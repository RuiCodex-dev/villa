"""Cube must not leave its downloads behind in the temp directory.

Without ``cache=True`` each NRRD (volume and mask) was written to a
``NamedTemporaryFile(delete=False)`` that was never removed: about 16 MB per
cube of the published instance labels, left in the system temp directory on
every load, ``activate_caching()`` and ``deactivate_caching()``.
"""

from __future__ import annotations

import os
import tempfile

import nrrd
import numpy as np
import pytest

import vesuvius.data.volume as volume_mod
from vesuvius.data.volume import Cube


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


@pytest.fixture
def offline_cube(tmp_path, monkeypatch):
    """Serve two small NRRD files instead of the remote cube, and isolate temp."""
    payloads = {}
    for name, value in (("volume", 11), ("mask", 1)):
        src = tmp_path / f"{name}.nrrd"
        nrrd.write(str(src), np.full((4, 5, 6), value, dtype=np.uint8))
        payloads[name] = src.read_bytes()

    requested = []

    def fake_get(url, *args, **kwargs):
        requested.append(url)
        return _Response(payloads["mask" if url.endswith("_mask.nrrd") else "volume"])

    temp_dir = tmp_path / "system-temp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    monkeypatch.setattr(volume_mod, "is_aws_ec2_instance", lambda: False)
    monkeypatch.setattr(volume_mod.requests, "get", fake_get)
    return temp_dir, requested


def test_uncached_cube_removes_its_temp_files(offline_cube):
    temp_dir, requested = offline_cube

    cube = Cube(1, 54, 7.91, 2256, 2512, 4816, cache=False)

    assert cube.volume.shape == (4, 5, 6) and int(cube.volume.max()) == 11
    assert int(cube.mask.max()) == 1
    assert os.listdir(temp_dir) == []
    # The URLs are joined with '/', whatever the platform.
    assert all("\\" not in url for url in requested)
    assert requested[0].endswith("/02256_02512_04816/02256_02512_04816_volume.nrrd")


def test_cached_cube_reads_the_complete_file_and_reuses_it(offline_cube, tmp_path):
    temp_dir, requested = offline_cube
    cache_dir = tmp_path / "cube-cache"

    first = Cube(1, 54, 7.91, 2256, 2512, 4816, cache=True, cache_dir=cache_dir)
    second = Cube(1, 54, 7.91, 2256, 2512, 4816, cache=True, cache_dir=cache_dir)

    assert len(requested) == 2  # the second cube came from the cache
    np.testing.assert_array_equal(first.volume, second.volume)
    assert sorted(os.listdir(cache_dir / "02256_02512_04816")) == [
        "02256_02512_04816_mask.nrrd",
        "02256_02512_04816_volume.nrrd",
    ]
    assert os.listdir(temp_dir) == []
