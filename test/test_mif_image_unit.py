"""Focused tests for MIF header/image read-write behavior."""

from __future__ import annotations

import io
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from modelarrayio.utils.mif_image import MifHeader, MifImage


def test_mif_header_write_to_and_from_fileobj_round_trip() -> None:
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
        ],
        dtype=np.float64,
    )
    header = MifHeader(
        shape=(2, 3, 4),
        zooms=(1.0, 2.0, 3.0),
        layout=[-1, 2, 3],
        dtype=np.dtype('<f4'),
        transform=transform,
        intensity_offset=0.5,
        intensity_scale=2.0,
        keyval={'foo': 'bar\nbaz'},
    )
    bio = io.BytesIO()
    data_offset = header.write_to(bio)
    bio.seek(0)
    parsed = MifHeader.from_fileobj(bio)

    assert data_offset >= 0
    assert parsed.get_data_shape() == (2, 3, 4)
    assert parsed.get_layout() == [-1, 2, 3]
    assert parsed.get_data_dtype() == np.dtype('<f4')
    assert parsed.get_intensity_scaling() == (0.5, 2.0)
    assert parsed.get_keyval()['foo'] == 'bar\nbaz'


@pytest.mark.parametrize(
    'header_text',
    [
        b'mrtrix image\nvox: 1\nlayout: +0\ndatatype: Float32LE\nEND\n',
        b'mrtrix image\ndim: 1\nlayout: +0\ndatatype: Float32LE\nEND\n',
        b'mrtrix image\ndim: 1\nvox: 1\ndatatype: Float32LE\nEND\n',
        b'mrtrix image\ndim: 1\nvox: 1\nlayout: +0\nEND\n',
    ],
)
def test_mif_header_missing_required_fields_raise(header_text: bytes) -> None:
    with pytest.raises(ValueError):
        MifHeader.from_fileobj(io.BytesIO(header_text))


def test_mif_header_rejects_non_mif_magic() -> None:
    with pytest.raises(ValueError, match='Not a MIF file'):
        MifHeader.from_fileobj(io.BytesIO(b'not mif\n'))


def test_mif_image_round_trip(tmp_path: Path) -> None:
    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 3.0, 0.0, 2.0],
            [0.0, 0.0, 4.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image = MifImage(data, affine)
    out_file = tmp_path / 'roundtrip.mif'
    image.to_filename(out_file)

    loaded = MifImage.from_filename(str(out_file))
    np.testing.assert_array_equal(loaded.get_fdata(dtype=np.float32), data)
    np.testing.assert_allclose(loaded.affine, affine)


def test_mif_image_from_file_map_requires_data_offset(tmp_path: Path) -> None:
    bad = tmp_path / 'bad.mif'
    bad.write_bytes(
        b'mrtrix image\ndim: 1\nvox: 1\nlayout: +0\ndatatype: Float32LE\ntransform: 1,0,0,0\nEND\n'
    )
    file_map = {'image': nb.FileHolder(filename=str(bad))}
    with pytest.raises(ValueError, match='Could not determine data offset'):
        MifImage.from_file_map(file_map)


def test_affine2header_updates_zooms_and_transform() -> None:
    data = np.ones((2, 2, 2), dtype=np.float32)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 3.0, 0.0, 6.0],
            [0.0, 0.0, 4.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = MifImage(data, affine)
    image._affine2header()
    zooms = image.header.get_zooms()
    assert zooms[:3] == (2.0, 3.0, 4.0)
    transform = image.header.get_transform()
    np.testing.assert_allclose(transform[:, 3], [5.0, 6.0, 7.0])
