"""Tests for h5-export-*-file commands."""

from __future__ import annotations

import h5py
import nibabel as nb
import numpy as np
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis

from modelarrayio.cli import h5_export_mif_file as export_mif_cli
from modelarrayio.cli.main import main as modelarrayio_main


def _make_nifti(data):
    return nb.Nifti1Image(data.astype(np.float32), affine=np.eye(4))


def _make_synthetic_cifti(mask_bool: np.ndarray, values: np.ndarray) -> nb.Cifti2Image:
    scalar_axis = ScalarAxis(['synthetic'])
    brain_axis = BrainModelAxis.from_mask(mask_bool)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    data_2d = values.reshape(1, -1).astype(np.float32)
    return nb.Cifti2Image(data_2d, header=header)


def test_h5_export_nifti_file_cli_column_index_and_source_file(tmp_path):
    shape = (3, 3, 3)
    group_mask = np.zeros(shape, dtype=bool)
    coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    for coord in coords:
        group_mask[coord] = True
    group_mask_file = tmp_path / 'group_mask.nii.gz'
    _make_nifti(group_mask.astype(np.uint8)).to_filename(group_mask_file)

    in_file = tmp_path / 'input.h5'
    with h5py.File(in_file, 'w') as h5:
        h5.create_dataset(
            'scalars/FA/values',
            data=np.array(
                [
                    [10.0, 20.0, 30.0],
                    [11.0, 21.0, np.nan],
                ],
                dtype=np.float32,
            ),
        )
        h5.create_dataset(
            'scalars/FA/column_names',
            data=np.array(['sub-01_scalar.nii.gz', 'sub-02_scalar.nii.gz'], dtype=object),
            dtype=h5py.string_dtype('utf-8'),
        )

    out_column_index = tmp_path / 'column_index.nii.gz'
    assert (
        modelarrayio_main(
            [
                'h5-export-nifti-file',
                '--input-hdf5',
                str(in_file),
                '--scalar-name',
                'FA',
                '--column-index',
                '1',
                '--group-mask-file',
                str(group_mask_file),
                '--output-file',
                str(out_column_index),
            ]
        )
        == 0
    )
    out_data = nb.load(out_column_index).get_fdata()
    assert out_data[coords[0]] == 11.0
    assert out_data[coords[1]] == 21.0
    assert np.isnan(out_data[coords[2]])

    out_source_file = tmp_path / 'source_file.nii.gz'
    assert (
        modelarrayio_main(
            [
                'h5-export-nifti-file',
                '--input-hdf5',
                str(in_file),
                '--scalar-name',
                'FA',
                '--source-file',
                'sub-01_scalar.nii.gz',
                '--group-mask-file',
                str(group_mask_file),
                '--output-file',
                str(out_source_file),
            ]
        )
        == 0
    )
    out_data_source = nb.load(out_source_file).get_fdata()
    assert out_data_source[coords[0]] == 10.0
    assert out_data_source[coords[1]] == 20.0
    assert out_data_source[coords[2]] == 30.0


def test_h5_export_cifti_file_cli(tmp_path):
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    template = tmp_path / 'template.dscalar.nii'
    _make_synthetic_cifti(mask, np.array([0.0, 0.0], dtype=np.float32)).to_filename(template)

    in_file = tmp_path / 'input.h5'
    with h5py.File(in_file, 'w') as h5:
        h5.create_dataset(
            'scalars/THICK/values',
            data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

    out_file = tmp_path / 'exported.dscalar.nii'
    assert (
        modelarrayio_main(
            [
                'h5-export-cifti-file',
                '--input-hdf5',
                str(in_file),
                '--scalar-name',
                'THICK',
                '--column-index',
                '1',
                '--example-cifti',
                str(template),
                '--output-file',
                str(out_file),
            ]
        )
        == 0
    )
    exported = nb.load(out_file).get_fdata().squeeze()
    assert np.array_equal(exported, np.array([3.0, 4.0], dtype=np.float64))


def test_h5_export_mif_file_cli(monkeypatch, tmp_path):
    in_file = tmp_path / 'input.h5'
    with h5py.File(in_file, 'w') as h5:
        h5.create_dataset(
            'scalars/FD/values',
            data=np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32),
        )

    example_mif = tmp_path / 'example.mif'
    example_mif.write_text('dummy')
    out_file = tmp_path / 'subject.mif'

    captured = {}

    def _fake_mif_to_nifti2(_path):
        template = nb.Nifti2Image(np.zeros((3, 1, 1), dtype=np.float32), affine=np.eye(4))
        return template, np.zeros(3, dtype=np.float32)

    def _fake_nifti2_to_mif(nifti2_image, mif_file):
        captured['data'] = nifti2_image.get_fdata().squeeze().copy()
        mif_file.write_text('fake mif')

    monkeypatch.setattr(export_mif_cli, 'mif_to_nifti2', _fake_mif_to_nifti2)
    monkeypatch.setattr(export_mif_cli, 'nifti2_to_mif', _fake_nifti2_to_mif)

    assert (
        modelarrayio_main(
            [
                'h5-export-mif-file',
                '--input-hdf5',
                str(in_file),
                '--scalar-name',
                'FD',
                '--column-index',
                '1',
                '--example-mif',
                str(example_mif),
                '--output-file',
                str(out_file),
            ]
        )
        == 0
    )
    assert out_file.exists()
    assert np.array_equal(captured['data'], np.array([8.0, 9.0, 10.0], dtype=np.float64))
