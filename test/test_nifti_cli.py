"""Tests for the to-modelarray and export-results CLI commands (NIfTI modality)."""

from __future__ import annotations

import csv
import logging
import os.path as op

import h5py
import nibabel as nb
import numpy as np
import pytest
import tiledb
from utils import make_nifti

from modelarrayio.cli.main import main as modelarrayio_main


def _ijk_value(i, j, k):
    return i * 100.0 + j * 10.0 + k * 1.0


def test_nifti_to_h5_creates_expected_hdf5(tmp_path, monkeypatch):
    # Small synthetic volume
    shape = (5, 6, 7)
    group_mask = np.zeros(shape, dtype=bool)
    # Create a sparse pattern of true voxels
    true_coords = [(0, 1, 1), (1, 2, 3), (2, 4, 5), (3, 0, 0), (4, 5, 6), (1, 1, 4), (2, 2, 2)]
    for i, j, k in true_coords:
        group_mask[i, j, k] = True

    # Save group mask
    group_mask_img = make_nifti(group_mask.astype(np.uint8))
    group_mask_file = tmp_path / 'group_mask.nii.gz'
    group_mask_img.to_filename(group_mask_file)

    # Create two subjects with individual masks (one drops a voxel)
    subjects = []
    for sidx in range(2):
        # Scalar volume encodes f(i,j,k)
        scalar = np.zeros(shape, dtype=np.float32)
        for i, j, k in true_coords:
            scalar[i, j, k] = _ijk_value(i, j, k) + sidx  # slight per-subject shift

        # Individual mask: subject 1 omits one voxel
        indiv_mask = group_mask.copy()
        if sidx == 1:
            omit = true_coords[1]
            indiv_mask[omit] = False

        scalar_img = make_nifti(scalar)
        mask_img = make_nifti(indiv_mask.astype(np.uint8))

        scalar_path = tmp_path / f'sub-{sidx + 1}_scalar.nii.gz'
        mask_path = tmp_path / f'sub-{sidx + 1}_mask.nii.gz'
        scalar_img.to_filename(scalar_path)
        mask_img.to_filename(mask_path)
        subjects.append((str(scalar_path.name), str(mask_path.name)))

    # Build cohort CSV (relative paths)
    cohort_csv = tmp_path / 'cohort.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scalar_name', 'source_file', 'source_mask_file'])
        writer.writeheader()
        for _sidx, (scalar_name, mask_name) in enumerate(subjects):
            writer.writerow(
                {
                    'scalar_name': 'FA',
                    'source_file': scalar_name,
                    'source_mask_file': mask_name,
                }
            )

    out_h5 = tmp_path / 'out.h5'
    monkeypatch.chdir(tmp_path)
    assert (
        modelarrayio_main(
            [
                'to-modelarray',
                '--mask',
                str(group_mask_file),
                '--cohort-file',
                str(cohort_csv),
                '--output',
                str(out_h5),
                '--backend',
                'hdf5',
                '--dtype',
                'float32',
                '--compression',
                'gzip',
                '--compression-level',
                '1',
                '--target-chunk-mb',
                '1.0',
            ]
        )
        == 0
    )
    assert op.exists(out_h5)

    # Validate HDF5 contents
    with h5py.File(out_h5, 'r') as h5:
        assert 'voxels' in h5
        vox = np.array(h5['voxels'])  # stored as transposed table (4, N): voxel_id, i, j, k
        assert vox.shape[0] == 4
        ijk = np.vstack(np.nonzero(group_mask))  # (3, N) ordered by i, then j, then k
        assert vox.shape[1] == ijk.shape[1]

        # Rows 1-3 are i, j, k (row 0 is voxel_id)
        assert np.array_equal(vox[1:], ijk)

        # Scalars dataset
        dset = h5['scalars/FA/values']
        n_files, n_voxels = dset.shape
        assert n_files == 2
        assert n_voxels == ijk.shape[1]

        # Column names exist and match subjects count
        grp = h5['scalars/FA']
        assert 'column_names' in grp
        colnames = [
            x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in grp['column_names'][...]
        ]
        assert len(colnames) == 2

        # Spot-check a voxel mapping (pick the third voxel)
        vidx = 2
        i, j, k = int(ijk[0, vidx]), int(ijk[1, vidx]), int(ijk[2, vidx])
        expected_s0 = _ijk_value(i, j, k) + 0
        expected_s1 = _ijk_value(i, j, k) + 1
        # If subject 1 omitted that voxel, it should be NaN (masked out becomes NaN on flatten)
        v0 = float(dset[0, vidx])
        v1 = float(dset[1, vidx])
        assert np.isclose(v0, expected_s0, equal_nan=True)
        # Determine whether subject 1 omitted this voxel
        omitted = False
        omit = true_coords[1]
        if (i, j, k) == omit:
            omitted = True
        if omitted:
            assert np.isnan(v1)
        else:
            assert np.isclose(v1, expected_s1, equal_nan=True)


def test_h5_to_nifti_writes_results_with_dataset_column_names(tmp_path):
    shape = (3, 3, 3)
    group_mask = np.zeros(shape, dtype=bool)
    true_coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    for coord in true_coords:
        group_mask[coord] = True

    group_mask_file = tmp_path / 'group_mask.nii.gz'
    make_nifti(group_mask.astype(np.uint8)).to_filename(group_mask_file)

    in_file = tmp_path / 'results.h5'
    with h5py.File(in_file, 'w') as h5:
        group = h5.require_group('results/lm')
        group.create_dataset(
            'results_matrix',
            data=np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.9, 0.8, 0.7],
                ],
                dtype=np.float32,
            ),
        )
        group.create_dataset(
            'column_names',
            data=np.array(['effect size', 'p.value'], dtype=h5py.string_dtype('utf-8')),
        )

    output_dir = tmp_path / 'nifti_results'
    assert (
        modelarrayio_main(
            [
                'export-results',
                '--mask',
                str(group_mask_file),
                '--analysis-name',
                'lm',
                '--input-hdf5',
                str(in_file),
                '--output-dir',
                str(output_dir),
            ]
        )
        == 0
    )

    effect_file = output_dir / 'lm_effect_size.nii.gz'
    pvalue_file = output_dir / 'lm_p.value.nii.gz'
    inv_pvalue_file = output_dir / 'lm_1m.p.value.nii.gz'
    assert effect_file.exists()
    assert pvalue_file.exists()
    assert inv_pvalue_file.exists()

    effect_data = nb.load(effect_file).get_fdata()
    pvalue_data = nb.load(pvalue_file).get_fdata()
    inv_pvalue_data = nb.load(inv_pvalue_file).get_fdata()

    for idx, coord in enumerate(true_coords):
        assert effect_data[coord] == pytest.approx([0.1, 0.2, 0.3][idx])
        assert pvalue_data[coord] == pytest.approx([0.9, 0.8, 0.7][idx])
        assert inv_pvalue_data[coord] == pytest.approx([0.1, 0.2, 0.3][idx])


def test_nifti_to_h5_scalar_columns_writes_prefixed_outputs(tmp_path, monkeypatch):
    shape = (3, 3, 3)
    group_mask = np.zeros(shape, dtype=bool)
    true_coords = [(0, 0, 1), (1, 1, 1), (2, 2, 0)]
    for i, j, k in true_coords:
        group_mask[i, j, k] = True

    group_mask_file = tmp_path / 'group_mask.nii.gz'
    make_nifti(group_mask.astype(np.uint8)).to_filename(group_mask_file)

    rows = []
    for sidx in range(2):
        subj_mask_file = tmp_path / f'sub-{sidx + 1}_mask.nii.gz'
        make_nifti(group_mask.astype(np.uint8)).to_filename(subj_mask_file)

        alpha_data = np.zeros(shape, dtype=np.float32)
        beta_data = np.zeros(shape, dtype=np.float32)
        for i, j, k in true_coords:
            alpha_data[i, j, k] = 10.0 + sidx
            beta_data[i, j, k] = 20.0 + sidx

        alpha_file = tmp_path / f'sub-{sidx + 1}_alpha.nii.gz'
        beta_file = tmp_path / f'sub-{sidx + 1}_beta.nii.gz'
        make_nifti(alpha_data).to_filename(alpha_file)
        make_nifti(beta_data).to_filename(beta_file)

        rows.append(
            {
                'subject_id': f'sub-{sidx + 1}',
                'alpha': alpha_file.name,
                'beta': beta_file.name,
                'source_mask_file': subj_mask_file.name,
            }
        )

    cohort_csv = tmp_path / 'cohort_wide.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'alpha', 'beta', 'source_mask_file'])
        writer.writeheader()
        writer.writerows(rows)

    out_h5 = tmp_path / 'voxelarray.h5'
    alpha_out = tmp_path / 'alpha_voxelarray.h5'
    beta_out = tmp_path / 'beta_voxelarray.h5'

    monkeypatch.chdir(tmp_path)
    assert (
        modelarrayio_main(
            [
                'to-modelarray',
                '--mask',
                str(group_mask_file),
                '--cohort-file',
                str(cohort_csv),
                '--scalar-columns',
                'alpha',
                'beta',
                '--output',
                str(out_h5),
            ]
        )
        == 0
    )

    assert alpha_out.exists()
    assert beta_out.exists()
    assert not out_h5.exists()

    with h5py.File(alpha_out, 'r') as h5:
        assert 'voxels' in h5
        assert sorted(h5['scalars'].keys()) == ['alpha']

    with h5py.File(beta_out, 'r') as h5:
        assert 'voxels' in h5
        assert sorted(h5['scalars'].keys()) == ['beta']


def _build_nifti_cohort(tmp_path):
    """Create a minimal NIfTI cohort (group mask + 2 subjects) and return CLI args."""
    shape = (4, 4, 4)
    true_coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]

    group_mask = np.zeros(shape, dtype=np.uint8)
    for coord in true_coords:
        group_mask[coord] = 1
    group_mask_file = tmp_path / 'group_mask.nii.gz'
    make_nifti(group_mask).to_filename(group_mask_file)

    rows = []
    for sidx in range(2):
        scalar = np.zeros(shape, dtype=np.float32)
        for i, j, k in true_coords:
            scalar[i, j, k] = float(i + j + k + sidx)
        scalar_file = tmp_path / f'sub-{sidx + 1}_scalar.nii.gz'
        mask_file = tmp_path / f'sub-{sidx + 1}_mask.nii.gz'
        make_nifti(scalar).to_filename(scalar_file)
        make_nifti(group_mask).to_filename(mask_file)
        rows.append(
            {
                'scalar_name': 'FA',
                'source_file': scalar_file.name,
                'source_mask_file': mask_file.name,
            }
        )

    cohort_csv = tmp_path / 'cohort.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scalar_name', 'source_file', 'source_mask_file'])
        writer.writeheader()
        writer.writerows(rows)

    return group_mask_file, cohort_csv


def test_nifti_tiledb_removes_existing_arrays_on_rerun(tmp_path, monkeypatch, caplog):
    """Regression test for https://github.com/PennLINC/ModelArrayIO/issues/39.

    The TileDB backend should succeed when the output directory already contains
    arrays from a previous run, removing and recreating them, and should emit a
    warning for each removed array.
    """
    group_mask_file, cohort_csv = _build_nifti_cohort(tmp_path)
    out_tdb = tmp_path / 'out.tdb'
    monkeypatch.chdir(tmp_path)

    cli_args = [
        'to-modelarray',
        '--mask',
        str(group_mask_file),
        '--cohort-file',
        str(cohort_csv),
        '--output',
        str(out_tdb),
        '--backend',
        'tiledb',
        '--compression',
        'gzip',
    ]

    # First run should succeed without any "Removing existing array" warnings.
    with caplog.at_level(logging.WARNING, logger='modelarrayio.storage.tiledb_storage'):
        assert modelarrayio_main(cli_args) == 0
    assert out_tdb.exists()
    assert tiledb.object_type(str(out_tdb / 'scalars' / 'FA' / 'values')) is not None
    assert not any('Removing existing array' in r.message for r in caplog.records)

    # Second run to the same output directory should succeed (regression for
    # issue #39) and emit a warning for the pre-existing array that was removed.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='modelarrayio.storage.tiledb_storage'):
        assert modelarrayio_main(cli_args) == 0
    assert any('Removing existing array' in r.message for r in caplog.records)
