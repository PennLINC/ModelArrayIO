"""Integration test for s3:// path support in nifti.

Requires network access and boto3. Run with:
    pytest test/test_voxels_s3.py -v

Skip in offline CI by excluding the 's3' mark:
    pytest -m "not s3"
"""

import csv
import shutil

import h5py
import numpy as np
import pytest

from modelarrayio.cli.main import main as modelarrayio_main

# Four confirmed ABIDE OHSU subjects used as test data
OHSU_SUBJECTS = [
    'OHSU_0050142',
    'OHSU_0050143',
    'OHSU_0050144',
    'OHSU_0050145',
]

_BUCKET = 'fcp-indi'
_PREFIX = 'data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global'


def _s3_alff(subject_id):
    return f's3://{_BUCKET}/{_PREFIX}/alff/{subject_id}_alff.nii.gz'


def _s3_mask(subject_id):
    return f's3://{_BUCKET}/{_PREFIX}/func_mask/{subject_id}_func_mask.nii.gz'


@pytest.fixture(scope='module')
def group_mask_path(tmp_path_factory):
    """Download one func_mask from S3 to use as the group mask for all tests."""
    boto3 = pytest.importorskip('boto3')
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError

    tmp = tmp_path_factory.mktemp('s3_group_mask')
    dest = tmp / 'group_mask.nii.gz'
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    key = f'{_PREFIX}/func_mask/{OHSU_SUBJECTS[0]}_func_mask.nii.gz'
    try:
        s3.download_file(_BUCKET, key, str(dest))
    except (OSError, BotoCoreError) as exc:
        pytest.skip(f'S3 download unavailable: {exc}')
    return dest


@pytest.mark.s3
def test_nifti_s3_parallel(tmp_path, group_mask_path, monkeypatch):
    """nifti downloads s3:// paths in parallel and produces a valid HDF5."""
    pytest.importorskip('boto3')

    shutil.copy(group_mask_path, tmp_path / 'group_mask.nii.gz')

    # Cohort CSV with s3:// paths
    cohort_csv = tmp_path / 'cohort.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scalar_name', 'source_file', 'source_mask_file'])
        writer.writeheader()
        for subj in OHSU_SUBJECTS:
            writer.writerow(
                {
                    'scalar_name': 'alff',
                    'source_file': _s3_alff(subj),
                    'source_mask_file': _s3_mask(subj),
                }
            )

    out_h5 = tmp_path / 'out.h5'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MODELARRAYIO_S3_ANON', '1')
    assert (
        modelarrayio_main(
            [
                'to-modelarray',
                '--mask',
                'group_mask.nii.gz',
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
                '--s3-workers',
                '4',
            ]
        )
        == 0
    )
    assert out_h5.exists()

    with h5py.File(out_h5, 'r') as h5:
        dset = h5['scalars/alff/values']
        n_files, n_voxels = dset.shape

        assert n_files == len(OHSU_SUBJECTS)
        assert n_voxels > 0

        # Each subject should have at least some non-NaN values
        for i in range(n_files):
            assert not np.all(np.isnan(dset[i, :]))

        # Column names recorded in the file
        assert 'column_names' in h5['scalars/alff']
        colnames = h5['scalars/alff/column_names'][...]
        assert len(colnames) == len(OHSU_SUBJECTS)


@pytest.mark.s3
def test_nifti_s3_serial_matches_parallel(tmp_path, group_mask_path, monkeypatch):
    """Serial (s3-workers=1) and parallel (s3-workers=4) produce identical data."""
    pytest.importorskip('boto3')

    shutil.copy(group_mask_path, tmp_path / 'group_mask.nii.gz')

    cohort_csv = tmp_path / 'cohort.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scalar_name', 'source_file', 'source_mask_file'])
        writer.writeheader()
        for subj in OHSU_SUBJECTS:
            writer.writerow(
                {
                    'scalar_name': 'alff',
                    'source_file': _s3_alff(subj),
                    'source_mask_file': _s3_mask(subj),
                }
            )

    base_argv = [
        'to-modelarray',
        '--mask',
        str(group_mask_path),
        '--cohort-file',
        str(cohort_csv),
        '--backend',
        'hdf5',
        '--dtype',
        'float32',
        '--compression',
        'none',
    ]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MODELARRAYIO_S3_ANON', '1')
    for workers, name in [('1', 'serial.h5'), ('4', 'parallel.h5')]:
        assert modelarrayio_main(base_argv + ['--output', name, '--s3-workers', workers]) == 0, (
            f'modelarrayio to-modelarray failed (workers={workers})'
        )

    with (
        h5py.File(tmp_path / 'serial.h5', 'r') as s,
        h5py.File(tmp_path / 'parallel.h5', 'r') as p,
    ):
        serial_data = s['scalars/alff/values'][...]
        parallel_data = p['scalars/alff/values'][...]

    # Row order in the parallel result may differ from cohort order, so sort both
    # by their row fingerprint before comparing
    assert serial_data.shape == parallel_data.shape
    serial_sorted = serial_data[np.lexsort(serial_data.T)]
    parallel_sorted = parallel_data[np.lexsort(parallel_data.T)]
    np.testing.assert_array_equal(serial_sorted, parallel_sorted)
