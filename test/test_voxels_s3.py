"""Integration test for s3:// path support in convoxel.

Requires network access and boto3. Run with:
    pytest test/test_voxels_s3.py -v

Skip in offline CI by excluding the 's3' mark:
    pytest -m "not s3"
"""

import csv
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

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
def test_convoxel_s3_parallel(tmp_path, group_mask_path):
    """convoxel downloads s3:// paths in parallel and produces a valid HDF5."""
    pytest.importorskip('boto3')

    # Copy the group mask into tmp_path so --relative-root resolves it
    shutil.copy(group_mask_path, tmp_path / 'group_mask.nii.gz')

    # Cohort CSV with s3:// paths — relative_root is not prepended to s3:// URIs
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
    cmd = [
        sys.executable,
        '-m',
        'modelarrayio.cli.voxels_to_h5',
        '--group-mask-file',
        'group_mask.nii.gz',
        '--cohort-file',
        str(cohort_csv),
        '--relative-root',
        str(tmp_path),
        '--output-hdf5',
        str(out_h5.name),
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
    env = {**os.environ, 'MODELARRAYIO_S3_ANON': '1'}
    proc = subprocess.run(
        cmd, cwd=str(tmp_path), capture_output=True, text=True, env=env, check=False
    )
    assert proc.returncode == 0, f'convoxel failed:\n{proc.stdout}\n{proc.stderr}'
    assert out_h5.exists()

    with h5py.File(out_h5, 'r') as h5:
        dset = h5['scalars/alff/values']
        num_subjects, num_voxels = dset.shape

        assert num_subjects == len(OHSU_SUBJECTS)
        assert num_voxels > 0

        # Each subject should have at least some non-NaN values
        for i in range(num_subjects):
            assert not np.all(np.isnan(dset[i, :]))

        # Column names recorded in the file
        assert 'column_names' in h5['scalars/alff']
        colnames = h5['scalars/alff/column_names'][...]
        assert len(colnames) == len(OHSU_SUBJECTS)


@pytest.mark.s3
def test_convoxel_s3_serial_matches_parallel(tmp_path, group_mask_path):
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

    base_cmd = [
        sys.executable,
        '-m',
        'modelarrayio.cli.voxels_to_h5',
        '--group-mask-file',
        'group_mask.nii.gz',
        '--cohort-file',
        str(cohort_csv),
        '--relative-root',
        str(tmp_path),
        '--backend',
        'hdf5',
        '--dtype',
        'float32',
        '--compression',
        'none',
    ]

    env = {**os.environ, 'MODELARRAYIO_S3_ANON': '1'}
    for workers, name in [('1', 'serial.h5'), ('4', 'parallel.h5')]:
        proc = subprocess.run(
            base_cmd + ['--output-hdf5', name, '--s3-workers', workers],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, f'convoxel failed (workers={workers}):\n{proc.stderr}'

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
