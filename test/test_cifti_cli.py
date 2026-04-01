import csv
import os.path as op

import h5py
import nibabel as nb
import numpy as np
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis

from modelarrayio.cli.main import main as modelarrayio_main


def _make_synthetic_cifti_dscalar(mask_bool: np.ndarray, values: np.ndarray) -> nb.Cifti2Image:
    # Build axes: single scalar and a brain model from a volumetric mask
    scalar_axis = ScalarAxis(['synthetic'])  # one scalar map
    brain_axis = BrainModelAxis.from_mask(mask_bool)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    # Data must be 2D: (nmaps, ngrayordinates)
    data_2d = values.reshape(1, -1).astype(np.float32)
    return nb.Cifti2Image(data_2d, header=header)


def test_concifti_cli_creates_expected_hdf5(tmp_path, monkeypatch):
    # Create a small volumetric mask for brain model axis
    vol_shape = (3, 3, 3)
    mask = np.zeros(vol_shape, dtype=bool)
    true_vox = [(0, 0, 0), (0, 1, 2), (1, 1, 1), (2, 2, 0), (2, 1, 2)]
    for ijk in true_vox:
        mask[ijk] = True
    n_grayordinates = int(mask.sum())

    # Create two subjects with simple sequences
    subjects = []
    for sidx in range(2):
        vals = np.arange(n_grayordinates, dtype=np.float32) + sidx
        img = _make_synthetic_cifti_dscalar(mask, vals)
        path = tmp_path / f'sub-{sidx + 1}.dscalar.nii'
        img.to_filename(path)
        subjects.append(str(path.name))

    # Build cohort CSV
    cohort_csv = tmp_path / 'cohort_cifti.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scalar_name', 'source_file'])
        writer.writeheader()
        for sname in subjects:
            writer.writerow(
                {
                    'scalar_name': 'THICK',
                    'source_file': sname,
                }
            )

    out_h5 = tmp_path / 'out_cifti.h5'
    monkeypatch.chdir(tmp_path)
    assert (
        modelarrayio_main(
            [
                'cifti-to-h5',
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
        assert 'greyordinates' in h5
        grey = np.array(h5['greyordinates'])  # stored as transposed table (2, N)
        assert grey.shape[0] == 2  # vertex_id, structure_id
        n = grey.shape[1]
        assert n == n_grayordinates

        # structure_names present
        g = h5['greyordinates']
        assert 'structure_names' in g.attrs
        struct_names = g.attrs['structure_names']
        assert len(struct_names) >= 1

        # Scalars dataset
        dset = h5['scalars/THICK/values']
        n_files, n_elements = dset.shape
        assert n_files == 2
        assert n_elements == n_grayordinates

        # Column names exist and match subjects count
        grp = h5['scalars/THICK']
        assert 'column_names' in grp
        colnames = [
            x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in grp['column_names'][...]
        ]
        assert len(colnames) == 2

        # Spot-check a couple values
        assert np.isclose(float(dset[0, 0]), 0.0)
        assert np.isclose(float(dset[1, 0]), 1.0)


def test_cifti_to_h5_scalar_columns_writes_prefixed_outputs(tmp_path, monkeypatch):
    vol_shape = (2, 2, 2)
    mask = np.zeros(vol_shape, dtype=bool)
    mask[(0, 0, 0)] = True
    mask[(1, 1, 1)] = True
    n_grayordinates = int(mask.sum())

    rows = []
    for sidx in range(2):
        alpha_vals = np.arange(n_grayordinates, dtype=np.float32) + sidx
        beta_vals = np.arange(n_grayordinates, dtype=np.float32) + 10 + sidx

        alpha_img = _make_synthetic_cifti_dscalar(mask, alpha_vals)
        beta_img = _make_synthetic_cifti_dscalar(mask, beta_vals)

        alpha_path = tmp_path / f'sub-{sidx + 1}_alpha.dscalar.nii'
        beta_path = tmp_path / f'sub-{sidx + 1}_beta.dscalar.nii'
        alpha_img.to_filename(alpha_path)
        beta_img.to_filename(beta_path)

        rows.append(
            {'subject_id': f'sub-{sidx + 1}', 'alpha': alpha_path.name, 'beta': beta_path.name}
        )

    cohort_csv = tmp_path / 'cohort_cifti_wide.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'alpha', 'beta'])
        writer.writeheader()
        writer.writerows(rows)

    out_h5 = tmp_path / 'greyordinatearray.h5'
    alpha_out = tmp_path / 'alpha_greyordinatearray.h5'
    beta_out = tmp_path / 'beta_greyordinatearray.h5'

    monkeypatch.chdir(tmp_path)
    assert (
        modelarrayio_main(
            [
                'cifti-to-h5',
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
        assert 'greyordinates' in h5
        assert sorted(h5['scalars'].keys()) == ['alpha']

    with h5py.File(beta_out, 'r') as h5:
        assert 'greyordinates' in h5
        assert sorted(h5['scalars'].keys()) == ['beta']
