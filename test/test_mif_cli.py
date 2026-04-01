import csv

import h5py
import numpy as np
import pandas as pd

import modelarrayio.cli.mif_to_h5 as mif_cli
from modelarrayio.cli.main import main as modelarrayio_main


def test_mif_to_h5_scalar_columns_writes_prefixed_outputs(tmp_path, monkeypatch):
    index_file = tmp_path / 'index.nii.gz'
    directions_file = tmp_path / 'directions.nii.gz'
    index_file.write_bytes(b'index')
    directions_file.write_bytes(b'directions')

    cohort_csv = tmp_path / 'cohort_mif_wide.csv'
    with cohort_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'alpha', 'beta'])
        writer.writeheader()
        writer.writerow({'subject_id': 'sub-1', 'alpha': 'a1.mif', 'beta': 'b1.mif'})
        writer.writerow({'subject_id': 'sub-2', 'alpha': 'a2.mif', 'beta': 'b2.mif'})

    fixel_table = pd.DataFrame({'fixel_id': [0, 1], 'x': [0.0, 1.0], 'y': [0.0, 1.0], 'z': [0.0, 1.0]})
    voxel_table = pd.DataFrame({'voxel_id': [0], 'i': [0], 'j': [0], 'k': [0]})

    def fake_gather_fixels(_index_file, _directions_file):
        return fixel_table, voxel_table

    def fake_load_cohort_mif(cohort_long, _s3_workers):
        scalars = {}
        sources = {}
        for scalar_name, group in cohort_long.groupby('scalar_name'):
            n_rows = group.shape[0]
            scalars[scalar_name] = [np.array([1.0, 2.0], dtype=np.float32) for _ in range(n_rows)]
            sources[scalar_name] = group['source_file'].tolist()
        return scalars, sources

    monkeypatch.setattr(mif_cli, 'gather_fixels', fake_gather_fixels)
    monkeypatch.setattr(mif_cli, 'load_cohort_mif', fake_load_cohort_mif)

    out_h5 = tmp_path / 'fixelarray.h5'
    alpha_out = tmp_path / 'alpha_fixelarray.h5'
    beta_out = tmp_path / 'beta_fixelarray.h5'
    assert (
        modelarrayio_main(
            [
                'mif-to-h5',
                '--index-file',
                str(index_file),
                '--directions-file',
                str(directions_file),
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
        assert 'fixels' in h5
        assert 'voxels' in h5
        assert sorted(h5['scalars'].keys()) == ['alpha']

    with h5py.File(beta_out, 'r') as h5:
        assert 'fixels' in h5
        assert 'voxels' in h5
        assert sorted(h5['scalars'].keys()) == ['beta']
