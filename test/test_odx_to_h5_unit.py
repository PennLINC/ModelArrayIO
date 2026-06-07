"""Unit tests for the ODX modality (detection + odx_to_h5 branching).

These tests monkeypatch the ODX readers so they run without the optional
``odx`` package installed, mirroring ``test_mif_to_h5_unit.py``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from modelarrayio.cli import odx_to_h5
from modelarrayio.utils.misc import detect_modality_from_path


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'scalar_name': ['angle_deg', 'angle_deg'],
            'source_file': ['sub-01.odx', 'sub-02.odx'],
            'subject': ['sub-01', 'sub-02'],
        }
    )


def _fixtures():
    fixel_table = pd.DataFrame(
        {'fixel_id': [0, 1], 'voxel_id': [0, 0], 'x': [1.0, 0.0], 'y': [0.0, 1.0], 'z': [0.0, 0.0]}
    )
    voxel_table = pd.DataFrame({'voxel_id': [0], 'i': [0], 'j': [0], 'k': [0]})
    scalars = {'angle_deg': [np.array([0.0, 5.0], np.float32), np.array([3.0, 7.0], np.float32)]}
    sources = {'angle_deg': ['sub-01.odx', 'sub-02.odx']}
    return fixel_table, voxel_table, scalars, sources


def test_detect_modality_odx() -> None:
    assert detect_modality_from_path('group/sub-01.odx') == 'odx'
    assert detect_modality_from_path('group/sub-01.odx/') == 'odx'  # directory layout


def test_odx_to_h5_raises_when_sources_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        odx_to_h5, 'gather_fixels_from_odx', lambda _src: (pd.DataFrame(), pd.DataFrame())
    )
    monkeypatch.setattr(odx_to_h5, 'load_cohort_odx', lambda cohort_long, s3_workers: ({}, {}))
    with pytest.raises(ValueError, match='Unable to derive scalar sources'):
        odx_to_h5.odx_to_h5(_cohort(), output=tmp_path / 'out.h5')


def test_odx_to_h5_writes_modelarray_schema(monkeypatch, tmp_path: Path) -> None:
    fixel_table, voxel_table, scalars, sources = _fixtures()
    monkeypatch.setattr(
        odx_to_h5, 'gather_fixels_from_odx', lambda *_a, **_k: (fixel_table, voxel_table)
    )
    monkeypatch.setattr(odx_to_h5, 'load_cohort_odx', lambda *_a, **_k: (scalars, sources))

    out = tmp_path / 'angle.h5'
    status = odx_to_h5.odx_to_h5(_cohort(), backend='hdf5', output=out)
    assert status == 0

    with h5py.File(out, 'r') as f:
        values = f['scalars/angle_deg/values']
        # ModelArray's per-scalar values matrix: (n_subjects, n_fixels)
        assert values.shape == (2, 2)
        assert np.allclose(values[0], [0.0, 5.0])
        assert np.allclose(values[1], [3.0, 7.0])
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in f['scalars/angle_deg/column_names'][()]]
        assert cols == ['sub-01.odx', 'sub-02.odx']
        # fixel geometry is carried for mapping results back
        assert 'fixels' in f
        assert 'voxels' in f
        assert list(f['fixels'].attrs['column_names']) == ['fixel_id', 'voxel_id', 'x', 'y', 'z']


def test_odx_to_h5_split_outputs(monkeypatch, tmp_path: Path) -> None:
    fixel_table, voxel_table, _scalars, _sources = _fixtures()
    scalars = {
        'angle_deg': [np.array([0.0, 5.0], np.float32), np.array([3.0, 7.0], np.float32)],
        'afd': [np.array([0.1, 0.2], np.float32), np.array([0.3, 0.4], np.float32)],
    }
    sources = {'angle_deg': ['sub-01.odx', 'sub-02.odx'], 'afd': ['sub-01.odx', 'sub-02.odx']}
    monkeypatch.setattr(
        odx_to_h5, 'gather_fixels_from_odx', lambda *_a, **_k: (fixel_table, voxel_table)
    )
    monkeypatch.setattr(odx_to_h5, 'load_cohort_odx', lambda *_a, **_k: (scalars, sources))

    status = odx_to_h5.odx_to_h5(_cohort(), backend='hdf5', output=tmp_path / 'fixels.h5', split_outputs=True)
    assert status == 0
    assert (tmp_path / 'angle_deg_fixels.h5').exists()
    assert (tmp_path / 'afd_fixels.h5').exists()
