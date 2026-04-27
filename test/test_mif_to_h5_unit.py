"""Unit tests for mif_to_h5 branching behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelarrayio.cli import mif_to_h5


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'scalar_name': ['FA', 'MD'],
            'source_file': ['fa.mif', 'md.mif'],
        }
    )


def test_mif_to_h5_raises_when_sources_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        mif_to_h5,
        'gather_fixels',
        lambda index_file, directions_file: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(mif_to_h5, 'load_cohort_mif', lambda cohort_long, s3_workers: ({}, {}))

    with pytest.raises(ValueError, match='Unable to derive scalar sources'):
        mif_to_h5.mif_to_h5('index.mif', 'directions.mif', _cohort(), output=tmp_path / 'out.h5')


def test_mif_to_h5_hdf5_split_outputs(monkeypatch, tmp_path: Path) -> None:
    fixel_table = pd.DataFrame({'fixel_id': [0], 'voxel_id': [0], 'x': [1], 'y': [0], 'z': [0]})
    voxel_table = pd.DataFrame({'voxel_id': [0], 'i': [0], 'j': [0], 'k': [0]})
    scalars = {
        'FA': [np.array([1.0], dtype=np.float32)],
        'MD': [np.array([2.0], dtype=np.float32)],
    }
    sources = {'FA': ['fa.mif'], 'MD': ['md.mif']}

    monkeypatch.setattr(
        mif_to_h5, 'gather_fixels', lambda *_args, **_kwargs: (fixel_table, voxel_table)
    )
    monkeypatch.setattr(mif_to_h5, 'load_cohort_mif', lambda *_args, **_kwargs: (scalars, sources))

    status = mif_to_h5.mif_to_h5(
        'index.mif',
        'directions.mif',
        _cohort(),
        backend='hdf5',
        output=tmp_path / 'fixels.h5',
        split_outputs=True,
    )
    assert status == 0
    assert (tmp_path / 'FA_fixels.h5').exists()
    assert (tmp_path / 'MD_fixels.h5').exists()


def test_mif_to_h5_tiledb_parallel_and_split(monkeypatch, tmp_path: Path) -> None:
    fixel_table = pd.DataFrame({'fixel_id': [0], 'voxel_id': [0], 'x': [1], 'y': [0], 'z': [0]})
    voxel_table = pd.DataFrame({'voxel_id': [0], 'i': [0], 'j': [0], 'k': [0]})
    scalars = {
        'FA': [np.array([1.0], dtype=np.float32)],
        'MD': [np.array([2.0], dtype=np.float32)],
    }
    sources = {'FA': ['fa.mif'], 'MD': ['md.mif']}
    calls = []

    monkeypatch.setattr(
        mif_to_h5, 'gather_fixels', lambda *_args, **_kwargs: (fixel_table, voxel_table)
    )
    monkeypatch.setattr(mif_to_h5, 'load_cohort_mif', lambda *_args, **_kwargs: (scalars, sources))
    monkeypatch.setattr(
        mif_to_h5.cli_utils,
        'write_tiledb_scalar_matrices',
        lambda output, scalars, sources, **kwargs: calls.append((Path(output), scalars, sources)),
    )

    status = mif_to_h5.mif_to_h5(
        'index.mif',
        'directions.mif',
        _cohort(),
        backend='tiledb',
        output=tmp_path / 'fixels.tdb',
        workers=2,
        split_outputs=True,
    )
    assert status == 0
    assert sorted(path.name for path, _, _ in calls) == ['FA_fixels.tdb', 'MD_fixels.tdb']
