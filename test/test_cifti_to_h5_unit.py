"""Focused unit tests for cifti_to_h5 branch coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tiledb
from utils import lazy_scalars

from modelarrayio.cli import cifti_to_h5


def test_cifti_to_h5_raises_when_scalar_sources_missing(monkeypatch) -> None:
    monkeypatch.setattr(cifti_to_h5, 'build_scalar_sources', lambda _cohort: {})
    with pytest.raises(ValueError, match='Unable to derive scalar sources'):
        cifti_to_h5.cifti_to_h5(pd.DataFrame(), output=Path('out.h5'))


def test_cifti_to_h5_tiledb_split_outputs_and_parcels(monkeypatch, tmp_path: Path) -> None:
    scalar_sources = {'FA': ['fa1.nii'], 'MD': ['md1.nii']}
    monkeypatch.setattr(cifti_to_h5, 'build_scalar_sources', lambda _cohort: scalar_sources)
    monkeypatch.setattr(
        cifti_to_h5,
        '_get_cifti_parcel_info',
        lambda _first: ('pscalar', {'parcel_id': np.array(['P1', 'P2'])}),
    )
    monkeypatch.setattr(
        cifti_to_h5,
        'load_cohort_cifti',
        lambda *_args, **_kwargs: (
            lazy_scalars(
                {
                    'FA': np.array([[1.0, 2.0]], dtype=np.float32),
                    'MD': np.array([[3.0, 4.0]], dtype=np.float32),
                }
            ),
            ['brain-a', 'brain-b'],
        ),
    )

    status = cifti_to_h5.cifti_to_h5(
        cohort_long=pd.DataFrame(
            {'scalar_name': ['FA', 'MD'], 'source_file': ['fa1.nii', 'md1.nii']}
        ),
        backend='tiledb',
        output=tmp_path / 'store.tdb',
        workers=2,
        split_outputs=True,
    )

    assert status == 0
    for scalar_name in scalar_sources:
        scalar_output = tmp_path / f'{scalar_name}_store.tdb'
        assert (
            tiledb.object_type(str(scalar_output / 'scalars' / scalar_name / 'values')) == 'array'
        )
        assert tiledb.object_type(str(scalar_output / 'parcels' / 'parcel_id')) == 'array'
        assert (
            tiledb.object_type(str(scalar_output / 'scalars' / scalar_name / 'column_names'))
            == 'array'
        )


def test_cifti_to_h5_hdf5_split_outputs_for_dscalar(monkeypatch, tmp_path: Path) -> None:
    scalar_sources = {'FA': ['fa1.nii']}
    scalars = lazy_scalars({'FA': np.array([[1.0, 2.0]], dtype=np.float32)})

    monkeypatch.setattr(cifti_to_h5, 'build_scalar_sources', lambda _cohort: scalar_sources)
    monkeypatch.setattr(cifti_to_h5, '_get_cifti_parcel_info', lambda _first: ('dscalar', {}))
    monkeypatch.setattr(
        cifti_to_h5,
        'load_cohort_cifti',
        lambda *_args, **_kwargs: (scalars, ['Left', 'Right']),
    )
    monkeypatch.setattr(
        cifti_to_h5,
        'brain_names_to_dataframe',
        lambda _brain_names: (pd.DataFrame({'i': [0, 1]}), ['Ctx']),
    )

    status = cifti_to_h5.cifti_to_h5(
        cohort_long=pd.DataFrame({'scalar_name': ['FA'], 'source_file': ['fa1.nii']}),
        backend='hdf5',
        output=tmp_path / 'grey.h5',
        split_outputs=True,
    )
    assert status == 0
    assert (tmp_path / 'FA_grey.h5').exists()
