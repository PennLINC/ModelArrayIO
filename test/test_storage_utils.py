"""Unit tests for shared storage helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelarrayio.storage import utils as storage_utils


def test_resolve_dtype_rejects_unknown_values() -> None:
    assert storage_utils.resolve_dtype('float32') == np.float32
    assert storage_utils.resolve_dtype(np.float64) == np.float64
    with pytest.raises(ValueError, match='Unsupported storage dtype'):
        storage_utils.resolve_dtype('int16')


def test_compute_full_subject_chunk_shape_auto() -> None:
    chunk = storage_utils.compute_full_subject_chunk_shape(
        n_files=4,
        n_elements=128,
        item_chunk=0,
        target_chunk_mb=1.0,
        storage_np_dtype='float32',
    )
    assert chunk[0] == 4
    assert 1 <= chunk[1] <= 128


def test_normalize_column_names_accepts_series_and_sequences() -> None:
    assert storage_utils.normalize_column_names(['a', 2]) == ['a', '2']
    assert storage_utils.normalize_column_names(pd.Series(['x', 'y'])).__class__ is list
