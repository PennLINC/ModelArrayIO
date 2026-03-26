"""Shared storage helpers used by HDF5 and TileDB backends."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_DTYPE_MAP = {
    'float32': np.float32,
    'float64': np.float64,
}


def resolve_dtype(
    storage_dtype: str | np.dtype | type[np.floating],
) -> type[np.float32 | np.float64]:
    """Resolve a configured storage dtype to a supported NumPy floating type."""
    if isinstance(storage_dtype, np.dtype):
        normalized = storage_dtype.name
    else:
        normalized = (
            np.dtype(storage_dtype).name
            if storage_dtype in (np.float32, np.float64)
            else str(storage_dtype).lower()
        )

    try:
        return _DTYPE_MAP[normalized]
    except KeyError as exc:
        supported = ', '.join(sorted(_DTYPE_MAP))
        raise ValueError(
            f'Unsupported storage dtype {storage_dtype!r}. Expected one of: {supported}.'
        ) from exc


def compute_full_subject_chunk_shape(
    num_subjects: int,
    num_items: int,
    item_chunk: int,
    target_chunk_mb: float,
    storage_np_dtype: str | np.dtype | type[np.floating],
) -> tuple[int, int]:
    """Compute a 2-D chunk/tile shape that spans all subjects."""
    num_subjects = int(num_subjects)
    num_items = int(num_items)
    if num_subjects <= 0 or num_items <= 0:
        raise ValueError(
            'Cannot compute chunk shape with zero-length dimension: '
            f'num_subjects={num_subjects}, num_items={num_items}'
        )

    subjects_per_chunk = num_subjects
    if int(item_chunk) > 0:
        items_per_chunk = min(int(item_chunk), num_items)
    else:
        bytes_per_value = np.dtype(resolve_dtype(storage_np_dtype)).itemsize
        target_bytes = float(target_chunk_mb) * 1024.0 * 1024.0
        items_per_chunk = max(1, int(target_bytes / (bytes_per_value * subjects_per_chunk)))
        items_per_chunk = min(items_per_chunk, num_items)

    return subjects_per_chunk, items_per_chunk


def normalize_column_names(sources: Sequence[str] | pd.Series) -> list[str]:
    """Return column names as a list of strings for metadata storage."""
    if isinstance(sources, pd.Series):
        return sources.astype(str).tolist()
    return [str(source) for source in sources]
