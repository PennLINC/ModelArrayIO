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
    n_files: int,
    n_elements: int,
    item_chunk: int,
    target_chunk_mb: float,
    storage_np_dtype: str | np.dtype | type[np.floating],
) -> tuple[int, int]:
    """Compute a 2-D chunk/tile shape that spans all subjects."""
    n_files = int(n_files)
    n_elements = int(n_elements)
    if n_files <= 0 or n_elements <= 0:
        raise ValueError(
            'Cannot compute chunk shape with zero-length dimension: '
            f'n_files={n_files}, n_elements={n_elements}'
        )

    subjects_per_chunk = n_files
    if int(item_chunk) > 0:
        items_per_chunk = min(int(item_chunk), n_elements)
    else:
        bytes_per_value = np.dtype(resolve_dtype(storage_np_dtype)).itemsize
        target_bytes = float(target_chunk_mb) * 1024.0 * 1024.0
        items_per_chunk = max(1, int(target_bytes / (bytes_per_value * subjects_per_chunk)))
        items_per_chunk = min(items_per_chunk, n_elements)

    return subjects_per_chunk, items_per_chunk


def normalize_column_names(sources: Sequence[str] | pd.Series) -> list[str]:
    """Return column names as a list of strings for metadata storage."""
    if isinstance(sources, pd.Series):
        return sources.astype(str).tolist()
    return [str(source) for source in sources]
