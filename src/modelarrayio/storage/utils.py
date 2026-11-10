"""Shared storage helpers used by HDF5 and TileDB backends."""

from __future__ import annotations

import operator
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

    try:
        if isinstance(item_chunk, bool):
            raise TypeError('a boolean is not an item chunk')  # bool passes operator.index
        resolved_item_chunk = operator.index(item_chunk)
    except TypeError as exc:
        raise ValueError(f'item chunk must be a non-negative integer, got {item_chunk!r}') from exc
    if resolved_item_chunk < 0:
        raise ValueError(f'item chunk must be non-negative, got {resolved_item_chunk}')

    subjects_per_chunk = n_files
    if resolved_item_chunk > 0:
        items_per_chunk = min(resolved_item_chunk, n_elements)
    else:
        bytes_per_value = np.dtype(resolve_dtype(storage_np_dtype)).itemsize
        target_bytes = float(target_chunk_mb) * 1024.0 * 1024.0
        if not np.isfinite(target_bytes) or target_bytes <= 0:
            raise ValueError(
                f'target chunk size must be a positive finite value, got {target_chunk_mb!r}'
            )
        items_per_chunk = max(1, int(target_bytes / (bytes_per_value * subjects_per_chunk)))
        items_per_chunk = min(items_per_chunk, n_elements)

    return subjects_per_chunk, items_per_chunk


def normalize_column_names(sources: Sequence[str] | pd.Series) -> list[str]:
    """Return column names as a list of strings for metadata storage."""
    if isinstance(sources, pd.Series):
        return sources.astype(str).tolist()
    return [str(source) for source in sources]


def validate_scalar_name(scalar_name: object) -> str:
    """Return a scalar identifier that is safe as one storage path segment."""
    name = str(scalar_name)
    if not name.strip():
        raise ValueError('Scalar names must not be empty or whitespace-only.')
    if '\x00' in name or '/' in name or '\\' in name or name in {'.', '..'}:
        raise ValueError(
            f'Unsafe scalar name {name!r}: names must be one path segment without '
            "'/', '\\', NUL, '.' or '..'."
        )
    return name


def resolve_backend_compression(
    backend: str,
    compression: str | None,
    compression_level: int | None,
) -> tuple[str, int | None]:
    """Resolve a supported backend codec and its effective level."""
    backend_name = str(backend).lower()
    supported = {
        'hdf5': {'gzip', 'lzf', 'none'},
        'tiledb': {'gzip', 'zstd', 'none'},
    }
    if backend_name not in supported:
        raise ValueError(f'Unsupported storage backend {backend!r}.')

    codec = (
        ('gzip' if backend_name == 'hdf5' else 'zstd')
        if compression is None
        else str(compression).lower()
    )
    if codec not in supported[backend_name]:
        choices = ', '.join(sorted(supported[backend_name]))
        raise ValueError(
            f'Compression {codec!r} is not supported by {backend_name}; choose one of: {choices}.'
        )
    if codec in {'none', 'lzf'}:
        return codec, None

    default_level = 4 if codec == 'gzip' else 5
    if compression_level is None:
        return codec, default_level
    try:
        if isinstance(compression_level, bool):
            raise TypeError('a boolean is not a compression level')  # bool passes operator.index
        level = operator.index(compression_level)
    except TypeError as exc:
        raise ValueError(
            f'compression level must be an integer, got {compression_level!r}'
        ) from exc
    if codec == 'gzip' and not 0 <= level <= 9:
        raise ValueError(f'gzip compression level must be between 0 and 9, got {level}.')
    return codec, level


def coerce_full_matrix(
    values: np.ndarray | Sequence[np.ndarray],
    *,
    dtype: str | np.dtype | type[np.floating],
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return one C-contiguous matrix, preserving an already-staged matrix.

    A 2-D array of the requested dtype is returned as-is (no copy). A sequence
    of 1-D rows is stacked into a new matrix. ``shape`` is enforced when given.
    """
    target_dtype = np.dtype(dtype)
    if isinstance(values, np.ndarray):
        if values.ndim != 2:
            raise ValueError(f'Expected a 2D scalar matrix, got shape {values.shape!r}')
        matrix = values
        if matrix.dtype != target_dtype:
            matrix = matrix.astype(target_dtype, copy=False)
        if not matrix.flags.c_contiguous:
            matrix = np.ascontiguousarray(matrix)
    else:
        rows = list(values)
        if shape is None:
            if not rows:
                raise ValueError('Cannot build a scalar matrix from an empty row sequence')
            first = np.asarray(rows[0])
            if first.ndim != 1:
                raise ValueError(f'Expected 1D scalar rows, got shape {first.shape!r}')
            shape = (len(rows), first.shape[0])
        if len(rows) != shape[0]:
            raise ValueError('rows length does not match the subjects dimension')
        matrix = np.empty(shape, dtype=target_dtype, order='C')
        for index, row in enumerate(rows):
            row_array = np.asarray(row)
            if row_array.ndim != 1 or row_array.shape[0] != shape[1]:
                raise ValueError(
                    f'Row {index} has shape {row_array.shape!r}; expected ({shape[1]},)'
                )
            matrix[index] = row_array

    if shape is not None and matrix.shape != shape:
        raise ValueError(f'Matrix shape {matrix.shape!r} does not match expected shape {shape!r}')
    return matrix
