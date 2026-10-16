"""Stream-compare ModelArray outputs (an HDF5 file or a TileDB root) one column block at a time.

Shared by ``compare_revisions.py`` (original vs optimized output), ``compare_to_reference.py``
(optimized output vs a deployed reference) and ``create_comparison_report.py`` (output shapes).
``h5py`` and ``tiledb`` are imported only when an output is opened, because libtiledb cannot
load on CPUs without AVX2 (the CUBIC login node) and the scripts' ``--help`` must still work there.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np

BlockReader = Callable[[int, int], np.ndarray]


def names_from_meta(array) -> list[str]:
    """Return the ``column_names`` a TileDB values array stores in its metadata."""
    raw = array.meta.get('column_names')
    return [str(value) for value in json.loads(raw)] if raw is not None else []


def _hdf5_column_names(handle, scalar: str) -> list[str]:
    group = handle[f'scalars/{scalar}']
    if 'column_names' in group:
        return [str(value) for value in group['column_names'].asstr()[:]]
    # Older outputs stored the names as an attribute of the values dataset.
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in group['values'].attrs.get('column_names', [])
    ]


class OutputReader:
    """Column-block reader over one scalar of an HDF5 file or a TileDB root."""

    def __init__(self, path: Path, scalar: str):
        self._h5 = None
        self._array = None
        if path.is_file():
            import h5py

            self._h5 = h5py.File(path, 'r')
            self._dataset = self._h5[f'scalars/{scalar}/values']
            self.shape = tuple(int(value) for value in self._dataset.shape)
            self.dtype = np.dtype(self._dataset.dtype)
            self.names = _hdf5_column_names(self._h5, scalar)
        else:
            import tiledb

            self._array = tiledb.open(str(path / 'scalars' / scalar / 'values'), 'r')
            self.shape = tuple(int(value) for value in self._array.shape)
            self.dtype = np.dtype(self._array.dtype)
            self.names = names_from_meta(self._array)

    def block(self, start: int, stop: int) -> np.ndarray:
        if self._h5 is not None:
            return np.asarray(self._dataset[:, start:stop])
        return np.asarray(self._array[:, start:stop]['values'])

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        if self._array is not None:
            self._array.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def compare_blocks(
    left: BlockReader, right: BlockReader, n_columns: int, block_columns: int = 4096
) -> tuple[bool, bool, float]:
    """Return ``(nan_equal, values_equal, max_abs_difference)`` over equal-shape matrices."""
    nan_equal = values_equal = True
    max_abs_difference = 0.0
    for start in range(0, n_columns, block_columns):
        stop = min(start + block_columns, n_columns)
        left_block = left(start, stop)
        right_block = right(start, stop)
        left_nan = np.isnan(left_block)
        right_nan = np.isnan(right_block)
        nan_equal = nan_equal and np.array_equal(left_nan, right_nan)
        finite = ~left_nan & ~right_nan
        if np.any(finite):
            difference = np.max(np.abs(left_block[finite] - right_block[finite]))
            max_abs_difference = max(max_abs_difference, float(difference))
        values_equal = values_equal and np.array_equal(left_block, right_block, equal_nan=True)
    return nan_equal, values_equal, max_abs_difference


def compare_outputs(
    left: Path, right: Path, scalar: str, block_columns: int = 4096
) -> dict[str, object]:
    """Compare one scalar of two outputs written by the same backend."""
    with OutputReader(left, scalar) as left_reader, OutputReader(right, scalar) as right_reader:
        shape_equal = left_reader.shape == right_reader.shape
        nan_equal = values_equal = False
        max_abs_difference = 0.0
        if shape_equal:
            nan_equal, values_equal, max_abs_difference = compare_blocks(
                left_reader.block, right_reader.block, left_reader.shape[1], block_columns
            )
        return {
            'shape_equal': shape_equal,
            'dtype_equal': left_reader.dtype == right_reader.dtype,
            'sources_equal': left_reader.names == right_reader.names,
            'nan_equal': nan_equal,
            'values_equal': values_equal,
            'max_abs_difference': max_abs_difference,
        }
