"""Utility functions for loading NIfTI data into staging matrices."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np

from modelarrayio.utils.loading import (
    LazyScalarMatrices,
    resolve_loader_workers,
    stage_source_rows,
)
from modelarrayio.utils.misc import build_scalar_sources, iter_scalar_source_rows
from modelarrayio.utils.s3_utils import is_s3_path, load_nibabel, release_image_buffers

_DEFAULT_MASK_CACHE_SIZE = 128


def _storage_dtype(dtype: str | np.dtype | type[np.floating]) -> np.dtype:
    """Return a nibabel-compatible floating-point dtype."""
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f'storage_dtype must be float32 or float64, got {resolved.name!r}.')
    return resolved


def _path_cache_key(path: str | os.PathLike[str]) -> Hashable:
    """Return a stable key while preserving remote URI semantics."""
    if is_s3_path(path):
        return str(path)
    return Path(path).expanduser().resolve(strict=False)


class ReducedMaskCache:
    """Bounded, thread-safe LRU cache of group-selected boolean masks.

    Concurrent requests for the same path share one in-flight load. Cached
    arrays are read-only so callers cannot make one subject's mask affect a
    later subject.
    """

    def __init__(self, max_size: int = _DEFAULT_MASK_CACHE_SIZE) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError(f'mask cache size must be a positive integer, got {max_size!r}.')
        self.max_size = max_size
        self._values: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._in_flight: dict[Hashable, Future[np.ndarray]] = {}
        self._lock = threading.Lock()

    def get_or_load(
        self,
        path: str | os.PathLike[str],
        loader: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Return the reduced mask for ``path``, loading it at most once concurrently."""
        key = _path_cache_key(path)
        owns_load = False
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                return cached

            pending = self._in_flight.get(key)
            if pending is None:
                pending = Future()
                self._in_flight[key] = pending
                owns_load = True

        if not owns_load:
            return pending.result()

        try:
            reduced = np.array(loader(), dtype=bool, copy=True, order='C')
            if reduced.ndim != 1:
                raise ValueError(
                    f'Reduced mask for {path!r} has shape {reduced.shape}; expected one dimension.'
                )
            reduced.setflags(write=False)
        except BaseException as exc:
            with self._lock:
                self._in_flight.pop(key, None)
            pending.set_exception(exc)
            raise

        with self._lock:
            self._values[key] = reduced
            self._values.move_to_end(key)
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)
            self._in_flight.pop(key, None)
            pending.set_result(reduced)
        return reduced

    def __len__(self) -> int:
        """Return the current number of cached masks."""
        with self._lock:
            return len(self._values)


def _as_nifti_image(source: Any):
    """Return an already-loaded image or load a local/S3 source."""
    return source if hasattr(source, 'get_fdata') else load_nibabel(source)


def _flat_group_indices(group_mask_matrix: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
    """Return the group-mask shape and its C-order selected flat indices."""
    group_mask = np.asarray(group_mask_matrix, dtype=bool)
    flat_indices = np.flatnonzero(group_mask)
    flat_indices.setflags(write=False)
    return group_mask.shape, flat_indices


def _selected_scalar_values(
    scalar_image: Any,
    *,
    group_shape: tuple[int, ...],
    flat_indices: np.ndarray,
    storage_dtype: np.dtype,
) -> np.ndarray:
    """Decode one scalar directly in its storage dtype and select group voxels."""
    loaded_here = not hasattr(scalar_image, 'get_fdata')
    image = _as_nifti_image(scalar_image)
    try:
        scalar_matrix = image.get_fdata(dtype=storage_dtype, caching='unchanged')
        if scalar_matrix.shape != group_shape:
            raise ValueError(
                'Scalar image and group mask must have matching shapes: '
                f'{scalar_matrix.shape} != {group_shape}'
            )

        # np.take with axis=None uses C-order flat indices and returns only the
        # selected row. It never exposes the decoded volume for in-place mutation.
        return np.take(scalar_matrix, flat_indices)
    finally:
        if loaded_here:
            release_image_buffers(image)


def _selected_mask_values(
    scalar_mask: Any,
    *,
    group_shape: tuple[int, ...],
    flat_indices: np.ndarray,
) -> np.ndarray:
    """Decode one mask without forcing float64 and reduce it to group voxels."""
    loaded_here = not hasattr(scalar_mask, 'get_fdata')
    image = _as_nifti_image(scalar_mask)
    try:
        mask_matrix = np.asanyarray(image.dataobj)
        if mask_matrix.shape != group_shape:
            raise ValueError(
                'Scalar mask and group mask must have matching shapes: '
                f'{mask_matrix.shape} != {group_shape}'
            )
        return np.take(mask_matrix, flat_indices) > 0
    finally:
        if loaded_here:
            release_image_buffers(image)


def flattened_image(
    scalar_image,
    scalar_mask,
    group_mask_matrix,
    *,
    storage_dtype: str | np.dtype | type[np.floating] = np.float32,
):
    """Flatten a scalar image to a typed 1-D group-mask selection.

    The decoded scalar volume is never modified. NaNs for an individual mask
    are applied only to the selected row, which keeps repeated image objects
    and repeated paths independent.

    Parameters
    ----------
    scalar_image : nibabel image or path-like
        Scalar image.
    scalar_mask : nibabel image or path-like
        Individual scalar mask.
    group_mask_matrix : numpy.ndarray
        Boolean group mask matrix.
    storage_dtype : {'float32', 'float64'}
        Floating-point dtype to request directly from nibabel.

    Returns
    -------
    numpy.ndarray
        One-dimensional selected scalar row.
    """
    dtype = _storage_dtype(storage_dtype)
    group_shape, flat_indices = _flat_group_indices(group_mask_matrix)
    row = _selected_scalar_values(
        scalar_image,
        group_shape=group_shape,
        flat_indices=flat_indices,
        storage_dtype=dtype,
    )
    reduced_mask = _selected_mask_values(
        scalar_mask,
        group_shape=group_shape,
        flat_indices=flat_indices,
    )
    row[~reduced_mask] = np.nan
    return row


def load_cohort_voxels(
    cohort_long,
    group_mask_matrix,
    s3_workers=None,
    *,
    workers=None,
    storage_dtype='float32',
    group_mask_path=None,
    mask_cache_size=None,
):
    """Return lazily staged per-scalar voxel matrices and ordered source paths.

    The third positional argument remains the legacy ``s3_workers`` count.
    New callers should pass ``workers``. Accessing one key in the returned
    mapping allocates and fills only that scalar's typed C-contiguous staging
    matrix, with no more than ``workers`` source loads submitted at once.

    Parameters
    ----------
    cohort_long : pandas.DataFrame
        Long-format cohort dataframe with ``scalar_name``, ``source_file``,
        and ``source_mask_file`` columns.
    group_mask_matrix : numpy.ndarray
        Boolean group mask array.
    s3_workers : int, optional
        Deprecated positional name for the loader worker count.
    workers : int, optional
        Maximum concurrent source loads.
    storage_dtype : {'float32', 'float64'}
        Dtype used for both nibabel decoding and the staging matrix.
    group_mask_path : path-like, optional
        Source path of ``group_mask_matrix``. A matching individual-mask path
        is not opened because it cannot exclude any selected group voxel.
    mask_cache_size : int, optional
        Maximum number of reduced individual masks retained. Defaults to 128.

    Returns
    -------
    scalars : modelarrayio.utils.loading.LazyScalarMatrices
        Mapping from scalar name to a lazily constructed 2-D numpy array.
    sources_by_scalar : dict[str, list[str]]
        Source paths in exact cohort order for each scalar.
    """
    worker_count = resolve_loader_workers(workers, s3_workers)
    dtype = _storage_dtype(storage_dtype)
    group_shape, flat_indices = _flat_group_indices(group_mask_matrix)
    if flat_indices.size == 0:
        raise ValueError('Cannot load NIfTI cohort: the group mask selects no voxels.')

    cache_size = _DEFAULT_MASK_CACHE_SIZE if mask_cache_size is None else mask_cache_size
    mask_cache = ReducedMaskCache(cache_size)
    group_mask_key = _path_cache_key(group_mask_path) if group_mask_path is not None else None

    sources_by_scalar = build_scalar_sources(cohort_long)
    source_pairs_by_scalar: dict[str, list[tuple[str, str]]] = {}
    for scalar_name, source_file, cohort_row in iter_scalar_source_rows(cohort_long):
        source_pairs_by_scalar.setdefault(scalar_name, []).append(
            (source_file, cohort_row.source_mask_file)
        )

    def _load_row(source_pair: tuple[str, str]) -> np.ndarray:
        source_file, source_mask_file = source_pair
        row = _selected_scalar_values(
            source_file,
            group_shape=group_shape,
            flat_indices=flat_indices,
            storage_dtype=dtype,
        )

        mask_key = _path_cache_key(source_mask_file)
        if group_mask_key is not None and mask_key == group_mask_key:
            return row

        reduced_mask = mask_cache.get_or_load(
            source_mask_file,
            lambda: _selected_mask_values(
                source_mask_file,
                group_shape=group_shape,
                flat_indices=flat_indices,
            ),
        )
        row[~reduced_mask] = np.nan
        return row

    scalar_names = tuple(sources_by_scalar)

    def _stage_scalar(scalar_name: str):
        return stage_source_rows(
            source_pairs_by_scalar[scalar_name],
            _load_row,
            scalar_name=scalar_name,
            storage_dtype=dtype,
            workers=worker_count,
            n_elements=int(flat_indices.size),
        )

    # The flat selection and the reduced-mask cache ride along explicitly, so
    # callers and tests need neither a second group-mask scan nor the volumes.
    scalars = LazyScalarMatrices(
        scalar_names,
        _stage_scalar,
        sources_by_scalar=sources_by_scalar,
        flat_indices=flat_indices,
        mask_cache=mask_cache,
    )
    return scalars, sources_by_scalar
