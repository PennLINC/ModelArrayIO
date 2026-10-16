"""Shared CLI helpers for logging, outputs, and result metadata."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from modelarrayio.storage import h5_storage, tiledb_storage
from modelarrayio.storage import utils as storage_utils


def configure_logging(log_level: str) -> None:
    """Configure package logging once for CLI entry points."""
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format='[%(levelname)s] %(name)s: %(message)s',
    )


def prepare_output_directory(output_dir: str | Path, logger: logging.Logger) -> Path:
    """Create an output directory and warn when reusing an existing path."""
    output_path = Path(output_dir)
    if output_path.exists():
        logger.warning('Output directory exists: %s', output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def prepare_output_parent(output_file: str | Path) -> Path:
    """Ensure the parent directory for an output file exists."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def prefixed_output_path(output_path: str | Path, prefix: str) -> Path:
    """Return output path with a sanitized prefix added to its filename."""
    path = Path(output_path)
    safe_prefix = sanitize_result_name(prefix)
    return path.with_name(f'{safe_prefix}_{path.name}')


def conversion_output_paths(
    output_path: str | Path,
    scalar_names: Sequence[str],
    *,
    split_outputs: bool,
) -> list[Path]:
    """Return final output paths and reject colliding split filenames."""
    if not split_outputs:
        for scalar_name in scalar_names:
            storage_utils.validate_scalar_name(scalar_name)
        return [Path(output_path)]

    validated_names = [storage_utils.validate_scalar_name(name) for name in scalar_names]
    paths = [prefixed_output_path(output_path, scalar_name) for scalar_name in validated_names]
    collision_keys: dict[str, tuple[str, Path]] = {}
    for scalar_name, path in zip(validated_names, paths, strict=True):
        key = str(path.resolve(strict=False)).casefold()
        if key in collision_keys:
            previous_name, previous_path = collision_keys[key]
            raise ValueError(
                f'Scalar names {previous_name!r} and {str(scalar_name)!r} map to the same '
                f'split output {str(previous_path)!r}.'
            )
        collision_keys[key] = (str(scalar_name), path)
    return paths


def write_table_dataset(
    h5_file: h5py.File,
    dataset_name: str,
    table: pd.DataFrame,
    *,
    extra_attrs: Mapping[str, Sequence[str]] | None = None,
) -> h5py.Dataset:
    """Write a dataframe as a transposed HDF5 dataset with column metadata."""
    dataset = h5_file.create_dataset(name=dataset_name, data=table.to_numpy().T)
    dataset.attrs['column_names'] = list(table.columns)
    for key, value in (extra_attrs or {}).items():
        dataset.attrs[key] = list(value)
    return dataset


def write_hdf5_scalar_matrices(
    h5_file: h5py.File,
    scalars: Mapping[str, np.ndarray | Sequence[np.ndarray]],
    sources_by_scalar: Mapping[str, Sequence[str]],
    *,
    storage_dtype: str,
    compression: str,
    compression_level: int,
    shuffle: bool,
    chunk_voxels: int,
    target_chunk_mb: float,
) -> dict[str, dict[str, object]]:
    """Write each scalar as one full matrix into an open HDF5 file."""
    metrics: dict[str, dict[str, object]] = {}
    for scalar_name, matrix in scalars.items():
        scalar_name = storage_utils.validate_scalar_name(scalar_name)
        n_files = len(matrix)
        if n_files == 0:
            continue
        scalar_metrics: dict[str, object] = {}
        h5_storage.create_scalar_matrix_dataset(
            h5_file,
            f'scalars/{scalar_name}/values',
            matrix,
            sources_by_scalar[scalar_name],
            storage_dtype=storage_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            chunk_voxels=chunk_voxels,
            target_chunk_mb=target_chunk_mb,
            metrics=scalar_metrics,
        )
        scalar_metrics['backend'] = 'hdf5'
        scalar_metrics['scalar_name'] = str(scalar_name)
        metrics[str(scalar_name)] = scalar_metrics
    return metrics


def write_tiledb_scalar_matrices(
    output_dir: str | Path,
    scalars: Mapping[str, np.ndarray | Sequence[np.ndarray]],
    sources_by_scalar: Mapping[str, Sequence[str]],
    *,
    storage_dtype: str,
    compression: str,
    compression_level: int,
    shuffle: bool,
    chunk_voxels: int,
    target_chunk_mb: float,
    write_column_name_arrays: bool = False,
) -> dict[str, dict[str, object]]:
    """Write each scalar in one dense submission into a TileDB directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, object]] = {}

    for scalar_name, matrix in scalars.items():
        scalar_name = storage_utils.validate_scalar_name(scalar_name)
        n_files = len(matrix)
        if n_files == 0:
            continue
        dataset_path = f'scalars/{scalar_name}/values'
        scalar_metrics: dict[str, object] = {}
        tiledb_storage.create_scalar_matrix_array(
            str(output_path),
            dataset_path,
            matrix,
            sources_by_scalar[scalar_name],
            storage_dtype=storage_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            tile_voxels=chunk_voxels,
            target_tile_mb=target_chunk_mb,
            metrics=scalar_metrics,
        )
        if write_column_name_arrays:
            tiledb_storage.write_column_names(
                str(output_path), scalar_name, sources_by_scalar[scalar_name]
            )
        scalar_metrics['backend'] = 'tiledb'
        scalar_metrics['scalar_name'] = str(scalar_name)
        metrics[str(scalar_name)] = scalar_metrics
    return metrics


def write_hdf5_parcel_arrays(
    h5_file: h5py.File,
    parcel_arrays: Mapping[str, np.ndarray],
) -> None:
    """Write parcellated CIFTI parcel name arrays as HDF5 string datasets.

    Creates one dataset per entry under the ``parcels/`` group.

    Parameters
    ----------
    h5_file : h5py.File
        Open, writable HDF5 file.
    parcel_arrays : mapping
        Keys are dataset names (e.g. ``'parcel_id'``, ``'parcel_id_from'``,
        ``'parcel_id_to'``); values are arrays of parcel name strings.
    """
    grp = h5_file.require_group('parcels')
    for name, values in parcel_arrays.items():
        grp.create_dataset(
            name,
            data=np.array(values, dtype=object),
            dtype=h5py.string_dtype(),
        )


def write_tiledb_parcel_arrays(
    base_uri: str | Path,
    parcel_arrays: Mapping[str, np.ndarray],
) -> None:
    """Write parcellated CIFTI parcel name arrays as TileDB string arrays.

    Creates one TileDB array per entry under the ``parcels/`` sub-path.

    Parameters
    ----------
    base_uri : str or Path
        Root directory of the TileDB store.
    parcel_arrays : mapping
        Keys are array names (e.g. ``'parcel_id'``); values are arrays of
        parcel name strings.
    """
    for name, values in parcel_arrays.items():
        tiledb_storage.write_parcel_names(
            str(base_uri),
            os.path.join('parcels', name),
            [str(v) for v in values],
        )


def sanitize_result_name(result_name: str) -> str:
    """Normalize an analysis result name for use in filenames."""
    return str(result_name).replace(' ', '_').replace('/', '_')


def read_result_names(
    h5_file: h5py.File,
    analysis_name: str,
    results_matrix: h5py.Dataset,
    *,
    logger: logging.Logger,
) -> list[str]:
    """Read result names from HDF5 metadata, with compatibility fallbacks."""
    names_attr = results_matrix.attrs.get('colnames')
    if names_attr is not None:
        decoded = _decode_names(names_attr)
        if decoded:
            return decoded

    candidate_paths = (
        f'results/{analysis_name}/column_names',
        f'results/{analysis_name}/results_matrix/column_names',
    )
    for path in candidate_paths:
        if path not in h5_file:
            continue
        try:
            decoded = _decode_names(h5_file[path][()])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug('Could not read column names from %s', path, exc_info=True)
            continue
        if decoded:
            return decoded

    logger.warning("Unable to read column names, using 'componentNNN' instead")
    return [f'component{n + 1:03d}' for n in range(results_matrix.shape[0])]


def _decode_names(values: object) -> list[str]:
    if isinstance(values, np.ndarray):
        sequence = values.tolist()
    elif isinstance(values, (list, tuple)):
        sequence = list(values)
    else:
        sequence = [values]

    decoded: list[str] = []
    for value in sequence:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            text = value.decode('utf-8', errors='ignore')
        else:
            text = str(value)
        decoded.append(text.rstrip('\x00').strip())
    return [name for name in decoded if name]
