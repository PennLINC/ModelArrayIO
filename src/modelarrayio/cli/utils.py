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
    scalars: Mapping[str, Sequence[np.ndarray]],
    sources_by_scalar: Mapping[str, Sequence[str]],
    *,
    storage_dtype: str,
    compression: str,
    compression_level: int,
    shuffle: bool,
    chunk_voxels: int,
    target_chunk_mb: float,
) -> None:
    """Write per-scalar matrices into an open HDF5 file."""
    for scalar_name, rows in scalars.items():
        num_subjects = len(rows)
        if num_subjects == 0:
            continue
        num_items = rows[0].shape[0]
        dataset = h5_storage.create_empty_scalar_matrix_dataset(
            h5_file,
            f'scalars/{scalar_name}/values',
            num_subjects,
            num_items,
            storage_dtype=storage_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            chunk_voxels=chunk_voxels,
            target_chunk_mb=target_chunk_mb,
            sources_list=sources_by_scalar[scalar_name],
        )
        h5_storage.write_rows_in_column_stripes(dataset, rows)


def write_tiledb_scalar_matrices(
    output_dir: str | Path,
    scalars: Mapping[str, Sequence[np.ndarray]],
    sources_by_scalar: Mapping[str, Sequence[str]],
    *,
    storage_dtype: str,
    compression: str,
    compression_level: int,
    shuffle: bool,
    chunk_voxels: int,
    target_chunk_mb: float,
    write_column_name_arrays: bool = False,
) -> None:
    """Write per-scalar matrices into a TileDB directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for scalar_name, rows in scalars.items():
        num_subjects = len(rows)
        if num_subjects == 0:
            continue
        num_items = rows[0].shape[0]
        dataset_path = f'scalars/{scalar_name}/values'
        tiledb_storage.create_empty_scalar_matrix_array(
            str(output_path),
            dataset_path,
            num_subjects,
            num_items,
            storage_dtype=storage_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            tile_voxels=chunk_voxels,
            target_tile_mb=target_chunk_mb,
            sources_list=sources_by_scalar[scalar_name],
        )
        if write_column_name_arrays:
            tiledb_storage.write_column_names(
                str(output_path), scalar_name, sources_by_scalar[scalar_name]
            )
        tiledb_storage.write_rows_in_column_stripes(str(output_path / dataset_path), rows)


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
    for name, values in parcel_arrays.items():
        h5_file.create_dataset(
            f'parcels/{name}',
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
