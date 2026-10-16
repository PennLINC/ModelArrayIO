"""Cohort, CSV and provenance helpers shared by the ``to-modelarray`` benchmark scripts.

The scripts import this module as a sibling: a script run by path has its own directory on
``sys.path``, and the test suite adds this directory through pytest's ``pythonpath`` setting.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import platform
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

PACKAGE_NAMES = ('modelarrayio', 'numpy', 'nibabel', 'h5py', 'tiledb')
# Header of a long-format cohort, which the converter combines into one output.
LONG_COHORT_FIELDS = ('subject_id', 'scalar_name', 'source_file')


def read_cohort(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str] | None = None
) -> None:
    """Write ``rows`` with ``\\n`` line endings, which shell consumers such as ``cut`` need.

    Without ``fieldnames`` the header is every key in first-seen order across the rows.
    """
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def output_path(case_dir: Path, backend: str, scalar: str | None = None) -> Path:
    """Return the converter output in ``case_dir``: the requested ``modelarray.<ext>`` path, or
    the ``<scalar>_modelarray.<ext>`` file it writes for one scalar of a split (wide) cohort."""
    suffix = '.h5' if backend == 'hdf5' else '.tdb'
    return case_dir / (f'modelarray{suffix}' if scalar is None else f'{scalar}_modelarray{suffix}')


def source_modality(paths: list[str]) -> str:
    """Return the single modality (``nifti``, ``cifti`` or ``mif``) shared by ``paths``."""
    modalities = set()
    for source in paths:
        path = urlparse(source).path.lower()
        if path.endswith(('.dscalar.nii', '.pscalar.nii', '.pconn.nii')):
            modalities.add('cifti')
        elif path.endswith(('.mif', '.mif.gz')):
            modalities.add('mif')
        elif path.endswith(('.nii', '.nii.gz')):
            modalities.add('nifti')
        else:
            raise ValueError(f'Cannot infer benchmark modality from {source!r}.')
    if len(modalities) != 1:
        raise ValueError(f'Benchmark cohort must contain one modality, got {sorted(modalities)}.')
    return modalities.pop()


def infer_scalar_columns(rows: list[dict[str, str]]) -> list[str]:
    """Return wide-format scalar columns, or [] for a long-format cohort."""
    if not rows:
        raise ValueError('Cohort CSV has no rows.')
    if {'scalar_name', 'source_file'} <= rows[0].keys():
        return []
    excluded = {'subject_id', 'source_mask_file'}
    columns = []
    for column in rows[0]:
        if column in excluded:
            continue
        values = [row[column] for row in rows if row.get(column)]
        if not values:
            continue
        try:
            source_modality(values)
        except ValueError:
            continue
        columns.append(column)
    if not columns:
        raise ValueError('Could not infer wide-format scalar columns; pass --scalar-columns.')
    return columns


def scalar_source_paths(rows: list[dict[str, str]], scalar_columns: list[str]) -> list[str]:
    """Return scalar inputs without companion mask paths."""
    if scalar_columns:
        return [row[column] for row in rows for column in scalar_columns if row.get(column)]
    return [row['source_file'] for row in rows if row.get('source_file')]


def scalar_names(rows: list[dict[str, str]], scalar_columns: list[str]) -> list[str]:
    if scalar_columns:
        return list(scalar_columns)
    return list(dict.fromkeys(row['scalar_name'] for row in rows if row.get('scalar_name')))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def output_stats(path: Path) -> tuple[int, int]:
    """Return (total bytes, file count) for an output file or directory tree."""
    files = [path] if path.is_file() else [child for child in path.rglob('*') if child.is_file()]
    return sum(child.stat().st_size for child in files), len(files)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {'python': platform.python_version()}
    for package in PACKAGE_NAMES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result
