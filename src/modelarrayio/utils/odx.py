"""Utility functions for ODX fixel data (the PennLINC odx-rs Python bindings).

ODX files are the native container produced by ``odx combine``. Each
template-space ODX carries its own group-fixel geometry (offsets + directions +
``compact_to_ijk``) plus per-fixel scalar arrays (DPF), so — unlike the MIF
path — no separate ``index``/``directions`` files are needed.

The cohort consumes **one per-subject ODX per row** (a single-column DPF), the
direct analogue of one ``.mif`` per subject in :mod:`modelarrayio.utils.mif`.
Produce them with ``odx combine --per-subject-odx DIR``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

__all__ = ['gather_fixels_from_odx', 'load_cohort_odx', 'write_odx_results']


def _import_odx():
    """Import the optional ``odx`` package with a helpful error if missing."""
    try:
        import odx
    except ImportError as exc:  # pragma: no cover - exercised only without odx
        raise ImportError(
            "The 'odx' package is required to read ODX fixel data. Install the "
            'odx-rs Python bindings (e.g. `pip install odx`) to use the odx modality.'
        ) from exc
    return odx


def gather_fixels_from_odx(odx_path):
    """Build ``(fixel_table, voxel_table)`` from an ODX file's geometry.

    Mirrors :func:`modelarrayio.utils.mif.gather_fixels`, but sources the fixel
    directions and per-voxel fixel layout straight from an ODX (``offsets`` +
    ``directions`` + ``compact_to_ijk``). ODX lays fixels out in compact-voxel
    order with monotonic ``offsets``, so the voxel/fixel ids need no resorting.

    Parameters
    ----------
    odx_path : path-like
        Path to an ODX file (archive or directory).

    Returns
    -------
    fixel_table : :obj:`pandas.DataFrame`
        Columns ``fixel_id, voxel_id, x, y, z``.
    voxel_table : :obj:`pandas.DataFrame`
        Columns ``voxel_id, i, j, k``.
    """
    odx = _import_odx()
    obj = odx.load(str(odx_path))
    offsets = np.asarray(obj.offsets, dtype=np.int64)  # (n_vox + 1,)
    directions = np.asarray(obj.directions, dtype=np.float32)  # (n_fixels, 3)
    ijk = np.asarray(obj.compact_to_ijk, dtype=np.int64)  # (n_vox, 3)
    n_vox = int(ijk.shape[0])
    counts = np.diff(offsets) if offsets.size else np.zeros(0, dtype=np.int64)
    n_fixels = int(offsets[-1]) if offsets.size else 0

    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(n_vox, dtype=np.int32),
            'i': ijk[:, 0],
            'j': ijk[:, 1],
            'k': ijk[:, 2],
        }
    )
    fixel_voxel_ids = np.repeat(np.arange(n_vox, dtype=np.int32), counts)
    fixel_table = pd.DataFrame(
        {
            'fixel_id': np.arange(n_fixels, dtype=np.int32),
            'voxel_id': fixel_voxel_ids,
            'x': directions[:, 0],
            'y': directions[:, 1],
            'z': directions[:, 2],
        }
    )
    return fixel_table, voxel_table


def _read_odx_column(obj, scalar_name, source):
    """Read one per-subject scalar column (a single-column DPF) from an ODX."""
    names = set(obj.dpf_names())
    if scalar_name not in names:
        raise ValueError(
            f"ODX '{source}' has no per-fixel scalar '{scalar_name}'. "
            f'Available DPF arrays: {sorted(names)}'
        )
    arr = np.asarray(obj.dpf(scalar_name), dtype=np.float32)  # (n_fixels, ncols)
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError(
            f"ODX '{source}' scalar '{scalar_name}' has shape {arr.shape}; the odx "
            'modality expects one per-subject ODX per cohort row (a single-column '
            'DPF). Produce per-subject ODX files with `odx combine --per-subject-odx DIR`.'
        )
    return np.ascontiguousarray(arr[:, 0])


def load_cohort_odx(cohort_long, s3_workers=1):
    """Load all ODX scalar rows from the cohort.

    Each cohort row points to one per-subject ODX with a single-column DPF,
    exactly as the MIF path uses one ``.mif`` per subject. Returns the same
    structures as :func:`modelarrayio.utils.mif.load_cohort_mif`.

    Parameters
    ----------
    cohort_long : :obj:`pandas.DataFrame`
        Long-format cohort dataframe with columns ``scalar_name`` and
        ``source_file``.
    s3_workers : :obj:`int`
        Accepted for signature parity with the MIF loader; ODX reads are
        memory-mapped and run serially.

    Returns
    -------
    scalars : dict[str, list[np.ndarray]]
        Per-scalar ordered list of 1-D subject arrays, ready for stripe-write.
    sources_lists : dict[str, list[str]]
        Per-scalar ordered list of source file paths (HDF5 column metadata).
    """
    odx = _import_odx()
    scalars: dict[str, list[np.ndarray]] = defaultdict(list)
    sources_lists: dict[str, list[str]] = defaultdict(list)
    for row in tqdm(list(cohort_long.itertuples(index=False)), desc='Loading ODX data'):
        scalar_name = str(row.scalar_name)
        source = str(row.source_file)
        obj = odx.load(source)
        scalars[scalar_name].append(_read_odx_column(obj, scalar_name, source))
        sources_lists[scalar_name].append(source)
    return scalars, sources_lists


def write_odx_results(template_odx, results, out_path):
    """Write per-fixel result arrays onto a template ODX's geometry.

    Rebuilds an ODX from ``template_odx``'s group-fixel geometry (mask, affine,
    offsets, directions) and attaches each result as a single-column DPF, so the
    statistics map back onto the same fixels (e.g. for visualization in trxviz).

    Parameters
    ----------
    template_odx : path-like
        An ODX whose geometry defines the output fixels (typically one of the
        per-subject ODX files, or the group ODX, used to build the HDF5).
    results : Mapping[str, numpy.ndarray]
        Maps a DPF name to a 1-D array of length ``n_fixels``.
    out_path : path-like
        Output ODX path (``.odx`` archive or directory).

    Returns
    -------
    out_path : :obj:`pathlib.Path`
    """
    odx = _import_odx()
    tmpl = odx.load(str(template_odx))
    dims = tuple(int(d) for d in tmpl.dimensions)
    affine = np.asarray(tmpl.affine, dtype=np.float64)
    mask = np.ascontiguousarray(np.asarray(tmpl.mask, dtype=np.uint8).reshape(-1))
    offsets = np.asarray(tmpl.offsets, dtype=np.int64)
    directions = np.ascontiguousarray(np.asarray(tmpl.directions, dtype=np.float32))
    n_fixels = int(offsets[-1]) if offsets.size else 0

    builder = odx.OdxBuilder(affine, dims, mask)
    for v in range(offsets.size - 1):
        builder.push_voxel_peaks(directions[offsets[v]:offsets[v + 1]])
    for name, arr in results.items():
        col = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(n_fixels, 1))
        builder.set_dpf(name, col)
    builder.finalize().save(str(out_path))
    return Path(out_path)
