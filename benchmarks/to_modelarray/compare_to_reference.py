"""Bitwise-compare a converter output against a reference TileDB conversion.

On CUBIC the original converter's TileDB output of the production CIFTI cohort
(``rbc_freesurferpost.tdb``, written by the stripe writer with one fragment per
column stripe) is the ground truth for the scalars it completed. This script
checks an optimized output — a per-scalar ``.h5`` file or a ``.tdb`` root —
against that reference scalar by scalar, without ever holding an 11 GB matrix
twice: it streams column blocks (old fragments are column stripes, so each
block touches one or two of them), aligns rows by ``column_names`` rather than
by position, and reports shape/dtype/NaN-map/value equality plus the maximum
absolute difference. It also flags reference scalars whose non-empty domain does
not cover the full matrix (the original run stalled part-way through a scalar).

Examples::

    # Which reference scalars are complete?
    python benchmarks/to_modelarray/compare_to_reference.py --reference REF.tdb --list

    # Check an optimized output for one scalar (rows may be a subset).
    python benchmarks/to_modelarray/compare_to_reference.py --reference REF.tdb \
      --candidate out/area_modelarray.h5 --scalar area --output-csv check.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import harness_common as common
import output_compare

FILELOCK_CONFIG = {'vfs.file.enable_filelocks': 'false'}


def _tiledb():
    import tiledb

    return tiledb


def _reference_names(root: Path, scalar: str, array) -> list[str]:
    names = output_compare.names_from_meta(array)
    if names:
        return names
    tiledb = _tiledb()
    column_uri = root / 'scalars' / scalar / 'column_names'
    if tiledb.object_type(str(column_uri)) == 'array':
        with tiledb.open(str(column_uri), 'r') as column_array:
            values = np.asarray(column_array[:]['values']).reshape(-1)
        return [v.decode() if isinstance(v, bytes) else str(v) for v in values]
    return []


def reference_completeness(root: Path) -> list[dict[str, object]]:
    """Report every reference scalar's shape and whether its domain is fully written."""
    tiledb = _tiledb()
    rows = []
    with tiledb.scope_ctx(tiledb.Config(FILELOCK_CONFIG)):
        for scalar_dir in sorted(p for p in (root / 'scalars').glob('*') if p.is_dir()):
            uri = scalar_dir / 'values'
            if tiledb.object_type(str(uri)) != 'array':
                continue
            with tiledb.open(str(uri), 'r') as array:
                n_rows, n_cols = array.shape
                nonempty = array.nonempty_domain()
                complete = nonempty is not None and all(
                    int(low) == 0 and int(high) == extent - 1
                    for (low, high), extent in zip(nonempty, (n_rows, n_cols), strict=True)
                )
                rows.append(
                    {
                        'scalar': scalar_dir.name,
                        'rows': n_rows,
                        'columns': n_cols,
                        'dtype': str(array.dtype),
                        'complete': complete,
                        'nonempty_domain': str(nonempty),
                    }
                )
    return rows


def compare_scalar(
    reference_root: Path,
    candidate_path: Path,
    scalar: str,
    *,
    block_columns: int = 4096,
) -> dict[str, object]:
    """Compare one scalar; rows are aligned by column_names, columns must match."""
    tiledb = _tiledb()
    result: dict[str, object] = {'scalar': scalar, 'candidate': str(candidate_path)}
    reference_uri = reference_root / 'scalars' / scalar / 'values'
    with tiledb.scope_ctx(tiledb.Config(FILELOCK_CONFIG)):
        if tiledb.object_type(str(reference_uri)) != 'array':
            # The original run completed only some scalars; say so instead of crashing.
            result.update(
                {
                    'reference_complete': False,
                    'reference_missing': True,
                    'values_equal': False,
                    'nan_equal': False,
                    'max_abs_difference': float('nan'),
                    'passed': False,
                }
            )
            return result
        result['reference_missing'] = False
        reference = tiledb.open(str(reference_uri), 'r')
        candidate = output_compare.OutputReader(candidate_path, scalar)
        try:
            ref_rows, ref_cols = (int(v) for v in reference.shape)
            ref_names = _reference_names(reference_root, scalar, reference)
            nonempty = reference.nonempty_domain()
            result['reference_complete'] = nonempty is not None and all(
                int(low) == 0 and int(high) == extent - 1
                for (low, high), extent in zip(nonempty, (ref_rows, ref_cols), strict=True)
            )
            result['columns_equal'] = candidate.shape[1] == ref_cols
            result['dtype_equal'] = candidate.dtype == np.dtype(reference.dtype)
            index_of = {name: i for i, name in enumerate(ref_names)}
            missing = [name for name in candidate.names if name not in index_of]
            result['rows_compared'] = len(candidate.names) - len(missing)
            result['rows_missing_in_reference'] = len(missing)
            aligned = bool(candidate.names) and not missing
            result['sources_aligned'] = aligned
            values_equal = aligned and result['columns_equal'] and result['dtype_equal']
            nan_equal = values_equal
            max_abs = 0.0
            if values_equal:
                # Reference rows are re-ordered to the candidate's column_names.
                ref_index = np.asarray([index_of[name] for name in candidate.names])

                def reference_block(start: int, stop: int) -> np.ndarray:
                    return np.asarray(reference[:, start:stop]['values'])[ref_index]

                nan_equal, values_equal, max_abs = output_compare.compare_blocks(
                    reference_block, candidate.block, ref_cols, block_columns
                )
            result['nan_equal'] = nan_equal
            result['values_equal'] = values_equal
            result['max_abs_difference'] = max_abs
            result['passed'] = bool(values_equal and nan_equal and result['reference_complete'])
        finally:
            candidate.close()
            reference.close()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Reference .tdb root')
    parser.add_argument('--list', action='store_true', help='List reference scalar completeness')
    parser.add_argument('--candidate', type=Path, help='Optimized .h5 file or .tdb root')
    parser.add_argument('--scalar', action='append', default=[], dest='scalars')
    parser.add_argument('--block-columns', type=int, default=4096)
    parser.add_argument('--output-csv', type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    reference = args.reference.expanduser().resolve()
    if args.list:
        rows = reference_completeness(reference)
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        if args.output_csv and rows:
            common.write_csv(args.output_csv, rows)
        return 0
    if args.candidate is None or not args.scalars:
        print('--candidate and at least one --scalar are required', file=sys.stderr)
        return 2
    results = [
        compare_scalar(
            reference,
            args.candidate.expanduser().resolve(),
            scalar,
            block_columns=args.block_columns,
        )
        for scalar in args.scalars
    ]
    for row in results:
        print(json.dumps(row, sort_keys=True))
    if args.output_csv:
        common.write_csv(args.output_csv, results)
    return int(not all(row['passed'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
