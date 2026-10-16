"""Python read gate for converter outputs, mirroring ``r_read_gate.R``.

The R gate needs the R ``tiledb`` package for TileDB outputs, which is not
available in every deployment (for example the only ModelArray container on
CUBIC). This gate performs the same checks from Python: every scalar matrix
opens, its ``column_names`` metadata matches the row count, and a
representative element range reads back. It handles both backends so a
dispatcher can fall back to it for HDF5 too. It prints ``read_seconds=`` like
the R gate and exits non-zero on any failure.

Usage: ``python tiledb_read_gate.py OUTPUT BACKEND [--n-columns N]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _fail(message: str) -> int:
    print(f'gate failed: {message}', file=sys.stderr)
    return 1


def _gate_tiledb(output: Path, n_columns: int) -> int:
    import numpy as np
    import tiledb

    roots = sorted(p for p in (output / 'scalars').glob('*') if p.is_dir())
    uris = [root / 'values' for root in roots if (root / 'values').exists()]
    if not uris:
        return _fail(f'no TileDB scalar matrices under {output}')
    config = tiledb.Config({'vfs.file.enable_filelocks': 'false'})
    with tiledb.scope_ctx(config):
        for uri in uris:
            with tiledb.open(str(uri), 'r') as array:
                rows, cols = array.shape
                names = json.loads(array.meta.get('column_names', '[]'))
                if len(names) != rows:
                    return _fail(f'{uri}: {len(names)} column names for {rows} rows')
                first = array[0, 0]['values']
                block = array[0 : min(64, rows), 0 : min(n_columns, cols)]['values']
                if not np.isfinite(np.asarray(first)).size or block.shape[0] == 0:
                    return _fail(f'{uri}: representative read returned nothing')
    return 0


def _gate_hdf5(output: Path, n_columns: int) -> int:
    import h5py

    with h5py.File(output, 'r') as handle:
        if 'scalars' not in handle:
            return _fail(f'no scalars group in {output}')
        names = list(handle['scalars'])
        if not names:
            return _fail(f'no scalar matrices in {output}')
        for name in names:
            group = handle[f'scalars/{name}']
            if 'values' not in group or 'column_names' not in group:
                return _fail(f'scalars/{name}: missing values or column_names')
            values = group['values']
            rows, cols = values.shape
            if group['column_names'].shape[0] != rows:
                return _fail(f'scalars/{name}: column_names length != rows')
            values[0, 0]
            block = values[0 : min(64, rows), 0 : min(n_columns, cols)]
            if block.shape[0] == 0:
                return _fail(f'scalars/{name}: representative read returned nothing')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('backend', choices=['hdf5', 'tiledb'])
    parser.add_argument('--n-columns', type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    gate = _gate_tiledb if args.backend == 'tiledb' else _gate_hdf5
    status = gate(args.output, args.n_columns)
    if status == 0:
        print(f'read_seconds={time.perf_counter() - started:.6f}')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
