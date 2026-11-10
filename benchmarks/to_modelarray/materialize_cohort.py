"""Download every ``s3://`` source in a cohort CSV and write a local-path cohort.

Public benchmark cohorts (``make_cohort.py``) reference ``s3://fcp-indi`` keys.
Cluster compute nodes may have no internet, so the files are fetched once on a
login node with unsigned requests and the cohort is rewritten to absolute local
paths. Existing files are kept, so re-running is cheap. Every ``s3://`` cell in
any column is materialized, which covers scalar columns and
``source_mask_file`` alike.

Example::

    python benchmarks/to_modelarray/materialize_cohort.py \
      --cohort benchmarks/to_modelarray/cohorts/abide-small.csv \
      --data-dir "$OPT_ROOT/data/abide" \
      --output "$OPT_ROOT/cohorts/abide-small-local.csv"
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import harness_common as common


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--extra-key',
        action='append',
        default=[],
        help='Additional s3:// object to download (for example the group mask).',
    )
    return parser


def local_path_for(uri: str, data_dir: Path) -> Path:
    """Map ``s3://bucket/a/b/derivative/file`` to ``data_dir/derivative/file``."""
    parsed = urlparse(uri)
    key = Path(parsed.path.lstrip('/'))
    return data_dir / key.parent.name / key.name


def _download(client, uri: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    parsed = urlparse(uri)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    client.download_file(parsed.netloc, parsed.path.lstrip('/'), str(temporary))
    temporary.replace(target)


def materialize(args: argparse.Namespace, client=None) -> int:
    """Download all s3 cells (and extra keys), write the local cohort, return file count."""
    with args.cohort.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    uris = {value for row in rows for value in row.values() if value and value.startswith('s3://')}
    uris.update(args.extra_key)
    if client is None and uris:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    targets = {uri: local_path_for(uri, args.data_dir) for uri in sorted(uris)}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(lambda item: _download(client, item[0], item[1]), targets.items()))

    for row in rows:
        for column, value in row.items():
            if value and value.startswith('s3://'):
                row[column] = str(targets[value].resolve())
    common.write_csv(args.output, rows, fieldnames)
    return len(targets)


def main() -> int:
    args = _parser().parse_args()
    count = materialize(args)
    print(f'Materialized {count} objects under {args.data_dir}; wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
