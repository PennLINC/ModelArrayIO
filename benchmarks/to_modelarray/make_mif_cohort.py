"""Fetch and normalize the public MIF (fixel) benchmark cohort.

The ModelArray walkthrough ships a fully public fixel demo — 100 FDC ``.mif``
subjects plus ``index.mif`` and ``directions.mif`` — hosted on OSF. It is the
MIF analogue of the public ABIDE NIfTI cohort: real diffusion fixel data whose
runtime, ``--workers`` scaling and compression ratios are all quotable.

This helper downloads and extracts that archive (unless it is already present)
and writes a normalized long-format ``subject_id,scalar_name,source_file``
cohort CSV with absolute paths, which ``run_benchmarks.py`` drives directly with
``--index-file`` and ``--directions-file``. The archive already contains a
long-format ``cohort_FDC_n100.csv``; normalizing it here just resolves the
relative source paths and drops the extra demographic columns so the harness and
report see a clean cohort.

Reference: https://pennlinc.github.io/ModelArray/articles/walkthrough.html
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import harness_common as common

DEFAULT_URL = 'https://osf.io/download/tce9d'
SCALAR_NAME = 'FDC'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('benchmarks/to_modelarray/data/mif-walkthrough'),
        help='Directory holding (or to receive) the extracted FDC archive.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmarks/to_modelarray/cohorts/mif-walkthrough.csv'),
        help='Normalized long-format cohort CSV to write.',
    )
    parser.add_argument('--url', default=DEFAULT_URL, help='OSF download URL for the FDC archive.')
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Use the already-extracted FDC directory without downloading.',
    )
    return parser


def _download_and_extract(url: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / 'download.tar.gz'
    print(f'Downloading FDC demo archive from {url} ...')
    with urllib.request.urlopen(url) as response, archive.open('wb') as handle:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    print(f'Extracting {archive} ...')
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter='data')
    archive.unlink()


def _find_fdc_dir(data_dir: Path) -> Path:
    for candidate in (data_dir / 'FDC', *sorted(data_dir.glob('**/FDC'))):
        if (candidate / 'index.mif').exists() and (candidate / 'directions.mif').exists():
            return candidate
    raise FileNotFoundError(
        f'Could not find an FDC directory with index.mif and directions.mif under {data_dir}.'
    )


def _subject_rows(fdc_dir: Path) -> list[dict[str, str]]:
    rows = [
        {
            'subject_id': mif_file.stem,
            'scalar_name': SCALAR_NAME,
            'source_file': str(mif_file.resolve()),
        }
        for mif_file in sorted(fdc_dir.glob('sub-*.mif'))
    ]
    if not rows:
        raise FileNotFoundError(f'No sub-*.mif files found in {fdc_dir}.')
    return rows


def make_mif_cohort(args: argparse.Namespace) -> int:
    """Write the normalized MIF cohort and return its row count."""
    if not args.skip_download and not (args.data_dir / 'FDC' / 'index.mif').exists():
        _download_and_extract(args.url, args.data_dir)
    fdc_dir = _find_fdc_dir(args.data_dir)
    rows = _subject_rows(fdc_dir)
    common.write_csv(args.output, rows, common.LONG_COHORT_FIELDS)
    print(
        f'Wrote {len(rows)} rows to {args.output}\n'
        f'  --index-file {fdc_dir / "index.mif"}\n'
        f'  --directions-file {fdc_dir / "directions.mif"}'
    )
    return len(rows)


def main() -> int:
    args = _parser().parse_args()
    make_mif_cohort(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
