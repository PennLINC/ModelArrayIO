"""Create deterministic wide-format ABIDE cohort CSVs from public S3 keys."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import harness_common as common

DEFAULT_BUCKET = 'fcp-indi'
DEFAULT_PREFIX = 'data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global'
DEFAULT_DERIVATIVES = ('alff',)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subjects', type=int, default=100)
    parser.add_argument('--derivatives', nargs='+', default=list(DEFAULT_DERIVATIVES))
    parser.add_argument('--repeat-factor', type=int, default=1)
    parser.add_argument('--bucket', default=DEFAULT_BUCKET)
    parser.add_argument('--prefix', default=DEFAULT_PREFIX)
    parser.add_argument(
        '--subject-manifest',
        type=Path,
        help='Optional newline-delimited subject IDs; avoids listing S3 and is useful offline.',
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.subjects < 1:
        raise ValueError('--subjects must be positive')
    if args.repeat_factor < 1:
        raise ValueError('--repeat-factor must be positive')
    for derivative in args.derivatives:
        if not re.fullmatch(r'[A-Za-z0-9_]+', derivative):
            raise ValueError(f'Invalid derivative name: {derivative!r}')


def _unsigned_s3_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            'Listing the public ABIDE bucket requires boto3; install modelarrayio[s3] '
            'or pass --subject-manifest.'
        ) from exc
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))


def _subjects_for_derivative(client, bucket: str, prefix: str, derivative: str) -> set[str]:
    key_prefix = f'{prefix.rstrip("/")}/{derivative}/'
    suffix = f'_{derivative}.nii.gz'
    subjects: set[str] = set()
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for item in page.get('Contents', []):
            filename = str(item['Key']).rsplit('/', 1)[-1]
            if filename.endswith(suffix):
                subjects.add(filename[: -len(suffix)])
    return subjects


def _available_subjects(args: argparse.Namespace) -> list[str]:
    if args.subject_manifest is not None:
        subjects = {
            line.strip()
            for line in args.subject_manifest.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        }
        return sorted(subjects)

    client = _unsigned_s3_client()
    required = [
        _subjects_for_derivative(client, args.bucket, args.prefix, derivative)
        for derivative in [*args.derivatives, 'func_mask']
    ]
    return sorted(set.intersection(*required)) if required else []


def _s3_uri(bucket: str, prefix: str, derivative: str, subject: str) -> str:
    return f's3://{bucket}/{prefix.rstrip("/")}/{derivative}/{subject}_{derivative}.nii.gz'


def make_cohort(args: argparse.Namespace) -> int:
    """Write the requested deterministic cohort and return its row count."""
    _validate_args(args)
    available = _available_subjects(args)
    if len(available) < args.subjects:
        raise ValueError(
            f'Requested {args.subjects} subjects, but only {len(available)} have every '
            f'requested derivative and func_mask.'
        )
    selected = available[: args.subjects]
    rows = []
    for repeat_index in range(args.repeat_factor):
        for subject in selected:
            row = {
                'subject_id': (
                    subject
                    if args.repeat_factor == 1
                    else f'{subject}__repeat-{repeat_index + 1:04d}'
                ),
                'source_mask_file': _s3_uri(args.bucket, args.prefix, 'func_mask', subject),
            }
            row.update(
                {
                    derivative: _s3_uri(args.bucket, args.prefix, derivative, subject)
                    for derivative in args.derivatives
                }
            )
            rows.append(row)
    common.write_csv(args.output, rows, ['subject_id', 'source_mask_file', *args.derivatives])
    return len(rows)


def main() -> int:
    args = _parser().parse_args()
    row_count = make_cohort(args)
    print(f'Wrote {row_count} rows to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
