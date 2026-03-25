"""Tests for S3 path detection (no boto3 required)."""

from __future__ import annotations

import pytest

from modelarrayio.utils.s3_utils import is_s3_path


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('s3://bucket/key.nii.gz', True),
        ('s3://my-bucket/prefix/sub/file.nii', True),
        ('S3://bucket/oops', False),
        ('', False),
        ('/local/path/file.nii.gz', False),
        ('relative/path.nii', False),
        ('https://example.com/x', False),
    ],
)
def test_is_s3_path(path: str, expected: bool) -> None:
    assert is_s3_path(path) is expected
