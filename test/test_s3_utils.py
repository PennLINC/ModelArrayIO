"""Tests for S3 path detection (no boto3 required)."""

from __future__ import annotations

import sys
import types

import pytest

from modelarrayio.utils import s3_utils
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


def test_s3_client_is_shared_and_pool_matches_workers(monkeypatch) -> None:
    calls = []
    client = object()

    class _FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            calls.append((service, kwargs))
            return client

    monkeypatch.setitem(sys.modules, 'boto3', _FakeBoto3)
    monkeypatch.setitem(sys.modules, 'botocore', types.SimpleNamespace(UNSIGNED='unsigned'))
    monkeypatch.setitem(
        sys.modules,
        'botocore.config',
        types.SimpleNamespace(Config=lambda **kwargs: ('config', kwargs)),
    )
    monkeypatch.setenv('MODELARRAYIO_S3_ANON', '1')
    s3_utils.clear_s3_client_cache()
    s3_utils.configure_s3_client_pool(23)

    assert s3_utils._make_s3_client() is client
    assert s3_utils._make_s3_client() is client
    assert calls == [
        (
            's3',
            {
                'config': (
                    'config',
                    {'max_pool_connections': 23, 'signature_version': 'unsigned'},
                )
            },
        )
    ]

    s3_utils.clear_s3_client_cache()
    s3_utils.configure_s3_client_pool(1)


def test_s3_client_cache_closes_replaced_and_cleared_clients(monkeypatch) -> None:
    clients = []

    class _Client:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeBoto3:
        @staticmethod
        def client(_service, **_kwargs):
            client = _Client()
            clients.append(client)
            return client

    monkeypatch.setitem(sys.modules, 'boto3', _FakeBoto3)
    monkeypatch.setitem(sys.modules, 'botocore', types.SimpleNamespace(UNSIGNED='unsigned'))
    monkeypatch.setitem(
        sys.modules,
        'botocore.config',
        types.SimpleNamespace(Config=lambda **kwargs: kwargs),
    )
    monkeypatch.setenv('MODELARRAYIO_S3_ANON', '0')
    s3_utils.clear_s3_client_cache()
    s3_utils.configure_s3_client_pool(10)

    first = s3_utils._make_s3_client()
    s3_utils.configure_s3_client_pool(24)
    assert first.closed is True
    second = s3_utils._make_s3_client()
    assert second is not first

    s3_utils.clear_s3_client_cache()
    assert second.closed is True
    s3_utils.configure_s3_client_pool(1)


def test_read_s3_bytes_closes_response_body(monkeypatch) -> None:
    class _Body:
        closed = False

        def read(self):
            return b'payload'

        def close(self):
            self.closed = True

    body = _Body()
    client = types.SimpleNamespace(get_object=lambda **_kwargs: {'Body': body})
    monkeypatch.setattr(s3_utils, '_make_s3_client', lambda: client)

    assert s3_utils.read_s3_bytes('s3://bucket/key') == b'payload'
    assert body.closed is True


def test_read_s3_bytes_closes_body_when_read_fails(monkeypatch) -> None:
    class _Body:
        closed = False

        def read(self):
            raise OSError('network interrupted')

        def close(self):
            self.closed = True

    body = _Body()
    client = types.SimpleNamespace(get_object=lambda **_kwargs: {'Body': body})
    monkeypatch.setattr(s3_utils, '_make_s3_client', lambda: client)

    with pytest.raises(OSError, match='network interrupted'):
        s3_utils.read_s3_bytes('s3://bucket/key')
    assert body.closed is True


def test_broken_datalad_symlink_has_actionable_error(tmp_path) -> None:
    source = tmp_path / 'image.nii.gz'
    source.symlink_to(tmp_path / '.git' / 'annex' / 'objects' / 'missing')

    with pytest.raises(FileNotFoundError, match='datalad get'):
        s3_utils._check_local_source(source)
