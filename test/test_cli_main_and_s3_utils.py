"""Unit tests for CLI entrypoint and S3 utilities."""

from __future__ import annotations

import gzip
import sys
import types

import nibabel as nb
import numpy as np
import pytest

from modelarrayio.cli import main as cli_main
from modelarrayio.utils import s3_utils


def test_main_prints_help_and_returns_1(capsys) -> None:
    status = cli_main.main([])
    captured = capsys.readouterr()
    assert status == 1
    assert 'usage:' in captured.out


def test_main_dispatches_to_selected_subcommand(monkeypatch) -> None:
    def _fake_run(**kwargs):
        assert kwargs['value'] == 'ok'
        return 7

    parser = cli_main._get_parser()
    parser.add_argument('--value', required=True)
    parser.set_defaults(func=_fake_run)
    monkeypatch.setattr(cli_main, '_get_parser', lambda: parser)

    assert cli_main.main(['--value', 'ok']) == 7


def test_get_version_fallbacks(monkeypatch) -> None:
    fake_about = types.ModuleType('modelarrayio.__about__')
    monkeypatch.setitem(sys.modules, 'modelarrayio.__about__', fake_about)
    monkeypatch.setattr(cli_main, 'version', lambda _: '1.2.3')
    assert cli_main._get_version() == '1.2.3'

    monkeypatch.setattr(
        cli_main,
        'version',
        lambda _: (_ for _ in ()).throw(cli_main.PackageNotFoundError('missing')),
    )
    assert cli_main._get_version() == '0+unknown'


def test_make_s3_client_anon_and_signed(monkeypatch) -> None:
    calls = []

    class _FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            calls.append((service, kwargs))
            return ('client', kwargs)

    fake_config_module = types.SimpleNamespace(Config=lambda **kwargs: ('cfg', kwargs))
    fake_botocore = types.SimpleNamespace(UNSIGNED='unsigned')
    monkeypatch.setitem(__import__('sys').modules, 'boto3', _FakeBoto3)
    monkeypatch.setitem(__import__('sys').modules, 'botocore', fake_botocore)
    monkeypatch.setitem(__import__('sys').modules, 'botocore.config', fake_config_module)

    monkeypatch.setenv('MODELARRAYIO_S3_ANON', '1')
    s3_utils._make_s3_client()
    assert calls[0][0] == 's3'
    assert 'config' in calls[0][1]

    monkeypatch.setenv('MODELARRAYIO_S3_ANON', '0')
    s3_utils._make_s3_client()
    assert calls[1] == ('s3', {})


def test_make_s3_client_requires_boto3(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == 'boto3':
            raise ImportError('no boto3')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _fake_import)
    with pytest.raises(ImportError, match='boto3 is required'):
        s3_utils._make_s3_client()


def test_load_nibabel_local_path(monkeypatch, tmp_path) -> None:
    nifti_path = tmp_path / 'image.nii.gz'
    data = np.zeros((2, 2, 2), dtype=np.float32)
    nb.Nifti1Image(data, np.eye(4)).to_filename(nifti_path)
    loaded = s3_utils.load_nibabel(str(nifti_path))
    np.testing.assert_array_equal(loaded.get_fdata(), data)


def test_load_nibabel_from_s3_bytes(monkeypatch, tmp_path) -> None:
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    image = nb.Nifti1Image(data, np.eye(4))
    file_path = tmp_path / 'local.nii'
    image.to_filename(file_path)
    raw = gzip.compress(file_path.read_bytes())

    class _FakeBody:
        def read(self):
            return raw

    class _FakeClient:
        def get_object(self, **kwargs):
            assert kwargs['Bucket'] == 'bucket'
            assert kwargs['Key'] == 'key.nii.gz'
            return {'Body': _FakeBody()}

    monkeypatch.setattr(s3_utils, '_make_s3_client', lambda: _FakeClient())
    loaded = s3_utils.load_nibabel('s3://bucket/key.nii.gz')
    np.testing.assert_array_equal(loaded.get_fdata(), data)
