"""Unit tests for MIF helper modules and CLI writer logic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from modelarrayio.cli import h5_to_mif as h5_to_mif_module
from modelarrayio.utils import mif


def test_mif_to_image_uses_mifimage_loader(monkeypatch, tmp_path: Path) -> None:
    class _FakeImage:
        def get_fdata(self, dtype=None):
            return np.array([[1.0, 2.0]], dtype=np.float32 if dtype is None else dtype)

    def _fake_from_filename(path):
        assert path == str(tmp_path / 'input.mif')
        return _FakeImage()

    monkeypatch.setattr(mif.MifImage, 'from_filename', staticmethod(_fake_from_filename))
    img, data = mif.mif_to_image(tmp_path / 'input.mif')
    assert isinstance(img, _FakeImage)
    np.testing.assert_array_equal(data, np.array([1.0, 2.0], dtype=np.float32))


def test_load_cohort_mif_sequential_and_parallel(monkeypatch) -> None:
    cohort = pd.DataFrame(
        {
            'scalar_name': ['FA', 'FA', 'MD'],
            'source_file': ['a', 'b', 'c'],
        }
    )

    def _fake_mif_to_image(path):
        mapping = {
            'a': np.array([1.0, 2.0], dtype=np.float32),
            'b': np.array([3.0, 4.0], dtype=np.float32),
            'c': np.array([5.0, 6.0], dtype=np.float32),
        }
        return object(), mapping[str(path)]

    monkeypatch.setattr(mif, 'mif_to_image', _fake_mif_to_image)
    serial_scalars, serial_sources = mif.load_cohort_mif(cohort, s3_workers=1)
    parallel_scalars, parallel_sources = mif.load_cohort_mif(cohort, s3_workers=2)

    assert list(serial_sources['FA']) == ['a', 'b']
    assert list(parallel_sources['MD']) == ['c']
    np.testing.assert_array_equal(np.array(serial_scalars['FA']), np.array(parallel_scalars['FA']))
    np.testing.assert_array_equal(np.array(serial_scalars['MD']), np.array(parallel_scalars['MD']))


def test_gather_fixels_builds_tables(monkeypatch) -> None:
    index_data = np.zeros((1, 2, 1, 2), dtype=np.float32)
    index_data[0, 0, 0, 0] = 1
    index_data[0, 0, 0, 1] = 0
    index_data[0, 1, 0, 0] = 2
    index_data[0, 1, 0, 1] = 1
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def _fake_mif_to_image(path):
        if str(path).endswith('index.mif'):
            return object(), index_data
        return object(), directions

    monkeypatch.setattr(mif, 'mif_to_image', _fake_mif_to_image)
    fixel_table, voxel_table = mif.gather_fixels('index.mif', 'directions.mif')

    assert list(voxel_table.columns) == ['voxel_id', 'i', 'j', 'k']
    assert list(fixel_table.columns) == ['fixel_id', 'voxel_id', 'x', 'y', 'z']
    assert len(fixel_table) == 3
    assert len(voxel_table) == 2


def test_gather_fixels_raises_when_terminal_count_missing(monkeypatch) -> None:
    index_data = np.zeros((1, 1, 1, 2), dtype=np.float32)
    index_data[..., 1] = np.nan
    directions = np.zeros((1, 3), dtype=np.float32)

    def _fake_mif_to_image(path):
        return object(), index_data if str(path).endswith('index.mif') else directions

    monkeypatch.setattr(mif, 'mif_to_image', _fake_mif_to_image)
    with pytest.raises(ValueError, match='Could not determine the final fixel count'):
        mif.gather_fixels('index.mif', 'directions.mif')


def test_write_mif_does_not_overwrite_existing_file(tmp_path: Path, caplog) -> None:
    existing = tmp_path / 'already.mif'
    existing.write_bytes(b'test')
    template = SimpleNamespace(shape=(2,), header=mif.MifHeader(shape=(2,)), affine=np.eye(4))

    with caplog.at_level('WARNING'):
        h5_to_mif_module.write_mif(np.array([1.0, 2.0], dtype=np.float32), template, existing)

    assert any('Output file already exists' in record.message for record in caplog.records)


def test_h5_to_mif_writes_pvalue_and_inverse(monkeypatch, tmp_path: Path) -> None:
    h5_path = tmp_path / 'results.h5'
    with h5py.File(h5_path, 'w') as h5_file:
        group = h5_file.require_group('results/lm')
        group.create_dataset(
            'results_matrix',
            data=np.array([[0.25, 0.75], [2.0, 3.0]], dtype=np.float32),
        )
        group.create_dataset(
            'column_names',
            data=np.array(['p.value', 'effect size'], dtype=h5py.string_dtype('utf-8')),
        )

    template_img = SimpleNamespace(shape=(2,), header=mif.MifHeader(shape=(2,)), affine=np.eye(4))
    monkeypatch.setattr(
        h5_to_mif_module, 'mif_to_image', lambda _: (template_img, np.array([0, 0]))
    )

    calls: list[tuple[np.ndarray, Path]] = []

    def _fake_write_mif(arr, template_img, out_file):
        calls.append((np.array(arr), Path(out_file)))

    monkeypatch.setattr(h5_to_mif_module, 'write_mif', _fake_write_mif)
    status = h5_to_mif_module.h5_to_mif(
        example_mif='template.mif',
        in_file=h5_path,
        analysis_name='lm',
        compress=False,
        output_dir=tmp_path / 'out',
    )

    assert status == 0
    written_names = sorted(path.name for _, path in calls)
    assert written_names == ['lm_1m.p.value.mif', 'lm_effect_size.mif', 'lm_p.value.mif']
