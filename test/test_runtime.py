"""Tests for conversion runtime policy and transactional publication."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import tiledb

from modelarrayio.cli import runtime
from modelarrayio.cli.runtime import (
    TimingRecorder,
    atomic_output,
    load_timing_metrics,
    path_stats,
    resolve_loader_workers,
    scalar_output_stats,
    validate_hdf5_output,
    validate_tiledb_output,
    validate_timing_jsonl_path,
)
from modelarrayio.storage import tiledb_storage
from modelarrayio.storage.tiledb_storage import create_scalar_matrix_array
from modelarrayio.utils.loading import LoadStats


def test_resolve_loader_workers_defaults_and_deprecated_alias() -> None:
    assert resolve_loader_workers(None, None) == 1
    assert resolve_loader_workers(3, None) == 3

    with pytest.warns(DeprecationWarning, match='--s3-workers is deprecated'):
        assert resolve_loader_workers(None, 4) == 4
    with pytest.warns(DeprecationWarning, match='--s3-workers is deprecated'):
        assert resolve_loader_workers(4, 4) == 4


def test_resolve_loader_workers_rejects_conflicts_and_nonpositive_values() -> None:
    with pytest.warns(DeprecationWarning, match='--s3-workers is deprecated'):
        with pytest.raises(ValueError, match='Conflicting loader worker counts'):
            resolve_loader_workers(2, 3)
    with pytest.raises(ValueError, match='positive integer'):
        resolve_loader_workers(0, None)


def test_load_timing_metrics_distinguishes_scheduler_and_pipeline_waiting() -> None:
    metrics = load_timing_metrics(
        LoadStats(
            allocation_seconds=0.25,
            wall_seconds=3.0,
            source_wall_seconds=2.0,
            cpu_seconds=1.5,
            source_seconds=4.0,
            validation_seconds=0.5,
            scheduler_wait_seconds=1.75,
            in_flight_sources_mean=1.5,
            in_flight_sources_max=2,
        )
    )

    assert metrics['source_load_seconds'] == 2.0
    assert metrics['staging_seconds'] == 3.0
    assert metrics['writer_idle_seconds'] == 3.0
    assert metrics['scheduler_wait_seconds'] == 1.75
    assert metrics['loader_blocked_seconds'] == 0.0
    assert metrics['queue_occupancy_mean'] == 0.0
    assert metrics['in_flight_sources_max'] == 2


def test_scalar_output_stats_avoids_cumulative_combined_file_size(tmp_path: Path) -> None:
    output = tmp_path / 'combined.h5'
    output.write_bytes(b'cumulative')

    assert scalar_output_stats({}, 'FA', output, split_output=False) == (None, None)
    assert scalar_output_stats({}, 'FA', output, split_output=True) == (
        len(b'cumulative'),
        1,
    )
    assert scalar_output_stats(
        {'FA': {'output_bytes': 123, 'output_file_count': 4}},
        'FA',
        output,
        split_output=False,
    ) == (123, 4)


def test_atomic_file_publication_replaces_only_after_success(tmp_path: Path) -> None:
    output = tmp_path / 'result.h5'
    output.write_bytes(b'old')

    with atomic_output(output, directory=False) as temporary:
        assert output.read_bytes() == b'old'
        temporary.write_bytes(b'new')

    assert output.read_bytes() == b'new'
    assert not list(tmp_path.glob('.*.modelarrayio-tmp-*'))


def test_atomic_directory_failure_preserves_published_output(tmp_path: Path) -> None:
    output = tmp_path / 'result.tdb'
    output.mkdir()
    (output / 'old.txt').write_text('old', encoding='utf-8')

    def _fail_during_conversion() -> None:
        with atomic_output(output, directory=True) as temporary:
            temporary.mkdir()
            (temporary / 'new.txt').write_text('new', encoding='utf-8')
            raise RuntimeError('injected failure')

    with pytest.raises(RuntimeError, match='injected failure'):
        _fail_during_conversion()

    assert (output / 'old.txt').read_text(encoding='utf-8') == 'old'
    assert not list(tmp_path.glob('.*.modelarrayio-tmp-*'))
    assert not list(tmp_path.glob('.*.modelarrayio-backup-*'))


@pytest.mark.parametrize('directory', [False, True])
def test_atomic_output_cleans_temporary_path_on_keyboard_interrupt(
    tmp_path: Path, directory: bool
) -> None:
    output = tmp_path / ('result.tdb' if directory else 'result.h5')
    if directory:
        output.mkdir()
        (output / 'old.txt').write_text('old', encoding='utf-8')
    else:
        output.write_bytes(b'old')

    def _interrupt_conversion() -> None:
        with atomic_output(output, directory=directory) as temporary:
            if directory:
                temporary.mkdir()
                (temporary / 'new.txt').write_text('new', encoding='utf-8')
            else:
                temporary.write_bytes(b'new')
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _interrupt_conversion()

    if directory:
        assert (output / 'old.txt').read_text(encoding='utf-8') == 'old'
    else:
        assert output.read_bytes() == b'old'
    assert not list(tmp_path.glob('.*.modelarrayio-tmp-*'))


def test_atomic_directory_publish_rolls_back_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / 'result.tdb'
    output.mkdir()
    (output / 'old.txt').write_text('old', encoding='utf-8')
    real_replace = Path.replace

    def _interrupt_temporary_publish(self, target):
        if '.modelarrayio-tmp-' in self.name and Path(target) == output:
            raise KeyboardInterrupt
        return real_replace(self, target)

    monkeypatch.setattr(Path, 'replace', _interrupt_temporary_publish)

    def _interrupt_publication() -> None:
        with atomic_output(output, directory=True) as temporary:
            temporary.mkdir()
            (temporary / 'new.txt').write_text('new', encoding='utf-8')

    with pytest.raises(KeyboardInterrupt):
        _interrupt_publication()

    assert (output / 'old.txt').read_text(encoding='utf-8') == 'old'
    assert not list(tmp_path.glob('.*.modelarrayio-tmp-*'))
    assert not list(tmp_path.glob('.*.modelarrayio-backup-*'))


def test_validate_hdf5_output_checks_source_name_count(tmp_path: Path) -> None:
    output = tmp_path / 'result.h5'
    with h5py.File(output, 'w') as h5_file:
        h5_file.create_dataset('scalars/FA/values', data=np.zeros((2, 3)))
        h5_file.create_dataset('scalars/FA/column_names', data=np.array([b'one']))

    with pytest.raises(RuntimeError, match='2 rows but 1 source names'):
        validate_hdf5_output(output, ['FA'])


def test_validate_hdf5_output_checks_full_contract(tmp_path: Path) -> None:
    output = tmp_path / 'result.h5'
    with h5py.File(output, 'w') as h5_file:
        h5_file.create_dataset('voxels', data=np.zeros((4, 3)))
        h5_file.create_dataset('scalars/FA/values', data=np.zeros((2, 3), dtype=np.float32))
        h5_file.create_dataset(
            'scalars/FA/column_names',
            data=np.array(['two', 'one'], dtype=h5py.string_dtype('utf-8')),
        )

    with pytest.raises(RuntimeError, match='source names or order'):
        validate_hdf5_output(
            output,
            ['FA'],
            expected_sources={'FA': ['one', 'two']},
            expected_dtype='float32',
            expected_elements=3,
            required_paths=('voxels',),
        )
    with pytest.raises(RuntimeError, match='elements != expected'):
        validate_hdf5_output(output, ['FA'], expected_elements=4)
    with pytest.raises(RuntimeError, match='missing required HDF5 path'):
        validate_hdf5_output(output, ['FA'], required_paths=('missing',))


def test_validate_tiledb_output_checks_metadata_and_cifti_name_array(tmp_path: Path) -> None:
    output = tmp_path / 'result.tdb'
    uri = create_scalar_matrix_array(
        str(output),
        'scalars/FA/values',
        np.zeros((2, 3), dtype=np.float32),
        ['one', 'two'],
        compression='none',
    )
    validate_tiledb_output(output, ['FA'])

    with pytest.raises(RuntimeError, match='missing TileDB array'):
        validate_tiledb_output(output, ['FA'], require_column_name_arrays=True)

    with tiledb.open(uri, 'w') as array:
        del array.meta['column_names']
    with pytest.raises(RuntimeError, match='0 source names'):
        validate_tiledb_output(output, ['FA'])


def test_validate_tiledb_output_checks_full_contract(tmp_path: Path) -> None:
    output = tmp_path / 'result.tdb'
    tiledb_storage.create_scalar_matrix_array(
        str(output),
        'scalars/FA/values',
        np.zeros((2, 3), dtype=np.float32),
        ['one', 'two'],
        compression='none',
    )
    tiledb_storage.write_column_names(str(output), 'FA', ['one', 'two'])
    validate_tiledb_output(
        output,
        ['FA'],
        require_column_name_arrays=True,
        expected_sources={'FA': ['one', 'two']},
        expected_dtype='float32',
        expected_elements=3,
    )

    tiledb_storage.write_column_names(str(output), 'FA', ['two', 'one'])
    with pytest.raises(RuntimeError, match='column-name array does not match'):
        validate_tiledb_output(output, ['FA'], require_column_name_arrays=True)
    with pytest.raises(RuntimeError, match='elements != expected'):
        validate_tiledb_output(output, ['FA'], expected_elements=4)


def test_validate_timing_jsonl_path_rejects_output_collisions(tmp_path: Path) -> None:
    hdf5_output = tmp_path / 'result.h5'
    tiledb_output = tmp_path / 'result.tdb'

    with pytest.raises(ValueError, match='conflicts with conversion output'):
        validate_timing_jsonl_path(
            hdf5_output,
            [hdf5_output],
            directory_outputs=False,
        )
    with pytest.raises(ValueError, match='conflicts with conversion output'):
        validate_timing_jsonl_path(
            tiledb_output / 'timing.jsonl',
            [tiledb_output],
            directory_outputs=True,
        )
    validate_timing_jsonl_path(
        tmp_path / 'timing.jsonl',
        [hdf5_output, tiledb_output],
        directory_outputs=False,
    )


def test_timing_recorder_appends_json_lines_and_path_stats(tmp_path: Path) -> None:
    metrics_path = tmp_path / 'metrics' / 'runtime.jsonl'
    recorder = TimingRecorder(metrics_path, modality='nifti', workers=2)
    recorder.emit('scalar_complete', scalar_name='FA', matrix_shape=[2, 3])
    recorder.finish(scalar_count=1)

    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [record['event'] for record in records] == [
        'scalar_complete',
        'conversion_complete',
    ]
    assert all(record['modality'] == 'nifti' for record in records)
    assert records[-1]['cpu_utilization_percent'] >= 0
    assert records[-1]['s3_pool_connections'] >= 10
    assert 'peak_open_file_descriptors_sampled' in records[-1]
    assert path_stats(metrics_path) == (metrics_path.stat().st_size, 1)


def test_timing_recorder_reports_tiledb_concurrency(tmp_path: Path) -> None:
    recorder = TimingRecorder(tmp_path / 'tiledb.jsonl', backend='tiledb')
    record = recorder.finish()

    assert record['tiledb_compute_concurrency_level'] >= 1
    assert record['tiledb_io_concurrency_level'] >= 1


def test_timing_recorder_jsonl_failure_does_not_fail_conversion_metrics(
    tmp_path: Path, caplog
) -> None:
    jsonl_directory = tmp_path / 'metrics.jsonl'
    jsonl_directory.mkdir()
    recorder = TimingRecorder(jsonl_directory)

    with caplog.at_level('WARNING', logger=runtime.__name__):
        record = recorder.emit('scalar_complete')

    assert record['event'] == 'scalar_complete'
    assert recorder.jsonl_path is None
    assert any('continuing without JSONL output' in item.message for item in caplog.records)


def test_publish_scalar_outputs_reports_loader_stats_filled_on_access(tmp_path) -> None:
    import numpy as np

    from modelarrayio.cli.runtime import TimingRecorder, publish_scalar_outputs
    from modelarrayio.utils.loading import LazyScalarMatrices, LoadStats

    def _stage(scalar_name):
        stats = LoadStats(source_count=2, workers=1, fast_path_rows=1)
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), stats

    scalars = LazyScalarMatrices(['FA'], _stage)
    assert scalars.stats == {}  # nothing staged yet: the loop must not snapshot this
    jsonl = tmp_path / 'timing.jsonl'
    recorder = TimingRecorder(jsonl, modality='test', backend='hdf5')

    published = publish_scalar_outputs(
        recorder,
        backend='hdf5',
        output=tmp_path / 'out.h5',
        split_outputs=False,
        scalar_names=['FA'],
        scalars=scalars,
        sources_by_scalar={'FA': ['a.nii', 'b.nii']},
        storage_dtype='float32',
        compression='gzip',
        compression_level=4,
        shuffle=True,
        chunk_voxels=0,
        target_chunk_mb=2.0,
        expected_elements=2,
    )

    assert published == [tmp_path / 'out.h5']
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    scalar_records = [record for record in records if record['event'] == 'scalar_complete']
    assert len(scalar_records) == 1
    assert scalar_records[0]['fast_path_rows'] == 1
    assert scalar_records[0]['matrix_shape'] == [2, 2]
