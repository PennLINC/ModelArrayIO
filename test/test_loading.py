"""Tests for bounded scalar staging."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from modelarrayio.utils.loading import LoadedRow, SourceLoadError, stage_source_rows


def test_stage_source_rows_is_bounded_ordered_and_typed() -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0

    def load(source: int) -> np.ndarray:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            # Force completion in reverse order.
            time.sleep((5 - source) * 0.002)
            return np.array([source + 0.25, source + 0.5], dtype=np.float64)
        finally:
            with lock:
                active -= 1

    matrix, stats = stage_source_rows(
        list(range(6)),
        load,
        scalar_name='FA',
        storage_dtype='float32',
        workers=2,
        n_elements=2,
    )

    assert maximum <= 2
    assert matrix.flags.c_contiguous
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[:, 0], np.arange(6, dtype=np.float32) + 0.25)
    assert 1 <= stats.in_flight_sources_max <= 2
    assert stats.in_flight_sources_mean >= 1
    assert stats.source_wall_seconds > 0
    assert stats.source_count == 6


def test_stage_source_rows_reuses_first_row_and_records_validation() -> None:
    calls: list[str] = []

    def load(source: str):
        calls.append(source)
        return LoadedRow(np.array([2.0, 3.0]), validation_seconds=0.125)

    matrix, stats = stage_source_rows(
        ['first', 'second'],
        load,
        scalar_name='MD',
        first_row=LoadedRow(
            np.array([0.0, 1.0]),
            validation_seconds=0.25,
            source_seconds=0.5,
            cpu_seconds=0.125,
        ),
        storage_dtype='float64',
        workers=1,
    )

    assert calls == ['second']
    np.testing.assert_array_equal(matrix, [[0.0, 1.0], [2.0, 3.0]])
    assert stats.validation_seconds == pytest.approx(0.375)
    assert stats.source_wall_seconds >= 0.5
    assert stats.source_seconds >= 0.5
    assert stats.wall_seconds >= 0.5
    assert stats.cpu_seconds >= 0.125


@pytest.mark.parametrize('workers', [0, -1, True, 2.0, '2', 'bad'])
def test_stage_source_rows_rejects_invalid_workers(workers) -> None:
    with pytest.raises(ValueError, match='positive integer'):
        stage_source_rows(
            ['source'],
            lambda _source: np.ones(1),
            scalar_name='FA',
            workers=workers,
            n_elements=1,
        )


def test_stage_source_rows_adds_scalar_and_source_to_errors() -> None:
    def fail(_source):
        raise OSError('unavailable')

    with pytest.raises(SourceLoadError, match=r"source 'broken\.nii\.gz'.*scalar 'ALFF'"):
        stage_source_rows(
            ['broken.nii.gz'],
            fail,
            scalar_name='ALFF',
            workers=1,
            n_elements=3,
        )


def test_stage_source_rows_rejects_wrong_row_shape() -> None:
    with pytest.raises(SourceLoadError, match=r'expected \(3,\)'):
        stage_source_rows(
            ['source'],
            lambda _source: np.ones(2),
            scalar_name='FA',
            workers=1,
            n_elements=3,
        )


def test_stage_source_rows_logs_progress_while_loading(caplog) -> None:
    import logging

    def load(source: int) -> np.ndarray:
        return np.array([float(source)], dtype=np.float32)

    with caplog.at_level(logging.INFO, logger='modelarrayio.utils.loading'):
        matrix, stats = stage_source_rows(
            list(range(6)),
            load,
            scalar_name='FA',
            workers=1,
            progress_interval_seconds=0.0000001,
        )
    np.testing.assert_array_equal(matrix[:, 0], np.arange(6, dtype=np.float32))
    progress = [record for record in caplog.records if 'Staging FA:' in record.getMessage()]
    assert progress, 'expected at least one progress line'
    # The last source is reported by the completion record, not a progress line.
    assert all('6/6' not in record.getMessage() for record in progress)
    assert stats.source_count == 6


def test_stage_source_rows_progress_can_be_disabled(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger='modelarrayio.utils.loading'):
        stage_source_rows(
            [0, 1, 2],
            lambda source: np.array([1.0], dtype=np.float32),
            scalar_name='FA',
            workers=2,
            progress_interval_seconds=None,
        )
    assert not [record for record in caplog.records if 'Staging FA:' in record.getMessage()]
