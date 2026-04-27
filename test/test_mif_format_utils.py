"""Unit tests for low-level MIF format parsing helpers."""

from __future__ import annotations

import io
import sys

import numpy as np
import pytest

from modelarrayio.utils import _mif_format


def test_readline_reads_until_newline() -> None:
    fileobj = io.BytesIO(b'first line\nsecond line')
    assert _mif_format._readline(fileobj) == b'first line\n'
    assert _mif_format._readline(fileobj) == b'second line'


@pytest.mark.parametrize(
    ('dtype_str', 'expected'),
    [
        ('Float32LE', np.dtype('<f4')),
        ('Float64BE', np.dtype('>f8')),
        ('UInt8', np.dtype('u1')),
    ],
)
def test_parse_dtype_valid(dtype_str: str, expected: np.dtype) -> None:
    parsed = _mif_format._mif_parse_dtype(dtype_str)
    assert parsed == expected


def test_parse_dtype_uses_native_endian_without_suffix(monkeypatch) -> None:
    monkeypatch.setattr(sys, 'byteorder', 'little')
    assert _mif_format._mif_parse_dtype('Int16') == np.dtype('<i2')


def test_parse_dtype_unknown_raises() -> None:
    with pytest.raises(ValueError, match='Unknown MIF datatype'):
        _mif_format._mif_parse_dtype('Bogus')


def test_dtype_to_str_round_trip_and_errors() -> None:
    assert _mif_format._mif_dtype_to_str(np.dtype('<f4')).startswith('Float32')
    assert _mif_format._mif_dtype_to_str(np.dtype('u1')) == 'UInt8'
    with pytest.raises(ValueError, match='Cannot represent numpy dtype'):
        _mif_format._mif_dtype_to_str(np.dtype(bool))


def test_layout_parse_and_format_round_trip() -> None:
    layout = _mif_format._mif_parse_layout('-0,+1,2', ndim=3)
    assert layout == [-1, 2, 3]
    assert _mif_format._mif_layout_to_str(layout) == '-0,+1,+2'


def test_layout_parse_rejects_wrong_axis_count() -> None:
    with pytest.raises(ValueError, match='Layout has 2 axes but dim has 3'):
        _mif_format._mif_parse_layout('0,1', ndim=3)
