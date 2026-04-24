"""Verify that MIF loading produces mrtrix-canonical (positive-stride) data."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from modelarrayio.utils._mif_format import _mif_apply_layout, _mif_apply_layout_for_write
from modelarrayio.utils.mif import mif_to_image

# ---------------------------------------------------------------------------
# Pure-Python synthetic fixtures: independent of downloaded data and mrconvert
# ---------------------------------------------------------------------------


def _write_mif(
    path: Path,
    data: np.ndarray,
    layout: list[int],
    scaling: tuple[float, float] | None = None,
) -> None:
    """Write a minimal MIF file with the given canonical-order *data* and *layout*."""
    shape = data.shape
    ndim = len(shape)
    assert len(layout) == ndim

    # Convert canonical-order data into the on-disk byte layout.
    disk = _mif_apply_layout_for_write(data, layout)

    layout_str = ','.join(('+' if s > 0 else '-') + str(abs(s) - 1) for s in layout)

    header_lines = [
        'mrtrix image',
        f'dim: {",".join(str(s) for s in shape)}',
        f'vox: {",".join("1.0" for _ in shape)}',
        f'layout: {layout_str}',
        'datatype: Float32LE',
        'transform: 1.0, 0.0, 0.0, 0.0',
        'transform: 0.0, 1.0, 0.0, 0.0',
        'transform: 0.0, 0.0, 1.0, 0.0',
    ]
    if scaling is not None:
        header_lines.append(f'scaling: {scaling[0]},{scaling[1]}')

    pre = ('\n'.join(header_lines) + '\n').encode('latin-1')
    # Compute offset the same way MifHeader.write_to does.
    file_prefix = b'file: . '
    end_suffix = b'\nEND\n'
    offset = len(pre) + len(file_prefix) + 5 + len(end_suffix)
    offset += (4 - offset % 4) % 4
    for _ in range(5):
        file_line = file_prefix + str(offset).encode() + end_suffix
        total = len(pre) + len(file_line)
        new_offset = total + (4 - total % 4) % 4
        if new_offset == offset:
            break
        offset = new_offset
    file_line = f'file: . {offset}\nEND\n'.encode('latin-1')

    payload = disk.astype('<f4').tobytes()
    with path.open('wb') as fh:
        fh.write(pre)
        fh.write(file_line)
        pad = offset - (len(pre) + len(file_line))
        fh.write(b'\x00' * pad)
        fh.write(payload)


def test_negative_stride_1d_is_flipped_on_read(tmp_path: Path) -> None:
    """Layout ``-0`` means disk bytes are stored reversed; loader must un-reverse."""
    canonical = np.arange(7, dtype=np.float32)
    path = tmp_path / 'neg_1d.mif'
    _write_mif(path, canonical, layout=[-1])

    # Inspect the raw bytes: they should be reversed relative to canonical.
    raw = path.read_bytes()[-canonical.nbytes :]
    on_disk = np.frombuffer(raw, dtype='<f4')
    np.testing.assert_array_equal(on_disk, canonical[::-1])

    _, loaded = mif_to_image(str(path))
    np.testing.assert_array_equal(loaded, canonical)


def test_mixed_negative_strides_3d_is_canonicalised(tmp_path: Path) -> None:
    """Layout ``-0,+1,-2`` should produce canonical-order numpy output on read."""
    canonical = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / 'neg_3d.mif'
    _write_mif(path, canonical, layout=[-1, 2, -3])

    _, loaded = mif_to_image(str(path))
    np.testing.assert_array_equal(loaded, canonical)


def test_layout_axis_reorder_only(tmp_path: Path) -> None:
    """Layout ``+1,+2,+0`` reorders axes on disk; loader returns canonical order."""
    canonical = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / 'reorder.mif'
    _write_mif(path, canonical, layout=[2, 3, 1])

    _, loaded = mif_to_image(str(path))
    np.testing.assert_array_equal(loaded, canonical)


def test_apply_layout_round_trips_through_write_and_read() -> None:
    """_mif_apply_layout is the inverse of _mif_apply_layout_for_write."""
    rng = np.random.default_rng(0)
    canonical = rng.random((3, 2, 4), dtype=np.float32)
    for layout in ([1, 2, 3], [-1, 2, 3], [-1, -2, 3], [3, -1, 2], [-3, 2, -1]):
        disk = _mif_apply_layout_for_write(canonical, layout)
        restored = _mif_apply_layout(disk.ravel(), canonical.shape, layout)
        np.testing.assert_array_equal(restored, canonical)


def test_intensity_scaling_applied_once(tmp_path: Path) -> None:
    """Verify the read path applies ``offset + scale * disk_value`` once."""
    disk_values = np.arange(5, dtype=np.float32)
    offset, scale = 2.5, 0.25
    path = tmp_path / 'scaled.mif'
    _write_mif(path, disk_values, layout=[1], scaling=(offset, scale))

    _, loaded = mif_to_image(str(path))
    np.testing.assert_allclose(loaded, offset + scale * disk_values, rtol=1e-6)


# ---------------------------------------------------------------------------
# mrconvert-based equivalence: loader matches canonical-stride output
# ---------------------------------------------------------------------------


_MRCONVERT = shutil.which('mrconvert')


def _canonical_strides(ndim: int) -> str:
    return ','.join(str(i + 1) for i in range(ndim))


def _downloaded_scalar_file(data_dir: Path) -> Path | None:
    """Return one fixel-scalar .mif from the cohort CSV, or ``None`` if unavailable."""
    import pandas as pd

    cohort_csv = data_dir / 'stat-alpha_cohort.csv'
    if not cohort_csv.exists():
        return None
    cohort = pd.read_csv(cohort_csv)
    candidate = data_dir / str(cohort['source_file'].iloc[0])
    return candidate if candidate.exists() else None


@pytest.mark.skipif(_MRCONVERT is None, reason='mrconvert not available on PATH')
@pytest.mark.downloaded_data
@pytest.mark.parametrize('filename', ['directions.mif', 'index.mif', '__scalar__'])
def test_loader_matches_mrconvert_canonical_strides(
    tmp_path: Path, downloaded_fixel_data_dir: Path, filename: str
) -> None:
    """Non-canonical stride files should load identically to their canonical-stride twin."""
    if filename == '__scalar__':
        src = _downloaded_scalar_file(downloaded_fixel_data_dir)
        if src is None:
            pytest.skip('no fixel scalar file listed in downloaded cohort CSV')
    else:
        src = downloaded_fixel_data_dir / filename
        if not src.exists():
            pytest.skip(f'{filename} missing from downloaded fixture directory')

    _, data_src = mif_to_image(str(src))
    ndim = data_src.ndim

    canonical = tmp_path / f'canonical_{src.name}'
    subprocess.run(
        [
            _MRCONVERT,
            str(src),
            str(canonical),
            '-strides',
            _canonical_strides(ndim),
            '-force',
            '-quiet',
        ],
        check=True,
    )
    _, data_can = mif_to_image(str(canonical))
    np.testing.assert_array_equal(data_src, data_can)


@pytest.mark.downloaded_data
def test_fixel_id_points_to_matching_scalar(downloaded_fixel_data_dir: Path) -> None:
    """The first_fixel_id stored in index.mif must index correctly into a fixel scalar.

    This fails when the loader returns disk-ordered data because ``scalar[id]``
    would fetch the reversed fixel.
    """
    index_file = downloaded_fixel_data_dir / 'index.mif'
    scalar_file = _downloaded_scalar_file(downloaded_fixel_data_dir)
    if not index_file.exists() or scalar_file is None:
        pytest.skip('fixel fixtures missing from downloaded data')

    _, index_data = mif_to_image(str(index_file))
    _, scalar = mif_to_image(str(scalar_file))

    count_vol = index_data[..., 0].astype(np.int64)
    id_vol = index_data[..., 1].astype(np.int64)
    n_fixels = scalar.shape[0]

    # Total count must equal the number of fixels in the scalar file.
    assert int(count_vol.sum()) == n_fixels

    # Every (count, id) pair must fit inside the scalar array.
    nonzero = count_vol > 0
    ids = id_vol[nonzero]
    counts = count_vol[nonzero]
    assert int((ids + counts).max()) == n_fixels
    assert int(ids.min()) >= 0
