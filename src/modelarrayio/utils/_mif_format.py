"""Low-level MIF format parsing helpers."""

import sys

import numpy as np


def _readline(fileobj) -> bytes:
    """Read one newline-terminated line from *fileobj* using only ``read(1)``.

    This works with any object that implements ``read(n)``, including
    nibabel's ``ImageOpener`` and gzip file objects that lack ``readline``.
    """
    buf = bytearray()
    while True:
        ch = fileobj.read(1)
        if not ch:
            break
        buf.extend(ch if isinstance(ch, (bytes, bytearray)) else ch.encode('latin-1'))
        if buf[-1:] == b'\n':
            break
    return bytes(buf)


_MIF_DTYPE_MAP: dict[str, str] = {
    'Int8': 'i1',
    'UInt8': 'u1',
    'Int16': 'i2',
    'UInt16': 'u2',
    'Int32': 'i4',
    'UInt32': 'u4',
    'Int64': 'i8',
    'UInt64': 'u8',
    'Float32': 'f4',
    'Float64': 'f8',
    'CFloat32': 'c8',
    'CFloat64': 'c16',
}

_NUMPY_TO_MIF_BASE: dict[tuple[str, int], str] = {
    ('i', 1): 'Int8',
    ('u', 1): 'UInt8',
    ('i', 2): 'Int16',
    ('u', 2): 'UInt16',
    ('i', 4): 'Int32',
    ('u', 4): 'UInt32',
    ('i', 8): 'Int64',
    ('u', 8): 'UInt64',
    ('f', 4): 'Float32',
    ('f', 8): 'Float64',
    ('c', 8): 'CFloat32',
    ('c', 16): 'CFloat64',
}


def _mif_parse_dtype(dtype_str: str) -> np.dtype:
    """Convert a MIF datatype string (e.g. ``'Float32LE'``) to a numpy dtype."""
    dtype_str = dtype_str.strip()
    if dtype_str.endswith('LE'):
        endian, base = '<', dtype_str[:-2]
    elif dtype_str.endswith('BE'):
        endian, base = '>', dtype_str[:-2]
    else:
        endian = '<' if sys.byteorder == 'little' else '>'
        base = dtype_str

    if base not in _MIF_DTYPE_MAP:
        raise ValueError(f'Unknown MIF datatype: {dtype_str!r}')

    type_char = _MIF_DTYPE_MAP[base]
    if type_char in ('i1', 'u1'):  # single-byte types have no endianness
        return np.dtype(type_char)
    return np.dtype(endian + type_char)


def _mif_dtype_to_str(dtype: np.dtype) -> str:
    """Convert a numpy dtype to a MIF datatype string."""
    dtype = np.dtype(dtype)
    base_name = _NUMPY_TO_MIF_BASE.get((dtype.kind, dtype.itemsize))
    if base_name is None:
        raise ValueError(f'Cannot represent numpy dtype {dtype!r} in MIF format')
    if dtype.itemsize == 1:
        return base_name

    byte_order = dtype.byteorder
    if byte_order == '=':
        byte_order = '<' if sys.byteorder == 'little' else '>'
    elif byte_order == '|':
        return base_name
    return base_name + ('LE' if byte_order == '<' else 'BE')


def _mif_parse_layout(layout_str: str, ndim: int) -> list[int]:
    """Parse a MIF layout string to a list of symbolic strides (1-indexed, signed).

    For example ``'-0,-1,+2'`` becomes ``[-1, -2, 3]``.  The absolute value
    encodes ordering (1 = fastest-varying axis) and the sign encodes direction.
    """
    strides = []
    for token in layout_str.strip().split(','):
        token = token.strip()
        if token.startswith('+'):
            sign, val = 1, int(token[1:])
        elif token.startswith('-'):
            sign, val = -1, int(token[1:])
        else:
            sign, val = 1, int(token)
        strides.append(sign * (val + 1))  # convert 0-indexed to 1-indexed
    if len(strides) != ndim:
        raise ValueError(f'Layout has {len(strides)} axes but dim has {ndim}: {layout_str!r}')
    return strides


def _mif_layout_to_str(layout: list[int]) -> str:
    """Convert symbolic strides list to a MIF layout string."""
    tokens = []
    for s in layout:
        sign = '+' if s >= 0 else '-'
        val = abs(s) - 1  # convert 1-indexed back to 0-indexed
        tokens.append(f'{sign}{val}')
    return ','.join(tokens)


def _mif_apply_layout(raw_flat: np.ndarray, shape: tuple, layout: list[int]) -> np.ndarray:
    """Reorder flat MIF disk data into a numpy array matching mrconvert's convention.

    MIF stores data with the axis whose ``|layout[i]|`` equals 1 varying
    fastest on disk.  This function reorders axes only — it does **not** flip
    axes for negative strides.  Instead, negative strides are encoded in the
    affine returned by :meth:`MifHeader.get_best_affine`, exactly as mrconvert
    does when writing NIfTI output.  This ensures that ``MifImage.get_fdata()``
    matches the data you would get from ``mrconvert file.mif file.nii`` followed
    by ``nibabel.load(file.nii).get_fdata()``.
    """
    ndim = len(shape)
    # Sort axes from fastest (|layout|=1) to slowest
    order = sorted(range(ndim), key=lambda i: abs(layout[i]))
    # Disk layout in C-order: [slowest, ..., fastest]
    disk_axes = list(reversed(order))
    disk_shape = tuple(shape[i] for i in disk_axes)

    data = raw_flat.reshape(disk_shape)

    # Transpose: output axis i came from disk position inv_perm[i]
    inv_perm = [0] * ndim
    for disk_pos, orig_axis in enumerate(disk_axes):
        inv_perm[orig_axis] = disk_pos
    data = data.transpose(inv_perm)

    return np.ascontiguousarray(data)


def _mif_apply_layout_for_write(data: np.ndarray, layout: list[int]) -> np.ndarray:
    """Reorder a numpy array into MIF disk layout for writing (axis ordering only)."""
    ndim = len(data.shape)

    # Transpose to disk order: [slowest, ..., fastest] in C-order
    order = sorted(range(ndim), key=lambda i: abs(layout[i]))
    disk_axes = list(reversed(order))
    data = data.transpose(disk_axes)
    return np.ascontiguousarray(data)
