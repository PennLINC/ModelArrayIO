"""Utility functions for fixel-wise data."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.filebasedimages import FileBasedHeader
from nibabel.spatialimages import SpatialImage


def image_to_mif(image: SpatialImage, mif_file):
    """Write a nibabel image to a `.mif` file.

    Parameters
    ----------
    image : :obj:`nibabel.spatialimages.SpatialImage`
        In-memory image to write.
    mif_file : :obj:`str`
        Path to a .mif file
    """
    output_path = Path(mif_file)
    if isinstance(image, MifImage):
        mif_image = image
    else:
        source_header = image.header if isinstance(image.header, MifHeader) else None
        header = source_header.copy() if source_header is not None else None
        mif_image = MifImage(np.asanyarray(image.dataobj), image.affine, header=header)

    mif_image.to_filename(output_path)

    if not output_path.exists():
        raise RuntimeError(f'Failed to create expected output file: {output_path}')


def nifti2_to_mif(nifti2_image, mif_file):
    """Convert an in-memory nibabel image to a `.mif` file.

    This compatibility wrapper now writes the data directly with
    :class:`MifImage` instead of shelling out through a temporary NIfTI file.
    """
    image_to_mif(nifti2_image, mif_file)


def mif_to_image(mif_file):
    """Load a `.mif` file into a :class:`MifImage`.

    Parameters
    ----------
    mif_file : :obj:`str`
        Path to a .mif file

    Returns
    -------
    mif_img : :obj:`MifImage`
        Loaded MIF image
    data : :obj:`numpy.ndarray`
        Data from the MIF image
    """
    input_path = Path(mif_file)
    mif_img = MifImage.from_filename(str(input_path))
    data = mif_img.get_fdata(dtype=np.float32).squeeze()
    return mif_img, data


# ============================================================
# MIF (MRtrix Image Format) reader/writer
# ============================================================


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


class MifHeader(FileBasedHeader):
    """Header for MIF (.mif / .mif.gz) image files.

    The MIF format uses a text header with ``key: value`` pairs followed by
    ``END``, then binary image data at the byte offset given by the
    ``file: . <offset>`` entry.

    The transform stored in the file contains *unit direction cosines* for
    each voxel axis; voxel sizes are stored separately in the ``vox`` field.
    The nibabel 4x4 affine is reconstructed as::

        affine[:3, :3] = transform[:3, :3] * zooms   # column-wise scaling
        affine[:3,  3] = transform[:3, 3]             # translation unchanged
    """

    def __init__(
        self,
        shape: tuple = (1,),
        zooms: tuple | None = None,
        layout: list[int] | None = None,
        dtype: np.dtype | None = None,
        transform: np.ndarray | None = None,
        intensity_offset: float = 0.0,
        intensity_scale: float = 1.0,
        keyval: dict | None = None,
    ) -> None:
        self._shape = tuple(int(s) for s in shape)
        ndim = len(self._shape)
        self._zooms = tuple(float(z) for z in zooms) if zooms is not None else (1.0,) * ndim
        self._layout = list(layout) if layout is not None else list(range(1, ndim + 1))
        self._dtype = np.dtype(dtype) if dtype is not None else np.dtype('f4')
        if transform is not None:
            self._transform = np.array(transform, dtype=np.float64).reshape(3, 4)
        else:
            self._transform = np.eye(3, 4, dtype=np.float64)
        self._intensity_offset = float(intensity_offset)
        self._intensity_scale = float(intensity_scale)
        self._keyval: dict[str, str] = dict(keyval) if keyval is not None else {}
        self._data_offset: int | None = None  # populated by from_fileobj

    @classmethod
    def from_header(cls, header=None):
        if header is None:
            return cls()
        if type(header) is cls:
            return header.copy()
        raise NotImplementedError(f'Cannot convert {type(header)} to {cls}')

    @classmethod
    def from_fileobj(cls, fileobj) -> MifHeader:
        """Read a MIF header from a binary file-like object.

        Uses only ``read(1)`` internally so it works with nibabel's
        ``ImageOpener`` and gzip streams as well as regular file objects.
        """
        first_line = _readline(fileobj).decode('latin-1').rstrip('\n\r')
        if first_line != 'mrtrix image':
            raise ValueError(f'Not a MIF file (expected "mrtrix image", got {first_line!r})')

        shape = None
        zooms = None
        layout_str = None
        dtype_str = None
        transform_rows: list[list[float]] = []
        scaling = None
        keyval: dict[str, str] = {}
        file_entry = None

        while True:
            line = _readline(fileobj).decode('latin-1')
            line = line.rstrip('\n\r')
            if line == 'END' or not line:
                break
            comment_pos = line.find('#')
            if comment_pos >= 0:
                line = line[:comment_pos]
            line = line.strip()
            if not line or ':' not in line:
                continue

            colon = line.index(':')
            key = line[:colon].strip()
            value = line[colon + 1 :].strip()
            if not key or not value:
                continue

            lkey = key.lower()
            if lkey == 'dim':
                shape = tuple(int(x.strip()) for x in value.split(','))
            elif lkey == 'vox':
                zooms = tuple(float(x.strip()) for x in value.split(','))
            elif lkey == 'layout':
                layout_str = value
            elif lkey == 'datatype':
                dtype_str = value
            elif lkey == 'transform':
                transform_rows.append([float(x.strip()) for x in value.split(',')])
            elif lkey == 'scaling':
                scaling = [float(x.strip()) for x in value.split(',')]
            elif lkey == 'file':
                file_entry = value
            else:
                # Preserve case and accumulate multi-line values
                keyval[key] = (keyval[key] + '\n' + value) if key in keyval else value

        if shape is None:
            raise ValueError('Missing "dim" in MIF header')
        if zooms is None:
            raise ValueError('Missing "vox" in MIF header')
        if dtype_str is None:
            raise ValueError('Missing "datatype" in MIF header')
        if layout_str is None:
            raise ValueError('Missing "layout" in MIF header')

        dtype = _mif_parse_dtype(dtype_str)
        layout = _mif_parse_layout(layout_str, len(shape))

        transform = np.eye(3, 4, dtype=np.float64)
        if len(transform_rows) >= 3:
            for r in range(3):
                for c in range(min(4, len(transform_rows[r]))):
                    transform[r, c] = transform_rows[r][c]

        intensity_offset, intensity_scale = 0.0, 1.0
        if scaling is not None and len(scaling) == 2:
            intensity_offset, intensity_scale = scaling[0], scaling[1]

        hdr = cls(
            shape=shape,
            zooms=zooms,
            layout=layout,
            dtype=dtype,
            transform=transform,
            intensity_offset=intensity_offset,
            intensity_scale=intensity_scale,
            keyval=keyval,
        )

        if file_entry is not None:
            parts = file_entry.split()
            if len(parts) >= 2:
                hdr._data_offset = int(parts[1])
            elif len(parts) == 1 and parts[0] != '.':
                hdr._data_offset = 0  # external data file (MIH format)

        return hdr

    def write_to(self, fileobj, data_offset: int | None = None) -> int:
        """Write the MIF header to *fileobj*, returning the data byte offset.

        The ``file: . <offset>\\nEND\\n`` footer and any alignment padding are
        written so that the caller can immediately append the binary data.
        """
        lines = ['mrtrix image']
        lines.append(f'dim: {",".join(str(s) for s in self._shape)}')
        lines.append(f'vox: {",".join(str(float(z)) for z in self._zooms)}')
        lines.append(f'layout: {_mif_layout_to_str(self._layout)}')
        lines.append(f'datatype: {_mif_dtype_to_str(self._dtype)}')

        for row in range(3):
            row_vals = ', '.join(repr(float(self._transform[row, col])) for col in range(4))
            lines.append(f'transform: {row_vals}')

        if self._intensity_offset != 0.0 or self._intensity_scale != 1.0:
            lines.append(f'scaling: {self._intensity_offset},{self._intensity_scale}')

        for key, value in self._keyval.items():
            lines.extend(f'{key}: {line_val}' for line_val in value.split('\n'))

        pre_file_bytes = ('\n'.join(lines) + '\n').encode('latin-1')
        pre_file_pos = len(pre_file_bytes)

        if data_offset is None:
            # Iteratively compute the offset so that the file: line fits exactly.
            file_prefix = b'file: . '
            end_suffix = b'\nEND\n'
            offset = pre_file_pos + len(file_prefix) + 5 + len(end_suffix)
            offset += (4 - offset % 4) % 4
            for _ in range(5):
                file_line = file_prefix + str(offset).encode() + end_suffix
                total = pre_file_pos + len(file_line)
                new_offset = total + (4 - total % 4) % 4
                if new_offset == offset:
                    break
                offset = new_offset
            data_offset = offset

        file_line = f'file: . {data_offset}\nEND\n'.encode('latin-1')
        fileobj.write(pre_file_bytes)
        fileobj.write(file_line)

        current_pos = pre_file_pos + len(file_line)
        padding = data_offset - current_pos
        if padding > 0:
            fileobj.write(b'\x00' * padding)

        return data_offset

    def copy(self) -> MifHeader:
        return deepcopy(self)

    # ------------------------------------------------------------------
    # Nibabel SpatialHeader protocol
    # ------------------------------------------------------------------

    def get_data_shape(self) -> tuple:
        return self._shape

    def set_data_shape(self, shape) -> None:
        self._shape = tuple(int(s) for s in shape)

    def get_zooms(self) -> tuple:
        return self._zooms

    def set_zooms(self, zooms) -> None:
        self._zooms = tuple(float(z) for z in zooms)

    def get_data_dtype(self) -> np.dtype:
        return self._dtype

    def set_data_dtype(self, dtype) -> None:
        self._dtype = np.dtype(dtype)

    def get_layout(self) -> list[int]:
        return list(self._layout)

    def get_transform(self) -> np.ndarray:
        """Return a copy of the 3x4 direction-cosine + translation matrix."""
        return self._transform.copy()

    def get_best_affine(self) -> np.ndarray:
        """Return the 4x4 affine mapping *disk* voxel indices to scanner space (mm).

        This follows mrconvert's NIfTI output convention: data is kept in the
        on-disk byte order (no axis flips are applied), and the affine is
        adjusted so that voxel ``(0, 0, 0, …)`` maps to the scanner position
        of the first element on disk.

        For axes with a positive layout stride the column equals::

            transform_col * vox

        For axes with a **negative** layout stride (stored reversed on disk) the
        column is **negated** and the origin is shifted by
        ``(dim - 1) * transform_col * vox`` so that disk voxel ``(0, …)``
        maps to the scanner position of the last mrtrix voxel along that axis::

            new_col    = -transform_col * vox
            new_origin  = origin + transform_col * vox * (dim - 1)
        """
        affine = np.eye(4, dtype=np.float64)
        n_spatial = min(3, len(self._zooms), len(self._shape))
        zooms = np.ones(3, dtype=np.float64)
        zooms[:n_spatial] = self._zooms[:n_spatial]

        rotation_cols = self._transform[:, :3] * zooms  # shape (3, 3)
        origin = self._transform[:, 3].copy()

        for i, s in enumerate(self._layout):
            if i < 3 and s < 0 and self._shape[i] > 1:
                # disk voxel 0 on this axis = mrtrix voxel (dim_i - 1)
                origin += rotation_cols[:, i] * (self._shape[i] - 1)
                rotation_cols[:, i] = -rotation_cols[:, i]

        affine[:3, :3] = rotation_cols
        affine[:3, 3] = origin
        return affine

    def get_intensity_scaling(self) -> tuple[float, float]:
        """Return ``(offset, scale)`` for intensity rescaling."""
        return self._intensity_offset, self._intensity_scale

    def get_keyval(self) -> dict[str, str]:
        return dict(self._keyval)

    __hash__ = None  # required because __eq__ is defined

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MifHeader):
            return NotImplemented
        return (
            self._shape == other._shape
            and self._zooms == other._zooms
            and self._layout == other._layout
            and self._dtype == other._dtype
            and np.allclose(self._transform, other._transform)
            and self._intensity_offset == other._intensity_offset
            and self._intensity_scale == other._intensity_scale
            and self._keyval == other._keyval
        )


class MifImage(SpatialImage):
    """Nibabel-style image class for MIF (.mif / .mif.gz) files.

    Supports reading and writing the MRtrix Image Format, including gzip
    compression.  The public API mirrors standard nibabel images::

        img = MifImage.load('image.mif')
        data = img.get_fdata()
        affine = img.affine

        new_img = MifImage(data, affine)
        new_img.to_filename('output.mif')
        new_img.to_filename('output.mif.gz')

    The MIF *layout* field (e.g. ``-0,-1,+2``) describes which axis varies
    fastest on disk and in which direction.  :meth:`get_fdata` always returns
    a C-contiguous array indexed as ``data[x, y, z, ...]`` regardless of the
    on-disk layout.
    """

    header_class = MifHeader
    files_types = (('image', '.mif'),)
    valid_exts = ('.mif',)
    _compressed_suffixes = ('.gz',)

    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None):
        super().__init__(dataobj, affine, header=header, extra=extra, file_map=file_map)
        # Ensure layout has the right number of axes for freshly created images
        if header is None and hasattr(dataobj, 'shape') and len(dataobj.shape) > 0:
            ndim = len(dataobj.shape)
            if len(self._header._layout) != ndim:
                self._header._layout = list(range(1, ndim + 1))
            self._header.set_data_dtype(np.asarray(dataobj).dtype)

    @classmethod
    def from_file_map(cls, file_map, *, mmap=False, keep_file_open=None):
        """Load a MIF image from a nibabel *file_map* dict."""
        img_fh = file_map['image']
        with img_fh.get_prepare_fileobj(mode='rb') as fileobj:
            header = cls.header_class.from_fileobj(fileobj)
            data_offset = header._data_offset
            if data_offset is None:
                raise ValueError('Could not determine data offset from MIF header')

            current_pos = fileobj.tell()
            skip = data_offset - current_pos
            if skip > 0:
                fileobj.read(skip)  # works for both seekable and gzip streams
            elif skip < 0:
                fileobj.seek(data_offset)

            shape = header.get_data_shape()
            dtype = header.get_data_dtype()
            n_bytes = int(np.prod(shape)) * dtype.itemsize
            raw = np.frombuffer(fileobj.read(n_bytes), dtype=dtype)

        data = _mif_apply_layout(raw, shape, header.get_layout())

        affine = header.get_best_affine()
        off, scale = header.get_intensity_scaling()
        if scale != 1.0 or off != 0.0:
            data = data.astype(np.float64) * scale + off

        img = cls(data, affine, header=header, file_map=file_map)
        img._affine = affine  # keep the exact affine from the header
        return img

    def to_file_map(self, file_map=None, dtype=None):
        """Save the image to the files described by *file_map*."""
        if file_map is None:
            file_map = self.file_map

        self.update_header()
        header = self._header

        if dtype is not None:
            header.set_data_dtype(np.dtype(dtype))

        data = np.asanyarray(self._dataobj)

        img_fh = file_map['image']
        with img_fh.get_prepare_fileobj(mode='wb') as fileobj:
            header.write_to(fileobj)
            disk_data = _mif_apply_layout_for_write(data, header.get_layout())
            fileobj.write(disk_data.astype(header.get_data_dtype()).tobytes())

    def _affine2header(self):
        """Sync the nibabel affine back into the MIF header fields."""
        if self._affine is None:
            return
        hdr = self._header
        affine = self._affine
        # Extract voxel sizes as column norms of the rotation+scale part
        zooms = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        ndim = len(hdr.get_data_shape())

        zooms_list = list(hdr.get_zooms())
        n_spatial = min(3, ndim)
        zooms_list[:n_spatial] = zooms[:n_spatial].tolist()
        hdr.set_zooms(zooms_list)

        # Store unit direction cosines and translation
        transform = np.zeros((3, 4), dtype=np.float64)
        safe_zooms = np.where(zooms > 0, zooms, 1.0)
        transform[:, :3] = affine[:3, :3] / safe_zooms
        transform[:, 3] = affine[:3, 3]
        hdr._transform = transform


def gather_fixels(index_file, directions_file):
    """Load the index and directions files to get lookup tables.

    Parameters
    ----------
    index_file : :obj:`str`
        Path to a Nifti2 index file
    directions_file : :obj:`str`
        Path to a Nifti2 directions file

    Returns
    -------
    fixel_table : :obj:`pandas.DataFrame`
        DataFrame with fixel_id, voxel_id, x, y, z
    voxel_table : :obj:`pandas.DataFrame`
        DataFrame with voxel_id, i, j, k
    """
    _index_img, index_data = mif_to_image(index_file)
    # number of fixels in each voxel; by index.mif definition
    count_vol = index_data[..., 0].astype(np.uint32)
    # index of the first fixel in this voxel, in the list of all fixels
    # (in directions.mif, FD.mif, etc)
    id_vol = index_data[..., 1]
    max_id = id_vol.max()
    # = the maximum id of fixels + 1 = # of fixels in entire image
    terminal_counts = count_vol[id_vol == max_id]
    if terminal_counts.size == 0:
        raise ValueError('Could not determine the final fixel count from the index image.')
    max_fixel_id = int(max_id) + int(terminal_counts.max())
    voxel_mask = count_vol > 0  # voxels that contains fixel(s), =1
    masked_ids = id_vol[voxel_mask]  # 1D array, len = # of voxels with fixel(s), value see id_vol
    masked_counts = count_vol[voxel_mask]  # dim as masked_ids; value see count_vol
    # indices that would sort array masked_ids value (i.e. first fixel's id in this voxel) from
    # lowest to highest; so it's sorting voxels by their first fixel id
    id_sort = np.argsort(masked_ids)
    sorted_counts = masked_counts[id_sort]
    # dim: [# of voxels with fixel(s)] x 3, each row is the subscript i.e. (i,j,k) in 3D
    # image of a voxel with fixel
    voxel_coords = np.column_stack(np.nonzero(count_vol))

    fixel_id = 0
    fixel_ids = np.arange(max_fixel_id, dtype=np.int32)
    fixel_voxel_ids = np.zeros_like(fixel_ids)
    for voxel_id, fixel_count in enumerate(sorted_counts):
        for _ in range(fixel_count):
            # fixel_voxel_ids: 1D, len = # of fixels; each value is the voxel_id of the voxel
            # where this fixel locates
            fixel_voxel_ids[fixel_id] = voxel_id
            fixel_id += 1
    sorted_coords = voxel_coords[id_sort]

    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(voxel_coords.shape[0]),
            'i': sorted_coords[:, 0],
            'j': sorted_coords[:, 1],
            'k': sorted_coords[:, 2],
        }
    )

    _directions_img, directions_data = mif_to_image(directions_file)
    fixel_table = pd.DataFrame(
        {
            'fixel_id': fixel_ids,
            'voxel_id': fixel_voxel_ids,
            'x': directions_data[:, 0],
            'y': directions_data[:, 1],
            'z': directions_data[:, 2],
        }
    )

    return fixel_table, voxel_table
