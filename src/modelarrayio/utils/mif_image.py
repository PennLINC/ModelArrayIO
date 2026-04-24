"""Nibabel-compatible image classes for MIF (.mif / .mif.gz) files."""

from copy import deepcopy
from typing import Self

import numpy as np
from nibabel.filebasedimages import FileBasedHeader
from nibabel.spatialimages import SpatialImage

from modelarrayio.utils._mif_format import (
    _mif_apply_layout,
    _mif_apply_layout_for_write,
    _mif_dtype_to_str,
    _mif_layout_to_str,
    _mif_parse_dtype,
    _mif_parse_layout,
    _readline,
)


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
    def from_fileobj(cls, fileobj) -> Self:
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

    def copy(self) -> Self:
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
        """Return the 4x4 affine mapping canonical voxel indices to scanner space (mm).

        The image data returned by :meth:`MifImage.get_fdata` is always in
        mrtrix-canonical (positive-stride) order, so the affine is simply
        built from the MIF ``transform`` and ``vox`` fields:

            affine[:3, :3] = transform[:, :3] * vox   # column-wise scale
            affine[:3,  3] = transform[:,  3]
        """
        affine = np.eye(4, dtype=np.float64)
        n_spatial = min(3, len(self._zooms), len(self._shape))
        zooms = np.ones(3, dtype=np.float64)
        zooms[:n_spatial] = self._zooms[:n_spatial]

        affine[:3, :3] = self._transform[:, :3] * zooms
        affine[:3, 3] = self._transform[:, 3]
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
