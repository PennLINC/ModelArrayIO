"""Utility functions for CIFTI data."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO

import nibabel as nb
import numpy as np
import pandas as pd

from modelarrayio.utils.loading import (
    LazyScalarMatrices,
    LoadedRow,
    SourceLoadError,
    resolve_loader_workers,
    stage_source_rows,
)
from modelarrayio.utils.misc import build_scalar_sources
from modelarrayio.utils.s3_utils import is_s3_path, load_nibabel, release_image_buffers

logger = logging.getLogger(__name__)

# NIfTI-2 fixed header length in bytes. The CIFTI XML extension and the data
# block follow it, and ``vox_offset`` in this header points past both.
_NIFTI2_HEADER_SIZE = 540

# Environment switch to disable the raw CIFTI fast path (diagnostics
# and strict-validation runs). Any of 0/false/no/off turns it off.
_RAW_READ_ENV = 'MODELARRAYIO_CIFTI_RAW'


def _hash_signature_value(hasher, value):
    """Feed nested axis state into a stable digest without materializing labels."""
    if value is None:
        hasher.update(b'none;')
    elif isinstance(value, dict):
        hasher.update(b'dict{')
        for key in sorted(value, key=str):
            _hash_signature_value(hasher, str(key))
            _hash_signature_value(hasher, value[key])
        hasher.update(b'}')
    elif isinstance(value, np.ndarray):
        hasher.update(f'array:{value.dtype.str}:{value.shape};'.encode())
        if value.dtype.hasobject:
            for item in value.flat:
                _hash_signature_value(hasher, item)
        elif value.dtype.kind in {'U', 'S'}:
            for item in value.flat:
                _hash_signature_value(hasher, str(item))
        else:
            hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        hasher.update(f'seq:{len(value)}['.encode())
        for item in value:
            _hash_signature_value(hasher, item)
        hasher.update(b']')
    elif isinstance(value, bytes):
        hasher.update(len(value).to_bytes(8, 'little'))
        hasher.update(value)
    else:
        encoded = str(value).encode('utf-8')
        hasher.update(len(encoded).to_bytes(8, 'little'))
        hasher.update(encoded)


def cifti_spatial_signature(cifti_file):
    """Return a short signature of all non-scalar CIFTI axis properties."""
    if not hasattr(cifti_file, 'get_fdata'):
        cifti = load_nibabel(cifti_file, cifti=True)
        try:
            return cifti_spatial_signature(cifti)
        finally:
            release_image_buffers(cifti)
    cifti = cifti_file
    hasher = hashlib.blake2b(digest_size=20)
    spatial_count = 0
    for axis_index in range(cifti.ndim):
        axis = cifti.header.get_axis(axis_index)
        if isinstance(axis, nb.cifti2.cifti2_axes.ScalarAxis):
            continue
        spatial_count += 1
        _hash_signature_value(hasher, axis_index)
        _hash_signature_value(hasher, type(axis).__name__)
        # Public nibabel properties only, read without a default: if one is ever
        # renamed, the AttributeError surfaces instead of silently weakening the
        # signature.
        if isinstance(axis, nb.cifti2.cifti2_axes.BrainModelAxis):
            for value in (
                axis.voxel,
                axis.vertex,
                axis.name,
                axis.nvertices,
                axis.affine,
                axis.volume_shape,
            ):
                _hash_signature_value(hasher, value)
        elif isinstance(axis, nb.cifti2.cifti2_axes.ParcelsAxis):
            for value in (
                axis.name,
                axis.voxels,
                axis.vertices,
                axis.affine,
                axis.volume_shape,
                axis.nvertices,
            ):
                _hash_signature_value(hasher, value)
        else:
            _hash_signature_value(hasher, vars(axis))
    if spatial_count == 0:
        raise ValueError('CIFTI image does not contain a supported spatial axis.')
    return spatial_count, hasher.hexdigest()


def extract_cifti_scalar_data(
    cifti_file,
    reference_brain_names=None,
    *,
    storage_dtype='float32',
    reference_signature=None,
    spatial_signature=None,
):
    """Load a scalar cifti file and get its data and mapping.

    Supports dscalar (.dscalar.nii), parcellated scalar (.pscalar.nii), and
    parcellated connectivity (.pconn.nii) files. For pconn files the 2-D
    connectivity matrix is flattened to a 1-D array (row-major order) and
    ``brain_structures`` contains the row parcel name repeated once per
    column, so callers receive a consistent 1-D array of element names.

    Parameters
    ----------
    cifti_file : :obj:`str` or :obj:`pathlib.Path` or :obj:`nibabel.Cifti2Image`
        CIFTI2 file on disk or already loaded CIFTI image.
    reference_brain_names : :obj:`numpy.ndarray`
        Array of vertex/parcel names used for cross-file consistency checks.
        For pconn files this must be the ``brain_structures`` value returned
        by a previous call to this function.

    Returns
    -------
    cifti_scalar_data: :obj:`numpy.ndarray`
        The scalar data from the cifti file, always 1-D.  For pconn files
        this is the row-major flattened connectivity matrix.
    brain_structures: :obj:`numpy.ndarray`
        Per-element spatial labels as strings.  For dscalar files these are
        the per-greyordinate brain structure names.  For pscalar files these
        are the parcel names.  For pconn files these are the row-parcel names
        repeated ``n_col_parcels`` times (one entry per flattened element).
    """
    if not hasattr(cifti_file, 'get_fdata'):
        cifti = load_nibabel(cifti_file, cifti=True)
        try:
            return extract_cifti_scalar_data(
                cifti,
                reference_brain_names,
                storage_dtype=storage_dtype,
                reference_signature=reference_signature,
                spatial_signature=spatial_signature,
            )
        finally:
            release_image_buffers(cifti)
    cifti = cifti_file
    if reference_signature is not None:
        signature = spatial_signature or cifti_spatial_signature(cifti)
        if signature != reference_signature:
            raise ValueError(f'Inconsistent spatial axis in CIFTI file {cifti_file!r}.')
    cifti_hdr = cifti.header
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    if len(axes) != 2:
        raise ValueError(
            f'Expected exactly 2 axes in scalar CIFTI data, found {len(axes)} for {cifti_file!r}.'
        )

    scalar_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ScalarAxis)]
    brain_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.BrainModelAxis)]
    parcel_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ParcelsAxis)]

    if len(scalar_axes) == 1 and len(brain_axes) == 1:
        # dscalar.nii: ScalarAxis (axis 0) + BrainModelAxis (axis 1)
        brain_axis = brain_axes[0]
        cifti_data = cifti.get_fdata(
            dtype=np.dtype(storage_dtype).type, caching='unchanged'
        ).reshape(-1)
        brain_names = (
            reference_brain_names
            if reference_signature is not None and reference_brain_names is not None
            else brain_axis.name
        )
        if cifti_data.shape[0] != brain_names.shape[0]:
            raise ValueError(
                f'Mismatch between brain names and data array in {cifti_file!r}: '
                f'{brain_names.shape[0]} names vs {cifti_data.shape[0]} values.'
            )
        if (
            reference_brain_names is not None
            and reference_signature is None
            and not np.array_equal(brain_names, reference_brain_names)
        ):
            raise ValueError(f'Inconsistent greyordinate names in CIFTI file {cifti_file!r}.')
        return cifti_data, brain_names

    elif len(scalar_axes) == 1 and len(parcel_axes) == 1:
        # pscalar.nii: ScalarAxis (axis 0) + ParcelsAxis (axis 1)
        parcel_axis = parcel_axes[0]
        cifti_data = cifti.get_fdata(
            dtype=np.dtype(storage_dtype).type, caching='unchanged'
        ).reshape(-1)
        parcel_names = (
            reference_brain_names
            if reference_signature is not None and reference_brain_names is not None
            else parcel_axis.name
        )
        if cifti_data.shape[0] != parcel_names.shape[0]:
            raise ValueError(
                f'Mismatch between parcel names and data array in {cifti_file!r}: '
                f'{parcel_names.shape[0]} names vs {cifti_data.shape[0]} values.'
            )
        if (
            reference_brain_names is not None
            and reference_signature is None
            and not np.array_equal(parcel_names, reference_brain_names)
        ):
            raise ValueError(f'Inconsistent parcel names in CIFTI file {cifti_file!r}.')
        return cifti_data, parcel_names

    elif len(parcel_axes) == 2:
        # pconn.nii: ParcelsAxis (axis 0, rows) + ParcelsAxis (axis 1, cols)
        row_axis = parcel_axes[0]
        col_axis = parcel_axes[1]
        cifti_data = cifti.get_fdata(dtype=np.dtype(storage_dtype).type, caching='unchanged')
        if cifti_data.ndim != 2:
            raise ValueError(
                f'Expected 2-D parcellated connectivity CIFTI data in {cifti_file!r}.'
            )
        n_rows, n_cols = cifti_data.shape
        if len(row_axis) != n_rows:
            raise ValueError(
                f'Mismatch between row parcel names and data array in {cifti_file!r}: '
                f'{len(row_axis)} names vs {n_rows} rows.'
            )
        if len(col_axis) != n_cols:
            raise ValueError(
                f'Mismatch between column parcel names and data array in {cifti_file!r}: '
                f'{len(col_axis)} names vs {n_cols} columns.'
            )
        if reference_signature is not None and reference_brain_names is not None:
            # The compact signature already covers both parcel axes, so avoid
            # rebuilding or comparing their expanded labels on every source.
            parcel_names = reference_brain_names
        else:
            # One parcel name per flattened entry: repeat each row parcel name
            # once per column for the reference metadata.
            parcel_names = np.repeat(row_axis.name, n_cols)
        cifti_data_flat = cifti_data.ravel()
        if reference_brain_names is not None:
            if reference_brain_names.ndim != 1:
                raise ValueError(
                    f'Expected 1-D reference_brain_names for pconn CIFTI file {cifti_file!r}, '
                    f'got {reference_brain_names.ndim}-D array.'
                )
            if reference_brain_names.shape[0] != cifti_data_flat.shape[0]:
                raise ValueError(
                    f'Mismatch between reference_brain_names and data array in {cifti_file!r}: '
                    f'{reference_brain_names.shape[0]} names vs '
                    f'{cifti_data_flat.shape[0]} flattened connectivity values.'
                )
            if reference_signature is None and not np.array_equal(
                parcel_names, reference_brain_names
            ):
                raise ValueError(f'Inconsistent parcel names in CIFTI file {cifti_file!r}.')
        return cifti_data_flat, parcel_names

    else:
        raise ValueError(
            f'Unsupported CIFTI axis combination in {cifti_file!r}. '
            'Supported types: dscalar (ScalarAxis+BrainModelAxis), '
            'pscalar (ScalarAxis+ParcelsAxis), pconn (ParcelsAxis+ParcelsAxis).'
        )


def _get_cifti_parcel_info(cifti_file):
    """Detect CIFTI type and return parcel name arrays for parcellated files.

    Parameters
    ----------
    cifti_file : str or :obj:`nibabel.Cifti2Image`
        Path to a CIFTI2 file, or an already-loaded image.

    Returns
    -------
    cifti_type : {'dscalar', 'pscalar', 'pconn'}
        The type of the CIFTI file.
    parcel_arrays : dict[str, numpy.ndarray]
        For ``'dscalar'``: empty dict (greyordinate table written separately).
        For ``'pscalar'``: ``{'parcel_id': parcel_names_array}`` — one entry per parcel.
        For ``'pconn'``: ``{'parcel_id_from': row_names, 'parcel_id_to': col_names}`` —
        one entry per unique row/column parcel axis.
    """
    if not hasattr(cifti_file, 'get_fdata'):
        cifti = load_nibabel(cifti_file, cifti=True)
        try:
            return _get_cifti_parcel_info(cifti)
        finally:
            release_image_buffers(cifti)
    cifti = cifti_file
    axes = [cifti.header.get_axis(i) for i in range(cifti.ndim)]

    scalar_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ScalarAxis)]
    brain_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.BrainModelAxis)]
    parcel_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ParcelsAxis)]

    if len(scalar_axes) == 1 and len(brain_axes) == 1:
        return 'dscalar', {}
    elif len(scalar_axes) == 1 and len(parcel_axes) == 1:
        return 'pscalar', {'parcel_id': parcel_axes[0].name}
    elif len(parcel_axes) == 2:
        return 'pconn', {
            'parcel_id_from': parcel_axes[0].name,
            'parcel_id_to': parcel_axes[1].name,
        }
    else:
        raise ValueError(
            f'Unsupported CIFTI axis combination in {cifti_file!r}. '
            'Supported types: dscalar, pscalar, pconn.'
        )


def brain_names_to_dataframe(brain_names):
    """Convert brain names to a dataframe.

    Parameters
    ----------
    brain_names : :obj:`numpy.ndarray`
        Array of brain names

    Returns
    -------
    greyordinate_df : :obj:`pandas.DataFrame`
        DataFrame with vertex_id and structure_id
    structure_name_strings : :obj:`list`
        List of structure names
    """
    # Make a lookup table for greyordinates
    structure_ids, structure_names = pd.factorize(brain_names)
    # Make them a list of strings
    structure_name_strings = list(map(str, structure_names))

    greyordinate_df = pd.DataFrame(
        {'vertex_id': np.arange(structure_ids.shape[0]), 'structure_id': structure_ids}
    )

    return greyordinate_df, structure_name_strings


@dataclass(frozen=True)
class _CiftiRawReadPlan:
    """On-disk layout for reading a CIFTI file's values without parsing its XML.

    A plan captures one reference file's layout so later sources with an
    identical cheap header can be read straight from disk. dscalar, pscalar
    and pconn files qualify (see :func:`_layout_digest`). Only unscaled
    floating data (no ``scl_slope``/``scl_inter``) is served; anything else is
    decoded by nibabel.

    NIfTI stores the ``(n_rows, n_cols)`` CIFTI matrix in Fortran order (the
    row index varies fastest), while the converter stores rows flattened in C
    order, as ``get_fdata().ravel()`` does. A single-row file (dscalar,
    pscalar) has ``n_rows == 1``, so both orders coincide; a pconn's block is
    reordered on read.
    """

    offset: int
    disk_dtype: np.dtype
    n_elements: int
    expected_file_size: int
    storage_dtype: np.dtype
    # Cheap header facts that must match the reference before a data block is
    # trusted: the NIfTI-2 dims and a digest of the CIFTI XML layout (see
    # :func:`_layout_digest`). vox_offset may differ between files — pipelines
    # can embed per-subject provenance strings in the CIFTI metadata, which
    # shifts the data block without changing its layout.
    dims: tuple[int, ...] = ()
    layout_digest: bytes = b''

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """The CIFTI ``(n_rows, n_cols)`` shape: the last two NIfTI dims."""
        return self.dims[-2], self.dims[-1]


_MATRIX_INDICES_MAP_OPEN = b'<MatrixIndicesMap'
_MATRIX_INDICES_MAP_CLOSE = b'</MatrixIndicesMap>'
_APPLIES_TO_DIMENSION_RE = re.compile(rb'AppliesToMatrixDimension\s*=\s*["\']([^"\']*)["\']')
_INDEX_TYPE_RE = re.compile(rb'IndicesMapToDataType\s*=\s*["\']([^"\']*)["\']')
_SCALARS_INDEX_TYPE = b'CIFTI_INDEX_TYPE_SCALARS'
_PARCELS_INDEX_TYPE = b'CIFTI_INDEX_TYPE_PARCELS'
_SINGLE_ROW_SPATIAL_INDEX_TYPES = frozenset(
    {b'CIFTI_INDEX_TYPE_BRAIN_MODELS', _PARCELS_INDEX_TYPE}
)


def _layout_digest(buffer: bytes, end: int | None = None) -> tuple[int, bytes] | None:
    """Return ``(required_rows, digest)`` for a supported CIFTI XML layout, or None.

    Only ``buffer[:end]`` is examined (all of it when *end* is None); it must
    contain the CIFTI XML. The XML is scanned with bounded byte searches, never
    parsed, and nothing is copied except the short opening tags. The supported
    layouts are exactly the ones :func:`extract_cifti_scalar_data` accepts:

    * dscalar / pscalar — a scalar map for dimension 0 and a brain-model or
      parcels map for dimension 1. The first item of the result is ``1``: the
      matrix must have exactly one row. Scalar map names are excluded from the
      digest because they may differ between subjects without changing the
      data layout.
    * pconn — parcels maps for dimensions 0 and 1, or one parcels map that
      applies to both (``AppliesToMatrixDimension="0,1"``). The first item is
      ``0``: no constraint on the number of rows. Every map's bytes are in the digest, so both
      row and column parcels must be byte-identical.

    Anything else — a label, series or dense-connectivity map, a missing or
    duplicated map, or unrecognised markup — returns None, which keeps that
    file on the nibabel path.
    """
    limit = len(buffer) if end is None else end
    xml_start = buffer.find(b'<CIFTI', 0, limit)
    if xml_start < 0:
        return None
    maps: dict[bytes, tuple[bytes, int, int]] = {}
    position = xml_start
    while (start := buffer.find(_MATRIX_INDICES_MAP_OPEN, position, limit)) >= 0:
        tag_end = buffer.find(b'>', start, limit)
        close = buffer.find(_MATRIX_INDICES_MAP_CLOSE, start, limit)
        if tag_end < 0 or close < 0:
            return None
        close += len(_MATRIX_INDICES_MAP_CLOSE)
        opening_tag = buffer[start : tag_end + 1]
        dimension = _APPLIES_TO_DIMENSION_RE.search(opening_tag)
        index_type = _INDEX_TYPE_RE.search(opening_tag)
        if dimension is None or index_type is None:
            return None
        key = dimension.group(1).replace(b' ', b'')
        if key in maps:
            return None
        maps[key] = (index_type.group(1).strip(), start, close)
        position = close

    view = memoryview(buffer)
    types = {key: index_type for key, (index_type, _, _) in maps.items()}
    if set(maps) == {b'0', b'1'} and types[b'0'] == _SCALARS_INDEX_TYPE:
        if types[b'1'] not in _SINGLE_ROW_SPATIAL_INDEX_TYPES:
            return None
        rows_kind, layout, hashed = 1, b'single-row', (b'1',)
    elif set(maps) == {b'0', b'1'} and types[b'0'] == types[b'1'] == _PARCELS_INDEX_TYPE:
        rows_kind, layout, hashed = 0, b'pconn', (b'0', b'1')
    elif set(maps) == {b'0,1'} and types[b'0,1'] == _PARCELS_INDEX_TYPE:
        rows_kind, layout, hashed = 0, b'pconn-shared-axis', (b'0,1',)
    else:
        return None
    hasher = hashlib.sha256()
    parts = [layout, *(types[key] for key in sorted(types))]
    parts += [view[maps[key][1] : maps[key][2]] for key in hashed]
    for part in parts:
        hasher.update(len(part).to_bytes(8, 'little'))
        hasher.update(part)
    return rows_kind, hasher.digest()


def _raw_read_enabled() -> bool:
    """Return True unless the raw CIFTI fast path is disabled by env."""
    return os.environ.get(_RAW_READ_ENV, '').strip().lower() not in {'0', 'false', 'no', 'off'}


def _is_unscaled(slope, inter) -> bool:
    """Return True when NIfTI ``scl_slope``/``scl_inter`` apply no scaling."""
    return slope in (None, 1.0) and inter in (None, 0.0)


@dataclass(frozen=True)
class _HeaderFacts:
    """What the fast path needs from a file's own 540-byte NIfTI-2 header."""

    offset: int
    dims: tuple[int, ...]
    disk_dtype: np.dtype


def _header_facts(header_bytes: bytes) -> _HeaderFacts | None:
    """Parse a file's on-disk NIfTI-2 header, or return None to use nibabel instead.

    The reference and every follower are judged by this same function, so
    they are compared on identical facts (the on-disk byte order and scaling,
    which nibabel normalises away in a loaded image's header). None is
    returned for anything that is not a NIfTI-2 header, for intensity scaling,
    and for an offset that cannot hold the CIFTI extension.
    """
    if len(header_bytes) != _NIFTI2_HEADER_SIZE:
        return None
    try:
        header = nb.Nifti2Header.from_fileobj(BytesIO(header_bytes), check=False)
        if int(header['sizeof_hdr']) != _NIFTI2_HEADER_SIZE:
            return None
        slope, inter = header.get_slope_inter()
        facts = _HeaderFacts(
            offset=int(header.get_data_offset()),
            dims=tuple(int(value) for value in header.get_data_shape()),
            disk_dtype=np.dtype(header.get_data_dtype()),
        )
    except Exception:  # noqa: BLE001 - any unreadable header must fall back to nibabel
        return None
    if not _is_unscaled(slope, inter) or facts.offset <= _NIFTI2_HEADER_SIZE:
        return None
    return facts


def _raw_values(data: bytes, plan: _CiftiRawReadPlan, offset: int = 0) -> np.ndarray:
    """Decode a data block into the converter's C-order flattened row."""
    values = np.frombuffer(data, dtype=plan.disk_dtype, count=plan.n_elements, offset=offset)
    matrix = values.reshape(plan.matrix_shape, order='F')
    return np.array(matrix, dtype=plan.storage_dtype, order='C').reshape(-1)


def _read_cifti_row_raw(source, plan: _CiftiRawReadPlan) -> np.ndarray | None:
    """Read one source's values straight from its data block, or None to fall back.

    Returns None unless *source* is a local, uncompressed file whose on-disk
    header facts (:func:`_header_facts`) match *plan* in dims and dtype, whose
    CIFTI XML layout digest equals the reference's, and whose file is large
    enough to hold the data block at its own ``vox_offset``. The offset itself
    may differ from the reference (per-subject provenance metadata lengthens
    the XML); the digest is what guarantees the same element order. Any
    mismatch means the caller decodes the source with nibabel.

    The file is opened once and read sequentially: header, XML extension and
    data block in a single read, so network filesystems see one open per file.
    """
    if is_s3_path(source):
        return None
    path = os.fspath(source)
    if path.endswith('.gz'):
        return None
    data_bytes = plan.n_elements * plan.disk_dtype.itemsize
    try:
        with open(path, 'rb') as handle:
            facts = _header_facts(handle.read(_NIFTI2_HEADER_SIZE))
            if facts is None or facts.dims != plan.dims or facts.disk_dtype != plan.disk_dtype:
                return None
            xml_length = facts.offset - _NIFTI2_HEADER_SIZE
            tail = handle.read(xml_length + data_bytes)
    except OSError:
        return None
    if len(tail) < xml_length + data_bytes:
        # Truncated file: the data block does not fit at its own offset.
        return None
    # The XML extension sits at the start of *tail*; digest it in place.
    layout = _layout_digest(tail, xml_length)
    if layout is None or layout[1] != plan.layout_digest:
        return None
    return _raw_values(tail, plan, offset=xml_length)


def _build_cifti_raw_read_plan(
    reference_row: np.ndarray, source, storage_dtype
) -> _CiftiRawReadPlan | None:
    """Capture a raw-read plan from a reference source, or None if ineligible.

    *reference_row* is the reference's values as decoded by nibabel. The plan
    is built from the reference file's own bytes through the same checks every
    follower gets (:func:`_header_facts`, :func:`_layout_digest`); no nibabel
    axis is rebuilt. The reference must be a local, uncompressed file with
    unscaled floating data, a supported layout, and dims consistent with that
    layout and with *reference_row*. As a final, independent check, reading the
    reference's own data block must reproduce *reference_row* exactly.
    """
    if is_s3_path(source):
        return None
    path = os.fspath(source)
    if path.endswith('.gz'):
        return None
    reference_row = np.asarray(reference_row)
    if reference_row.ndim != 1:
        return None
    n_elements = int(reference_row.shape[0])
    try:
        with open(path, 'rb') as handle:
            facts = _header_facts(handle.read(_NIFTI2_HEADER_SIZE))
            if facts is None:
                return None
            xml = handle.read(facts.offset - _NIFTI2_HEADER_SIZE)
    except OSError:
        return None
    if facts.disk_dtype.kind != 'f':
        return None
    dims = facts.dims
    # The CIFTI matrix is the last two NIfTI dims; every earlier dim must be 1.
    if len(dims) < 2 or int(np.prod(dims[:-2])) != 1 or dims[-2] * dims[-1] != n_elements:
        return None
    layout = _layout_digest(xml)
    if layout is None:
        return None
    required_rows, digest = layout
    if required_rows and dims[-2] != required_rows:
        return None
    plan = _CiftiRawReadPlan(
        offset=facts.offset,
        disk_dtype=facts.disk_dtype,
        n_elements=n_elements,
        expected_file_size=facts.offset + n_elements * facts.disk_dtype.itemsize,
        storage_dtype=np.dtype(storage_dtype),
        dims=dims,
        layout_digest=digest,
    )
    probe = _read_cifti_row_raw(path, plan)
    if probe is None:
        return None
    expected = reference_row.astype(plan.storage_dtype, copy=False)
    if not np.array_equal(probe, expected, equal_nan=True):
        logger.debug('Raw CIFTI fast path disabled for %s: reference probe mismatch.', path)
        return None
    return plan


def load_cohort_cifti(
    cohort_long,
    s3_workers=None,
    *,
    workers=None,
    storage_dtype='float32',
):
    """Return lazy per-scalar CIFTI matrices and reference spatial metadata.

    Each scalar's first source is opened and decoded with nibabel once, to fix
    its reference row and spatial axis. Later sources are read straight from
    their data block when their cheap NIfTI-2 header and CIFTI XML layout match
    that reference exactly (see :func:`_read_cifti_row_raw`), which avoids
    re-parsing the CIFTI XML per file. Any source that does not match gets a
    full nibabel decode validated against a compact spatial signature, so both
    paths accept the same cohorts and produce the same values. Set
    ``MODELARRAYIO_CIFTI_RAW=0`` to force the nibabel path for every source.

    Rows are derived with :func:`~modelarrayio.utils.misc.build_scalar_sources`,
    the same helper the CLI uses for ``column_names``, so the staged rows and
    the stored source names always agree.

    Parameters
    ----------
    cohort_long : :obj:`pandas.DataFrame`
        Long-format cohort dataframe
    s3_workers : :obj:`int`
        Deprecated positional alias for the loader worker count.
    workers : :obj:`int`
        Maximum number of concurrent source loads.
    storage_dtype : {'float32', 'float64'}
        Dtype used for decoding and staging.

    Returns
    -------
    scalars : :class:`modelarrayio.utils.loading.LazyScalarMatrices`
        Lazy mapping that stages one ordered, typed matrix per scalar. Its
        ``reference_image``, ``reference_brain_names`` and
        ``reference_signature`` attributes carry the decoded first source.
    reference_brain_names : :obj:`numpy.ndarray`
        Brain structure names from the first file, for building greyordinate table.
    """
    worker_count = resolve_loader_workers(workers, s3_workers)

    sources_by_scalar = build_scalar_sources(cohort_long)
    if not sources_by_scalar:
        raise ValueError('Cannot load CIFTI cohort without any sources.')

    first_scalar = next(iter(sources_by_scalar))
    first_src = sources_by_scalar[first_scalar][0]
    first_started = time.perf_counter()
    first_cpu_started = time.process_time()
    first_image = None
    try:
        first_image = load_nibabel(first_src, cifti=True)
        first_data, reference_brain_names = extract_cifti_scalar_data(
            first_image,
            storage_dtype=storage_dtype,
        )
        validation_started = time.perf_counter()
        reference_signature = cifti_spatial_signature(first_image)
        first_validation_seconds = time.perf_counter() - validation_started
    except Exception as exc:
        raise SourceLoadError(
            f'Failed to load source {str(first_src)!r} for scalar {first_scalar!r}: {exc}'
        ) from exc
    finally:
        if first_image is not None:
            release_image_buffers(first_image)
    first_loaded_row = LoadedRow(
        first_data,
        validation_seconds=first_validation_seconds,
        source_seconds=time.perf_counter() - first_started,
        cpu_seconds=time.process_time() - first_cpu_started,
    )

    raw_enabled = _raw_read_enabled()
    first_plan = (
        _build_cifti_raw_read_plan(first_data, first_src, storage_dtype) if raw_enabled else None
    )

    def _validated_signature(image, source):
        """Return the image's spatial signature, rejecting any axis mismatch."""
        validation_started = time.perf_counter()
        signature = cifti_spatial_signature(image)
        if signature != reference_signature:
            raise ValueError(f'Inconsistent spatial axis in CIFTI file {source!r}.')
        return signature, time.perf_counter() - validation_started

    def _decode_reference_row(source):
        """Full nibabel decode of a scalar's first source, validated against the cohort."""
        started = time.perf_counter()
        cpu_started = time.process_time()
        image = load_nibabel(source, cifti=True)
        try:
            signature, validation_seconds = _validated_signature(image, source)
            values, _ = extract_cifti_scalar_data(
                image,
                reference_brain_names=reference_brain_names,
                storage_dtype=storage_dtype,
                reference_signature=reference_signature,
                spatial_signature=signature,
            )
        except Exception:
            release_image_buffers(image)
            raise
        row = LoadedRow(
            values,
            validation_seconds=validation_seconds,
            source_seconds=time.perf_counter() - started,
            cpu_seconds=time.process_time() - cpu_started,
        )
        return row, image

    def _load(source, plan=None):
        if plan is not None:
            raw_values = _read_cifti_row_raw(source, plan)
            if raw_values is not None:
                return LoadedRow(raw_values, fast_path=True)
        image = load_nibabel(source, cifti=True)
        try:
            signature, validation_seconds = _validated_signature(image, source)
            values, _ = extract_cifti_scalar_data(
                image,
                reference_brain_names=reference_brain_names,
                storage_dtype=storage_dtype,
                reference_signature=reference_signature,
                spatial_signature=signature,
            )
        finally:
            release_image_buffers(image)
        return LoadedRow(values, validation_seconds=validation_seconds)

    def _stage(scalar_name):
        sources = sources_by_scalar[scalar_name]
        if scalar_name == first_scalar:
            reusable_first = first_loaded_row
            plan = first_plan
        else:
            try:
                reusable_first, reference_image = _decode_reference_row(sources[0])
            except Exception as exc:
                raise SourceLoadError(
                    f'Failed to load source {str(sources[0])!r} for scalar {scalar_name!r}: {exc}'
                ) from exc
            try:
                plan = (
                    _build_cifti_raw_read_plan(reusable_first.values, sources[0], storage_dtype)
                    if raw_enabled
                    else None
                )
            finally:
                release_image_buffers(reference_image)

        def _load_source(source):
            return _load(source, plan)

        return stage_source_rows(
            sources,
            _load_source,
            scalar_name=scalar_name,
            storage_dtype=storage_dtype,
            workers=worker_count,
            first_row=reusable_first,
        )

    scalars = LazyScalarMatrices(
        list(sources_by_scalar),
        _stage,
        sources_by_scalar=sources_by_scalar,
        reference_image=first_image,
        reference_brain_names=reference_brain_names,
        reference_signature=reference_signature,
    )
    return scalars, reference_brain_names
