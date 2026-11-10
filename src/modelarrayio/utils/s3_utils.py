"""S3 utilities."""

from __future__ import annotations

import gzip
import logging
import os
import threading
from io import BytesIO
from os import PathLike
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_S3_CLIENT_CACHE = {'client': None, 'key': None}
_S3_CLIENT_LOCK = threading.Lock()
_S3_SETTINGS = {'pool_connections': 10}


def is_s3_path(path: str | PathLike[str]) -> bool:
    """Return True if path is an S3 URI (s3://)."""
    return str(path).startswith('s3://')


def configure_s3_client_pool(workers: int) -> None:
    """Ensure new shared S3 clients have enough connections for loaders."""
    pool_connections = max(10, int(workers))
    if pool_connections == _S3_SETTINGS['pool_connections']:
        return
    _S3_SETTINGS['pool_connections'] = pool_connections
    clear_s3_client_cache()


def s3_client_pool_connections() -> int:
    """Return the configured shared-client connection-pool size."""
    return _S3_SETTINGS['pool_connections']


def clear_s3_client_cache() -> None:
    """Clear cached clients (primarily for tests and credential changes)."""
    with _S3_CLIENT_LOCK:
        client = _S3_CLIENT_CACHE['client']
        _S3_CLIENT_CACHE.update(client=None, key=None)
    close = getattr(client, 'close', None)
    if close is not None:
        close()


def _make_s3_client():
    """Create a boto3 S3 client.

    Uses anonymous (unsigned) access when the environment variable
    MODELARRAYIO_S3_ANON=1 is set (useful for public buckets such as
    fcp-indi).  Otherwise the standard boto3 credential chain is used
    (env vars, ~/.aws/credentials, IAM instance profile, etc.).

    Raises ImportError if boto3 is not installed.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            'boto3 is required for s3:// paths. Install with: pip install modelarrayio[s3]'
        ) from None
    anon = os.environ.get('MODELARRAYIO_S3_ANON', '').lower() in ('1', 'true', 'yes')
    pool_connections = s3_client_pool_connections()
    cache_key = (anon, pool_connections)
    with _S3_CLIENT_LOCK:
        if _S3_CLIENT_CACHE['client'] is not None and _S3_CLIENT_CACHE['key'] == cache_key:
            return _S3_CLIENT_CACHE['client']

        previous_client = _S3_CLIENT_CACHE['client']

        from botocore.config import Config

        config_kwargs = {'max_pool_connections': pool_connections}
        if anon:
            from botocore import UNSIGNED

            config_kwargs['signature_version'] = UNSIGNED
        client = boto3.client('s3', config=Config(**config_kwargs))
        _S3_CLIENT_CACHE.update(client=client, key=cache_key)
    close = getattr(previous_client, 'close', None)
    if close is not None:
        close()
    return client


def read_s3_bytes(path: str | PathLike[str], *, decompress_gzip: bool = False) -> bytes:
    """Read an S3 object with the shared client and always close its body."""
    if not is_s3_path(path):
        raise ValueError(f'Expected an s3:// URI, got {path!r}.')
    parsed = urlparse(str(path))
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    if not bucket or not key:
        raise ValueError(f'Invalid S3 URI: {path!r}.')

    logger.debug('Loading s3://%s/%s into memory', bucket, key)
    response = _make_s3_client().get_object(Bucket=bucket, Key=key)
    body = response['Body']
    try:
        data = body.read()
    finally:
        close = getattr(body, 'close', None)
        if close is not None:
            close()
    if decompress_gzip and os.path.basename(key).endswith('.gz'):
        return gzip.decompress(data)
    return data


def _check_local_source(path: str | PathLike[str]) -> Path:
    """Validate a local source and give actionable git-annex diagnostics."""
    local_path = Path(path)
    if local_path.exists():
        return local_path
    hint = ''
    if local_path.is_symlink():
        hint = (
            ' The path is a broken symlink; if it belongs to DataLad/git-annex, '
            'materialize it first (for example, with `datalad get`).'
        )
    raise FileNotFoundError(f'Image source does not exist: {local_path}.{hint}')


def release_image_buffers(image) -> None:
    """Close in-memory file objects after a nibabel image has been decoded.

    Images loaded from S3 retain their downloaded bytes through ``FileHolder``
    objects. Callers that have already copied the decoded data can release
    those buffers while continuing to use the parsed image header.
    """
    seen: set[int] = set()
    for holder in getattr(image, 'file_map', {}).values():
        fileobj = getattr(holder, 'fileobj', None)
        if fileobj is None or id(fileobj) in seen:
            continue
        seen.add(id(fileobj))
        close = getattr(fileobj, 'close', None)
        if close is not None:
            close()


def load_nibabel(path: str | PathLike[str], *, cifti: bool = False):
    """Load a nibabel image from a local path or an s3:// URI.

    For s3:// paths the object is downloaded directly into memory via
    ``get_object``; no temporary file is written to disk.  The bytes are
    decompressed in-memory if the key ends with ``.gz``, then handed to
    nibabel through its ``FileHolder`` / ``from_file_map`` API.

    Parameters
    ----------
    path : str
        Local file path or s3:// URI.
    cifti : bool
        Pass ``True`` for CIFTI-2 files (``.dscalar.nii`` etc.) so that
        nibabel returns a ``Cifti2Image`` with proper axes.  ``False``
        (default) returns a ``Nifti1Image``.

    Returns
    -------
    nibabel.FileBasedImage
    """
    import nibabel as nb

    if not is_s3_path(path):
        return nb.load(_check_local_source(path))

    data = read_s3_bytes(path, decompress_gzip=True)

    from nibabel.filebasedimages import FileHolder

    fh = FileHolder(fileobj=BytesIO(data))
    file_map = {'header': fh, 'image': fh}

    if cifti:
        return nb.Cifti2Image.from_file_map(file_map)
    return nb.Nifti1Image.from_file_map(file_map)
