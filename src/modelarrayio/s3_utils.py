import gzip
import logging
import os
from io import BytesIO
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_s3_path(path: str) -> bool:
    """Return True if path is an S3 URI (s3://)."""
    return str(path).startswith('s3://')


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
    if anon:
        from botocore import UNSIGNED
        from botocore.config import Config

        return boto3.client('s3', config=Config(signature_version=UNSIGNED))
    return boto3.client('s3')


def load_nibabel(path: str, *, cifti: bool = False):
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
        return nb.load(path)

    parsed = urlparse(path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    logger.debug('Loading s3://%s/%s into memory', bucket, key)
    data = _make_s3_client().get_object(Bucket=bucket, Key=key)['Body'].read()

    if os.path.basename(key).endswith('.gz'):
        data = gzip.decompress(data)

    from nibabel.filebasedimages import FileHolder

    fh = FileHolder(fileobj=BytesIO(data))
    file_map = {'header': fh, 'image': fh}

    if cifti:
        return nb.Cifti2Image.from_file_map(file_map)
    return nb.Nifti1Image.from_file_map(file_map)
