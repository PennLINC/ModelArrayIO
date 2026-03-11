import os
import logging
import tempfile
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_s3_path(path: str) -> bool:
    """Return True if path is an S3 URI (s3://)."""
    return str(path).startswith("s3://")


def download_s3_file(s3_path: str) -> str:
    """Download an S3 object to a local temporary file.

    Returns the local path. The caller is responsible for deleting the file.
    Requires boto3: pip install modelarrayio[s3].
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for s3:// paths. "
            "Install with: pip install modelarrayio[s3]"
        )
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    # Preserve compound extensions (e.g. .nii.gz) so nibabel can identify
    # the format from the tempfile name.
    fname = os.path.basename(key)
    if fname.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif fname.endswith(".mif.gz"):
        suffix = ".mif.gz"
    else:
        suffix = os.path.splitext(fname)[-1] or ".tmp"
    fd, local_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    # Honour MODELARRAYIO_S3_ANON=1 for public buckets (e.g. testing against
    # fcp-indi). In production the standard boto3 credential chain is used.
    anon = os.environ.get("MODELARRAYIO_S3_ANON", "").lower() in ("1", "true", "yes")
    if anon:
        from botocore import UNSIGNED
        from botocore.config import Config
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    else:
        s3 = boto3.client("s3")
    logger.debug("Downloading s3://%s/%s -> %s", bucket, key, local_path)
    s3.download_file(bucket, key, local_path)
    return local_path


def open_path(path: str) -> tuple[str, bool]:
    """Resolve a path to a local file, downloading from S3 if needed.

    Intended for use inside worker functions where relative_root has already
    been applied to local paths before job submission.

    Returns
    -------
    local_path : str
    is_temp : bool
        True if the caller must delete ``local_path`` after use.
    """
    if is_s3_path(path):
        return download_s3_file(path), True
    return path, False
