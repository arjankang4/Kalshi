"""S3 access: region-pinned session, verified upload, streaming download.

The upload path exists to make one thing safe: deleting the local copy. A local
segment is unlinked only after S3 has independently confirmed it holds the same
bytes, because the raw order book capture cannot be re-fetched from any REST
endpoint if it is lost.
"""
import base64
import hashlib
import io
import time

import boto3
from botocore.config import Config as BotoConfig

import config

# Single-part put ceiling. Segments roll at 64 MB so they always take this path,
# which yields a plain (non-composite) SHA-256 that we can compare directly.
SINGLE_PUT_MAX = 128 * 1024 * 1024
_CHUNK = 1 << 20

_session = None


def client():
    """Region is pinned: nothing is configured on this box and boto3 would
    otherwise resolve to us-east-1 only by accident."""
    global _session
    if _session is None:
        _session = boto3.session.Session(region_name=config.AWS_REGION)
    return _session.client(
        "s3",
        config=BotoConfig(
            region_name=config.AWS_REGION,
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=32,
        ),
    )


def sha256_file(path):
    """Streaming digest -- never reads the whole file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_gz_plaintext(path):
    """Digest of the UNCOMPRESSED stream.

    gzip output varies with mtime, level and zlib version, so the compressed
    digest is not a stable identity for the content. The plaintext digest
    survives re-compression and is what the manifests record.
    """
    import gzip
    h = hashlib.sha256()
    n = 0
    with gzip.open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(blk)
            n += len(blk)
    return h.hexdigest(), n


def _b64(hexdigest):
    return base64.b64encode(bytes.fromhex(hexdigest)).decode()


class UploadVerificationError(RuntimeError):
    pass


def put_verified(local_path, bucket, key, extra=None):
    """Upload and prove S3 holds the same bytes. Returns the object metadata.

    Verification is by SHA-256, never by ETag: a multipart ETag is not an MD5 of
    the content, and comparing against one is a well-known way to feel safe while
    being wrong.
    """
    size = local_path.stat().st_size
    if size > SINGLE_PUT_MAX:
        raise ValueError(f"{local_path} is {size} B; use put_large() for multipart")

    digest = sha256_file(local_path)
    c = client()
    with open(local_path, "rb") as fh:
        resp = c.put_object(
            Bucket=bucket, Key=key, Body=fh,
            ChecksumAlgorithm="SHA256",
            **(extra or {}),
        )

    returned = resp.get("ChecksumSHA256")
    if returned != _b64(digest):
        raise UploadVerificationError(
            f"s3://{bucket}/{key}: checksum mismatch "
            f"(local {digest}, s3 returned {returned!r})")

    head = c.head_object(Bucket=bucket, Key=key, VersionId=resp["VersionId"])
    if head["ContentLength"] != size:
        raise UploadVerificationError(
            f"s3://{bucket}/{key}: length {head['ContentLength']} != local {size}")

    return {"bucket": bucket, "key": key, "version_id": resp["VersionId"],
            "object_sha256": digest, "bytes": size}


def put_large(local_path, bucket, key, extra=None):
    """Multipart upload for Parquet. The composite checksum S3 returns is not
    comparable to a whole-object digest, so correctness here rests on the
    read-back verification in verify.py, not on the transport."""
    size = local_path.stat().st_size
    digest = sha256_file(local_path)
    c = client()
    c.upload_file(str(local_path), bucket, key, ExtraArgs=extra or {})
    head = c.head_object(Bucket=bucket, Key=key)
    if head["ContentLength"] != size:
        raise UploadVerificationError(
            f"s3://{bucket}/{key}: length {head['ContentLength']} != local {size}")
    return {"bucket": bucket, "key": key, "version_id": head.get("VersionId"),
            "object_sha256": digest, "bytes": size}


def put_bytes(data, bucket, key, content_type="application/json"):
    c = client()
    digest = hashlib.sha256(data).hexdigest()
    resp = c.put_object(Bucket=bucket, Key=key, Body=data,
                        ContentType=content_type, ChecksumAlgorithm="SHA256")
    if resp.get("ChecksumSHA256") != _b64(digest):
        raise UploadVerificationError(f"s3://{bucket}/{key}: checksum mismatch")
    return {"bucket": bucket, "key": key, "version_id": resp["VersionId"],
            "object_sha256": digest, "bytes": len(data)}


def open_stream(bucket, key, version_id=None):
    """File-like over an S3 object. Body is a stream -- the day is never staged
    locally, which is what keeps the converter's footprint independent of input
    size."""
    kw = {"Bucket": bucket, "Key": key}
    if version_id:
        kw["VersionId"] = version_id
    return client().get_object(**kw)["Body"]


def list_keys(bucket, prefix):
    """Sorted, so downstream input ordering is deterministic and reruns are
    byte-reproducible."""
    c = client()
    out = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        r = c.list_objects_v2(**kw)
        out.extend((o["Key"], o["Size"], o["LastModified"]) for o in r.get("Contents", []))
        if not r.get("IsTruncated"):
            return sorted(out)
        token = r["NextContinuationToken"]


def exists(bucket, key):
    try:
        client().head_object(Bucket=bucket, Key=key)
        return True
    except client().exceptions.ClientError:
        return False


def copy_object(bucket, src_key, dst_key):
    """Final promotion step. CopyObject is atomic; a reader sees the old object
    or the new one, never a torn write."""
    c = client()
    r = c.copy_object(Bucket=bucket, Key=dst_key,
                      CopySource={"Bucket": bucket, "Key": src_key})
    return r.get("VersionId")
