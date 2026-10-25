"""Bucket lifecycle configuration, applied as code rather than console clicks.

Both buckets have versioning enabled. A lifecycle rule that only sets Expiration
therefore does not delete anything -- it writes a delete marker and leaves the
bytes as a noncurrent version, billed forever. Every rule below that looks
redundant is closing one of those holes:

  NoncurrentVersionExpiration    the actual delete under versioning
  ExpiredObjectDeleteMarker      reaps the markers left behind
  AbortIncompleteMultipartUpload orphaned parts from a failed upload are billed
                                 but invisible to ListObjects

Usage:  python -m storage.setup_buckets [--apply]
"""
import argparse
import json
import sys

import config
from storage import s3util

# Short-term raw capture. 7-day expiry is the requested retention; the day is
# promoted to the historical bucket at T+3, giving a 4-day overlap.
WS_RULES = [
    {
        "ID": "expireAfter7",
        "Filter": {},
        "Status": "Enabled",
        "Expiration": {"Days": 7},
    },
    {
        "ID": "noncurrentAndMultipart",
        "Filter": {},
        "Status": "Enabled",
        # 1 day: the current version is the only one anything reads. A noncurrent
        # version here is either an overwrite (shouldn't happen; keys are unique
        # per segment) or the delete-marker shadow of the 7-day expiry.
        "NoncurrentVersionExpiration": {"NoncurrentDays": 1},
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
    },
    {
        "ID": "reapDeleteMarkers",
        "Filter": {},
        "Status": "Enabled",
        # Cannot share a rule with Expiration.Days -- S3 rejects the combination.
        "Expiration": {"ExpiredObjectDeleteMarker": True},
    },
]

# Long-term Parquet. Glacier Instant Retrieval stays directly queryable by
# Athena with no restore step, so the 365-day transition does not break tables.
def _hist_rules():
    rules = []
    for name, prefix in (("parquet", config.S3_PREFIX_PARQUET + "/"),
                         ("raw", config.S3_PREFIX_RAW_ARCHIVE + "/")):
        rules.append({
            "ID": f"glacierAfter365-{name}",
            "Filter": {"Prefix": prefix},
            "Status": "Enabled",
            "Transitions": [{"Days": 365, "StorageClass": "GLACIER_IR"}],
            "Expiration": {"Days": 1095},
        })
    rules.append({
        # Manifests are read to validate reruns and to decide whether a partition
        # is already current. Transitioning them to Glacier would mean paying
        # retrieval to answer "is the pipeline caught up?", so they stay standard.
        "ID": "manifestsExpireNoTransition",
        "Filter": {"Prefix": config.S3_PREFIX_MANIFEST + "/"},
        "Status": "Enabled",
        "Expiration": {"Days": 1095},
    })
    rules.append({
        # Athena spills query results here. They are derived, reproducible, and
        # would otherwise accumulate forever inside the archive bucket.
        "ID": "athenaResultsExpire",
        "Filter": {"Prefix": "athena-results/"},
        "Status": "Enabled",
        "Expiration": {"Days": 7},
    })
    rules.append({
        "ID": "noncurrentAndMultipart",
        "Filter": {},
        "Status": "Enabled",
        # 7 days here rather than 1: promotion overwrites a partition in place on
        # rerun, and a week is long enough to roll back to the prior version if a
        # rerun turns out to have been wrong.
        "NoncurrentVersionExpiration": {"NoncurrentDays": 7},
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
    })
    rules.append({
        "ID": "reapDeleteMarkers",
        "Filter": {},
        "Status": "Enabled",
        "Expiration": {"ExpiredObjectDeleteMarker": True},
    })
    return rules


HIST_RULES = _hist_rules()


def current(bucket):
    c = s3util.client()
    try:
        return c.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
    except c.exceptions.ClientError as e:
        if "NoSuchLifecycleConfiguration" in str(e):
            return []
        raise


def audit(bucket, rules):
    """What the current config fails to cover. Returns a list of findings."""
    have = current(bucket)
    findings = []
    ids = {r["ID"] for r in have}

    if not any("NoncurrentVersionExpiration" in r for r in have):
        findings.append(
            "no NoncurrentVersionExpiration: with versioning enabled, expired "
            "objects persist as noncurrent versions and are billed indefinitely")
    if not any("AbortIncompleteMultipartUpload" in r for r in have):
        findings.append(
            "no AbortIncompleteMultipartUpload: failed uploads leave orphaned "
            "parts that are billed but do not appear in ListObjects")
    if not any(r.get("Expiration", {}).get("ExpiredObjectDeleteMarker") for r in have):
        findings.append("no ExpiredObjectDeleteMarker cleanup: delete markers accumulate")
    if bucket == config.S3_BUCKET_HIST:
        if not any("Days" in r.get("Expiration", {}) for r in have):
            findings.append("no object expiration: the 3-year retention is not enforced")
    return findings


def apply(bucket, rules):
    s3util.client().put_bucket_lifecycle_configuration(
        Bucket=bucket, LifecycleConfiguration={"Rules": rules})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the configuration; default is audit only")
    a = ap.parse_args(argv)

    rc = 0
    for bucket, rules in ((config.S3_BUCKET_WS, WS_RULES),
                          (config.S3_BUCKET_HIST, HIST_RULES)):
        print(f"\n=== {bucket} ===")
        print(f"versioning: "
              f"{s3util.client().get_bucket_versioning(Bucket=bucket).get('Status', 'Disabled')}")
        findings = audit(bucket, rules)
        if findings:
            rc = 1
            for f in findings:
                print(f"  LEAK  {f}")
        else:
            print("  no leaks found")

        if a.apply:
            apply(bucket, rules)
            print(f"  applied {len(rules)} rules")
            after = audit(bucket, rules)
            print("  post-apply audit:", "clean" if not after else after)
            rc = 0 if not after else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
