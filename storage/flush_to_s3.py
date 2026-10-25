"""Local segments -> s3://<S3_BUCKET_WS>, and the 23:59 ET day seal.

The local delete is strictly conditional on S3 independently confirming it holds
the same bytes. Raw order book capture has no REST backfill path, so a delete
that races an upload failure is unrecoverable data loss, not an inconvenience.

A segment is eligible for flush only once its `.meta.json` sidecar exists --
that file is what the recorder writes on close, so its presence is the
"this file is complete" marker. The open segment is skipped by construction.

Usage:
    python -m storage.flush_to_s3            # flush closed segments
    python -m storage.flush_to_s3 seal       # + write the day manifest
"""
import argparse
import json
import sys
import time
from datetime import date, timedelta

import config
from storage import paths, s3util


def closed_segments(dt=None):
    """(gz_path, meta_path) for every complete, not-yet-flushed segment."""
    root = config.WS_DIR
    if not root.exists():
        return []
    out = []
    pattern = f"dt={dt}/*/*.jsonl.gz" if dt else "dt=*/*/*.jsonl.gz"
    for gz in sorted(root.glob(pattern)):
        meta = paths.segment_meta_path(gz)
        if meta.exists():
            out.append((gz, meta))
    return out


def _dt_and_channel(gz_path):
    # .../dt=YYYY-MM-DD/channel=<ch>/seg-*.jsonl.gz
    return (gz_path.parent.parent.name.split("=", 1)[1],
            gz_path.parent.name.split("=", 1)[1])


def flush_one(gz, meta, bucket=None, archive=True):
    """Upload, verify, then delete. Returns the upload record."""
    bucket = bucket or config.S3_BUCKET_WS
    dt, channel = _dt_and_channel(gz)
    meta_doc = json.loads(meta.read_text())

    key = paths.ws_key(dt, channel, gz.name)
    rec = s3util.put_verified(gz, bucket, key)

    meta_doc["s3"] = rec
    meta_key = paths.ws_key(dt, channel, meta.name)
    s3util.put_bytes(json.dumps(meta_doc, indent=1).encode(), bucket, meta_key)

    if archive:
        # Copy the raw to the historical bucket as well. The short-term bucket
        # expires at 7 days and promotion runs at T+3, so without this the window
        # to fix a conformance bug and re-derive is only four days.
        s3util.put_verified(gz, config.S3_BUCKET_HIST,
                            paths.raw_archive_key(dt, channel, gz.name))
        # The sidecar goes with it. Without it the archive is unusable rather than
        # merely inconvenient: `first_recv_ns` is the only copy-proof ordering key
        # (S3 LastModified is rewritten by any copy, which is what silently broke
        # book replay for every migrated day), and `plaintext_sha256` is the only
        # way to prove the archived bytes are intact.
        s3util.put_bytes(json.dumps(meta_doc, indent=1).encode(), config.S3_BUCKET_HIST,
                         paths.raw_archive_key(dt, channel, meta.name))

    # Only now. Verification passed, so the bytes exist in S3.
    gz.unlink()
    meta.unlink()
    return rec


def flush(dt=None, archive=True):
    segs = closed_segments(dt)
    if not segs:
        print("nothing to flush")
        return 0
    ok = failed = 0
    for gz, meta in segs:
        try:
            rec = flush_one(gz, meta, archive=archive)
            ok += 1
            print(f"[up] {gz.name} -> s3://{rec['bucket']}/{rec['key']} "
                  f"({rec['bytes']}B sha={rec['object_sha256'][:12]})")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            # Left in place deliberately: the next tick retries it.
            print(f"[!!] {gz.name} kept locally: {type(e).__name__}: {e}",
                  file=sys.stderr)
    print(f"flushed {ok}, failed {failed}")
    return 1 if failed else 0


def seal(dt=None):
    """Aggregate the day into one manifest.

    Everything downstream trusts this document: promotion refuses to run against
    a day with no manifest, or one whose counts disagree with what it reads.
    Aggregated from S3 rather than local state so it survives a box restart.
    """
    dt = dt or str(config.today_et_date())
    bucket = config.S3_BUCKET_WS
    keys = s3util.list_keys(bucket, paths.ws_prefix(dt))

    metas = []
    for key, _size, _mod in keys:
        if not key.endswith(".meta.json"):
            continue
        metas.append(json.loads(s3util.open_stream(bucket, key).read()))
    metas.sort(key=lambda m: m["segment_id"])

    sids, gaps = {}, []
    for m in metas:
        for sid, s in (m.get("sids") or {}).items():
            # sid keys are already namespaced by capture session upstream; older
            # segments may carry a bare sid, so namespace those here too.
            if ":" not in sid:
                sid = f"{m.get('capture_session_id', '?')}:{sid}"
            agg = sids.setdefault(sid, {"first_seq": None, "last_seq": None,
                                        "n_delta": 0, "n_snapshot": 0,
                                        "n_control": 0})
            for f in ("n_delta", "n_snapshot", "n_control"):
                agg[f] += s.get(f, 0)
            if s.get("first_seq") is not None:
                agg["first_seq"] = (s["first_seq"] if agg["first_seq"] is None
                                    else min(agg["first_seq"], s["first_seq"]))
            if s.get("last_seq") is not None:
                agg["last_seq"] = (s["last_seq"] if agg["last_seq"] is None
                                   else max(agg["last_seq"], s["last_seq"]))

    for sid, s in sids.items():
        if s["first_seq"] is None:
            continue
        expected = s["last_seq"] - s["first_seq"] + 1
        seen = s["n_delta"] + s["n_snapshot"] + s["n_control"]
        s["frames_expected"] = expected
        s["frames_seen"] = seen
        s["is_contiguous"] = expected == seen
        if not s["is_contiguous"]:
            gaps.append({"sid": sid, "expected": expected, "seen": seen,
                         "missing": expected - seen})

    doc = {
        "dt": dt,
        "sealed_at": time.time(),
        "segment_count": len(metas),
        "line_count": sum(m["line_count"] for m in metas),
        "uncompressed_bytes": sum(m["uncompressed_bytes"] for m in metas),
        "compressed_bytes": sum(m["compressed_bytes"] for m in metas),
        "dropped_frames": sum(m.get("dropped_frames", 0) for m in metas),
        "sids": sids,
        "gaps": gaps,
        "epochs_opened": [e for m in metas for e in m.get("epochs_opened", [])],
        "epochs_closed": [e for m in metas for e in m.get("epochs_closed", [])],
        "segments": [{"segment_id": m["segment_id"],
                      "line_count": m["line_count"],
                      "plaintext_sha256": m["plaintext_sha256"],
                      "s3_key": (m.get("s3") or {}).get("key"),
                      "s3_version_id": (m.get("s3") or {}).get("version_id")}
                     for m in metas],
    }
    doc["compression_ratio"] = round(
        doc["uncompressed_bytes"] / max(1, doc["compressed_bytes"]), 2)

    body = json.dumps(doc, indent=1).encode()
    # Written to BOTH buckets, same key, for the same reason flush_one archives the
    # segments beside it: the short-term bucket's `expireAfter7` rule carries
    # `Filter: {}`, so it reaps ws/_manifest/ exactly like the ws/ tape it describes.
    # The seal is the only record that a day was captured and complete, and every
    # seal-scoped check in monitor.py treats a failed GET as "not a captured day" --
    # so without the archived copy the audit trail does not break loudly at day 7,
    # it reports a clean bill of health for days it can no longer see.
    # paths.seal_day_locations() is the reader half; keep the key identical in both
    # so the fallback stays a pure bucket swap with no key rewriting.
    for seal_bucket, seal_key in paths.seal_day_locations(dt):
        s3util.put_bytes(body, seal_bucket, seal_key)
    print(f"[seal] dt={dt} segments={doc['segment_count']} "
          f"lines={doc['line_count']} gz={doc['compressed_bytes']}B "
          f"ratio={doc['compression_ratio']}x dropped={doc['dropped_frames']} "
          f"gaps={len(gaps)}")
    return 1 if gaps or doc["dropped_frames"] else 0


def catchup(days=7):
    """Seal every one of the last `days` completed ET days that has no seal manifest.

    The 23:59 ET seal timer only fires if the box is up. This box is stopped for days at a time,
    and an unsealed day makes promote_ws refuse it ("has no seal manifest") -- every run, with no
    way to recover. Capture is gated off on most of those days, so the catch-up seal is usually
    an empty-day manifest; it still records the day truthfully (segment_count from S3). Run at
    boot by kalshi-ws-seal-catchup.service, after network and credentials are up.
    """
    rc = 0
    today = config.today_et_date()
    for i in range(1, days + 1):
        d = str(today - timedelta(days=i))
        if any(s3util.exists(b, k) for b, k in paths.seal_day_locations(d)):
            continue
        print(f"[catchup] dt={d} unsealed -> sealing")
        rc |= flush(d, archive=True)
        rc |= seal(d)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="flush", choices=("flush", "seal", "catchup"))
    ap.add_argument("--days", type=int, default=7, help="catchup: how many past ET days to check")
    ap.add_argument("--dt", default=None, help="ET date; defaults to today ET")
    ap.add_argument("--no-archive", action="store_true",
                    help="skip the raw copy to the historical bucket")
    a = ap.parse_args(argv)

    if a.action == "catchup":
        return catchup(a.days)
    rc = flush(a.dt, archive=not a.no_archive)
    if a.action == "seal":
        # Flush first: the seal must see the final segment.
        rc |= seal(a.dt)
    return rc


if __name__ == "__main__":
    sys.exit(main())
