"""Storage health and leak detection, reported in risk-agent's input shape.

The checks that look paranoid are the ones that have already been true on this
account. Both buckets shipped with versioning enabled and no noncurrent
expiration, which means an object "expired" by lifecycle is not deleted -- it
becomes a noncurrent version and is billed indefinitely. Incomplete multipart
uploads are worse: billed, and invisible to ListObjects.

Usage:
    python -m storage.monitor            # human-readable
    python -m storage.monitor --json     # risk-agent payload
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
# `time` the module and `time` the datetime class collided here: this line used to import
# `time` from datetime and silently shadowed the module, so every `time.time()` below was
# `datetime.time.time` -- an AttributeError. It only ever fired on the branches that alarm
# (check_local_backlog's backlog path), so the flush alarm would have degraded to a generic
# "check could not be run" warn at exactly the moment it mattered. Aliased 2026-08-24.
from datetime import datetime, time as dtime, timedelta, timezone

import config
from storage import paths, s3util

WARN, CRIT = "warn", "critical"

THRESHOLDS = {
    "disk_free_gb": (5, 2),
    "recorder_rss_mb": (500, 1024),
    "unflushed_age_min": (30, 45),
    "noncurrent_pct": (10, 25),
    "scratch_age_h": (6, 24),
}

# A finished external sort deletes its own spill directory. One that is still there hours later
# is a sort that DIED -- and it died holding the disk, which is the resource capture needs most.
# Below this it is probably a sort in flight; the age threshold is what separates the two.
SCRATCH_LEAK_BYTES = 64 << 20         # 64 MB

# A percentage alone is meaningless on a nearly-empty bucket: 92% noncurrent of
# 200 KB is arithmetically alarming and materially nothing. Severity requires
# both a high ratio and enough absolute waste to be worth acting on, or the
# monitor trains its reader to ignore it.
NONCURRENT_WARN_BYTES = 10 << 20      # 10 MB
NONCURRENT_CRIT_BYTES = 100 << 20     # 100 MB


def _finding(sev, where, issue, evidence, fix=None, bytes_=None):
    return {"severity": sev, "where": where, "issue": issue,
            "evidence": evidence, "fix": fix, "bytes": bytes_}


# LEVEL-1 (P1) disk alert, Arjan 2026-09-29: at or above this share of `/` used, the monitor writes
# a P1 line to alerts.log immediately AND fails (exit 2 -> OnFailure -> kalshi-alert@). The BTC dry
# quoter alone can write ~2.5 GB per 5-minute run; a full disk stops capture, flush and the quoter.
# KALSHI_DISK_P1_PCT overrides it, for testing the alert path without filling the disk.
DISK_P1_PCT = float(os.environ.get("KALSHI_DISK_P1_PCT", "90"))
ALERTS_LOG = str(config.ROOT / "var" / "log" / "alerts.log")


def check_disk():
    out = []
    st = shutil.disk_usage("/")
    free_gb = st.free / 1e9
    used_pct = 100.0 * st.used / st.total
    w, c = THRESHOLDS["disk_free_gb"]
    if used_pct >= DISK_P1_PCT:
        msg = (f"{datetime.now(timezone.utc).isoformat()} P1 ALERT disk / at {used_pct:.1f}% used "
               f"(>= {DISK_P1_PCT:.0f}%), {free_gb:.1f} GB free -- flush and prune local data, and check "
               f"local logs and caches\n")
        with open(ALERTS_LOG, "a") as fh:
            fh.write(msg)
        out.append(_finding(CRIT, "local disk", f"P1: disk {used_pct:.1f}% used",
                            f"{free_gb:.1f} GB free on / (df)",
                            "flush + prune local data; then grow EBS if needed"))
    elif free_gb < c:
        out.append(_finding(CRIT, "local disk", "free space critical",
                            f"{free_gb:.1f} GB free on / (df)",
                            "grow the EBS volume; capture will stall"))
    elif free_gb < w:
        out.append(_finding(WARN, "local disk", "free space low",
                            f"{free_gb:.1f} GB free on /"))
    return out, {"disk_free_gb": round(free_gb, 1),
                 "disk_used_pct": round(100 * st.used / st.total, 1)}


def check_recorder():
    out, m = [], {}
    try:
        active = subprocess.run(["systemctl", "is-active", "kalshi-ws-recorder"],
                                capture_output=True, text=True).stdout.strip()
        m["recorder_active"] = active
        if active != "active":
            out.append(_finding(
                CRIT, "kalshi-ws-recorder", "recorder is not running",
                f"systemctl is-active -> {active}",
                "order book history has no REST backfill path; every second down "
                "is permanently unrecoverable"))
        props = subprocess.run(
            ["systemctl", "show", "kalshi-ws-recorder",
             "-p", "MemoryCurrent", "-p", "NRestarts"],
            capture_output=True, text=True).stdout
        vals = dict(l.split("=", 1) for l in props.strip().splitlines() if "=" in l)
        rss_mb = int(vals.get("MemoryCurrent", 0) or 0) / 1e6
        m["recorder_rss_mb"] = round(rss_mb, 1)
        m["recorder_restarts"] = int(vals.get("NRestarts", 0) or 0)
        w, c = THRESHOLDS["recorder_rss_mb"]
        if rss_mb > c:
            out.append(_finding(CRIT, "kalshi-ws-recorder", "recorder RSS critical",
                                f"MemoryCurrent={rss_mb:.0f} MB",
                                "a buffer is growing; frames are not draining"))
        elif rss_mb > w:
            out.append(_finding(WARN, "kalshi-ws-recorder", "recorder RSS elevated",
                                f"MemoryCurrent={rss_mb:.0f} MB"))
    except Exception as e:                           # noqa: BLE001
        out.append(_finding(WARN, "kalshi-ws-recorder", "could not read service state",
                            f"{type(e).__name__}: {e}"))
    return out, m


def check_local_backlog():
    """Segments closed but not yet in S3."""
    out, m = [], {}
    pending = []
    if config.WS_DIR.exists():
        pending = [p for p in config.WS_DIR.glob("dt=*/*/*.jsonl.gz")
                   if paths.segment_meta_path(p).exists()]
    m["unflushed_segments"] = len(pending)
    if pending:
        oldest = min(p.stat().st_mtime for p in pending)
        age_min = (time.time() - oldest) / 60
        m["unflushed_age_min"] = round(age_min, 1)
        m["unflushed_bytes"] = sum(p.stat().st_size for p in pending)
        w, c = THRESHOLDS["unflushed_age_min"]
        if age_min > c:
            out.append(_finding(
                CRIT, "local buffer", "segments are not reaching S3",
                f"{len(pending)} closed segments, oldest {age_min:.0f} min old",
                "check kalshi-ws-flush.timer and S3 credentials"))
        elif age_min > w:
            out.append(_finding(WARN, "local buffer", "flush is falling behind",
                                f"oldest closed segment {age_min:.0f} min old"))
    return out, m


def check_scratch():
    """Spill directories left behind by a sort that did not finish.

    `rest_backfill.SpillSorter` removes its own directory in a `finally`, so a leftover is not
    untidiness -- it is the fingerprint of a process that was killed rather than raised. A
    SIGTERM does not run `finally`, so a box shutdown mid-backfill leaks the whole spill.

    Real case this was written for: `kalshi-rest-eod` was SIGTERMed by a poweroff at
    2026-08-24 04:56 and left 490 MB in `var/scratch/trades/dt=2026-08-23`. It sat there for
    twelve hours while every 15-minute audit reported `findings: none`, because nothing here
    looked at scratch at all. On a 32 GB volume that is 1.5% of the disk per interrupted run.
    """
    out, m = [], {}
    root = config.SCRATCH_DIR
    m["scratch_bytes"] = 0
    m["scratch_stale_jobs"] = []
    if not root.exists():
        return out, m
    now = time.time()
    total = 0
    for job_dt in sorted(root.glob("*/dt=*")):
        if not job_dt.is_dir():
            continue
        files = [p for p in job_dt.rglob("*") if p.is_file()]
        if not files:
            continue
        nbytes = sum(p.stat().st_size for p in files)
        total += nbytes
        age_h = (now - max(p.stat().st_mtime for p in files)) / 3600
        w, c = THRESHOLDS["scratch_age_h"]
        if nbytes < SCRATCH_LEAK_BYTES or age_h < w:
            continue
        rel = f"{job_dt.parent.name}/{job_dt.name}"
        m["scratch_stale_jobs"].append(rel)
        sev = CRIT if age_h > c else WARN
        out.append(_finding(
            sev, "local scratch", "spill left behind by a sort that did not finish",
            f"{rel}: {nbytes/1e6:.0f} MB, untouched for {age_h:.0f} h",
            f"confirm the job's output reached S3, then rm -rf {root}/{rel}",
            bytes_=nbytes))
    m["scratch_bytes"] = total
    return out, m


def _bucket_bytes(bucket):
    """Current vs noncurrent bytes. The gap is the leak."""
    c = s3util.client()
    cur = non = 0
    cur_n = non_n = 0
    token = None
    while True:
        kw = {"Bucket": bucket, "MaxKeys": 1000}
        if token:
            kw["KeyMarker"] = token
        r = c.list_object_versions(**kw)
        for v in r.get("Versions", []):
            if v.get("IsLatest"):
                cur += v["Size"]; cur_n += 1
            else:
                non += v["Size"]; non_n += 1
        if not r.get("IsTruncated"):
            break
        token = r.get("NextKeyMarker")
    return cur, non, cur_n, non_n


def check_buckets():
    out, m = [], {}
    c = s3util.client()
    for bucket in (config.S3_BUCKET_WS, config.S3_BUCKET_HIST):
        short = bucket.replace("kalshi-bucket", "b")
        cur, non, cur_n, non_n = _bucket_bytes(bucket)
        m[f"{short}_current_bytes"] = cur
        m[f"{short}_noncurrent_bytes"] = non
        m[f"{short}_objects"] = cur_n
        pct = 100 * non / max(1, cur)
        w, cth = THRESHOLDS["noncurrent_pct"]
        if pct > cth and non > NONCURRENT_CRIT_BYTES:
            out.append(_finding(
                CRIT, bucket, "noncurrent versions are accumulating",
                f"{non/1e6:.1f} MB noncurrent vs {cur/1e6:.1f} MB current "
                f"({pct:.0f}%), {non_n} versions",
                "NoncurrentVersionExpiration is missing or too long", non))
        elif pct > w and non > NONCURRENT_WARN_BYTES:
            out.append(_finding(
                WARN, bucket, "noncurrent versions accumulating",
                f"{non/1e6:.1f} MB noncurrent ({pct:.0f}%), {non_n} versions",
                "expected to be reaped by NoncurrentVersionExpiration", non))

        # Incomplete multipart uploads: billed, and invisible to ListObjects.
        try:
            mpu = c.list_multipart_uploads(Bucket=bucket).get("Uploads", []) or []
        except Exception:                            # noqa: BLE001
            mpu = []
        m[f"{short}_incomplete_mpu"] = len(mpu)
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        stale = [u for u in mpu if u.get("Initiated") and u["Initiated"] < cutoff]
        if stale:
            out.append(_finding(
                WARN, bucket, "stale incomplete multipart uploads",
                f"{len(stale)} uploads older than 1 day",
                "AbortIncompleteMultipartUpload should reap these"))

        # Lifecycle completeness -- the config itself, not just its effects.
        try:
            rules = c.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
        except Exception:                            # noqa: BLE001
            rules = []
        versioning = c.get_bucket_versioning(Bucket=bucket).get("Status")
        if versioning == "Enabled":
            if not any("NoncurrentVersionExpiration" in r for r in rules):
                out.append(_finding(
                    CRIT, bucket, "versioned bucket has no NoncurrentVersionExpiration",
                    "get-bucket-versioning=Enabled and no matching lifecycle rule",
                    "expired objects become noncurrent versions and are billed forever"))
            if not any("AbortIncompleteMultipartUpload" in r for r in rules):
                out.append(_finding(
                    WARN, bucket, "no AbortIncompleteMultipartUpload rule",
                    "lifecycle rules do not abort failed uploads",
                    "orphaned parts are billed but absent from ListObjects"))
        if bucket == config.S3_BUCKET_HIST and not any(
                "Days" in r.get("Expiration", {}) for r in rules):
            out.append(_finding(CRIT, bucket, "no object expiration",
                                "no Expiration.Days rule",
                                "the 3-year retention is not enforced"))

    # Short-term bucket must not hold anything past its 7-day retention.
    old = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=8)
    for key, size, mod in s3util.list_keys(config.S3_BUCKET_WS,
                                           config.S3_PREFIX_WS + "/"):
        if mod < cutoff:
            old.append((key, size))
    m["b_objects_past_retention"] = len(old)
    if old:
        out.append(_finding(
            WARN, config.S3_BUCKET_WS, "objects older than the 7-day retention",
            f"{len(old)} objects, oldest {old[0][0]}",
            "the expireAfter7 lifecycle rule is not taking effect"))
    return out, m


def check_promotion(lag_days=3, lookback=7):
    """Days sealed but never promoted.

    The sentinel is `book_epochs`, not `orderbook_deltas`. promote_ws skips a
    zero-row table and writes no manifest for it, so a delta-less day looked
    unpromoted forever: dt=2026-08-19 promoted every table it had and still
    reported CRITICAL on every run for 18 hours. book_epochs is written whenever
    a seal exists -- the recorder cannot connect without opening an epoch -- so
    its manifest means "promotion ran" and nothing else.

    That is the whole job of this check now. Whether the day actually contains a
    book is a different question with a different blast radius, and it belongs to
    check_delta_capture() below. One predicate answering both could not tell
    "promotion is broken" from "the order book is gone", which are not remotely
    the same incident.
    """
    out, m = [], {}
    missing, bad_epochs, empty = [], {}, []
    today = config.today_et_date()
    for i in range(lag_days, lag_days + lookback):
        d = today - timedelta(days=i)
        dt = str(d)
        seal = _seal_doc(dt)
        if seal is None:
            continue
        # A SEALED DAY WITH NO FRAMES HAS NOTHING TO PROMOTE, and demanding a manifest for one is
        # how this check spent 2026-08-22 onward firing a CRITICAL every fifteen minutes. Capture
        # was gated off that day (`config.CAPTURE_SESSION_ENABLED`), so the recorder sealed
        # `line_count: 0, segment_count: 0` and `promote_ws` correctly wrote no manifest -- every
        # table logged "0 rows, skipped". The check could not tell "promotion failed" from "there
        # was nothing to promote".
        #
        # That is not a cosmetic bug. This alarm is the one that says the raw is about to age out
        # of the 7-day bucket unpromoted, and an alarm that has cried wolf for days is one nobody
        # reads on the day it is right.
        #
        # Recorded as a measurement rather than dropped: a run of empty sealed days is exactly what
        # a silently-dead recorder looks like, and that belongs in check_delta_capture, which owns
        # the question of whether frames are arriving at all.
        if not int(seal.get("line_count") or 0):
            empty.append(dt)
            continue
        mkey = paths.promotion_manifest_key("book_epochs", d)
        if not s3util.exists(config.S3_BUCKET_HIST, mkey):
            missing.append(dt)
            continue
        # Epochs that cannot be rebuilt (not contiguous, or opened without a
        # snapshot). Read from promote_ws's own manifest rather than recomputed
        # here: promote_ws holds the promoted book_epochs rows and is the only
        # thing that can derive this, because `opened_with_snapshot` exists in
        # no other artefact -- the seal's per-sid is_contiguous plus gaps is a
        # related but different quantity. Reading its number instead of writing
        # a second one is what keeps the monitor and promote_ws's own warning
        # from disagreeing.
        #
        # Measurement only, deliberately no finding: these are historical facts
        # about days already captured, and there is nothing to action. A day
        # that starts trending upward is the signal, and that is what a series
        # of numbers shows and a recurring WARN would only bury.
        #
        # None means "not recorded", NOT zero. Manifests written before the
        # field existed omit it, and 2026-08-19 really has 4 and 2026-08-20 has 1.
        try:
            acct = _get_json(config.S3_BUCKET_HIST, mkey).get("accounting") or {}
            bad_epochs[dt] = acct.get("unreconstructable_epochs")
        except Exception:                              # noqa: BLE001
            bad_epochs[dt] = None
    m["unpromoted_days"] = missing
    m["empty_sealed_days"] = empty
    m["unreconstructable_epochs_by_day"] = bad_epochs
    if missing:
        out.append(_finding(
            CRIT, "promotion", "sealed days were never promoted",
            f"days with a seal but no promotion manifest: {missing}",
            "the short-term bucket expires at 7 days; promote before then or the "
            "raw is gone"))
    return out, m


def _get_json(bucket, key):
    """Read a JSON object and CLOSE the stream.

    The close is not tidiness. An unclosed StreamingBody keeps its HTTP
    connection alive, and if anything in the same process has also imported
    pyarrow the two teardowns race at interpreter exit and abort the process:
    reproduced as SIGABRT with `terminate called without an active exception`,
    exit 134, and it goes away the moment the stream is closed. The monitor does
    not import pyarrow today, so here it is only a leaked connection every 15
    minutes -- but this is a 15-minute loop, and the callers that DO pair the two
    are one import away.
    """
    stream = s3util.open_stream(bucket, key)
    try:
        return json.loads(stream.read())
    finally:
        stream.close()


def _load_seal(dt):
    """The day's seal manifest, or None. Falls back to the archive copy.

    Preferred bucket first; see paths.seal_day_locations() for why the fallback
    has to exist at all.
    """
    for bucket, key in paths.seal_day_locations(dt):
        try:
            return _get_json(bucket, key)
        except Exception:                              # noqa: BLE001
            continue
    return None


def _sealed(dt):
    return any(s3util.exists(b, k) for b, k in paths.seal_day_locations(dt))


def _seal_doc(dt):
    """The seal manifest for a day, or None if the day is not sealed.

    Separate from `_sealed` because the promotion check needs the day's LINE COUNT, not merely the
    existence of a seal. An unreadable seal returns None and is therefore treated as "not sealed" --
    the conservative direction here, because the alternative is asserting something about a day we
    cannot read.
    """
    for b, k in paths.seal_day_locations(dt):
        if not s3util.exists(b, k):
            continue
        try:
            return _get_json(b, k)
        except Exception:                              # noqa: BLE001
            return None
    return None


def _et(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=config.ET)


def _capture_window(dt):
    """(first, last) ET datetimes actually spanned by a day's segments, or (None, None).

    Read from the `.meta.json` sidecars, never from S3 LastModified: any copy
    rewrites LastModified, and the us-east-1 -> us-east-2 sync already proved that
    once by collapsing a day's 59 segments into 5 timestamps. `first_recv_ns` and
    `last_recv_ns` are stamped at capture and survive being copied. Same reasoning
    as book_replay._ordered_keys().
    """
    for bucket, prefix in paths.raw_day_locations(dt):
        firsts, lasts = [], []
        for key, _size, _lm in s3util.list_keys(bucket, prefix):
            if not key.endswith(".meta.json"):
                continue
            try:
                doc = _get_json(bucket, key)
            except Exception:                          # noqa: BLE001
                continue                               # unreadable sidecar -> ignore
            if doc.get("first_recv_ns"):
                firsts.append(doc["first_recv_ns"])
            if doc.get("last_recv_ns"):
                lasts.append(doc["last_recv_ns"])
        if firsts:
            # the short-term copy answered; don't pay to list the archive as well
            return _et(min(firsts)), _et(max(lasts or firsts))
    return None, None


def _overlaps_session(first, last):
    """Did this capture window touch the gated trading session at all?"""
    day = first.date()
    open_ = datetime.combine(day, dtime(*config.CAPTURE_SESSION_OPEN), tzinfo=config.ET)
    close = datetime.combine(day, dtime(*config.CAPTURE_SESSION_CLOSE), tzinfo=config.ET)
    return first < close and last > open_


def check_delta_capture(lag_days=1, lookback=8):
    """Sealed days that captured a live session but recorded no order book.

    This is the most expensive failure the pipeline can have and, until now, the
    only thing that would have caught it was check_promotion's sentinel noticing
    it by accident. Trades, candles and metadata are all re-fetchable; the book is
    not. Kalshi exposes no historical orderbook endpoint -- /markets/{t}/orderbook
    is current-state only -- so a session recorded without deltas is gone, and no
    amount of later work brings it back.

    Zero deltas is not automatically an incident, which is exactly why this needs
    its own check rather than a bare non-zero assertion. Off the session the
    correct delta count IS zero: dt=2026-08-19 captured 21:52-23:27 ET, after the
    close, and its 3,888 snapshots with no deltas are a faithful record of a
    market that was not moving. The discriminator is whether the capture window
    overlapped 09:30-16:15 ET.

    Counts come from the seal manifest's per-sid `n_delta`, so the common case
    costs one GET. The sidecar listing that establishes the window is only paid
    for when the count is zero -- the rare branch.
    """
    out, m = [], {}
    counts = {}
    today = config.today_et_date()
    for i in range(lag_days, lag_days + lookback):
        dt = str(today - timedelta(days=i))
        seal = _load_seal(dt)
        if seal is None:
            continue        # no seal: not a captured day, and check_promotion's business
        n = sum(sid.get("n_delta") or 0 for sid in (seal.get("sids") or {}).values())
        counts[dt] = n
        if n:
            continue
        first, last = _capture_window(dt)
        if first is None or not _overlaps_session(first, last):
            continue        # off-hours or pre-flight: no book is the right answer
        out.append(_finding(
            CRIT, "delta capture",
            "captured a live session but recorded no order book",
            f"dt={dt} sealed with {seal.get('line_count')} lines spanning "
            f"{first:%H:%M}-{last:%H:%M} ET, which overlaps the "
            f"{config.CAPTURE_SESSION_OPEN[0]:02d}:{config.CAPTURE_SESSION_OPEN[1]:02d}"
            f"-{config.CAPTURE_SESSION_CLOSE[0]:02d}:"
            f"{config.CAPTURE_SESSION_CLOSE[1]:02d} ET session, but n_delta=0 "
            f"across every sid",
            "the book has no REST backfill path -- check the orderbook_delta "
            "subscription and sid_channel routing in ws_recorder before the next "
            "session, because this day is already unrecoverable"))
    # Reported even when nothing is wrong: a day that falls from 2.7M deltas to a
    # few thousand is the interesting signal, and it never trips a zero-check.
    m["delta_frames_by_day"] = counts
    return out, m


def check_rest_tape(lag_days=2, lookback=7):
    """Trading days with no REST trade tape.

    Scoped to days that also have a seal manifest: a seal means the recorder ran,
    which is the closest thing we have to "the exchange was open". Checking every
    calendar day would flag weekends and holidays forever.

    WARN, not CRIT, and deliberately: unlike promotion -- where the short-term raw
    expires at 7 days and a miss is permanent -- REST history stays fetchable, so a
    gap here costs a re-run rather than the data.
    """
    out, m = [], {}
    missing = []
    today = config.today_et_date()
    for i in range(lag_days, lag_days + lookback):
        d = today - timedelta(days=i)
        if not _sealed(str(d)):
            continue
        if not s3util.exists(config.S3_BUCKET_HIST,
                             paths.promotion_manifest_key("trades", d, src="rest")):
            missing.append(str(d))
    m["days_missing_rest_tape"] = missing
    if missing:
        out.append(_finding(
            WARN, "rest tape", "captured days have no REST trade tape",
            f"days with a seal but no src=rest trades manifest: {missing}",
            "systemctl start kalshi-rest-eod.service, or re-run "
            "storage.rest_backfill trades --start <d> --end <d+1>"))
    return out, m


def run():
    findings, measurements = [], {}
    for fn in (check_disk, check_recorder, check_local_backlog, check_scratch,
               check_buckets, check_promotion, check_delta_capture,
               check_rest_tape):
        try:
            f, m = fn()
        except Exception as e:                       # noqa: BLE001
            # Never report an unrun check as passing.
            f = [_finding(WARN, fn.__name__, "check could not be run",
                          f"{type(e).__name__}: {e}",
                          "evidence gap -- re-run before trusting this report")]
            m = {}
        findings.extend(f)
        measurements.update(m)

    crit = [f for f in findings if f["severity"] == CRIT]
    return {
        "stage": "audit",
        "status": "failed" if crit else ("degraded" if findings else "ok"),
        "measurements": measurements,
        "leaks": [{"where": f["where"], "bytes": f["bytes"], "cause": f["issue"],
                   "fix": f["fix"]} for f in findings if f["bytes"]],
        "issue_reports": [{"agent": "storage-agent",
                           "issue": f"[{f['severity']}] {f['where']}: {f['issue']}",
                           "evidence": f["evidence"]} for f in findings],
        "actions_taken": [],
        "next_check": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = run()
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(f"status: {rep['status'].upper()}")
        print("measurements:")
        for k, v in sorted(rep["measurements"].items()):
            print(f"  {k:<32} {v}")
        if rep["issue_reports"]:
            print("findings:")
            for i in rep["issue_reports"]:
                print(f"  {i['issue']}\n      evidence: {i['evidence']}")
        else:
            print("findings: none")
    return 2 if rep["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
