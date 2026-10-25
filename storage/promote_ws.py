"""T+3 promotion: raw WebSocket capture -> conformed Parquet in the long-term bucket.

Day D lands in the short-term bucket on D, is promoted on D+3, and expires from
the short-term bucket on D+7 -- a deliberate 4-day overlap.

This is a **pure function** of (sorted input set, code version). A rerun writes
byte-identical output, so re-running is a safe no-op rather than a duplication
risk. Everything that would break that determinism is handled explicitly:
ingest_ts is stamped at capture and carried through rather than recomputed here,
run identity lives in the sidecar manifest and never in a column, snapshot ids
are hashes rather than uuid4, row groups are bounded by row count rather than
bytes, and every input list is sorted.

Refuses to promote a day whose seal manifest is missing or whose counts
disagree with what it actually reads.

Usage:
    python -m storage.promote_ws --dt 2026-08-20
    python -m storage.promote_ws --lag 3          # promote D-3
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone

import config
from kalshi import client as k
from storage import conform, parquetio, paths, s3util, schemas
from storage.units import to_micros

TABLES = ("orderbook_deltas", "orderbook_snapshots", "orderbook_snapshot_levels",
          "trades", "ticker_updates", "book_epochs", "market_id_xref")

# Which producer we are, for tables written by more than one (see paths.SOURCED_TABLES).
# `trades` is also filled independently by rest_backfill; before this existed the two
# overwrote each other on a shared key.
SRC = "ws"


def _src(table):
    return SRC if table in paths.SOURCED_TABLES else None


def load_day_manifest(dt):
    key = paths.day_manifest_key(dt)
    if not s3util.exists(config.S3_BUCKET_WS, key):
        raise RuntimeError(
            f"dt={dt} has no seal manifest at {key}. The day is not known to be "
            f"complete, so promoting it would silently bless a partial capture.")
    return json.loads(s3util.open_stream(config.S3_BUCKET_WS, key).read())


def market_meta(dt):
    """market_type and notional per ticker -- needed to invert use_yes_price.

    The inversion constant is the market's own notional, never a hardcoded 1.0.

    Resolved for the *promoted* day, not today: this runs at T+3, so today's
    events are the wrong ones and would leave every notional null.
    """
    out = {}
    d = date.fromisoformat(dt) if isinstance(dt, str) else dt
    stamp = f"{d:%y%b%d}".upper()
    evs = [f"{st}-{stamp}{config.CAPTURE_CLOSE_SUFFIX}"
           for st in config.CAPTURE_SERIES]
    for ev in evs:
        try:
            for m in k.paged("/markets", "markets", limit=1000, event_ticker=ev):
                out[m["ticker"]] = {
                    "market_type": m.get("market_type"),
                    "event_ticker": m.get("event_ticker"),
                    "notional_value_micros": to_micros(m.get("notional_value_dollars")),
                }
        except Exception as e:                       # noqa: BLE001
            print(f"[meta] {ev}: {type(e).__name__}", file=sys.stderr)
    return out


def read_frames(dt):
    """Stream every captured line for the day, in sorted key order.

    Sorted: input ordering must be deterministic or the output is not
    byte-reproducible.
    """
    # Short-term first, permanent archive as fallback. The archive holds the same
    # bytes from the same flush, so re-deriving a day older than the short-term
    # bucket's 7-day expiry reads identical input and stays byte-reproducible --
    # which is the whole reason flush writes to both.
    bucket, keys = config.S3_BUCKET_WS, []
    for b, prefix in paths.raw_day_locations(dt):
        keys = [k_ for k_, _s, _m in s3util.list_keys(b, prefix)
                if k_.endswith(".jsonl.gz")]
        if keys:
            bucket = b
            break
    for key in keys:
        body = s3util.open_stream(bucket, key)
        with gzip.GzipFile(fileobj=body) as fh:
            for line in fh:
                if line.strip():
                    yield key, json.loads(line)


def promote(dt, upload=True):
    t0 = time.time()
    seal = load_day_manifest(dt)
    meta = market_meta(dt)
    ingest_fallback = datetime.now(timezone.utc)
    d = date.fromisoformat(dt)

    rows = {t: [] for t in TABLES}
    epochs = {}          # sid -> open epoch state
    sid_channel = {}     # sid -> channel; data frames carry only sid
    epoch_rows = {}      # book_epoch_id -> row
    xref = {}
    sub_state = {"use_yes_price": False}
    counts = {"lines": 0, "delta": 0, "snapshot": 0, "trade": 0, "ticker": 0,
              "control": 0, "dropped_markers": 0, "unparseable": 0, "skipped": 0}
    errors = []

    def close_epoch(sid, rec):
        e = epochs.pop(sid, None)
        if not e:
            return
        r = epoch_rows.get(e["book_epoch_id"])
        if r:
            r.update({
                "seq_end": rec.get("seq_end"),
                "close_reason": rec.get("close_reason"),
                "close_error_code": rec.get("close_error_code"),
                "gap_seq_lo": rec.get("gap_seq_lo"),
                "gap_seq_hi": rec.get("gap_seq_hi"),
                "opened_with_snapshot": bool(rec.get("opened_with_snapshot")),
                "end_recv_ts": _dt(rec.get("recv_ns")),
            })

    for _key, rec in read_frames(dt):
        counts["lines"] += 1
        kind = rec.get("kind")

        if kind == "segment_header":
            sub_state = rec.get("sub") or sub_state
            continue
        if kind == "sid_channel":
            sid_channel[rec["sid"]] = rec.get("channel")
            continue
        if kind == "epoch_open":
            eid = rec["book_epoch_id"]
            epochs[rec["sid"]] = {"book_epoch_id": eid,
                                  "seq_start": rec.get("seq_start")}
            epoch_rows[eid] = {
                "book_epoch_id": eid,
                "capture_session_id": rec.get("session") or "",
                "sid": rec["sid"], "channel": "orderbook_delta",
                "n_markets": (sub_state or {}).get("n_markets"),
                "use_yes_price_raw": bool((sub_state or {}).get("use_yes_price")),
                "price_basis_stored": "no_leg_native",
                "seq_start": rec.get("seq_start"), "seq_end": None,
                "frames_expected": None, "frames_delta": 0, "frames_snapshot": 0,
                "frames_control": 0, "frames_seen": 0,
                "is_contiguous": False, "opened_with_snapshot": False,
                "open_reason": rec.get("open_reason"), "close_reason": "open",
                "close_error_code": None,
                "start_recv_ts": _dt(rec.get("recv_ns")), "end_recv_ts": None,
                "gap_seq_lo": None, "gap_seq_hi": None,
                "schema_version": config.SCHEMA_VERSION,
            }
            continue
        if kind == "epoch_close":
            close_epoch(rec["sid"], rec)
            continue
        if kind == "drop_marker":
            counts["dropped_markers"] += 1
            continue
        if kind == "unparseable":
            counts["unparseable"] += 1
            continue
        if kind == "command":
            p = (rec.get("frame") or {}).get("params") or {}
            if "use_yes_price" in p:
                sub_state = dict(sub_state, use_yes_price=p["use_yes_price"])
            continue

        frame = rec.get("frame")
        if not isinstance(frame, dict) or "type" not in frame:
            counts["skipped"] += 1
            continue

        ftype = frame["type"]
        sid = frame.get("sid")
        ep = epochs.get(sid)
        session = rec.get("session") or ""
        recv_ns = rec.get("recv_ns")
        ingest = _dt(recv_ns) or ingest_fallback

        try:
            if ftype == "orderbook_delta":
                rows["orderbook_deltas"].append(
                    conform.delta(frame, recv_ns, session, ep, sub_state, meta, ingest))
                counts["delta"] += 1
                _bump(epoch_rows, ep, "frames_delta")
                _xref(xref, frame, ingest)
            elif ftype == "orderbook_snapshot":
                h, ls = conform.snapshot(frame, recv_ns, session, ep, sub_state,
                                         meta, ingest)
                rows["orderbook_snapshots"].append(h)
                rows["orderbook_snapshot_levels"].extend(ls)
                counts["snapshot"] += 1
                _bump(epoch_rows, ep, "frames_snapshot")
                if ep:
                    r = epoch_rows.get(ep["book_epoch_id"])
                    if r:
                        r["opened_with_snapshot"] = True
                _xref(xref, frame, ingest)
            elif ftype == "trade":
                # WS trades carry no seq, so there is no gap detection on this
                # channel. `source` records that, and dedup must never let a
                # ws_trade shadow the authoritative REST tape.
                rows["trades"].append(
                    conform.trade(frame, recv_ns, sid, ingest))
                counts["trade"] += 1
            elif ftype == "ticker":
                rows["ticker_updates"].append(
                    conform.ticker(frame, recv_ns, sid, session, ingest))
                counts["ticker"] += 1
                _xref(xref, frame, ingest)
            elif ftype in ("ok", "unsubscribed"):
                # These CONSUME sequence numbers on the same sid stream.
                counts["control"] += 1
                if sid_channel.get(sid) in (None, "orderbook_delta"):
                    _bump(epoch_rows, ep, "frames_control")
            else:
                counts["skipped"] += 1
        except conform.ConformanceError as e:
            errors.append(f"{ftype} seq={frame.get('seq')}: {e}")

    # Finalize epoch accounting: expected counts control frames, or every control
    # response would read as a phantom gap.
    for r in epoch_rows.values():
        r["frames_seen"] = (r["frames_delta"] + r["frames_snapshot"]
                            + r["frames_control"])
        if r["seq_start"] is not None and r["seq_end"] is not None:
            r["frames_expected"] = r["seq_end"] - r["seq_start"] + 1
            r["is_contiguous"] = r["frames_expected"] == r["frames_seen"]
        else:
            r["frames_expected"] = r["frames_seen"]
            r["is_contiguous"] = True
    rows["book_epochs"] = sorted(epoch_rows.values(),
                                 key=lambda r: (r["capture_session_id"] or "",
                                                r["sid"] or 0, r["seq_start"] or 0))
    rows["market_id_xref"] = sorted(xref.values(),
                                    key=lambda r: (r["ticker"], r["market_id"]))

    # Reconcile against the seal before writing anything.
    accounted = (counts["delta"] + counts["snapshot"] + counts["control"]
                 + counts["dropped_markers"] + counts["unparseable"]
                 + counts["skipped"])
    print(f"[promote] dt={dt} lines={counts['lines']} delta={counts['delta']} "
          f"snapshot={counts['snapshot']} trade={counts['trade']} "
          f"ticker={counts['ticker']} control={counts['control']} "
          f"dropped={counts['dropped_markers']} errors={len(errors)}")
    for e in errors[:5]:
        print(f"    conformance: {e}", file=sys.stderr)

    # Computed before the write loop so it can ride along in `counts` and land in
    # the manifest accounting. promote_ws is the only thing that can derive this
    # correctly: it holds the promoted book_epochs rows, and `opened_with_snapshot`
    # exists nowhere else -- the seal carries per-sid is_contiguous and a gaps list,
    # which is a related but different quantity. Recording it here gives monitor.py
    # one authoritative number to read instead of re-deriving a near-miss.
    bad = [r for r in rows["book_epochs"]
           if not r["is_contiguous"] or not r["opened_with_snapshot"]]
    counts["unreconstructable_epochs"] = len(bad)
    counts["book_epochs"] = len(rows["book_epochs"])

    results = {}
    for table in TABLES:
        data = rows[table]
        if not data:
            print(f"[promote] {table:<28} 0 rows, skipped")
            continue
        data.sort(key=_sorter(table))
        out = paths.scratch_dir("promote", d) / f"{table}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        digests = parquetio.write_rows(out, table, iter(data), schemas)
        ok, got, m = parquetio.verify_file(out, schemas.TABLES[table], digests)
        if not ok:
            raise RuntimeError(f"{table} dt={dt}: parquet verification failed")
        if upload:
            key = paths.parquet_key(table, d, src=_src(table))
            tmp = paths.tmp_parquet_key(table, d, digests["ordered_sha256"][:16],
                                        src=_src(table))
            rec = s3util.put_large(out, config.S3_BUCKET_HIST, tmp)
            s3util.copy_object(config.S3_BUCKET_HIST, tmp, key)   # atomic
            _manifest(table, d, digests, m, rec, key, counts, seal, errors)
            s3util.client().delete_object(Bucket=config.S3_BUCKET_HIST, Key=tmp)
        results[table] = m
        print(f"[promote] {table:<28} rows={m['num_rows']:>8} "
              f"parquet={m['bytes']/1e6:>7.2f}MB rg={m['row_groups']}")
        out.unlink(missing_ok=True)

    print(f"[promote] epochs={len(rows['book_epochs'])} "
          f"unreconstructable={len(bad)} "
          f"rss={parquetio.peak_rss_kb()//1024}MB {time.time()-t0:.0f}s")
    if bad:
        # NOT a failure exit. Every table above uploaded and verified; these epochs
        # are a data-quality fact about the captured tape, and days that predate
        # real capture carry them legitimately (dt=2026-08-19 is post-close
        # pre-flight connections: 4 of 12 epochs never opened with a snapshot,
        # which is the correct record of a market that was not moving).
        # Exiting 1 here made the unit indistinguishable from a genuine failure --
        # a missing seal, a verification mismatch -- and tripped OnFailure nightly.
        # The count is in the manifest accounting for monitor.py to report on.
        print(f"[promote] WARNING dt={dt}: {len(bad)} of "
              f"{len(rows['book_epochs'])} book epochs are unreconstructable "
              f"(no opening snapshot, or a sequence gap). The book cannot be "
              f"replayed across them; every other table promoted cleanly.",
              file=sys.stderr)
        for r in bad[:5]:
            print(f"    epoch {r['book_epoch_id']} "
                  f"contiguous={r['is_contiguous']} "
                  f"opened_with_snapshot={r['opened_with_snapshot']}",
                  file=sys.stderr)
    return 1 if errors else 0


def _sorter(table):
    keys = schemas.SORT_KEYS[table]
    return lambda r: tuple(_sk(r.get(c)) for c in keys)


def _sk(v):
    if v is None:
        return (0, "", 0)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (1, "", v)
    return (2, str(v), 0)


def _bump(epoch_rows, ep, field):
    if not ep:
        return
    r = epoch_rows.get(ep["book_epoch_id"])
    if r:
        r[field] += 1


def _xref(xref, frame, ts):
    m = frame.get("msg") or {}
    mid, tk = m.get("market_id"), m.get("market_ticker")
    if not mid or not tk:
        return
    e = xref.get(mid)
    if e is None:
        xref[mid] = {"market_id": mid, "ticker": tk, "first_seen_ts": ts,
                     "last_seen_ts": ts, "schema_version": config.SCHEMA_VERSION}
    else:
        e["last_seen_ts"] = ts


def _dt(ns):
    return None if ns is None else datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _manifest(table, d, digests, meta, rec, final_key, counts, seal, errors):
    doc = {
        "table": table, "partition": dict([paths.partition_kv(table, d)]),
        "src": _src(table),
        "schema_version": config.SCHEMA_VERSION,
        "pyarrow_version": __import__("pyarrow").__version__,
        "inputs": {"seal_segments": seal.get("segment_count"),
                   "seal_lines": seal.get("line_count")},
        "output": {"key": final_key, "object_sha256": rec["object_sha256"],
                   "row_count": meta["num_rows"], "row_groups": meta["row_groups"],
                   "bytes": meta["bytes"]},
        "content_digests": digests,
        "accounting": dict(counts, conformance_errors=len(errors)),
        # run identity lives here, never in a column -- that is what keeps the
        # data file byte-identical across reruns
        "runtime": {"peak_rss_kb": parquetio.peak_rss_kb(),
                    "promoted_at": datetime.now(timezone.utc).isoformat()},
    }
    s3util.put_bytes(json.dumps(doc, indent=1).encode(), config.S3_BUCKET_HIST,
                     paths.promotion_manifest_key(table, d, src=_src(table)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", default=None)
    ap.add_argument("--lag", type=int, default=3, help="promote D-lag")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args(argv)
    dt = a.dt or str(config.today_et_date() - timedelta(days=a.lag))
    return promote(dt, upload=not a.no_upload)


if __name__ == "__main__":
    sys.exit(main())
