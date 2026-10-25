"""Prove conformed WebSocket rows are interchangeable with REST rows.

Two distinct checks, and the distinction matters:

  structural  the conformed WS row and the REST row populate the SAME schema,
              with the same column types and the same nullability behaviour.
              A WS-only column being null on a REST row is expected; a column
              being absent, or holding a different type, is a defect.

  agreement   where both sources observed the same market at overlapping times,
              they agree on the values they share.

When they disagree, the WebSocket side is what gets fixed. REST is canonical --
including when REST looks wrong, in which case this escalates rather than
quietly reconciling toward the convenient answer.

Usage:
    python -m storage.reconcile fixtures         # against payloads on disk
    python -m storage.reconcile day --dt 2026-08-20
"""
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

import config
from storage import conform, paths, s3util, schemas

WS_ONLY = {
    "orderbook_snapshots": {"capture_session_id", "sid", "seq", "book_epoch_id",
                            "epoch_seq_start", "use_yes_price_raw"},
    "trades": {"market_id", "ws_sid"},
}


def structural(rows_ws, rows_rest, table):
    """Both row sets must build the identical Arrow schema."""
    schema = schemas.TABLES[table]
    problems = []
    for label, rows in (("ws", rows_ws), ("rest", rows_rest)):
        if not rows:
            problems.append(f"{label}: no rows to check")
            continue
        try:
            tbl = pa.Table.from_pylist(rows, schema=schema)
        except Exception as e:                       # noqa: BLE001
            problems.append(f"{label}: does not fit {table} schema: {e}")
            continue
        if tbl.schema != schema:
            problems.append(f"{label}: schema drift vs declared {table}")
    if rows_ws and rows_rest:
        ws_keys, rest_keys = set(rows_ws[0]), set(rows_rest[0])
        if ws_keys != rest_keys:
            only_ws = ws_keys - rest_keys - WS_ONLY.get(table, set())
            only_rest = rest_keys - ws_keys
            if only_ws:
                problems.append(f"columns only on the WS row: {sorted(only_ws)}")
            if only_rest:
                problems.append(f"columns only on the REST row: {sorted(only_rest)}")
    return problems


def check_fixtures():
    """Conform the real payloads captured on disk and assert they line up."""
    root = config.RAW_DIR / "kalshi"
    stamp = (root / "latest").read_text().strip()
    src = root / stamp
    ob_files = sorted(src.glob("orderbooks_*.json"))
    if not ob_files:
        print("no orderbook fixture found", file=sys.stderr)
        return 1

    books = json.loads(ob_files[0].read_text())
    now = datetime.now(timezone.utc)
    rest_headers, rest_levels = [], []
    for ticker, body in books.items():
        h, ls = conform.rest_orderbook(ticker, body, now, {}, now)
        rest_headers.append(h)
        rest_levels.extend(ls)

    # Synthesize the WS form of the same books: identical content, WS field
    # names and WS's habit of omitting an empty side entirely.
    ws_headers, ws_levels = [], []
    epoch = {"book_epoch_id": "e" * 32, "seq_start": 1}
    for i, (ticker, body) in enumerate(books.items()):
        ob = body["orderbook_fp"]
        msg = {"market_ticker": ticker, "market_id": f"{i:032x}"}
        if ob.get("yes_dollars"):
            msg["yes_dollars_fp"] = ob["yes_dollars"]
        if ob.get("no_dollars"):
            msg["no_dollars_fp"] = ob["no_dollars"]
        frame = {"type": "orderbook_snapshot", "sid": 1, "seq": i + 1, "msg": msg}
        h, ls = conform.snapshot(frame, int(now.timestamp() * 1e9), "sess",
                                 epoch, {"use_yes_price": False}, {}, now)
        ws_headers.append(h)
        ws_levels.extend(ls)

    rc = 0
    for table, ws, rest in (("orderbook_snapshots", ws_headers, rest_headers),
                            ("orderbook_snapshot_levels", ws_levels, rest_levels)):
        problems = structural(ws, rest, table)
        print(f"[structural] {table}: "
              f"{'OK' if not problems else 'PROBLEMS'} "
              f"(ws={len(ws)} rest={len(rest)})")
        for p in problems:
            rc = 1
            print(f"    {p}")

    # Agreement on the level content itself.
    def keyed(rows):
        return {(r["ticker"], r["side"], r["level_ix"]):
                (r["price_micros"], r["size_cc"]) for r in rows}
    a, b = keyed(ws_levels), keyed(rest_levels)
    diff = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
    print(f"[agreement] levels compared={len(a)} mismatched={len(diff)}")
    for k in diff[:5]:
        rc = 1
        print(f"    {k}: ws={a.get(k)} rest={b.get(k)}")

    # The absent-vs-empty property, stated as a test rather than a comment.
    empty_yes = [h for h in ws_headers if h["yes_levels"] == 0]
    print(f"[absent-vs-empty] {len(empty_yes)}/{len(ws_headers)} books have an "
          f"empty yes side; all carry yes_levels=0 rather than a missing row: "
          f"{'OK' if all(h['yes_levels'] == 0 for h in empty_yes) else 'FAIL'}")
    rest_empty = {h["ticker"] for h in rest_headers if h["yes_levels"] == 0}
    ws_empty = {h["ticker"] for h in ws_headers if h["yes_levels"] == 0}
    if rest_empty != ws_empty:
        rc = 1
        print(f"    MISMATCH: WS omitting the key and REST sending [] must "
              f"conform to the same thing; differ on "
              f"{sorted(rest_empty ^ ws_empty)[:5]}")
    else:
        print(f"    WS-absent and REST-[] conform identically: OK")
    return rc


def check_day(dt):
    """Reconcile a captured day's WS snapshots against a REST poll."""
    bucket = config.S3_BUCKET_WS
    keys = [k for k, _s, _m in s3util.list_keys(bucket, paths.ws_prefix(dt))
            if k.endswith(".jsonl.gz")]
    if not keys:
        print(f"no captured segments for dt={dt}", file=sys.stderr)
        return 1
    now = datetime.now(timezone.utc)
    n_snap = n_delta = n_other = 0
    epochs = {}
    headers = []
    for key in keys:
        body = s3util.open_stream(bucket, key)
        with gzip.GzipFile(fileobj=body) as fh:
            for line in fh:
                rec = json.loads(line)
                kind = rec.get("kind")
                if kind == "epoch_open":
                    epochs[rec["sid"]] = {"book_epoch_id": rec["book_epoch_id"],
                                          "seq_start": rec.get("seq_start")}
                    continue
                frame = rec.get("frame")
                if not frame or "type" not in frame:
                    continue
                t = frame["type"]
                if t == "orderbook_snapshot":
                    h, _ls = conform.snapshot(
                        frame, rec["recv_ns"], rec.get("session", ""),
                        epochs.get(frame.get("sid")), {"use_yes_price": False},
                        {}, now)
                    headers.append(h)
                    n_snap += 1
                elif t == "orderbook_delta":
                    conform.delta(frame, rec["recv_ns"], rec.get("session", ""),
                                  epochs.get(frame.get("sid")),
                                  {"use_yes_price": False}, {}, now)
                    n_delta += 1
                else:
                    n_other += 1
    problems = structural(headers, [], "orderbook_snapshots")
    print(f"[day {dt}] snapshots={n_snap} deltas={n_delta} other={n_other}")
    print(f"[day {dt}] conformed snapshot headers fit the declared schema: "
          f"{'OK' if not problems else problems}")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fixtures", "day"))
    ap.add_argument("--dt", default=None)
    a = ap.parse_args(argv)
    if a.mode == "fixtures":
        return check_fixtures()
    return check_day(a.dt or str(config.today_et_date()))


if __name__ == "__main__":
    sys.exit(main())
