"""Scoped Kalshi REST backfill: the index 4pm-close board, one ET session per partition.

This exists alongside `rest_backfill` rather than inside it because the two jobs
have opposite shapes. The exchange-wide sweep is time-ordered and filtered
locally -- ~1.29M requests for a year -- because 34% of the tape is in scope. For
the ~180 markets/day of the index close board that trade is inverted: discovery
is arithmetic (the event ticker is a pure function of series and date) and trades
come back in exactly one unpaginated request per market. Tier A is ~30k requests.

Three things worth knowing before changing anything here:

  * **Output goes to `*_index_close` tables.** `already_done()` keys resume on
    (table, dt), so writing `trades` here would make the exchange-wide sweep skip
    every day this touched, leaving partitions that hold six series behind a
    manifest claiming they are complete.

  * **`dt` is the ET date of the *event*, not of `created_time`.** These events
    open 22:30 ET the previous evening, so a `created_time` partition would split
    a session across two files. The study wants the day's board together, and
    these are separate tables, so the session date is the useful key. This is a
    deliberate divergence from `trades`; do not "fix" it to match.

  * The historical/live split is the same `/historical/cutoff` boundary the
    exchange-wide sweep uses. Settled markets are reachable *only* via
    /historical/markets -- /markets?event_ticker= returns zero rows for them.

Usage:
    python -m storage.scoped_backfill --tier A --start 2025-08-20 --end 2026-08-20
    python -m storage.scoped_backfill --series KXINX --start 2026-05-28 --end 2026-05-29 --no-upload
"""
import argparse
import sys
import time
from datetime import date, timedelta

import config
from kalshi import client as k
from storage import paths, parquetio, s3util, schemas
from storage.rest_backfill import (
    _now, already_done, cutoff, market_rows, raise_rate_limit, trade_row,
    _write_manifest,
)

# Tier A is the range-bracket board: KXINX/KXNASDAQ100 are 28 `between` + 1
# `greater` + 1 `less`. The U series are all `greater_or_equal`, which the pricer
# now handles (theo.STRIKE_TYPE_RULE), so B and C are priceable -- they are held
# separate only because they are ~5x the markets for a ladder of the same board.
TIERS = {
    "A": ("KXINX", "KXNASDAQ100", "KXINXDUD", "KXNASDAQDUD"),
    "B": ("KXINX", "KXNASDAQ100", "KXINXDUD", "KXNASDAQDUD", "KXINXU"),
    "C": ("KXINX", "KXNASDAQ100", "KXINXDUD", "KXNASDAQDUD", "KXINXU",
          "KXNASDAQ100U"),
}

CLOSE_SUFFIX = "H1600"


def event_ticker(series, d, suffix=CLOSE_SUFFIX):
    """KXINX-25SEP15H1600. Pure arithmetic -- no API call to discover an event."""
    return f"{series}-{d:%y%b%d}".upper() + suffix


def _is_missing(exc):
    """A 404 here means the event was never listed (holiday, pre-listing)."""
    return getattr(exc, "code", None) in (404,)


def fetch_event_markets(series, d, historical, suffix=CLOSE_SUFFIX):
    """Every market on one event. Verified to return in a single page, no cursor."""
    ev = event_ticker(series, d, suffix)
    path = "/historical/markets" if historical else "/markets"
    try:
        return ev, k.paged(path, "markets", limit=1000, event_ticker=ev)
    except Exception as e:                           # noqa: BLE001
        if _is_missing(e):
            return ev, []
        print(f"[scoped] {ev}: {type(e).__name__} {str(e)[:100]}", file=sys.stderr)
        return ev, []


def fetch_market_trades(ticker, historical):
    """All trades for one market. Measured: 1 request, never paginated."""
    path = "/historical/trades" if historical else "/markets/trades"
    try:
        return k.paged(path, "trades", limit=1000, ticker=ticker)
    except Exception as e:                           # noqa: BLE001
        if _is_missing(e):
            return []
        print(f"[scoped] trades {ticker}: {type(e).__name__}", file=sys.stderr)
        return []


# --- driver --------------------------------------------------------------

def _emit_table(table, d, rows, upload, ingest, historical):
    """Write one partition. Returns (row_count, bytes) or (0, 0) if empty."""
    rows = list(rows)
    if not rows:
        return 0, 0
    scratch = paths.scratch_dir(f"scoped_{table}", d)
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "part-00000.snappy.parquet"
    key_cols = schemas.SORT_KEYS[table]
    rows.sort(key=lambda r: tuple(str(r.get(c) or "") for c in key_cols))
    digests = parquetio.write_rows(out, table, iter(rows), schemas)
    ok, got, meta = parquetio.verify_file(out, schemas.TABLES[table], digests)
    if not ok:
        raise RuntimeError(f"{table} dt={d}: parquet verification failed")
    if upload:
        rec = s3util.put_large(out, config.S3_BUCKET_HIST, paths.parquet_key(table, d))
        _write_manifest(table, d, digests, meta, rec, historical, ingest)
    out.unlink(missing_ok=True)
    return meta["num_rows"], meta["bytes"]


def backfill_day(d, series_list, cut, upload=True, resume=True):
    ingest = _now()
    tables = ["trades" + schemas.SCOPED_SUFFIX, "markets" + schemas.SCOPED_SUFFIX]
    if resume and upload and all(already_done(t, d) for t in tables):
        return {"skipped": True}

    lo, hi = config.et_day_bounds(d)
    historical = hi <= cut["market_settled_ts"]
    t0 = time.time()

    mkt_rows, trd_rows = [], []
    n_markets = n_trades = 0

    for st in series_list:
        ev, raw_markets = fetch_event_markets(st, d, historical)
        if not raw_markets:
            continue
        n_markets += len(raw_markets)
        mkt_rows.extend(market_rows(
            raw_markets, {}, ingest,
            "rest_historical_markets" if historical else "rest_markets"))
        for m in raw_markets:
            for t in fetch_market_trades(m["ticker"], historical):
                trd_rows.append(trade_row(
                    t, None, ingest,
                    "rest_historical_trades" if historical else "rest_trades"))
                n_trades += 1

    if n_markets == 0:
        print(f"[scoped] dt={d} no events listed (holiday or pre-listing)")
        return {"markets": 0, "trades": 0}

    rows_t, _ = _emit_table("trades" + schemas.SCOPED_SUFFIX, d, trd_rows,
                            upload, ingest, historical)
    rows_m, _ = _emit_table("markets" + schemas.SCOPED_SUFFIX, d, mkt_rows,
                            upload, ingest, historical)
    print(f"[scoped] dt={d} markets={rows_m:>4} trades={rows_t:>6} "
          f"{'hist' if historical else 'live'} "
          f"{time.time()-t0:>5.0f}s rss={parquetio.peak_rss_kb()//1024}MB",
          flush=True)
    return {"markets": rows_m, "trades": rows_t}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="A")
    ap.add_argument("--series", default=None, help="CSV; overrides --tier")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--oldest-first", action="store_true")
    a = ap.parse_args(argv)

    raise_rate_limit()
    cut = cutoff()
    upload = not a.no_upload
    series_list = ([s.strip() for s in a.series.split(",") if s.strip()]
                   if a.series else list(TIERS[a.tier]))

    end_d = date.fromisoformat(a.end) if a.end else config.today_et_date()
    start_d = (date.fromisoformat(a.start) if a.start
               else end_d - timedelta(days=a.days))

    days = []
    dd = start_d
    while dd < end_d:
        if dd.weekday() < 5:            # no 4pm close market on a weekend
            days.append(dd)
        dd += timedelta(days=1)
    if not a.oldest_first:
        days.reverse()

    print(f"[scoped] series={','.join(series_list)} "
          f"{start_d}..{end_d} ({len(days)} weekdays)")

    totals = {"days": 0, "skipped": 0, "markets": 0, "trades": 0}
    for d in days:
        r = backfill_day(d, series_list, cut, upload, not a.no_resume)
        if r.get("skipped"):
            totals["skipped"] += 1
            continue
        totals["days"] += 1
        for kk in ("markets", "trades"):
            totals[kk] += r.get(kk, 0)

    print(f"[scoped] DONE days={totals['days']} skipped={totals['skipped']} "
          f"markets={totals['markets']} trades={totals['trades']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
