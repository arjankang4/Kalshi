"""One year of Kalshi REST history -> Parquet in the long-term bucket.

The shape of this job is dictated by two measured facts:

  * Financials+Crypto spans ~1M markets over a year (KXBTC alone has 297k, from
    its 15-minute range markets). A per-market loop is ~1M requests.
  * `/markets/trades` and `/historical/trades` accept min_ts/max_ts/cursor with
    no series filter, and 34.1% of the exchange-wide tape is in scope.

So trades are swept **time-ordered, exchange-wide** and filtered locally: ~700k
requests for the whole exchange instead of ~1M for a subset of it.

The 2026-06-20 historical cutoff splits every call: trades filled before it live
only under /historical/trades, and markets settled before it only under
/historical/markets.

Usage:
    python -m storage.rest_backfill dims                    # series/events/markets
    python -m storage.rest_backfill trades --days 365
    python -m storage.rest_backfill trades --start 2026-08-01 --end 2026-08-20
"""
import argparse
import gzip
import hashlib
import heapq
import json
import os
import shutil
import sys
import threading
from concurrent import futures
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import config
from kalshi import client as k
from storage import paths, parquetio, s3util, schemas
from storage.units import to_cc, to_micros, to_strike

# Scope is a category AND a tag filter. Kalshi has no sub_category field -- the subsections
# the UI shows are `tags`, and /series accepts them as a comma-separated OR filter.
#
# `Indices` is in the list and must stay: KXNASDAQ100, KXNASDAQ100U, KXNASDAQDUD and KXINXDUD
# are tagged ONLY `Indices`, so a Markets+Interest Rates filter silently drops the whole Nasdaq
# side. Restricting to Financials also drops 6 Commodities series carrying the `Markets` tag.
#
# Measured: 146 series, down from 1,172 under the old (Financials, Crypto) categories -- an 8x
# cut, with Crypto (the largest market-count contributor by far) leaving entirely.
# Per-category rules, because the two halves of the scope are asked for
# differently: Financials is wanted only in three of its UI subsections, Crypto is
# wanted whole. A single `category AND tag` test cannot express that -- it either
# drops all of Crypto or drags in every Financials series.
#
#   value = tuple of tags -> in scope if the series carries ANY of them
#   value = None          -> the whole category, regardless of tags
#
# Measured against the live exchange 2026-08-21 (13,359 series total):
#   Financials AND {Markets, Interest Rates, Indices}   146
#   Crypto (all tags)                                   272
#                                             TOTAL     418
#
# Crypto carries tags the Financials rule would never match ('BTC', 'ETH', '15 min',
# 'Hourly', 'Pre-Market', ...), which is exactly why it needs the None rule.
SCOPE_RULES = {
    "Financials": ("Markets", "Interest Rates", "Indices"),
    "Crypto": None,
}
IN_SCOPE_CATEGORIES = tuple(SCOPE_RULES)
IN_SCOPE_TAGS = SCOPE_RULES["Financials"]
SPILL_BUCKETS = 64
# Spill runs are capped by SERIALIZED BYTES, not by row count. Buckets are
# series ranges, so a single hot series can otherwise put an unbounded amount
# into one bucket; capping bounds what sorted_rows() holds at once.
#
# Bytes rather than rows because memory tracks row WIDTH, and the widths here
# differ by 3x: markets is 54 columns, trades is 19. A 200k-row cap measured
# 2.19 GB peak on the markets emit -- over the unit's MemoryMax=2G -- while the
# same cap is harmless for trades. One byte budget is correct for both.
# Python dicts run ~2-3x their JSON length, so this is a ~150 MB working set.
RUN_BYTES = 48 << 20
RUN_ROWS = 500_000          # secondary bound, for pathologically narrow rows

# The trading path keeps the conservative bucket; the backfill runs at the real
# account limit with headroom. Verified against GET /account/limits 2026-08-21:
# usage_tier=advanced, read.refill_rate=300 tok/s, i.e. 30 req/s at 10 tokens per
# request. raise_rate_limit() clamps to min(BACKFILL_RATE, rate*0.9) = 27.
# This is a CEILING on our own side; the server-side budget is shared per API
# key, so a concurrent quote cycle is left ~3 req/s while this runs.
BACKFILL_RATE = 27.0

# Concurrent ET days. A day is serial by construction (cursor pagination), so the
# only way to use the rate budget is to run several days at once. 4 x the measured
# ~7.7-13.9 req/s single-day rate lands at or above the 27 req/s ceiling, where the
# shared token bucket becomes the binding constraint again -- which is the goal.
# Memory is ~340 MB per day at peak, so 4 workers is ~1.4 GB against MemoryMax=8G.
TRADES_WORKERS = 4


def raise_rate_limit():
    try:
        lim = k._request("GET", "/account/limits", signed=True)
        rate = lim["read"]["refill_rate"] / 10.0
        k._RATE_PER_SEC = max(1.0, min(BACKFILL_RATE, rate * 0.9))
        print(f"[rate] account tier={lim.get('usage_tier')} "
              f"read={lim['read']['refill_rate']} tok/s -> {k._RATE_PER_SEC:.1f} req/s")
    except Exception as e:                           # noqa: BLE001
        k._RATE_PER_SEC = 8.0
        print(f"[rate] could not read /account/limits ({e}); staying at 8 req/s")


def cutoff():
    c = k.get("/historical/cutoff")
    out = {kk: _ts(v) for kk, v in c.items()}
    print(f"[cutoff] market_settled={c['market_settled_ts']} "
          f"trades_created={c['trades_created_ts']}")
    return out


def _ts(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _now():
    return datetime.now(timezone.utc)


def _canon_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _jstr(o):
    return None if o is None else json.dumps(o, sort_keys=True, separators=(",", ":"))


STATUS_GROUP = {
    "initialized": "unopened", "inactive": "unopened", "active": "open",
    "closed": "closed", "determined": "closed", "disputed": "closed",
    "amended": "closed", "finalized": "settled",
}


# --- dimensions ----------------------------------------------------------

def in_scope_series():
    """The capture universe, per SCOPE_RULES: a whole category or a tag subset of it.

    Filtered locally rather than via the `tags` query param even though /series supports it --
    one unpaginated call returns everything, so a server-side filter saves no requests, and
    doing it here keeps the scope auditable in one place instead of split across a URL.
    """
    allser = k.paged("/series", "series", limit=200)
    out = []
    for s in allser:
        if s.get("category") not in SCOPE_RULES:
            continue
        tags = SCOPE_RULES[s["category"]]
        if tags is None or (set(tags) & set(s.get("tags") or ())):
            out.append(s)
    return out


def series_rows(raw, ingest):
    for s in raw:
        yield {
            "ticker": s["ticker"], "title": s.get("title"),
            "category": s.get("category"), "frequency": s.get("frequency"),
            "tags": s.get("tags") or [],
            "settlement_sources": [{"name": x.get("name"), "url": x.get("url")}
                                   for x in (s.get("settlement_sources") or [])],
            "contract_url": s.get("contract_url"),
            "contract_terms_url": s.get("contract_terms_url"),
            "fee_type": s.get("fee_type"), "fee_multiplier": s.get("fee_multiplier"),
            "additional_prohibitions": s.get("additional_prohibitions") or [],
            "product_metadata_json": _jstr(s.get("product_metadata")),
            "volume_cc": to_cc(s.get("volume_fp")) if s.get("volume_fp") else None,
            "exchange_index": s.get("exchange_index"),
            "last_updated_ts": _ts(s.get("last_updated_ts")),
            "row_hash": _canon_hash(s), "observed_ts": ingest, "ingest_ts": ingest,
            "ts_quality": "observed", "source": "rest_series",
            "schema_version": config.SCHEMA_VERSION,
        }


def event_rows(raw, ingest):
    for e in raw:
        yield {
            "event_ticker": e["event_ticker"], "series_ticker": e.get("series_ticker"),
            "title": e.get("title"), "sub_title": e.get("sub_title"),
            "category_legacy": e.get("category"),
            "collateral_return_type": e.get("collateral_return_type"),
            "mutually_exclusive": e.get("mutually_exclusive"),
            "available_on_brokers": e.get("available_on_brokers"),
            "strike_date": _ts(e.get("strike_date")),
            "strike_period": e.get("strike_period"),
            "settlement_sources": [{"name": x.get("name"), "url": x.get("url")}
                                   for x in (e.get("settlement_sources") or [])],
            "product_metadata_json": _jstr(e.get("product_metadata")),
            "fee_type_override": e.get("fee_type_override"),
            "fee_multiplier_override": e.get("fee_multiplier_override"),
            "exchange_index": e.get("exchange_index"),
            "last_updated_ts": _ts(e.get("last_updated_ts")),
            "row_hash": _canon_hash(e), "observed_ts": ingest, "ingest_ts": ingest,
            "ts_quality": "observed", "source": "rest_events",
            "schema_version": config.SCHEMA_VERSION,
        }


def market_rows(raw, series_of, ingest, source):
    for m in raw:
        pr = m.get("price_ranges") or []
        ranges, steps = [], []
        for r in pr:
            step = to_micros(r.get("step"))
            steps.append(step)
            ranges.append({"start_micros": to_micros(r.get("start")),
                           "end_micros": to_micros(r.get("end")),
                           "step_micros": step})
        yield {
            "ticker": m["ticker"], "event_ticker": m.get("event_ticker"),
            "series_ticker": series_of.get(m.get("event_ticker")) or
                             m["ticker"].split("-")[0],
            "market_type": m.get("market_type"),
            "exchange_index": m.get("exchange_index"),
            "yes_sub_title": m.get("yes_sub_title"), "no_sub_title": m.get("no_sub_title"),
            "created_time": _ts(m.get("created_time")),
            "updated_time": _ts(m.get("updated_time")),
            "open_time": _ts(m.get("open_time")), "close_time": _ts(m.get("close_time")),
            "expected_expiration_time": _ts(m.get("expected_expiration_time")),
            "latest_expiration_time": _ts(m.get("latest_expiration_time")),
            "settlement_ts": _ts(m.get("settlement_ts")),
            "settlement_timer_seconds": m.get("settlement_timer_seconds"),
            "status": m.get("status"),
            "status_query_group": STATUS_GROUP.get(m.get("status")),
            "result": m.get("result"), "can_close_early": m.get("can_close_early"),
            "is_provisional": m.get("is_provisional"),
            "yes_bid_micros": to_micros(m.get("yes_bid_dollars")),
            "yes_bid_size_cc": to_cc(m.get("yes_bid_size_fp")),
            "yes_ask_micros": to_micros(m.get("yes_ask_dollars")),
            "yes_ask_size_cc": to_cc(m.get("yes_ask_size_fp")),
            "no_bid_micros": to_micros(m.get("no_bid_dollars")),
            "no_ask_micros": to_micros(m.get("no_ask_dollars")),
            "last_price_micros": to_micros(m.get("last_price_dollars")),
            "previous_yes_bid_micros": to_micros(m.get("previous_yes_bid_dollars")),
            "previous_yes_ask_micros": to_micros(m.get("previous_yes_ask_dollars")),
            "previous_price_micros": to_micros(m.get("previous_price_dollars")),
            "volume_cc": to_cc(m.get("volume_fp")),
            "volume_24h_cc": to_cc(m.get("volume_24h_fp")),
            "open_interest_cc": to_cc(m.get("open_interest_fp")),
            "notional_value_micros": to_micros(m.get("notional_value_dollars")),
            "settlement_value_micros": to_micros(m.get("settlement_value_dollars")),
            "strike_type": m.get("strike_type"),
            "floor_strike": to_strike(m.get("floor_strike")),
            "cap_strike": to_strike(m.get("cap_strike")),
            "functional_strike": m.get("functional_strike"),
            "custom_strike_json": _jstr(m.get("custom_strike")),
            "expiration_value": m.get("expiration_value"),
            "price_level_structure": m.get("price_level_structure"),
            "price_ranges": ranges,
            "min_tick_micros": min([s for s in steps if s], default=None),
            "rules_primary": m.get("rules_primary"),
            "rules_secondary": m.get("rules_secondary"),
            "mve_collection_ticker": m.get("mve_collection_ticker"),
            "row_hash": _canon_hash(m), "observed_ts": ingest,
            "source_ts": _ts(m.get("updated_time")), "ingest_ts": ingest,
            "ts_quality": "observed", "source": source,
            "schema_version": config.SCHEMA_VERSION,
        }


# --- trades --------------------------------------------------------------

def trade_row(t, series_of_ticker, ingest, source):
    series = t["ticker"].split("-")[0]
    return {
        "trade_id": t["trade_id"], "ticker": t["ticker"],
        "event_ticker": None, "series_ticker": series, "market_id": None,
        "yes_price_micros": to_micros(t["yes_price_dollars"]),
        "no_price_micros": to_micros(t["no_price_dollars"]),
        "count_cc": to_cc(t["count_fp"]),
        "taker_outcome_side": t.get("taker_outcome_side"),
        "taker_book_side": t.get("taker_book_side"),
        "taker_side_legacy": t.get("taker_side"),
        "is_block_trade": bool(t.get("is_block_trade")),
        "created_time": _ts(t["created_time"]), "source_ts": _ts(t["created_time"]),
        "ingest_ts": ingest, "ts_quality": "exchange_us",
        "source": source, "ws_sid": None,
        "schema_version": config.SCHEMA_VERSION,
    }


def cutoff_windows(lo_ts, hi_ts, cut_ts):
    """Split [lo_ts, hi_ts) at the historical cutoff -> [(lo, hi, historical), ...].

    THE CUTOFF IS AN INSTANT, THE PARTITION IS A DAY, and on exactly one day of the year those
    two disagree. `backfill_trades` decided the endpoint once per day with `historical = hi <=
    trades_cut`, so on the straddling day the whole 24 hours was swept from `/markets/trades`.
    That endpoint only serves the LIVE side of the cutoff, so `[lo, cut)` came back EMPTY -- and
    an empty sweep is not an error here, it is a quiet `0 rows`. Worse, the day was then sealed:
    `_write_manifest` wrote its manifest and `already_done` skips any partition that has one, so
    the missing hours could never be re-fetched without deleting the manifest by hand. The gap is
    permanent, silent, and lands on a day whose OTHER hours look perfectly healthy.

    Returning a list rather than a flag is the fix: each side of the cutoff is swept from the
    endpoint that actually serves it, and the day is one Parquet partition as before.
    """
    if cut_ts is None or hi_ts <= cut_ts:
        return [(lo_ts, hi_ts, True)]
    if lo_ts >= cut_ts:
        return [(lo_ts, hi_ts, False)]
    return [(lo_ts, cut_ts, True), (cut_ts, hi_ts, False)]


def sweep_trades_window(lo_ts, hi_ts, scope, historical, tickers=None):
    """Yield in-scope raw trades for [lo_ts, hi_ts). Cursor-paginated.

    `scope` filters by series prefix; `tickers`, when given, narrows further to an
    exact market set. Both are client-side because the endpoint takes no series,
    category or ticker filter -- the whole exchange tape is fetched either way, so
    narrowing costs nothing extra and saves only local work."""
    path = "/historical/trades" if historical else "/markets/trades"
    source = "rest_historical_trades" if historical else "rest_trades"
    cursor, pages, seen, kept = None, 0, 0, 0
    while True:
        p = {"limit": 1000, "min_ts": lo_ts, "max_ts": hi_ts}
        if cursor:
            p["cursor"] = cursor
        r = k.get(path, **p)
        batch = r.get("trades") or []
        seen += len(batch)
        pages += 1
        for t in batch:
            if t["ticker"].split("-")[0] in scope and (
                    tickers is None or t["ticker"] in tickers):
                kept += 1
                yield t, source
        cursor = r.get("cursor")
        if pages % 250 == 0:
            print(f"    ...{pages} pages, {seen} seen, {kept} in scope", flush=True)
        if not batch or not cursor:
            break
    return


class SpillSorter:
    """External sort by the leading sort key, with bounded memory.

    Buckets are contiguous *ranges* of series_ticker, not hash buckets, so
    processing them in order yields true global sort order -- which is what makes
    the declared sorting_columns metadata honest and lets Athena prune on
    row-group statistics.
    """

    def __init__(self, scratch, series_sorted, nbuckets=SPILL_BUCKETS,
                 run_rows=RUN_ROWS, run_bytes=RUN_BYTES):
        self.dir = scratch
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        per = max(1, (len(series_sorted) + nbuckets - 1) // nbuckets)
        self.bucket_of = {s: i // per for i, s in enumerate(series_sorted)}
        self.nbuckets = max(self.bucket_of.values()) + 1 if self.bucket_of else 1
        self.run_rows = run_rows
        self.run_bytes = run_bytes
        self._fh = {}         # bucket -> open handle for its current run
        self._run = {}        # bucket -> index of its current run
        self._in_run = {}     # bucket -> rows written to the current run
        self._bytes_in_run = {}   # bucket -> serialized bytes in current run
        self.count = 0

    def _run_path(self, b, n):
        return self.dir / f"b{b:03d}-r{n:03d}.jsonl.gz"

    def add(self, series, row):
        b = self.bucket_of.get(series, self.nbuckets - 1)
        fh = self._fh.get(b)
        # Buckets are series *ranges*, so one hot series can otherwise make a
        # single bucket unbounded. Roll to a new run once this one is full;
        # sorted_rows() merges the runs back, so ordering is unaffected.
        if fh is not None and (self._bytes_in_run[b] >= self.run_bytes
                               or self._in_run[b] >= self.run_rows):
            fh.close()
            fh = None
            self._run[b] += 1
        if fh is None:
            n = self._run.setdefault(b, 0)
            fh = self._fh[b] = gzip.open(self._run_path(b, n), "wt",
                                         compresslevel=1)
            self._in_run[b] = 0
            self._bytes_in_run[b] = 0
        line = json.dumps(row, separators=(",", ":"), default=str) + "\n"
        fh.write(line)
        self._in_run[b] += 1
        self._bytes_in_run[b] += len(line)
        self.count += 1

    def close_writers(self):
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()

    def sorted_rows(self, sort_key):
        """Bucket by bucket in range order, runs merged within each bucket.

        Peak is ONE RUN, not one bucket: each run is sorted and rewritten on its
        own, then the sorted runs are streamed through heapq.merge a row at a
        time. Sorting the runs lazily inside the merge would not work -- merge
        primes every input immediately, so every run would be resident at once.
        """
        keyf = lambda r: tuple(str(r.get(c) or "") for c in sort_key)  # noqa: E731
        for b in range(self.nbuckets):
            runs = sorted(self.dir.glob(f"b{b:03d}-r*.jsonl.gz"))
            if not runs:
                continue
            if len(runs) == 1:
                # Common case: no merge pass, and no second trip through disk.
                with gzip.open(runs[0], "rt") as fh:
                    rows = [json.loads(line) for line in fh]
                rows.sort(key=keyf)
                for r in rows:
                    yield _revive(r)
                rows.clear()
                continue

            sorted_runs = []
            for p in runs:                      # peak here is one run
                with gzip.open(p, "rt") as fh:
                    rows = [json.loads(line) for line in fh]
                rows.sort(key=keyf)
                # 's' prefix, deliberately not 'r': the run glob above must
                # never pick these up on a re-entry.
                sp = self.dir / p.name.replace("-r", "-s", 1)
                with gzip.open(sp, "wt", compresslevel=1) as out:
                    for r in rows:
                        out.write(json.dumps(r, separators=(",", ":"),
                                             default=str) + "\n")
                rows.clear()
                sorted_runs.append(sp)

            handles = [gzip.open(sp, "rt") for sp in sorted_runs]
            try:
                streams = [(json.loads(line) for line in h) for h in handles]
                for r in heapq.merge(*streams, key=keyf):
                    yield _revive(r)
            finally:
                for h in handles:
                    h.close()

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


_TS_COLS = ("created_time", "source_ts", "ingest_ts", "updated_time",
             "open_time", "close_time", "expected_expiration_time",
             "latest_expiration_time", "settlement_ts", "observed_ts",
             "strike_date", "last_updated_ts", "end_period_ts", "start_period_ts")
_DEC_COLS = ("floor_strike", "cap_strike")


def _revive(r):
    """Undo the JSON round-trip the spill imposes.

    json.dumps(default=str) stringifies datetimes and Decimals; without this the
    Arrow builder would reject them, or worse, coerce them differently than the
    digest expects.
    """
    for c in _TS_COLS:
        v = r.get(c)
        if isinstance(v, str):
            r[c] = datetime.fromisoformat(v)
    for c in _DEC_COLS:
        v = r.get(c)
        if isinstance(v, str):
            r[c] = Decimal(v)
    return r


# Which producer we are, for tables written by more than one (see paths.SOURCED_TABLES).
# `trades` is also filled by promote_ws from the WebSocket tape. They used to share one
# key: promote overwrote ours, or our already_done() saw promote's manifest and skipped
# the day. Keyed by src, both tapes survive and reconcile.py can compare them.
SRC = "rest"


def _src(table):
    return SRC if table in paths.SOURCED_TABLES else None


def already_done(table, d):
    """A partition with a manifest is complete. Promotion and backfill are pure
    functions of their inputs, so re-deriving it would produce the same bytes --
    skipping is not a shortcut, it is the correct no-op.

    Scoped to OUR src: promote_ws writing the WebSocket trades for a day says
    nothing about whether the REST tape for it exists."""
    return s3util.exists(config.S3_BUCKET_HIST,
                         paths.promotion_manifest_key(table, d, src=_src(table)))


def backfill_trades(start_d, end_d, scope, series_sorted, cut, upload=True,
                    newest_first=True, resume=True, tickers=None):
    """One Parquet partition per ET day.

    Newest-first by default: the full year takes ~20 h at this account's rate
    limit, and recent history is the part that is useful soonest.
    """
    ingest = _now()
    trades_cut = cut["trades_created_ts"]
    totals = {"days": 0, "rows": 0, "bytes": 0, "skipped": 0}

    days = []
    d = start_d
    while d < end_d:
        days.append(d)
        d += timedelta(days=1)
    if newest_first:
        days.reverse()

    def one_day(d):
        """Sweep, sort, write and upload a single ET day. Runs on a worker thread.

        Days are independent: each has its own [min_ts, max_ts) window, its own
        scratch dir (paths.scratch_dir keys on dt), its own Parquet object and its
        own manifest. Nothing is shared but the rate limiter, which is what we
        WANT saturated -- see the note on workers above.
        """
        if resume and upload and already_done("trades", d):
            return {"skipped": 1}
        lo, hi = config.et_day_bounds(d)
        wins = cutoff_windows(int(lo.timestamp()), int(hi.timestamp()),
                              None if trades_cut is None else int(trades_cut.timestamp()))
        # One label for the manifest and the log. A day that straddles the cutoff is genuinely
        # both, and calling it either one would misdescribe half its rows.
        kinds = {w[2] for w in wins}
        historical = True in kinds and False not in kinds
        label = "hist" if kinds == {True} else ("live" if kinds == {False} else "hist+live")
        t0 = time.time()

        scratch = paths.scratch_dir("trades", d)
        sorter = SpillSorter(scratch, series_sorted)
        try:
            for w_lo, w_hi, w_hist in wins:
                for t, source in sweep_trades_window(w_lo, w_hi, scope, w_hist, tickers):
                    row = trade_row(t, None, ingest, source)
                    sorter.add(row["series_ticker"], row)
            sorter.close_writers()

            if sorter.count == 0:
                print(f"[trades] dt={d} 0 rows ({label})", flush=True)
                return {}

            out = scratch / "part-00000.snappy.parquet"
            digests = parquetio.write_rows(
                out, "trades", sorter.sorted_rows(schemas.SORT_KEYS["trades"]), schemas)
            ok, got, meta = parquetio.verify_file(out, schemas.TABLES["trades"], digests)
            if not ok:
                raise RuntimeError(f"dt={d}: parquet verification failed {got} != {digests}")

            if upload:
                key = paths.parquet_key("trades", d, src=_src("trades"))
                rec = s3util.put_large(out, config.S3_BUCKET_HIST, key)
                _write_manifest("trades", d, digests, meta, rec, historical, ingest,
                                windows=wins)

            print(f"[trades] dt={d} rows={meta['num_rows']:>7} "
                  f"parquet={meta['bytes']/1e6:>6.1f}MB rg={meta['row_groups']} "
                  f"{label} {time.time()-t0:>5.0f}s "
                  f"rss={parquetio.peak_rss_kb()//1024}MB", flush=True)
            return {"days": 1, "rows": meta["num_rows"], "bytes": meta["bytes"]}
        finally:
            sorter.cleanup()

    # Parallel across DAYS, not within one. A day's pages are cursor-paginated and
    # therefore strictly serial -- page N+1 needs the cursor from page N -- so a
    # single day cannot go faster than its round-trip latency allows. Measured
    # 2026-08-21 on dt=2026-08-20: ~8,800 pages in 1141 s = 7.7 req/s achieved,
    # against a 27 req/s budget. The job was never rate-limited; it was latency-
    # bound, and the standing "parallelising will not help" note was wrong.
    #
    # k._throttle() holds a module-level threading.Lock over a shared token bucket,
    # so N workers cannot exceed the account limit no matter how N is set -- the
    # pool converts idle wait into throughput and nothing more.
    #
    # Days are submitted newest-first, so the useful recent history still lands
    # first even though completion order is not guaranteed.
    errors = []
    lock = threading.Lock()
    with futures.ThreadPoolExecutor(max_workers=TRADES_WORKERS) as ex:
        fut_day = {ex.submit(one_day, d): d for d in days}
        for f in futures.as_completed(fut_day):
            d = fut_day[f]
            try:
                got = f.result()
            except Exception as e:                   # noqa: BLE001
                # One bad day must not abort the other 181. It wrote no manifest,
                # so a later run picks it up again -- that is the resume contract.
                errors.append((d, e))
                print(f"[trades] dt={d} FAILED {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            with lock:
                for kk, v in got.items():
                    totals[kk] = totals.get(kk, 0) + v

    print(f"[trades] DONE days={totals['days']} skipped={totals['skipped']} "
          f"rows={totals['rows']} bytes={totals['bytes']/1e9:.2f}GB "
          f"failed={len(errors)} workers={TRADES_WORKERS}")
    if errors:
        print("[trades] failed days (no manifest written; re-run to retry): "
              + ", ".join(str(d) for d, _ in errors), file=sys.stderr)
    return totals


def _write_manifest(table, d, digests, meta, rec, historical, ingest, windows=None):
    # `windows` records WHICH endpoint served WHICH span of the day. On the one day a year that
    # straddles the historical cutoff the answer is "both", and a manifest that claimed a single
    # source would make an incomplete partition indistinguishable from a complete one -- which is
    # what let the pre-cutoff hours go missing and stay missing. See `cutoff_windows`.
    src_label = ("rest_historical" if historical else "rest_live")
    if windows and len({w[2] for w in windows}) > 1:
        src_label = "rest_historical+rest_live"
    doc = {"table": table, "partition": dict([paths.partition_kv(table, d)]),
           "src": _src(table),
           "schema_version": config.SCHEMA_VERSION,
           "pyarrow_version": __import__("pyarrow").__version__,
           "source": src_label,
           "windows": [{"min_ts": w[0], "max_ts": w[1],
                        "endpoint": "/historical/trades" if w[2] else "/markets/trades"}
                       for w in (windows or ())],
           "output": {"key": rec["key"], "version_id": rec.get("version_id"),
                      "object_sha256": rec["object_sha256"],
                      "row_count": meta["num_rows"], "row_groups": meta["row_groups"],
                      "bytes": meta["bytes"]},
           "content_digests": digests,
           "runtime": {"peak_rss_kb": parquetio.peak_rss_kb(),
                       "written_at": ingest.isoformat()}}
    s3util.put_bytes(json.dumps(doc, indent=1).encode(), config.S3_BUCKET_HIST,
                     paths.promotion_manifest_key(table, d, src=_src(table)))


def backfill_dims(upload=True, resume=True):
    """series, events, markets. Cheap relative to trades; run first.

    `resume=False` re-derives every table even if it already has a manifest.
    Needed whenever the scope filters change: the tables carry whatever scope
    was in force when they were written, and already_done() cannot see that.
    """
    ingest = _now()
    raw_series = in_scope_series()
    scope = {s["ticker"] for s in raw_series}
    print(f"[dims] {len(scope)} in-scope series "
          f"({', '.join(IN_SCOPE_CATEGORIES)})")

    d = config.today_et_date()
    out_dir = paths.scratch_dir("dims", d)
    out_dir.mkdir(parents=True, exist_ok=True)

    # series
    _emit("series", d, out_dir, series_rows(raw_series, ingest), upload, ingest,
          resume)

    # events -- spilled, not accumulated. This is ~226k events across the scope;
    # holding them as dicts is a few hundred MB that then sits in memory for the
    # whole markets phase as well.
    #
    # The resume check is here rather than inside _emit because the enumeration
    # itself is ~40 minutes of API calls. `series_of` only refines the
    # series attribution for markets, and the ticker-prefix fallback below is
    # reliable for Kalshi (a market ticker always begins with its series
    # ticker), so skipping the sweep costs nothing.
    series_of, n_events = {}, 0
    if resume and upload and already_done("events", d):
        print("[dims] events     already present, skipping enumeration")
    else:
      ev_sorter = SpillSorter(paths.scratch_dir("dims_events", d), sorted(scope))
      for i, st in enumerate(sorted(scope), 1):
          try:
            for e in k.paged_iter("/events", "events", limit=200,
                                  series_ticker=st):
                series_of[e["event_ticker"]] = e.get("series_ticker") or st
                for row in event_rows([e], ingest):
                    ev_sorter.add(row["series_ticker"] or st, row)
                n_events += 1
          except Exception as e:                     # noqa: BLE001
            print(f"[dims] events {st}: {type(e).__name__}", file=sys.stderr)
            continue
          if i % 100 == 0:
            print(f"[dims] events {i}/{len(scope)} series, {n_events} events, "
                  f"rss={parquetio.peak_rss_kb()//1024}MB", flush=True)
      ev_sorter.close_writers()
      _emit("events", d, out_dir,
            ev_sorter.sorted_rows(schemas.SORT_KEYS["events"]), upload, ingest,
            resume)
      ev_sorter.cleanup()

    # markets: live and historical, unioned on ticker. /historical/markets takes
    # no timestamp filters, so it must be enumerated by series.
    #
    # Spilled rather than accumulated: as a list of dicts this is multiple GB.
    # Peak here is one series' worth of markets plus one spill run.
    #
    # The union is deduped PER SERIES, not globally. A global `seen` set held
    # every ticker in scope for the whole run -- ~140 B each with set overhead,
    # which at the old (Financials, Crypto) scope was ~1.5 GB against the unit's
    # MemoryMax=2G, and was the dominant term in the 11 crashes of 2026-08-20.
    # Per-series is equivalent because of the invariant noted above: a market
    # ticker always begins with its series ticker, so the same ticker can never
    # arrive under two different series. The loops are inverted so that both
    # sources for one series are visited together and the set can be dropped.
    sorter = SpillSorter(paths.scratch_dir("dims_markets", d), sorted(scope))
    n_markets = 0
    for i, st in enumerate(sorted(scope), 1):
        seen_st = set()
        for label, path, source in (("live", "/markets", "rest_markets"),
                                    ("hist", "/historical/markets",
                                     "rest_historical_markets")):
            try:
                # Streamed: a single paged() call on a large series would
                # materialize every market dict at once, which is most of a GB.
                for m in k.paged_iter(path, "markets", limit=1000,
                                      series_ticker=st):
                    if m["ticker"] in seen_st:
                        continue
                    seen_st.add(m["ticker"])
                    for row in market_rows([m], series_of, ingest, source):
                        sorter.add(row["series_ticker"], row)
            except Exception as e:                   # noqa: BLE001
                print(f"[dims] {label} markets {st}: {type(e).__name__}",
                      file=sys.stderr)
                continue
        n_markets += len(seen_st)
        del seen_st
        if i % 100 == 0:
            print(f"[dims] markets {i}/{len(scope)} series, "
                  f"{n_markets} markets, rss={parquetio.peak_rss_kb()//1024}MB",
                  flush=True)
    sorter.close_writers()
    _emit("markets", d, out_dir,
          sorter.sorted_rows(schemas.SORT_KEYS["markets"]), upload, ingest,
          resume)
    sorter.cleanup()
    return 0


def _emit(table, d, out_dir, rows, upload, ingest, resume=True):
    if resume and upload and already_done(table, d):
        # Same reasoning as trades: a partition with a manifest is complete, and
        # re-deriving it would produce identical bytes.
        print(f"[dims] {table:<10} already present, skipped")
        for _ in rows:
            pass
        return
    out = out_dir / f"{table}.parquet"
    digests = parquetio.write_rows(out, table, rows, schemas)
    ok, got, meta = parquetio.verify_file(out, schemas.TABLES[table], digests)
    if not ok:
        raise RuntimeError(f"{table}: parquet verification failed")
    if upload:
        rec = s3util.put_large(out, config.S3_BUCKET_HIST, paths.parquet_key(table, d))
        _write_manifest(table, d, digests, meta, rec, False, ingest)
    print(f"[dims] {table:<10} rows={meta['num_rows']:>8} "
          f"parquet={meta['bytes']/1e6:>7.2f}MB rss={parquetio.peak_rss_kb()//1024}MB",
          flush=True)
    out.unlink(missing_ok=True)


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else []


def _trades_scope(a):
    """(series scope, exact ticker set or None) for the trades sweep.

    A market ticker always begins with its series ticker, so --tickers can derive
    its own series scope: the cheap prefix test still does the bulk of the
    filtering and the exact set only refines it.
    """
    series, tickers = _csv(a.series), _csv(a.tickers)
    if not series and not tickers:
        return {s["ticker"] for s in in_scope_series()}, None
    scope = set(series) or {t.split("-")[0] for t in tickers}
    if tickers:
        # Anything named by --tickers must survive the prefix test too, or it
        # would be fetched and then silently dropped.
        scope |= {t.split("-")[0] for t in tickers}
    print(f"[trades] scope={len(scope)} series"
          + (f", {len(tickers)} exact tickers" if tickers else ""))
    return scope, (set(tickers) or None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("dims", "trades", "all"))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--oldest-first", action="store_true",
                    help="default is newest-first: recent history is useful soonest")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-derive partitions that already have a manifest")
    ap.add_argument("--series", default=None,
                    help="CSV of series tickers; overrides the configured scope")
    ap.add_argument("--tickers", default=None,
                    help="CSV of full market tickers; implies their series")
    a = ap.parse_args(argv)

    raise_rate_limit()
    cut = cutoff()
    upload = not a.no_upload

    if a.what in ("dims", "all"):
        backfill_dims(upload, resume=not a.no_resume)
    if a.what in ("trades", "all"):
        end_d = date.fromisoformat(a.end) if a.end else config.today_et_date()
        start_d = (date.fromisoformat(a.start) if a.start
                   else end_d - timedelta(days=a.days))
        scope, tickers = _trades_scope(a)
        backfill_trades(start_d, end_d, scope, sorted(scope), cut, upload,
                        newest_first=not a.oldest_first, resume=not a.no_resume,
                        tickers=tickers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
