"""Pin the three memory fixes in the REST backfill's spill/write path.

    python -m storage.test_backfill_memory

Synthetic only -- no Kalshi API calls, no S3, no credentials. This has to be
runnable before committing the shared rate limit to a ~20 h job.

What it guards:
  1. SpillSorter splits a bucket into size-capped runs and merges them back, so
     peak is one RUN rather than one bucket. Ordering must survive the merge --
     the declared sorting_columns metadata is only honest if it does.
  2. BoundedWriter flushes on BYTES as well as rows, so a wide schema (markets
     is 54 columns) no longer buffers its way past ARROW_BUDGET before the
     1M-row flush is reached.
"""
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa

from storage import parquetio, schemas
from storage.rest_backfill import SpillSorter

FAILS = []


def _short(v):
    t = repr(v)
    return t if len(t) <= 120 else t[:117] + "..."


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: "
          f"got={_short(got)} want={_short(want)}")
    if not ok:
        FAILS.append(label)


def ok_(label, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}{(' ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


SORT_KEY = ["series_ticker", "ticker", "observed_ts"]
TS = datetime(2026, 8, 21, 0, 0, 0, tzinfo=timezone.utc)


def keyf(r):
    return tuple(str(r.get(c) or "") for c in SORT_KEY)


def idk(r):
    """Identity/order key over the columns _revive() leaves as strings.

    sorted_rows() yields revived rows -- observed_ts comes back as a datetime,
    not the ISO string that was written -- so the raw key is not comparable
    across the boundary. series_ticker+ticker are untouched, and ticker is
    unique within a series, so they fully determine the expected order.
    """
    return (str(r["series_ticker"]), str(r["ticker"]))


def rows_for(series, n, salt=0):
    """Deliberately emitted OUT of sort order so a no-op sort would be caught."""
    for i in range(n):
        j = (i * 7919 + salt) % n              # scrambled, still a permutation
        yield {"series_ticker": series,
               "ticker": f"{series}-{j:06d}",
               "observed_ts": f"2026-08-21T00:00:{j % 60:02d}Z"}


print("SpillSorter: one hot series forced across many runs")
tmp = Path(tempfile.mkdtemp(prefix="spilltest-"))
try:
    series = [f"KX{c}" for c in "ABCDEFGH"]
    # run_rows far below the per-series volume => many runs inside one bucket
    s = SpillSorter(tmp / "hot", series, nbuckets=4, run_rows=50,
                    run_bytes=1 << 30)
    expected = []
    for st in series:
        for r in rows_for(st, 260, salt=len(st)):
            s.add(st, r)
            expected.append(r)
    s.close_writers()

    runs = sorted(p.name for p in (tmp / "hot").glob("b*-r*.jsonl.gz"))
    ok_("bucket was split into multiple runs", len(runs) > len(series),
        f"({len(runs)} run files)")

    out = list(s.sorted_rows(SORT_KEY))
    check("no rows lost or duplicated", len(out), len(expected))

    got_keys = [idk(r) for r in out]
    ok_("output is globally sorted across run+bucket boundaries",
        got_keys == sorted(got_keys))
    check("multiset matches input (n distinct)",
          sorted(idk(r) for r in expected) == sorted(got_keys), True)

    leaked = list((tmp / "hot").glob("b*-s*.jsonl.gz"))
    ok_("sorted-run files do not collide with the run glob",
        all("-r" not in p.name for p in leaked))

    print("\nSpillSorter: single-run bucket takes the no-merge path")
    s2 = SpillSorter(tmp / "cold", series, nbuckets=4, run_rows=10_000,
                     run_bytes=1 << 30)
    exp2 = []
    for st in series:
        for r in rows_for(st, 20, salt=1):
            s2.add(st, r)
            exp2.append(r)
    s2.close_writers()
    out2 = list(s2.sorted_rows(SORT_KEY))
    check("single-run row count", len(out2), len(exp2))
    k2 = [idk(r) for r in out2]
    ok_("single-run output sorted", k2 == sorted(k2))
    check("single-run multiset matches",
          sorted(idk(r) for r in exp2) == sorted(k2), True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)


print("\nSpillSorter: a wide-row bucket stays inside a memory budget")
# The regression this pins: RUN_ROWS alone measured 2.19 GB peak on the real
# markets emit (54 columns, 2.59M rows) -- over the unit's MemoryMax=2G --
# because a 200k-row cap means something very different for a 54-column row
# than for a 19-column one. The byte cap is what makes one budget fit both.
tmp3 = Path(tempfile.mkdtemp(prefix="widespill-"))
try:
    import resource
    wide_series = ["KXWIDE"]
    sw = SpillSorter(tmp3 / "w", wide_series, nbuckets=1,
                     run_bytes=8 << 20)          # small budget, easy to observe
    pad = "x" * 2000                              # ~2 KB/row, markets-like
    N = 60_000
    for i in range(N):
        sw.add("KXWIDE", {"series_ticker": "KXWIDE",
                          "ticker": f"KXWIDE-{(i * 7919) % N:07d}",
                          "observed_ts": "2026-08-21T00:00:00Z",
                          "pad": pad})
    sw.close_writers()
    nruns = len(list((tmp3 / "w").glob("b*-r*.jsonl.gz")))
    ok_("byte cap split a single-series bucket into many runs", nruns > 8,
        f"({nruns} runs for ~{N * 2 // 1000} MB of rows)")

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kk = []
    for r in sw.sorted_rows(SORT_KEY):
        kk.append(r["ticker"])
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    check("all wide rows came back", len(kk), N)
    ok_("wide-bucket output is sorted", kk == sorted(kk))
    grew = (after - before) / 1024
    ok_("merge did not materialize the whole bucket", grew < 200,
        f"(peak RSS grew {grew:.0f} MB for a {N * 2 // 1000} MB bucket)")
finally:
    shutil.rmtree(tmp3, ignore_errors=True)


print("\nBoundedWriter: wide schema flushes on bytes, not on the 1M row bound")
schema = schemas.TABLES["markets"]
check("markets is the wide table", len(schema), 54)

tmp2 = Path(tempfile.mkdtemp(prefix="pqtest-"))
try:
    # Force the byte bound to fire well inside a single row group's worth of
    # rows, which is exactly the markets case at full scale.
    saved = parquetio.ROW_GROUP_BYTES
    parquetio.ROW_GROUP_BYTES = 4 << 20
    try:
        n = 120_000

        def _val(f, i):
            """A type-correct value for every column.

            Every field is populated rather than nulled: 8 of the 54 are
            non-nullable, and a fully-populated row is also the widest one,
            which is the case the byte bound has to handle.
            """
            t = f.type
            if pa.types.is_string(t):
                return f"{f.name}-{i % 1000}"
            if pa.types.is_timestamp(t):
                return TS
            if pa.types.is_decimal(t):
                return Decimal(f"{7500 + (i % 500)}.0001")
            if pa.types.is_boolean(t):
                return bool(i % 2)
            if pa.types.is_integer(t):
                return i % 1000
            if pa.types.is_list(t):
                return [{"start_micros": i, "end_micros": i + 1,
                         "step_micros": 1}]
            raise AssertionError(f"unhandled type {t} on {f.name}")

        def wide_rows():
            for i in range(n):
                r = {f.name: _val(f, i) for f in schema}
                r["series_ticker"] = f"KX{i % 8}"
                r["ticker"] = f"KX{i % 8}-{i:07d}"
                yield r

        out = tmp2 / "markets.parquet"
        digests = parquetio.write_rows(out, "markets", wide_rows(), schemas)
        good, got, meta = parquetio.verify_file(out, schema, digests)
        ok_("no MemoryError; file written", True)
        check("verify_file round-trips", good, True)
        check("all rows written", meta["num_rows"], n)
        ok_("byte bound produced >1 row group while rows < ROW_GROUP_ROWS",
            meta["row_groups"] > 1 and n < parquetio.ROW_GROUP_ROWS,
            f"(row_groups={meta['row_groups']}, rows={n})")
    finally:
        parquetio.ROW_GROUP_BYTES = saved
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

# ---------------------------------------------------------------------------
# Parallel day loop. Added 2026-08-21 when the measured single-day rate
# (~7.7-13.9 req/s against a 27 req/s budget) showed the sweep was latency-bound,
# not rate-limited, and the day loop was parallelised to use the headroom.
# ---------------------------------------------------------------------------
print("\nbackfill_trades: days run concurrently, totals aggregate, one bad day is isolated")

from datetime import date as _date, timedelta as _timedelta   # noqa: E402
import threading as _threading                                 # noqa: E402
from storage import rest_backfill as _rb                       # noqa: E402

_live = {"n": 0, "max": 0}
_livelock = _threading.Lock()
BAD_DAY = _date(2026, 8, 15)


def _fake_sweep(lo_ts, hi_ts, scope, historical, tickers=None):
    """Stand in for the real cursor sweep: no network, but the same shape.

    Keep this signature in step with sweep_trades_window(). It grew a 5th
    parameter (`tickers`) on 2026-08-21 and this stub did not, which surfaced as
    all 8 days failing with a TypeError rather than as an obvious stub mismatch --
    the day loop catches per-day exceptions by design, so a bad stub looks
    exactly like a bad day.
    """
    with _livelock:
        _live["n"] += 1
        _live["max"] = max(_live["max"], _live["n"])
    try:
        import time as _t
        _t.sleep(0.25)                       # long enough for overlap to be real
        day_tag = lo_ts % 100000
        if _date.fromtimestamp(lo_ts).day == BAD_DAY.day:
            raise RuntimeError("synthetic sweep failure")
        for i in range(50):
            yield {
                "trade_id": f"t-{day_tag}-{i:05d}",
                "ticker": f"KXBTC-{day_tag}-{i:03d}",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "3.00",
                "taker_outcome_side": "yes",
                "taker_book_side": "yes",
                "taker_side": "yes",
                "is_block_trade": False,
                "created_time": "2026-08-20T12:00:00Z",
            }, "rest_trades"
    finally:
        with _livelock:
            _live["n"] -= 1


_real_sweep = _rb.sweep_trades_window
_rb.sweep_trades_window = _fake_sweep
try:
    end = _date(2026, 8, 20)
    start = end - _timedelta(days=8)
    cut = {"trades_created_ts": None}
    # upload=False keeps this entirely local: no S3, no manifests, and
    # already_done() is never consulted.
    import datetime as _dt
    cut = {"trades_created_ts": _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc)}
    totals = _rb.backfill_trades(start, end, {"KXBTC"}, ["KXBTC"], cut,
                                 upload=False, newest_first=True, resume=False)
    ok_("more than one day ran at once", _live["max"] > 1,
        f"(peak concurrent days = {_live['max']}, workers={_rb.TRADES_WORKERS})")
    check("8 days attempted, 1 synthetic failure -> 7 written", totals["days"], 7)
    check("rows aggregated across threads", totals["rows"], 7 * 50)
    ok_("a failing day did not abort the rest", totals["days"] == 7)
finally:
    _rb.sweep_trades_window = _real_sweep

print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
sys.exit(0 if not FAILS else 1)
