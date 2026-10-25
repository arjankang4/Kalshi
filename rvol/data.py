"""Pull and cache 1-second bars for the realized-vol study.

Readers only, and never a vendor call from a notebook -- the same discipline as `backtest/data.py`.
`ensure_range` materialises the cache once; everything downstream reads Parquet off local disk.

WHAT IS PULLED, AND WHY TWO SOURCES PER UNDERLYING

    SPX -> I:SPX (index, RTH only)  + SPY (proxy, 04:00-19:59 ET)
    NDX -> I:NDX (index, RTH only)  + QQQ (proxy, 04:00-19:59 ET)

The index is the thing Kalshi settles on and is what the targets and the RTH features are built
from. But **an index has no pre-open bars at all** -- it is only computed while its constituents
trade -- so a 3-hour lookback taken at 09:30 has nothing to look back at. The ETF's extended-hours
tape is the substitute.

**CORRECTION 2026-08-22: the reason originally given for that substitution was wrong.** This
comment used to read "Massive carries no futures of any kind (probed 2026-08-21: /futures/vX 404,
market=futures empty, ESZ6 empty)". That probe produced a false negative three ways: `/futures/vX`
is a literal placeholder, not an endpoint; `market=futures` is the wrong namespace (v3 reference is
equities); and the resolution values it would have needed are `1sec`/`1min`, not `1s`/`1min`-style
guesses -- `/futures/v1/aggs` answers a bad resolution with **HTTP 400 "Invalid resolution"**, which
reads like a dead endpoint if you do not open the body. Re-probed live 2026-08-22, all of these
return data on this subscription:

    /futures/v1/aggs/{ESU6,NQU6}?resolution=1sec   {window_start(ns), open, high, low, close,
                                                    volume, transactions, dollar_volume}
    /futures/v1/quotes/{ticker}   bid/ask/sizes at ns timestamps
    /futures/v1/trades/{ticker}   price/size at ns timestamps
    /futures/v1/products, /futures/v1/contracts

Measured 2026-08-20, 1-second bar counts per ET hour, ESU6 vs the SPY proxy this module actually
uses:

    ET hour   04    05    06    07    08    09  | 10-15
    ESU6     62%   46%   41%   70%   88%   90%  | 85-99%
    SPY       7%    2%    3%   12%   21%   58%  | ~94%

That is **3.8x the pre-open observations** (14,269 vs 3,721 over 04:00-09:00 ET), concentrated in
exactly the hours where the proxy is thinnest. So the pre-open half of this feature set rests on a
substitution that is no longer justified by availability. Switching it is a real change with real
consequences (ES is a forward on a different clock, with its own basis and roll), so it has NOT
been made here -- but do not repeat "Massive has no futures" as the reason not to.

That substitution is legitimate here for the same reason `SPY x 10` is illegitimate as a level:
returns are ratio-invariant to the smooth dividend accrual that wrecks the level. Measured at the
30s scale, RTH, four sessions: SPY/I:SPX 1.019-1.035 and QQQ/I:NDX 1.021-1.052 -- a stable few
percent, which a fitted coefficient absorbs. `features` carries `proxy_frac` so the model knows
how much of each window came from the proxy.

Pre-open coverage is thin and the study must not pretend otherwise. Measured 2026-08-20, SPY
second-bar counts out of 3600 possible per ET hour: 04h 255, 05h 79, 06h 95, 07h 421, 08h 769,
09h 2102, then ~3400/h through RTH.

    python -m rvol.data --build              # ~2 years, all four tickers, resumable
    python -m rvol.data --build --days 30
"""
import argparse
import datetime
import sys
from concurrent.futures import ThreadPoolExecutor

import config
from massive import futures as fut
from massive import history as hist

CACHE = config.ROOT / "var" / "rvol" / "bars"

# index ticker and extended-hours proxy per underlying
SOURCES = {
    "SPX": {"index": "I:SPX", "proxy": "SPY", "futures": "ES"},
    "NDX": {"index": "I:NDX", "proxy": "QQQ", "futures": "NQ"},
}
ALL_TICKERS = sorted({t for v in SOURCES.values() for t in v.values()})

# The futures legs are cached under their PRODUCT code, not the delivery contract, because the
# contract changes four times a year and every reader downstream wants one continuous series per
# underlying. Which contract a given ET date resolves to is recorded in CHAIN_PATH, and the roll
# dates it implies are what `pricing/gapcal` uses to drop the roll-spanning return -- see the roll
# discussion in massive/futures.
FUTURES_PRODUCTS = {v["futures"] for v in SOURCES.values()}
CHAIN_PATH = config.ROOT / "var" / "rvol" / "futures_chain.json"

DEFAULT_YEARS = 2

# One RTH day of index second-bars is ~22.5-23k rows against a 50,000 page limit, so a per-day
# request never pages. Per-day files rather than monthly because the pull is resumable and a
# half-finished month is worse than a missing day; these live on local disk, not in Glacier, so
# the small-object argument in storage/paths.py does not apply.
WORKERS = 8

# Futures legs get fewer workers than equities legs. Not because concurrency caused the 502s -- it
# did not, the broken cursor did (see PAGE_LIMIT in massive/futures) -- but because a busy futures
# day now costs two bisected requests instead of one, and each carries ~2x the rows of an index day.
FUTURES_WORKERS = 6


def cache_path(ticker, d):
    return CACHE / ticker.replace(":", "_") / f"dt={d}.parquet"


def _write(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({"t_ms": pa.array([r[0] for r in rows], pa.int64()),
                    "close": pa.array([r[1] for r in rows], pa.float64())})
    tmp = path.with_suffix(".tmp")
    pq.write_table(tbl, tmp, compression="snappy")
    tmp.replace(path)                       # atomic: a killed pull leaves no half-file


def fetch_day(ticker, d):
    """[(epoch_ms, close)] for one ET session date, ascending.

    Only `c` is read. For an index every bar has o == c == h == l -- a computed level has no range
    inside the interval -- so h/l would read zero variance. For the ETF proxy the close is the
    right choice anyway: using the high/low of a sparse pre-open bar imports the spread.
    """
    if ticker in FUTURES_PRODUCTS:
        return _fetch_futures_day(ticker, d)
    bars = hist.second_bars(ticker, str(d), str(d))
    return [(b["t"], b["c"]) for b in bars if b.get("c") is not None and b["c"] > 0]


def load_chain():
    """{product: {ISO date: contract ticker}}, or {} before `--chain` has been run."""
    import json
    if not CHAIN_PATH.exists():
        return {}
    return json.loads(CHAIN_PATH.read_text())


def build_chain(products=None, years=DEFAULT_YEARS, end=None, days=None, resume=True):
    """Resolve the front-month contract for every session and persist it.

    Two vendor requests per product per day, so this is the slow part of a cold build -- but it
    runs once and the answer is a fact about the market rather than about our sample. Resumable:
    dates already resolved are kept, which matters because a partial chain would otherwise make
    `fetch_day` silently skip days.
    """
    import json
    products = sorted(products or FUTURES_PRODUCTS)
    end = end or config.today_et()
    start = end - datetime.timedelta(days=days if days else int(365.25 * years))
    chain = load_chain() if resume else {}
    for pc in products:
        have = chain.setdefault(pc, {})
        todo = [d for d in sessions(start, end) if str(d) not in have]
        print(f"[rvol.data] chain {pc}: {len(todo)} sessions to resolve "
              f"({len(have)} already known)")
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for d, tk in zip(todo, pool.map(lambda dd: fut.front_month(pc, dd), todo)):
                if tk:
                    have[str(d)] = tk
        n_fixed = _repair_excursions(have)
        if n_fixed:
            print(f"[rvol.data] chain {pc}: repaired {n_fixed} short-lived excursions")
        rolls = _roll_dates(have)
        print(f"[rvol.data] chain {pc}: {len(have)} sessions, {len(rolls)} rolls -> "
              f"{[f'{d}:{a}->{b}' for d, a, b in rolls]}")
    CHAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHAIN_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(chain, indent=1, sort_keys=True))
    tmp.replace(CHAIN_PATH)
    return chain


def _roll_dates(by_date):
    """[(date, from_ticker, to_ticker)] -- the sessions where the front month changed.

    The return spanning one of these is a contract spread, not a price move. Measured at the
    2024-09 roll it was 61.5 ES points (1.09%) in a single step. Any variance estimator that
    crosses it must drop that return rather than measure it.
    """
    out = []
    prev_d = prev_tk = None
    for d in sorted(by_date):
        tk = by_date[d]
        if prev_tk and tk != prev_tk:
            out.append((d, prev_tk, tk))
        prev_d, prev_tk = d, tk
    return out


# A quarterly front month is held for roughly 63 sessions. Anything held for a handful is not a
# roll, it is a selection error -- and each phantom roll costs two overnight cycles AND, if it
# were left in, would splice a multi-month basis step into the tape. The known cause is an
# incomplete vendor reference list (see MIN_FRONT_VOLUME in massive/futures); this is the
# belt-and-braces catch for causes not yet seen.
MIN_TENURE_SESSIONS = 5


def _repair_excursions(by_date, min_tenure=MIN_TENURE_SESSIONS):
    """Overwrite runs shorter than `min_tenure` with the contract that surrounds them.

    Mutates `by_date` in place and returns how many dates were rewritten.
    """
    days = sorted(by_date)
    if not days:
        return 0
    runs, start = [], 0
    for i in range(1, len(days) + 1):
        if i == len(days) or by_date[days[i]] != by_date[days[start]]:
            runs.append((start, i))
            start = i
    fixed, unrepaired = 0, []
    for j, (lo, hi) in enumerate(runs):
        # The first and last runs are short because the SAMPLE ends, not because the pick was
        # wrong. Repairing them would rewrite a genuine roll that happened days before the cache
        # was built back into the contract it replaced.
        if hi - lo >= min_tenure or j == 0 or j == len(runs) - 1:
            continue
        prev_tk = by_date[days[runs[j - 1][0]]]
        next_tk = by_date[days[runs[j + 1][0]]]
        # Only an excursion AND BACK is repairable: the surrounding runs must agree. If they
        # differ this is a short middle leg of a real two-step roll, and guessing would be worse
        # than leaving it for the roll-drop logic to handle as two boundaries.
        if prev_tk != next_tk:
            # A short run flanked by two DIFFERENT contracts is a real roll sitting next to a bad
            # pick, or a two-step roll. Guessing would be worse than leaving it, but it must not
            # be silent -- otherwise it is indistinguishable from "no excursion found here".
            unrepaired.append((days[lo], by_date[days[lo]], hi - lo, prev_tk, next_tk))
            continue
        for k in range(lo, hi):
            if by_date[days[k]] != prev_tk:
                by_date[days[k]] = prev_tk
                fixed += 1
    for d, tk, n, a, b in unrepaired:
        print(f"[rvol.data]   WARN short run NOT repaired: {tk} held {n} session(s) from {d}, "
              f"flanked by {a} and {b} -- inspect before trusting the rolls here")
    return fixed


def _fetch_futures_day(product, d):
    """[(epoch_ms, close)] for the whole ET CALENDAR day `d` on that day's front month.

    Calendar day, not CME session: the gap calendar splices 16:00 -> 09:30 by reading two
    adjacent day files, exactly as the SPY path already does, and a session-shaped file would put
    one night's data across a boundary that does not line up with either.
    """
    chain = load_chain().get(product, {})
    tk = chain.get(str(d))
    if not tk:
        return []
    start_ns = int(datetime.datetime.combine(
        d, datetime.time(0, 0), tzinfo=config.ET).timestamp() * 1e9)
    end_ns = int(datetime.datetime.combine(
        d + datetime.timedelta(days=1), datetime.time(0, 0), tzinfo=config.ET).timestamp() * 1e9)
    return fut.closes_ms(tk, start_ns, end_ns)


def ensure_day(ticker, d, resume=True):
    """(status, n_rows). status is 'cached' | 'written' | 'empty' | 'failed'.

    A vendor error is caught and reported rather than raised. `pool.map` propagates the first
    exception out of the whole build, so before this a single transient 502 anywhere in 1,044
    day-requests destroyed the run and discarded every day still in flight -- observed at day 129
    of the ES/NQ pull. A failed day is simply not written, so re-running with `resume=True` picks
    up exactly those days. `ensure_range` prints them; the failure is never silent.
    """
    p = cache_path(ticker, d)
    if resume and p.exists():
        return "cached", None
    try:
        rows = fetch_day(ticker, d)
    except Exception as e:                  # noqa: BLE001 -- a vendor error is data, not a bug
        _FAILED.append((ticker, d, repr(e)[:120]))
        return "failed", 0
    if not rows:
        return "empty", 0                   # holiday, or before the ticker existed
    _write(p, rows)
    return "written", len(rows)


_FAILED = []


def sessions(start, end):
    """Weekdays in [start, end]. Holidays fall out as empty fetches rather than a holiday table --
    the same choice varcal makes with MIN_SESSION_BARS, and for the same reason: a table goes
    stale silently and this does not."""
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def ensure_range(tickers=None, years=DEFAULT_YEARS, end=None, days=None, resume=True,
                 workers=WORKERS):
    tickers = tickers or ALL_TICKERS
    end = end or config.today_et()
    start = end - datetime.timedelta(days=days if days else int(365.25 * years))
    todo = [(t, d) for t in tickers for d in sessions(start, end)]
    print(f"[rvol.data] {len(tickers)} tickers x {len(todo)//len(tickers)} weekdays "
          f"= {len(todo)} day-requests, {start} -> {end}")
    counts = {"cached": 0, "written": 0, "empty": 0, "failed": 0}
    rows = 0
    _FAILED.clear()
    if any(t in FUTURES_PRODUCTS for t in tickers):
        workers = min(workers, FUTURES_WORKERS)
        print(f"[rvol.data] futures in the pull -> {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (status, n) in enumerate(pool.map(lambda p: ensure_day(p[0], p[1], resume), todo)):
            counts[status] += 1
            rows += n or 0
            if (i + 1) % 250 == 0:
                print(f"[rvol.data]   {i+1}/{len(todo)} {counts}")
    print(f"[rvol.data] done {counts}, {rows:,} new rows")
    if _FAILED:
        print(f"[rvol.data] {len(_FAILED)} day-requests FAILED and were not written. "
              f"Re-run the same command to retry exactly these:")
        for t, d, err in _FAILED[:20]:
            print(f"[rvol.data]   {t} {d}  {err}")
        if len(_FAILED) > 20:
            print(f"[rvol.data]   ... and {len(_FAILED) - 20} more")
    return counts


def load_day(ticker, d):
    """(t_ms, close) as numpy arrays, or (None, None) when the day is not cached."""
    import numpy as np
    import pyarrow.parquet as pq
    p = cache_path(ticker, d)
    if not p.exists():
        return None, None
    t = pq.read_table(p)
    return (np.asarray(t.column("t_ms"), dtype="int64"),
            np.asarray(t.column("close"), dtype="float64"))


def cached_days(ticker):
    d = CACHE / ticker.replace(":", "_")
    if not d.exists():
        return []
    return sorted(datetime.date.fromisoformat(p.stem.split("=", 1)[1]) for p in d.glob("dt=*.parquet"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="cache 1-second bars for the realized-vol study")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--years", type=float, default=DEFAULT_YEARS)
    ap.add_argument("--days", type=int)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--chain", action="store_true",
                    help="resolve the futures front-month chain (do this before --build)")
    a = ap.parse_args(argv)
    if a.chain:
        build_chain(years=a.years, days=a.days, resume=not a.no_resume)
    if a.build:
        if any(t in FUTURES_PRODUCTS for t in (a.tickers or ALL_TICKERS)) and not CHAIN_PATH.exists():
            raise SystemExit("[rvol.data] no futures chain yet -- run --chain first, or the "
                             "futures days would all resolve empty and look like holidays")
        ensure_range(a.tickers, years=a.years, days=a.days, resume=not a.no_resume)
    for t in (a.tickers or ALL_TICKERS):
        ds = cached_days(t)
        print(f"  {t:7s} {len(ds):4d} days cached"
              + (f"  {ds[0]} .. {ds[-1]}" if ds else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
