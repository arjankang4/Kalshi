"""Massive option NBBO at Kalshi trade instants -> Parquet.

The study only ever evaluates a theo at the moment a Kalshi trade printed, so
this fetches exactly that: for each ET session, the prevailing option NBBO at
each distinct trade timestamp, for the contracts that back the traded boundaries.

**The adaptive fetch is the whole design.** Per (contract, day) let K be the
number of instants needed and P the number of 50k-quote pages in the tape:

    P == 1                -> tape  (one request, and it serves every instant)
    P <= K                -> tape
    otherwise             -> K as-of queries

P is unknown up front, and that is fine. Tape pages are time-ascending, so a
partial tape resolves a *prefix* of the instants exactly (verified: reading 8 of
31 pages resolved all 12 early-session instants, byte-identical to as-of). So we
start a tape with an abort guard at K+1 pages and fall back to as-of for whatever
is still pending. Worst case is one wasted request; never worse than 2x optimal.

Measured spread that makes this matter: a near-money 0DTE contract carries
1.5-2.3M quotes/session, a far-OTM one 1,255.

Usage:
    python -m storage.massive_backfill --start 2026-05-28 --end 2026-05-29 --no-upload
"""
import argparse
import concurrent.futures as cf
import sys
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import config
from massive import history as mh
from pricing import theo
from storage import paths, parquetio, s3util, schemas
from storage.rest_backfill import _now, already_done, _write_manifest
from storage.units import to_micros

UNDERLYING_OF_SERIES = {
    "KXINX": "SPX", "KXINXU": "SPX", "KXINXDUD": "SPX",
    "KXNASDAQ100": "NDX", "KXNASDAQ100U": "NDX", "KXNASDAQDUD": "NDX",
}

# Strikes either side of the board's centre that get *every* instant, so put-call
# parity always has near-money two-sided pairs to work with. Without these, spot
# at an instant where only a wing market traded would be inferred from wing
# strikes, which is exactly the contamination path theo.parity_spot suffers from.
PARITY_ANCHORS = 3
# Tape pages to attempt before falling back to as-of. Zero -- i.e. as-of only --
# after measuring both sides:
#
#   * Massive sustains 115 req/s (96 concurrent, zero errors), so a request is
#     cheap. The whole board needs ~6,400 as-of fetches a day => ~56 s/day.
#   * A tape page is 50,000 records. Probing even one page per contract means
#     parsing ~9M records/day to save a few hundred requests, and a near-money
#     contract's full tape is 2.3M records on its own. That CPU, not the network,
#     is what made the first two attempts at this take >10 minutes per day.
#
# The tape path and AsOfIndex are kept: they are what the agreement check in
# `verify_asof_agreement` uses to prove the as-of results are right, and they are
# the right tool if a continuous mid grid is ever wanted. Set >0 to re-enable.
MAX_TAPE_PAGES = 0
WORKERS = 192           # I/O-bound; the limiter in massive_history is the throttle
# Full-board snapshot cadence, in minutes. See the grid block in backfill_day.
GRID_MINUTES = 15


# --- board -----------------------------------------------------------------

def strike_grid(contracts, root):
    return sorted({Decimal(str(c["strike_price"])) for c in contracts
                   if c["ticker"][2:-15] == root})


def occ(root, expiry_yymmdd, right, strike):
    return f"O:{root}{expiry_yymmdd}{right}{round(float(strike) * 1000):08d}"


def _grid_instants(d, minutes):
    """Evenly spaced ET session instants for the full-board snapshots."""
    lo, hi = mh.et_session_ns(d)
    step = int(minutes) * 60 * 10**9
    return list(range(lo, hi + 1, step))


def market_strikes(boundaries, grid_floats, min_interval):
    """The strikes needed to price one market's boundaries, both variants.

    theo.neighbors is the live pricer's own function, so the boundary->strike
    mapping cannot drift from production. K+-m covers the min-interval variant.
    """
    listed = set(grid_floats)
    want = set()
    for b in boundaries:
        lo, hi = theo.neighbors(b, grid_floats)
        if lo is not None:
            want.update((lo, hi))
        if min_interval:
            for cand in (b - min_interval, b, b + min_interval):
                if cand in listed:
                    want.add(cand)
    return want


def anchor_strikes(boundaries, grid_floats, n=PARITY_ANCHORS):
    """Near-money strikes for parity spot. The Kalshi 4pm board is centred on the
    expected close by construction, so the median boundary is a good proxy for
    spot without needing spot to find it."""
    if not boundaries or not grid_floats:
        return set()
    centre = sorted(boundaries)[len(boundaries) // 2]
    ordered = sorted(grid_floats, key=lambda s: abs(s - centre))
    return set(ordered[:n * 2])


# --- fetch -----------------------------------------------------------------

def fetch_contract_day(option_ticker, instants_ns, session_ns, page_budget=None):
    """Resolve every instant for one contract. Returns (resolved, mode).

    `page_budget` caps the tape attempt; beyond it we abandon and as-of the
    remainder. The default balances two costs that are not the same thing:
    a request, and the 50,000 records a full page makes us parse. Break-even on
    requests alone is K pages, but a 46-page near-money tape means parsing 2.3M
    records to keep a handful of instants, so the cap also bounds CPU.
    """
    lo, hi = session_ns
    budget = (page_budget if page_budget is not None
              else min(len(instants_ns), MAX_TAPE_PAGES))
    idx = mh.AsOfIndex(instants_ns)
    if budget:
        pages = 0
        try:
            for page in mh.quote_pages(option_ticker, lo, hi, max_pages=budget):
                pages += 1
                if not idx.feed(page):
                    break
            idx.finish()
        except mh.MassiveHistoryError as e:
            print(f"[mv] tape {option_ticker}: {str(e)[:100]}", file=sys.stderr)

    pending = idx.pending()
    if not pending:
        return idx.resolved, "tape"

    # Either the tape was abandoned at the budget or the contract simply has no
    # quote before those instants. One as-of each settles both cases.
    mode = "tape" if idx.resolved else "asof"
    for t in pending:
        try:
            q = mh.quote_asof(option_ticker, t)
        except mh.MassiveHistoryError as e:
            print(f"[mv] asof {option_ticker}@{t}: {str(e)[:80]}", file=sys.stderr)
            continue
        if q is not None:
            idx.resolved[t] = q
            mode = "asof" if mode == "asof" else "tape+asof"
    return idx.resolved, mode


def _price_micros(v):
    """Massive sends floats. repr() is shortest-round-trip, so to_micros' hard
    fail on excess precision still means something."""
    return None if v is None else to_micros(repr(float(v)))


def quote_rows(meta, resolved, mode, instants_ns, ingest):
    for t in instants_ns:
        q = resolved.get(t)
        qts = int(q["sip_timestamp"]) if q else None
        bid = q.get("bid_price") if q else None
        ask = q.get("ask_price") if q else None
        yield {
            "option_ticker": meta["option_ticker"], "underlying": meta["underlying"],
            "root": meta["root"], "expiration_date": meta["expiration_date"],
            "contract_type": meta["contract_type"], "strike": meta["strike"],
            "asof_ts_ns": int(t), "asof_ts": mh.from_ns(t),
            "has_quote": q is not None,
            "had_bid": (bid is not None) if q else None,
            "had_ask": (ask is not None) if q else None,
            "quote_ts_ns": qts,
            "quote_ts": mh.from_ns(qts) if qts is not None else None,
            "quote_age_ns": (int(t) - qts) if qts is not None else None,
            "bid_micros": _price_micros(bid),
            "bid_size_contracts": int(q["bid_size"]) if q and q.get("bid_size") is not None else None,
            "ask_micros": _price_micros(ask),
            "ask_size_contracts": int(q["ask_size"]) if q and q.get("ask_size") is not None else None,
            "bid_exchange": q.get("bid_exchange") if q else None,
            "ask_exchange": q.get("ask_exchange") if q else None,
            "sequence_number": q.get("sequence_number") if q else None,
            "fetch_mode": mode,
            "source": "massive_v3_quotes",
            "source_ts": mh.from_ns(qts) if qts is not None else None,
            "ingest_ts": ingest, "ts_quality": "sip_ns",
            "schema_version": config.SCHEMA_VERSION,
        }


# --- driver ----------------------------------------------------------------

def _read_parquet(table, d, columns=None):
    """Read one partition back from S3.

    Was a bytes-then-io.BytesIO read, which kills the interpreter at teardown
    after the work is already done -- so this job could upload a correct day and
    still exit non-zero. parquetio.read_s3_table carries the measurements.
    """
    return parquetio.read_s3_table(config.S3_BUCKET_HIST,
                                   paths.parquet_key(table, d), columns=columns)


def backfill_day(d, upload=True, resume=True, workers=WORKERS,
                 grid_minutes=GRID_MINUTES):
    table = "option_quotes_asof"
    if resume and upload and already_done(table, d):
        return {"skipped": True}
    ingest = _now()

    trades = _read_parquet("trades" + schemas.SCOPED_SUFFIX, d,
                           ["ticker", "series_ticker", "created_time"])
    markets = _read_parquet("markets" + schemas.SCOPED_SUFFIX, d,
                            ["ticker", "series_ticker", "floor_strike", "cap_strike",
                             "strike_type"])
    if trades is None or markets is None or trades.num_rows == 0:
        print(f"[mv] dt={d} no scoped Kalshi partition yet -- skipping")
        return {"skipped": True}

    tr = trades.to_pylist()
    mk = {m["ticker"]: m for m in markets.to_pylist()}
    session_ns = mh.et_session_ns(d)

    # Floor for the as-of lookback: the start of the ET day *containing the
    # instant*, not of `d`. These events open at 02:30Z -- 22:30 ET the previous
    # evening -- so a trade can legitimately predate d's own ET midnight, and
    # flooring at d would send timestamp.gte past timestamp.lte and 400.
    # Per-instant keeps pre-open quotes while still excluding the day the strike
    # was listed. Memoised: two distinct dates per session at most.
    _floor_cache = {}

    def _floor_ns(t):
        et_date = mh.from_ns(t).astimezone(config.ET).date()
        if et_date not in _floor_cache:
            _floor_cache[et_date] = mh.to_ns(config.et_day_bounds(et_date)[0])
        return _floor_cache[et_date]

    # Group trades by underlying, and by market within it. A contract is only
    # needed at the instants its own market traded -- fetching every contract at
    # every instant is a cross product that costs ~10x more for no extra data.
    per_und = {}
    for t in tr:
        und = UNDERLYING_OF_SERIES.get(t["series_ticker"])
        m = mk.get(t["ticker"])
        if und is None or m is None:
            continue
        ts = t["created_time"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        slot = per_und.setdefault(und, {"by_market": {}, "boundaries": set()})
        slot["by_market"].setdefault(t["ticker"], set()).add(mh.to_ns(ts))
        for f in ("floor_strike", "cap_strike"):
            if m.get(f) is not None:
                slot["boundaries"].add(theo.normalize_boundary(m[f]))

    out_rows, stats = [], {"contracts": 0, "instants": 0, "modes": {}, "req": 0}
    for und, slot in sorted(per_und.items()):
        contracts = mh.contracts_for_expiry(und, d)
        if not contracts:
            print(f"[mv] dt={d} {und}: no contracts listed", file=sys.stderr)
            continue
        root = mh.pm_settled_root(contracts)
        grid = [float(s) for s in strike_grid(contracts, root)]
        if not grid:
            continue
        m_int = min((grid[i + 1] - grid[i] for i in range(len(grid) - 1)), default=0.0)
        all_instants = sorted({t for s in slot["by_market"].values() for t in s})
        stats["instants"] = max(stats["instants"], len(all_instants))

        # strike -> the instants at which some market needing it traded
        need = {}
        for ticker, instants in slot["by_market"].items():
            m = mk[ticker]
            bounds = [theo.normalize_boundary(m[f]) for f in ("floor_strike", "cap_strike")
                      if m.get(f) is not None]
            for s in market_strikes(bounds, grid, m_int):
                need.setdefault(s, set()).update(instants)
        # Parity anchors carry every instant, so spot is always inferable.
        for s in anchor_strikes(sorted(slot["boundaries"]), grid):
            need.setdefault(s, set()).update(all_instants)

        # A coarse grid on which the WHOLE board is priced at once. Trade-driven
        # instants alone can never price a full board -- a trade only pulls the
        # strikes its own market needs -- which would leave the single best
        # global check unavailable: the 30 mutually-exclusive KXINX brackets must
        # sum to 1, and drift away from 1 is what exposes a width, sign or
        # neighbour-selection bug that no single-market comparison would show.
        # Costs ~28% more requests at the default 15 minutes.
        if grid_minutes:
            for t in _grid_instants(d, grid_minutes):
                for s in need:
                    need[s].add(t)

        expiry = f"{d:%y%m%d}"
        jobs = []
        for s, instants in sorted(need.items()):
            for right, ctype in (("C", "call"), ("P", "put")):
                jobs.append(({
                    "option_ticker": occ(root, expiry, right, s),
                    "underlying": und, "root": root,
                    "expiration_date": str(d), "contract_type": ctype,
                    "strike": Decimal(str(s)),
                }, sorted(instants)))
        stats["contracts"] += len(jobs)

        if MAX_TAPE_PAGES:
            def _one(job):
                meta, instants = job
                res, mode = fetch_contract_day(meta["option_ticker"], instants,
                                               session_ns)
                return list(quote_rows(meta, res, mode, instants, ingest))
            with cf.ThreadPoolExecutor(workers) as ex:
                for rows in ex.map(_one, jobs):
                    out_rows.extend(rows)
                    if rows:
                        md = rows[0]["fetch_mode"]
                        stats["modes"][md] = stats["modes"].get(md, 0) + 1
            continue

        # Flat fan-out over (contract, instant). Parallelising per *contract*
        # instead leaves the parity anchors -- one contract with every instant --
        # running hundreds of round trips serially while the pool idles.
        flat = [(mi, meta, t) for mi, (meta, instants) in enumerate(jobs)
                for t in instants]
        stats["req"] += len(flat)
        resolved = [{} for _ in jobs]

        def _fetch(item):
            mi, meta, t = item
            try:
                return mi, t, mh.quote_asof(meta["option_ticker"], t, _floor_ns(t))
            except mh.MassiveHistoryError as e:
                print(f"[mv] asof {meta['option_ticker']}@{t}: {str(e)[:80]}",
                      file=sys.stderr)
                return mi, t, None

        with cf.ThreadPoolExecutor(workers) as ex:
            for mi, t, q in ex.map(_fetch, flat):
                if q is not None:
                    resolved[mi][t] = q
        for mi, (meta, instants) in enumerate(jobs):
            out_rows.extend(quote_rows(meta, resolved[mi], "asof", instants, ingest))
        stats["modes"]["asof"] = stats["modes"].get("asof", 0) + len(jobs)

    if not out_rows:
        print(f"[mv] dt={d} nothing to write")
        return {"rows": 0}

    scratch = paths.scratch_dir("massive_asof", d)
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / "part-00000.snappy.parquet"
    key_cols = schemas.SORT_KEYS[table]
    out_rows.sort(key=lambda r: tuple(str(r.get(c) or "") for c in key_cols))
    digests = parquetio.write_rows(out, table, iter(out_rows), schemas)
    ok, got, meta = parquetio.verify_file(out, schemas.TABLES[table], digests)
    if not ok:
        raise RuntimeError(f"{table} dt={d}: parquet verification failed")
    if upload:
        rec = s3util.put_large(out, config.S3_BUCKET_HIST, paths.parquet_key(table, d))
        _write_manifest(table, d, digests, meta, rec, True, ingest)
    out.unlink(missing_ok=True)

    have = sum(1 for r in out_rows if r["has_quote"])
    print(f"[mv] dt={d} contracts={stats['contracts']:>4} instants={stats['instants']:>4} "
          f"rows={meta['num_rows']:>7} quoted={have/max(1,len(out_rows)):>5.1%} "
          f"modes={stats['modes']} {meta['bytes']/1e6:.1f}MB", flush=True)
    return {"rows": meta["num_rows"]}


def verify_asof_agreement(option_ticker, d, n=25, seed=7):
    """Prove the as-of path against an independent full-tape reconstruction.

    These are two different code paths to the same definition -- one asks the
    server for the last row at or before t, the other streams the tape forward
    and keeps the last row it passed. If they disagree, the tie-break on equal
    sip_timestamp is wrong and every quote in the study is suspect. Cheap enough
    to run before trusting a sweep; use a contract whose tape is small.
    """
    import random
    lo, hi = mh.et_session_ns(d)
    rnd = random.Random(seed)
    instants = sorted(rnd.randrange(lo, hi) for _ in range(n))

    idx = mh.AsOfIndex(instants)
    for page in mh.quote_pages(option_ticker, lo, hi):
        if not idx.feed(page):
            break
    idx.finish()

    agree = mismatch = missing = 0
    for t in instants:
        a, b = idx.resolved.get(t), mh.quote_asof(option_ticker, t)
        if a is None or b is None:
            missing += 1
            continue
        ka = (a["sip_timestamp"], a.get("sequence_number"),
              a.get("bid_price"), a.get("ask_price"),
              a.get("bid_size"), a.get("ask_size"))
        kb = (b["sip_timestamp"], b.get("sequence_number"),
              b.get("bid_price"), b.get("ask_price"),
              b.get("bid_size"), b.get("ask_size"))
        if ka == kb:
            agree += 1
        else:
            mismatch += 1
            print(f"  MISMATCH t={t}\n    tape={ka}\n    asof={kb}", file=sys.stderr)
    print(f"[verify] {option_ticker} {d}: agree={agree} mismatch={mismatch} "
          f"unresolved={missing}")
    return mismatch == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default=None,
                    help="OPTION_TICKER to run the tape-vs-asof agreement check on")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--grid-minutes", type=int, default=GRID_MINUTES,
                    help="full-board snapshot cadence; 0 disables")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--oldest-first", action="store_true")
    a = ap.parse_args(argv)

    end_d = date.fromisoformat(a.end) if a.end else config.today_et_date()
    start_d = (date.fromisoformat(a.start) if a.start
               else end_d - timedelta(days=a.days))

    if a.verify:
        return 0 if verify_asof_agreement(a.verify, start_d) else 1

    days = []
    dd = start_d
    while dd < end_d:
        if dd.weekday() < 5:
            days.append(dd)
        dd += timedelta(days=1)
    if not a.oldest_first:
        days.reverse()

    print(f"[mv] {start_d}..{end_d} ({len(days)} weekdays) workers={a.workers}")
    tot = {"days": 0, "rows": 0, "skipped": 0}
    for d in days:
        t0 = time.time()
        r = backfill_day(d, not a.no_upload, not a.no_resume, a.workers,
                         a.grid_minutes)
        if r.get("skipped"):
            tot["skipped"] += 1
            continue
        tot["days"] += 1
        tot["rows"] += r.get("rows", 0)
    print(f"[mv] DONE days={tot['days']} skipped={tot['skipped']} rows={tot['rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
