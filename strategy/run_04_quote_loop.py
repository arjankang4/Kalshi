"""Stage 4: the continuous quoter -- risk-priced edge, three-rung ladders, deadband requoting.

This runs a real loop, charges edge scaled to each market's measured risk, rests three rungs a
side, and leaves an order alone unless it is actually wrong.

THREE CADENCES, and they are different on purpose:

    quote      QUOTE_INTERVAL_S          book snapshot -> theo -> ladders -> reconcile. The fast path.
    edge       config.EDGE_RECOMPUTE_S   the three dollar Greeks and R per market (strategy/edgemgr).
    lean       config.LEAN_RECOMPUTE_S   realized-vs-implied vol (strategy/leanchain), cached.

The loop sleeps `interval - elapsed`, so its PERIOD is the interval rather than interval-plus-work
-- the drift-compensated pattern from strategy/latency.py. A cycle that overruns simply free-runs.

TIERS AND EDGE ARE ON. `config.TIERS_ENABLED` and `config.EDGE_MANAGER_ENABLED` were both built
to default False -- which made this loop quote one rung a side at `QUOTE_OFFSET`, reproducing the
pre-calibration quoter -- and both were switched on 2026-08-21. This docstring claimed otherwise
until 2026-08-23; check `config.py` rather than this line.

`--edge` off is still reachable and is now an EXPLICIT OVERRIDE: since a market without a usable R
is no longer quoted at all (see strategy/edgemgr), turning edge off would otherwise quote nothing.
It installs a FlatCurve, sets `em.operator_flat`, and prints a banner.

    python -m strategy.run_04_quote_loop --minutes 5              # dry, sends nothing
    python -m strategy.run_04_quote_loop --minutes 5 --tiers --edge
    python -m strategy.run_04_quote_loop --minutes 30 --tiers --edge --live
"""
import argparse
import datetime
import json
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config
from kalshi import client as k
from massive import client as mv
from massive import index as ix
from massive import streams as mstreams
from kalshi import ws as kws
from pricing import atmvol, edge as pedge, lean as pricing_lean, theo as ptheo
from pricing import basis as basismod
from pricing import universe, validate_theo
from pricing import volspike as pvolspike
from storage import feed_tee as ftee
from strategy import fastcancel
from strategy import risktape
from strategy import (band as bandmod, budget, cancel_all, deltaadj, edgemgr,
                      journal as jnl, leanchain, orders as omgr, quoting, risklimit,
                      skewstate)

# Main cycle and `--interval` default. Every market is fully repriced (spot -> forward -> theo ->
# edge -> ladder -> reconcile) once per cycle. This is the theo recompute rate. Illustrative value;
# the live value is not published.
QUOTE_INTERVAL_S = 1.0
# `--warmup` default: seconds to wait for option-quote coverage before the first cycle. Illustrative
# value; the live value is not published.
WARMUP_S = 15.0

# THE STOP EVENT, and why a signal handler is not optional here.
#
# Cancel-on-exit lives in `main`'s `finally` block. Python runs `finally` on a normal return and on
# KeyboardInterrupt, and NOT on SIGTERM -- the default disposition kills the process outright. So
# every way this loop is stopped by anything other than a human at a keyboard -- `systemctl stop`,
# a unit's TimeoutStartSec expiring, a plain `kill`, the box shutting down -- skipped the cancel and
# left GTC orders resting on a live market until it closed. `storage/ws_recorder:679` had handlers
# from the start; the process that sends the orders did not.
#
# The handler only SETS the event. It does not cancel, log or exit: doing real work in a signal
# handler risks re-entering whatever it interrupted, and the cancel path makes network calls. The
# loop notices within one `--interval` and unwinds through the normal path.
_STOP = threading.Event()


def _install_signal_handlers():
    """SIGINT/SIGTERM set the stop event. Idempotent, so a second signal during the cancel sweep
    does not abort it -- an operator hitting Ctrl-C twice because the first looked slow must not be
    the reason orders are left resting."""
    def _handler(signum, _frame):
        if not _STOP.is_set():
            print(f"\n[run_04] signal {signum} -- stopping after this cycle, then cancelling",
                  file=sys.stderr, flush=True)
        _STOP.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except (ValueError, OSError):
            # Not the main thread, or a platform without it. Reported, never fatal.
            print(f"[run_04] *** could not install a handler for {s!r} -- cancel-on-exit is "
                  f"NOT guaranteed on that signal ***", file=sys.stderr)


def discover_boards(daily=None, hourly=None, date=None):
    """(daily_markets, hourly_markets) for today, across every board we quote.

    ONE REQUEST PER SERIES, not one per event. `/markets` takes `series_ticker`, so a single call
    returns every event of a series; fetching per event would be seven requests for KXINXU alone.

    FILTERED BY `min_close_ts`, NOT BY `status`, and this is not a preference. The documented
    `MarketStatusQuery` enum is `[unopened, open, paused, closed, settled]`
    (`kalshi_api.yaml:4286`) but the `status` field on a live market reads `active` or
    `finalized` -- measured 2026-08-25 on three real events. The two vocabularies do not meet:
    `status=open` returns **zero markets** against a board that is plainly trading, and
    `status=active` is rejected outright as an invalid filter. A quoter built on `status=open`
    would have found the hourly boards empty every single cycle and quoted nothing, silently,
    while reporting a healthy run. `min_close_ts` returns exactly what we want -- markets that
    have not closed yet -- in one request: measured 420 markets across 7 events.

    Unfiltered is not an option: `series_ticker=KXINXU` alone returns **40,400 markets across 329
    events**, i.e. every board the series has ever listed, at 41 pages and 410 read tokens.

    MEASURED 2026-08-21: every remaining hour is listed and active simultaneously and the set only
    shrinks through the session (10:21 ET showed H1100-H1600, 11:49 showed H1200-H1600). So this
    returns up to seven hourly events at the open and fewer later, and nothing new appears
    mid-session -- which is what lets the WebSocket subscription be taken once at startup.

    Filtered to TODAY's stamp as well, because `min_close_ts` legitimately carries the NEXT day's
    board once it lists -- measured 2026-08-25, tomorrow's seven events were already active
    alongside today's. Pricing tomorrow's event off today's 0DTE option strip is nonsense rather
    than an error, so it would not announce itself.
    """
    d = date or config.today_et_date()
    stamp = f"{d:%y%b%d}".upper()
    daily = config.QUOTE_DAILY_SERIES if daily is None else daily
    hourly = config.QUOTE_HOURLY_SERIES if hourly is None else hourly

    def _today(ms):
        return [m for m in ms if f"-{stamp}" in (m.get("event_ticker") or m.get("ticker") or "")]

    dm = []
    for sname in daily:
        dm.extend(_today(k.markets(event_ticker=f"{sname}-{stamp}H1600")))
    hm = []
    now_ts = int(time.time())
    for sname in hourly:
        hm.extend(_today(k.markets(series_ticker=sname, min_close_ts=now_ts)))
    return dm, hm


def _index_quotes(underlyings=None, feed=None):
    """{underlying: IndexQuote} for every board we quote, in ONE request.

    Batched deliberately (`massive.index.snapshot` takes `ticker.any_of`): both indices then
    carry the SAME vendor timestamp, so the SPX and NDX spots a cycle prices from cannot be
    minutes apart. An underlying the vendor omits is simply absent from the dict rather than
    present as None -- callers must use `.get()`.
    """
    underlyings = tuple(underlyings or (config.OPTION_UNDERLYING,))
    # OFF THE SOCKET WHEN THERE IS ONE. Measured 2026-08-26, the same index level read over the
    # WebSocket is a median 21.3ms old against 711ms over REST -- the vendor's one-second
    # publication grid is unchanged, but 690ms of the lag we used to add on top of it was our own
    # polling. REST stays as the fallback so a socket that has not warmed up cannot stop a cycle.
    if feed is not None:
        got = feed.snapshot(underlyings)
        if got:
            return got
    try:
        return ix.snapshot(underlyings)
    except Exception as e:                                   # noqa: BLE001
        print(f"[index] snapshot failed ({type(e).__name__}: {str(e)[:80]})", file=sys.stderr)
        return {}


def _sigma(fit, calendar, now):
    """Annualised ATM implied vol on the intraday variance clock, or None.

    Same clock as `HourlyPricer.sigma_annualized` and as `rvol` -- one RTH session is one
    variance-day, 252 a year, and the overnight gap never enters.
    """
    if fit is None or not fit.ok:
        return None
    left = 1.0 - calendar.phi(now)
    if left <= 1e-6:
        return None
    return fit.w / ((left / 252.0) ** 0.5)


# Tunable. Illustrative value; the live value is not published.
BOARD_REFRESH_S = 120.0


def _still_open(markets, now):
    """Drop markets whose `close_time` has passed, without asking the exchange.

    This is what makes a cached market list safe between refreshes. A closed market that stayed in
    the list would be priced, selected and quoted against a board that no longer trades -- and the
    exchange would reject the orders one at a time, which reads in the cycle record like a broken
    quoter rather than a stale list.
    """
    out = []
    for m in markets:
        ct = m.get("close_time")
        if not ct:
            out.append(m)                       # no stamp: keep it, the refresh will settle it
            continue
        try:
            t = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            out.append(m)
            continue
        if t > now:
            out.append(m)
    return out


# --- one underlying's worth of everything -------------------------------------------------
#
# ADDED 2026-08-26, when NDX was turned on. Everything in this class is per UNDERLYING and none
# of it is per hour: the option strip (SPXW vs NDXP), the variance calendar, the realized-vol
# model, the smile, and the futures contract the theo is carried forward on. The loop merges the
# boards only AFTER pricing, because the two things that must stay global are the two that ration
# a shared resource -- the capital budget and the write budget -- and neither of those can be
# decided one underlying at a time.
#
# What is deliberately NOT here, i.e. what stays shared:
#   - the Massive options socket. ONE connection per account, 1000 tickers; SPXW's 210 and NDXP's
#     404 both fit with room to spare (measured 2026-08-26), so the collector spans both strips
#     and `quotes_by_strike` filters by root. Alternating would be worse, not cheaper: a newly
#     subscribed symbol arrives BLIND -- 15 s after re-adding 8 near-money strikes only 2 had
#     quoted -- so every switch buys a multi-second hole in whichever chain we switch to.
#   - the Kalshi market-data socket, the resting book, the risk limit and the journal.
class Board:
    """The per-underlying half of a cycle: its strip, its models, its futures leg, its scratch."""

    __slots__ = ("underlying", "daily_series", "hourly_series", "root", "expiry", "strikes",
                 "tickers", "em", "lc", "sk", "da", "trk", "_disc", "_disc_t",
                 # per-cycle scratch, rewritten every pass and read only within it
                 "daily_m", "hourly_m", "spot_res", "pricer", "hourly_pricer", "sigma",
                 "em_map", "lean_map", "skip", "fast_spot", "fast_detail", "move_scale",
                 "spike")

    def __init__(self, underlying, daily_series, hourly_series):
        self.underlying = underlying
        self.daily_series = tuple(daily_series)
        self.hourly_series = tuple(hourly_series)
        self.root = self.expiry = self.strikes = self.tickers = None
        self.em = self.lc = self.sk = self.da = self.trk = None
        self._disc, self._disc_t = None, None      # cached (daily, hourly) and when it was taken
        self.reset()

    def reset(self):
        self.daily_m, self.hourly_m = [], []
        self.spot_res = self.pricer = self.hourly_pricer = self.sigma = None
        self.fast_spot, self.fast_detail = None, None
        self.move_scale = None
        self.em_map, self.lean_map = {}, {}
        self.skip = None
        self.spike = None

    def discover(self, now, mono, refresh_s=BOARD_REFRESH_S, force=False):
        """(daily, hourly) for this board, from the cache unless it is due a refresh."""
        due = force or self._disc is None or (mono - self._disc_t) >= refresh_s
        if due:
            self._disc, self._disc_t = discover_boards(self.daily_series, self.hourly_series), mono
        d, h = self._disc
        return _still_open(d, now), _still_open(h, now)

    @property
    def markets(self):
        return self.daily_m + self.hourly_m

    @property
    def series(self):
        return self.daily_series + self.hourly_series

    def detail(self):
        return {"underlying": self.underlying, "root": self.root,
                "n_daily": len(self.daily_m), "n_hourly": len(self.hourly_m),
                "spot": None if self.spot_res is None else self.spot_res.value,
                # THE LABEL MUST NAME THE PROVENANCE, not the channel it arrived on. The fast spot
                # is handed to `spot.resolve` AS an IndexQuote -- deliberately, so the parity
                # cross-check and the staleness rule still run -- and `resolve` therefore stamps
                # it `"index"`. On 2026-08-26 that made a reader conclude from 719,734 journal
                # rows that the futures-derived spot was never used, when `spot == fast_spot` on
                # 5,640 of 5,650 board-cycles and differed on none.
                "spot_source": (None if self.spot_res is None else
                                ("fast_futures" if self.fast_spot is not None
                                 else self.spot_res.source)),
                "spot_resolve_source": None if self.spot_res is None else self.spot_res.source,
                "sigma": self.sigma, "skip": self.skip,
                "fast_spot": self.fast_spot,
                "basis": (self.trk.detail() if self.trk is not None else None),
                "fast_reason": (self.fast_detail or {}).get("reason"),
                "move_scale": self.move_scale,
                "edge": self.em.summary() if self.em is not None else None,
                "delta_adj": self.da.detail() if self.da is not None else None,
                "skew": self.sk.detail() if self.sk is not None else None}


class EdgeUnion:
    """`.get(ticker)` across one `EdgeManager` per underlying.

    `build_desired` wants one object it can ask about any ticker on the merged board. Routing by
    ticker rather than merging the caches keeps each manager's own `summary()` and `calendar`
    intact -- they are different calendars, and averaging them would be meaningless.
    """

    def __init__(self, boards):
        self.boards = list(boards)
        self._owner = {}

    def bind(self, boards):
        """Rebuild the ticker -> manager index from this cycle's markets."""
        self._owner = {m["ticker"]: b.em for b in boards for m in b.markets}
        return self

    def get(self, ticker):
        em = self._owner.get(ticker)
        if em is not None:
            return em.get(ticker)
        # Unknown ticker: ask any manager so the caller still gets the well-formed
        # `{"quotable": False, "reason": "unknown_ticker"}` shape rather than a KeyError.
        return self.boards[0].em.get(ticker) if self.boards else {}

    def summary(self):
        return {b.underlying: b.em.summary() for b in self.boards if b.em is not None}


def hourly_cutoff(market, now):
    """(refuse, minutes_left) for the hourly pre-settlement cutoff. See HOURLY_MIN_MINUTES_TO_CLOSE.

    A market is "hourly" when its settlement is not 16:00 ET. Unknown settlement -> not refused
    here; the edge manager already refuses a market it cannot time.
    """
    settle = edgemgr.settlement_of(market)
    if settle is None:
        return False, None
    et = settle.astimezone(config.ET)
    if (et.hour, et.minute) == (16, 0):
        return False, None
    left = (settle - now).total_seconds() / 60.0
    return left < config.HOURLY_MIN_MINUTES_TO_CLOSE, left


def build_desired(markets, fairs, books, em, lean_map, tiers_enabled, bands, positions=None,
                  book=None, mono_now=None, now=None, spike_map=None):
    """{(ticker, side): [Rung]} plus a per-market record of why.

    `spike_map` is `{ticker: LeanChain.short_signal(...)}` -- the near-term vol-spike signal of the
    market's underlying. It is read ONLY when `config.VOLSPIKE_ENABLED`; with the flag off this
    function does not look at it, adds no key to any note, and returns exactly what it returned
    before `pricing/volspike` existed (`strategy.test_volspike` pins that).

    `bands` is `strategy.band.classify`'s per-market record -- it carries this market's rung sizes
    (three in the band, one outside) and its net-position limit. A plain tuple or None is still
    accepted and means "the same counts everywhere", which is what the pure ladder tests pass.

    `positions` is `{ticker: net contracts}`, signed, from `kalshi.ws.MarketFeed`.
    """
    desired, notes = {}, {}
    mono_now = time.monotonic() if mono_now is None else mono_now
    now = datetime.datetime.now(datetime.timezone.utc) if now is None else now
    book = book if book is not None else omgr.RestingBook()
    for m in markets:
        tk = m["ticker"]
        detail = fairs.get(tk) or {}
        fair = detail.get("fair")
        e = em.get(tk)
        entry = books.get(tk) if isinstance(books, dict) else None
        bb, ba, bsz, asz = quoting.top_of_book(entry)
        bd = (bands or {}).get(tk) if isinstance(bands, dict) else None
        counts = (bd or {}).get("counts") or bandmod.resolve_counts(bands, tk)
        note = {"fair": fair, "best_bid": bb, "best_ask": ba,
                "inband": (bd or {}).get("inband"), "sigmas": (bd or {}).get("sigmas"),
                "rungs_wanted": len(counts), **{
            f"edge_{kk}": e.get(kk) for kk in ("R", "edge", "risk_u", "source", "reason")},
            # CARRIED OFF `detail`, and it has to be explicit because `note` is a whitelist.
            # These four were journalled from `notes` from the day they were added and none of
            # them was ever ON `notes`: the delta-adjust block writes them onto the `fairs`
            # entry (see the loop above `CAPITAL SELECTION`), `build_desired` then builds a
            # fresh dict here, and the journal read the fresh one. Every `delta_adj`,
            # `delta_adj_reason`, `f_move` and `theo_vol` ever written to the tape is null --
            # 8,259 of 8,259 rows on 2026-08-26, 309,096 of 309,096 on 2026-08-25.
            #
            # `theo` was never wrong: it reads `note["fair"]`, which IS the adjusted value. What
            # was missing is everything needed to take the adjustment APART -- the unadjusted
            # theo beside it, the size of the carry, and why it was refused when it was. With
            # two underlyings that is no longer an audit nicety: it is the only per-market
            # evidence that NDX is being carried on NQ and SPX on ES.
            **{kk: detail.get(kk) for kk in
               ("fair_vol", "delta_adj", "delta_adj_reason", "f_move")}}
        if fair is None:
            notes[tk] = {**note, "skip": detail.get("reason") or "no_fair"}
            continue
        if not e.get("quotable"):
            notes[tk] = {**note, "skip": e.get("reason") or "not_quotable"}
            continue
        cut, left = hourly_cutoff(m, now)
        if cut:
            notes[tk] = {**note, "skip": f"hourly_within_{config.HOURLY_MIN_MINUTES_TO_CLOSE:g}min",
                         "minutes_to_close": left}
            continue

        # THE NEAR-TERM VOL-SPIKE WIDENING (`pricing/volspike`). Symmetric -- the same extra edge
        # on both sides -- and entirely skipped while VOLSPIKE_ENABLED is False, so the flag-off
        # path is the pre-existing code with `spike_m == 0` and no new note keys.
        spike_m = 0
        if config.VOLSPIKE_ENABLED:
            spike_m, sdet = pvolspike.edge_adjust(((spike_map or {}).get(tk) or {}).get("s"))
            note["spike_s"] = sdet["spike_s"]
            note["spike_cents"] = sdet["spike_cents"]
            note["spike_reason"] = sdet["spike_reason"]
            if sdet["spike_pull"]:
                notes[tk] = {**note, "skip": "volspike_pull"}
                continue

        # THE INVENTORY GATE. `position` is signed net contracts: a bid buys YES and ADDS to it,
        # an ask sells YES and REDUCES it. At the limit we stop quoting the side that would add and
        # keep quoting the side that flattens us -- which is what stops one market picking us off
        # repeatedly on a single modelling disagreement. It is a gate on NEW quotes only; orders
        # already resting are left to `plan_side`, because pulling them is a separate decision that
        # costs queue position.
        pos = float((positions or {}).get(tk) or 0.0)
        limit = (bd or {}).get("position_limit")
        blocked = set()
        if limit:
            if pos >= limit - 1e-9:
                blocked.add("bid")
            if pos <= -limit + 1e-9:
                blocked.add("ask")
        note["position"] = pos
        note["position_limit"] = limit
        if blocked:
            note["position_blocked"] = sorted(blocked)

        rv = (lean_map or {}).get(tk) or {}
        for side in ("bid", "ask"):
            if side in blocked:
                note[f"{side}_rungs"] = 0
                continue
            # THE REALIZED-VOL LEAN, and it moves the required EDGE rather than the price.
            #
            # This replaces the offset-and-clamp mechanism (`lean.side_offset` +
            # `lean.clamp_price`) that stood here until 2026-08-25. That one shifted the quoted
            # PRICE and then hard-clamped it, so the worst case was a level on the wire that the
            # implied model never chose. Shifting the edge requirement instead means the worst
            # case is quoting wider on one side and tighter on the other, off a theo that is still
            # entirely the implied-vol digital. The two old functions remain for the backtest
            # (`backtest/analysis`) and are no longer on the live path.
            #
            # `p_ref` is the VRP-centred reference chain, not the raw implied one -- see
            # `strategy/leanchain.chain`. Comparing against raw implied would trade the variance
            # risk premium as if it were a signal.
            # THE CHARGE, IN THREE PARTS, and only two of them gate the exit.
            #
            #   floor    what we charge to be willing to quote at all (EDGE_FLOOR, 0.5c).
            #   lean     the realized-vol disagreement, per SIDE: -0.5c at most on the aggressive
            #            side, +2c at most on the defensive one, scaled by the disagreement
            #            against a 10-PROBABILITY-POINT clamp.
            #   bonus    what this market's own R earns. 0 at R_min, 1c at R_p99, 5c at R_max.
            #   backoff  a decaying tax on a side we have just been filled on: 1c now, 0.5c at
            #            5s, 0c at 10s. Per (market, side), so flattening is never taxed.
            #
            # TO POST:  floor + lean + bonus + backoff.  TO STAY: edge > 0.
            #
            # A `floor + lean` EXIT bar was built here on 2026-08-26 and removed the same day:
            # backtesting it on that session's book tape found the fills below 0.25-0.50x their
            # posting bar were the ones that made money, and the only band where cutting helped
            # cancelled 40-67% of resting rungs a cycle. See the long note in `orders.plan_side`.
            # The bonus and the backoff remain what we charge for TAKING a position on.
            adj, ldet = pricing_lean.edge_adjust(rv.get("p_ref"), rv.get("p_rv"), side)
            floor_m = int(e.get("floor_micros") or 0)
            bonus_m = int(e.get("bonus_micros") or 0)
            back_m = pricing_lean.backoff_micros(
                None if (ft := book.state(tk).last_fill(side)) is None else (mono_now - ft))
            min_m = max(0, floor_m + adj + spike_m)
            entry_m = max(0, min_m + bonus_m + back_m)
            rungs = quoting.decide_ladder(side, fair, bb, ba, counts,
                                          min_edge_micros=entry_m,
                                          scale_pos=e.get("scale_pos", 0.0),
                                          tiers_enabled=tiers_enabled)
            if rungs:
                desired[(tk, side)] = rungs
            note[f"{side}_rungs"] = len(rungs)
            note[f"{side}_lean"] = ldet.get("lean_reason")
            note[f"{side}_lean_cents"] = ldet.get("lean_cents")
            note[f"{side}_min_edge"] = entry_m
            note[f"{side}_bonus"] = bonus_m
            note[f"{side}_backoff"] = back_m
        notes[tk] = note
    return desired, notes


# --- the verdict ---------------------------------------------------------
#
# A scheduled check that cannot fail is not a check. Until 2026-08-21 this loop returned 0 no
# matter what happened, so "quoted cleanly for twelve minutes", "every cycle threw and was
# caught", and "the market never opened" were indistinguishable from outside -- all three wrote a
# log and exited 0. That is the shape of the us-east-2 cutover failure: a step that quietly did
# not happen reads exactly like a step that succeeded.

PASS, FAIL, NO_SESSION, NO_SPOT = "PASS", "FAIL", "NO_SESSION", "NO_SPOT"

# Grounded in a measured 2-minute dry run: 60 cycles against a 2 s interval (~30/min, i.e. the
# loop kept up), 0 errors, 91 orders resting, 52% quiet.
MIN_CYCLE_FRACTION = 0.5
MAX_ERROR_RATE = 0.10
MIN_SPOT_RATE = 0.5


def verdict(stats, expected_cycles, in_session):
    """(verdict, exit_code, detail) from a finished run. Pure -- no clock, no I/O.

    Kept free of side effects so the four outcomes can be pinned against synthetic stats rather
    than by waiting for a Monday.

    `NO_SESSION` and `NO_SPOT` are the same observation -- spot never resolved -- split by whether
    the market was supposed to be open. On a holiday that is expected and exits 0; inside
    `config.in_capture_session()` it means the feeds are down and must exit non-zero. Holidays are
    deliberately not modelled anywhere in this repo (see config.in_capture_session), so a holiday
    will report NO_SPOT and alarm once. That is the cheap direction to be wrong in.
    """
    n = int(stats.get("cycles") or 0)
    errors = int(stats.get("errors") or 0)
    no_spot = int(stats.get("no_spot") or 0)
    placed = int(stats.get("placed") or 0)
    n_ok = max(0, n - errors)

    d = {"cycles": n, "expected": int(expected_cycles), "errors": errors,
         "error_rate": (errors / n) if n else 1.0,
         "no_spot": no_spot,
         "spot_rate": ((n_ok - no_spot) / n_ok) if n_ok else 0.0,
         "placed": placed, "in_session": bool(in_session), "reasons": []}

    if n == 0:
        d["reasons"].append("no cycles ran")
        return FAIL, 1, d

    if d["spot_rate"] <= 0.0:
        if in_session:
            d["reasons"].append("spot never resolved while inside the session gate")
            return NO_SPOT, 1, d
        d["reasons"].append("market not open")
        return NO_SESSION, 0, d

    if n < MIN_CYCLE_FRACTION * expected_cycles:
        d["reasons"].append(f"only {n} of an expected {int(expected_cycles)} cycles")
    if d["error_rate"] >= MAX_ERROR_RATE:
        d["reasons"].append(f"error rate {d['error_rate']:.0%}")
    if d["spot_rate"] < MIN_SPOT_RATE:
        d["reasons"].append(f"spot resolved on only {d['spot_rate']:.0%} of cycles")
    if placed == 0:
        # A run that never planned an order never exercised the ladder, the edge manager or the
        # order manager. It is clean in the way an unplugged machine is clean.
        d["reasons"].append("no order was ever planned")

    return (PASS, 0, d) if not d["reasons"] else (FAIL, 1, d)


def cycle(state, collector, boards, edges, book, args, jrnl):
    """One pass over every board. Returns a dict of what happened.

    `boards` is a list of `Board`, one per UNDERLYING, each carrying its own strip, calendar,
    edge manager, lean chain, smile and futures leg. `edges` is the `EdgeUnion` over them that
    `build_desired` asks about any ticker. Everything downstream of the merge -- selection, the
    write budget, the risk gate, `reconcile` -- sees one board, because capital and write tokens
    are shared and cannot be rationed one underlying at a time.
    """
    t0 = time.perf_counter()
    now = datetime.datetime.now(config.ET)

    # FILLS FIRST -- ahead of every gate and every early return in this function.
    #
    # A fill is the exchange telling us what we now own. Applying it is never wrong, never depends
    # on being able to price anything, and must not wait on a feed that has nothing to do with it:
    # the gate below is about the MASSIVE option feed, and a Massive outage has no bearing on
    # whether a Kalshi order filled. Draining these after the gate would let fills queue up
    # unapplied for the whole outage, and `RestingBook` would come out the far side describing
    # orders that no longer exist -- measured for edge, counted against the per-side cap, and
    # eventually cancelled by an order_id the exchange had already retired.
    #
    # This is the reconciliation the loop never had: `RestingBook` was written from create
    # responses and not checked against the exchange again until `cancel_all` at exit.
    feed = state.get("mdfeed")
    risk = state.get("risk")
    n_fills = 0
    if feed is not None:
        drained = feed.drain_fills()
        for fl in drained:
            n_fills += omgr.apply_fill(book, fl)
        # Cash is folded from the RAW frames, not from what `apply_fill` matched. A fill for an
        # order this process did not place -- a restart, a hand-entered order, the tail of a
        # previous session -- moves the account's money whether or not `RestingBook` recognises
        # the id, and a loss limit that only counts our own bookkeeping is blind to exactly the
        # positions we did not mean to have.
        if risk is not None:
            risk.on_fills(drained)

    # THE HALT CHECK, ahead of every gate including the feed one. A halt must work on a cycle we
    # would have skipped anyway -- that is the cycle an operator is most likely to be halting on.
    if risk is not None:
        why = risk.halt_reason()
        if why:
            print(f"[HALT] {why} -- stopping and cancelling", file=sys.stderr, flush=True)
            _STOP.set()
            return {"skipped": "halted", "reason": why, "fills": n_fills,
                    "risk": risk.summary()}

    # THE FEED GATE, before anything is read off it.
    #
    # `collector.snapshot()` cannot tell a live board from a frozen one -- a stale quote and a
    # current quote are the same dict -- and the pump thread can die on any socket error. Spot
    # does not catch this, because the index arrives on a separate REST call and stays fresh
    # while the option board is hours old. So the health check comes first, and a dead or stale
    # feed skips the cycle rather than pricing off it.
    #
    # Skipping leaves the existing orders resting. That is the right call at this altitude: the
    # ladder was priced off a board that WAS live, and pulling a whole book on one bad cycle
    # costs more queue position than a few seconds of exposure is worth. `cancel_all` in the
    # `finally` block is the kill path, and an operator watching `feed_stale` in the printed
    # cycle record is the escalation.
    h = collector.health()
    if not h["alive"]:
        okc = collector.reconnect()
        print(f"[feed] pump died ({h['pump_error']}) -- reconnect "
              f"{'ok' if okc else 'FAILED'} (#{collector.reconnects})", file=sys.stderr)
        return {"skipped": "feed_down", "reconnected": okc, "fills": n_fills,
                "feed": collector.health()}
    age = h["quote_age_s"]
    if age is not None and age > config.MASSIVE_MAX_QUOTE_AGE_S:
        print(f"[feed] no quote in {age:.1f}s (> {config.MASSIVE_MAX_QUOTE_AGE_S}s) -- "
              f"not quoting this cycle", file=sys.stderr)
        return {"skipped": "feed_stale", "quote_age_s": round(age, 1), "fills": n_fills,
                "feed": h}

    # The futures anchor is sampled HERE, beside the option snapshot, not later. It is one end of
    # the delta increment and the option board is the other; sampling it after the board has been
    # priced would fold the pricing latency into the "move" and adjust the theo for time that had
    # already passed. See strategy/deltaadj.
    #
    # ONE ANCHOR PER UNDERLYING, and all of them at this instant. SPX is carried forward on ES and
    # NDX on NQ, and the two are not interchangeable: NQ moves about 29,305/7,699 = 3.8x as many
    # POINTS as ES for the same proportional move, while an NDX dollar delta is per NDX point. A
    # single shared adjuster would therefore carry every NDX theo at roughly a quarter of its
    # correct size -- in the right direction, every cycle, and completely silently, because a
    # quarter-sized adjustment is too SMALL to trip MAX_ADJ_DOLLARS.
    for b in boards:
        if b.da is not None:
            b.da.anchor_now()

    # THE BOOK COMES OFF THE WEBSOCKET when one is running, and off REST when it is not.
    #
    # This used to be an unconditional `k.orderbooks(state["tickers"])` every cycle. Two problems
    # at the full universe, and only one of them is about budget: the endpoint caps `tickers` at
    # 100 (`kalshi_api.yaml`, now respected by `client.orderbooks`), and a polled book is only ever
    # as fresh as the poll. The WS book is maintained frame by frame and costs nothing to keep --
    # subscribe once, receive forever. See `kalshi/ws`.
    #
    # REST REMAINS THE FALLBACK, and it is a real one rather than a gesture: at 10 tokens per
    # request flat (measured, `GET /account/endpoint_costs`) a 532-market board is 6 requests and
    # 60 tokens, so falling back for a whole cycle is affordable. We take it when WS coverage
    # drops below the floor -- a gap storm, a reconnect, a warm-up that has not finished -- because
    # quoting a fraction of the board without saying so is the failure mode that looks healthy.
    book_src, cov = "ws", None
    if feed is not None and feed.alive:
        cov = feed.health()["coverage"]
        if cov >= config.WS_BOOK_MIN_COVERAGE:
            books = feed.books(state["tickers"])
        else:
            book_src = "rest_fallback"
    elif feed is not None:
        book_src = "rest_feed_down"
    else:
        book_src = "rest"

    # THE MARKET LIST IS CACHED, not re-fetched every cycle -- see `BOARD_REFRESH_S`. Between
    # refreshes `Board.discover` returns the cached list with closed markets filtered out locally,
    # which costs nothing and is correct because the set only ever shrinks intra-session. The
    # refresh itself still goes in this pool, beside the option snapshot and the index, so the
    # cycle that pays for it pays once.
    mono = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        f_m = {b.underlying: pool.submit(b.discover, now, mono) for b in boards}
        f_q = pool.submit(collector.snapshot)
        f_i = pool.submit(_index_quotes, tuple(b.underlying for b in boards),
                          state.get("indexfeed"))
        f_b = (None if book_src == "ws"
               else pool.submit(lambda: k.orderbooks(state["tickers"], depth=10)))
        snap, iq = f_q.result(), f_i.result()
        for b in boards:
            b.reset()
            b.daily_m, b.hourly_m = f_m[b.underlying].result()
        if f_b is not None:
            books = f_b.result()
    if isinstance(books, dict) and "orderbooks" in books:
        books = books["orderbooks"]
    if book_src != "ws":
        print(f"[book] {book_src}"
              + (f" (ws coverage {cov:.0%} < {config.WS_BOOK_MIN_COVERAGE:.0%})"
                 if cov is not None else "")
              + (f" ({feed.health()['pump_error']})"
                 if feed is not None and feed.pump_error else ""), file=sys.stderr)

    # PRICE EACH UNDERLYING SEPARATELY, THEN MERGE. `resolve_and_price_boards` was already
    # parameterised by (root, expiry, strikes, calendar, skew_state), so this is a loop rather
    # than a rewrite -- every one of those is per underlying and none of them is per hour.
    #
    # A BOARD THAT CANNOT PRICE IS DROPPED ON ITS OWN. Before NDX, `spot_res.value is None`
    # returned `{"skipped": "no_spot"}` for the whole cycle, which was right when there was one
    # underlying and is wrong now: a stale NDXP strip must not stop us quoting SPX. The cycle only
    # skips when EVERY board is out.
    #
    # `b.sk` is a SkewState carried ACROSS cycles, one per underlying (a smile is a fact
    # about SPXW or about NDXP, never about both). Passing it here is what makes the smoother
    # reachable at all: before 2026-08-24 the loop built a Pricer with no curve and refit the skew
    # from scratch every cycle, while `theo.Pricer.skew` claimed otherwise.
    #
    # TWO PRICERS PER BOARD, ONE SPOT RESOLUTION, ONE SKEW OBSERVATION. The 4pm brackets go
    # through `theo.Pricer` (its digital is only valid at 16:00, the one instant a listed contract
    # expires); every hourly ladder goes through `pricing.hourly.HourlyPricer`, which rescales the
    # same 16:00 total vol through the variance calendar to each market's own close_time. ONE
    # HourlyPricer covers all seven open events -- `market_fair` reads each market's own
    # settlement -- so this is two pricer objects per underlying, not eight.
    fairs, markets = {}, []
    for b in boards:
        if not b.markets:
            b.skip = "no_markets"
            continue
        # THE FAST SPOT, when the basis can justify one. `BasisTracker` returns
        # `F_now - EMA(F - index)` -- the futures level carried onto the index's scale, which is
        # the scale KXINX settles on. It is handed in as a substitute INDEX QUOTE rather than
        # forced past `spot.resolve`, so the parity cross-check, the staleness rule and the
        # divergence alarm all still run and still get the last word.
        iq_b = (iq or {}).get(b.underlying)
        fast_spot, fast_d = (None, {"reason": "no_tracker"})
        if b.trk is not None:
            fmid, _fage = (state["futures"].mid(b.underlying)
                           if state.get("futures") is not None else (None, None))
            fast_spot, fast_d = b.trk.spot(fmid)
            if fast_spot is not None and iq_b is not None:
                iq_b = ix.IndexQuote(b.underlying, iq_b.ticker, fast_spot,
                                     time.time_ns(), iq_b.market_status, iq_b.name)
        b.fast_spot, b.fast_detail = fast_spot, fast_d
        pricers, fairs_b, res = validate_theo.resolve_and_price_boards(
            b.daily_m, b.hourly_m, snap, b.root, b.expiry, b.strikes,
            now=now, calendar=b.em.calendar, index_quote=iq_b,
            skew_state=b.sk)
        b.pricer, b.spot_res = pricers["daily"], res
        b.hourly_pricer = pricers["hourly"]
        if res.value is None:
            b.skip = f"no_spot:{res.reason}"
            if state.get("guard") is not None:
                state["guard"].retire(b.underlying)
            print(f"[spot:{b.underlying}] {res.reason} -- board not quoted this cycle",
                  file=sys.stderr)
            continue
        fairs.update({tk: v[1] for tk, v in fairs_b.items()})
        markets.extend(b.markets)
    live_boards = [b for b in boards if not b.skip]
    if not live_boards:
        return {"skipped": "no_spot", "fills": n_fills,
                "reason": ";".join(f"{b.underlying}={b.skip}" for b in boards)}

    # Roll the parity band to follow spot. Ported from run_03 on 2026-08-23 when that stage was
    # deleted: it was the ONLY caller of `recenter` in the repo, so without this run_04 held the
    # subscription it seeded at startup from a board-midpoint proxy and the +-2.5% band drifted
    # off spot for the whole session with no correction.
    #
    # ONE re-centre over the UNION of both strips, because there is one socket. The collector
    # diffs against what it currently holds, so handing it the union is the only way to express
    # "keep SPXW where it is and move NDXP" -- handing it one underlying's list would unsubscribe
    # the other's entirely.
    #
    # `b.daily_m`, NOT `b.markets`, and it must match the seed in `main` -- see the long note
    # there on why strikes are pinned from the daily board only. Passing the hourly rungs through
    # `bracketing_strikes` pins one strike pair per rung: measured 2026-08-26, NDX daily-only pins
    # 58 strikes and NDX daily+hourly pins 318, which is 636 tickers on its own and squeezes the
    # parity band to nothing. And it does not degrade -- `desired_tickers` never sacrifices the
    # pinned set, so it returns an over-cap list, `QuoteCollector.recenter` raises, the handler
    # below swallows it, and the subscription FREEZES at the seed for the rest of the session with
    # the parity spot no longer following spot. It never fired only because every session so far
    # ran n_hourly=0.
    #
    # Rate-limited inside the collector (RECENTER_MIN_INTERVAL), so calling it every cycle is
    # safe -- it returns None and does nothing if the last re-centre was recent. The bracketing
    # strikes are pinned, so this can never unprice a market; the worst case is a briefly thinner
    # spot anchor while the new strikes wait for their first tick.
    recentered = None
    try:
        want = []
        for b in live_boards:
            _, w = universe.desired_tickers(b.daily_m, b.strikes, b.root, b.expiry,
                                            b.spot_res.value)
            want.extend(w)
        if want:
            recentered = collector.recenter(want)
    except Exception as e:                           # noqa: BLE001
        # A failed re-centre must never take the cycle down: the existing subscription is still
        # valid, just centred on a staler spot.
        print(f"[recenter] skipped ({type(e).__name__}: {str(e)[:80]})", file=sys.stderr)

    # THE VOL, THE EDGE MANAGER, THE LEAN AND THE DELTA CARRY -- all per underlying, in one pass
    # so that each board's `quotes_by_strike` is computed once and shared by the three readers.
    for b in live_boards:
        quotes = universe.quotes_by_strike(snap, b.root, b.expiry, b.strikes)
        fit = atmvol.fit(quotes, b.strikes)
        b.sigma = _sigma(fit, b.em.calendar, now) if b.em.calendar else None

        # close_time normalisation lives in edgemgr.settlement_of -- see the note there on why it
        # is not done here.
        # THE ADVERSE-SELECTION GATE's input: the futures move we expect over the window between
        # a tick moving and our cancel landing. Estimated live from the feed's own recent mid
        # changes, so it follows the regime rather than a constant fitted once. None -- a feed
        # too thin to estimate from, or the gate switched off -- leaves every market quotable,
        # which is the pre-2026-08-26 behaviour.
        b.move_scale = None
        if (getattr(config, "ADVERSE_GATE_ENABLED", False)
                and state.get("futures") is not None):
            b.move_scale = state["futures"].move_scale(
                b.underlying, config.QUOTE_EXPOSURE_WINDOW_S, config.ADVERSE_MOVE_SIGMAS)
        b.em_map = b.em.refresh(b.markets, b.spot_res.value, b.sigma, now,
                                move_scale=b.move_scale)

        if b.lc is not None and b.lc.ok:
            try:
                b.lc.observe((iq or {}).get(b.underlying))
                # The SAME curve the pricer installed -- `pricer.skew` is the smoothed SkewState
                # fit when there is one. Comparing a smiled implied chain against a flat realized
                # one would book the smile as a disagreement, worst exactly in the wings.
                b.lean_map = b.lc.chain(b.markets, quotes, b.strikes, now, b.spot_res.value,
                                        skew_fit=getattr(b.pricer, "skew", None))
            except Exception as e:                           # noqa: BLE001
                print(f"[lean:{b.underlying}] {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)

        # THE NEAR-TERM VOL-SPIKE SIGNAL, computed and journalled every cycle whether or not
        # VOLSPIKE_ENABLED is set -- the evidence has to accumulate before anything is enabled.
        # `short_signal` never raises; it rides the lean's 5s feature cache.
        b.spike = b.lc.short_signal(now, sigma_impl=b.sigma) if b.lc is not None else None
        vsj = state.get("vsjrnl")
        if vsj is not None and b.spike is not None:
            vsj.write({"v": 1, "ts_ns": time.time_ns(), "underlying": b.underlying,
                       "enabled": config.VOLSPIKE_ENABLED, **b.spike,
                       "would": pvolspike.edge_adjust(b.spike.get("s"))[1]})

        # DELTA-ADJUST, between the edge manager and the ladder. It has to be here and nowhere
        # else: the dollar delta comes from `em_map` (so the edge manager must have run), and
        # every consumer downstream -- the ladder, the requote deadband, `book.state().theo` --
        # has to see ONE theo. Adjusting after the ladder was decided would leave `plan_side`
        # measuring resting orders against a theo the ladder never used.
        #
        # `fairs` is the {ticker: detail} the rest of the cycle reads, so the adjusted value
        # replaces `detail["fair"]` in place and the unadjusted one is kept beside it for the
        # journal. `b.da` is this underlying's adjuster and no other's.
        # THE DELTA ADJUSTMENT IS OFF WHEN THE FAST SPOT IS ON, and this is a correctness point
        # rather than a tidy-up. Both carry theo forward on the SAME futures move: the adjustment
        # approximates it as `delta * (F_now - F_anchor)`, and the fast spot has already put that
        # move into the forward the board was just priced from. Running both would count it twice.
        if b.da is not None and b.fast_spot is None:
            for m in b.markets:
                tk = m["ticker"]
                detail = fairs.get(tk)
                if not detail:
                    continue
                adj_fair, adet = b.da.adjust(detail.get("fair"),
                                             (b.em_map.get(tk) or {}).get("delta"))
                detail["fair_vol"] = detail.get("fair")      # the pure option-snapshot theo
                detail["fair"] = adj_fair
                detail["delta_adj"] = adet["delta_adj"]
                detail["delta_adj_reason"] = adet["reason"]
                detail["f_move"] = adet["f_move"]

    # MERGED VIEWS. From here the cycle is one board again, because the two resources still to be
    # rationed -- capital and write tokens -- are shared and cannot be divided per underlying
    # without deciding the split in advance.
    em_map, lean_map, owner, spike_map = {}, {}, {}, {}
    for b in live_boards:
        em_map.update(b.em_map)
        lean_map.update(b.lean_map)
        owner.update({m["ticker"]: b for m in b.markets})
        spike_map.update({m["ticker"]: b.spike for m in b.markets})
    em = edges.bind(live_boards)

    # HAND THIS CYCLE'S PRICERS TO THE FAST GUARD. After pricing, before any order is planned: the
    # guard measures its increment from the forward the board was actually priced at, so
    # publishing earlier would give it an origin the board never used, and publishing later would
    # leave it acting on the previous cycle's forward while this cycle's orders go out.
    guard = state.get("guard")
    if guard is not None:
        for b in live_boards:
            if b.fast_spot is not None:
                guard.publish(b.underlying, b.pricer, b.hourly_pricer,
                              b.daily_m, b.hourly_m, b.fast_spot)
            else:
                guard.retire(b.underlying)

    # CAPITAL SELECTION, and it is the constraint that binds hardest -- far harder than the write
    # budget below. A two-sided ladder ties up its full face value whatever the fair is (a bid at p
    # risks p, an ask at p risks 1-p), so MAX_RESTING_ORDER_VALUE and MAX_RESTING_CONTRACTS_TOTAL
    # buy only a few dozen of the several hundred SPX/NDX markets listed.
    #
    # Without this the loop plans a ~3,200-order board every cycle, `within_risk_limits` rejects
    # essentially all of it, and nothing is ever quoted -- a failure that would look, in the
    # cycle record, exactly like a quiet market. See `strategy/budget.select_markets`.
    #
    # INCUMBENTS are the markets we already hold, and they are passed in so the selector can
    # apply hysteresis: R refreshes every EDGE_RECOMPUTE_S and a market at the boundary would
    # otherwise flip in and out every 30 s, each flip costing a six-order ladder and every queue
    # position on it.
    sel_theos = {m["ticker"]: (fairs.get(m["ticker"]) or {}).get("fair") for m in markets}
    incumbents = {t for t in book.tickers() if book.contracts(t) > 0}

    # THE BAND, first: it decides each market's SHAPE (three rungs a side or one) and therefore
    # what it costs, which selection needs before it can ration anything. Defined in
    # sigma-moneyness rather than in strikes because a strike count is not a constant width --
    # measured on these two boards, +-3 strikes is 3.99 sigma on the 4pm brackets and 0.80 sigma on
    # the hourly rungs. See `strategy/band`.
    #
    # The incumbent set is the same one selection uses, so a market already quoted in depth is held
    # to the wider exit boundary and does not surrender its extra rungs on a wobble of the forward.
    #
    # PER UNDERLYING, because `classify` takes ONE forward and sigma-moneyness is measured
    # against it. Handing it a merged board with SPX's spot would put every NDX market roughly
    # 21,000 points away from the money and classify the whole Nasdaq board as wing.
    bands, bsum = {}, {}
    for b in live_boards:
        bb_, bs_ = bandmod.classify(b.markets, b.spot_res.value, b.em_map,
                                    incumbents=incumbents)
        bands.update(bb_)
        bsum[b.underlying] = bs_
    counts_map = {t: v["counts"] for t, v in bands.items()}

    chosen, sdet = budget.select_markets(markets, sel_theos, em_map,
                                         counts=counts_map,
                                         incumbents=incumbents)

    # A market we hold but no longer choose is NOT simply forgotten. It is absent from `desired`,
    # so `reconcile` plans `no_longer_quoting` cancels for it -- which are exposure-reducing, cost
    # 2 tokens, and are never deferred by the budget. Exiting is always affordable.
    positions = feed.positions() if feed is not None else {}

    # MARK THE BOOK. The loss limit reads ACCOUNT equity (REST balance + REST positions, polled by
    # `risklimit.AccountPoller`) marked to the Kalshi MID, against the day's opening equity -- see
    # `strategy/risklimit` for why the old fill-cash mark to our own theo read +$1,744.50 on a
    # -$899.05 day. It is what the loss limit reads on the NEXT cycle's halt check.
    if risk is not None:
        acct = state.get("acct")
        snap = acct.latest() if acct is not None else None
        held = snap[2] if snap else {}
        tops = {}
        for tk in held:
            ent = books.get(tk) if isinstance(books, dict) else None
            if ent is None and feed is not None:
                ent = feed.entry(tk)
            bb, ba, _, _ = quoting.top_of_book(ent)
            tops[tk] = (bb, ba)
        risk.mark_account(snap, tops, {t: (d or {}).get("fair") for t, d in fairs.items()})

    # THE RISK TABLE. Here because this is the first point where positions, the Greeks and both
    # vol views all exist for the same instant -- earlier and the positions are last cycle's,
    # later and the row would be stamped after orders moved. One row per underlying per second.
    rt = state.get("risktape")
    if rt is not None and rt.due():
        for b in live_boards:
            rv = next((r for r in (b.lean_map or {}).values() if r.get("sigma_rv")), {})
            try:
                rt.snapshot(b.underlying, now, positions, b.em_map, b.markets, b.pricer,
                            b.em.calendar, sigma_impl=b.sigma, sigma_rv=rv.get("sigma_rv"),
                            spot=b.spot_res.value, fast_spot=b.fast_spot,
                            hourly_markets=b.hourly_m)
            except Exception as e:                           # noqa: BLE001
                print(f"[risktape:{b.underlying}] {type(e).__name__}: {str(e)[:80]}",
                      file=sys.stderr)

    # RE-PRICE WHAT WE ARE ABOUT TO POST, off the spot as of NOW.
    #
    # Everything above this line -- the vol fit, the greeks, the edge bar, the band, the capital
    # selection -- tolerates a spot that is a cycle old, because none of it moves materially on a
    # tenth of a point. The LADDER PRICE does not: it is the number on the wire, and until
    # 2026-08-26 it was computed from a spot read before ~150ms of that work (measured: em.refresh
    # 89ms + lc.chain 105ms + the rest), so an order became liftable carrying a spot ~225ms old
    # against a feed that delivers at 74ms.
    #
    # This is not "price without spot", which is impossible -- the digital IS a function of spot.
    # It separates VOL from SPOT: hold the slow leg, restamp the fast one. `strategy/deltaadj`
    # states the principle in its first line and only ever applied it to the theo.
    #
    # A FULL RE-PRICE, not `delta * dF`. Measured 2026-08-26: 1.31ms (NDX) + 2.16ms (SPX) for 60
    # markets with vol held fixed, and the fast guard does the same call at 0.53ms p50 two hundred
    # times a second. For ~3ms we get the exact number -- smile slope and gamma included -- rather
    # than a first-order approximation that drops precisely the terms that matter at the money,
    # which is where the money was lost.
    # Gated on the FAST SPOT, not on the guard: the guard is a separate control arm and turning
    # it off must not silently take the posted price back to a 225ms-old spot.
    n_repriced = 0
    reprice_ms = 0.0
    if state.get("futures") is not None and not args.no_reprice_at_send:
        t_rp = time.perf_counter()
        for b in live_boards:
            hourly_tk = {m["ticker"] for m in b.hourly_m}
            mine = [(m, (b.hourly_pricer if m["ticker"] in hourly_tk else b.pricer))
                    for m in b.markets if m["ticker"] in chosen]
            if not mine:
                continue
            fmid, _ = (state["futures"].mid(b.underlying)
                       if state.get("futures") is not None else (None, None))
            spot_now, _sd = (b.trk.spot(fmid) if b.trk is not None else (None, None))
            if spot_now is None or b.fast_spot is None:
                continue
            fresh = fastcancel.reprice_shifted(
                b.pricer, b.hourly_pricer,
                getattr(b.pricer, "spot", None),
                (b.hourly_pricer.fit.forward if b.hourly_pricer is not None
                 and b.hourly_pricer.fit else None),
                spot_now - b.fast_spot, mine)
            for tk, fair in fresh.items():
                d = fairs.get(tk)
                if d is not None:
                    d["fair_at_price"] = d.get("fair")   # what the slow leg said, for the tape
                    d["fair"] = fair
                    n_repriced += 1
        reprice_ms = (time.perf_counter() - t_rp) * 1000.0

    desired, notes = build_desired([m for m in markets if m["ticker"] in chosen],
                                   fairs, books, em, lean_map,
                                   args.tiers, bands, positions=positions, book=book,
                                   spike_map=spike_map)
    # The manager decides against the theo we most recently priced each market at, so stamp it
    # before reconciling -- that theo is what `plan_side` measures every resting order's edge
    # against, and a stale one would call a live order negative-edge that is not.
    #
    # Stamped for EVERY market, not just the chosen ones: a market we are exiting still needs a
    # current theo, or `plan_side` measures its resting orders against a stale one and can call a
    # live order negative-edge that is not.
    #
    # THE TOUCH IS STAMPED IN THE SAME PASS, off the same `books` snapshot `build_desired` priced
    # the ladder from. The zombie hold's exception asks whether somebody has posted inside our best
    # resting order (`orders.zombie_held`); answering it from a book read at a different moment
    # would lift the hold for a market state that never existed.
    for m in markets:
        tk = m["ticker"]
        d = fairs.get(tk) or {}
        if d.get("fair") is not None:
            book.state(tk).theo = d["fair"]
        bb_s, ba_s, _, _ = quoting.top_of_book(books.get(tk) if isinstance(books, dict) else None)
        book.state(tk).set_touch(bb_s, ba_s)
    place, cancel, kept = omgr.reconcile(book, desired, counts=counts_map)

    # WRITE BUDGET, between the plan and the risk gate. Kalshi bills writes per item -- 10 tokens
    # per create, 2 per cancel -- against a 900-token bucket refilling at 300/s, so a 2 s cycle
    # can afford ~50 requote pairs. The full SPX/NDX universe plans up to 3,192 orders. Something
    # has to choose, and before this it was the exchange: `apply` sent every chunk unmetered and
    # the back half 429'd by construction, after its cancels had gone out and before their places
    # had. See `strategy/budget`.
    #
    # BEFORE the risk gate, not after, and the ordering is deliberate. Admission only ever drops
    # whole units -- a cancel with its place, or a solo top-up -- so every plan it hands on is
    # exposure-neutral or exposure-reducing relative to the one it received. Risk-checking the
    # smaller plan is therefore both correct and cheaper. Doing it the other way would risk-check
    # orders that were never going to be sent.
    theos = {t: book.state(t).theo for t in book.tickers()}
    theos.update({m["ticker"]: (fairs.get(m["ticker"]) or {}).get("fair") for m in markets})
    place, cancel, bdet = budget.admit(place, cancel, k.write_bucket.available(),
                                       theos=theos, em_map=em_map,
                                       resting=book.resting_sides())
    if bdet["n_deferred"]:
        print(f"[budget] {bdet['n_admitted']} admitted / {bdet['n_deferred']} deferred "
              f"({bdet['deferred_cost']:.0f} tokens of work), "
              f"{bdet['tokens_left']:.0f} left: {bdet['deferred_by_class']}", file=sys.stderr)
    if bdet["reduce_over_budget"]:
        # Not a deferral -- they were all sent. But the bucket is now in debt to the exchange and
        # the next write is the one that 429s, so this is the line an operator needs to see.
        print(f"[budget] ALARM exposure-reducing cancels alone cost "
              f"{bdet['reduce_cost']:.0f} tokens against {bdet['tokens_in']:.0f} available",
              file=sys.stderr)

    # The cancels go in too: `apply` sends them FIRST, so what will actually rest is
    # held - cancelled + placed. Checking only the places double-counts every order being
    # replaced, and since MAX_RESTING_CONTRACTS_PER_MARKET_SIDE is exactly one 3-tier ladder,
    # a side quoting its full ladder then trips on any requote at all.
    ok, violations = omgr.within_risk_limits(book, place, cancel)
    if not ok:
        # Drop the replacement cancels with the places they belonged to. Sending them alone
        # unquotes the side: an `improve_1t` cancel exists only so a better-priced order can
        # take its slot, and if that order is blocked the right move is to leave the resting one
        # where it is. Only `edge_negative` and `no_longer_quoting` survive -- both REDUCE
        # exposure, which is the correct direction when a risk limit has tripped.
        kept_cancels = omgr.reducing_only(cancel)
        print(f"[risk] BLOCKED {len(place)} orders and "
              f"{len(cancel) - len(kept_cancels)} replacement cancels "
              f"({len(kept_cancels)} reducing cancels still sent): {violations[:3]}",
              file=sys.stderr)
        place, cancel = [], kept_cancels
    # UNDER THE SAME LOCK THE GUARD TAKES. Both threads send cancels and both mutate the resting
    # book, so the sends are serialised. Only `apply` is inside it, never `reconcile` -- holding it
    # across the whole plan would block the guard for most of a cycle, which is the one thing it
    # exists not to do. The residual race is benign and already handled: an order the guard pulled
    # a millisecond earlier comes back `not_found`, which `apply` counts as `cancel_gone` and
    # removes from the local book rather than treating as a failure.
    with state["apply_lock"]:
        res = omgr.apply(place, cancel, book, live=args.live, max_batch=state.get("max_batch"),
                         budget=k.write_bucket)

    # SAY WHAT THE EXCHANGE SAID. `apply` has collected `place_errors`/`cancel_errors` since it
    # was written and nothing ever read them, so the only signal a refusal produced was the count
    # in the cycle record. On 2026-08-26 that meant a live session sending ~60 orders a cycle,
    # having every batch refused, and reporting `placed: 0, errors: 1` -- with the status and body
    # already in hand and discarded. Deduplicated by message, because sixty identical rejections
    # are one fact.
    for kind in ("place_errors", "cancel_errors"):
        seen = {}
        for e in res[kind]:
            msg = e.get("body") or e.get("error") or e.get("batch_error") or e.get("batch_shape")
            key = str(msg)[:200]
            seen[key] = seen.get(key, 0) + int(e.get("n") or 1)
        for msg, n in seen.items():
            print(f"[{kind[:-7]}] x{n} {msg}", file=sys.stderr)

    if jrnl is not None:
        for (tk, side), rungs in desired.items():
            d = notes.get(tk) or {}
            ob = owner.get(tk)
            if ob is None:                       # cannot happen; a row with no board is a bug
                continue
            for r in rungs:
                jrnl.write({"v": 2, "ts_ns": time.time_ns(), "live": args.live,
                            "ticker": tk, "side": side, "rung": r.index,
                            "price": r.price, "count": r.count, "edge": r.edge,
                            # PER-RUNG placement bar. `min_edge` below is the side-level one, i.e.
                            # rank 0's; `scale_pos` is not journalled, so without this a replay
                            # cannot reconstruct what rungs 1 and 2 were placed against.
                            "rung_min_edge": r.min_edge,
                            "rule": r.reason, "theo": d.get("fair"),
                            "best_bid": d.get("best_bid"), "best_ask": d.get("best_ask"),
                            "R": d.get("edge_R"), "required_edge": d.get("edge_edge"),
                            "risk_u": d.get("edge_risk_u"), "edge_source": d.get("edge_source"),
                            "theo_vol": d.get("fair_vol"),
                            # THE LEAN, journalled. `docs/PIPELINE.md` claimed since the lean was
                            # written that "the shadow chain is computed and journalled every
                            # cycle, so evidence accumulates before anything is acted on" -- and
                            # it was computed and never journalled. So there is NO recorded
                            # history of what the realized-vol view would have done, which is
                            # exactly the evidence needed to judge whether to trust it. Fixed
                            # 2026-08-25; from here the record accumulates.
                            # `p_rv` used to be read off `notes`, which never carried it, so
                            # it was null on every row. It belongs with its five siblings on the
                            # lean row -- so the walrus binds here, at the first of the six.
                            "p_rv": (rv_row := (lean_map or {}).get(tk) or {}).get("p_rv"),
                            "p_rv_shadow": rv_row.get("p_rv_shadow"),
                            "p_impl_ref": rv_row.get("p_ref"),
                            "sigma_rv": rv_row.get("sigma_rv"),
                            "rv_horizon": rv_row.get("rv_horizon"),
                            "lean_reason": rv_row.get("reason"),
                            "lean_cents": d.get(f"{side}_lean_cents"),
                            "min_edge": d.get(f"{side}_min_edge"),
                            # THE BAR, TAKEN APART -- `min_edge` is the total it cost to
                            # POST, these are its parts. Journalled because the afternoon's
                            # analysis could not tell an order that decayed from one that was
                            # never good, and had to reconstruct the bar by joining back to the
                            # quote record (which landed on the wrong rung: see the notebook).
                            "bonus_edge": d.get(f"{side}_bonus"),
                            "backoff_edge": d.get(f"{side}_backoff"),
                            "delta_adj": d.get("delta_adj"),
                            "delta_adj_reason": d.get("delta_adj_reason"),
                            "f_move": d.get("f_move"),
                            # FROM THE OWNING BOARD, not from a cycle-level variable. Four
                            # boards share one tape, and an SPX spot stamped on an NDX row would
                            # be worse than no spot at all -- it reads as a number.
                            "underlying": ob.underlying,
                            "spot": ob.spot_res.value,
                            # `fast_futures` when the futures-derived spot was substituted; see
                            # `Board.detail`. `spot.resolve` calls it "index" because that is the
                            # slot it arrived in, which is not the same question.
                            "spot_source": ("fast_futures" if ob.fast_spot is not None
                                            else ob.spot_res.source),
                            # The A/B for skew smoothing is a diff of `theo` per
                            # (ticker, rung) between a run with this flag and one without.
                            "sigma": ob.sigma,
                            "skew_smoothed": ob.sk is not None})

    return {"ms": round((time.perf_counter() - t0) * 1000, 1),
            "n_markets": len(markets),
            "n_daily": sum(len(b.daily_m) for b in boards),
            "n_hourly": sum(len(b.hourly_m) for b in boards),
            "n_chosen": len(chosen), "n_desired": len(desired),
            "placed": res["placed"], "cancelled": res["cancelled"], "kept": len(kept),
            "resting": len(book), "contracts": book.contracts(),
            # ONE ENTRY PER UNDERLYING. `spot`, `sigma`, `edge`, `delta_adj` and `skew` were
            # all scalars when there was one board and are all per-underlying now; collapsing
            # them to a single number would silently report SPX's and call it the cycle's.
            "boards": {b.underlying: b.detail() for b in boards},
            # None = rate-limited (refused), (0, 0) = ran with nothing to do, (a, r) = moved.
            "recentered": recentered,
            "select": sdet,
            "band": bsum,
            "positions_held": sum(1 for v in positions.values() if v),
            "book_src": book_src,
            "ws_coverage": (None if cov is None else round(cov, 3)),
            "fills": n_fills,
            "pnl": (None if risk is None or risk.pnl is None else round(risk.pnl, 2)),
            "budget": bdet,
            "bucket": k.write_bucket.detail(),
            # Rate limiting is broken out from the generic error count on purpose. It used to be
            # indistinguishable from any other exception here and fed MAX_ERROR_RATE anonymously,
            # so a cycle throttled by our own budget read exactly like a cycle that crashed.
            "rate_limited": res["rate_limited"],
            # Cancels the exchange had already retired -- filled, or gone another way. Not an
            # error and not a cancellation; broken out because a rising count is the signature of
            # fills landing faster than the fill feed applies them.
            "cancel_gone": res["cancel_gone"],
            "repriced_at_send": n_repriced,
            "reprice_ms": round(reprice_ms, 3),
            # The tape's own health. `dropped` rising means the writer is behind the socket and
            # the day's book is no longer complete -- which a replay cannot tell you afterwards.
            "tape": (state["tee"].detail() if state.get("tee") is not None else None),
            "spot_tape": (state["spot_tape"].detail()
                          if state.get("spot_tape") is not None else None),
            "guard": (state["guard"].detail() if state.get("guard") is not None else None),
            "feeds": {"futures": (state["futures"].health()
                                  if state.get("futures") is not None else None),
                      "index": (state["indexfeed"].health()
                                if state.get("indexfeed") is not None else None)},
            "unsent": res["unsent_cancels"] + res["unsent_places"],
            "orphaned_places": res["orphaned_places"],
            "errors": len(res["place_errors"]) + len(res["cancel_errors"])}


def startup_reconcile(live, adopt=False):
    """Deal with orders already resting before this process starts. Returns a summary dict.

    THE HOLE THIS CLOSES. `RestingBook` starts empty and until 2026-08-25 nothing checked that
    against the exchange. A restart -- a crash, a redeploy, a second window someone opened --
    therefore laddered on top of orders it could not see: invisible to `reconcile`, uncounted by
    `within_risk_limits`, and unswept until `cancel_all` at exit. The per-side cap is EXACTLY one
    ladder, so two ladders is twice the intended size, at a price a previous process's model chose.

    CANCELLING IS THE DEFAULT, NOT ADOPTING. Adopting an order means inheriting a position this
    model never took, at a price it never named. `--adopt-resting` is for the operator who knows
    better, and it deliberately does NOT teach the book about them: they stay outside the risk
    arithmetic, which is exactly why it is not the default.

    A read that FAILS does not stop the run -- it is announced as running blind, because refusing
    to start on a transient `/portfolio/orders` error would be a worse failure than starting. A
    live cancel that leaves anything behind DOES stop it: quoting on top of orders we could not
    pull is the specific outcome this function exists to prevent.
    """
    try:
        events = cancel_all.today_events()
        pre = cancel_all.resting_across(events)
    except Exception as e:                                   # noqa: BLE001
        print(f"[run_04] *** could not read resting orders at startup "
              f"({type(e).__name__}: {str(e)[:90]}) -- proceeding BLIND to anything already "
              f"resting ***", file=sys.stderr)
        return {"read": False, "found": None, "action": "blind", "left": None}
    if not pre:
        print("[run_04] startup: nothing resting")
        return {"read": True, "found": 0, "action": "none", "left": 0}

    print(f"[run_04] startup: {len(pre)} order(s) already resting across today's events")
    if adopt:
        print("[run_04] *** --adopt-resting: leaving them. They are NOT in the local book, so "
              "they do not count against the risk limits and will not be requoted. ***",
              file=sys.stderr)
        return {"read": True, "found": len(pre), "action": "adopted", "left": len(pre)}
    if not live:
        print("[run_04] *** dry run: NOT cancelling them. A --live run would have. ***",
              file=sys.stderr)
        return {"read": True, "found": len(pre), "action": "dry_noop", "left": len(pre)}

    left = cancel_all.cancel_events(events)
    print(f"[run_04] startup: cancelled, {left} still resting")
    if left:
        raise SystemExit("startup cancel left orders resting -- refusing to quote on top of "
                         "them; run  python -m strategy.cancel_all")
    return {"read": True, "found": len(pre), "action": "cancelled", "left": 0}


def main(argv=None):
    ap = argparse.ArgumentParser(description="continuous risk-priced quoter")
    ap.add_argument("--live", action="store_true", help="send real orders")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--interval", type=float, default=QUOTE_INTERVAL_S)
    ap.add_argument("--warmup", type=float, default=WARMUP_S)
    ap.add_argument("--tiers", action="store_true", default=config.TIERS_ENABLED)
    ap.add_argument("--edge", action="store_true", default=config.EDGE_MANAGER_ENABLED)
    ap.add_argument("--no-delta-adj", action="store_true",
                    help="do not carry the theo forward on futures between option snapshots "
                         "(the control arm; see strategy/deltaadj)")
    ap.add_argument("--no-smooth-skew", action="store_true",
                    help="refit the skew from each snapshot instead of reusing the smoothed "
                         "curve -- the control arm for the smoothing A/B")
    ap.add_argument("--no-fast-spot", action="store_true",
                    help="poll the index over REST and carry the theo with strategy/deltaadj, "
                         "as before the WebSocket feeds (the control arm)")
    ap.add_argument("--no-reprice-at-send", action="store_true",
                    help="post at the spot the cycle opened with instead of re-pricing the "
                         "chosen markets just before apply (the control arm)")
    ap.add_argument("--no-guard", action="store_true",
                    help="do not run the fast cancel guard (the control arm; see "
                         "strategy/fastcancel)")
    ap.add_argument("--no-tee", action="store_true",
                    help="do not write the Kalshi socket's frames to a capture segment "
                         "(the control arm; see storage/feed_tee)")
    ap.add_argument("--no-ws", action="store_true",
                    help="price off polled /markets/orderbooks instead of the WebSocket book "
                         "(the control arm; see kalshi/ws)")
    ap.add_argument("--no-gate", action="store_true",
                    help="always exit 0; for exploratory runs that should not alarm")
    ap.add_argument("--adopt-resting", action="store_true",
                    help="keep orders already resting at startup instead of cancelling them "
                         "(see the startup reconciliation below -- cancelling is the default)")
    args = ap.parse_args(argv)

    _install_signal_handlers()
    _STOP.clear()

    stamp = config.run_stamp()

    # ONE Board PER UNDERLYING. Everything below this point that is not shared hangs off one of
    # these; see `Board` for what is deliberately left global (the Massive socket, the Kalshi
    # socket, the resting book, the risk limit, the journal).
    boards = [Board(u, d, h) for u, (d, h) in sorted(config.QUOTE_BOARDS.items())]
    _now, _mono = datetime.datetime.now(config.ET), time.monotonic()
    for b in boards:
        b.daily_m, b.hourly_m = b.discover(_now, _mono, force=True)
    boards = [b for b in boards if b.markets]
    if not boards:
        raise SystemExit("no markets for today's boards")
    markets = [m for b in boards for m in b.markets]

    # THE OPTION STRIPS, one per underlying, against ONE Massive connection.
    #
    # Massive permits one options WebSocket per account and caps it at 1000 tickers, so the two
    # strips share a budget. Measured 2026-08-26: SPXW 58 pinned + 77 band = 105 strikes = 210
    # tickers, NDXP 58 pinned + 174 band = 202 strikes = 404 tickers, 614 of 1000 with both on.
    # It fits because `universe.parity_band` is a PERCENT of spot rather than a strike count -- a
    # fixed count would be +-3.3% on SPX's 5-wide grid and +-0.85% on NDX's, which sits at ~29,300.
    #
    # Alternating between the strips instead would be worse, not cheaper: a newly subscribed
    # symbol arrives BLIND (there is no snapshot on subscribe -- measured, 15 s after re-adding 8
    # near-money strikes only 2 had quoted), so every switch buys a multi-second hole in whichever
    # chain we switch to. See `QuoteCollector.recenter`.
    #
    # STRIKES ARE PINNED FROM THE DAILY BOARD ONLY, and that is a pricing fact rather than a
    # shortcut. `universe.bracketing_strikes` exists to serve `theo.Pricer.digital_above`, which
    # reads the smile at listed neighbours of each boundary. `HourlyPricer` does not: its
    # `p_above_for` is analytic in K -- `bs.p_above`, or a spline with flat extrapolation outside
    # its knots -- so an hourly rung needs no option contract at its own strike, only enough
    # near-money strikes to fit atmvol and the smile, which the daily board already pins.
    #
    # Passing the hourly rungs through here is expensive and, on NDX, fatal: measured 2026-08-26,
    # NDX daily-only pins 58 strikes and NDX daily+hourly pins 318, which is 636 tickers on its
    # own. `desired_tickers` "never sacrifices the pinned set", so it would not trim -- it would
    # hand `QuoteCollector` an over-cap list and the subscription would freeze. The re-centre in
    # `cycle` passes `daily_m` for exactly this reason and the two MUST stay in step.
    tickers = []
    for b in boards:
        # NOTE the order: (root, strikes, contracts). Unpacking it as (root, expiry, strikes) puts
        # the contract dicts where the strike floats belong and `theo.neighbors` dies comparing a
        # dict to a float. `expiry` is derived from the date, not returned here.
        b.root, b.strikes, _contracts = universe.option_root_and_strikes(underlying=b.underlying)
        b.expiry = config.occ_expiry()
        bounds = ptheo.boundaries(b.daily_m)
        seed = (min(bounds) + max(bounds)) / 2 if bounds else None
        _, b.tickers = universe.desired_tickers(b.daily_m, b.strikes, b.root, b.expiry, seed)
        tickers.extend(b.tickers)
    if len(tickers) > mv.QuoteCollector.MAX_TICKERS:
        raise SystemExit(
            f"{len(tickers)} option tickers over the {mv.QuoteCollector.MAX_TICKERS} "
            f"Massive per-connection cap ("
            + ", ".join(f"{b.underlying}/{b.root}={len(b.tickers)}" for b in boards) + ")")

    # `wbudget`, not `budget` -- `strategy.budget` is imported at module scope and a local of the
    # same name here would read as the module three hundred lines away in `cycle`.
    wbudget = k.set_write_budget()
    reads = k.set_read_budget(share=config.READ_SHARE_TRADING)
    for b in boards:
        n_ev = len({m.get("event_ticker") for m in b.hourly_m})
        print(f"[run_04] {b.underlying}: {len(b.markets)} markets "
              f"({len(b.daily_m)} daily + {len(b.hourly_m)} hourly across {n_ev} event(s)), "
              f"strip {b.root} {len(b.tickers)} tickers")
    print(f"[run_04] {len(markets)} markets over {len(boards)} underlying(s), "
          f"{len(tickers)}/{mv.QuoteCollector.MAX_TICKERS} option tickers | "
          f"live={args.live} tiers={args.tiers} edge={args.edge}")
    print(f"[run_04] write budget: tier={wbudget.get('usage_tier')} "
          f"capacity={wbudget.get('bucket_capacity')} refill={wbudget.get('refill_rate')}/s "
          f"max_batch={wbudget.get('max_batch')} tiers_seen={wbudget.get('tiers_seen') or 'none'} "
          f"| reserve={config.WRITE_RESERVE_TOKENS:.0f} tok")
    print(f"[run_04] read budget: {reads['req_per_s_before']:.1f} -> "
          f"{reads['req_per_s_after']:.1f} req/s (share {reads['share']:.0%}, "
          f"cap {reads['cap']:.0f}), tiers seen {reads['tiers_seen'] or 'none'}")

    # THE PER-UNDERLYING MODELS. Four objects per board and not one of them is shareable: the
    # edge curve is calibrated per underlying, the variance calendar is per underlying, the smile
    # is a fact about one option root, the realized-vol model is fit on one index, and the
    # futures contract is ES or NQ.
    for b in boards:
        b.em = edgemgr.EdgeManager(b.underlying, enabled=args.edge)
        if not args.edge:
            # Explicit operator override. Since 2026-08-23 a market without an R is NOT quoted,
            # so without this flag `--edge` off would quote nothing at all rather than
            # reproducing the pre-calibration flat quoter.
            b.em.curve = pedge.FlatCurve()
            b.em.operator_flat = True
        print(f"[run_04] {b.underlying} edge manager: available={b.em.available} "
              f"curve={b.em.curve!r} operator_flat={b.em.operator_flat}")
        if args.edge and not b.em.available:
            print(f"[run_04] *** {b.underlying} edge manager UNAVAILABLE ({b.em.reason}) -- no "
                  f"{b.underlying} market will be quotable. Fix the calibration or pass "
                  f"--no-edge deliberately. ***")

        b.lc = leanchain.LeanChain(b.underlying)
        b.lc.start()
        print(f"[run_04] {b.underlying} lean: enabled={config.LEAN_ENABLED} available={b.lc.ok}")

        b.sk = None if args.no_smooth_skew else skewstate.SkewState(b.underlying)
        print(f"[run_04] {b.underlying} skew: smoothed={b.sk is not None}"
              + ("" if b.sk is None else f" window={b.sk.window_s}s "
                                         f"recompute={b.sk.recompute_s}s "
                                         f"half_life={b.sk.half_life_s}s"))

        # Promoted to production 2026-08-24. It was implemented and validated in the backtest
        # (`theopipe.delta_adjusted_theo`, gated by `validate_ws` section 5) and never reached
        # the live loop, which re-priced spot only every 2 s -- eleven cents of ATM mispricing per
        # index point of move, sitting on the book. `--no-delta-adj` is the control arm.
        #
        # ONE TICKER PER UNDERLYING, from `massive.futures.PRODUCT` (SPX->ES, NDX->NQ). Both legs
        # verified live 2026-08-26: ESU6 and NQU6 both resolved, 20/20 samples fresh, median
        # quote age 0.353 s and 0.347 s, 0 errors. Each ticker polls at FUTURES_POLL_S (0.25 s),
        # so two boards is 8 req/s on the Massive REST budget rather than 4 -- a budget the
        # option socket and the Kalshi order path both stay out of.
        if not args.no_delta_adj:
            ft = deltaadj.FuturesTicker(b.underlying).start()
            if ft.ok:
                b.da = deltaadj.DeltaAdjuster(ft)
            else:
                # Failing soft, PER BOARD: no futures feed for this underlying means its theo is
                # simply not carried forward, which is the pre-2026-08-24 behaviour. It must
                # never take the session down and must never affect the other underlying.
                print(f"[run_04] *** {b.underlying} delta adjustment UNAVAILABLE "
                      f"({ft.last_error}) -- its theo is the option snapshot's, re-priced every "
                      f"{args.interval}s ***", file=sys.stderr)
        print(f"[run_04] {b.underlying} delta-adj: {'on' if b.da is not None else 'OFF'}"
              + ("" if b.da is None else f" contract={b.da.ticker.ticker} "
                                         f"poll={b.da.ticker.poll_s}s "
                                         f"max_adj=${b.da.max_adj:.2f}"))
    edges = EdgeUnion(boards)

    # --- THE FAST SPOT PATH ---------------------------------------------------------------
    #
    # Two WebSockets and a per-underlying basis, replacing two REST polls. Measured 2026-08-26:
    # futures age 320ms -> 73.6ms and 4/s -> 114-220/s; index age 711ms -> 21.3ms; and the
    # per-read TCP+TLS handshake (40.4ms, 29% of every 139ms poll) becomes one 103ms connect.
    #
    # The feeds are process-wide, not per board -- both are multi-underlying by construction and
    # a second connection would buy nothing.
    us = tuple(b.underlying for b in boards)
    futures = idxfeed = spot_tape = None
    if not args.no_fast_spot:
        # THE SPOT TAPE. Its own channel, and it records only what CHANGED -- futures mid moves
        # and index publications. On 2026-08-26 both feeds ran all afternoon and nothing wrote
        # them down, so when the session lost money the decomposition of a fill's loss into
        # `delta * dF` could not be measured below the 2s quote cycle. Roughly 10-20 distinct
        # futures mids a second against 114-220 frames, so this is a few percent of the raw feed.
        if not args.no_tee and getattr(config, "QUOTE_TEE_ENABLED", False):
            spot_tape = ftee.FeedTee(
                channel=getattr(config, "SPOT_TEE_CHANNEL", "spot"),
                subscribe_state={"underlyings": list(us), "feeds": ["futures", "index"],
                                 "records": "updates_only", "live": args.live,
                                 "run": stamp}).start()
            if not spot_tape.ok:
                print(f"[run_04] *** spot tape UNAVAILABLE ({spot_tape.error}) ***",
                      file=sys.stderr)
                spot_tape = None
        futures = mstreams.FuturesFeed(us, tee=spot_tape).start()
        idxfeed = mstreams.IndexFeed(us, tee=spot_tape).start()
        time.sleep(1.0)                          # let auth + the first frames land
        print(f"[run_04] futures ws: ok={futures.ok} contracts={futures.contract}"
              + (f" ERR {futures.last_error}" if futures.last_error else ""))
        print(f"[run_04] index ws:   ok={idxfeed.ok} "
              f"underlyings={sorted(idxfeed.snapshot())}"
              + (f" ERR {idxfeed.last_error}" if idxfeed.last_error else ""))
        if not futures.ok or not idxfeed.ok:
            # Failing soft, all the way back to today's behaviour: no fast spot means the cycle
            # reads the index over REST and `deltaadj` carries the theo, exactly as before.
            print("[run_04] *** fast spot UNAVAILABLE -- falling back to REST index + "
                  "delta-adjust ***", file=sys.stderr)
            for f in (futures, idxfeed):
                if f is not None:
                    f.stop()
            futures = idxfeed = None
    for b in boards:
        b.trk = basismod.BasisTracker(b.underlying) if futures is not None else None

    # THE BASIS IS FOLDED ON THE INDEX TICK, in its own thread, because that is the only instant
    # when both legs are current -- see `pricing/basis`. One thread for every underlying: it wakes
    # far more often than the index publishes and does nothing when the timestamp has not moved.
    basis_stop = threading.Event()

    def _fold_basis():
        seen = {}
        while not basis_stop.is_set():
            try:
                snap = idxfeed.snapshot()
                for u, q in snap.items():
                    if seen.get(u) == q.ts_ns:
                        continue
                    seen[u] = q.ts_ns
                    trk = next((b.trk for b in boards if b.underlying == u), None)
                    if trk is not None:
                        mid, _ = futures.mid(u)
                        trk.observe(mid, q)
            except Exception as e:                           # noqa: BLE001
                print(f"[basis] {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            basis_stop.wait(0.05)

    basis_thread = None
    if futures is not None:
        basis_thread = threading.Thread(target=_fold_basis, daemon=True)
        basis_thread.start()
    apply_lock = threading.RLock()

    col = mv.QuoteCollector(tickers)
    col.start()
    col.wait_for_coverage(len(tickers) // 2, timeout=args.warmup)
    time.sleep(2)

    book = omgr.RestingBook()
    jrnl = jnl.open_journal(stamp, name="quotes.jsonl")
    rjrnl = jnl.open_journal(stamp, name="risk.jsonl")
    rtape = risktape.RiskTape(rjrnl)
    # One row per underlying per cycle: the vol-spike signal and what it WOULD charge. Its own
    # file so no existing reader of quotes.jsonl meets a row without a ticker.
    vsjrnl = jnl.open_journal(stamp, name="volspike.jsonl")

    # THE KALSHI MARKET-DATA SOCKET. One subscribe covers the whole board for the session; the
    # frames that follow are not metered. Failing soft is deliberate and matches every other feed
    # here: no socket means the loop polls REST, which is a slower book, not a broken one.
    # THE TAPE, opened before the socket so no frame is received without somewhere to go.
    #
    # `storage/ws_recorder` has been gated off since 2026-08-22, so the last order-book tape in S3
    # is 2026-08-21 and both live sessions have none. Order book data has no REST backfill path: a
    # day not captured is gone. `/portfolio/fills` and this journal say what we quoted and what
    # filled; only the book says whether the fill was any good. Teeing the socket we already hold
    # costs no connection, no subscription and no tokens -- Kalshi meters commands, not frames --
    # and covers exactly the markets we traded. See `storage/feed_tee`.
    tee = None
    if not args.no_tee and getattr(config, "QUOTE_TEE_ENABLED", False):
        tee = ftee.FeedTee(subscribe_state={"channels": list(kws.CHANNELS),
                                            "n_markets": len(markets),
                                            "series": sorted({s for b in boards for s in b.series}),
                                            "live": args.live, "run": stamp}).start()
        if not tee.ok:
            # Failing soft: no tape is a lost research question, not a broken quoter. It must
            # never take the session down.
            print(f"[run_04] *** feed tape UNAVAILABLE ({tee.error}) -- quoting without one ***",
                  file=sys.stderr)
            tee = None
    print(f"[run_04] feed tape: {'on' if tee is not None else 'OFF'}"
          + ("" if tee is None else f" dt={tee.dt} channel={tee.channel} "
                                    f"session={tee.session_id} -> {config.WS_DIR}"))

    mdfeed = None
    if not args.no_ws:
        try:
            mdfeed = kws.MarketFeed([m["ticker"] for m in markets], tee=tee).start()
            cov = mdfeed.wait_for_coverage(config.WS_BOOK_MIN_COVERAGE,
                                           timeout=config.WS_BOOK_WARMUP_S)
            h = mdfeed.health()
            print(f"[run_04] ws book: {h['n_valid']}/{h['n_wanted']} markets "
                  f"({cov:.0%} coverage) in {h['commands']} commands, "
                  f"{h['frames']} frames | channels={','.join(kws.CHANNELS)}")
            if cov < config.WS_BOOK_MIN_COVERAGE:
                print(f"[run_04] *** ws coverage {cov:.0%} below the "
                      f"{config.WS_BOOK_MIN_COVERAGE:.0%} floor after "
                      f"{config.WS_BOOK_WARMUP_S:.0f}s -- cycles will fall back to REST ***",
                      file=sys.stderr)
        except Exception as e:                               # noqa: BLE001
            mdfeed = None
            print(f"[run_04] *** ws book UNAVAILABLE ({type(e).__name__}: {str(e)[:90]}) -- "
                  f"polling /markets/orderbooks ***", file=sys.stderr)
    else:
        print("[run_04] ws book: OFF (--no-ws), polling /markets/orderbooks")

    startup_reconcile(live=args.live, adopt=args.adopt_resting)

    # SEED THE POSITION MAP FROM REST. `market_positions` only reports CHANGES, so a position
    # opened before this process started is invisible to the socket until it next moves -- and on
    # 2026-08-26 a restart mid-session began with five real positions worth $63.78 that the
    # inventory gate read as flat. Cheap (one read), and it can only ever add information.
    if mdfeed is not None:
        try:
            held = k.open_positions()
            n = mdfeed.seed_positions(held)
            print(f"[run_04] positions at startup: {len(held)} held, feed seeded ({n} known)"
                  + ("" if not held else "  " + ", ".join(f"{t}={v:+.0f}"
                                                          for t, v in list(held.items())[:6])))
        except Exception as e:                               # noqa: BLE001
            print(f"[run_04] *** could not read positions at startup "
                  f"({type(e).__name__}: {str(e)[:80]}) -- the inventory gate starts blind ***",
                  file=sys.stderr)

    risk = risklimit.SessionRisk()
    acct = risklimit.AccountPoller().start()
    snap0 = acct.latest()
    print(f"[run_04] risk: loss limit ${risk.limit:,.2f} on ACCOUNT equity marked to Kalshi mid "
          f"(poll {acct.interval_s:g}s) | balance "
          + (f"${snap0[1]:,.2f}, {len(snap0[2])} positions" if snap0
             else f"UNREADABLE ({acct.last_error})")
          + f" | halt file {risk.halt_path}"
          + (" *** PRESENT AT STARTUP ***" if risk.halt_reason() else ""))

    # `state` is what survives BETWEEN cycles and is shared by all boards. Everything
    # per-underlying that used to live here -- the delta adjuster, the smile -- now hangs off its
    # `Board` instead, which is why those keys are gone rather than plural.
    # THE FAST GUARD, started after the resting book exists and after the startup reconcile, so it
    # can never act on orders this process did not put there.
    guard = None
    if futures is not None and not args.no_guard:
        guard = fastcancel.FastGuard(
            futures, {b.underlying: b.trk for b in boards}, book,
            live=args.live, apply_lock=apply_lock,
            on_cancel=lambda cs, res: print(
                f"[guard] pulled {res['cancelled'] + res.get('cancel_gone', 0)}/{len(cs)} "
                f"on a fast move: {', '.join(sorted({c['ticker'] for c in cs}))[:120]}",
                file=sys.stderr)).start()
    print(f"[run_04] fast guard: {'on' if guard is not None else 'OFF'}"
          + ("" if guard is None else f" interval={guard.interval_s * 1000:.0f}ms "
                                      f"min_move={guard.min_move}pts (cancel-only)"))

    state = {"tickers": [m["ticker"] for m in markets],
             "max_batch": wbudget.get("max_batch"),
             "risk": risk, "acct": acct,
             "mdfeed": mdfeed,
             "tee": tee,
             "futures": futures,
             "indexfeed": idxfeed,
             "spot_tape": spot_tape,
             "guard": guard,
             "risktape": rtape,
             "vsjrnl": vsjrnl,
             "apply_lock": apply_lock}
    deadline = time.time() + args.minutes * 60
    n = 0
    stats = {"quiet": 0, "placed": 0, "cancelled": 0, "errors": 0, "no_spot": 0,
             "feed": 0, "ms": []}
    try:
        while time.time() < deadline and not _STOP.is_set():
            t_start = time.time_ns()
            try:
                r = cycle(state, col, boards, edges, book, args, jrnl)
            except Exception as e:                           # noqa: BLE001
                print(f"[cycle] {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                r = {"error": type(e).__name__}
            n += 1
            if "error" in r:
                stats["errors"] += 1
            else:
                if r.get("skipped") == "no_spot":
                    stats["no_spot"] += 1
                if r.get("skipped") in ("feed_down", "feed_stale"):
                    stats["feed"] += 1
                stats["placed"] += r.get("placed", 0)
                stats["cancelled"] += r.get("cancelled", 0)
                stats["ms"].append(r.get("ms", 0))
                if not r.get("placed") and not r.get("cancelled"):
                    stats["quiet"] += 1
            if n <= 3 or n % 10 == 0 or r.get("placed") or r.get("cancelled"):
                print(f"[{n:4d}] {json.dumps(r, default=str)}", flush=True)
            # `_STOP.wait` rather than `time.sleep`, so a signal arriving mid-sleep is acted on
            # now instead of after the rest of the interval. On a 2 s cycle that is the difference
            # between a stop taking 2 s and taking 4.
            _STOP.wait(max(0.0, args.interval - (time.time_ns() - t_start) / 1e9))
    finally:
        col.stop()
        if mdfeed is not None:
            h = mdfeed.health()
            # The command count is the number that matters for the ONE WebSocket limit that is
            # real (error 27, per-subscription command rate) and the one nobody has a figure for.
            # Printing it every run is how a figure gets established.
            print(f"[run_04] ws book: {h['frames']} frames, {h['commands']} commands, "
                  f"{h['gaps']} gaps, {h['reconnects']} resubscribes, "
                  f"{h['n_valid']}/{h['n_wanted']} valid at exit"
                  + (f" | command errors: {h['command_errors']}" if h["command_errors"] else ""))
            mdfeed.stop()
        if tee is not None:
            # AFTER the feed, so frames still in flight are written rather than dropped on the
            # floor. `stop` drains the queue, closes the segment and writes its manifest, which is
            # what makes the day flushable by the existing storage timers.
            tee.stop()
            print(f"[run_04] feed tape: {tee.frames:,} frames, {tee.segments_closed} segment(s), "
                  f"{tee.bytes_written/1e6:.1f} MB gz, {tee.dropped} dropped"
                  + (f" | ERROR {tee.error}" if tee.error else ""))
        if guard is not None:
            guard.stop()
            print(f"[run_04] fast guard: {guard.detail()}")
        basis_stop.set()
        if basis_thread is not None:
            basis_thread.join(timeout=3)
        for f in (futures, idxfeed):
            if f is not None:
                print(f"[run_04] {f.name} ws: {f.health()}")
                f.stop()
        if spot_tape is not None:
            # AFTER both feeds, so updates still in flight are written rather than dropped.
            spot_tape.stop()
            print(f"[run_04] spot tape: {spot_tape.frames:,} updates, "
                  f"{spot_tape.segments_closed} segment(s), "
                  f"{spot_tape.bytes_written/1e6:.1f} MB gz, {spot_tape.dropped} dropped"
                  + (f" | ERROR {spot_tape.error}" if spot_tape.error else ""))
        for b in boards:
            if b.da is not None:
                b.da.ticker.stop()
        jrnl.close()
        vsjrnl.close()
        print(f"[run_04] risk tape: {rtape.detail()}")
        rjrnl.close()
        ms = sorted(stats["ms"])
        # The quiet rate is the deadband's whole justification: on a price-time exchange every
        # order we DON'T move keeps its queue position. A rate near zero means the deadband is
        # too tight and the quoter is churning its own queue.
        print(f"[run_04] {n} cycles | quiet {stats['quiet']}/{n - stats['errors']} "
              f"({stats['quiet'] / max(1, n - stats['errors']):.0%}) | "
              f"placed {stats['placed']} cancelled {stats['cancelled']} errors {stats['errors']}")
        if ms:
            print(f"[run_04] cycle ms: p50 {ms[len(ms)//2]:.0f} "
                  f"p95 {ms[int(len(ms)*0.95)]:.0f} max {ms[-1]:.0f}")
        print(f"[run_04] {len(book)} orders resting, {book.contracts():.0f} contracts")
        print(f"[run_04] risk: {json.dumps(risk.summary(), default=str)}")
        expected = max(1.0, args.minutes * 60.0 / max(args.interval, 0.001))
        # `in_trading_session`, NOT `in_capture_session`. The latter is hard-wired False while
        # CAPTURE_SESSION_ENABLED is off -- a storage decision -- and reading it here downgraded a
        # feed outage in the middle of a trading day from NO_SPOT (exit 1, alarms) to NO_SESSION
        # (exit 0, silent). See config.in_trading_session.
        v, code, vd = verdict({**stats, "cycles": n}, expected, config.in_trading_session())
        print(f"[run_04] VERDICT {v}  cycles {vd['cycles']}/{vd['expected']} "
              f"errors {vd['errors']} ({vd['error_rate']:.0%}) "
              f"spot {vd['spot_rate']:.0%} placed {vd['placed']} "
              f"quiet {stats['quiet'] / max(1, n - stats['errors']):.0%}"
              + (f"  -- {'; '.join(vd['reasons'])}" if vd["reasons"] else ""), flush=True)
        if _STOP.is_set():
            print(f"[run_04] stopped early: {risk.halted or 'signal'}", flush=True)
        exit_code = 0 if args.no_gate else code
        # A halt is never a passing run, whatever the cycle statistics say. `--no-gate` is for
        # exploratory runs that should not alarm; a loss limit or an operator halt is not one.
        if risk.halted:
            exit_code = 1
        if args.live:
            try:
                # `cancel_all.today_events()` names 18 events -- every daily, hourly and DUD
                # board of BOTH index families, chunked into two queries. Deliberately wider than
                # what we quote: cancelling an event we never touched is a free no-op, and the
                # cost of omitting one is an order resting on a live market with nothing to pull
                # it. Until 2026-08-25 it named only the two H1600 boards, so every hourly order
                # would have survived this block.
                left = cancel_all.cancel_events(cancel_all.today_events())
                print(f"[exit] cancel-on-exit complete, {left} still resting")
                if left:
                    print("[exit] WARNING: run  python -m strategy.cancel_all  NOW",
                          file=sys.stderr)
            except Exception as e:                           # noqa: BLE001
                print(f"[exit] CANCEL FAILED ({type(e).__name__}: {e}) -- "
                      f"run  python -m strategy.cancel_all  NOW", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
