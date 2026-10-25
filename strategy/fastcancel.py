"""The fast guard: re-price the resting book from the futures, and pull what has gone bad.

WHAT THIS IS FOR. On 2026-08-26 SPX fell 5.15 points in 16 seconds, through the strike of an
hourly board nine minutes from settlement. 154 orders filled in that minute, 76 of them inside one
2-second cycle, and the markout against our own theo was -18.17c mean. The loop reacted as fast as
its architecture allowed -- it cancelled 104 orders in the same cycle that took 76 fills, and the
write budget deferred exactly zero reducing cancels all session. The architecture was the problem:
theo only moved when the cycle moved it.

WHY IT REPLACES THE DELTA ADJUSTMENT RATHER THAN HELPING IT. `strategy/deltaadj` carries theo
forward by `delta * (F_now - F_anchor)`, and it works exactly as designed -- but its anchor is
re-sampled at the top of every cycle, so `f_move` only ever spans the few hundred milliseconds
between the anchor and the pricing instant. Measured through that crash: SPX `f_move` ranged
-0.50 to +1.25 points while the index moved 5.15, and the adjustment contributed -1.39c and
-6.14c against a 29-39 cent theo move. It is a linear approximation of a re-price we could not
previously afford. Measured 2026-08-26, re-pricing the ~60 markets we actually quote costs
**3.5ms** (NDX 1.31 + SPX 2.16) with vol held fixed -- so we can afford the exact thing, smile
slope included. Running both would double-count the same move.

IT CANCELS. IT NEVER PLACES. That is not a simplification, it is the constraint that makes the
whole design fit:

  * A cancel costs 2 tokens and is admitted unconditionally as a REDUCE (`strategy/budget`), so
    the 300 tokens/s bucket allows ~150 of them a second. A place costs 10 against a hard 30
    orders/s ceiling and a 30-per-request batch cap.
  * Re-quoting at 200Hz would forfeit queue position on every rung, every time, on a price-time
    exchange where the queue position is most of what a resting order is worth.
  * Deciding WHAT to quote needs the option board, the band, the edge curve and the capital
    budget. Deciding whether a resting order has gone negative needs only a forward.

So: the slow loop (2s) owns vol, selection and placement; this owns the forward and the exit.

THE MOVE IS APPLIED AS A DELTA, NOT AS A LEVEL. Each pricer keeps its own forward convention --
`theo.Pricer` takes the resolved index spot because KXINX settles on the index, while
`hourly.HourlyPricer` takes atmvol's PARITY forward because substituting the index inflates
`cp_spread` and trips `MAX_CP_VOL_SPREAD` (see `pricing/validate_theo`). Setting either to the
fast spot would erase a distinction those modules argue for at length. Shifting both by the same
increment preserves it.
"""
import threading
import time

import config
from strategy import orders as omgr

# Same getattr pattern: the fallback is live. Illustrative value; the live value is not published.
FAST_REPRICE_S = getattr(config, "FAST_REPRICE_S", 0.01)

# An index move of this many points triggers a guard reprice. Read via getattr(config,
# "FAST_MIN_MOVE_POINTS"), which config.py does not define. Illustrative value; the live value is
# not published.
MIN_MOVE_POINTS = getattr(config, "FAST_MIN_MOVE_POINTS", 0.1)


def reprice_shifted(daily, hourly, base_daily_fwd, base_hourly_fwd, move, items):
    """{ticker: fair} for `items` with both forwards shifted by `move`. Restores them after.

    `items` is [(market, pricer)]. The forwards are RESTORED in a finally, including when pricing
    raises: the slow cycle owns these objects and reads `pricer.spot` again when it journals, so a
    caller that left them shifted would have the next cycle measure its increment from a forward
    it never priced at.

    THE MOVE IS APPLIED AS A DELTA, not as a level, because each pricer keeps its own forward
    convention -- `theo.Pricer` takes the resolved index spot because KXINX settles on the index,
    while `hourly.HourlyPricer` takes atmvol's PARITY forward because substituting the index
    inflates `cp_spread` and trips MAX_CP_VOL_SPREAD. Setting either to a common level would erase
    a distinction those modules argue for at length.

    Shared by the fast guard (which uses it to decide cancels at 200Hz) and by the quote cycle
    (which uses it to price what it is about to POST off a spot read milliseconds earlier rather
    than one read before 150ms of vol, greek, band and selection work). One implementation,
    because two would drift and the difference would be invisible.

    THE SMILE CURVE CARRIES ITS OWN FORWARD, and it has to move too (fixed 2026-09-23). With a
    non-flat curve installed -- the live case, `skew_smoothed=True` -- both pricers price the
    digital through `SkewFit.p_above_at`, which reads `SkewFit.forward`, not `Pricer.spot` or
    `HourlyPricer.fit.forward`. Shifting only those left the re-priced fair almost unchanged, so the
    guard and reprice-at-send were near no-ops whenever a smile was on. Each distinct curve (the
    4pm and hourly pricers usually SHARE one) is shifted once, by the same increment, and its
    projection cache -- keyed on vol alone, but computed at a forward -- is swapped out for the
    duration so neither a stale nor a shifted projection survives the call.
    """
    out = {}
    curves = []
    for p in (daily, hourly):
        sf = getattr(p, "_skew", None) if p is not None else None
        if sf is not None and not getattr(sf, "flat", True) and sf.forward \
                and all(sf is not c[0] for c in curves):
            curves.append((sf, sf.forward, sf._proj))
    try:
        if daily is not None and base_daily_fwd is not None:
            daily.spot = base_daily_fwd + move
        if hourly is not None and base_hourly_fwd is not None:
            hourly.fit.forward = base_hourly_fwd + move
        for sf, f0, _p in curves:
            sf.forward = f0 + move
            sf._proj = {}
        for m, pricer in items:
            fair, _d = pricer.market_fair(m)
            if fair is not None:
                out[m["ticker"]] = fair
    finally:
        if daily is not None and base_daily_fwd is not None:
            daily.spot = base_daily_fwd
        if hourly is not None and base_hourly_fwd is not None:
            hourly.fit.forward = base_hourly_fwd
        for sf, f0, proj in curves:
            sf.forward = f0
            sf._proj = proj
    return out


class BoardState:
    """One underlying's fast-path inputs, refreshed by the slow cycle."""

    __slots__ = ("underlying", "daily", "hourly", "by_ticker",
                 "base_spot", "base_daily_fwd", "base_hourly_fwd", "stamp")

    def __init__(self, underlying):
        self.underlying = underlying
        self.daily = self.hourly = None
        # {ticker: (market, pricer)}. An INDEX, not two lists, because the tick walks the RESTING
        # book -- tens of markets -- and must never walk the board, which is thousands. Scanning
        # both lists per tick cost a measured 23.4ms p50 against a 5ms budget and stole enough CPU
        # to push the main cycle's p95 from 511ms to 1,364ms.
        self.by_ticker = {}
        self.base_spot = self.base_daily_fwd = self.base_hourly_fwd = None
        self.stamp = 0.0


class FastGuard:
    """Watches the futures, re-prices resting markets, cancels what has gone negative."""

    def __init__(self, futures, trackers, book, live=False, apply_lock=None,
                 interval_s=None, min_move=None, on_cancel=None):
        self.futures = futures                  # massive.streams.FuturesFeed
        self.trackers = trackers                # {underlying: pricing.basis.BasisTracker}
        self.book = book
        self.live = live
        self.apply_lock = apply_lock or threading.RLock()
        self.interval_s = FAST_REPRICE_S if interval_s is None else interval_s
        self.min_move = MIN_MOVE_POINTS if min_move is None else min_move
        self.on_cancel = on_cancel
        self.boards = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self.wakes = 0
        self.repriced = 0                       # times a move was large enough to re-price
        self.cancels = 0
        self.errors = 0
        self.last_error = None
        self.max_reprice_ms = 0.0
        self._reprice_ms = []

    # -- handed over by the slow cycle --

    def publish(self, underlying, daily_pricer, hourly_pricer, daily_m, hourly_m, spot):
        """Install this cycle's pricers and the forward they were priced at.

        Called once per board per cycle, AFTER the board is priced and BEFORE the guard is allowed
        to act on it. The forwards recorded here are the origin the fast increment is measured
        from; a guard reading last cycle's origin against this cycle's book would apply a move
        that had already been priced in.
        """
        with self._lock:
            st = self.boards.setdefault(underlying, BoardState(underlying))
            st.daily, st.hourly = daily_pricer, hourly_pricer
            st.by_ticker = {m["ticker"]: (m, daily_pricer) for m in daily_m}
            if hourly_pricer is not None:
                st.by_ticker.update({m["ticker"]: (m, hourly_pricer) for m in hourly_m})
            st.base_spot = spot
            st.base_daily_fwd = getattr(daily_pricer, "spot", None)
            st.base_hourly_fwd = (hourly_pricer.fit.forward
                                  if hourly_pricer is not None and hourly_pricer.fit else None)
            st.stamp = time.monotonic()

    def retire(self, underlying):
        """Stop acting on a board -- it failed to price, or is no longer quoted."""
        with self._lock:
            self.boards.pop(underlying, None)

    # -- lifecycle --

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        return self

    @property
    def ok(self):
        return bool(self._thread and self._thread.is_alive())

    # -- the loop --

    def _run(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.wakes += 1
                self._tick()
            except Exception as e:                           # noqa: BLE001
                # A guard that dies silently is worse than no guard: the loop would keep quoting
                # believing it was protected. Record and keep going.
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
            self._stop.wait(max(0.0, self.interval_s - (time.perf_counter() - t0)))

    def _tick(self):
        with self._lock:
            boards = list(self.boards.values())
        if not boards:
            return
        doomed = []
        for st in boards:
            if st.daily is None or st.base_spot is None:
                continue
            trk = self.trackers.get(st.underlying)
            if trk is None:
                continue
            mid, _age = self.futures.mid(st.underlying)
            spot, sd = trk.spot(mid)
            if spot is None:
                continue
            move = spot - st.base_spot
            if abs(move) < self.min_move:
                continue
            doomed.extend(self._reprice_board(st, move, spot))
        if doomed:
            self._cancel(doomed)

    def _reprice_board(self, st, move, spot):
        """Shift both forwards by `move`, re-price the RESTING markets, return dead orders."""
        t0 = time.perf_counter()
        # WALK THE BOOK, NOT THE BOARD. These are the only markets a cancel could apply to.
        # `resting_tickers`, NOT `tickers`: the slow cycle creates a MarketState for every market
        # on the board when it stamps the theo and the touch, so `tickers()` is thousands and only
        # tens of them hold an order. See `RestingBook.resting_tickers`.
        mine = [st.by_ticker[t] for t in self.book.resting_tickers() if t in st.by_ticker]
        if not mine:
            return []
        # Restore the origin afterwards: the slow cycle owns these objects and will read
        # `pricer.spot` again when it journals. A guard that left them shifted would have the next
        # cycle measure its increment from a forward it never priced at.
        fairs = reprice_shifted(st.daily, st.hourly, st.base_daily_fwd, st.base_hourly_fwd,
                                move, mine)
        dead = []
        for m, _pricer in mine:
            dead.extend(self._judge_at(m, fairs.get(m["ticker"])))
        ms = (time.perf_counter() - t0) * 1000.0
        self.max_reprice_ms = max(self.max_reprice_ms, ms)
        self._reprice_ms.append(ms)
        if len(self._reprice_ms) > 2000:
            del self._reprice_ms[:1000]
        self.repriced += 1
        return dead

    def _judge_at(self, market, fair):
        """Resting orders on this market whose edge has gone non-positive at the new fair."""
        tk = market["ticker"]
        if fair is None:
            return []                            # cannot price it: not evidence that it is bad
        out = []
        st = self.book.state(tk)
        for side in ("bid", "ask"):
            for o in st.side(side):
                # THE SAME RULE `plan_side` USES, deliberately: edge is theo - price on the bid and
                # price - theo on the ask, and non-positive means we no longer believe the order.
                # A second, looser rule here would mean two definitions of "losing money".
                edge = (fair - o["price"]) if side == "bid" else (o["price"] - fair)
                if edge <= 0:
                    out.append({**o, "side": side, "ticker": tk, "reason": "fast_edge_negative"})
        return out

    def _cancel(self, dead):
        # De-duplicate on order_id: two boards cannot own the same order, but a market can be
        # judged twice within one tick if the slow cycle republished mid-pass.
        seen, uniq = set(), []
        for c in dead:
            oid = c.get("order_id")
            if oid is None or oid in seen:
                continue
            seen.add(oid)
            uniq.append(c)
        if not uniq:
            return
        with self.apply_lock:
            res = omgr.apply([], uniq, self.book, live=self.live)
        self.cancels += res["cancelled"] + res.get("cancel_gone", 0)
        if self.on_cancel:
            try:
                self.on_cancel(uniq, res)
            except Exception:                                # noqa: BLE001
                pass

    def detail(self):
        ms = sorted(self._reprice_ms)
        return {"ok": self.ok, "wakes": self.wakes, "repriced": self.repriced,
                "cancels": self.cancels, "errors": self.errors, "last_error": self.last_error,
                "boards": sorted(self.boards),
                "reprice_ms_p50": (round(ms[len(ms) // 2], 3) if ms else None),
                "reprice_ms_p95": (round(ms[int(0.95 * (len(ms) - 1))], 3) if ms else None),
                "reprice_ms_max": round(self.max_reprice_ms, 3)}
