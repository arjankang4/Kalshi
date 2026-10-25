"""The LIVE delta-adjusted theo: hold vol on the slow cadence, move spot on the fast one.

WHAT THIS PROMOTES, AND FROM WHERE. `backtest/theopipe.delta_adjusted_theo` has priced the taking
study this way since it was written; `backtest/fastspot` states the motivation for the live loop
outright and then only backtests it:

    "`strategy/run_04_quote_loop` re-prices a board every 2 seconds from a fresh option snapshot.
     That is fine for volatility, which is a diffusion, and far too slow for spot: an ATM rung's
     delta runs $0.1149 per index point into the close, so a one-point move between recomputes is
     eleven cents of mispricing sitting on the book for somebody to lift."

This is the production half of that sentence. The board is still priced from the option snapshot
on the quote cycle; between snapshots the theo is carried forward by

    theo_now = theo_at_snapshot + delta * (F_now - F_at_snapshot)

with `delta` the dollar delta per index point the edge manager already computes for every market
(`greeks.market_greeks`, recomputed every `EDGE_RECOMPUTE_S`).

BOTH ENDS OF THE INCREMENT MUST BE THE SAME INSTRUMENT, and this is the one way to get it wrong.
Differencing a futures mid against the OPTION-IMPLIED forward measures the futures-cash BASIS --
a level, not a move. Measured on 2026-08-21 that basis is ~16.7 index points against an ATM dollar
delta of up to $0.11/point, i.e. more than a dollar of fabricated adjustment on a contract that
pays at most a dollar. In the backtest that single error fired a take on 29% of all book updates.
So the anchor is a FUTURES mid sampled at the snapshot instant, and the fast leg is a futures mid
sampled now, and the basis cancels because it is the same series at both ends.

WHERE THE LIVE FEED COMES FROM. `/futures/v1/quotes/{ticker}` -- REST, top-of-book, nanosecond
timestamps, measured at 131 quotes/second for ES at 14:30 ET (`backtest/fastspot`). There is no
futures WebSocket on this account and the single options WS slot is already spent on the board,
so this polls. Polling is bounded by `FUTURES_POLL_S` and runs on its own thread against the same
`kalshi`-independent Massive REST budget, so it competes with nothing on the order path.

FAILING SOFT IS THE CONTRACT. No quote, a stale quote, an unknown delta, or an adjustment past
`MAX_ADJ_DOLLARS` all return the UNADJUSTED theo with a reason. A theo carried forward on a bad
increment is worse than one that is two seconds old, because the two-second-old one is at least
internally consistent with the vol it was priced from.
"""
import threading
import time

import config
from massive import futures as fut

# Futures mid poll rate for the delta adjustment. Same getattr pattern: the fallback is live.
# Illustrative value; the live value is not published.
FUTURES_POLL_S = getattr(config, "FUTURES_POLL_S", 0.5)

# The futures mid is stale after this many seconds, and the theo goes unadjusted. Read via
# getattr(config, "FUTURES_MAX_STALE_S"), which config.py does not define. Illustrative value; the
# live value is not published.
MAX_QUOTE_AGE_S = getattr(config, "FUTURES_MAX_STALE_S", 2.0)

# Cap on the futures delta adjustment to theo. Read via getattr(config, "DELTA_ADJ_MAX_DOLLARS"),
# which config.py does not define. Illustrative value; the live value is not published.
MAX_ADJ_DOLLARS = getattr(config, "DELTA_ADJ_MAX_DOLLARS", 0.5)


class FuturesTicker:
    """Latest top-of-book mid for one product's front-month contract. Polls on its own thread."""

    def __init__(self, underlying, poll_s=FUTURES_POLL_S):
        self.underlying = underlying
        self.product = fut.PRODUCT.get(underlying, underlying)
        self.poll_s = poll_s
        self.ticker = None
        self._lock = threading.Lock()
        self._t_ns = None
        self._mid = None
        self._stop = threading.Event()
        self._thread = None
        self.polls = 0
        self.errors = 0
        self.last_error = None

    # -- lifecycle --

    def start(self, day=None):
        """Resolve the front month and begin polling. Returns self; never raises."""
        import datetime
        day = day or datetime.datetime.now(config.ET).date()
        try:
            # By VOLUME, not by calendar -- see `futures.front_month` on the roll.
            self.ticker = fut.front_month(self.product, day)
        except Exception as e:                                  # noqa: BLE001
            self.last_error = f"front_month: {type(e).__name__}: {str(e)[:120]}"
        if self.ticker is None:
            self.last_error = self.last_error or f"no front-month contract for {self.product}"
            return self
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self

    @property
    def ok(self):
        return bool(self.ticker and self._thread and self._thread.is_alive())

    # -- polling --

    def _poll(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                # limit=1 and the vendor's own descending order give the newest quote in one row.
                # `order=asc` is ignored by this endpoint (fastspot, behaviour 2), so never ask.
                page = fut._get(f"/futures/v1/quotes/{self.ticker}", limit=1)
                rows = page.get("results") or []
                if rows:
                    r = rows[0]
                    bid, ask = r.get("bid_price"), r.get("ask_price")
                    if bid and ask and ask >= bid:
                        with self._lock:
                            self._t_ns = int(r["timestamp"])
                            self._mid = (float(bid) + float(ask)) / 2.0
                self.polls += 1
            except Exception as e:                              # noqa: BLE001
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
            self._stop.wait(max(0.0, self.poll_s - (time.monotonic() - t0)))

    def mid(self, max_age_s=MAX_QUOTE_AGE_S):
        """(mid, age_s) or (None, age_s|None). Age is measured on the VENDOR's nanosecond stamp."""
        with self._lock:
            t_ns, mid = self._t_ns, self._mid
        if mid is None or t_ns is None:
            return None, None
        age = time.time() - t_ns / 1e9
        if age < 0 or age > max_age_s:
            return None, age
        return mid, age

    def detail(self):
        mid, age = self.mid()
        return {"product": self.product, "contract": self.ticker, "ok": self.ok,
                "mid": mid, "age_s": None if age is None else round(age, 3),
                "polls": self.polls, "errors": self.errors, "last_error": self.last_error}


class DeltaAdjuster:
    """Anchors on a futures mid at each board snapshot and carries theos forward from it."""

    def __init__(self, ticker, max_adj=MAX_ADJ_DOLLARS):
        self.ticker = ticker
        self.max_adj = max_adj
        self.anchor = None          # futures mid at the instant the board was priced
        self.anchor_age_s = None
        self.stats = {"adjusted": 0, "unadjusted": 0, "refused_size": 0}

    def anchor_now(self):
        """Sample the anchor. Call this AS CLOSE AS POSSIBLE to taking the option snapshot.

        The anchor and the fast leg must be sampled under one rule -- the same series, the same
        endpoint, the same staleness bound -- because the whole increment is their difference.
        """
        self.anchor, self.anchor_age_s = (self.ticker.mid() if self.ticker is not None
                                          else (None, None))
        return self.anchor

    def adjust(self, theo, delta):
        """(theo', detail). Returns `theo` unchanged, with a reason, whenever it cannot adjust."""
        d = {"delta_adj": 0.0, "f_anchor": self.anchor, "f_now": None,
             "f_move": None, "reason": None}
        if theo is None:
            d["reason"] = "no_theo"
            return theo, d
        if self.ticker is None or self.anchor is None:
            d["reason"] = "no_anchor"
            self.stats["unadjusted"] += 1
            return theo, d
        if delta is None:
            # An unquotable market, or one inside MIN_MINUTES_TO_CLOSE -- the edge manager
            # returns delta=None for both, and neither should be carried forward.
            d["reason"] = "no_delta"
            self.stats["unadjusted"] += 1
            return theo, d
        f_now, age = self.ticker.mid()
        if f_now is None:
            d["reason"] = f"futures_stale_{age:.1f}s" if age is not None else "no_futures_quote"
            self.stats["unadjusted"] += 1
            return theo, d
        move = f_now - self.anchor
        adj = float(delta) * move
        if abs(adj) > self.max_adj:
            # See MAX_ADJ_DOLLARS. On a $1 binary this size of "repricing" is a broken increment.
            d.update({"f_now": f_now, "f_move": move, "reason": "adj_over_max"})
            self.stats["refused_size"] += 1
            return theo, d
        self.stats["adjusted"] += 1
        d.update({"delta_adj": adj, "f_now": f_now, "f_move": move})
        return theo + adj, d

    def detail(self):
        return {**self.stats, "anchor": self.anchor,
                "futures": (self.ticker.detail() if self.ticker is not None else None)}
