"""Fast spot: the futures level, carried on the index's own scale.

    spot_fast = F_now - EMA(F - index, sampled when the index ticks)

WHAT THIS IS AND IS NOT. It is NOT a change of pricing reference: KXINX settles on the INDEX at
16:00, so the index remains the thing being predicted, and everything downstream -- `theo.Pricer`,
`band.classify`, `edgemgr.refresh` -- keeps taking a number on the index's scale. What changes is
only how fast we learn that number. The futures supply the motion, the EMA supplies the level, and
the output is an estimate of the index NOW rather than of the index up to 1.7 seconds ago.

WHY IT IS WORTH DOING, measured 2026-08-26 on the session that lost money:

  * The index publishes on a one-second grid and we read it over REST at a median age of 711ms
    (p95 1,472ms, max 3.0s). So on a move we learn late by up to ~1.7s.
  * SPX fell 5.15 points in 16 seconds at 11:51:41 -- 0.86 pt/s -- straight through the strike of
    an hourly board nine minutes from settlement. 154 fills landed in that minute, 76 of them
    inside one 2-second cycle, and the markout against our own theo was -18.17c mean.
  * The basis is far quieter than the move it would let us see: across 1,081 cycles that session,
    SPX `futures - index` had median 14.15, p05 13.64, p95 14.64, **stdev 0.316 points**.

So the trade is 0.32 points of basis noise against 1-3 points of lag. That is the whole argument,
and it is an empirical one -- if the basis ever becomes as noisy as the lag it replaces, this
stops being worth it, which is why `detail()` reports the residual and `alarm` fires on it.

WHY THE EMA IS SAMPLED ON THE INDEX TICK, not on a timer and not on every futures frame. The
quantity being averaged is `F - index`, and it is only meaningful at an instant when BOTH are
current. The index is the slow leg, so its arrival is the only moment when the difference is not
mostly a measurement of how far F has moved since the index was last computed. Sampling on
futures frames instead would average in F's own noise 114 times a second and bias the basis toward
whatever F did most recently -- exactly the signal we are trying to keep.

FAILURE IS SOFT AND DIRECTIONAL. No futures, no basis, or a residual outside the alarm band, and
`spot()` returns None with a reason; the caller falls back to `pricing/spot.resolve`, i.e. to
today's behaviour. A fast spot we cannot justify is worse than a slow one we can.
"""
import math
import time

import config

# Half-life of the index-minus-futures basis EMA. Read via getattr(config, "BASIS_EMA_HALF_LIFE_S"),
# which config.py does not define. Illustrative value; the live value is not published.
DEFAULT_HALF_LIFE_S = 45.0

# Alarm when the basis breaks by more than this fraction of spot. Same getattr pattern. Illustrative
# value; the live value is not published.
DEFAULT_ALARM_PCT = 0.001

# Basis samples required before the EMA is trusted. Illustrative value; the live value is not
# published.
MIN_SAMPLES = 20


class BasisTracker:
    """One underlying's `EMA(futures - index)`, and the fast spot it yields."""

    __slots__ = ("underlying", "half_life_s", "alarm_pct", "min_samples",
                 "_ema", "_t", "_n", "_last_index_ts", "_last_resid", "_max_resid",
                 "_refused", "_last_level")

    def __init__(self, underlying, half_life_s=None, alarm_pct=None,
                 min_samples=MIN_SAMPLES):
        self.underlying = underlying
        self.half_life_s = DEFAULT_HALF_LIFE_S if half_life_s is None else half_life_s
        self.alarm_pct = DEFAULT_ALARM_PCT if alarm_pct is None else alarm_pct
        self._last_level = None
        self.min_samples = min_samples
        self._ema = None
        self._t = None
        self._n = 0
        self._last_index_ts = None
        self._last_resid = None
        self._max_resid = 0.0
        self._refused = {}

    # -- the slow leg --

    def observe(self, futures_mid, index_quote, now=None):
        """Fold one (futures, index) pair into the EMA. Call this when the INDEX ticks.

        Returns True if it was folded. Ignores a repeated index timestamp, because the index's
        one-second grid means a repeat is the same observation seen twice, and folding it twice
        would silently halve the effective half-life.
        """
        if futures_mid is None or index_quote is None or index_quote.value is None:
            return False
        ts = index_quote.ts_ns
        if ts is not None and ts == self._last_index_ts:
            return False
        self._last_index_ts = ts
        self._last_level = float(index_quote.value)
        now = time.monotonic() if now is None else now
        b = float(futures_mid) - float(index_quote.value)
        if self._ema is None:
            self._ema, self._t, self._n = b, now, 1
            return True
        # TIME-WEIGHTED, not per-sample: the index grid slips and a reconnect can drop ticks, and
        # a per-sample EMA would then weight a 3-second gap the same as a 1-second one.
        dt = max(0.0, now - self._t)
        w = 0.5 ** (dt / self.half_life_s) if self.half_life_s > 0 else 0.0
        self._ema = w * self._ema + (1.0 - w) * b
        self._t = now
        self._n += 1
        self._last_resid = b - self._ema
        self._max_resid = max(self._max_resid, abs(self._last_resid))
        return True

    # -- the fast leg --

    @property
    def ready(self):
        return self._ema is not None and self._n >= self.min_samples

    @property
    def basis(self):
        return self._ema

    def spot(self, futures_mid):
        """(fast_spot, detail). `fast_spot` is None whenever it cannot be justified."""
        d = {"underlying": self.underlying, "basis": self._ema, "n": self._n,
             "residual": self._last_resid, "reason": None, "alarm": False}
        if futures_mid is None:
            d["reason"] = "no_futures"
            self._refused["no_futures"] = self._refused.get("no_futures", 0) + 1
            return None, d
        if not self.ready:
            d["reason"] = f"basis_warming_{self._n}/{self.min_samples}"
            self._refused["warming"] = self._refused.get("warming", 0) + 1
            return None, d
        # THE BREAK DETECTOR. The residual is measured on the last INDEX tick, so this catches a
        # basis that has stepped -- a contract roll, a halt, a feed pointed at the wrong month --
        # rather than a basis that is merely drifting, which the EMA already tracks.
        band = self.alarm_pct * (self._last_level or 0.0)
        if self._last_resid is not None and band > 0 and abs(self._last_resid) > band:
            d["reason"] = f"basis_break_{self._last_resid:+.2f}pts"
            d["alarm"] = True
            self._refused["break"] = self._refused.get("break", 0) + 1
            return None, d
        s = float(futures_mid) - self._ema
        if not math.isfinite(s) or s <= 0:
            d["reason"] = "non_finite"
            return None, d
        d["spot"] = s
        return s, d

    def detail(self):
        return {"underlying": self.underlying, "basis": self._ema, "n": self._n,
                "half_life_s": self.half_life_s, "ready": self.ready,
                "last_residual": self._last_resid,
                "max_abs_residual": round(self._max_resid, 4),
                "alarm_pct": self.alarm_pct,
                "alarm_points": (None if self._last_level is None
                                 else round(self.alarm_pct * self._last_level, 3)),
                "refused": dict(self._refused)}
