"""The live realized-vol reader: a ring buffer of index levels, sampled at the quote cadence.

The quote loop has no second-bar stream -- it polls `/v3/snapshot/indices`. So this backfills the
trailing three hours ONCE from `/v2/aggs/.../1/second/` and then appends the snapshot value on
every poll. Three hours at 5-second polling is 2,160 points, which is nothing.

**Polling at 5s does not reintroduce the 5s Epps bias.** The bias comes from SAMPLING an index at
5s, not from observing it that often. Realized variance is read off this buffer at each lookback's
OWN fitted scale -- 60s for 3h/1h/30m and 10s for 15m/5m (`dataset.LOOKBACK_SCALE_S`) -- so the
estimator sees the same scale its coefficient was fitted at. A denser buffer only makes the
sub-grid averaging better.

That per-lookback scale was NOT passed until 2026-08-24: this module called `window_sigma` with
its default `scale_s`, which is 30 -- the TARGET scale, correct for the thing being predicted and
wrong for every one of the five predictors. The training set built `f_3h` at 60s and served it at
30s, and `f_5m` at 10s and served it at 30s, so two of the features were biased in opposite
directions against the betas multiplying them. This is not a rounding matter: measured 2026-08-20,
annualized RTH realized vol on `I:NDX` reads 9.26% at 1s against 13.76% at 30s (the Epps effect,
because constituents do not all trade every second), so scale IS the level.

What this deliberately does NOT do is invent history. If the backfill fails, the buffer starts
empty and the long lookbacks return None until it fills -- which makes the lean inert rather than
wrong. A lean built on a 20-minute buffer pretending to be a 3-hour lookback is worse than no lean.
"""
import datetime
import threading

import numpy as np

import config
from massive import index as ix
from pricing import varcal
from massive import history as mhist
from rvol import data as rdata
from rvol import features as F
from rvol.dataset import LOOKBACK_SCALE_S, LOOKBACKS, SHORT_LOOKBACKS

BUFFER_HOURS = 3.0


class IndexBuffer:
    """Thread-safe (t_ms, level) ring for one underlying, plus the pre-open proxy series."""

    def __init__(self, underlying, hours=BUFFER_HOURS):
        self.underlying = underlying
        self.hours = hours
        self._lock = threading.Lock()
        self._t = []
        self._p = []
        self.proxy_t = None
        self.proxy_p = None
        self.backfilled = False

    def backfill(self, now=None):
        """Load today's index seconds and the ETF proxy's extended-hours seconds.

        Both come from the aggregates endpoint rather than the cache, because the cache is a
        research artefact that stops at the last build. This is the live path.
        """
        now = now or datetime.datetime.now(config.ET)
        d = now.astimezone(config.ET).date()
        src = rdata.SOURCES[self.underlying]
        try:
            rows = ix.closes_ms(self.underlying, d)
            with self._lock:
                self._t = [r[0] for r in rows]
                self._p = [r[1] for r in rows]
            bars = mhist.second_bars(src["proxy"], str(d), str(d))
            pr = [(b["t"], b["c"]) for b in bars if b.get("c") and b["c"] > 0]
            self.proxy_t = np.asarray([r[0] for r in pr], dtype="int64")
            self.proxy_p = np.asarray([r[1] for r in pr], dtype="float64")
            self.backfilled = True
        except Exception as e:                                   # noqa: BLE001
            print(f"[rvol.live] backfill failed ({type(e).__name__}: {str(e)[:80]}) -- "
                  f"lean stays inert until the buffer fills", flush=True)
            self.backfilled = False
        return self.backfilled

    def append(self, quote):
        """Append one IndexQuote. Ignores a repeat of the same vendor timestamp."""
        if quote is None or quote.ts_ns is None:
            return False
        t_ms = quote.ts_ns // 1_000_000
        with self._lock:
            if self._t and t_ms <= self._t[-1]:
                return False
            self._t.append(t_ms)
            self._p.append(float(quote.value))
            cutoff = t_ms - int(self.hours * 3600 * 1000)
            while self._t and self._t[0] < cutoff:
                self._t.pop(0)
                self._p.pop(0)
        return True

    def arrays(self):
        with self._lock:
            return np.asarray(self._t, dtype="int64"), np.asarray(self._p, dtype="float64")

    def span_s(self):
        with self._lock:
            return (self._t[-1] - self._t[0]) / 1000.0 if len(self._t) > 1 else 0.0


class LiveRealizedVol:
    """Features off the buffer, cached to `config.LEAN_RECOMPUTE_S`."""

    def __init__(self, underlying, calendar=None, buffer=None):
        self.underlying = underlying
        self.cal = calendar or varcal.load(underlying)
        self.buf = buffer or IndexBuffer(underlying)
        self._cache = None
        self._cache_t = 0.0
        self._cache_now = None
        self._short = None
        self._short_t = 0.0

    def features(self, now=None, force=False):
        """{'f_3h':..., 'pf_3h':..., 'cov_3h':...} or None. Recomputed at most every 5s."""
        import time
        t = time.monotonic()
        if not force and self._cache is not None and (t - self._cache_t) < config.LEAN_RECOMPUTE_S:
            return self._cache
        now = now or datetime.datetime.now(config.ET)
        ts, px = self.buf.arrays()
        if ts.size < 2:
            return None
        out = {}
        for name, mins in LOOKBACKS:
            # `scale_s=` is the whole point -- see the module docstring. Serving a feature at a
            # scale its beta was not fitted at is a silent level error, not an approximation.
            sig, pf, cov = F.window_sigma(self.cal, ts, px, self.buf.proxy_t, self.buf.proxy_p,
                                          now - datetime.timedelta(minutes=mins), now,
                                          scale_s=LOOKBACK_SCALE_S[name])
            out[f"f_{name}"], out[f"pf_{name}"], out[f"cov_{name}"] = sig, pf, cov
        self._cache, self._cache_t, self._cache_now = out, t, now
        return out

    def short_forecasts(self, model, now=None, force=False):
        """{'1m': sigma|None, '3m': ..., '30m': ..., 'f_1m': ...} or None, on the same 5s cadence.

        The near-term forecasts for `pricing/volspike`. Reuses `features()` (and its cache) for the
        five long lookbacks, and computes only what a short model reads beyond them -- today at most
        `f_1m`, one extra `window_sigma` over a 60-second window at 10s. `features()` itself is left
        exactly as it was, so the lean's journalled `rv_features` do not change.

        `30m` is the LONG model's 30m forecast -- the reference `volspike.signal` divides by.
        Every value is None when the model has no `short_models` (a production artifact that
        predates them), which the caller journals as the reason.
        """
        import time
        t = time.monotonic()
        if not force and self._short is not None and (t - self._short_t) < config.LEAN_RECOMPUTE_S:
            return self._short
        feats = self.features(now, force=force)
        if not feats:
            return None
        feats = dict(feats)
        want = set(model.short_features()) if hasattr(model, "short_features") else set()
        at = self._cache_now or now or datetime.datetime.now(config.ET)
        if want:
            ts, px = self.buf.arrays()
            for name, mins in SHORT_LOOKBACKS:
                if f"f_{name}" not in want:
                    continue
                sig, pf, cov = F.window_sigma(self.cal, ts, px, self.buf.proxy_t,
                                              self.buf.proxy_p,
                                              at - datetime.timedelta(minutes=mins), at,
                                              scale_s=LOOKBACK_SCALE_S[name])
                feats[f"f_{name}"], feats[f"pf_{name}"], feats[f"cov_{name}"] = sig, pf, cov
        out = {h: model.predict_short(h, feats) for h in getattr(model, "short_models", {})}
        out["30m"] = model.predict("30m", feats)
        for name, _ in SHORT_LOOKBACKS:
            if f"f_{name}" in feats:
                out[f"f_{name}"] = feats[f"f_{name}"]
        self._short, self._short_t = out, t
        return out
