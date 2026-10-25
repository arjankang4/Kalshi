"""The live skew curve: refit on a slow cadence, smoothed over a short window.

`pricing/skew` is the pure fitter -- one board snapshot in, one curve out, no memory.  This is the
stateful half that knows what time it is.  Same split as `pricing/varcal` (pure calendar) vs
`strategy/leanchain` (live state), and `pricing/edge` vs `strategy/edgemgr`.

WHY SMOOTH AT ALL.  A single-snapshot fit inherits every quote flicker on the board.  One leg
momentarily crossed or one bid pulled moves that bucket's implied vol, the spline pivots on it, and
the ratio it hands the whole ladder moves with it -- a bad quote on one strike repricing markets
that strike has nothing to do with.  Averaging the last 30 seconds costs a little responsiveness on
a genuine vol move and removes almost all of that.

WHY THE QUEUE IS KEYED BY BUCKET LABEL, NOT BY STRIKE.  Spot moves during a session, so the strike
sitting nearest +0.3 sigma changes -- several times an hour near the money.  Keying the history by
strike would start a fresh, empty series every time that happened, exactly when the curve is moving
and the smoothing is most wanted.  Keying by label means the +0.3 bucket keeps its history and the
new observation lands in the right series.

WHY RATIOS, NOT VOLS.  The level decays all day as variance is consumed; the shape does not.
Averaging w over 30 seconds smears the term structure into the smile.  Averaging w/w_atm does not.

WHY THE SMOOTHED CURVE IS BUILT ON NOMINAL x.  The raw fit puts each observation at the strike's
actual position in sigmas, which is right for a single snapshot.  Across a window those positions
drift, so the smoothed curve is rebuilt on each bucket's canonical position instead -- the same
place its label names.  Averaging observations taken at slightly different x and then serving them
at their nominal x is the approximation this module makes, and it is why the buckets are 0.3 sigma
apart rather than 0.05: the drift inside one bucket over 30 seconds is far smaller than the spacing.

FAILING SOFT IS THE CONTRACT.  Any failure -- no ATM vol, too few buckets, a non-monotone ladder --
returns a flat curve with a reason, which prices exactly as the pre-skew model did.  A skew that is
wrong is worse than no skew.
"""
import collections
import math
import time

import config
from pricing import skew as skewmod

# Defaults, overridable from config so a live run can be retuned without an edit here.
WINDOW_S = getattr(config, "SKEW_WINDOW_S", 60.0)
RECOMPUTE_S = getattr(config, "SKEW_RECOMPUTE_S", 3.0)
HALF_LIFE_S = getattr(config, "SKEW_HALF_LIFE_S", 15.0)

# Minimum fits per bucket before the smoothed value is served. Illustrative value; the live value is
# not published.
MIN_SAMPLES = 3


class SkewState:
    """A smoothed skew curve for one underlying.

    Call `observe(...)` as often as you like -- it refits at most every `recompute_s` and returns
    the current smoothed curve either way, so a fast quote loop pays only for the slower fit cadence.
    """

    def __init__(self, underlying, window_s=None, recompute_s=None, half_life_s=None):
        self.underlying = underlying
        self.window_s = WINDOW_S if window_s is None else window_s
        self.recompute_s = RECOMPUTE_S if recompute_s is None else recompute_s
        self.half_life_s = HALF_LIFE_S if half_life_s is None else half_life_s
        # label -> deque of (monotonic_t, ratio)
        self.queues = collections.defaultdict(collections.deque)
        self._last_fit_t = None      # None means never fitted, which is not the
                                     # same as 'fitted at t=0' -- see observe().
        self._raw = None            # the most recent single-snapshot fit
        self._smoothed = None       # what callers are served
        self.n_fits = 0
        self.reason = "no_observation"

    # --- queue -----------------------------------------------------------

    def _push(self, raw_fit, t):
        for b in raw_fit.buckets:
            q = self.queues[b["label"]]
            q.append((t, b["ratio"]))
        self._prune(t)

    def _prune(self, t):
        cutoff = t - self.window_s
        for label, q in list(self.queues.items()):
            while q and q[0][0] < cutoff:
                q.popleft()
            if not q:
                del self.queues[label]

    def _ema(self, label, t):
        """Recency-weighted mean of a bucket's ratios, weight 0.5 ** (age / half_life).

        An EMA over an explicitly bounded window rather than a recursive one: the window is short
        and the cadence is not perfectly regular (a slow REST cycle skips a beat), so weighting by
        measured age is both simpler to reason about and correct when a sample is late.
        """
        q = self.queues.get(label)
        if not q or len(q) < MIN_SAMPLES:
            return None
        num = den = 0.0
        for ts, r in q:
            wt = 0.5 ** ((t - ts) / self.half_life_s)
            num += wt * r
            den += wt
        return num / den if den > 0.0 else None

    # --- serving ---------------------------------------------------------

    def _rebuild(self, forward, w_atm, t, strikes=()):
        """Build the served curve from the smoothed per-bucket ratios.

        `strikes` is the LISTED grid, and passing it is load-bearing on two separate counts --
        both of which were missing until 2026-08-24, on the one code path that reaches the live
        quote loop:

          1. **`SkewFit(strikes=...)` is what arms `_projection`.** Without it the fit's strike
             tuple is empty, `_projection` short-circuits to `((), ())`, and the corrected
             digital is served RAW. The raw corrected ladder is non-monotone on 98.8% of boards
             (see `skew._projection`); the projection onto the non-increasing cone is what
             removes the negative rungs. Those rungs do not fail loudly -- `theo.bracket_fair`
             ends in `clamp(total)`, so a negative bracket becomes exactly 0.00 and the board
             quietly loses a rung. Inside 2 sigma the projection moves nothing (0.0000pp SPX /
             0.022pp NDX), which is why it was reasonable to call the omission nominal; on the
             wings it is the difference between a priced rung and a dead one.

          2. **`skewmod.ladder_monotone` is the acceptance guard, and it ran only inside
             `skew.fit`.** The smoothed curve is a DIFFERENT spline through DIFFERENT knots
             (EMA'd ratios at nominal x), so a raw fit passing the guard says nothing about the
             smoothed one. This path installed it into the live pricer via
             `validate_theo.resolve_and_price` without ever checking it. Now it fails soft to
             flat with `skew_non_monotone`, which is this module's stated contract: "a skew that
             is wrong is worse than no skew".

        Callers that genuinely have no grid may omit it; the guard then trivially passes and the
        projection stays off, i.e. exactly the old behaviour. Nothing live does that.
        """
        pts = []
        for x_nom in skewmod.bucket_offsets():
            r = self._ema(skewmod.bucket_label(x_nom), t)
            if r is None or not (skewmod.RATIO_MIN <= r <= skewmod.RATIO_MAX):
                continue
            pts.append((x_nom, r))
        if len(pts) < skewmod.MIN_BUCKETS:
            self.reason = "too_few_smoothed_buckets"
            return skewmod.SkewFit(forward, w_atm, None, [], self.reason)

        pts.sort()
        sp = skewmod.Spline([x for x, _ in pts], [r for _, r in pts])
        buckets = [{"label": skewmod.bucket_label(x), "x_nominal": x, "x": x,
                    "strike": None, "w": (w_atm * r if w_atm else None), "ratio": r}
                   for x, r in pts]
        ks = sorted(float(k) for k in (strikes or ()))
        out = skewmod.SkewFit(forward, w_atm, sp, buckets, None, strikes=ks)

        if ks and not skewmod.ladder_monotone(out, ks):
            self.reason = "skew_non_monotone"
            return skewmod.SkewFit(forward, w_atm, None, buckets, self.reason)

        self.reason = None
        return out

    def observe(self, quotes, strikes, forward, w_atm, now=None, force=False):
        """Fold one board snapshot in and return the curve to price with.

        `now` is a `time.monotonic()` reading; leave it None outside tests.
        """
        t = time.monotonic() if now is None else now
        # `is None` first: a monotonic clock can legitimately read near zero, and treating
        # 'never fitted' as 'fitted at 0' silently skips the very first fit.
        due = force or self._last_fit_t is None or (t - self._last_fit_t) >= self.recompute_s

        if due:
            self._last_fit_t = t
            raw = skewmod.fit(quotes, strikes, forward, w_atm)
            self._raw = raw
            if raw.buckets:
                self._push(raw, t)
                self.n_fits += 1
            else:
                self.reason = raw.reason or "raw_fit_empty"
            self._prune(t)

        # Rebuilt every call, not only on a refit: the queue ages between fits, so the weights
        # move even when no new observation arrived. It is a handful of arithmetic over <= 21
        # buckets -- cheaper than deciding whether it was worth skipping.
        self._smoothed = self._rebuild(forward, w_atm, t, strikes=strikes)
        return self._smoothed

    @property
    def curve(self):
        """The last served curve, or None before the first observation."""
        return self._smoothed

    def detail(self):
        """Diagnostics for the journal."""
        sm, raw = self._smoothed, self._raw
        return {"underlying": self.underlying,
                "skew_reason": self.reason,
                "skew_flat": (sm.flat if sm is not None else True),
                "n_fits": self.n_fits,
                "n_buckets_smoothed": (sm.n_buckets if sm is not None else 0),
                "n_buckets_raw": (raw.n_buckets if raw is not None else 0),
                "raw_reason": (raw.reason if raw is not None else None),
                "queue_depth": {k: len(v) for k, v in sorted(self.queues.items())},
                "window_s": self.window_s, "half_life_s": self.half_life_s,
                "recompute_s": self.recompute_s}
