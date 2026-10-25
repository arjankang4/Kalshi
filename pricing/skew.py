"""A fitted volatility skew across strikes, in units of standard deviations from the money.

Everything upstream of this file prices every strike off one ATM vol.  `pricing/hourly` names that
as its assumption #1 -- "Zero skew.  One ATM vol prices every strike.  Defensible where it matters
-- inside ~3 sigma of the forward -- and increasingly not, outside it."  On a 60-rung SPX ladder
only ~19 rungs are inside that window, so most of the board is priced by an assumption the board
itself contradicts.  SPX and NDX have real skew, and it is directly observable in the quotes we
already collect.

**The x-axis is standard deviations, not strikes.**  `x = log(K/F) / w_atm`.  That is the only
choice that makes the curve portable: a fixed strike offset means something different on SPX at
7700 than on NDX at 29000, and something different at 09:30 than at 15:55 as `w` decays toward
zero.  In x-space the smile is roughly stationary over the session, which is what makes a queue of
past observations meaningful at all.

**What is fitted is a RATIO, not a level.**  Each bucket contributes `r_i = w_i / w_atm`, and the
spline interpolates `r`.  Serving is `w(K) = w_atm * r(x(K))`.  Two reasons.  The level moves all
day as variance is consumed while the *shape* is far more stable, so ratios are what a 30-second
queue can average without smearing the term structure into it.  And it makes the zero-skew case an
exact identity: a board with no skew fits `r == 1` everywhere and reproduces the flat-vol pricer
bit for bit, which is the strongest check in `validate_skew` and the same trick `varcal` uses at
16:00.

**Per-strike vol comes from both legs, weighted by inverse width IN VOLS.**  For a leg,
`width_vol = (ask - bid) / vega_w`, and the weights are `1/width_vol` normalised.  A leg quoted 6
vols wide against one quoted 2 vols wide gets a quarter of the weight.

That is not the vega weighting `theo.implied_w_at` warns about.  Put-call parity makes vega
*identical* for the call and the put at a strike, so weighting **by vega** is a silent no-op -- an
earlier version of `implied_w_at` did exactly that and just averaged the legs.  Weighting by width
*expressed in vols* is a different quantity: vega cancels out of the ratio, leaving inverse price
width, and the deep wings protect themselves because a small vega inflates `width_vol` for both
legs equally while the ITM leg's wider price quote sinks its own weight.  The measured pathology
that motivated OTM-only -- a 7975 put quoted 259.5/280.0, 20.5 points wide, implying w = 0.0139
against an ATM 0.0024 -- is exactly the case this down-weights to near nothing rather than
excluding by rule.

**Degrade, never fabricate.**  A bucket with no invertible two-sided quote contributes nothing.
If too few buckets survive, or if the resulting digital ladder is not monotone in strike, the fit
refuses and the caller falls back to flat ATM vol with a stated reason.  A skew curve that is
wrong is worse than no skew curve, because the flat model is at least wrong in a known direction.
"""
import math

import config
from pricing import bs, theo

MISSING = None

# Smile knot spacing, in sigmas. Illustrative value; the live value is not published.
BUCKET_STEP = 0.25
# Tunable. Illustrative value; the live value is not published.
BUCKET_MAX = 2.5

# GATE: with fewer populated buckets than this, fall back to flat vol. Illustrative value; the live
# value is not published.
MIN_BUCKETS = 4

# GATE: reject a smile bucket whose vol / ATM vol is below this. Illustrative value; the live value
# is not published.
RATIO_MIN = 0.2
# GATE: reject a smile bucket whose vol / ATM vol is above this. Illustrative value; the live value
# is not published.
RATIO_MAX = 5.0

# Smallest width, in total-vol units, that a leg is credited with. Only bites on a zero-wide
# (synthetic or touching) quote, where it keeps the weight finite instead of infinite.
MIN_WIDTH_VOL = 1e-9


# Finite-difference step for dw/dK in the slope term. Illustrative value; the live value is not
# published.
SLOPE_STEP_X = 0.04

# The projection (see `SkewFit._projection`) only has to cover the strikes where the digital is
# not already saturated: beyond 5 sigma `N(d2)` is inside 3e-7 of 0 or 1 and the slope term is
# numerically zero, so the ladder is monotone there whatever the curve does. Every violation
# measured on the 1s board sits inside |x| = 3.1. Bounding the window is worth 2.2x on the whole
# fit-plus-board path (5.5 ms -> 2.5 ms on a 500-strike SPX grid), which matters because the live
# quote loop prices a board per cycle.
PROJECTION_XMAX = 5.0


def _pava_nonincreasing(y):
    """L2 projection onto the non-increasing cone (pool-adjacent-violators). O(n), no scipy."""
    n = len(y)
    val = [0.0] * n
    wt = [0.0] * n
    idx = 0
    for v in y:
        val[idx] = float(v)
        wt[idx] = 1.0
        idx += 1
        while idx > 1 and val[idx - 2] < val[idx - 1]:
            w = wt[idx - 2] + wt[idx - 1]
            val[idx - 2] = (wt[idx - 2] * val[idx - 2] + wt[idx - 1] * val[idx - 1]) / w
            wt[idx - 2] = w
            idx -= 1
    out = []
    for k in range(idx):
        out.extend([val[k]] * int(wt[k]))
    return out


def _interp(x, xs, ys):
    """Linear interpolation on a sorted grid, flat outside it."""
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0.0:
        return ys[lo]
    return ys[lo] + (x - xs[lo]) / span * (ys[hi] - ys[lo])


def bucket_offsets(step=BUCKET_STEP, xmax=BUCKET_MAX):
    """The nominal knot positions, in sigmas: -3.0, -2.7, ... 0.0 ... +2.7, +3.0."""
    n = int(round(xmax / step))
    return [round(i * step, 10) for i in range(-n, n + 1)]


def bucket_label(x):
    """Stable name for a bucket.

    The label, not the strike, is the identity that a live queue keys on.  Spot moves during a
    session, so the strike sitting in the "+0.3 sigma" bucket changes -- and when it does we still
    want the new observation appended to the +0.3 series rather than starting a new one.
    """
    if abs(x) < 1e-12:
        return "atm"
    return f"{x:+.1f}"


def leg_vol(F, K, right, q, w_ref):
    """(implied w, weight) for one leg, or (MISSING, 0.0).

    Requires BOTH sides quoted. `theo.quote_value` deliberately returns the bid alone for a
    one-sided quote, and such a quote has no width, so it would enter the average carrying a
    confidence it has not earned. Zero width is fine (see the floor below); absent is not.
    """
    if not q:
        return MISSING, 0.0
    bid, ask = q.get("bp"), q.get("ap")
    if bid is None or ask is None or ask < bid:
        return MISSING, 0.0
    price = theo.quote_value(q)
    if price is MISSING:
        return MISSING, 0.0
    w = bs.implied_w(F, K, price, right)
    if w is MISSING or w <= 0.0:
        return MISSING, 0.0
    vega = bs.vega_w(F, K, w_ref)
    if vega <= 0.0:
        return MISSING, 0.0
    # Floored, not rejected, at zero width. A synthetic or crossed-to-touching quote has no
    # width to divide by, and the honest reading of a zero-wide market is "maximally informative",
    # not "uninformative" -- rejecting it would make an exact board unfittable.
    width_vol = max((ask - bid) / vega, MIN_WIDTH_VOL)
    return w, 1.0 / width_vol


def strike_vol(quotes, F, K, w_ref):
    """Total vol at one strike, from both legs weighted by inverse width-in-vols.

    Returns `(w, detail)`.  `w` is MISSING when neither leg inverts.
    """
    d = {"w_call": None, "w_put": None, "wt_call": 0.0, "wt_put": 0.0, "w": None}
    wc, tc = leg_vol(F, K, "C", quotes.get(("C", K)), w_ref)
    wp, tp = leg_vol(F, K, "P", quotes.get(("P", K)), w_ref)
    d["w_call"], d["w_put"], d["wt_call"], d["wt_put"] = wc, wp, tc, tp
    tot = tc + tp
    if tot <= 0.0:
        return MISSING, d
    acc = 0.0
    if wc is not MISSING:
        acc += wc * tc
    if wp is not MISSING:
        acc += wp * tp
    d["w"] = acc / tot
    return d["w"], d


# --- natural cubic spline ------------------------------------------------
#
# Hand-rolled rather than scipy: every module under pricing/ imports stdlib and config only, and
# this has to run in the seconds before a quote cycle.  A natural cubic spline over <=21 knots is
# a tridiagonal solve -- the Thomas algorithm below is O(n) and exact.

class Spline:
    """Natural cubic spline through (x, y), with FLAT extrapolation outside the knots.

    Flat, not linear or cubic: past the last knot a cubic diverges and can drive the vol ratio
    negative, and these are precisely the far wings whose quotes are least trustworthy.  Holding
    the endpoint ratio is the conservative reading -- it says "we stop learning out here", which
    is true.
    """

    __slots__ = ("x", "y", "m")

    def __init__(self, x, y):
        self.x, self.y = list(x), list(y)
        n = len(self.x)
        if n < 3:
            self.m = [0.0] * n
            return
        h = [self.x[i + 1] - self.x[i] for i in range(n - 1)]
        alpha = [0.0] * n
        for i in range(1, n - 1):
            alpha[i] = 3.0 * ((self.y[i + 1] - self.y[i]) / h[i]
                              - (self.y[i] - self.y[i - 1]) / h[i - 1])
        l, mu, z = [1.0] + [0.0] * (n - 1), [0.0] * n, [0.0] * n
        for i in range(1, n - 1):
            l[i] = 2.0 * (self.x[i + 1] - self.x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        c = [0.0] * n
        for i in range(n - 2, -1, -1):
            c[i] = z[i] - mu[i] * c[i + 1]
        self.m = c

    def __call__(self, xq):
        x, y, c = self.x, self.y, self.m
        n = len(x)
        if n == 0:
            return MISSING
        if n == 1:
            return y[0]
        if xq <= x[0]:
            return y[0]                       # flat extrapolation
        if xq >= x[-1]:
            return y[-1]                      # flat extrapolation
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x[mid] <= xq:
                lo = mid
            else:
                hi = mid
        h = x[lo + 1] - x[lo]
        if n < 3:
            f = (xq - x[lo]) / h
            return y[lo] + f * (y[lo + 1] - y[lo])
        b = (y[lo + 1] - y[lo]) / h - h * (2.0 * c[lo] + c[lo + 1]) / 3.0
        dd = (c[lo + 1] - c[lo]) / (3.0 * h)
        dx = xq - x[lo]
        return y[lo] + b * dx + c[lo] * dx * dx + dd * dx * dx * dx


class SkewFit:
    """The fitted curve, plus everything needed to see why it looks the way it does."""

    __slots__ = ("forward", "w_atm", "spline", "buckets", "reason", "n_buckets", "flat",
                 "strikes", "_proj")

    def __init__(self, forward, w_atm, spline, buckets, reason, strikes=None):
        self.forward, self.w_atm = forward, w_atm
        self.spline, self.buckets, self.reason = spline, buckets, reason
        self.n_buckets = len(buckets)
        self.flat = spline is None
        # The listed grid the curve was fitted against. Needed only to project the digital ladder
        # back onto the monotone cone -- see `_projection`. A fit built without it (the live
        # `SkewState` EMA path) still gets the slope term, just not the projection; measured, the
        # projection is worth <0.001pp inside 2 sigma, so that degradation is nominal.
        self.strikes = tuple(sorted(float(k) for k in strikes)) if strikes else ()
        self._proj = {}

    @property
    def ok(self):
        return self.reason is None and self.spline is not None

    def x_of(self, K):
        if not self.forward or K <= 0.0 or not self.w_atm:
            return MISSING
        return math.log(K / self.forward) / self.w_atm

    def ratio(self, K):
        """w(K) / w_atm.  Exactly 1.0 when the fit refused, which is the flat-vol fallback."""
        if self.flat:
            return 1.0
        x = self.x_of(K)
        if x is MISSING:
            return 1.0
        r = self.spline(x)
        if r is MISSING or not (RATIO_MIN <= r <= RATIO_MAX):
            return 1.0
        return r

    def w_of(self, K):
        if self.w_atm is None:
            return MISSING
        return self.w_atm * self.ratio(K)

    # ---- the digital -----------------------------------------------------------------
    #
    # A binary is the strike-DERIVATIVE of the call curve, and with a strike-dependent vol the
    # call depends on K twice:
    #
    #     C(K)   = BS(F, K, w(K))
    #     -dC/dK = N(d2)  -  F * phi(d1) * dw/dK
    #
    # `p_above_flatvol` is the FIRST TERM ONLY, which is what this repo priced until 2026-08-24 and
    # what `notebooks/05_taking_postmortem.ipynb` §7 shows is wrong by up to +4.0 probability points
    # at the money. `p_above` is the whole derivative. See bias-ledger F27/F28.

    def dratio_dK(self, K, step_x=SLOPE_STEP_X):
        """d(ratio)/dK of the fitted curve at K, differenced `step_x` sigmas either side.

        Zero for a refused fit, which is right: a flat curve has no slope and the term vanishes,
        so the corrected digital degrades to exactly the old one wherever the smile is off.
        """
        if self.flat or not self.forward or not self.w_atm or K <= 0.0:
            return 0.0
        x = self.x_of(K)
        if x is MISSING:
            return 0.0
        ka = self.forward * math.exp((x - step_x) * self.w_atm)
        kb = self.forward * math.exp((x + step_x) * self.w_atm)
        if kb <= ka:
            return 0.0
        return (self.ratio(kb) - self.ratio(ka)) / (kb - ka)

    def slope_term(self, K, w_atm=None):
        """`- F * phi(d1) * dw/dK`, in probability units, at ATM vol `w_atm`.

        `w_atm` defaults to the fit's own. The HOURLY board passes its horizon vol instead: the
        ratio curve is read where it was observed (16:00 sigmas) but the vol it multiplies is the
        horizon's, so `dw/dK = w_horizon * dratio/dK` and the term scales with it.
        """
        wa = self.w_atm if w_atm is None else w_atm
        if self.flat or not wa or wa <= 0.0 or not self.forward or K <= 0.0:
            return 0.0
        w_k = wa * self.ratio(K)
        if w_k <= 0.0:
            return 0.0
        d1 = math.log(self.forward / K) / w_k + w_k / 2.0
        return -self.forward * bs.norm_pdf(d1) * (wa * self.dratio_dK(K))

    def p_above_flatvol(self, K):
        """`N(d2)` at the smile's vol for K. The pre-2026-08-24 digital. Kept for the guard and
        for A/B: `_ladder_monotone` deliberately still runs on THIS, so which boards accept a fit
        is unchanged by the correction and old and new runs share one population."""
        w = self.w_of(K)
        if w is MISSING:
            return MISSING
        return bs.p_above(self.forward, K, w)

    def _projection(self, wa):
        """`projected - raw` on the listed ladder: the monotone repair, which is ~always zero.

        A cubic spline through noisy bucket ratios is not butterfly-arbitrage-free, and the old
        digital could not see that because it never used the slope. The corrected one can: on the
        1s option board the raw corrected ladder is non-monotone on 98.8% of boards -- by a mean
        worst step of 0.47pp (SPX) / 0.90pp (NDX) on ~3 and ~8 rungs of ~450, essentially all
        beyond |x| = 2.7. Projecting onto the non-increasing cone (PAVA, the L2 projection) repairs
        it and removes every negative rung. *Measured* displacement: mean 0.003pp (SPX) / 0.014pp
        (NDX) over the whole ladder, and inside 2 sigma **0.0000pp / 0.022pp** -- i.e. nothing the
        study reads moves. Carried as an ADJUSTMENT rather than a replacement so that off-grid
        boundaries keep the exact analytic value wherever the projection does not bind.
        """
        key = round(float(wa), 12)
        if key in self._proj:
            return self._proj[key]
        ks = self.strikes
        if self.flat or not ks:
            out = ((), ())
            self._proj[key] = out
            return out
        lo = self.forward * math.exp(-PROJECTION_XMAX * self.w_atm)
        hi = self.forward * math.exp(PROJECTION_XMAX * self.w_atm)
        ks = tuple(k for k in ks if lo <= k <= hi)
        if len(ks) < 3:
            out = ((), ())
            self._proj[key] = out
            return out
        raw = []
        for k in ks:
            r = self.ratio(k)
            w_k = wa * r
            raw.append(bs.p_above(self.forward, k, w_k) + self.slope_term(k, w_atm=wa)
                       if w_k > 0.0 else 0.0)
        adj = tuple(v - r_ for v, r_ in zip(_pava_nonincreasing(raw), raw))
        out = (ks, adj)
        self._proj[key] = out
        return out

    def p_above_at(self, K, w_atm=None):
        """`P(S > K) = -dC/dK` on the fitted curve, at ATM vol `w_atm` (default: the fit's own)."""
        wa = self.w_atm if w_atm is None else w_atm
        if wa is None:
            return MISSING
        r = self.ratio(K)
        w_k = wa * r
        if w_k <= 0.0:
            return MISSING
        p = bs.p_above(self.forward, K, w_k)
        if not config.DIGITAL_SLOPE_TERM or self.flat:
            return p
        p += self.slope_term(K, w_atm=wa)
        ks, adj = self._projection(wa)
        if ks:
            p += _interp(K, ks, adj)
        return min(1.0, max(0.0, p))

    def p_above(self, K):
        return self.p_above_at(K)

    def as_dict(self):
        return {"forward": self.forward, "w_atm": self.w_atm, "reason": self.reason,
                "n_buckets": self.n_buckets, "flat": self.flat,
                "buckets": [{"label": b["label"], "x": b["x"], "strike": b["strike"],
                             "w": b["w"], "ratio": b["ratio"]} for b in self.buckets]}


def atm_vol(quotes, strikes, F, max_steps=3):
    """Total vol at the money, for callers that do not already have an `atmvol.AtmFit`.

    `HourlyPricer` carries `fit.w` from `atmvol` and should pass that.  The daily pricer does not:
    it anchors on its own spot (the index when fresh, parity otherwise), and pulling in atmvol's
    separately-derived parity forward would mix two forwards into one curve -- the ratio would then
    absorb the difference between them and call it skew.

    Walks outward from the nearest listed strike so a single uninvertible ATM quote does not sink
    the whole fit.
    """
    if not strikes or not F:
        return MISSING
    ks = sorted(float(k) for k in strikes)
    i0 = min(range(len(ks)), key=lambda i: abs(ks[i] - F))
    for step in range(max_steps + 1):
        # A LIST ordered by distance from the forward, not a set. `{i0 - step, i0 + step}` is a
        # set literal, so which side of the money is tried first came out of Python's hash order
        # -- meaning the ATM vol served when the nearest strike is uninvertible was arbitrary but
        # STABLE within a process, so it never looked random, it just silently favoured one wing.
        # The strike grid is uneven (5-wide near the money, up to 200-wide in the tails), so the
        # two candidates are equidistant in INDEX and not in price; the nearer one in price is the
        # better ATM proxy and is now the one tried first. `i` breaks exact ties deterministically.
        cand = [i0] if step == 0 else [i for i in (i0 - step, i0 + step) if 0 <= i < len(ks)]
        cand.sort(key=lambda i: (abs(ks[i] - F), i))
        for i in cand:
            w, _ = strike_vol(quotes, F, ks[i], _W_PROBE)
            if w is not MISSING:
                return w
    return MISSING


# A reference total-vol used only to turn a price width into a vol width when no ATM vol is known
# yet.  It cancels out of the leg WEIGHTS (both legs share it), so its level only matters for the
# vega>0 guard -- any plausible 0DTE number does.
_W_PROBE = 0.005


def fit(quotes, strikes, forward, w_atm, step=BUCKET_STEP, xmax=BUCKET_MAX,
        min_buckets=MIN_BUCKETS):
    """Fit the skew curve for one board snapshot.

    `quotes` is the canonical `{('C'|'P', strike): Q-frame}` board; `strikes` is the FULL listed
    grid, not the subscribed subset.  `forward` and `w_atm` come from `atmvol.fit`.

    A refusal is a `SkewFit` with `flat=True` and a `reason` -- never an exception, and never a
    half-fitted curve.  `ratio()` then returns 1.0 everywhere, so the caller gets the flat-vol
    model it would have had anyway.
    """
    strikes = sorted(float(k) for k in (strikes or ()))
    if not strikes or not forward or not w_atm or w_atm <= 0.0:
        return SkewFit(forward, w_atm, None, [], "no_atm_fit")

    buckets, seen_strikes = [], set()
    for x_nom in bucket_offsets(step, xmax):
        target = forward * math.exp(x_nom * w_atm)
        K = min(strikes, key=lambda k: abs(k - target))
        if K in seen_strikes:
            # Two knots landed on the same listed strike -- the grid is coarser than the bucket
            # spacing here.  Taking it twice would double-weight one quote and, worse, pin the
            # spline to two different x for one observation.
            continue
        w, d = strike_vol(quotes, forward, K, w_atm)
        if w is MISSING:
            continue
        r = w / w_atm
        if not (RATIO_MIN <= r <= RATIO_MAX):
            continue
        seen_strikes.add(K)
        # The abscissa is the strike's ACTUAL position in sigmas, not the knot's nominal one.
        # The vol was observed at K, so that is where it belongs on the curve; the nominal label
        # survives only as the bucket's identity for the live queue.
        buckets.append({"label": bucket_label(x_nom), "x_nominal": x_nom,
                        "x": math.log(K / forward) / w_atm, "strike": K,
                        "w": w, "ratio": r, "detail": d})

    if len(buckets) < min_buckets:
        return SkewFit(forward, w_atm, None, buckets, "too_few_buckets")

    buckets.sort(key=lambda b: b["x"])
    sp = Spline([b["x"] for b in buckets], [b["ratio"] for b in buckets])
    out = SkewFit(forward, w_atm, sp, buckets, None, strikes=strikes)

    if not ladder_monotone(out, strikes):
        return SkewFit(forward, w_atm, None, buckets, "skew_non_monotone")
    return out


def ladder_monotone(sf, strikes, tol=1e-12):
    """P(S > K) must be non-increasing in K.

    A spline through noisy quotes can wobble enough to invert the digital ladder, which shows up
    downstream as a bracket with negative probability -- `bracket_fair` would clamp it to zero and
    the board would silently lose a rung.  Cheaper to catch it here and fall back to flat.

    **Deliberately run on `p_above_flatvol`, not on the corrected digital.** Two reasons. It keeps
    the set of boards that accept a fit identical to what it was before the slope term existed, so
    the before/after comparison in nb05 §8 changes exactly one thing. And the corrected ladder is
    monotone BY CONSTRUCTION after `_projection`, so running the guard on it would test nothing.
    The wobble the guard exists to catch is now also reported, per fit, by `_projection`.
    """
    prev = None
    for K in strikes:
        p = sf.p_above_flatvol(K)
        if p is MISSING:
            continue
        if prev is not None and p > prev + tol:
            return False
        prev = p
    return True


# Public since 2026-08-24. It was private, and the consequence was that the ONE caller outside
# this module that needed it -- `strategy.skewstate._rebuild`, which builds the curve the LIVE
# quote loop prices off -- did not call it. A guard that only the in-module path runs is not a
# guard on the system.
_ladder_monotone = ladder_monotone
