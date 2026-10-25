"""Black-Scholes in TOTAL-VOL space, for pricing markets that have no option of their own.

`pricing/theo.py` reads probabilities off the 4pm option board, which only answers the question
that board is asked: what happens at 16:00 ET.  The hourly Kalshi boards settle at 10:00..15:00,
and no listed option expires then.  So those need a distribution, and this is the smallest one
that will do.

(Historical note, because this docstring said "model-free off vertical spreads" until 2026-08-24
and the difference is not cosmetic: `theo.digital_above` reads a vol off the fitted smile and
takes `-dC/dK` of the resulting curve -- `N(d2) - F*phi(d1)*dw/dK`, `pricing/skew.SkewFit`.  The
vertical spread was removed on 2026-08-21 and for three days the slope term went with it.  See
`docs/pricing-invariants.md`.)

Everything here is parameterised by **w = sigma * sqrt(T)**, the standard deviation of the log
return over the window -- not by an annualised sigma and a separate T.  Black-Scholes depends on
the pair only through their product, so carrying the product:

  * removes the annualisation round trip (pick a day-count, divide, later multiply back), which is
    where a factor of sqrt(252/365) hides;
  * lets the variance calendar scale a 4pm number to a 14:00 number by a pure ratio, with no
    day-count convention entering at all (see pricing/varcal);
  * makes the degenerate cases legible: w == 0 is intrinsic, and w is what a bisection solves for.

Rates and dividends are zero.  These are 0DTE index options: the financing component over a few
hours is smaller than a tick, and Kalshi settles on the index, not on a total-return series.  So
`F` below is both the forward and the spot, and prices are undiscounted.

`math.erf` rather than `scipy.stats.norm`: scipy is not installed, and this needs to run in the
minutes before a session.  The error function is in the C library and is correctly rounded, so
`norm_cdf` is accurate to ~1 ulp -- better than the tabulated rational approximations that get
reached for when scipy is missing.
"""
import math

MISSING = None

SQRT2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)

# Bisection bounds for the implied-vol solve.  The upper bound is deliberately absurd (500% total
# stdev): a real 0DTE ATM straddle is w ~ 0.002-0.02, and a bound that can be hit silently returns
# a wrong answer instead of None.
W_MIN = 1e-9
W_MAX = 5.0
W_TOL = 1e-12
MAX_ITER = 200

# Minimum time value, in dollars, for a price to identify a vol at all.
#
# Vega vanishes far from the money, so past a few sigma the price stops responding to w and the
# bisection converges on whatever the last representable price happened to be. Measured: a
# 50-point-ITM call at w = 5e-4 (a 3.9 point stdev, i.e. 13 sigma) round-trips with |dw| = 3e-06,
# which is 0.6% of w -- not a solver bug, just a question the price cannot answer.
#
# One cent is the natural floor: it is the resolution of the price grid these quotes live on, so
# an option whose entire time value is under a tick carries less than one tick of information
# about vol. Returning MISSING keeps such a quote out of the ATM aggregate instead of letting it
# vote with a fabricated number.
MIN_TIME_VALUE = 0.01


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / SQRT_2PI


def intrinsic(F, K, cp):
    return max(0.0, F - K) if cp == "C" else max(0.0, K - F)


def d1_d2(F, K, w):
    """(d1, d2) with r = q = 0.  Caller guarantees F > 0, K > 0, w > 0."""
    d1 = (math.log(F / K) + 0.5 * w * w) / w
    return d1, d1 - w


def bs_price(F, K, w, cp):
    """Undiscounted Black-Scholes.  `w` is total stdev, not annualised vol."""
    if F <= 0.0 or K <= 0.0:
        return MISSING
    if w <= 0.0:
        return intrinsic(F, K, cp)
    d1, d2 = d1_d2(F, K, w)
    if cp == "C":
        return F * norm_cdf(d1) - K * norm_cdf(d2)
    return K * norm_cdf(-d2) - F * norm_cdf(-d1)


def vega_w(F, K, w):
    """d(price)/dw -- sensitivity to TOTAL stdev, so no sqrt(T) factor.

    Same for calls and puts.  Used to weight strikes and to sanity-check that a solved w sits
    somewhere the price is actually informative about it.
    """
    if F <= 0.0 or K <= 0.0 or w <= 0.0:
        return 0.0
    d1, _ = d1_d2(F, K, w)
    return F * norm_pdf(d1)


def p_above(F, K, w):
    """P(S_T > K) under the risk-neutral lognormal -- this is N(d2), NOT the delta N(d1).

    The delta is the hedge ratio; N(d2) is the probability.  They differ by roughly phi(0)*w,
    which on a one-sided Kalshi rung is ~0.08c at a quiet hour's vol and ~0.32c in a stressed
    one.  Under a 1c tick, but a real fraction of the 1c quote offset -- and N(d2) is not more
    expensive to compute, so there is no reason to carry the bias.

    On a two-sided bracket the bias very largely cancels (<0.06c on a 25-wide range), which is
    why the daily board could be priced off spreads without ever confronting this.
    """
    if F <= 0.0 or K <= 0.0:
        return MISSING
    if w <= 0.0:
        return 1.0 if F > K else 0.0
    _, d2 = d1_d2(F, K, w)
    return norm_cdf(d2)


def implied_w(F, K, price, cp, lo=W_MIN, hi=W_MAX, min_time_value=MIN_TIME_VALUE):
    """Total stdev implied by an option price, or MISSING if the price implies none.

    Bisection rather than Newton: the price is monotone in w with no local structure, 200
    halvings of [1e-9, 5] is exact to ~1e-12, and it cannot diverge on the near-zero vega that a
    deep-OTM 0DTE quote presents.  Measured at ~40k solves/second, so the whole 400-rung NDX
    board costs well under a millisecond.

    MISSING (not an exception, and not a clamp to the bound) when the price is at or through a
    no-arbitrage boundary, or when its time value is under `min_time_value` and so identifies no
    vol.  Those quotes are common on a 0DTE board -- a deep-ITM call whose bid is gone prices to
    intrinsic -- and clamping them to W_MIN would feed a zero vol into the aggregate as though it
    were an observation.
    """
    if F <= 0.0 or K <= 0.0 or price is MISSING or price < 0.0:
        return MISSING
    intr = intrinsic(F, K, cp)
    cap = F if cp == "C" else K          # call <= F, put <= K, both undiscounted
    if price <= intr + min_time_value or price >= cap - 1e-12:
        return MISSING
    for _ in range(MAX_ITER):
        mid = 0.5 * (lo + hi)
        if bs_price(F, K, mid, cp) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < W_TOL:
            break
    return 0.5 * (lo + hi)

