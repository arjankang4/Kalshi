"""Binary-option Greeks, in DOLLARS PER CONTRACT.

A Kalshi index market is a binary that settles $0 or $1, so its price *is* its probability and a
Greek in probability units is already a Greek in dollars. That makes the three sensitivities
directly comparable and directly addable:

    delta   dollars of value per $1 move in the index
    vega    dollars of value per 1 volatility POINT (1%) of annualised sigma
    theta   dollars of value per 1 minute of time passing

They are deliberately NOT normalised. Their raw magnitudes are the point: measured on a real SPX
board, an at-the-money rung's delta runs $0.0115/pt at 10:00, $0.0318 at 15:00 and $0.1149 at
15:58, while the +-25pt wings decay from $0.0296 to zero over the same session. Vega dominates the
wings in the morning, delta dominates the money into the close, and theta only matters in the last
half hour. Which Greek dominates, and when, is the signal.

WHAT IS *NOT* HERE, AND WHY

  * **Gamma.** For a binary it is not independent information. Since d2 = (ln(F/K) - w^2/2)/w,
    ln(F/K) = d2*w + w^2/2, so

        vega_w  = phi(d2)*(-ln(F/K)/w^2 - 1/2) = -phi(d2)*(1 + d2/w)
        gamma   = -phi(d2)*(1 + d2/w)/(F^2*w)  = vega_w / (F^2 * w)

    Verified to 3.7e-16 over 54 (F, w, K) combinations. F and w are constant across a board at any
    instant, so gamma is vega rescaled and adding it just double-weights the vega axis.

  * **Any sensitivity of the vertical-spread leg.** `theo.digital_above_spread` reads option
    quotes; bumping spot does not tell you what those quotes would have been. Its only response to
    a spot move is the call/put branch flip, which is a discontinuity, not a derivative. These
    Greeks describe the Black-Scholes leg, which is the part of the blend that HAS derivatives.
    That is a limitation of the measure, recorded rather than papered over.

`pricing/bs.py` also exports a `vega_w` -- that one is the VANILLA's vega and a different
quantity. The names here are distinct on purpose.
"""
import datetime
import math

import config
from pricing import bs, theo

MISSING = None

TRADING_DAYS = 252.0

# Tunable. Illustrative value; the live value is not published.
MIN_VAR_LEFT_FRACTION = 0.01

# Floor on clock time left for theta. Illustrative value; the live value is not published.
THETA_MIN_SECONDS = 30.0


def delta_per_point(F, K, w):
    """dP/dS: dollars per $1 move in the index. Always positive; peaks at the money."""
    if F is None or K is None or w is None or F <= 0 or K <= 0 or w <= 0:
        return MISSING
    _, d2 = bs.d1_d2(F, K, w)
    return bs.norm_pdf(d2) / (F * w)


def vega_per_volpoint(F, K, w, var_left):
    """dP/dsigma_annualised, per ONE vol point (1%). Signed; zero at the money.

    The chain rule matters and is easy to get wrong: `w = sigma * sqrt(var_left / 252)`, so

        dP/dsigma = dP/dw * sqrt(var_left / 252)

    and a vol POINT is a further factor of 0.01. Skipping either rescales the whole axis and
    silently reweights R against delta and theta.

    Sign is kept here -- it says which way vol helps -- and the caller takes |.| for the risk norm.
    """
    if F is None or K is None or w is None or F <= 0 or K <= 0 or w <= 0:
        return MISSING
    if var_left is None or var_left <= 0:
        return MISSING
    _, d2 = bs.d1_d2(F, K, w)
    dP_dw = bs.norm_pdf(d2) * (-math.log(F / K) / (w * w) - 0.5)
    return dP_dw * math.sqrt(var_left / TRADING_DAYS) * 0.01


def var_left(calendar, now, settlement, floor=True):
    """Remaining variance-time from `now` to `settlement`, in session-variance units.

    Floored at `MIN_VAR_LEFT_FRACTION` of the market's whole life so theta cannot diverge -- see
    the module constant. Returns MISSING when the calendar has nothing left to apportion, which is
    `varcal`'s own refusal and is not overridden here.
    """
    if settlement is None or now is None or settlement <= now:
        return MISSING
    ratio = calendar.remaining_ratio(now, settlement)
    if ratio is None:
        return MISSING
    left = (1.0 - calendar.phi(now)) * ratio
    if not floor:
        return left
    return max(left, MIN_VAR_LEFT_FRACTION * (1.0 - calendar.phi(_session_open(now))))


def _session_open(now):
    et = now.astimezone(config.ET)
    return datetime.datetime.combine(et.date(), datetime.time(9, 30), tzinfo=config.ET)


def total_vol(sigma_annualised, var_left_units):
    """sigma on the intraday variance clock -> total vol w over the remaining window."""
    if sigma_annualised is None or var_left_units is None or var_left_units <= 0:
        return MISSING
    return sigma_annualised * math.sqrt(var_left_units / TRADING_DAYS)


def theta_per_minute(F, K, sigma_annualised, now, settlement, calendar):
    """Dollars of value lost per ONE minute of time passing. Signed.

    A FINITE one-minute difference through the variance calendar, not an analytic dP/dT, and the
    difference is not cosmetic. `varcal.phi` is piecewise-linear and deliberately not smooth: the
    15:59 bucket carries ~3x a flat minute (the MOC ramp) and midday carries ~0.7x. The next
    minute's ACTUAL variance share is the honest number; an analytic derivative would smooth away
    exactly the feature the calendar exists to capture.

    Returned as `P(now+1min) - P(now)` -- the change in value the position experiences as that
    minute elapses with spot and vol unchanged.
    """
    # Floor the remaining clock time: a market inside THETA_MIN_SECONDS is measured as if it had
    # exactly that much left, which is the spec ("set the theta of the <1 min to expiration to
    # its theta if it was =1 min to expiration").
    gap = (settlement - now).total_seconds()
    if gap <= 0:
        return MISSING          # already settled: there is no next minute to decay through
    if THETA_MIN_SECONDS > 0 and gap < THETA_MIN_SECONDS:
        now = settlement - datetime.timedelta(seconds=THETA_MIN_SECONDS)

    vl0 = var_left(calendar, now, settlement, floor=False)
    if vl0 is MISSING:
        return MISSING
    w0 = total_vol(sigma_annualised, vl0)
    p0 = bs.p_above(F, K, w0) if w0 else MISSING
    if p0 is MISSING:
        return MISSING

    nxt = now + datetime.timedelta(minutes=1)
    if nxt >= settlement:
        # Inside the last minute the option is a spot comparison, not a distribution.
        p1 = 1.0 if F > K else 0.0
        return p1 - p0
    vl1 = var_left(calendar, nxt, settlement, floor=False)
    if vl1 is MISSING:
        return MISSING
    w1 = total_vol(sigma_annualised, vl1)
    p1 = bs.p_above(F, K, w1) if w1 else MISSING
    if p1 is MISSING:
        return MISSING
    return p1 - p0


# --- bracket aggregation -------------------------------------------------

def one_sided_greeks(g_above):
    """The four one-sided spellings of a Greek at one boundary, from d/dx of P(S > K).

    Mirrors `theo.one_sided_probs` so `STRIKE_TYPE_RULE` is a lookup here too -- but the mapping
    is NOT the same, and that difference is the whole reason this function exists. A probability
    maps `less` to `1 - p`; a DERIVATIVE maps it to `-g`, because the constant 1 differentiates
    away. Reusing `one_sided_probs` on a Greek returns `1 - delta`, which on a real board read
    +0.985 where the true delta was -0.015.
    """
    if g_above is None:
        return {k: None for k in theo.PROB_KEYS}
    return {"less": -g_above, "less_or_equal": -g_above,
            "greater": g_above, "greater_or_equal": g_above}


def bracket_greek(strike_type, floor_strike, cap_strike, g_at):
    """A Kalshi market's Greek, from a per-boundary Greek function. None if unavailable.

    Same signed-term arithmetic as `theo.bracket_fair` over the same `STRIKE_TYPE_RULE` table, and
    deliberately NOT a call into it: `bracket_fair` ends in `clamp(total)`, which is right for a
    probability and destroys a Greek. On a real board that clamp turned a `between` delta of
    -0.004156 into +0.000000 -- a market reading as risk-free at the moment it was most exposed.
    """
    rule = theo.STRIKE_TYPE_RULE.get(strike_type)
    if rule is None:
        return MISSING
    bounds = {"floor_strike": theo.normalize_boundary(floor_strike),
              "cap_strike": theo.normalize_boundary(cap_strike)}
    memo = {}
    total = 0.0
    for term in rule:
        b = bounds.get(term.bound)
        if b is None:
            return MISSING
        if b not in memo:
            memo[b] = one_sided_greeks(g_at(b))
        v = memo[b][term.prob]
        if v is None:
            return MISSING
        total += term.sign * v
    return total


def market_greeks(market, F, sigma_annualised, now, settlement, calendar):
    """The three dollar Greeks for one Kalshi market, plus the inputs they were built from.

    Returns a dict with `delta`, `vega`, `theta` (all dollars per contract, signed) and `R`, the
    Euclidean norm that `pricing/edge` consumes. `R` uses absolute values because the sign of a
    Greek is direction, not risk.
    """
    out = {"delta": None, "vega": None, "theta": None, "R": None,
           "w": None, "var_left": None, "sigma": sigma_annualised,
           "forward": F, "reason": None}
    vl = var_left(calendar, now, settlement)
    if vl is MISSING:
        out["reason"] = "no_variance_left"
        return out
    w = total_vol(sigma_annualised, vl)
    if w is MISSING or w <= 0:
        out["reason"] = "no_vol"
        return out
    out["w"], out["var_left"] = w, vl

    st = market.get("strike_type")
    lo, cap = market.get("floor_strike"), market.get("cap_strike")
    d = bracket_greek(st, lo, cap, lambda k: delta_per_point(F, k, w))
    v = bracket_greek(st, lo, cap, lambda k: vega_per_volpoint(F, k, w, vl))
    t = bracket_greek(st, lo, cap,
                      lambda k: theta_per_minute(F, k, sigma_annualised, now, settlement, calendar))
    out["delta"], out["vega"], out["theta"] = d, v, t
    if d is None or v is None or t is None:
        out["reason"] = "no_greek"
        return out
    out["R"] = math.sqrt(d * d + v * v + t * t)
    return out
