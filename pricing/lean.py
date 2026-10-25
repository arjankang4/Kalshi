"""Turn a realized-vs-implied vol disagreement into a per-side quote offset.

WHAT THIS DOES NOT DO: it does not move the theo. `pricing/hourly` keeps producing the fair value
from implied vol, and that stays the number we think a market is worth. The realized-vol model
only changes how much edge we insist on before quoting. Mixing the two would mean trading on a
model whose out-of-sample record is 126 held-out sessions of vol forecasting -- not of PnL.

THE SHAPE OF THE LEAN

    expected   = sigma_impl * LEAN_VRP_RATIO
    k          = |sigma_rv - expected| / LEAN_VOL_SCALE        (vol points of SURPRISE)
    offset     = BASE_OFFSET * sqrt(1 / max(1, k))

So no surprise charges the full BASE_OFFSET, and a two-vol-point surprise charges
sqrt(1/2) * BASE_OFFSET -- the scaling specified. The offset only ever SHRINKS, never grows:
`max(1, k)` floors it, so a surprise can make us more aggressive on the side we like and never
wider than base.

**Why the centring, and why the lean is inert without it.** Realized vol sits systematically BELOW
implied vol -- the variance risk premium is one of the most persistent facts in options. Measured
live 2026-08-21 13:36 ET on a quiet SPX board: sigma_rv 4.79% against sigma_impl 7.87%, a ratio of
0.61. A lean on the RAW difference would therefore read "realized is 3 vol points cheap" on
essentially every observation and narrow the same side all day, which is not a signal -- it is a
constant, and a constant belongs in QUOTE_OFFSET, not in a model.

`LEAN_VRP_RATIO` is the typical sigma_rv / sigma_impl, and only the deviation from it is
information. It is `None` until calibrated, and with it `None` `side_offset` returns the base
offset unchanged. Calibrating it needs a historical implied-vol series, which is the same missing
measurement that keeps `LEAN_ENABLED` False.

The offset narrows on the side the realized-vol chain favours and stays at base on the other. The
disagreement is directional per strike -- a higher realized vol pushes near-money probabilities
toward 0.5 and far ones away from it -- which is why the caller builds a full shadow probability
chain rather than passing one scalar.

THE CLAMP IS THE SAFETY PROPERTY, NOT THE FORMULA. Whatever the scaling does, the bid may never
exceed the realized-vol chain's value and the offer may never sit below it. "Never quote bad to
our realized vol number" is a hard constraint that survives any bug in the arithmetic above it,
and it is checked independently in rvol/validate.
"""
import math

import config

MISSING = None


def side_offset(p_impl, p_rv, sigma_impl, sigma_rv, side,
                base_offset=None, vol_scale=None):
    """(offset, detail) for one side of one market.

    `p_impl` is the theo (what we quote around); `p_rv` is the shadow chain's value at the same
    strike. `side` is 'bid' or 'ask' in YES terms.
    """
    base = config.QUOTE_OFFSET if base_offset is None else base_offset
    scale = config.LEAN_VOL_SCALE if vol_scale is None else vol_scale

    d = {"lean_applied": False, "lean_k": None, "lean_offset": base,
         "p_rv": p_rv, "sigma_rv": sigma_rv, "sigma_impl": sigma_impl,
         "lean_reason": None, "clamped": False}

    if p_rv is MISSING or sigma_rv is MISSING or sigma_impl is MISSING or p_impl is MISSING:
        d["lean_reason"] = "no_rv"
        return base, d
    if not (math.isfinite(sigma_rv) and math.isfinite(sigma_impl) and sigma_impl > 0):
        d["lean_reason"] = "bad_sigma"
        return base, d

    vrp = config.LEAN_VRP_RATIO
    if vrp is None:
        # Uncalibrated: the raw difference is dominated by the variance risk premium, so acting
        # on it would narrow one side permanently. Record the numbers, change nothing.
        d["lean_reason"] = "vrp_uncalibrated"
        d["sigma_expected"] = None
        return base, d

    expected = sigma_impl * vrp
    k = abs(sigma_rv - expected) / scale
    d["lean_k"] = k
    d["sigma_expected"] = expected
    d["vrp_ratio"] = vrp

    # Which side does the shadow chain favour? p_rv > p_impl means the realized-vol view thinks
    # the contract is worth MORE, so buying is the attractive direction and the bid may narrow.
    # Surprise direction, not level direction: realized coming in ABOVE the premium-adjusted
    # expectation is what makes the shadow chain informative, and p_rv encodes that in strike
    # space (higher vol pushes near-money probabilities toward 0.5 and far ones away from it).
    favours_bid = p_rv > p_impl
    narrow = (side == "bid" and favours_bid) or (side == "ask" and not favours_bid)

    offset = base * math.sqrt(1.0 / max(1.0, k)) if narrow else base
    d["lean_applied"] = narrow and k > 1.0
    d["lean_offset"] = offset
    return offset, d


def clamp_price(price, p_rv, side):
    """Never quote through the realized-vol value. Returns (price, clamped).

    A bid may not exceed `p_rv`; an offer may not sit below it. This is applied AFTER the ladder
    has chosen a price, because the ladder can improve/penny/join to a level the offset never
    named -- so clamping the offset alone would not actually bound the quote.
    """
    if price is None or p_rv is MISSING:
        return price, False
    if side == "bid" and price > p_rv:
        return p_rv, True
    if side == "ask" and price < p_rv:
        return p_rv, True
    return price, False

# =========================================================================
# THE EDGE LEAN -- how the realized-vol view reaches a quote, since 2026-08-25.
#
# IT NEVER MOVES THE THEO. The fair value stays the digital priced off IMPLIED vol through the
# smile, delta-adjusted on futures. What the realized-vol view changes is how much edge we DEMAND
# on each side -- charge more where it says we are about to be picked off, accept less where it
# says we are being offered something cheap.
#
# That is a materially safer instrument than the offset-and-clamp mechanism below, which moved
# the quoted PRICE. The worst case here is quoting wider on one side and tighter on the other; the
# worst case there was a hard clamp putting a price on the wire the implied model never chose.
#
# THE MAPPING, in probability points. d = (p_rv - p_ref) x 100, clamped to +-10, u = |d| / 10:
#
#     defensive side (the one the disagreement says is dangerous):   required edge + 2.0u cents
#     aggressive side (the one it says is cheap):                    required edge - 0.5u cents
#
# So at d = +5 points, against the curve's 0.5c floor: offer 0.5 + 1.0 = 1.5c, bid 0.5 - 0.25 =
# 0.25c. At d = 0 both adjustments are 0 and the quote is exactly what the implied model wanted.
# At the +-10 clamp: +2.0c defensive, -0.5c aggressive (which floors the aggressive side at zero
# from a 0.5c base -- deliberately: that is as aggressive as this lean is allowed to make us).
#
# SIGN. d > 0 means realized says the market is MORE likely than implied does. So we want to BUY
# yes -- aggressive on the bid, defensive on the offer.
#
# ASYMMETRY IS THE POINT. Widening by up to 2c and tightening by at most 0.5c is not a mistake:
# being wrong about a lean costs a fill at a bad price, and being right only earns a slightly
# better one. The defensive leg is four times the aggressive leg because the payoffs are not
# symmetric.
#
# THE ADJUSTMENT IS APPLIED ON TOP OF THE CALIBRATED CURVE, not instead of it. `edgemgr` produces
# the required edge from R (EDGE_FLOOR at the calm end, rising across the top 1%); this shifts that
# number. Everything downstream -- the ladder, the width cap, the requote deadband -- then runs on
# the shifted value, which is what makes a change in realized vol able to trigger a requote at all.
# =========================================================================

# Probability-point disagreement at which the lean is fully applied. Illustrative value; the live
# value is not published.
LEAN_DISAGREEMENT_CLAMP = 8.0
# Edge added on the side the RV model says is dangerous, at the clamp. Illustrative value; the live
# value is not published.
LEAN_DEFENSIVE_CENTS = 1.5
# Edge removed on the side the RV model says is cheap, at the clamp. Illustrative value; the live
# value is not published.
LEAN_AGGRESSIVE_CENTS = 0.75


# Extra edge charged on a side right after it is filled. Illustrative value; the live value is not
# published.
BACKOFF_START_CENTS = 1.5
# Backoff remaining at BACKOFF_HALF_S. Illustrative value; the live value is not published.
BACKOFF_MID_CENTS = 0.75
# End of the exponential leg of the backoff. Illustrative value; the live value is not published.
BACKOFF_HALF_S = 4.0
# The backoff reaches zero here. Illustrative value; the live value is not published.
BACKOFF_END_S = 8.0


def backoff_cents(age_s, start=None, mid=None, half=None, end=None):
    """The backoff charge in cents for a fill `age_s` old. 0 when there is no recent fill."""
    if age_s is None or age_s < 0:
        return 0.0
    start = BACKOFF_START_CENTS if start is None else start
    mid = BACKOFF_MID_CENTS if mid is None else mid
    half = BACKOFF_HALF_S if half is None else half
    end = BACKOFF_END_S if end is None else end
    if age_s >= end:
        return 0.0
    if age_s <= half:
        # start -> mid with a `half`-second half-life, which for mid = start/2 is exactly
        # start * 0.5**(age/half).
        return start * ((mid / start) ** (age_s / half)) if start > 0 else 0.0
    return mid * (end - age_s) / (end - half)


def backoff_micros(age_s, **kw):
    return int(round(backoff_cents(age_s, **kw) * 10_000))


def edge_adjust(p_impl, p_rv, side, clamp=None, defensive=None, aggressive=None):
    """(delta_micros, detail) -- how much to shift this side's required edge, in micro-dollars.

    Positive widens (defensive), negative tightens (aggressive). Returns 0 with a reason whenever
    the lean has nothing to say, so a caller can add it unconditionally.
    """
    d = {"lean_d": None, "lean_u": None, "lean_side": side, "lean_cents": 0.0,
         "lean_reason": None}
    if p_impl is None or p_rv is None or p_impl is MISSING or p_rv is MISSING:
        d["lean_reason"] = "no_chain"
        return 0, d
    clamp = LEAN_DISAGREEMENT_CLAMP if clamp is None else clamp
    defensive = LEAN_DEFENSIVE_CENTS if defensive is None else defensive
    aggressive = LEAN_AGGRESSIVE_CENTS if aggressive is None else aggressive

    diff = (float(p_rv) - float(p_impl)) * 100.0
    d["lean_d"] = round(diff, 4)
    u = min(abs(diff), clamp) / clamp
    d["lean_u"] = round(u, 4)
    if u <= 0.0:
        d["lean_reason"] = "no_disagreement"
        return 0, d

    # d > 0: realized says MORE likely -> buy yes. The bid is the aggressive side, the ask the
    # defensive one. d < 0 mirrors it.
    aggressive_side = "bid" if diff > 0 else "ask"
    cents = -aggressive * u if side == aggressive_side else defensive * u
    d["lean_cents"] = round(cents, 4)
    d["lean_reason"] = "aggressive" if side == aggressive_side else "defensive"
    return int(round(cents * 10_000)), d       # cents -> micro-dollars
