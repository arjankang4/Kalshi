"""Which markets are near enough the money to be worth quoting in depth.

The band decides the SHAPE of a quote -- three rungs a side or one -- and, with it, the market's
inventory limit. It does not decide whether a market is quoted at all; that is
`budget.select_markets`, which rations capital, and `edgemgr`, which refuses a market with no `R`.

WHY SIGMA AND NOT STRIKES. Measured on the two SPX boards, 2026-08-21 11:49 ET
(F = 7691.1, w to 16:00 = 0.00244, so one sigma is 18.8 index points):

    KXINX  4pm brackets, 25 wide  ->  +-3 strikes = +-75 pts = +-3.99 sigma
    KXINXU hourly rungs,  5 wide  ->  +-3 strikes = +-15 pts = +-0.80 sigma

A strike count means five different things on two boards of the same underlying, so counting
strikes would quote the hourly ladder five times tighter than the daily one for no stated reason.
`config.QUOTE_BAND_PCT` already carries this argument for the option strip; this is the same
argument for the Kalshi board.

Sigma-moneyness also self-adjusts to the horizon for free, because `w` is the total vol to THAT
market's own settlement -- `edgemgr` computes it per market through the variance calendar. An
hourly rung ten minutes from close gets almost no band, which is correct: it has no optionality
left anywhere but at the money.

HYSTERESIS. Entering the band costs two extra rungs a side; leaving it cancels them. On a
price-time exchange that is queue position, so a market that oscillates across the boundary pays
repeatedly for a classification that keeps flipping. An incumbent stays in the band until it is
`BAND_EXIT_MULT` times further out than it would have needed to be to enter.

Pure -- no network, no clock, no state. `python -m strategy.test_band`.
"""
import config
from pricing.theo import STRIKE_TYPE_RULE, normalize_boundary

# A `between` market has two boundaries and is not a point on the moneyness axis. Its distance is
# measured from the MIDPOINT of the bracket, which is where its digital is centred and where its
# Greeks peak -- the same place `bracket_fair` differences its two terms.
BRACKET_TYPE = "between"


def market_strike(market):
    """The single level this market's moneyness is measured from, or None if unpriceable.

    `floor_strike` for a `greater*` market, `cap_strike` for a `less*` one -- taken from
    `theo.STRIKE_TYPE_RULE` rather than restated, so a new strike type cannot be handled here and
    not there. A `between` market uses the midpoint of its two boundaries.
    """
    st = market.get("strike_type")
    terms = STRIKE_TYPE_RULE.get(st)
    if not terms:
        return None
    if st == BRACKET_TYPE:
        lo = normalize_boundary(market.get("floor_strike"))
        hi = normalize_boundary(market.get("cap_strike"))
        if lo is None or hi is None:
            return None
        return 0.5 * (lo + hi)
    return normalize_boundary(market.get(terms[0].bound))


def band_width(forward, w, sigma=None):
    """Half-width of the band in index points: `sigma * w * F`. None when it cannot be computed.

    `w` is TOTAL vol (sigma * sqrt(T)), which is the unit everything in this repo is parameterised
    by -- see `pricing/bs`. So `w * F` is one standard deviation of the terminal level in points,
    with no `T` appearing anywhere.
    """
    if forward is None or w is None or forward <= 0 or w <= 0:
        return None
    s = config.BAND_SIGMA if sigma is None else float(sigma)
    return s * float(w) * float(forward)


def distance(market, forward):
    """|K - F| in index points, or None."""
    k = market_strike(market)
    if k is None or forward is None:
        return None
    return abs(float(k) - float(forward))


def in_band(market, forward, w, sigma=None, incumbent=False, exit_mult=None):
    """(bool, detail) -- does this market get three rungs?

    `incumbent` is whether it is ALREADY quoted in depth. An incumbent is held to the wider exit
    boundary, so the band is sticky and does not churn a ladder on every wobble of the forward.

    Fails CLOSED: no strike, no forward or no `w` means one rung, not three. Depth is the
    expensive choice and an unknown does not earn it.
    """
    exit_mult = config.BAND_EXIT_MULT if exit_mult is None else float(exit_mult)
    d = distance(market, forward)
    half = band_width(forward, w, sigma)
    if d is None or half is None:
        return False, {"reason": "no_strike" if d is None else "no_width",
                       "distance": d, "half_width": half}
    limit = half * (exit_mult if incumbent else 1.0)
    ok = d <= limit + 1e-9
    # `sigmas` is the market's actual moneyness in standard deviations -- d / (w*F) -- NOT its
    # position within the band. The distinction matters in a log: "1.4 sigma out" is a fact about
    # the market, while "0.7 of the way to the edge" changes meaning whenever BAND_SIGMA is tuned.
    # `band_frac` carries the second reading for anyone who wants it.
    one_sigma = half / (config.BAND_SIGMA if sigma is None else float(sigma))
    return ok, {"reason": None, "distance": round(d, 4), "half_width": round(half, 4),
                "limit": round(limit, 4),
                "sigmas": round(d / one_sigma, 3) if one_sigma else None,
                "band_frac": round(d / half, 3) if half else None,
                "incumbent": bool(incumbent)}


def counts_for(inband, counts=None):
    """The rung sizes for a market: the full ladder in-band, its first rung outside.

    Slicing the SAME tuple rather than naming a second one is deliberate -- a wing rung must be
    the same size as a band rung, because the lot is the unit the inventory limits are counted in.
    """
    counts = counts or config.QUOTE_TIER_COUNTS
    return tuple(counts) if inband else tuple(counts[:1])


def position_limit(inband, lot=None):
    """Net contracts we are willing to hold in this market, as a positive bound."""
    lot = config.LOT_SIZE if lot is None else lot
    lots = config.POSITION_LIMIT_LOTS_BAND if inband else config.POSITION_LIMIT_LOTS_WING
    return float(lots) * float(lot)


def classify(markets, forward, em_map, incumbents=(), sigma=None):
    """{ticker: {inband, counts, position_limit, ...}} for a whole board, plus a summary.

    `w` is read per market from `em_map` -- `edgemgr.refresh` already computes it through the
    variance calendar for each market's own settlement, so nothing is re-priced here.
    """
    incumbents = set(incumbents)
    out, n_band = {}, 0
    for m in markets:
        t = m.get("ticker")
        w = ((em_map or {}).get(t) or {}).get("w")
        ok, d = in_band(m, forward, w, sigma=sigma, incumbent=t in incumbents)
        n_band += bool(ok)
        out[t] = {"inband": bool(ok), "counts": counts_for(ok),
                  "position_limit": position_limit(ok), **d}
    summary = {"n": len(out), "band": n_band, "wing": len(out) - n_band,
               "sigma": config.BAND_SIGMA if sigma is None else sigma,
               "exit_mult": config.BAND_EXIT_MULT}
    return out, summary


def resolve_counts(counts, ticker, default=None):
    """The rung sizes for `ticker`, from either a per-market map or one global tuple.

    Every consumer of `counts` -- `orders.reconcile`, `budget.select_markets`,
    `run_04.build_desired` -- now takes either shape, so a caller that has not been taught about
    the band keeps working unchanged and a caller that has passes a `{ticker: counts}` map. One
    helper rather than three copies of the isinstance check, because three copies is how the two
    halves of a rule drift apart.
    """
    default = default or config.QUOTE_TIER_COUNTS
    if isinstance(counts, dict):
        return tuple(counts.get(ticker) or default)
    return tuple(counts or default)
