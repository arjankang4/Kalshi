"""Kalshi bracket arithmetic: strike-type rules and bracket fair value from a digital.

Config-free on purpose, so a non-equity project (~/Kalshi_BTC) can import it without
loading the equity config. `pricing.theo` re-exports every name here.

Bracket arithmetic is a lookup table, not an if-chain.  Kalshi's `rules_primary` text is what
pins each type's inclusivity, and it does NOT match the OpenAPI field docs (`floor_strike` is
documented as the minimum value that settles YES, i.e. inclusive, yet `greater`/`less` use
their bound exclusively):

    greater           floor_strike=7691.76   "is above 7691.76"          S >  K
    greater_or_equal  floor_strike=7870      "is above 7869.9999"        S >= K
    less              cap_strike=7375        "is below 7375"             S <  K
    less_or_equal     cap_strike             (none listed today)         S <= K
    between           floor=7375 cap=7399.9999  "is between 7375-7399.9999"  lo <= S <= hi

All four one-sided structures are computed and exposed, and `strike_type` selects which one is
used -- see `one_sided_probs` and `STRIKE_TYPE_RULE`.

On why four when only two numbers exist: for a continuous index P(S > K) == P(S >= K), so the
four collapse to two.  The atom P(S == K) is not literally zero -- Kalshi settles to the index
rounded to 2dp -- but it is negligible.  With 0DTE sigma ~0.8% of 7700 (~62 pts), peak density
is ~1/(62*sqrt(2pi)) ~ 0.0064/pt, so a 0.01-wide atom is ~6e-5, i.e. 0.006 cents against a
1-cent tick.  Five minutes from the close (sigma ~7 pts) it is ~6e-4; only inside the last
~30 seconds does it reach ~2e-3, and only for the bracket straddling spot.  `digital_above`
reads a spread off a 5- or 10-point strike grid, so its own resolution floor is ~1/w per point
-- orders of magnitude coarser than the atom it would need to resolve.

So the strict/non-strict split is carried as SEMANTICS (the record says which event was priced,
auditable against Kalshi's rules text) and deliberately not as NUMERICS.  It is one place to
change if a settlement-grid model is ever fitted.
"""
from collections import namedtuple

MISSING = None

# Kalshi spells one boundary two ways: a `between` market caps at 7574.9999 and the next floors
# at 7575.  Rounding only inside a tolerance collapses those without moving a genuinely
# fractional level -- KXINXDUD floors at the prior close, e.g. 7691.76.
BOUNDARY_ROUND_TOL = 0.02


def clamp(x, lo=0.0, hi=1.0):
    return min(hi, max(lo, x))


def normalize_boundary(x):
    """Kalshi's two spellings of one boundary -> one float.

    Rounds only inside BOUNDARY_ROUND_TOL, so 7574.9999 -> 7575 but KXINXDUD's 7691.76 keeps
    its fraction.  Accepts float or Decimal (storage.units.to_strike returns Decimal).
    """
    if x is None:
        return None
    f = float(x)
    r = round(f)
    return float(r) if abs(f - r) <= BOUNDARY_ROUND_TOL else f


# --- one-sided structures ------------------------------------------------

PROB_KEYS = ("less", "less_or_equal", "greater", "greater_or_equal")


def one_sided_probs(p_above):
    """The four one-sided probabilities at one boundary, derived from P(S > K).

    Keys are Kalshi's own strike_type spellings so STRIKE_TYPE_RULE is a lookup rather than a
    translation.  All four are returned even though only two are distinct -- see the module
    docstring for why the split is semantic, not numeric.
    """
    if p_above is None:
        return {k: None for k in PROB_KEYS}
    below = 1.0 - p_above
    return {"less": below, "less_or_equal": below,
            "greater": p_above, "greater_or_equal": p_above}


# One signed term of a bracket's payoff, in terms of a one-sided probability at a boundary.
Term = namedtuple("Term", "sign bound prob")

STRIKE_TYPE_RULE = {
    "greater":          (Term(+1, "floor_strike", "greater"),),
    "greater_or_equal": (Term(+1, "floor_strike", "greater_or_equal"),),
    "less":             (Term(+1, "cap_strike", "less"),),
    "less_or_equal":    (Term(+1, "cap_strike", "less_or_equal"),),
    # lo <= S <= hi  ==  P(S >= lo) - P(S > hi).  Mixing a non-strict with a strict term is
    # deliberate: it matches Kalshi's inclusive `between` rules text, and it is numerically
    # identical to the old P(>lo) - P(>hi) since the two spellings share a value.
    "between":          (Term(+1, "floor_strike", "greater_or_equal"),
                         Term(-1, "cap_strike", "greater")),
}

# Distinguishable reasons beat one generic "unhandled": these three price off something other
# than a scalar strike, so no digital can reach them.
UNPRICEABLE_REASON = {
    "functional": "functional_strike",
    "custom": "custom_strike",
    "structured": "structured_strike",
    None: "null_strike_type",
}


def is_priceable(strike_type):
    return strike_type in STRIKE_TYPE_RULE


def bracket_fair(strike_type, floor_strike, cap_strike, p_above):
    """Kalshi bracket -> (fair, detail), given a digital function.

    `p_above(K) -> P(S > K) | None`.  Pure: no board, no spot, no Pricer -- which is what lets
    the backtest share this exact arithmetic instead of reimplementing it.

    The detail dict always carries the same keys so downstream tables stay rectangular.
    Exactly one `use_*` flag is True for a priceable market; all are False otherwise.
    """
    lo = normalize_boundary(floor_strike)
    hi = normalize_boundary(cap_strike)
    d = {"strike_type": strike_type, "lo": lo, "hi": hi,
         "prob_used": None,
         "use_less": False, "use_less_or_equal": False,
         "use_greater": False, "use_greater_or_equal": False, "use_between": False,
         "probs_lo": None, "probs_hi": None,
         "p_above_lo": None, "p_above_hi": None,
         "terms": [], "fair": None, "reason": None, "error": None}

    rule = STRIKE_TYPE_RULE.get(strike_type)
    if rule is None:
        d["reason"] = UNPRICEABLE_REASON.get(strike_type, "unknown_strike_type")
        d["error"] = f"unhandled strike_type {strike_type!r}"
        return MISSING, d

    d["prob_used"] = strike_type
    d[f"use_{strike_type}"] = True

    bounds = {"floor_strike": lo, "cap_strike": hi}
    memo = {}                                    # boundary -> probs dict, so `between` calls once
    total = 0.0
    for term in rule:
        b = bounds[term.bound]
        if b is None:
            d["reason"] = f"missing_{term.bound}"
            d["error"] = f"{strike_type} needs {term.bound}"
            return MISSING, d
        if b not in memo:
            pa = p_above(b)
            memo[b] = one_sided_probs(pa)
            if term.bound == "floor_strike":
                d["p_above_lo"], d["probs_lo"] = pa, memo[b]
            else:
                d["p_above_hi"], d["probs_hi"] = pa, memo[b]
        val = memo[b][term.prob]
        d["terms"].append({"sign": term.sign, "bound": term.bound, "boundary": b,
                           "prob": term.prob, "value": val})
        if val is None:
            d["reason"] = "no_digital"
            d["error"] = (f"no digital at boundary {b} "
                          f"(missing quotes or no bracketing strikes)")
            return MISSING, d
        total += term.sign * val

    d["fair"] = clamp(total)
    return d["fair"], d
