"""Theoretical value: cube-root microprice -> digital probabilities -> bracket fair value.

Departures from the spec as literally written (all three flagged in the plan):
  1. Call digital is (C(K_lo) - C(K_hi))/w.  The spec's (C(7705)-C(7695)) is negative, since
     calls decrease in strike.  The put form in the spec is already correctly signed.
  2. The divisor is the *actual* strike width (K_hi - K_lo), not a hardcoded 2.  The spec's "/2"
     fits SPY $1 strikes; on the SPX 7695/7705 strikes it also names, the width is 10.
  3. The microprice cube-roots BOTH sizes.  The spec's worked example gives $1.04 by mixing
     cbrt(8)=2 with the raw 8; applying the stated rule consistently gives $1.0333.

Bracket arithmetic (strike-type rules, boundary spelling, `bracket_fair`) lives in
`pricing/brackets.py`, which carries no config and is re-exported here.
"""
from statistics import median

from pricing import bs
from pricing import skew as skewmod
from pricing.brackets import (  # noqa: F401  re-exported: callers use theo.<name>
    BOUNDARY_ROUND_TOL, MISSING, PROB_KEYS, STRIKE_TYPE_RULE, UNPRICEABLE_REASON, Term,
    bracket_fair, clamp, is_priceable, normalize_boundary, one_sided_probs)


def cbrt(x):
    return max(x, 0.0) ** (1.0 / 3.0)


def micro(bid, ask, bid_size, ask_size):
    """Cube-root size-weighted microprice.

    Weight leans toward the side with more size: heavy offer -> fair sits near the bid.
    Equivalent to bid + w_bid * (ask - bid) with w_bid = cbrt(bs)/(cbrt(bs)+cbrt(as)).
    """
    if bid is None or ask is None:
        return MISSING
    wb, wa = cbrt(bid_size or 0.0), cbrt(ask_size or 0.0)
    if wb + wa == 0:
        return (bid + ask) / 2.0
    return (bid * wa + ask * wb) / (wb + wa)


def quote_value(q):
    """Massive Q frame -> microprice.

    Massive omits `bp`/`bs` entirely when there is no bid (very common for OTM 0DTE options).
    An absent bid means zero, not unknown, so it is read as 0.00 x 0 -- which the cube-root
    weighting then pins the fair value to, since the bid carries no size.
    """
    if not q:
        return MISSING
    bid, ask = q.get("bp"), q.get("ap")
    if bid is None and ask is None:
        return MISSING
    if bid is None:
        bid = 0.0
    if ask is None:
        return bid  # one-sided: bid only
    return micro(bid, ask, q.get("bs") or 0.0, q.get("as") or 0.0)


def two_sided(q, require_ask_size=True):
    """Is this quote a genuine two-sided market?  THE canonical predicate.

    Stricter than `quote_value`, deliberately, and the strictness is load-bearing.

    Massive omits `bp`/`bs` entirely when there is no bid, and `quote_value` correctly reads
    that as 0.00 -- the right call for pricing a wing, fatal for anchoring parity. A deep-ITM
    call whose bid has gone prices to 0.00, which makes |C - P| smallest at *its* strike, which
    wins the anchor and implies S at that strike. Measured on real tape: spot 7387.5 on a day
    whose true spot was 7706.

    This lives in `theo` rather than `atmvol` because `theo` is the lower module (atmvol imports
    quote_value from here, so the reverse would cycle) and because the rule had been implemented
    three separate times -- here it was simply MISSING, which is what let the live parity path
    carry the bug the other two guarded against. `pricing.atmvol.two_sided` and
    `backtest.theo_bt.Board._two_sided` now both delegate here.

    `require_ask_size=False` reproduces the backtest's weaker form (bid size only); a vol anchor
    needs both sides quoted because the microprice interpolates between them.
    """
    if not q:
        return False
    if q.get("bp") is None or q.get("ap") is None:
        return False
    if (q.get("bs") or 0) <= 0:
        return False
    return not require_ask_size or (q.get("as") or 0) > 0


def neighbors(strike, strikes):
    """Closest listed strikes strictly below and strictly above `strike`."""
    below = [k for k in strikes if k < strike]
    above = [k for k in strikes if k > strike]
    if not below or not above:
        return None, None
    return max(below), min(above)


class Pricer:
    """Prices digitals off one snapshot of option quotes.

    `quotes` maps ('C'|'P', strike) -> Massive Q frame.
    """

    def __init__(self, quotes, strikes, spot=None, skew_fit=None):
        self.quotes = quotes
        self.strikes = sorted(strikes)
        self._strike_set = set(self.strikes)
        self.spot = spot if spot is not None else self.parity_spot()
        self._skew = skew_fit
        self._skew_built = skew_fit is not None

    def use_skew(self, skew_fit):
        """Install a curve fitted elsewhere, after construction.

        The live loop reuses one `strategy.skewstate.SkewState` across cycles, but the curve's
        x-axis is measured from the forward, so it cannot be fitted until the spot is resolved --
        which happens after this Pricer exists. Hence a setter rather than a constructor argument
        on that path. `skew_fit=` still works for callers that have a curve up front.
        """
        self._skew = skew_fit
        self._skew_built = True
        return self

    @property
    def skew(self):
        """The fitted vol curve, built once on first use.

        Lazy because `parity_spot` has to run first -- the curve's x-axis is measured from the
        forward, so a Pricer whose spot is still unknown cannot have a curve yet.

        This runs on every call that does NOT hand a curve over: the backtest, the validators,
        and `validate_theo.resolve_and_price` when it is given no `skew_state`. The live quote
        loop passes one, so there it refits at `SKEW_RECOMPUTE_S` rather than every cycle and
        prices off the smoothed curve. Until 2026-08-24 this docstring claimed the live loop did
        that and it did not -- `resolve_and_price` had no way to pass a curve at all.
        """
        if not self._skew_built:
            w_atm = skewmod.atm_vol(self.quotes, self.strikes, self.spot)
            self._skew = skewmod.fit(self.quotes, self.strikes, self.spot, w_atm)
            self._skew_built = True
        return self._skew

    def val(self, right, strike):
        return quote_value(self.quotes.get((right, strike)))

    def parity_spot(self, width=50.0, strict=True):
        """S = K + C - P at 0DTE (discounting ~ 0). Median over near-money strikes.

        A rough anchor comes from the strike where |C - P| is smallest, then we median around it.

        This was the ONLY spot source until the Index entitlement landed on 2026-08-21; `I:SPX`
        now serves a real level (massive/index.py) and the caller passes it in as `spot=`. Parity
        is kept as the fallback and the cross-check, not deleted: it is derived from the same
        quote snapshot the theos are, so its errors partially cancel against them, and its
        divergence from the index is a health signal on both feeds.

        **`strict=True` (the default since 2026-08-24) is the whole safety property.** Both legs
        at a strike must be genuinely two-sided (`theo.two_sided`). Without it a deep-ITM call
        with no bid prices to exactly 0.00 -- absent means zero, not unknown -- so |C - P|
        collapses to the tiny OTM premium, that strike wins the anchor, and the result is
        S = K, its own strike. Measured on 2026-08-19: spot 7387.5 on a day whose true spot was
        ~7706. `atmvol.two_sided` and `theo_bt.parity_spot(strict=True)` had carried this guard
        since they were written; this method did not, and it is the one the LIVE loop reaches --
        via `validate_theo.resolve_and_price` -> `spot.resolve` whenever the index is absent or
        stale. Bracketing strikes span about +-4.6% of spot, so a wrong anchor is not a wing
        effect: it reprices the entire board off a forward hundreds of points out.

        Returning MISSING when no strike is two-sided is deliberate and fails SAFE:
        `spot.resolve` then reports `..._+no_parity` with `value=None`, and `run_04` skips the
        cycle rather than quoting off a fabricated spot. That is the same clean skip the loop
        already takes outside 09:30-16:00 ET, when SPXW stops quoting.

        `strict=False` reproduces the pre-2026-08-24 behaviour and exists only so a study can
        measure the gap. Nothing on the live path passes it.
        """
        pairs = []
        for k in self.strikes:
            if strict and not (two_sided(self.quotes.get(("C", k)))
                               and two_sided(self.quotes.get(("P", k)))):
                continue
            c, p = self.val("C", k), self.val("P", k)
            if c is not None and p is not None:
                pairs.append((abs(c - p), k, k + c - p))
        if not pairs:
            return MISSING
        anchor = min(pairs)[2]
        near = [s for _, k, s in pairs if abs(k - anchor) <= width]
        return median(near) if near else anchor

    def implied_w_at(self, strike):
        """Total vol implied by the options AT `strike`, or (MISSING, detail).

        **Only the OUT-OF-THE-MONEY leg is used**, and this is the load-bearing decision.

        An ITM option is almost all intrinsic, so its time value is a small difference of two
        large numbers and inherits the full width of its quote. Measured on the 2026-08-19 board
        at spot 7704.30: the 7975 put was quoted 259.5 / 280.0 -- **20.5 points wide** -- giving a
        microprice of 270.93 against an intrinsic of 270.70, i.e. 0.23 of "time value" that is
        entirely quote noise. Inverting that yields w = 0.0139 against an ATM w of 0.0024, a 5.7x
        overstatement, and a far-OTM digital priced off it came out at 0.6% instead of ~0. The OTM
        leg at the same strike is all time value and carries no intrinsic to subtract.

        Note why vega weighting cannot substitute for this: put-call parity makes vega IDENTICAL
        for the call and the put at a given strike (see bs.vega_w -- "same for calls and puts"),
        so a vega-weighted mean of the two legs is just their plain average and cannot discriminate
        between them at all. An earlier version of this method did exactly that and was silently
        a no-op.

        `bs.implied_w` returns MISSING -- not a clamp -- when a price is through a no-arbitrage
        bound or its time value is under a tick. Those quotes are the norm on a 0DTE board, and
        letting them vote a fabricated zero into the answer is the failure this avoids.

        No time-to-expiry appears anywhere: `bs` is parameterised by total vol w = sigma*sqrt(T),
        and the options being read expire at exactly the instant the market settles, so the w
        they imply IS the w the digital needs. That identity is why this is valid only on the
        16:00 board (pricing/hourly handles every other settlement).
        """
        d = {"w_call": None, "w_put": None, "w_leg": None, "w": None}
        F = self.spot
        if F is None:
            return MISSING, d
        # Both are computed for visibility; only the OTM one is used.
        c, pu = self.val("C", strike), self.val("P", strike)
        d["w_call"] = bs.implied_w(F, strike, c, "C") if c is not None else MISSING
        d["w_put"] = bs.implied_w(F, strike, pu, "P") if pu is not None else MISSING

        otm = "C" if strike >= F else "P"
        w = d["w_call"] if otm == "C" else d["w_put"]
        if w is MISSING:
            # No fallback to the ITM leg on purpose. If the OTM option cannot identify a vol,
            # the honest answer is that this strike does not identify one -- the ITM leg would
            # supply a number built from quote width, which is worse than no number.
            return MISSING, d
        d["w_leg"], d["w"] = otm, w
        return w, d

    def digital_above_delta(self, strike):
        """P(final index > strike) from the option price LEVEL at `strike`, via Black-Scholes.

        This is `N(d2)`, not the delta `N(d1)`. They differ by roughly phi(0)*w -- ~0.08c at a
        quiet hour's vol and ~0.32c in a stressed one (see pricing/bs.p_above). Under the tick,
        but a real fraction of the 1c quote offset, and N(d2) is not more expensive to compute.
        `N(d1)` is recorded alongside in the detail dict so the adds/removes study can settle the
        choice with PnL rather than argument.

        When `strike` is not a listed strike -- KXINXDUD's 7691.76, and NDX boundaries off the
        10-wide grid -- w is interpolated between the bracketing strikes IN TOTAL-VOL SPACE. Vol
        is far smoother across strikes than price is, so interpolating price would import the
        curvature that the whole smile has.
        """
        F = self.spot
        if F is None:
            return MISSING
        if strike in self._strike_set:
            w, _ = self.implied_w_at(strike)
        else:
            lo, hi = neighbors(strike, self.strikes)
            if lo is None:
                return MISSING
            w_lo, _ = self.implied_w_at(lo)
            w_hi, _ = self.implied_w_at(hi)
            if w_lo is MISSING and w_hi is MISSING:
                return MISSING
            if w_lo is MISSING:
                w = w_hi
            elif w_hi is MISSING:
                w = w_lo
            else:
                f = (strike - lo) / (hi - lo)
                w = w_lo + f * (w_hi - w_lo)
        if w is MISSING:
            return MISSING
        return bs.p_above(F, strike, w)

    def digital_above(self, strike):
        """P(final index > strike). The only estimator: Black-Scholes on the skew curve.

        `N(d2)` at a vol read off the fitted smile -- `w(K) = w_atm * ratio(K)` -- rather than at
        one flat ATM vol, and rather than off the slope of the option curve. When the curve refuses
        to fit, `ratio` is 1.0 everywhere and this degrades exactly to the flat-vol model.
        """
        sf = self.skew
        if sf is not None and sf.w_atm is not None:
            return sf.p_above(strike)
        # No ATM vol at all -- not even a flat curve can be served. Fall back to the per-strike
        # implied vol at this boundary, which is what priced the board before the curve existed.
        # Strictly better than refusing: it is the same N(d2), just unsmoothed.
        return self.digital_above_delta(strike)

    def digital_detail(self, strike):
        """Everything behind one digital, for the journal and the study. Never used to price."""
        sf = self.skew
        out = sf.p_above(strike) if sf is not None else MISSING
        d = {"boundary": strike, "p_skew": out,
             "skew_reason": (sf.reason if sf is not None else "no_skew_fit"),
             "skew_flat": (sf.flat if sf is not None else True),
             "w_atm": (sf.w_atm if sf is not None else None),
             "skew_ratio": (sf.ratio(strike) if sf is not None else None),
             "w_used": (sf.w_of(strike) if sf is not None else None)}
        # The raw per-strike vol at this boundary, unsmoothed -- recorded so a boundary whose own
        # quote disagrees with the fitted curve is visible in the journal rather than only in the
        # aggregate.
        w, wd = (self.implied_w_at(strike) if strike in self._strike_set else (None, {}))
        d.update(wd)
        d["p_raw_strike_vol"] = (bs.p_above(self.spot, strike, w)
                                 if (w not in (None, MISSING) and self.spot) else None)
        return out, d

    def market_fair(self, market):
        """Fair value for one Kalshi market. Returns (fair, detail-dict).

        A thin adapter: all bracket arithmetic lives in the pure `bracket_fair` so the
        backtest can share it rather than maintain a parallel copy that drifts.
        """
        fair, d = bracket_fair(market.get("strike_type"),
                               market.get("floor_strike"),
                               market.get("cap_strike"),
                               self.digital_above)
        d["spot"] = self.spot
        sf = self.skew
        d["skew_reason"] = sf.reason if sf is not None else "no_skew_fit"
        d["skew_flat"] = sf.flat if sf is not None else True
        d["skew_buckets"] = sf.n_buckets if sf is not None else 0
        d["w_atm"] = sf.w_atm if sf is not None else None
        # Per-boundary provenance, attached HERE rather than inside bracket_fair so that function
        # stays pure and the backtest can keep sharing it. Keyed by boundary because a `between`
        # market prices off two and they can degrade differently -- one on the fitted curve,
        # one on the flat fallback.
        #
        # `priced` is per BOUNDARY. `d["reason"]` is per MARKET, and on a `between` whose cap has
        # no digital but whose floor priced fine, reading the market's "no_digital" against each
        # boundary in this dict tags the floor as unpriced when it was not. The term's own
        # recorded value is the authority, so it is carried across.
        d["digitals"] = {}
        for t in d.get("terms", []):
            det = self.digital_detail(t["boundary"])[1]
            det["priced"] = t.get("value") is not None
            det["boundary_reason"] = None if det["priced"] else "no_digital"
            d["digitals"][t["boundary"]] = det
        return fair, d


def boundaries(markets):
    """Every distinct boundary a *priceable* market actually prices from.

    Rule-aware rather than "every non-null strike field": an unpriceable type has no boundary
    worth subscribing option strikes for, and a one-sided market prices from one end only.
    For today's KXINX event this yields the identical set it always did.
    """
    out = set()
    for m in markets:
        for term in STRIKE_TYPE_RULE.get(m.get("strike_type"), ()):
            b = normalize_boundary(m.get(term.bound))
            if b is not None:
                out.add(b)
    return sorted(out)
