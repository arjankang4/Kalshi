"""Prices Kalshi's HOURLY index markets, which have no option expiring when they settle.

The daily board is priced off the option board itself: `theo.Pricer` fits a smile and takes
P(S > K) = -dC/dK of it. That only answers the question the option board is asked -- what the
index does by 16:00 ET -- because 16:00 is the only instant a listed SPXW/NDXP contract expires
at. (This said "model-free off a vertical call or put spread" until 2026-08-24; the vertical was
removed on 2026-08-21 and the slope term `-F*phi(d1)*dw/dK` went missing with it for three days.
`docs/pricing-invariants.md`.) The hourly boards (`KXINXU`,
`KXNASDAQ100U`, events H1000..H1600) settle at 10:00..15:00, where nothing expires.

So the hourly path is the daily path plus a distribution:

    pricing/atmvol   quotes  -> forward F and total vol w_4pm to 16:00   (measured)
    pricing/varcal   now, h  -> what fraction of the remaining variance lands by h  (fitted)
    pricing/bs       F, K, w -> P(S > K) = N(d2)                          (assumed lognormal)

and then `theo.bracket_fair` does the Kalshi bracket arithmetic, unchanged. That last point is
the design: `bracket_fair` takes a `p_above(K)` callable and nothing else, so every strike_type
rule, the boundary normalisation that collapses 7574.9999 into 7575, the rectangular detail dict
and the null-reason taxonomy are shared with the live daily pricer rather than reimplemented
next to it.

`HourlyPricer` exposes `.spot` and `.market_fair(m) -> (fair, detail)` -- the same duck type as
`theo.Pricer` -- so wiring it into a quote cycle later is a swap at
`validate_theo.price_markets`, and `strategy/quoting.py` needs no change at all.

WHAT IS ASSUMED, stated plainly:
  * Zero skew. One ATM vol prices every strike. Defensible where it matters -- inside ~3 sigma of
    the forward -- and increasingly not, outside it. On a 60-rung SPX hourly ladder only ~19 rungs
    are inside that window; the rest are worth ~0 or ~1 whatever the smile does.
  * Flat vol across the day. The 4pm number is stretched to a sub-hour horizon by the variance
    calendar, which knows the average shape of a session but nothing about *today*. On an FOMC or
    CPI day the 14:00 hour needs more variance left in it than the average day allocates.
  * The parity forward is the forward to 16:00, not to h. Intraday dividends make those differ by
    <0.02% except around quarterly ex-dates. Not modelled.
"""
from datetime import datetime, time as dtime, timezone

import config
from pricing import atmvol, bs, skew as skewmod, theo
from pricing.theo import MISSING

CLOSE_T = dtime(16, 0)

# Refuse to price an hourly market with less remaining variance than this. Illustrative value; the
# live value is not published.
MIN_VARIANCE_LEFT = 1e-05

# Presentation only. One intraday session counts as one "variance day", so a year is 252 of them.
# Nothing in the pricing path divides or multiplies by this -- the model works in total vol and
# scales by a ratio, in which the annualisation cancels. It exists so the fitted vol can be read
# in the units a trader thinks in, and so a wrong-looking number is obvious at a glance.
TRADING_DAYS_PER_YEAR = 252


def parse_settlement(market):
    """The instant a Kalshi market settles against.

    **`close_time`, not `expiration_time`.** On `KXINXU-26AUG20H1500-T7819.9999` the settlement
    instant is `close_time = 2026-08-20T19:00:00Z` (3pm ET), while `expiration_time` is
    `2026-08-27T22:00:00Z` -- a week later, when the contract is administratively expired. Using
    the latter as a time to expiry would price an hourly market as an eight-day option.
    """
    raw = market.get("close_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def expiry_4pm(now):
    """16:00 ET on the ET date of `now` -- when the 0DTE options being read actually expire."""
    et = now.astimezone(config.ET)
    return datetime.combine(et.date(), CLOSE_T, tzinfo=config.ET)


class HourlyPricer:
    """Prices any strike_type on any settlement instant up to today's 16:00.

    Construct one per quote snapshot; it holds no state across time.
    """

    def __init__(self, quotes, strikes, now, calendar, fit=None, index_forward=None,
                 skew_fit=None):
        self.quotes = quotes
        self.strikes = sorted(strikes)
        self.now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        self.calendar = calendar
        # index_forward is a FALLBACK for when parity cannot price itself, not an override --
        # see atmvol.fit on why substituting it trips the cp_vol_spread gate.
        self.fit = (fit if fit is not None
                    else atmvol.fit(quotes, self.strikes, index_forward=index_forward))
        self.expiry_4pm = expiry_4pm(self.now)
        self._w_cache = {}
        self._skew = skew_fit
        self._skew_built = skew_fit is not None

    @property
    def skew(self):
        """The fitted vol curve for this board, built once on first use.

        Fitted against the 16:00 forward and 16:00 ATM vol, because that is the expiry the listed
        options actually have -- there is no separate option board at 15:00 to read a smile from.
        """
        if not self._skew_built:
            self._skew = (skewmod.fit(self.quotes, self.strikes, self.fit.forward, self.fit.w)
                          if self.fit.ok else None)
            self._skew_built = True
        return self._skew

    def use_skew(self, skew_fit):
        """Install a curve fitted elsewhere, after construction. Mirrors `theo.Pricer.use_skew`.

        The live loop reuses ONE `strategy.skewstate.SkewState` across cycles and across boards --
        the smile is a property of the SPXW option board, not of the Kalshi board being priced, so
        the 4pm brackets and every hourly ladder read the same curve. Without this setter the
        hourly pricer could only take a curve at construction, and the smoothed one does not exist
        until spot has been resolved. That gap is why `validate_theo.resolve_and_price` could not
        price an hourly board at all before 2026-08-25.

        Note the curve is fitted against the 16:00 forward and 16:00 ATM vol -- see `.skew`. The
        hourly rescaling happens in `total_vol_to`, not in the curve.
        """
        self._skew = skew_fit
        self._skew_built = True
        return self

    # `theo.Pricer` calls this `spot`, and with r = q = 0 at 0DTE the forward is the spot.
    # Same name so the two are interchangeable at the seam.
    @property
    def spot(self):
        return self.fit.forward

    @property
    def w_4pm(self):
        return self.fit.w

    @property
    def sigma_annualized(self):
        """The fitted ATM vol as an annualised sigma, on the variance clock. Diagnostic only.

        w_4pm is the total stdev from now to 16:00, which is `1 - Phi(now)` variance-days of
        exposure, so sigma = w / sqrt((1 - Phi(now)) / 252).

        Reading it on the VARIANCE clock rather than the clock clock matters: at 10:15 there are
        5h45m left of a 6.5h session (88% by the clock) but only 78% of the variance. Annualising
        on wall time would report a sigma ~6% below the one the model is actually using, and the
        two would silently disagree.
        """
        if not self.fit.ok:
            return None
        left = 1.0 - self.calendar.phi(self.now)
        if left <= MIN_VARIANCE_LEFT:
            return None
        return self.fit.w / ((left / TRADING_DAYS_PER_YEAR) ** 0.5)

    def total_vol_to(self, settlement):
        """(w, ratio, reason) for a settlement instant. Cached per instant."""
        key = settlement
        if key in self._w_cache:
            return self._w_cache[key]
        out = self._compute_w(settlement)
        self._w_cache[key] = out
        return out

    def _compute_w(self, settlement):
        if not self.fit.ok:
            return None, None, self.fit.reason or "no_atm_fit"
        if settlement is None:
            return None, None, "no_close_time"
        if settlement <= self.now:
            return None, None, "expired"
        if 1.0 - self.calendar.phi(self.now) <= MIN_VARIANCE_LEFT:
            return None, None, "no_variance_left"
        ratio = self.calendar.remaining_ratio(self.now, settlement)
        if ratio is None:
            return None, None, "no_variance_left"
        if ratio <= 0.0:
            return None, None, "no_variance_in_window"
        # w scales as the square root of variance, and the ratio is a variance ratio.
        #
        # Nothing here is annualised. The 4pm vol already prices only the variance from now to
        # 16:00, so the overnight gap is outside both the numerator and the denominator and no
        # intraday fraction has to be applied -- or double-applied. At settlement == 16:00 the
        # ratio is exactly 1 and this returns w_4pm untouched, which is the invariant that ties
        # this model to the validated daily one.
        return self.fit.w * (ratio ** 0.5), ratio, None

    def p_above_for(self, settlement):
        """`p_above(K) -> P(S_settlement > K) | None`, the callable bracket_fair consumes.

        The vol is per-strike, not one scalar: `w(K) = w_atm(settlement) * ratio(K)`, where the
        ratio comes off the skew curve fitted to the 16:00 board.

        **The ratio is read at the strike's position on the board where it was OBSERVED**, i.e. in
        16:00 sigma units, and then applied to the horizon's ATM vol. The same strike sits at more
        sigmas when less variance remains, so evaluating the curve at the horizon's own x instead
        would be a sticky-delta-across-horizons assumption -- plausible, untested here, and it
        would make a 15:00 market read its skew off a part of the curve no quote was taken from.
        Reading it where it was measured is the conservative choice; the alternative is worth an
        experiment, not a default.
        """
        w, _, _ = self.total_vol_to(settlement)
        if w is None:
            return lambda k: MISSING
        F = self.fit.forward
        sf = self.skew
        if sf is None or sf.flat:
            return lambda k: bs.p_above(F, k, w)
        # `p_above_at`, not `bs.p_above(F, k, w * ratio(k))`: the digital is `-dC/dK`, so a
        # strike-dependent vol contributes `-F*phi(d1)*dw/dK` on top of `N(d2)`. The ratio is still
        # read in 16:00 sigmas (see the docstring above) but the vol it multiplies is the
        # HORIZON's, so the term scales with `w` and `w_atm=w` is what carries that through.
        return lambda k: sf.p_above_at(k, w_atm=w)

    def market_fair(self, market):
        """Fair value for one Kalshi market. Returns (fair, detail), same contract as theo.Pricer."""
        settlement = parse_settlement(market)
        w, ratio, reason = self.total_vol_to(settlement)

        fair, d = theo.bracket_fair(market.get("strike_type"),
                                    market.get("floor_strike"),
                                    market.get("cap_strike"),
                                    self.p_above_for(settlement))
        d["spot"] = self.fit.forward
        d["forward"] = self.fit.forward
        d["w"] = w
        d["w_4pm"] = self.fit.w
        d["var_ratio"] = ratio
        d["settlement"] = settlement.isoformat() if settlement else None
        d["minutes_to_settlement"] = (
            (settlement - self.now).total_seconds() / 60.0 if settlement else None)
        d["sigma_annualized"] = self.sigma_annualized
        d["variance_left_to_4pm"] = 1.0 - self.calendar.phi(self.now)
        d["atm_reason"] = self.fit.reason
        d["forward_source"] = self.fit.forward_source
        d["parity_forward"] = self.fit.parity_forward
        d["index_minus_parity"] = self.fit.index_minus_parity
        d["cp_spread"] = self.fit.cp_spread
        d["vol_dispersion"] = self.fit.dispersion
        # bracket_fair can only report "no_digital" when p_above returns None. Overwrite with the
        # specific cause so a board full of unpriced markets says WHY, not just that it failed.
        if reason and fair is MISSING:
            d["reason"] = reason
            d["error"] = f"no vol for settlement {d['settlement']}: {reason}"
        return fair, d


def price_markets(markets, quotes, strikes, now, calendar, index_forward=None):
    """Mirror of `validate_theo.price_markets`, so the two are swappable at the call site."""
    pricer = HourlyPricer(quotes, strikes, now, calendar, index_forward=index_forward)
    return pricer, {m["ticker"]: pricer.market_fair(m) for m in markets}
