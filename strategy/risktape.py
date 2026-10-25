"""A row a second of what we are actually holding, for post-session risk analysis.

WHAT THIS ANSWERS THAT NOTHING ELSE DOES. The quote journal records what we INTENDED -- every
desired rung with the theo and the edge behind it -- and `/portfolio/fills` records what TRADED.
Neither says what the book looked like as a position: how much we make if the index rises a point,
how much if vol rises a point, and above all WHERE in the strike space that vol risk sits. On
2026-08-26 the session lost money in the last minutes of an hourly board, and the question "were we
long vol everywhere, or long downside vol and short upside vol?" could not be answered from
anything on disk.

THE VEGA BUCKETS ARE THE POINT. A binary's vega is signed -- zero at the money, positive on one
side and negative on the other -- so a portfolio vega of, say, +$100 can mean two completely
different books: long vol evenly, or long $400 of downside vol against short $300 of upside. Those
have opposite P&L under a skew move and identical summary statistics. Bucketing by the delta of the
VANILLA CALL at each market's reference strike puts the risk on the axis an options desk actually
reasons about, so "long the 25-delta wing, short the 60-delta body" becomes readable.

THE REFERENCE STRIKE, per market type:
  * `greater_or_equal` / `greater` -- the "at or above" hourly rungs: their own `floor_strike`.
  * `between` -- the 4pm brackets: the BOTTOM strike, i.e. `floor_strike`.
  * `less` -- the bottom bracket, which has no floor: `cap_strike`.

THE VANILLA IS THE 4PM ONE, deliberately. There are no listed hourly options; the only vanillas
that exist are the SPXW/NDXP contracts expiring at 16:00, which are also the surface every theo in
this repo is built from. Using each market's own horizon instead would make the bucket axis mean
something different for an 11:00 rung than for a 15:00 one, and the buckets would stop being
comparable across the session -- which is the one thing they are for. Delta is Black-76 forward
delta `N(d1)` at the smile's own `w(K)`, undiscounted, which at 0DTE with r=q=0 is the delta.

The table is one row per (instant, underlying). Flat, so it loads straight into a frame; the
buckets are columns rather than a nested dict for the same reason.
"""
import datetime
import time

import config
from pricing import bs
from pricing import greeks as gk
from strategy import edgemgr

# Cadence of the risk snapshot tape. Illustrative value; the live value is not published.
RISK_TAPE_S = getattr(config, "RISK_TAPE_S", 2.0)

# Ten-delta buckets, the axis an options desk reads. The last one is closed at 1.0 so a strike far
# below the forward -- call delta ~1 -- has somewhere to go.
BUCKET_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BUCKET_NAMES = tuple(f"d{int(BUCKET_EDGES[i] * 100):02d}_{int(BUCKET_EDGES[i + 1] * 100):02d}"
                     for i in range(len(BUCKET_EDGES) - 1))


def reference_strike(market):
    """The strike whose vanilla call delta buckets this market. None if it has neither bound."""
    lo, hi = market.get("floor_strike"), market.get("cap_strike")
    return lo if lo is not None else hi


def vanilla_call_delta(forward, strike, skew_fit=None, w_atm=None):
    """Black-76 forward delta `N(d1)` of the 4pm vanilla call at `strike`, or None.

    The smile is read where it was fitted -- `SkewFit.ratio(K)` returns `w(K)/w_atm` -- so a wing
    strike is bucketed at its own vol rather than at the money's. A flat fit returns ratio 1.0 and
    this degrades to Black-76 at the ATM vol, which is the right fallback and not a silent one:
    `skew_flat` is already reported in the cycle record.
    """
    if forward is None or strike is None or forward <= 0 or strike <= 0:
        return None
    w = None
    if skew_fit is not None and getattr(skew_fit, "w_atm", None):
        w = skew_fit.w_atm * skew_fit.ratio(strike)
    elif w_atm:
        w = w_atm
    if not w or w <= 0:
        return None
    d1, _d2 = bs.d1_d2(forward, strike, w)
    return bs.norm_cdf(d1)


def bucket_of(delta):
    """Index into BUCKET_NAMES, or None."""
    if delta is None:
        return None
    d = min(max(float(delta), 0.0), 1.0)
    for i in range(len(BUCKET_EDGES) - 1):
        if d < BUCKET_EDGES[i + 1] or i == len(BUCKET_EDGES) - 2:
            return i
    return None


class RiskTape:
    """Writes one flat row per (instant, underlying)."""

    def __init__(self, journal, interval_s=None):
        self.journal = journal
        self.interval_s = RISK_TAPE_S if interval_s is None else interval_s
        self._last = 0.0
        self.rows = 0
        self.errors = 0
        self.last_error = None

    def due(self, now_mono=None):
        now_mono = time.monotonic() if now_mono is None else now_mono
        return (now_mono - self._last) >= self.interval_s

    def snapshot(self, underlying, now, positions, em_map, markets, pricer, calendar,
                 sigma_impl=None, sigma_rv=None, spot=None, fast_spot=None,
                 hourly_markets=(), now_mono=None, write=True):
        """Build (and by default write) one row. Returns the row dict."""
        now_mono = time.monotonic() if now_mono is None else now_mono
        forward = getattr(pricer, "spot", None)
        skew_fit = getattr(pricer, "skew", None)
        by_ticker = {m["ticker"]: m for m in markets}

        # A FALLBACK ATM VOL, so the bucket axis never collapses quietly. Without one, a cycle with
        # no fitted smile -- a cold start, a refused fit, or any caller that hands over a bare
        # pricer -- puts EVERY market in `vega_unbucketed`, which reads as "we could not attribute
        # the risk" when the truth is only "we had no curve". Reconstructed from the annualised ATM
        # vol and the variance still to run, which is exactly the inverse of `run_04._sigma`:
        #   sigma = w / sqrt(var_left / 252)   =>   w = sigma * sqrt(var_left / 252)
        # Flat, so a wing strike is bucketed at the money's vol. That is a worse axis than the
        # smile's and a far better one than none; `skew_flat` in the row says which was used.
        w_fallback = None
        try:
            if sigma_impl and calendar is not None:
                left = 1.0 - calendar.phi(now)
                if left > 0:
                    w_fallback = float(sigma_impl) * ((left / 252.0) ** 0.5)
        except Exception:                                    # noqa: BLE001
            w_fallback = None

        row = {"v": 1, "ts_ns": time.time_ns(),
               "et": now.isoformat() if isinstance(now, datetime.datetime) else str(now),
               "underlying": underlying, "spot": spot, "fast_spot": fast_spot,
               "forward": forward,
               "sigma_impl": sigma_impl, "sigma_rv": sigma_rv,
               "n_positions": 0, "net_contracts": 0.0, "abs_contracts": 0.0,
               "delta_dollars": 0.0, "vega_dollars": 0.0, "theta_dollars": 0.0,
               "vega_bucketed": 0.0, "vega_unbucketed": 0.0,
               "skew_flat": bool(getattr(skew_fit, "flat", True)) if skew_fit else None}
        for n in BUCKET_NAMES:
            row[f"vega_{n}"] = 0.0
            row[f"pos_{n}"] = 0.0

        for tk, pos in (positions or {}).items():
            pos = float(pos or 0.0)
            if not pos:
                continue
            m = by_ticker.get(tk)
            if m is None:
                continue                       # another underlying's market
            g = (em_map or {}).get(tk) or {}
            row["n_positions"] += 1
            row["net_contracts"] += pos
            row["abs_contracts"] += abs(pos)
            # PER CONTRACT, SIGNED, times the signed position. `edgemgr` reports delta as dollars
            # per ONE index point and vega as dollars per ONE vol point (see pricing/greeks), which
            # is exactly the unit this table is specified in -- no rescaling here, deliberately, so
            # a unit change upstream shows up as a wrong number rather than a silently corrected
            # one.
            d, v, th = g.get("delta"), g.get("vega"), g.get("theta")
            if d is not None:
                row["delta_dollars"] += pos * float(d)
            if th is not None:
                row["theta_dollars"] += pos * float(th)
            if v is None:
                continue
            vega = pos * float(v)
            row["vega_dollars"] += vega
            k = reference_strike(m)
            cd = vanilla_call_delta(forward, k, skew_fit, w_atm=w_fallback)
            b = bucket_of(cd)
            if b is None:
                row["vega_unbucketed"] += vega
                continue
            row[f"vega_{BUCKET_NAMES[b]}"] += vega
            row[f"pos_{BUCKET_NAMES[b]}"] += pos
            row["vega_bucketed"] += vega

        # THE VARIANCE CLOCK, both horizons. `phi` is the fraction of the session's variance already
        # elapsed, so `1 - phi` is what is left to 16:00; `var_left` is the same clock measured to
        # an arbitrary settlement, which is what an hourly rung actually expires against. One
        # session is one variance-day, so both are fractions of a session and directly comparable.
        try:
            if calendar is not None:
                row["var_left_day"] = 1.0 - calendar.phi(now)
        except Exception as e:                                   # noqa: BLE001
            row["var_left_day"] = None
            self.last_error = f"phi: {type(e).__name__}: {str(e)[:60]}"
        row.setdefault("var_left_day", None)

        settle, tkr = self._soonest(hourly_markets, now)
        row["next_hourly"] = settle.isoformat() if settle is not None else None
        row["next_hourly_ticker"] = tkr
        row["minutes_to_hourly"] = (None if settle is None
                                    else round((settle - now).total_seconds() / 60.0, 3))
        vl = None
        if settle is not None and calendar is not None:
            try:
                vl = gk.var_left(calendar, now, settle)
            except Exception:                                    # noqa: BLE001
                vl = None
        row["var_left_hourly"] = None if vl is None or vl is gk.MISSING else float(vl)

        if write and self.journal is not None:
            try:
                self.journal.write(row)
                self.rows += 1
                self._last = now_mono
            except Exception as e:                               # noqa: BLE001
                self.errors += 1
                self.last_error = f"write: {type(e).__name__}: {str(e)[:80]}"
        return row

    @staticmethod
    def _soonest(hourly_markets, now):
        best = None
        for m in hourly_markets or ():
            try:
                # `edgemgr.settlement_of`, not a local parse: it owns the close_time
                # normalisation and a second copy would drift from it.
                s = edgemgr.settlement_of(m)
            except Exception:                                    # noqa: BLE001
                s = None
            if s is None or s <= now:
                continue
            if best is None or s < best[0]:
                best = (s, m.get("ticker"))
        return best or (None, None)

    def detail(self):
        return {"rows": self.rows, "errors": self.errors, "last_error": self.last_error,
                "interval_s": self.interval_s}
