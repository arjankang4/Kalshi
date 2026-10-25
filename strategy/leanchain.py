"""Build the realized-vol shadow probability chain the lean reads.

Two probability chains over the same board:

    p_impl(K)  the theo -- the fitted smile's -dC/dK on the 4pm board, the same curve scaled
               by the variance calendar on the hourly ones
    p_rv(K)    the same strikes priced off the realized-vol model's sigma

`pricing/lean` turns the disagreement into a per-side offset. This module only produces `p_rv`
and the two sigmas, and it is written to fail soft: any missing piece yields an empty chain, which
makes `side_offset` return the base offset and the quote comes out exactly as it would have.

COMPUTED EVEN WHEN THE LEAN IS DISABLED. `config.LEAN_ENABLED` gates whether the chain reaches
`quoting.decide`, not whether it is calculated, so a session's worth of would-be leans lands in
the journal before any of them is acted on. That is the evidence needed to decide whether to turn
it on, and it costs one extra board evaluation every 5 seconds.
"""
import datetime
import math

import config
from pricing import atmvol, bs, varcal
from strategy import edgemgr
from rvol import live as rlive
from rvol import model as rmodel

CLOSE_T = datetime.time(16, 0)
TRADING_DAYS = 252.0


class LeanChain:
    """Holds the per-session state: calendar, fitted model, and the index ring buffer."""

    def __init__(self, underlying):
        self.underlying = underlying
        self.ok = False
        self.reason = None
        try:
            self.cal = varcal.load(underlying)
            self.model = rmodel.load(underlying)
            self.buf = rlive.IndexBuffer(underlying)
            self.rv = rlive.LiveRealizedVol(underlying, calendar=self.cal, buffer=self.buf)
            self.ok = True
        except Exception as e:                                   # noqa: BLE001
            self.reason = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"[lean] unavailable ({self.reason}) -- quoting unleaned", flush=True)

    def start(self):
        if self.ok:
            self.buf.backfill()
        return self.ok and self.buf.backfilled

    def observe(self, index_quote):
        if self.ok:
            self.buf.append(index_quote)

    def sigmas(self, now, quotes, strikes):
        """(sigma_rv, sigma_impl, detail). Either may be None."""
        d = {"lean_reason": None, "buffer_span_s": self.buf.span_s() if self.ok else 0.0}
        if not self.ok:
            d["lean_reason"] = self.reason or "unavailable"
            return None, None, d

        left = 1.0 - self.cal.phi(now)
        d["variance_left"] = left
        if left <= 1e-6:
            d["lean_reason"] = "no_variance_left"
            return None, None, d

        fit = atmvol.fit(quotes, strikes)
        if not fit.ok:
            d["lean_reason"] = f"atm_{fit.reason or 'unfit'}"
            return None, None, d
        # Same clock both sides: one RTH session is one variance-day, 252 a year, overnight never
        # enters. This is HourlyPricer.sigma_annualized, and the realized side is annualized the
        # same way in rvol.features.
        sigma_impl = fit.w / math.sqrt(left / TRADING_DAYS)
        d["w_impl"], d["sigma_impl"], d["forward"] = fit.w, sigma_impl, fit.forward

        feats = self.rv.features(now)
        if not feats:
            d["lean_reason"] = "no_rv_features"
            return None, sigma_impl, d
        d["rv_features"] = {k: (None if v is None or v != v else round(v, 6))
                            for k, v in feats.items() if k.startswith(("f_", "pf_"))}

        minutes_left = max(1.0, (self._close(now) - now).total_seconds() / 60.0)
        sigma_rv, horizon = self.model.predict_for_minutes(minutes_left, feats)
        d["rv_horizon"], d["minutes_left"] = horizon, minutes_left
        if sigma_rv is None:
            d["lean_reason"] = "rv_predict_none"
            return None, sigma_impl, d
        d["sigma_rv"] = sigma_rv
        return sigma_rv, sigma_impl, d

    @staticmethod
    def _close(now):
        et = now.astimezone(config.ET)
        return datetime.datetime.combine(et.date(), CLOSE_T, tzinfo=config.ET)

    def short_signal(self, now, sigma_impl=None):
        """The near-term vol-spike signal for `pricing/volspike`, one dict per underlying.

        `s` is log(sigma_<VOLSPIKE_HORIZON> / sigma_30m) -- realized vs realized, same clock.
        `s_impl` is log(sigma_1m / sigma_impl), journalled for COMPARISON only: it carries the
        variance risk premium as an offset. Computed whether or not VOLSPIKE_ENABLED is set.
        Never raises; a missing piece comes back as `s=None` with a reason.
        """
        from pricing import volspike
        out = {"s": None, "reason": None, "horizon": config.VOLSPIKE_HORIZON,
               "sigma_impl": sigma_impl}
        if not self.ok:
            out["reason"] = self.reason or "unavailable"
            return out
        if not getattr(self.model, "short_models", None):
            out["reason"] = "no_short_models"
            return out
        try:
            fc = self.rv.short_forecasts(self.model, now)
        except Exception as e:                                   # noqa: BLE001
            out["reason"] = f"{type(e).__name__}: {str(e)[:60]}"
            return out
        if not fc:
            out["reason"] = "no_rv_features"
            return out
        for h in ("1m", "3m", "30m"):
            out[f"sigma_{h}"] = fc.get(h)
        out["f_1m"] = fc.get("f_1m")
        out["s_1m"] = volspike.signal(fc.get("1m"), fc.get("30m"))
        out["s_3m"] = volspike.signal(fc.get("3m"), fc.get("30m"))
        out["s_impl"] = volspike.signal(fc.get("1m"), sigma_impl)
        out["s"] = out.get(f"s_{config.VOLSPIKE_HORIZON}")
        if out["s"] is None:
            out["reason"] = "rv_predict_none"
        return out

    def chain(self, markets, quotes, strikes, now, forward, enabled=None, skew_fit=None):
        """{ticker: {...}} for every market the realized-vol view has something to say about.

        THREE RULES, all of them 2026-08-25:

        1. **The 4pm board never gets a lean.** Its digital is priced off options that expire at
           exactly that instant, so implied vol is not a forecast there -- it is the market's own
           terminal distribution, and the realized-vol model has no horizon that reaches it (the
           longest fitted one is 60 minutes against a 390-minute session). Only the hourly ladders
           are leaned.

        2. **`p_rv` goes through the SMILE**, not through flat Black-Scholes. We hold a fitted
           smile in production and the implied chain uses it; comparing a smiled chain against a
           flat one would book the smile itself as a disagreement, largest exactly in the wings
           where the lean is most aggressive. `skew_fit` is the same curve the pricer installed.

        3. **The comparison is against RAW IMPLIED**, with no variance-risk-premium centring.
           Operator decision, 2026-08-25, taken with the following measured in front of it:
           realized sits systematically below implied (4.79% against 7.87%, ratio 0.61), and with
           NO information content at all that ratio reads **-9.33 points** on a mid-board hourly
           rung -- near the +-10 clamp, from premium alone. Across the hourly board at 0.61:
           median |d| = 0.00 (near-the-money digitals barely move with vol), p90 = 5.79, 2.3%
           beyond the clamp, all of it in the wings.

           SO BE CLEAR ABOUT WHAT THIS LEAN IS. Uncentred, it harvests the variance risk premium:
           realized below implied makes `p_rv < p_impl` above the forward and `p_rv > p_impl`
           below it, so we systematically sell upside and buy downside -- a short-vol posture. That
           is a real and well-documented edge, and it is NOT the claim "our realized-vol model
           forecasts better than implied does". It pays most days and loses badly when vol jumps.
           What bounds the tail is the risk plumbing, not this module: per-market position limits
           (band and wing), per-market collateral, and the book-wide collateral cap.

           `config.LEAN_VRP_RATIO` no longer exists on this path. It survives only as a parameter
           of the retired `lean.side_offset` arm that `backtest/analysis` still studies.
        """
        enabled = config.LEAN_ENABLED if enabled is None else enabled
        sigma_rv_session, sigma_impl, d = self.sigmas(now, quotes, strikes)
        out = {"_meta": {**d, "enabled": enabled,
                         "smiled": skew_fit is not None and getattr(skew_fit, "ok", False)}}
        if sigma_impl is None or forward is None:
            return out

        feats = self.rv.features(now) if self.ok else None
        w_impl_4pm = d.get("w_impl")

        def _p(k, w):
            if w is None or w <= 0:
                return None
            if skew_fit is not None and getattr(skew_fit, "ok", False) and not skew_fit.flat:
                return skew_fit.p_above_at(k, w_atm=w)
            return bs.p_above(forward, k, w)

        for m in markets:
            tk = m["ticker"]
            settlement = edgemgr.settlement_of(m)
            row = {"p_rv": None, "p_rv_shadow": None, "p_ref": None,
                   "sigma_rv": None, "sigma_impl": sigma_impl, "w_rv": None,
                   "enabled": enabled, "reason": None}
            if settlement is None:
                row["reason"] = "no_settlement"
                out[tk] = row
                continue
            if settlement.astimezone(config.ET).time() >= CLOSE_T:
                # Rule 1. Not an error -- the 4pm board is quoted on implied vol by design.
                row["reason"] = "4pm_board_implied_only"
                out[tk] = row
                continue
            ratio = self.cal.remaining_ratio(now, settlement)
            if not ratio or feats is None or w_impl_4pm is None:
                row["reason"] = ("no_variance_left" if not ratio
                                 else "no_rv_features" if feats is None else "no_w_impl")
                out[tk] = row
                continue

            minutes = max(1.0, (settlement - now).total_seconds() / 60.0)
            sigma_rv, horizon = self.model.predict_for_minutes(minutes, feats)
            if sigma_rv is None:
                row["reason"] = "rv_predict_none"
                out[tk] = row
                continue

            w_impl = w_impl_4pm * math.sqrt(ratio)
            w_rv = w_impl * (sigma_rv / sigma_impl) if sigma_impl else None
            k = m.get("floor_strike") if m.get("strike_type") != "between" else None
            p_rv = _p(k, w_rv) if k is not None else None
            # The reference is the IMPLIED chain itself. No premium centring -- see rule 3.
            p_ref = _p(k, w_impl) if k is not None else None
            row.update({"p_rv_shadow": p_rv, "p_ref": p_ref, "sigma_rv": sigma_rv,
                        "w_rv": w_rv, "w_impl": w_impl, "rv_horizon": horizon,
                        "minutes": round(minutes, 1),
                        "p_rv": p_rv if (enabled and p_ref is not None) else None})
            out[tk] = row
        return out

    @staticmethod
    def _close(now):
        et = now.astimezone(config.ET)
        return datetime.datetime.combine(et.date(), CLOSE_T, tzinfo=config.ET)
