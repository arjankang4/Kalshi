"""The live edge manager: dollar Greeks per market, refreshed on a slow cadence.

`pricing/edge` is the pure curve; this is the stateful half that knows what time it is and caches.
Same split as `pricing/varcal` (pure calendar) vs `strategy/leanchain` (live state).

WHY 30 SECONDS. The Greeks move with spot, vol and the clock, none of which change materially in
two seconds -- but the board is re-priced every two, so recomputing three Greeks across ~400
markets on every pass would be most of the loop's work for none of its value. `config
.EDGE_RECOMPUTE_S` bounds only the Greek work; the theo and the book stay on the fast cadence.

NO R, NO QUOTE. A market without a usable R is not quoted, rather than falling back to the flat
`config.QUOTE_OFFSET` quoter. The argument is simply that our fair
values are not proven enough to justify it -- a quote we cannot risk-price is a quote we should
not be showing, and "degrade to the old quoter" quietly reintroduces a pricing model we have since
decided against.

So every path that cannot produce an R now returns `quotable: False` and the market is skipped.
There are five, and they are all treated the same: no calendar, no calibration artifact, missing
board inputs (forward/sigma), no settlement timestamp, and a per-market R that comes back None.

The one deliberate exception is the operator override: `--edge` off installs a `pedge.FlatCurve`
on purpose, for reproducing pre-calibration behaviour. That path still quotes flat, and it says so
loudly at startup rather than looking like a healthy run.
"""
import datetime
import time

import config
from pricing import edge as pedge
from pricing import greeks, varcal


def settlement_of(market):
    """The instant a market's payoff is fixed, as a tz-aware datetime, or None.

    `close_time` is a STRING off the live REST API and a datetime out of Parquet, and normalising
    it here rather than at the call site is deliberate: the first version relied on the caller
    stashing a parsed copy, and every cycle then died comparing a str to a datetime because
    `close_time` won the `or`. A Greek path must not depend on who prepared its input.

    `close_time`, not `expected_expiration_time` -- the latter is a data-availability stamp and
    sits `close_time + 3h` on most daily rows.
    """
    v = market.get("_settlement") or market.get("close_time")
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if v.tzinfo is None:
        v = v.replace(tzinfo=datetime.timezone.utc)
    return v


class EdgeManager:
    """Per-market required edge and risk percentile, recomputed at most every EDGE_RECOMPUTE_S."""

    def __init__(self, underlying, curve=None, calendar=None, enabled=None):
        self.underlying = underlying
        self.enabled = config.EDGE_MANAGER_ENABLED if enabled is None else enabled
        self.reason = None
        self.calendar = calendar
        self.curve = curve
        if self.calendar is None:
            try:
                self.calendar = varcal.load(underlying)
            except Exception as e:                           # noqa: BLE001
                self.reason = f"no_calendar: {type(e).__name__}"
        if self.curve is None:
            try:
                self.curve = pedge.Curve.load(underlying)
            except Exception as e:                           # noqa: BLE001
                self.reason = f"no_calibration: {type(e).__name__}"
                self.curve = pedge.FlatCurve()
                self.enabled = False
        # Set by an operator asking for the flat quoter (`--edge` off). Distinct from
        # `enabled`, which also goes False when the calibration artifact is missing.
        self.operator_flat = False
        self._cache = {}
        self._cache_t = 0.0
        self.last_computed = None

    @property
    def available(self):
        return self.calendar is not None and isinstance(self.curve, pedge.Curve)

    def _flat(self, reason=None):
        """The no-R record. NOT quotable -- see the module docstring.

        Kept shaped like a priced record (and still carrying QUOTE_OFFSET) so the journal and any
        downstream reader see the same keys either way; `quotable: False` is what actually stops
        the quote at run_04_quote_loop's skip.
        """
        return {"edge_micros": int(round(config.QUOTE_OFFSET * 1e6)),
                "edge": config.QUOTE_OFFSET,
                "floor_micros": int(round(config.QUOTE_OFFSET * 1e6)), "bonus_micros": 0,
                "risk_u": 0.0, "scale_pos": 0.0, "R": None,
                "delta": None, "vega": None, "theta": None,
                "quotable": False, "source": "no_R",
                "reason": reason or self.reason or "no_R"}

    def _flat_override(self):
        """The ONLY quotable no-R record: the explicit `--edge` off operator override."""
        return {**self._flat(reason="edge_manager_disabled_by_operator"),
                "quotable": True, "source": "flat_override"}

    def _no_r(self, reason=None):
        """What every path that cannot produce an R returns.

        Not quotable, unless an operator explicitly asked for the flat quoter with `--edge` off.
        `operator_flat` is deliberately a separate flag from `enabled`: `enabled` also goes False
        when the calibration artifact fails to load, and that is a failure, not a choice.
        """
        return self._flat_override() if self.operator_flat else self._flat(reason)

    def refresh(self, markets, forward, sigma, now, force=False, move_scale=None):
        """Recompute R and the required edge for every market. Returns {ticker: detail}.

        Cached: a call inside the recompute window returns the previous map untouched, which is
        what keeps a 2-second quote loop from paying for a 30-second calculation.
        """
        t = time.monotonic()
        if not force and self._cache and (t - self._cache_t) < config.EDGE_RECOMPUTE_S:
            return self._cache
        if not self.available or forward is None or sigma is None or sigma <= 0:
            self._cache = {m["ticker"]: self._no_r() for m in markets}
            self._cache_t = t
            return self._cache

        out = {}
        for m in markets:
            settle = settlement_of(m)
            if settle is None:
                out[m["ticker"]] = self._no_r("no_settlement_ts")
                continue
            mins_left = (settle - now.astimezone(datetime.timezone.utc)).total_seconds() / 60.0
            if mins_left < config.MIN_MINUTES_TO_CLOSE:
                # Not a Greek problem to solve -- once the next minute reaches settlement the
                # contract resolves and its theta is (payoff - price), measured at -0.4999 for an
                # ATM rung at 15:59. Refuse explicitly. See config.MIN_MINUTES_TO_CLOSE.
                out[m["ticker"]] = {"edge_micros": None, "edge": None,
                                    "floor_micros": None, "bonus_micros": None, "risk_u": 1.0,
                                    "R": None, "delta": None, "vega": None, "theta": None,
                                    "scale_pos": 1.0, "quotable": False, "source": "gate",
                                    "reason": f"within_{config.MIN_MINUTES_TO_CLOSE}min_of_close",
                                    "minutes_to_close": mins_left}
                continue
            g = greeks.market_greeks(m, forward, sigma, now, settle, self.calendar)
            R = g["R"]
            em = self.curve.edge_micros(R) if R is not None else None
            if R is None or em is None:
                # No R, or an R past anything a year of history produced. The first degrades to
                # flat; the second is a refusal and must NOT degrade -- an R beyond the calibrated
                # range is a market we do not understand, not one to charge more for.
                if R is None:
                    out[m["ticker"]] = self._no_r(g.get("reason") or "no_R")
                else:
                    out[m["ticker"]] = {"edge_micros": None, "edge": None,
                                        "floor_micros": None, "bonus_micros": None,
                                        "risk_u": 1.0,
                                        "R": R, "delta": g["delta"], "vega": g["vega"],
                                        "theta": g["theta"], "scale_pos": 1.0, "quotable": False,
                                        "source": "curve", "reason": "R_above_calibrated_max"}
                continue
            # THE TWO PARTS, CARRIED SEPARATELY. `edge_micros` stays the total required to POST,
            # so every existing reader is unchanged; `floor_micros` and `bonus_micros` let the
            # quote path build the entry bar and the EXIT bar out of different pieces -- the exit
            # bar is the floor plus the lean and never the bonus. See `pricing/edge`.
            # THE ADVERSE-SELECTION GATE. `move_scale` is the futures move we expect over the
            # window between a tick moving and our cancel landing -- measured 2026-08-26 at ~85ms
            # (74ms feed + 5ms guard wake + 0.5ms reprice + 5.6ms order round trip), of which
            # about 80ms is vendor and network and none of it is ours to shorten.
            #
            # A resting quote is a free option over that window: anyone who sees the futures
            # before we do can lift us for `|delta| * move` and we cannot pull in time. If that
            # exceeds the edge we charged, the quote is mispriced by construction and no amount of
            # engineering at this stack fixes it -- the only move is not to write the option.
            #
            # MEASURED, on the session this comes from: the worst decile of fills was 78 trades
            # that cost -$341.80, all at-the-money with dollar deltas of 0.086-0.17, hit by
            # 0.75-2.75 point ES moves. `delta * dSpot` explained 79.6% of our own theo's move
            # there against -7.6% on every other fill -- a difference in kind, not degree. At the
            # low-delta wings the same option is worth a fraction of a cent and is fine to write.
            if move_scale is not None and g["delta"] is not None and em is not None:
                adverse = abs(float(g["delta"])) * float(move_scale)
                if adverse > em / 1e6:
                    out[m["ticker"]] = {
                        "edge_micros": None, "edge": None,
                        "floor_micros": None, "bonus_micros": None,
                        "risk_u": self.curve.risk_u(R), "R": R,
                        "delta": g["delta"], "vega": g["vega"], "theta": g["theta"],
                        "w": g["w"], "var_left": g["var_left"],
                        "scale_pos": self.curve.scale_position(R),
                        "quotable": False, "source": "curve",
                        "reason": "adverse_selection",
                        "adverse_dollars": adverse, "move_scale": float(move_scale)}
                    continue
            bonus = self.curve.bonus_dollars(R)
            out[m["ticker"]] = {
                "edge_micros": em, "edge": em / 1e6,
                "floor_micros": int(round(pedge.EDGE_FLOOR * 1e6)),
                "bonus_micros": (0 if bonus is None or bonus is pedge.MISSING
                                 else int(round(bonus * 1e6))),
                "risk_u": self.curve.risk_u(R),
                "scale_pos": self.curve.scale_position(R), "R": R,
                "delta": g["delta"], "vega": g["vega"], "theta": g["theta"],
                "w": g["w"], "var_left": g["var_left"],
                "quotable": True, "source": "curve", "reason": None}
        self._cache, self._cache_t, self.last_computed = out, t, now
        return out

    def get(self, ticker):
        return self._cache.get(ticker) or self._no_r("unknown_ticker")

    def summary(self):
        """One line's worth of what the manager is currently charging."""
        vals = [v for v in self._cache.values() if v.get("R") is not None]
        if not vals:
            return {"n": 0, "source": "flat", "reason": self.reason}
        es = sorted(v["edge"] for v in vals if v.get("edge") is not None)
        refused = sum(1 for v in self._cache.values() if not v.get("quotable"))
        return {"n": len(self._cache), "n_priced": len(es), "refused": refused,
                "edge_min": es[0] if es else None,
                "edge_med": es[len(es) // 2] if es else None,
                "edge_max": es[-1] if es else None,
                "enabled": self.enabled}
