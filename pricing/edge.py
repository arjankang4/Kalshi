"""How much edge a market has to offer before we are willing to quote it.

The old rule was one constant: `QUOTE_OFFSET = 0.01`, on every market, all day. Measured on a real
SPX board, an at-the-money rung's delta runs $0.0115 per index point at 10:00 and $0.1149 at
15:58 -- a tenfold change in what a one-point move costs. One offset cannot be right for both ends
of that.

THE RISK NUMBER

    R = sqrt(delta^2 + |vega|^2 + |theta|^2)         dollars per contract

Raw dollars, deliberately NOT normalised to [0,1] per axis. The magnitudes are the signal: vega
dominates the wings in the morning, delta dominates the money into the close, and theta only
matters in the last half hour. Normalising each axis would erase exactly that rotation and give a
market with tiny absolute risk the same score as one with large risk, purely because it was
extreme relative to its own axis.

THE CURVE -- two segments, continuous at the join

    R <= R_min          ->  FLOOR                   a production R below the calibration minimum
                                                    is overridden UP to the floor
    R_min < R <= R_p99  ->  FLOOR -> KNEE, exponential in u = (R - R_min) / (R_p99 - R_min)
    R_p99 < R <= R_max  ->  KNEE -> CAP, linear in    v = (R - R_p99) / (R_max - R_p99)
    R > R_max           ->  DO NOT QUOTE

Both branches give exactly KNEE at R_p99. Exponential through the bulk because the R distribution is
heavily right-skewed -- most of a board is wings that genuinely deserve the floor -- and a linear
ramp there would charge real edge for markets carrying almost none. Linear across the top 1%
because that tail is short and a second exponential would put the whole of it at the cap.

Above R_max we refuse rather than clamp. A production R beyond anything a year of history produced
is not a market to charge more for; it is a market we do not understand.
"""
import math

import config

MISSING = None

# Base edge charged on every quote, and the exit bar (cancel once edge <= this). Illustrative value;
# the live value is not published.
EDGE_FLOOR = 0.01

# Extra edge at R_p99. Illustrative value; the live value is not published.
BONUS_KNEE = 0.02
# Extra edge at R_max. Illustrative value; the live value is not published.
BONUS_CAP = 0.08

# Curvature of the exponential leg between the knee and the cap. Illustrative value; the live value
# is not published.
BONUS_BASE = 4.0

# Kept so `scale_position` and every caller that still speaks in totals keep working. The total
# charge to POST is EDGE_FLOOR + bonus: EDGE_FLOOR at R_min, EDGE_CAP at R_max.
EDGE_KNEE = EDGE_FLOOR + BONUS_KNEE     # at R_p99
EDGE_CAP = EDGE_FLOOR + BONUS_CAP       # at R_max

MICROS = 1_000_000


class Curve:
    """The calibrated R -> edge map for one underlying."""

    __slots__ = ("underlying", "R_min", "R_p99", "R_max", "built_utc", "n_rows",
                 "n_sessions", "_pct_r", "_pct_p")

    def __init__(self, art):
        self.underlying = art["underlying"]
        self.R_min = float(art["R_min"])
        self.R_p99 = float(art["R_p99"])
        self.R_max = float(art["R_max"])
        self.built_utc = art.get("built_utc")
        self.n_rows = art.get("n_rows")
        self.n_sessions = art.get("n_sessions")
        if not (self.R_min < self.R_p99 < self.R_max):
            raise ValueError(f"degenerate anchors: {self.R_min} {self.R_p99} {self.R_max}")
        pct = art.get("percentiles") or {}
        pairs = sorted((float(v), float(k)) for k, v in pct.items())
        self._pct_r = [r for r, _ in pairs]
        self._pct_p = [p / 100.0 for _, p in pairs]

    @classmethod
    def load(cls, underlying):
        from edgecal import fit as efit
        return cls(efit.load(underlying))

    def bonus_dollars(self, R):
        """The RISK part of the charge, in dollars. 0 at R_min, BONUS_KNEE at R_p99, BONUS_CAP at
        R_max. None means DO NOT QUOTE, same as before.

        Required to POST and never to STAY: a resting order is judged against the floor plus the
        lean, because the bonus paid for taking the position on, not for keeping it.
        """
        if R is None or not math.isfinite(R):
            return MISSING
        if R > self.R_max:
            return MISSING
        if R <= self.R_min:
            return 0.0
        if R <= self.R_p99:
            u = (R - self.R_min) / (self.R_p99 - self.R_min)
            return BONUS_KNEE * ((BONUS_BASE ** u) - 1.0) / (BONUS_BASE - 1.0)
        v = (R - self.R_p99) / (self.R_max - self.R_p99)
        return BONUS_KNEE + (BONUS_CAP - BONUS_KNEE) * v

    def edge_dollars(self, R):
        """Total required edge to POST, in dollars: the floor plus the bonus. None = do not quote.

        The lean is NOT in here -- it is applied per SIDE by `pricing/lean.edge_adjust`, and a
        curve has no side.
        """
        b = self.bonus_dollars(R)
        if b is MISSING:
            return MISSING
        return EDGE_FLOOR + b

    def edge_micros(self, R):
        """The same, in integer micro-dollars -- the unit every threshold comparison uses.

        Float threshold comparisons have already cost this repo once: theo 0.115 against a 0.11
        quote is 0.0050000000000000044, so `edge > 0.005` was True on noise and pennied a market
        it should have joined.
        """
        e = self.edge_dollars(R)
        return MISSING if e is MISSING else int(round(e * MICROS))

    def scale_position(self, R):
        """Where R sits on the EDGE curve, in [0,1] -- exponential then linear, same as the edge.

        `(edge(R) - floor) / (cap - floor)`. This is what the tiered rungs step back on, so the
        back rungs move on exactly the scale the edge does rather than on a separate percentile
        rank. Because the bulk of the curve is exponential, most markets sit near 0 here: at the
        median R the position is ~0.04, at R_p99 ~0.26, and only the extreme tail approaches 1.
        That compression is inherited from the edge curve on purpose -- a market that barely earns
        more edge should not suddenly rest much further back.

        A refused R (above R_max) returns 1.0, the most conservative answer, though the caller
        should not be laddering a market it will not quote.
        """
        e = self.edge_dollars(R)
        if e is None:
            return 1.0
        span = EDGE_CAP - EDGE_FLOOR
        return 0.0 if span <= 0 else max(0.0, min(1.0, (e - EDGE_FLOOR) / span))

    def risk_u(self, R):
        """R's percentile rank in the calibration distribution, in [0,1].

        This is what sets how far behind the tiered rungs sit -- "the same scaled percentile
        logic". Interpolated across the stored percentile grid rather than recomputed, so the live
        path never touches the sweep.
        """
        if R is None or not math.isfinite(R) or not self._pct_r:
            return 0.0
        if R <= self._pct_r[0]:
            return 0.0
        if R >= self._pct_r[-1]:
            return 1.0
        lo = 0
        for i in range(1, len(self._pct_r)):
            if self._pct_r[i] >= R:
                lo = i - 1
                break
        r0, r1 = self._pct_r[lo], self._pct_r[lo + 1]
        p0, p1 = self._pct_p[lo], self._pct_p[lo + 1]
        if r1 <= r0:
            return p0
        return p0 + (p1 - p0) * (R - r0) / (r1 - r0)

    def __repr__(self):
        return (f"<Curve {self.underlying} R_min={self.R_min:.5f} R_p99={self.R_p99:.5f} "
                f"R_max={self.R_max:.5f} from {self.n_sessions} sessions>")


class FlatCurve:
    """A curve that charges `config.QUOTE_OFFSET` on everything, always quotable.

    Exists so the tiered quoter can be proven to reproduce the old one bit for bit before any
    calibrated number touches it -- the same role `ratio == 1 at 16:00` plays for the hourly
    model. (It used to also cite `BLEND_W_SPREAD = 1.0`; that constant and the two-estimator
    blend it weighted were removed with the skew model.)
    """

    underlying = "flat"
    R_min = R_p99 = R_max = float("inf")
    built_utc = None
    n_rows = n_sessions = 0

    def edge_dollars(self, R):
        return config.QUOTE_OFFSET

    def edge_micros(self, R):
        return int(round(config.QUOTE_OFFSET * MICROS))

    def risk_u(self, R):
        return 0.0

    def scale_position(self, R):
        return 0.0

    def __repr__(self):
        return f"<FlatCurve {config.QUOTE_OFFSET}>"


def load(underlying):
    return Curve.load(underlying)
