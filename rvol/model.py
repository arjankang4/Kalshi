"""Serving side of the realized-vol model. Mirrors pricing/varcal's load/serve split.

The live path never re-derives anything: `fit` writes coefficients to
`var/model/rvol/{SPX,NDX}.json`, this loads them and evaluates.
"""
import math

import numpy as np

from rvol import fit as rfit


class RealizedVolModel:
    """Predicted forward realized vol, as annualized sigma on the intraday variance clock."""

    def __init__(self, art):
        self.underlying = art["underlying"]
        self.built_utc = art.get("built_utc")
        self.spec_version = art.get("spec_version")
        self.features = art["features"]
        self.extra = art["extra"]
        self.models = art["models"]
        # The 3m / 1m near-term models. A SEPARATE key, read only by `predict_short`: nothing in
        # `predict_for_minutes`, `horizons()` or the C++ loader iterates it, so the lean's blend is
        # unchanged whether or not an artifact carries them. {} for artifacts that do not.
        self.short_models = art.get("short_models") or {}
        self.n_sessions = art.get("n_sessions")

    @classmethod
    def load(cls, underlying):
        return cls(rfit.load(underlying))

    def horizons(self):
        return sorted(self.models, key=lambda h: float(h.rstrip("m")))

    def predict(self, horizon, feats):
        """feats: {'f_3h': sigma, ..., 'pf_3h': frac} -> predicted annualized sigma, or None.

        Returns the conditional **MEAN**, not the median: `exp(beta'x + s^2/2)`.

        The HAR is fitted on log sigma and is unbiased there, but `exp` is convex, so `exp(beta'x)`
        alone is the conditional median and sits below `E[sigma | x]`. Measured on the held-out set
        before the correction, `mean(pred)/mean(actual)` was 0.971-0.984 against an `exp(s^2/2)` of
        1.035 -- the gap and the correction agree, which is what distinguishes this from a regime
        shift. The lean compares this number directly against implied vol, so a systematic 3.5%
        understatement is a permanent tilt toward leaning the same way every time.

        `sigma_resid` is absent from artifacts built before SPEC_VERSION gained it; those degrade
        to the uncorrected median rather than failing, and `self.retransform` says which happened.

        Returns None rather than a number when any feature is missing or non-positive. A log-HAR
        has no defensible answer without its inputs, and substituting a mean would hand the lean a
        confident-looking prediction built from nothing.
        """
        m = self.models.get(horizon)
        if m is None:
            return None
        row = []
        for c in self.features:
            v = feats.get(c)
            if v is None or not np.isfinite(v) or v <= 0:
                return None
            row.append(math.log(v))
        for c in self.extra:
            v = feats.get(c, 0.0)
            if v is None or not np.isfinite(v):
                return None
            row.append(float(v))
        row.append(1.0)
        y = float(np.dot(np.asarray(row), np.asarray(m["beta"])))
        if not np.isfinite(y):
            return None
        s = m.get("sigma_resid")
        if s is None or not np.isfinite(s) or s <= 0:
            return math.exp(y)                       # uncorrected median; see retransform
        return math.exp(y + 0.5 * s * s)

    def short_horizons(self):
        return sorted(self.short_models, key=lambda h: float(h.rstrip("m")))

    def predict_short(self, horizon, feats):
        """Near-term (3m / 1m) forecast, annualized sigma on the 30s level, or None.

        Same contract as `predict` -- conditional mean via exp(s^2/2), None on any missing or
        non-positive input -- but reads the SHORT model's own `features`/`extra`, which may include
        `f_1m`. The 10s-target Epps rescale is already folded into the intercept at fit time.
        Deliberately never consulted by `predict_for_minutes`.
        """
        m = self.short_models.get(horizon)
        if m is None:
            return None
        row = []
        for c in m.get("features", self.features):
            v = feats.get(c)
            if v is None or not np.isfinite(v) or v <= 0:
                return None
            row.append(math.log(v))
        for c in m.get("extra", self.extra):
            v = feats.get(c, 0.0)
            if v is None or not np.isfinite(v):
                return None
            row.append(float(v))
        row.append(1.0)
        if len(row) != len(m["beta"]):
            return None
        y = float(np.dot(np.asarray(row), np.asarray(m["beta"])))
        if not np.isfinite(y):
            return None
        s = m.get("sigma_resid")
        if s is None or not np.isfinite(s) or s <= 0:
            return math.exp(y)
        return math.exp(y + 0.5 * s * s)

    def short_features(self):
        """Every feature any short model reads beyond the long models' set (e.g. 'f_1m')."""
        extra = set()
        for m in self.short_models.values():
            extra |= set(m.get("features", [])) - set(self.features)
        return sorted(extra)

    def predict_for_minutes(self, minutes, feats, blend=True):
        """Prediction for an arbitrary horizon. Returns (sigma, label).

        BLENDED between the two bracketing horizons by default since 2026-08-25: a 40-minute
        market gets (60-40)/(60-30) = 2/3 of the 30m model and 1/3 of the 60m. Outside the fitted
        range -- under 7.5m or over 60m -- it clamps to the nearest end rather than extrapolating.

        This reverses an earlier deliberate choice, so the earlier argument is worth keeping:
        "the three models are separate fits on overlapping samples; interpolating their outputs
        would imply a smoothness between them that was never estimated." That objection is right
        about interpolating *parameters* and weaker about averaging *outputs* -- a weighted mean
        of two forecasts is an ensemble, not a claim that a model was fitted at 40 minutes. What it
        does buy is continuity: under `blend=False` a market drifting from 44 to 46 minutes to
        settlement jumps horizon, its sigma steps, and -- now that the lean moves required edge --
        that step would requote the market for no reason but a labelling boundary.

        `blend=False` restores the nearest-horizon behaviour and is what the accuracy studies use,
        since those score each fitted horizon on its own sample.
        """
        if minutes is None or minutes <= 0:
            return None, None
        # float, not int: the short horizon is "7.5m" since 2026-08-22 and int("7.5") raises.
        # Every horizon label is parsed this way -- see backtest/rvin for the other site.
        pairs = sorted((float(h.rstrip("m")), h) for h in self.models)
        if not pairs:
            return None, None
        if not blend or len(pairs) == 1:
            _, h = min((abs(v - minutes), h) for v, h in pairs)
            return self.predict(h, feats), h

        if minutes <= pairs[0][0]:
            return self.predict(pairs[0][1], feats), pairs[0][1]
        if minutes >= pairs[-1][0]:
            return self.predict(pairs[-1][1], feats), pairs[-1][1]
        for (v0, h0), (v1, h1) in zip(pairs, pairs[1:]):
            if v0 <= minutes <= v1:
                lo, hi = self.predict(h0, feats), self.predict(h1, feats)
                if lo is None or hi is None:
                    return (lo if hi is None else hi), (h0 if hi is None else h1)
                wt = 0.0 if v1 == v0 else (minutes - v0) / (v1 - v0)
                return (1.0 - wt) * lo + wt * hi, f"{h0}/{h1}@{wt:.2f}"
        return None, None

    def retransform(self, horizon):
        """(applied?, factor) for the median->mean correction at this horizon.

        Diagnostic, so a stale artifact serving uncorrected medians is visible rather than silent.
        """
        m = self.models.get(horizon) or {}
        s = m.get("sigma_resid")
        if s is None or not np.isfinite(s) or s <= 0:
            return False, 1.0
        return True, float(math.exp(0.5 * s * s))

    def metrics(self, horizon):
        m = self.models.get(horizon) or {}
        return {k: m.get(k) for k in ("oos_r2", "oos_r2_rw", "oos_r2_cal",
                                      "qlike", "qlike_rw", "qlike_cal",
                                      "sessions_test", "n_test")}

    def beats_baselines(self, horizon):
        m = self.models.get(horizon)
        if not m:
            return False
        return (m["oos_r2"] > max(m["oos_r2_rw"], m["oos_r2_cal"])
                and m["qlike"] < min(m["qlike_rw"], m["qlike_cal"]))


def load(underlying):
    return RealizedVolModel.load(underlying)
