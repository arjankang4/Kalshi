"""Log-HAR fits: next 60m / 30m / 7.5m realized vol (`models`), plus 3m / 1m (`short_models`).

The three long horizons are what the lean blends (`RealizedVolModel.predict_for_minutes`). The two
short ones are a near-term risk signal for `pricing/volspike`, stored under a SEPARATE artifact key
so nothing that reads `models` -- the lean, `backtest/rvin`, `edgecal/greeks_bt`, the C++ loader --
can pick them up. They are fitted only from a cache built with `rvol.dataset --short`.

    log y = b0 + b1 log f_3h + b2 log f_1h + b3 log f_30m + b4 log f_15m + b5 log f_5m
               + b6 proxy_frac_3h

Fitted by `numpy.linalg.lstsq`. No statsmodels, no scipy -- this repo does its statistics by hand
on purpose, and a 7-parameter OLS is a normal-equations solve.

OVERLAPPING SAMPLES, STATED PLAINLY. Observations five minutes apart share up to 55 of the 60
minutes of a 1-hour lookback and most of their target window. Residuals are therefore strongly
autocorrelated, and the naive OLS standard errors are understated by a large and unknown factor.

The response is NOT a cleverer standard error. It is to **do no inference at all** -- no p-values,
no t-statistics, no "significant" anywhere in this module -- and to judge the model only on a
held-out period it never saw. The train/test split is strictly temporal (first 75% of sessions vs
last 25%), never random: a random split would put an observation's near-duplicate neighbour on the
other side of the split and report a wildly optimistic score.

TWO BASELINES, BOTH OF WHICH MUST BE BEATEN for the lean to be worth building:

  * **random walk** -- next period's vol equals the last hour's vol. If the multi-horizon
    structure cannot beat "look at the last hour", it is not earning its place.
  * **slow trailing** -- next period's vol equals the 3h trailing estimate. This is the slower,
    more calendar-like reference; it is NOT the calendar's own forecast.

    **Corrected 2026-08-22.** This baseline used to be called "calendar" and described as
    "the calendar's own expectation, which in these units is exactly 1.0x". Both halves were
    wrong, and the metric keys `oos_r2_cal` / `qlike_cal` still carry the misleading name.
    (a) The units are NOT a ratio. `features.window_sigma` returns `annualized(total_rv / vt)`
        -- an ANNUALIZED SIGMA. Measured on the held-out set, `mean log y` is -2.48 for SPX and
        -2.07 for NDX, i.e. ~8.4% and ~12.7% annualized. There is no 1.0x anywhere in these
        units; scoring a constant 1.0 forecast gives R^2 = -28.9 (SPX) because it predicts 100%
        vol against an 8% truth.
    (b) More fundamentally, **the calendar cannot make this forecast at all.** It supplies the
        SHAPE of variance within a session, not the LEVEL. Any level has to come from a trailing
        realized estimate, so a "calendar-only" baseline does not exist in this parameterisation.
        `log(f_3h)` is a trailing-RV baseline and nothing more.
    So "beats the calendar baseline" was never tested and should not be claimed. What IS
    supported -- and does hold -- is that the model beats both TRAILING baselines: measured
    2026-08-22 on the held-out set, SPX oosR2 0.660/0.667/0.639 (15m/30m/60m) against 0.578/
    0.594/0.563 for the 1h random walk and 0.217/0.233/0.195 for the 3h trailing; NDX
    0.658/0.661/0.627 against 0.579/0.590/0.554 and 0.196/0.197/0.140.

Reported on the held-out set: out-of-sample R^2 on log vol, and QLIKE on the variance scale.
QLIKE (v/f - log(v/f) - 1) is reported because it is the loss that penalises UNDER-forecasting
variance asymmetrically, which is the error that costs money when quoting.

    python -m rvol.fit --fit --underlying SPX
    python -m rvol.fit --report
"""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

import config
from rvol import dataset as D

MODEL_DIR = config.MODEL_DIR / "rvol"

FEATURES = [f"f_{n}" for n, _ in D.LOOKBACKS]
EXTRA = ["pf_3h"]
SPEC_VERSION = 1

TRAIN_FRACTION = 0.75          # by SESSION, not by row

# SHORT-MODEL SPECS. `names` is stored per model, so a short model may read a different feature set
# from the long ones; `RealizedVolModel.predict_short` reads each model's own `features`/`extra`.
#   base   the long models' exact design (5 lookbacks + pf_3h)
#   +1m    the same plus a 1-minute trailing lookback at 10s
# Which one ships per horizon is decided on the held-out set in `fit_short_all` and recorded there.
SHORT_VARIANTS = {
    "base": (FEATURES, EXTRA),
    "+1m": (FEATURES + ["f_1m"], EXTRA),
}


def design(df):
    """(X, names) -- log features plus the proxy fraction plus an intercept."""
    cols, names = [], []
    for c in FEATURES:
        cols.append(np.log(df[c].to_numpy(dtype="float64")))
        names.append(f"log_{c}")
    for c in EXTRA:
        cols.append(df[c].to_numpy(dtype="float64"))
        names.append(c)
    X = np.column_stack(cols + [np.ones(len(df))])
    return X, names + ["const"]


def _finite_mask(X, y):
    return np.isfinite(X).all(axis=1) & np.isfinite(y)


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def qlike(y_log, yhat_log):
    """QLIKE on the VARIANCE scale from log-sigma inputs.

    v = exp(2 y), f = exp(2 yhat); loss = mean(v/f - log(v/f) - 1), minimised at 0 when f == v.
    Asymmetric: under-forecasting variance is penalised harder than over-forecasting, which is the
    right asymmetry for a quoter -- being short vol you did not price is what costs money.
    """
    ratio = np.exp(2.0 * (y_log - yhat_log))
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def exclusion_counts(df, ycol, features, extra):
    """How many rows a fit drops, and why -- counted, never silent.

    The drop itself is unchanged (it is what the production `models` were fitted with). What used
    to be invisible is its size: a flat window returns NaN from `features.window_sigma`, and
    dropping those rows biases the target UP if they are common. Measured 2026-09-29 the in-session
    loss is 0.57% of the grid, all of it five early-close sessions plus the truncated 2024-08-22
    bar file; this makes the next change to that visible.
    """
    y = df[ycol].to_numpy(dtype="float64")
    y_ok = np.isfinite(y) & (y > 0)
    f_ok = np.ones(len(df), dtype=bool)
    for c in features:
        v = df[c].to_numpy(dtype="float64")
        f_ok &= np.isfinite(v) & (v > 0)
    for c in extra:
        f_ok &= np.isfinite(df[c].to_numpy(dtype="float64"))
    out = {"n_rows": int(len(df)), "target_missing": int((~y_ok).sum()),
           "feature_missing": int((y_ok & ~f_ok).sum()), "used": int((y_ok & f_ok).sum())}
    mins = dict(D.HORIZONS + D.SHORT_HORIZONS).get(ycol.split("_", 1)[1])
    if mins is not None and "obs_minute" in df.columns:
        # Past 16:00 is NaN BY DESIGN (never truncated -- see rvol/dataset); the remainder is
        # the in-session loss, which is the number that could bias the target.
        past = int((df["obs_minute"].to_numpy() + mins > D.SESSION_MIN).sum())
        out["target_past_close"] = past
        out["target_missing_in_session"] = out["target_missing"] - past
    flag = f"yflag_{ycol.split('_', 1)[1]}"
    if flag in df.columns:
        out["target_by_reason"] = {str(k): int(v) for k, v in
                                   df.loc[~y_ok, flag].value_counts().items()}
    return out


def fit_one(df, horizon, train_frac=TRAIN_FRACTION, exclusions=None):
    """Fit one horizon. Returns a dict of coefficients and held-out metrics.

    `exclusions`, if a dict, receives `exclusion_counts` for this horizon. It is an out-parameter
    rather than a key on the return value so the fitted `models` entry stays byte-identical.
    """
    ycol = f"y_{horizon}"
    if exclusions is not None:
        exclusions.update(exclusion_counts(df, ycol, FEATURES, EXTRA))
    d = df.dropna(subset=FEATURES + EXTRA + [ycol]).copy()
    d = d[(d[ycol] > 0)]
    for c in FEATURES:
        d = d[d[c] > 0]

    sessions = sorted(d["dt"].unique())
    if len(sessions) < 40:
        raise RuntimeError(f"only {len(sessions)} sessions with usable {ycol}")
    cut = sessions[int(len(sessions) * train_frac)]
    tr, te = d[d["dt"] < cut], d[d["dt"] >= cut]

    Xtr, names = design(tr)
    ytr = np.log(tr[ycol].to_numpy(dtype="float64"))
    m = _finite_mask(Xtr, ytr)
    Xtr, ytr = Xtr[m], ytr[m]
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    # Residual scale of the TRAINING fit, in log space, with the regression's dof removed.
    #
    # This is what turns the served prediction from a median into a mean. The fit is unbiased in
    # logs, but `exp` is convex, so `exp(beta'x)` is the conditional MEDIAN of sigma and sits
    # systematically below E[sigma | x] by exp(s^2/2) -- the retransformation (Jensen) gap.
    # Measured on the held-out set before this was applied: mean(pred)/mean(actual) ran 0.971-0.984
    # across both underlyings and all three horizons, against an exp(s^2/2) of 1.035. The gap is
    # almost exactly the correction, which is what says it is this and not a regime shift.
    #
    # It matters here because the lean compares sigma_rv DIRECTLY against implied vol. A 3.5%
    # systematic understatement is a 3.5% permanent tilt toward "realized is below implied", i.e.
    # toward always leaning the same way, which is the one error a lean cannot survive.
    resid = ytr - Xtr @ beta
    dof = max(1, len(ytr) - Xtr.shape[1])
    sigma_resid = float(np.sqrt(float((resid ** 2).sum()) / dof))

    Xte, _ = design(te)
    yte = np.log(te[ycol].to_numpy(dtype="float64"))
    m = _finite_mask(Xte, yte)
    Xte, yte = Xte[m], yte[m]
    pred = Xte @ beta

    # baseline 1: random walk on the 1h lookback
    rw = np.log(te["f_1h"].to_numpy(dtype="float64"))[m]
    # baseline 2: the SLOW TRAILING reference -- the 3h lookback. Named `cal` and reported as
    # `oos_r2_cal`/`qlike_cal` for artifact compatibility, but it is NOT the calendar: the
    # calendar gives intraday shape, not a vol level, so it makes no forecast in these
    # annualized-sigma units. See the module docstring.
    cal = np.log(te["f_3h"].to_numpy(dtype="float64"))[m]

    return {
        "horizon": horizon,
        "spec_version": SPEC_VERSION,
        "names": names,
        "beta": [float(b) for b in beta],
        "sigma_resid": sigma_resid,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "sessions_train": int(len({*tr["dt"]})), "sessions_test": int(len({*te["dt"]})),
        "train_end": str(cut),
        "oos_r2": r2(yte, pred),
        "oos_r2_rw": r2(yte, rw),
        "oos_r2_cal": r2(yte, cal),
        "qlike": qlike(yte, pred),
        "qlike_rw": qlike(yte, rw),
        "qlike_cal": qlike(yte, cal),
        "in_sample_r2": r2(ytr, Xtr @ beta),
    }


def _design_cols(df, features, extra):
    cols = [np.log(df[c].to_numpy(dtype="float64")) for c in features]
    cols += [df[c].to_numpy(dtype="float64") for c in extra]
    X = np.column_stack(cols + [np.ones(len(df))])
    return X, [f"log_{c}" for c in features] + list(extra) + ["const"]


def fit_short(df, horizon, variant="base", target="y", train_frac=TRAIN_FRACTION, cut=None):
    """Fit one SHORT horizon (3m / 1m) under one feature variant. Dict like `fit_one`, plus:

    `target_scale_s` / `epps_rescale`. The 10s target is Epps-biased ~6% LOW against the 30s scale
    every other target uses (see rvol/dataset). `epps_rescale` is sqrt(mean y_10s^2 / mean
    y_30s^2) on the SAME windows over the TRAINING sessions only, and the fitted target is
    y_10s / epps_rescale -- so the served forecast is on the 30s level. It is folded into the
    intercept; nothing at serve time applies it.

    `target="yref"` fits the 30s target directly (the comparison arm; `epps_rescale` is 1).

    `*_ref` metrics score every arm against the SAME truth -- log of the 30s target on the held-out
    rows -- which is the only way to compare a 10s-fitted model with a 30s-fitted one.
    """
    features, extra = SHORT_VARIANTS[variant]
    ycol, rcol = f"{target}_{horizon}", f"yref_{horizon}"
    excl = exclusion_counts(df, ycol, features, extra)
    d = df.dropna(subset=features + extra + [ycol, rcol]).copy()
    d = d[(d[ycol] > 0) & (d[rcol] > 0)]
    for c in features:
        d = d[d[c] > 0]
    sessions = sorted(d["dt"].unique())
    if len(sessions) < 40:
        raise RuntimeError(f"only {len(sessions)} sessions with usable {ycol}")
    cut = cut or sessions[int(len(sessions) * train_frac)]
    tr, te = d[d["dt"] < cut], d[d["dt"] >= cut]

    y_tr = tr[ycol].to_numpy(dtype="float64")
    r_tr = tr[rcol].to_numpy(dtype="float64")
    rescale = 1.0 if target == "yref" else float(np.sqrt(np.mean(y_tr ** 2) / np.mean(r_tr ** 2)))

    Xtr, names = _design_cols(tr, features, extra)
    ytr = np.log(y_tr / rescale)
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    resid = ytr - Xtr @ beta
    dof = max(1, len(ytr) - Xtr.shape[1])
    sigma_resid = float(np.sqrt(float((resid ** 2).sum()) / dof))

    Xte, _ = _design_cols(te, features, extra)
    yte = np.log(te[ycol].to_numpy(dtype="float64") / rescale)
    ref = np.log(te[rcol].to_numpy(dtype="float64"))
    pred = Xte @ beta
    rw = np.log(te["f_1h"].to_numpy(dtype="float64"))
    cal = np.log(te["f_3h"].to_numpy(dtype="float64"))
    rw5 = np.log(te["f_5m"].to_numpy(dtype="float64"))
    return {
        "horizon": horizon, "spec_version": SPEC_VERSION, "variant": variant,
        "features": list(features), "extra": list(extra),
        "names": names, "beta": [float(b) for b in beta], "sigma_resid": sigma_resid,
        "target": target, "target_scale_s": (D.SHORT_REFERENCE_SCALE_S if target == "yref"
                                             else D.HORIZON_SCALE_S[horizon]),
        "epps_rescale": rescale,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "sessions_train": int(tr["dt"].nunique()), "sessions_test": int(te["dt"].nunique()),
        "train_end": str(cut),
        "oos_r2": r2(yte, pred), "oos_r2_rw": r2(yte, rw), "oos_r2_cal": r2(yte, cal),
        "oos_r2_rw5m": r2(yte, rw5),
        "qlike": qlike(yte, pred), "qlike_rw": qlike(yte, rw), "qlike_cal": qlike(yte, cal),
        "qlike_rw5m": qlike(yte, rw5),
        # QLIKE of the SERVED forecast, exp(beta'x + s^2/2) -- what `predict_short` returns. The
        # `qlike*` keys above score the log-median `beta'x`, as `fit_one` does for the long
        # models; on a target this noisy the median sits well below E[v], QLIKE punishes
        # under-forecasting, and the median-vs-baseline comparison is tilted toward whichever
        # baseline happens to sit high. Both are reported; neither is tuned on.
        "qlike_served": qlike(yte, pred + 0.5 * sigma_resid ** 2),
        "qlike_ref_served": qlike(ref, pred + 0.5 * sigma_resid ** 2),
        # Squared correlation with the 30s truth: information content net of any LEVEL
        # difference, which is what separates the 10s and 30s arms (see SHORT_CHOICE).
        "corr2_ref": float(np.corrcoef(ref, pred)[0, 1] ** 2),
        "oos_r2_ref": r2(ref, pred), "qlike_ref": qlike(ref, pred),
        "qlike_ref_rw": qlike(ref, rw), "qlike_ref_cal": qlike(ref, cal),
        "in_sample_r2": r2(ytr, Xtr @ beta),
        "mean_ratio_test": float(np.mean(np.exp(pred + 0.5 * sigma_resid ** 2))
                                 / np.mean(np.exp(yte))),
        "exclusions": excl,
    }


# The spec that ships per short horizon, chosen on the held-out set 2026-09-29 (see
# `fit_short_all`'s printed table and the report in the artifact's `short_selection`). Four arms
# were tried per horizon and underlying: {base, +1m} x {10s rescaled, 30s}. All scored against the
# same held-out truth, the 30s target:
#
#   10s vs 30s TARGET. corr^2 with the truth is identical to the third decimal on all four
#   (horizon, underlying) cells -- the two arms carry the same information and differ only in
#   LEVEL. The 30s arm wins log-R^2 (1m: 0.330 vs 0.272 SPX) because log of a 1-2-return target
#   has a lower mean than its variance implies, and fitting it in logs reproduces that Jensen
#   shift; the 10s arm, rescaled on variance, wins served QLIKE on every cell (1m SPX 0.620 vs
#   0.668, NDX 0.621 vs 0.666; 3m SPX 0.290 vs 0.297, NDX 0.267 vs 0.271). The signal divides by
#   sigma_30m, a variance-level quantity, so 10s ships.
#   +1m LOOKBACK. corr^2 +0.001..0.003 and served QLIKE -0.001..-0.004 on every cell: consistent
#   but negligible, and it adds a 6-return feature to the live path. Not shipped.
SHORT_CHOICE = {"3m": ("base", "y"), "1m": ("base", "y")}


def fit_short_all(df, train_frac=TRAIN_FRACTION, cut=None, verbose=True):
    """(short_models, selection). Fits every arm, ships SHORT_CHOICE, records all arms' metrics."""
    chosen, arms = {}, {}
    for h, _ in D.SHORT_HORIZONS:
        arms[h] = {}
        for variant in SHORT_VARIANTS:
            for target in ("y", "yref"):
                m = fit_short(df, h, variant, target, train_frac, cut)
                arms[h][f"{variant}|{target}"] = {
                    k: m[k] for k in ("oos_r2", "qlike", "oos_r2_ref", "qlike_ref",
                                      "qlike_ref_served", "corr2_ref",
                                      "sigma_resid", "epps_rescale", "sessions_test")}
                if (variant, target) == SHORT_CHOICE[h]:
                    chosen[h] = m
    return chosen, {"choice": {h: list(v) for h, v in SHORT_CHOICE.items()},
                    "n_arms_per_horizon": len(SHORT_VARIANTS) * 2, "arms": arms}


def calendar_fingerprint(underlying):
    """Identify the variance calendar these coefficients are defined against.

    Every feature and every target is divided by `features.variance_time(cal, ...)`, so the
    fitted betas mean nothing except relative to one specific calendar. Repoint
    `varcal.load(underlying)` at a refit calendar and the coefficients are silently stale --
    the model keeps predicting, `validate` keeps printing the metrics stored at fit time, and
    nothing anywhere notices. Recording the fingerprint is what makes that detectable.

    `shares_sha256` is the load-bearing field: built_utc changes on a rebuild that reproduces
    the same numbers, and overnight_share alone would miss a pure intraday reshape.
    """
    from pricing import varcal
    # Read the artifact directly: VarianceCalendar.__init__ keeps only shares/cum/
    # overnight_share/sessions/built_utc, and the provenance fields we need to compare
    # (profile, source_ticker, estimator) are dropped on load.
    art = json.loads(varcal.calendar_path(underlying).read_text())
    shares = list(art.get("shares") or [])
    blob = json.dumps([round(float(x), 12) for x in shares], separators=(",", ":"))
    return {
        "profile": art.get("profile", "default"),
        "source_ticker": art.get("source_ticker"),
        "built_utc": art.get("built_utc"),
        "estimator": art.get("estimator"),
        "overnight_share": art.get("overnight_share"),
        "n_buckets": len(shares),
        "shares_sha256": hashlib.sha256(blob.encode()).hexdigest(),
    }


def clock_fingerprint(underlying):
    """Hash the COMPLETE input surface of `features.variance_time`, in one field.

    Why this exists when `calendar_fingerprint` and `gap_calendar_fingerprint` already do: each of
    those hashes its own `shares` vector and nothing else, so neither can see a change to
    **`overnight_share`** -- and since 2026-08-22 that scalar does far more work than it used to.
    `variance_time` scales the ENTIRE overnight shape by it, so every lookback reaching before
    09:30 is proportional to it. A calendar rebuild that moved the scalar and left `shares`
    untouched would leave both existing hashes identical while changing every pre-open feature.

    That is not hypothetical. `shares` and `overnight_share` are fitted by two different estimators
    with two different exclusions in the current varcal rebuild, and the open-anchor staleness
    issue -- see the docstring in pricing/gapcal -- would move the scalar alone if it were ever
    corrected. Credit to the varcal session for spotting the gap.

    The three components are hashed together because they are consumed together: a model is valid
    against one (intraday shape, overnight level, overnight shape) triple, not against three
    independently-versioned artifacts.
    """
    from pricing import gapcal, varcal
    parts = []
    art = json.loads(varcal.calendar_path(underlying).read_text())
    parts.append(json.dumps([round(float(x), 12) for x in (art.get("shares") or [])],
                            separators=(",", ":")))
    parts.append(repr(round(float(art.get("overnight_share") or 0.0), 12)))
    gp = gapcal.gap_path(underlying)
    if gp.exists():
        g = json.loads(gp.read_text())
        parts.append(json.dumps([round(float(x), 12) for x in (g.get("shares") or [])],
                                separators=(",", ":")))
    else:
        parts.append("no-gap-calendar")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def gap_calendar_fingerprint(underlying):
    """Identify the OVERNIGHT calendar these coefficients are defined against.

    Exactly the same hazard as `calendar_fingerprint`, and if anything a sharper one. Since
    2026-08-22 `features.variance_time` apportions the pre-open portion of a window by
    `pricing/gapcal`'s measured shape instead of uniformly across 17.5 hours, so every lookback
    that reaches before 09:30 -- which is every 3h lookback before 12:30 -- is normalised by this
    artifact. Refit the gap calendar and those features change while `rvol.validate` keeps
    replaying the metrics stored at fit time and keeps printing PASS.

    Returns None rather than raising when the artifact is absent, so a model fitted before the
    gap calendar existed is still identifiable as such.
    """
    from pricing import gapcal
    path = gapcal.gap_path(underlying)
    if not path.exists():
        return None
    art = json.loads(path.read_text())
    shares = list(art.get("shares") or [])
    blob = json.dumps([round(float(x), 12) for x in shares], separators=(",", ":"))
    return {
        "profile": art.get("profile", "default"),
        "sources": art.get("sources"),
        "built_utc": art.get("built_utc"),
        "estimator": art.get("estimator"),
        "cycles": art.get("cycles"),
        "halt_capped": (art.get("halt") or {}).get("capped"),
        "n_buckets": len(shares),
        "shares_sha256": hashlib.sha256(blob.encode()).hexdigest(),
    }


def fit_underlying(underlying, train_frac=TRAIN_FRACTION, write=True, obs_dir=None,
                   short=False, out_path=None):
    """The full artifact. `short=True` adds `short_models` (needs a `--short` cache in `obs_dir`).

    Writes to `var/model/rvol/` by default -- the PRODUCTION path. A candidate goes to `out_path`.
    """
    df = D.load(underlying, obs_dir=obs_dir)
    out = {"underlying": underlying,
           "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "spec_version": SPEC_VERSION,
           "features": FEATURES, "extra": EXTRA,
           "n_rows": int(len(df)), "n_sessions": int(df["dt"].nunique()),
           "first_session": str(df["dt"].min()), "last_session": str(df["dt"].max()),
           "calendar": calendar_fingerprint(underlying),
           "gap_calendar": gap_calendar_fingerprint(underlying),
           "clock_sha256": clock_fingerprint(underlying),
           "models": {}}
    excl = {}
    for h, _ in D.HORIZONS:
        excl[h] = {}
        out["models"][h] = fit_one(df, h, train_frac, exclusions=excl[h])
    if short:
        # Same session cut as the long models, so the 126 held-out sessions are the same ones.
        cut = datetime.date.fromisoformat(out["models"][D.HORIZONS[0][0]]["train_end"])
        out["short_models"], out["short_selection"] = fit_short_all(df, train_frac, cut)
        for h, m in out["short_models"].items():
            excl[h] = m["exclusions"]
    out["exclusions"] = excl
    if write:
        p = out_path or (MODEL_DIR / f"{underlying}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print(f"[rvol.fit] wrote {p}")
    return out


def report(art):
    print(f"\n=== {art['underlying']} === {art['n_rows']:,} obs, {art['n_sessions']} sessions "
          f"({art['first_session']} .. {art['last_session']})")
    for h, m in art["models"].items():
        print(f"\n  horizon {h}: train {m['sessions_train']} sessions / {m['n_train']:,} obs, "
              f"test {m['sessions_test']} / {m['n_test']:,}  (split {m['train_end']})")
        print(f"    {'coefficient':>14} {'value':>9}")
        for n, b in zip(m["names"], m["beta"]):
            print(f"    {n:>14} {b:9.4f}")
        print(f"    {'metric':>14} {'HAR':>9} {'rw(1h)':>9} {'slow(3h)':>9}")
        print(f"    {'oos R2 (log)':>14} {m['oos_r2']:9.4f} {m['oos_r2_rw']:9.4f} {m['oos_r2_cal']:9.4f}")
        print(f"    {'QLIKE (var)':>14} {m['qlike']:9.4f} {m['qlike_rw']:9.4f} {m['qlike_cal']:9.4f}")
        beats = m["oos_r2"] > max(m["oos_r2_rw"], m["oos_r2_cal"]) and \
                m["qlike"] < min(m["qlike_rw"], m["qlike_cal"])
        print(f"    -> beats both baselines on both metrics: {beats}")
    for h, m in (art.get("short_models") or {}).items():
        print(f"\n  SHORT horizon {h} [{m['variant']}, target {m['target_scale_s']}s, "
              f"epps_rescale {m['epps_rescale']:.4f}]: train {m['sessions_train']} sessions / "
              f"{m['n_train']:,} obs, test {m['sessions_test']} / {m['n_test']:,}")
        print(f"    {'metric':>14} {'HAR':>9} {'rw(1h)':>9} {'slow(3h)':>9} {'rw(5m)':>9}")
        print(f"    {'oos R2 (log)':>14} {m['oos_r2']:9.4f} {m['oos_r2_rw']:9.4f} "
              f"{m['oos_r2_cal']:9.4f} {m['oos_r2_rw5m']:9.4f}")
        print(f"    {'QLIKE (var)':>14} {m['qlike']:9.4f} {m['qlike_rw']:9.4f} "
              f"{m['qlike_cal']:9.4f} {m['qlike_rw5m']:9.4f}")
        print(f"    {'QLIKE served':>14} {m['qlike_served']:9.4f}   (exp(beta'x + s^2/2), what predict_short returns)")
        print(f"    sigma_resid {m['sigma_resid']:.4f}  exp(s^2/2) {math.exp(0.5 * m['sigma_resid'] ** 2):.4f}"
              f"  mean(pred)/mean(actual) {m['mean_ratio_test']:.4f}")
        beats = m["oos_r2"] > max(m["oos_r2_rw"], m["oos_r2_cal"]) and \
                m["qlike"] < min(m["qlike_rw"], m["qlike_cal"])
        print(f"    -> beats both baselines on both metrics: {beats}")
    sel = art.get("short_selection")
    if sel:
        print(f"\n  short-model arms tried ({sel['n_arms_per_horizon']} per horizon), scored on the "
              f"held-out 30s truth:")
        for h, arms in sel["arms"].items():
            for k, v in arms.items():
                print(f"    {h:>3} {k:>10}  R2_ref {v['oos_r2_ref']:.4f}  corr2_ref {v['corr2_ref']:.4f}"
                      f"  QLIKE_ref {v['qlike_ref']:.4f} (served {v['qlike_ref_served']:.4f})"
                      f"  own-target R2 {v['oos_r2']:.4f}  s {v['sigma_resid']:.3f}")


def load(underlying):
    p = MODEL_DIR / f"{underlying}.json"
    if not p.exists():
        raise FileNotFoundError(f"no rvol model for {underlying} at {p} -- "
                                f"run: python -m rvol.fit --fit --underlying {underlying}")
    return json.loads(p.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit the log-HAR realized-vol models")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--underlying", nargs="*", default=["SPX", "NDX"])
    ap.add_argument("--short", action="store_true",
                    help="also fit 3m/1m into short_models; needs --obs-dir and --out-dir")
    ap.add_argument("--obs-dir", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="write {UND}.short.json here instead of the production var/model/rvol")
    a = ap.parse_args(argv)
    if a.short and a.fit and not a.out_dir:
        ap.error("--short never writes to the production var/model/rvol; give --out-dir")
    for u in a.underlying:
        if a.fit:
            od = pathlib.Path(a.obs_dir) if a.obs_dir else None
            op = (pathlib.Path(a.out_dir) / f"{u}.short.json") if a.out_dir else None
            art = fit_underlying(u, obs_dir=od, short=a.short, out_path=op)
        elif a.out_dir:
            art = json.loads((pathlib.Path(a.out_dir) / f"{u}.short.json").read_text())
        else:
            art = load(u)
        report(art)
    return 0


if __name__ == "__main__":
    sys.exit(main())
