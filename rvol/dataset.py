"""Assemble the (features, targets) matrix from cached bars. Resumable, one Parquet per session.

Mirrors `backtest/run.py`: materialise once to `var/rvol/obs/{UND}/dt=*.parquet`, then `load()`.
Nothing downstream re-derives, so a restart cannot move the numbers under you.

ONE OBSERVATION every 5 minutes of RTH, not every second. Consecutive overlapping windows are near
duplicates -- two observations 5 minutes apart share 55 of the 60 minutes of a 1-hour lookback --
so a denser grid buys autocorrelated near-copies, not information. 76 observations a session over
~500 sessions is ~38k rows per underlying, which is ample for a 6-parameter regression.

FEATURES are trailing realized vol over 3h / 1h / 30m / 15m / 5m, TARGETS are forward realized vol
over 60m / 30m / 7.5m (and, in the separate short-horizon cache, 3m / 1m -- see SHORT_HORIZONS),
all as annualized sigma on the intraday variance clock (see rvol.features).

The three long lookbacks are sampled at 60s and the two short ones at 10s; the three long targets
are sampled at 30s and the 3m/1m short ones at 10s, rescaled to the 30s level at fit time.
See LOOKBACK_SCALE_S below for why the targets deliberately do not follow the features.

A target that would run past 16:00 is NaN, never truncated. Truncating would silently make the
late-session targets measure a shorter window than the early ones, which reads as "volatility
falls into the close" -- an artefact that would then be learned.
"""
import argparse
import datetime
import sys

import numpy as np

import config
from pricing import varcal
from rvol import data as rdata
from rvol import features as F

OBS_DIR = config.ROOT / "var" / "rvol" / "obs"

LOOKBACKS = (("3h", 180), ("1h", 60), ("30m", 30), ("15m", 15), ("5m", 5))
HORIZONS = (("60m", 60), ("30m", 30), ("7.5m", 7.5))

# THE SHORT HORIZONS, kept OUT of `HORIZONS` on purpose. They are a near-term risk signal
# (`pricing/volspike`), not part of the to-settlement blend, so they are fitted into the artifact's
# separate `short_models` key and served only by `RealizedVolModel.predict_short`. Everything that
# loops `HORIZONS` -- `fit.fit_underlying`'s `models`, `rvol/accuracy`, `backtest/rvbt` -- therefore
# sees exactly the three horizons it always did. Never label one "15m": the golden-vector generator
# uses that label as its missing-horizon probe.
SHORT_HORIZONS = (("3m", 3), ("1m", 1))
# A 1-minute trailing lookback, read ONLY by a short-model variant (see fit.SHORT_VARIANTS). Kept
# out of `LOOKBACKS` so the long models' design matrix, and the live feature dict they read, are
# unchanged.
SHORT_LOOKBACKS = (("1m", 1),)

# SAMPLING SCALE PER WINDOW. Kept as separate maps rather than a third tuple element so that
# `fit.py`'s `for n, _ in LOOKBACKS` unpacking keeps working -- and because "30m" means two
# different scales depending on whether it is a lookback (60s) or a target (30s), which a single
# name-keyed map could not express.
#
# 60s for the three long lookbacks and 10s for the two short ones is the study's specification.
# All three TARGETS stay at 30s with sub-grid averaging, which is not: a feature's scale bias is
# absorbed by its fitted coefficient, a target's is not. See the SCALE_S note in rvol/features
# for the measured Epps and microstructure-noise numbers on both sides.
LOOKBACK_SCALE_S = {"3h": 60, "1h": 60, "30m": 60, "15m": 10, "5m": 10, "1m": 10}
HORIZON_SCALE_S = {"60m": 30, "30m": 30, "7.5m": 30, "3m": 10, "1m": 10}

# THE SHORT TARGETS ARE SAMPLED AT 10s, NOT 30s, and are then RESCALED to the 30s level.
# Measured 2026-09-29 over all 502 cached sessions, 77 obs a session (`rvol.validate` does not
# re-run this; the numbers are the decision record):
#
#     pooled RTH vol     1s      5s     10s     15s     30s
#     I:SPX            8.95%  10.47%  11.10%  11.42%  11.89%
#     I:NDX           10.96%  13.33%  14.18%  14.59%  15.20%
#
#     sd(log y) of the window target      1s     5s    10s    15s    30s
#     1m  SPX / NDX                     0.52/0.53 0.56/0.56 0.60/0.59 0.63/0.63 0.73/0.73
#     3m  SPX / NDX                     0.49/0.50 0.52/0.52 0.54/0.53 0.55/0.54 0.59/0.57
#
# A 1-minute window at 30s holds one or two returns per sub-grid (coverage 0.517, just over
# MIN_COVERAGE), so its log target is dominated by chi-square sampling noise. 10s cuts that noise
# by ~18% (1m) and ~9% (3m) for an Epps level bias of ~6.4% (same windows, 10s vs 30s: SPX 0.938 /
# 0.939, NDX 0.935 / 0.936 for 1m / 3m). 5s buys only ~6% more noise reduction for a bias that
# grows to ~11% on NDX, so 10s is the knee. The bias is a near-constant level factor, so it is
# removed rather than lived with: `fit.fit_short` measures sqrt(mean y_10s^2 / mean y_30s^2) on
# the TRAINING sessions (from `yref_*`, the same window at 30s) and divides it out, which keeps
# every short forecast on the same 30s level as the long models and as `sigma_30m`. On the
# per-window annualized sigmas that factor is 0.915 / 0.921 (1m SPX / NDX) and 0.925 / 0.925 (3m)
# -- larger than the pooled-RV 6.4% because it is an unweighted mean over windows.
#
# Zero/NaN targets are NOT a scale problem here: at every scale 1s..30s the NaN rate is identical,
# 222 of 38,654 windows (0.57%), all in six sessions: five early closes (the index stops at
# 13:00, the forward-fill is flat, RV is 0; 34 windows each) and 2024-08-22, whose cached bar file
# ends at 11:35 ET (52 windows). Sub-grid averaging leaves no genuinely flat window in a full
# session even at 1s. They are now COUNTED (`yflag_*`, and `exclusions` in the fitted
# artifact) rather than silently dropped.
SHORT_REFERENCE_SCALE_S = 30

OBS_STEP_MIN = 5
OBS_FIRST_MIN = 0        # 09:30
# 15:50 -- the last instant on the 5-minute grid where the SHORTEST target still fits inside the
# session (380 + 7.5 = 387.5 <= 390). Longer targets past their own limit come back NaN, which is
# why this is set by the shortest one. It moved out from 375 when the short target went 15m -> 7.5m.
OBS_LAST_MIN = 380

SESSION_MIN = 390


def obs_path(underlying, d, obs_dir=None):
    return (obs_dir or OBS_DIR) / underlying / f"dt={d}.parquet"


def _et(d, minute):
    return (datetime.datetime.combine(d, F.OPEN_T, tzinfo=config.ET)
            + datetime.timedelta(minutes=minute))


def build_session(underlying, d, cal, short=False):
    """One session -> list of observation dicts. [] when the day is unusable.

    `short=True` adds the short-horizon columns (`_short_columns`). Off by default so the
    production cache in `var/rvol/obs` is built exactly as before.
    """
    src = rdata.SOURCES[underlying]
    ts_i, px_i = rdata.load_day(src["index"], d)
    ts_p, px_p = rdata.load_day(src["proxy"], d)
    if ts_i is None or len(ts_i) < 1000:
        return []

    rows = []
    for m in range(OBS_FIRST_MIN, OBS_LAST_MIN + 1, OBS_STEP_MIN):
        now = _et(d, m)
        rec = {"underlying": underlying, "dt": str(d), "obs_minute": m,
               "obs_et": now.isoformat()}
        for name, mins in LOOKBACKS:
            sig, pf, cov = F.window_sigma(cal, ts_i, px_i, ts_p, px_p,
                                          now - datetime.timedelta(minutes=mins), now,
                                          scale_s=LOOKBACK_SCALE_S[name])
            rec[f"f_{name}"] = sig
            rec[f"pf_{name}"] = pf
            rec[f"cov_{name}"] = cov
        for name, mins in HORIZONS:
            end_m = m + mins
            if end_m > SESSION_MIN:
                rec[f"y_{name}"] = np.nan
                rec[f"ycov_{name}"] = np.nan
                continue
            sig, _pf, cov = F.window_sigma(cal, ts_i, px_i, ts_p, px_p,
                                           now, _et(d, end_m),
                                           scale_s=HORIZON_SCALE_S[name])
            rec[f"y_{name}"] = sig
            rec[f"ycov_{name}"] = cov
        if short:
            _short_columns(rec, cal, ts_i, px_i, ts_p, px_p, d, m, now)
        rows.append(rec)
    return rows


def _target_flag(sig, cov):
    """Why a target is missing, or 'ok'. The reasons `features.window_sigma` collapses into NaN."""
    if np.isfinite(sig) and sig > 0:
        return "ok"
    if cov < F.MIN_COVERAGE:
        return "low_coverage"
    return "zero_rv"


def _short_columns(rec, cal, ts_i, px_i, ts_p, px_p, d, m, now):
    """The short-horizon columns. Written into the SAME row, and never touching an existing key,
    so every long-horizon feature and target stays bit-identical to a cache built without them."""
    for name, mins in SHORT_LOOKBACKS:
        sig, pf, cov = F.window_sigma(cal, ts_i, px_i, ts_p, px_p,
                                      now - datetime.timedelta(minutes=mins), now,
                                      scale_s=LOOKBACK_SCALE_S[name])
        rec[f"f_{name}"], rec[f"pf_{name}"], rec[f"cov_{name}"] = sig, pf, cov
    for name, mins in SHORT_HORIZONS:
        end_m = m + mins
        if end_m > SESSION_MIN:
            rec[f"y_{name}"] = rec[f"yref_{name}"] = np.nan
            rec[f"ycov_{name}"] = np.nan
            rec[f"yflag_{name}"] = "past_close"
            continue
        sig, _pf, cov = F.window_sigma(cal, ts_i, px_i, ts_p, px_p, now, _et(d, end_m),
                                       scale_s=HORIZON_SCALE_S[name])
        ref, _pf, _cov = F.window_sigma(cal, ts_i, px_i, ts_p, px_p, now, _et(d, end_m),
                                        scale_s=SHORT_REFERENCE_SCALE_S)
        rec[f"y_{name}"], rec[f"ycov_{name}"], rec[f"yref_{name}"] = sig, cov, ref
        rec[f"yflag_{name}"] = _target_flag(sig, cov)


def build_range(underlying, start=None, end=None, resume=True, verbose=True,
                obs_dir=None, short=False):
    import pandas as pd
    obs_dir = obs_dir or OBS_DIR
    cal = varcal.load(underlying)
    days = rdata.cached_days(rdata.SOURCES[underlying]["index"])
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    obs_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for i, d in enumerate(days):
        p = obs_path(underlying, d, obs_dir)
        if resume and p.exists():
            continue
        rows = build_session(underlying, d, cal, short=short)
        if not rows:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(p, index=False)
        n_new += 1
        if verbose and n_new % 50 == 0:
            print(f"[rvol.dataset] {underlying} {i+1}/{len(days)} days, {n_new} written")
    if verbose:
        print(f"[rvol.dataset] {underlying}: {n_new} sessions written, "
              f"{len(list((obs_dir / underlying).glob('dt=*.parquet')))} cached")
    return n_new


def load(underlying, start=None, end=None, obs_dir=None):
    import pandas as pd
    d = (obs_dir or OBS_DIR) / underlying
    files = sorted(d.glob("dt=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no observations for {underlying} -- run: python -m rvol.dataset --build")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    if start:
        df = df[df["dt"] >= start]
    if end:
        df = df[df["dt"] <= end]
    return df.sort_values(["dt", "obs_minute"]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the realized-vol observation matrix")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--underlying", nargs="*", default=["SPX", "NDX"])
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--obs-dir", default=None,
                    help="write/read this cache instead of var/rvol/obs (the short-horizon cache "
                         "is var/backtest/rvol_obs_short)")
    ap.add_argument("--short", action="store_true",
                    help="also build the 1m/3m short-horizon columns")
    a = ap.parse_args(argv)
    import pathlib
    od = pathlib.Path(a.obs_dir) if a.obs_dir else None
    if a.short and od is None:
        ap.error("--short writes new columns; give it its own --obs-dir, never var/rvol/obs")
    for u in a.underlying:
        if a.build:
            build_range(u, resume=not a.no_resume, obs_dir=od, short=a.short)
        else:
            df = load(u, obs_dir=od)
            print(f"  {u}: {len(df):,} rows, {df['dt'].nunique()} sessions, "
                  f"{df['dt'].min()} .. {df['dt'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
