"""Realized variance: the estimator, the clock, and the feature/target matrix.

THE ESTIMATOR -- averaged (subsampled) realized variance at a 30-SECOND scale.

Not 1s, and not the 5s the original spec asked for. An index is a computed level whose
constituents do not all trade every second, so high-frequency sampling is biased LOW -- the Epps
effect. Measured 2026-08-20, annualized RTH realized vol:

    ticker    1s      5s     30s     60s
    I:SPX   6.64%   7.23%   7.76%   7.76%
    I:NDX   9.26%  11.61%  13.76%  13.43%
    SPY     8.23%   7.99%   7.80%   7.60%

SPY is flat across intervals because it is one traded instrument; the indices are not, and I:NDX
reads 33% low at 1s and 16% low at 5s. As a FEATURE that is survivable -- the regression rescales
it. As a TARGET it is not: the lean compares the model's output directly against implied vol, so a
15-minute target sampled at 5s would bias every short-horizon lean by ~16% on NDX, permanently.

Sampling everything at 30s would then leave a 5-minute window with only 10 returns. The fix is to
average the RV over all 30 one-second-offset 30s grids: the estimate keeps the unbiased 30s scale
while using every observation. On a synthetic GBM path it must be invariant to the offset, which
is what distinguishes it from a plain 30s RV and is asserted in rvol/validate.

THE CLOCK -- everything is in SESSION-VARIANCE units, then annualized at 252.

A raw RV over a window is not comparable across times of day: 09:30-10:00 carries 2.4x a flat
minute and 14:30-15:00 carries 0.53x. Dividing by the variance the calendar expects over that
exact window turns each measurement into "what this window implies about the WHOLE session's
variance", which is comparable across horizons and across times of day, and is the same clock
`HourlyPricer.sigma_annualized` already uses (`w / sqrt((1 - Phi(now)) / 252)`).

So the model never re-learns the intraday seasonality that pricing/varcal already measures, and a
prediction of "1.0x the calendar" means the calendar was right.

THE PRE-OPEN. A 3-hour lookback taken at 09:30 reaches into the overnight gap, where there is no
index at all. The gap's variance-time now comes from `pricing/gapcal`, which measures the shape of
the night on 1-minute buckets from ES/NQ futures (and SPY/QQQ across the 17:00-18:00 CME halt),
scaled to the total `varcal.overnight_share` gives it. Before that it was spread UNIFORMLY in clock
time across 17.5 hours, which weighted 02:00 exactly like 09:00.

Two things still hold and are worth keeping: `proxy_frac` tells the model how much of a window was
built from the pre-open leg, and **targets are always entirely inside RTH**, so no target is ever
contaminated by the gap at all.

NOTE what deliberately did NOT change with it: the pre-open PRICE series is still the ETF proxy,
not the futures, even though ES gives ~3.8x the pre-open observations. Changing the calendar and
the price source in one step would make `pf_3h` uninterpretable -- and `pf_3h` moving off -0.30 is
the acceptance test for the calendar change. The source swap is a separate experiment.
"""
import datetime
import functools
import math

import numpy as np

import config
from pricing import gapcal

# The DEFAULT scale, and the one every TARGET uses. Features now sample per-horizon -- see
# rvol/dataset's LOOKBACK_SCALE_S -- but targets deliberately do not.
#
# A feature may be biased and survive it: the regression fits a coefficient per lookback, so a
# systematic scale bias is rescaled away. A TARGET cannot. It is the thing being predicted and the
# thing the eventual realized-vs-implied comparison is scored against, so a bias in it is a
# permanent bias in every forecast the model makes. The two biases available here point in
# opposite directions and neither is small:
#
#   * Indices are Epps-biased LOW at high frequency, because their constituents do not all trade
#     every second. Measured 2026-08-20, annualized RTH realized vol:
#         I:SPX 6.64% @1s, 7.23% @5s, 7.76% @30s, 7.76% @60s
#         I:NDX 9.26% @1s, 11.61% @5s, 13.76% @30s, 13.43% @60s     <- 33% low at 1s
#   * The extended-hours ETF tape is microstructure-noise-biased HIGH. Measured over the overnight
#     gap against the index close-to-open move, SPY realized variance reads 2.78x at 10s, 1.55x at
#     30s, 1.36x at 60s and 0.76x at 900s -- a signature plot, not a level error.
#
# So: 10s for the two short lookbacks because that is what the study asked for and a coefficient
# absorbs it, 60s for the three long ones, and 30s with sub-grid averaging for all three targets.
SCALE_S = 30
N_OFFSETS = None                # default: one sub-grid per second of whatever scale is used
TRADING_DAYS = 252.0

OPEN_T = datetime.time(9, 30)
CLOSE_T = datetime.time(16, 0)


# An RV window with less data coverage than this returns NaN. Illustrative value; the live value is
# not published.
MIN_COVERAGE = 0.6


def averaged_rv(ts_ms, px, t0_ms, t1_ms, scale_s=SCALE_S, n_offsets=N_OFFSETS):
    # `n_offsets` defaults to the scale so the sub-grids are one second apart whatever the scale
    # is -- 60 grids at 60s, 10 at 10s. Pinning it to a constant 30 would put the 10s grids a
    # third of a second apart (finer than the data) and the 60s grids two seconds apart.
    """Averaged realized variance over [t0_ms, t1_ms), at `scale_s`, or (nan, 0.0).

    Returns (rv, coverage) where `coverage` is the fraction of the window spanned by grid points
    that had an observation at or before them. The RV is expressed as a RATE times the nominal
    window length, so ragged coverage across sub-grids does not make one offset contribute more
    than another.
    """
    if t1_ms <= t0_ms or ts_ms is None or len(ts_ms) < 2:
        return math.nan, 0.0
    n_offsets = n_offsets or scale_s
    step = scale_s * 1000
    span = float(t1_ms - t0_ms)
    rates, covered = [], []
    for off in range(n_offsets):
        g0 = t0_ms + off * (step // n_offsets if n_offsets else 0)
        grid = np.arange(g0, t1_ms + 1, step, dtype="int64")
        if grid.size < 2:
            continue
        idx = np.searchsorted(ts_ms, grid, side="right") - 1
        ok = idx >= 0
        if ok.sum() < 2:
            continue
        # Forward-fill: each grid point takes the last observation at or before it. Grid points
        # before the first observation are dropped, not back-filled -- back-filling would invent
        # a price the market had not printed.
        p = px[idx[ok]]
        p = p[p > 0]
        if p.size < 2:
            continue
        r = np.diff(np.log(p))
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        g = grid[ok]
        cov_span = float(g[-1] - g[0])
        if cov_span <= 0:
            continue
        rates.append(float((r ** 2).sum()) / cov_span)
        covered.append(cov_span / span)
    if not rates:
        return math.nan, 0.0
    return float(np.mean(rates)) * span, float(np.mean(covered))


@functools.lru_cache(maxsize=8)
def gap_calendar(underlying, profile="default"):
    """The overnight shape, cached. Raises if it has not been built -- deliberately.

    There is no fallback to the old uniform apportionment. A silent fallback would make a missing
    artifact look like a working model with slightly different numbers, which is exactly the
    failure mode this change exists to remove.
    """
    return gapcal.load(underlying, profile)


def variance_time(cal, t0_et, t1_et, gcal=None):
    """Variance-time between two ET datetimes, in units where one RTH session == 1.0.

    `t1_et` is assumed to be inside RTH (every window this study builds ends there). `t0_et` may
    fall in the preceding gap, and that portion is now apportioned by SHAPE rather than by clock
    time: `pricing/gapcal` measures where in the night the variance actually lands, and
    `cal.overnight_share` says how much of it there is in total.

    Units are the load-bearing part. `gapcal.share_between` returns a fraction of the GAP, and
    multiplying by `overnight_share` -- which varcal expresses in multiples of one RTH session --
    puts it on the same clock as `cal.phi`. The two calendars meet here and nowhere else.

    What this replaced: `overnight_share * (gap_seconds / 63000)`, the gap spread uniformly across
    17.5 hours, weighting 02:00 exactly like 09:00. Every pre-open lookback was normalised by that,
    and the regression responded by fitting `pf_3h` to -0.30 -- it learned to discount the feature
    rather than use it. That coefficient moving toward zero is the test of whether this worked.
    """
    open_dt = datetime.datetime.combine(t1_et.date(), OPEN_T, tzinfo=config.ET)
    v = 0.0
    lo = max(t0_et, open_dt)
    if t1_et > lo:
        v += cal.phi(t1_et) - cal.phi(lo)
    if t0_et < open_dt:
        g = gcal or gap_calendar(cal.underlying)
        v += cal.overnight_share * g.share_between(t0_et, open_dt)
    return max(v, 0.0)


def annualized(session_var):
    """session-variance units -> annualized sigma, on the intraday variance clock.

    One RTH session is one variance-day and there are 252 of them, exactly as
    `HourlyPricer.sigma_annualized` assumes. The overnight gap never enters, which is the whole
    reason both sides of the eventual comparison live on this clock.
    """
    if session_var is None or not np.isfinite(session_var) or session_var < 0:
        return math.nan
    return math.sqrt(session_var * TRADING_DAYS)


def _ms(dt_et):
    return int(dt_et.timestamp() * 1000)


def window_sigma(cal, ts_idx, px_idx, ts_px, px_px, t0_et, t1_et, scale_s=SCALE_S):
    """Annualized sigma implied by one window, plus diagnostics.

    Splices the index series (RTH) with the ETF proxy (pre-open). Returns
    (sigma_ann, proxy_frac, coverage) with NaN sigma when the window is too sparsely covered.
    """
    open_dt = datetime.datetime.combine(t1_et.date(), OPEN_T, tzinfo=config.ET)
    parts, proxy_var = [], 0.0
    total_rv = 0.0
    cov_num = cov_den = 0.0

    if t0_et < open_dt:
        rv_p, cov_p = averaged_rv(ts_px, px_px, _ms(t0_et), _ms(min(open_dt, t1_et)), scale_s)
        if np.isfinite(rv_p):
            total_rv += rv_p
            proxy_var += rv_p
        w = (min(open_dt, t1_et) - t0_et).total_seconds()
        cov_num += cov_p * w
        cov_den += w

    lo = max(t0_et, open_dt)
    if t1_et > lo:
        rv_i, cov_i = averaged_rv(ts_idx, px_idx, _ms(lo), _ms(t1_et), scale_s)
        if np.isfinite(rv_i):
            total_rv += rv_i
        w = (t1_et - lo).total_seconds()
        cov_num += cov_i * w
        cov_den += w

    coverage = cov_num / cov_den if cov_den > 0 else 0.0
    vt = variance_time(cal, t0_et, t1_et)
    if vt <= 0 or coverage < MIN_COVERAGE or total_rv <= 0:
        return math.nan, math.nan, coverage
    proxy_frac = proxy_var / total_rv if total_rv > 0 else 0.0
    return annualized(total_rv / vt), proxy_frac, coverage
