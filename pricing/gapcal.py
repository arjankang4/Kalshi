"""Overnight variance calendar: how the 16:00 -> 09:30 gap's variance is distributed across its minutes.

`pricing/varcal` measures the 390 RTH minutes and then knows the whole overnight gap as ONE
NUMBER, `overnight_share`. `rvol/features.variance_time` spreads that number uniformly in clock
time across 17.5 hours. That is wrong in a way the model has already noticed: the 3h lookback
feature, which reaches into the gap for every observation before 12:30, fits a coefficient of
**-0.30** -- the regression learned to discount a feature built from a badly-apportioned pre-open
rather than to use it.

This module gives the gap a shape, on the same 1-minute grid and with the same estimator varcal
uses for the session, so the two halves of the day are measured the same way.

    varcal   09:30 -> 16:00   391 buckets   normalised to 1.0
    gapcal   16:00 -> 09:30  1050 buckets   normalised to 1.0, then SCALED BY overnight_share

WHY THIS IS A SEPARATE MODULE. varcal's artifact, its public surface (`phi`, `remaining_ratio`,
`overnight_share`) and the `phi(16:00) == 1.0` identity that `pricing/validate_hourly` asserts are
all untouched by anything here. The hourly pricer is a pure variance RATIO and the gap cancels out
of it, so nothing in the quoting path changes; the only consumer is `rvol/features`.

WHERE THE PRICES COME FROM

    16:00-17:00   ES / NQ      futures trade through the close
    17:00-18:00   SPY / QQQ    CME maintenance halt; the ETF post-market runs to 20:00
    18:00-09:30   ES / NQ      including 20:00-04:00, where the ETFs have NOTHING

The ETF tape alone cannot do this, and not only because of the 8-hour hole. Measured over 71-126
cycles, SPY realized variance over the 9.5 hours it DOES cover, against the SPX close-to-open
squared gap (a clean estimator must read well below 1.0, since 8 hours are missing):

    10s 2.783    30s 1.551    60s 1.356    300s 0.832    900s 0.758

That is a microstructure-noise signature plot -- implied noise sd ~0.77bp per print, about a
half-spread in extended hours. Crucially the bias is NOT a level error that the final scaling step
would absorb: noise enters in proportion to PRINT DENSITY, which varies tenfold across the night,
so it distorts the shape too. Hence futures, and hence the noise check in `signature_plot` rather
than an assumption that the futures tape is clean.

THE SPX INDEX OPEN PRINT IS STALE, AND IT UNDERSTATES THE LEVEL WE INHERIT

Measured here 2026-08-22 over 388 cycles, mean squared log move from the prior index close to:

                        SPX                 NDX
    09:30 first print   2.5696e-05          6.7383e-05      <- what varcal calls the overnight gap
    09:35               3.6210e-05  1.41x   6.7573e-05  1.00x
    09:30 -> 09:35      3.0077e-06          3.7642e-06

For SPX the gap measured to 09:35 is **41% larger** than the gap measured to the 09:30 print, and
the first five minutes on their own are nowhere near big enough to account for it: independent
increments would give 2.57e-5 + 0.30e-5 = 2.87e-5, not 3.62e-5. The excess is a positive covariance
between "the overnight gap" and "the first five minutes", which is the signature of a stale open --
`I:SPX` first prints at 09:30:01, and a 500-name index computed one second into the session is
still carrying prior closes for the constituents that have not opened yet. `I:NDX` first prints at
09:30:15, has 100 constituents, and shows **no staleness at all** (1.00x): a clean control.

THE CLEANEST IDENTIFICATION IS THE CORRELATION, not the variance ratio. Under a martingale with a
clean opening print, the overnight gap and the first five minutes are independent increments and
their correlation is zero. Measured over 388 cycles:

                     corr(gap, 09:30-09:35)    placebo: corr(09:30-09:35, 09:35-09:40)
    SPX                    +0.427                            +0.067
    NDX                    -0.111                            +0.026

The placebo column matters: adjacent five-minute windows inside the session are uncorrelated for
both indices, so +0.427 is not "windows near each other correlate in this data". Only the
gap-to-first-five-minutes pair correlates, and only for SPX. That is the leak itself rather than an
inference drawn from a variance decomposition. Independently reproduced by the varcal session on a
different sample (286 event-clean nights): SPX +0.419, NDX -0.091.

HOW BIG IT IS -- AND WHY THE OBVIOUS MEASUREMENT OVERSTATES IT BY ABOUT HALF

Re-anchoring the gap at 09:35 raises `overnight_share`. It is tempting to read the whole rise as
staleness. It is not, and NDX proves it:

                     base (09:30)   numerator only    anchor moved
    SPX                    0.2916   0.3657 (+25.4%)   0.3756 (+28.8%)
    NDX                    0.3873   0.4165 ( +7.5%)   0.4325 (+11.7%)

NDX has no staleness by either measure above, and it still gains 7.5-11.7%. The reason is
mechanical: 09:30-09:35 is the busiest block of the session at ~2.7x a flat minute, so removing it
from the denominator `I` inflates `f = r_on^2 / (I + r_on^2)` whether or not anything is stale.

So the staleness-attributable part is the EXCESS OVER THE NDX CONTROL: **~17-18 percentage points**
(25.4 - 7.5 = 17.9, or 28.8 - 11.7 = 17.1), not the ~29% the raw shift shows and not the 41-58% the
variance ratios suggest. Still material and worth having; about half of the naive reading.

WHICH NUMBER TO BELIEVE IF THEY EVER DISAGREE. Three measurements appear here and they answer
different questions, so their agreement is corroboration and not replication:

  * the CORRELATION identifies whether the open is stale at all -- fewest assumptions, use this;
  * the NDX-CONTROLLED EXCESS sizes it -- use this for the magnitude;
  * the FUTURES-vs-INDEX-GAP ratio (SPX 1.577, NDX 1.025) compares integrated variance over a whole
    night against a single close-to-open return. It also carries basis, the index's own 16:00-16:04
    tail, and any genuine ES-vs-cash difference, so it is an UPPER BOUND on total level error and
    must not be used to size staleness.

The consequence lands on this module either way. `varcal.overnight_share` is fitted as
`r_on^2 / (I + r_on^2)` off exactly that stale open, so for SPX it is biased low -- and it is the
number `gap_shares` is scaled by.

DO NOT "FIX" THIS BY SCALING `overnight_share`. If the opening print is stale the defect is in the
ANCHOR DEFINITION, and moving the anchor takes the busiest five minutes of the session out of the
intraday calendar that `pricing/hourly` reads. That is a structural change to `shares`, not a
multiplier on one scalar. Note also that it is almost entirely an `overnight_share` problem and
barely a quoting one: `hourly` reads `remaining_ratio(now, expiry)`, which depends on the shape from
`now` forward, so the 09:30-09:35 attribution is invisible to any quote placed after 09:35. It bites
`rvol/features.variance_time` and anything else reading the scalar -- which is why it surfaced here.

This is a `pricing/varcal` issue, not a `gapcal` one -- the SHAPE here is unaffected, because it is
fitted on per-cycle fractions that are normalised within each night. Only the LEVEL is inherited.
It is left uncorrected on purpose: inventing a correction factor here would bury a varcal bug
inside a different module. `--check` reports the staleness alongside the ratio so the two are never
confused, and `validate_gapcal` treats a high ratio as EXPECTED where the open is stale.

THE LEVEL COMES FROM THE INDEX, NEVER FROM THE FUTURES

`gap_shares` is a SHAPE and sums to 1. The level is `varcal`'s `overnight_share`, measured on the
index's own close-to-open return. ES is a forward: it carries basis to cash, it rolls quarterly,
and it trades a different session. Returns are near-invariant to a smooth overnight carry, which
is what licenses it as a shape proxy -- exactly the argument that licenses SPY for returns and
forbids it for levels. Anchoring the level to ES would import basis and the roll into the answer.

This also yields the first genuine out-of-sample check this calendar has ever had: total measured
16:00->09:30 variance, divided by the mean squared index close-to-open return, should be **~1.0**.
`--check` reports it. varcal's own gates are scored in-sample against the window it was fitted on.

THE ROLL IS A DISCONTINUITY, NOT VARIANCE

At the 2024-09 roll, ESU4 closed 5639.00 and ESZ4 closed 5700.50 on the same day: a 61.5 point,
1.09% step that is carry, not risk. Cycles whose two calendar days resolve to different contracts
are DROPPED rather than spliced -- there are only a handful of rolls in two years, and a partial
splice would put a pure basis step into one overnight bucket and let it dominate.

SINGLE-NIGHT CYCLES ONLY. The shape here should be fitted on the SAME population as the level it
will be scaled by, or the two describe different things -- and AS OF 2026-08-24 THEY DO. varcal now
fits `overnight_share` on ALL gaps pooled (weeknights, weekends, holidays), while this shape is
still weeknight-only. See `varcal.gap_span` and the box-archive note for why that was chosen.

The mismatch is bounded and one-signed. A weekend carries more variance than a weeknight, so the
pooled level sits above the weeknight level: SPX 0.2773 vs 0.2496 (+11.1%), NDX 0.3762 vs 0.3281
(+14.7%). The ONLY consumer is `rvol/features.variance_time`, and only for a lookback whose window
reaches back into a gap. On a weeknight it now assigns ~11% (SPX) / ~15% (NDX) too much variance to
the pre-open leg; on a weekend it assigns too little, because a Friday-to-Monday cycle is scaled by
the pooled average rather than the larger weekend figure. It does NOT touch the quote path: the 4pm
board settles at `phi == 1.0`, where the calendar cancels out entirely.

THE FIX, IF THE PRE-OPEN LOOKBACK NEEDS IT, is to make `variance_time` span-aware -- scale by the
gap's own span rather than by one pooled scalar -- NOT to re-fit this shape on all gaps. Pooling
this shape would splice a 65-hour weekend onto a 17.5-hour grid, which is a different and worse
error than the one it would fix.

Fitting on single nights also has a free consequence -- DST transitions are always Sunday 02:00, so
no weekday-to-weekday cycle can straddle one. `validate_gapcal` asserts that rather than trusting
it.
"""
import argparse
import datetime
import json
import math
import sys
from datetime import date, datetime as dt, time as dtime, timedelta

import numpy as np

import config
from pricing import eventcal, varcal

CACHE = config.ROOT / "var" / "rvol" / "bars"
CHAIN_PATH = config.ROOT / "var" / "rvol" / "futures_chain.json"

# 16:00 -> 09:30 is 17.5 hours = 1050 wall-clock minutes, and 1050 = 210 x 5 exactly, so the fine
# and coarse grids nest with no ragged final block (unlike varcal, whose 391st bucket is a
# singleton). Buckets are labelled by WALL CLOCK, so the grid is 1050 long on every date.
GAP_START_T = dtime(16, 0)
GAP_END_T = dtime(9, 30)
GAP_MINUTES = 1050
FINE_WIDTH = 60
COARSE_WIDTH = 300
FINE_PER_BLOCK = COARSE_WIDTH // FINE_WIDTH          # 5
N_BLOCKS = GAP_MINUTES // FINE_PER_BLOCK             # 210

# The CME maintenance halt. Futures are dark here and the ETF post-market is the only tape.
HALT_START_T = dtime(17, 0)
HALT_END_T = dtime(18, 0)
HALT_FIRST = 60                                      # bucket index of 17:00
HALT_LAST = 120                                      # bucket index of 18:00 (exclusive end)

# Underlying -> (futures product, extended-hours ETF). Both are pooled into one shape, exactly as
# varcal pools SPX and NDX: the overnight session is a property of the global equity complex --
# the same European open, the same 08:30 ET prints -- not of a particular index.
SOURCES = {
    "SPX": {"futures": "ES", "etf": "SPY"},
    "NDX": {"futures": "NQ", "etf": "QQQ"},
}
UNDERLYINGS = ("SPX", "NDX")

DEFAULT_YEARS = 2

# A cycle needs this many of its 1050 buckets to carry a real observation before it is allowed to
# contribute a shape. Overnight coverage is genuinely thin -- ES prints in 25-50% of seconds
# between 20:00 and 04:00 -- but a night where most buckets are forward-filled from one stale
# price is not a measurement of anything.
MIN_COVERED = 0.60

# Report-only, as varcal's WARN_BUCKET_SHARE is: a night whose busiest single minute holds more
# than this much of its own gap variance is printed, not dropped. The 2024-09 roll step would have
# read 0.9+ here, which is what this is for -- catching a splice error, not a violent night.
WARN_BUCKET_SHARE = 0.35


def minute_labels():
    """['16:00', '16:01', ..., '09:29'] -- 1050 wall-clock labels."""
    out = []
    t = dt(2000, 1, 1, GAP_START_T.hour, GAP_START_T.minute)
    for _ in range(GAP_MINUTES):
        out.append(t.strftime("%H:%M"))
        t += timedelta(minutes=1)
    return out


def gap_edges(d):
    """The 1051 ET instants bounding a cycle's buckets, from `d` 16:00 to `d+1` 09:30.

    Built by walking wall-clock minutes and localising each one through `config.ET`, never by
    adding a fixed offset to a single anchor. `pricing/varbars._et_seconds` reads the UTC offset
    ONCE and reuses it, which is safe only because an RTH session cannot straddle the 02:00 Sunday
    transition -- a 16:00 -> 09:30 window can, and this one does not inherit that shortcut.
    """
    base = dt.combine(d, GAP_START_T)
    out = []
    for k in range(GAP_MINUTES + 1):
        naive = base + timedelta(minutes=k)
        aware = naive.replace(tzinfo=config.ET)
        out.append(aware)
    return out


def _cache_path(ticker, d):
    return CACHE / ticker.replace(":", "_") / f"dt={d}.parquet"


def _load_day(ticker, d):
    """(t_ms, close) numpy arrays for one ET calendar date, or (None, None).

    `pricing` deliberately does not import `rvol` -- `rvol/{dataset,live,features}` already import
    `pricing.varcal`, so reaching the other way would make the dependency circular. The few lines
    of Parquet access are repeated here for the same reason `pricing/varbars` repeats them.
    """
    import pyarrow.parquet as pq
    p = _cache_path(ticker, d)
    if not p.exists():
        return None, None
    t = pq.read_table(p)
    return (np.asarray(t.column("t_ms"), dtype="int64"),
            np.asarray(t.column("close"), dtype="float64"))


def load_chain():
    return json.loads(CHAIN_PATH.read_text()) if CHAIN_PATH.exists() else {}


def _segment_grid(ts_ms, px, edges_ms, lo_ms, hi_ms):
    """Prices on `edges_ms` using only source prints inside [lo_ms, hi_ms]. Also a covered mask.

    Restricting the source to the segment is the whole point. Forward-fill is right for a sparse
    overnight tape -- an edge with no print of its own carries the last one, yielding a zero
    return in that bucket and the full move in the next -- but an UNRESTRICTED forward-fill would
    carry the 17:00 futures price across the CME maintenance halt and hand the 18:00 reopen jump
    to the first post-halt bucket. That jump is the same hour SPY is being used to measure, so it
    would be counted twice.

    The opening edge is the one exception: it takes the first print at or AFTER `lo_ms` when
    nothing precedes it, so a segment whose first print lands a few seconds late still measures
    its first bucket instead of discarding it.
    """
    if ts_ms is None or len(ts_ms) < 2:
        return None, None
    sel = (ts_ms >= lo_ms) & (ts_ms <= hi_ms)
    if sel.sum() < 2:
        return None, None
    t, q = ts_ms[sel], px[sel]
    idx = np.searchsorted(t, edges_ms, side="right") - 1
    out = np.full(len(edges_ms), np.nan)
    ok = idx >= 0
    out[ok] = q[idx[ok]]
    if not ok[0]:
        out[0] = q[0]                       # opening edge: back-fill to the segment's first print
    advanced = np.zeros(len(edges_ms), dtype=bool)
    advanced[1:] = idx[1:] > idx[:-1]
    return out, advanced


# The gap is measured as three independent SEGMENTS, and no return is ever taken across a seam
# between them. Each segment's buckets draw both of their edges from one source, so a futures
# level is never differenced against an ETF level -- that would price the basis, not the market.
#
#     [16:00, 17:00)  futures    60 buckets
#     [17:00, 18:00)  ETF        60 buckets   -- the CME maintenance halt
#     [18:00, 09:30)  futures   930 buckets
#
SEGMENTS = ((0, HALT_FIRST, "futures"),
            (HALT_FIRST, HALT_LAST, "etf"),
            (HALT_LAST, GAP_MINUTES, "futures"))


def _cycle_returns(und, d0, d1, chain, scale_s=FINE_WIDTH):
    """One cycle -> (returns, covered) on a `scale_s` grid over the gap, or None.

    The grid always spans 16:00 -> 09:30 and always divides it evenly: 63000 seconds is a whole
    multiple of 10, 30, 60, 300 and 900, so every scale the signature plot uses nests cleanly.
    """
    src = SOURCES[und]
    tickers = {"futures": src["futures"], "etf": src["etf"]}

    # A roll between the two calendar days makes the tape a different instrument on each side of
    # midnight. Drop the cycle: a handful of rolls in two years is a cheaper loss than splicing a
    # multi-month basis step into one overnight bucket.
    pc_chain = chain.get(src["futures"], {})
    a, b = pc_chain.get(str(d0)), pc_chain.get(str(d1))
    if a and b and a != b:
        return None

    if (60 * GAP_MINUTES) % scale_s:
        raise ValueError(f"scale {scale_s}s does not divide the {GAP_MINUTES}-minute gap evenly")
    per_bucket = scale_s // 60 if scale_s >= 60 else None

    edges_dt = gap_edges(d0)
    start_ms = int(edges_dt[0].timestamp() * 1000)
    end_ms = int(edges_dt[-1].timestamp() * 1000)
    if per_bucket:
        # Wall-clock anchored, so a DST transition would be handled correctly if one could ever
        # occur here (it cannot -- see the module docstring on single-night cycles).
        edges_ms = np.array([int(e.timestamp() * 1000) for e in edges_dt[::per_bucket]],
                            dtype="int64")
    else:
        edges_ms = np.arange(start_ms, end_ms + 1, scale_s * 1000, dtype="int64")

    series = {}
    for kind, tk in tickers.items():
        t0, p0 = _load_day(tk, d0)
        t1, p1 = _load_day(tk, d1)
        parts_t = [x for x in (t0, t1) if x is not None]
        parts_p = [x for x in (p0, p1) if x is not None]
        if not parts_t:
            return None
        series[kind] = (np.concatenate(parts_t), np.concatenate(parts_p))

    n = len(edges_ms) - 1
    r = np.full(n, np.nan)
    cov = np.zeros(n, dtype=bool)
    for lo_min, hi_min, kind in SEGMENTS:
        lo_i = lo_min * 60 // scale_s
        hi_i = hi_min * 60 // scale_s
        seg_edges = edges_ms[lo_i:hi_i + 1]
        if len(seg_edges) < 2:
            continue
        ts, px = series[kind]
        g, adv = _segment_grid(ts, px, seg_edges, seg_edges[0], seg_edges[-1])
        if g is None:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            good = np.isfinite(g[:-1]) & np.isfinite(g[1:]) & (g[:-1] > 0) & (g[1:] > 0)
            seg_r = np.where(good, np.log(np.where(good, g[1:], 1.0)
                                          / np.where(good, g[:-1], 1.0)), np.nan)
        r[lo_i:hi_i] = seg_r
        cov[lo_i:hi_i] = adv[1:]
    return r, cov


def signature_plot(und, cycles, scales=(10, 30, 60, 300, 900)):
    """Mean gap RV by sampling scale, plus the index close-to-open gap it should reproduce.

    The diagnostic that condemned the ETF tape. Run it on the futures tape before trusting the
    fine grid: if RV still falls materially as the scale coarsens, microstructure noise is being
    read as variance and the shape is distorted toward wherever prints are densest.
    """
    out = {}
    chain = load_chain()
    for sc in scales:
        tot = []
        for d0, d1 in cycles:
            got = _cycle_returns(und, d0, d1, chain, scale_s=sc)
            if got is None:
                continue
            r = got[0]
            r = r[np.isfinite(r)]
            if r.size:
                tot.append(float((r ** 2).sum()))
        out[sc] = float(np.mean(tot)) if tot else float("nan")
    return out


# --- fitting -------------------------------------------------------------
#
# THE ESTIMATOR IS varcal's, DELIBERATELY.
#
# Per-CYCLE normalisation to fractions summing to 1, equal-weight average across cycles, one-sided
# per-bucket p99 winsorisation, then a coarse/fine rescale. Not a choice made for symmetry -- it
# was re-measured here and it matters more overnight than it does intraday. On the SPY tape, the
# 07:30-08:30 slice reads:
#
#     mean-of-squares 0.318      per-cycle fraction 0.228      median 0.153
#
# and the top five nights each put 70-77% of the whole gap into that one hour. A mean of squared
# returns is a variance-weighted average of nights, so a handful of them own the shape. The gap's
# distribution across nights is far more right-skewed than the session's, which is exactly why
# varcal's old estimator read `overnight_share` nine points too high.
#
# The winsoriser clips the per-bucket series of daily FRACTIONS at its own p99. That is an order
# statistic over ~300 real observations and cannot collapse, unlike a MAD-derived scale applied to
# raw returns -- the version of that idea which was measured to destroy 63% of I:NDX's closing
# minute.


def _pooled_cycle(d0, d1, chain, loud=None):
    """One cycle -> pooled fine/coarse fraction vectors across underlyings, or None.

    Each underlying is normalised to sum 1 on its OWN gap before pooling, so a night when NQ
    happened to move more than ES does not tilt the shape toward NQ's minutes. Same construction
    as `varcal._day_fractions` followed by its per-day pooling.
    """
    per_u = {}
    for u in UNDERLYINGS:
        fracs, cov_frac = {}, None
        ok = True
        for w in (FINE_WIDTH, COARSE_WIDTH):
            got = _cycle_returns(u, d0, d1, chain, scale_s=w)
            if got is None:
                ok = False
                break
            r, cov = got
            sq = np.where(np.isfinite(r), r, 0.0) ** 2
            tot = float(sq.sum())
            if tot <= 0:
                ok = False
                break
            fracs[w] = sq / tot
            if w == FINE_WIDTH:
                cov_frac = float(cov.mean())
        if not ok or cov_frac is None or cov_frac < MIN_COVERED:
            continue
        if loud is not None and fracs[FINE_WIDTH].max() > WARN_BUCKET_SHARE:
            loud.append((d0, u, float(fracs[FINE_WIDTH].max()),
                         int(fracs[FINE_WIDTH].argmax())))
        per_u[u] = {"fracs": fracs, "cov": cov_frac}
    if not per_u:
        return None
    return {
        "fine": np.mean([v["fracs"][FINE_WIDTH] for v in per_u.values()], axis=0),
        "coarse": np.mean([v["fracs"][COARSE_WIDTH] for v in per_u.values()], axis=0),
        "cov": float(np.mean([v["cov"] for v in per_u.values()])),
        "underlyings": sorted(per_u),
    }


def cached_cycles(years=DEFAULT_YEARS, end=None):
    """[(d0, d1)] single-night cycles whose BOTH dates have every leg cached.

    Single-night only, matching `varcal._select_gaps(span=1)`: a weekend gap carries ~1.37x a
    single overnight, and the shape must be fitted on the same population as the level that will
    scale it. It also guarantees no cycle straddles a DST transition, since those are always
    Sunday 02:00.
    """
    end = end or config.today_et()
    start = end - timedelta(days=int(365.25 * years))
    tickers = sorted({t for v in SOURCES.values() for t in v.values()})
    have = []
    d = start
    while d <= end:
        if d.weekday() < 5 and all(_cache_path(t, d).exists() for t in tickers):
            have.append(d)
        d += timedelta(days=1)
    return [(a, b) for a, b in zip(have, have[1:]) if (b - a).days == 1]


def _rescale_to_blocks(fine, coarse):
    """Rescale each 5-minute block of `fine` to carry the total `coarse` gives it.

    Same contract as `varcal.rescale_to_blocks` -- and the same identity afterwards, for every k:
    `sum(out[5k:5k+5]) == coarse[k]` -- but the gap grid is 1050/210 with no ragged terminal
    bucket, so varcal's version cannot be called directly. An empty block is spread flat rather
    than left at zero: a zero-width step in `phi_gap` is a window with no variance in it, which
    a caller dividing by variance-time reads as infinite.
    """
    if len(fine) != GAP_MINUTES:
        raise ValueError(f"fine grid has {len(fine)} buckets, want {GAP_MINUTES}")
    if len(coarse) != N_BLOCKS:
        raise ValueError(f"coarse grid has {len(coarse)} blocks, want {N_BLOCKS}")
    out = np.array(fine, dtype="float64")
    for k in range(N_BLOCKS):
        sl = slice(k * FINE_PER_BLOCK, (k + 1) * FINE_PER_BLOCK)
        tot = out[sl].sum()
        out[sl] = out[sl] * (coarse[k] / tot) if tot > 0 else coarse[k] / FINE_PER_BLOCK
    return out / out.sum()


def _apply_halt_cap(shares, verbose=True):
    """Cap the 17:00-18:00 hour at an average hour of 18:00-09:30, one-sided.

    Arjan's rule, implemented literally: the halt hour is the one slice with no futures behind it,
    priced off a post-market ETF tape thin enough that its variance is mostly spread. If it comes
    in ABOVE what a normal hour of the night carries, hold it there and redistribute the surplus
    proportionally across the futures-sourced buckets. If it comes in below, leave it -- a
    one-sided rule, because the concern is the ETF inflating a slice it cannot measure well, not
    the ETF being quiet.

    Returns (shares, report).
    """
    halt = slice(HALT_FIRST, HALT_LAST)
    rest_idx = np.r_[0:HALT_FIRST, HALT_LAST:GAP_MINUTES]
    halt_share = float(shares[halt].sum())
    # 'an average hour from 18:00 to 09:30' -- the futures-sourced night after the halt.
    night = shares[HALT_LAST:]
    avg_hour = float(night.sum()) / (len(night) / 60.0)
    report = {"halt_share": halt_share, "avg_hour": avg_hour, "capped": False}
    # Tolerance, not pedantry: on a perfectly flat shape the two are equal in exact arithmetic and
    # differ in the last bit, which would report a 0.0-surplus 'cap' and read as a real one.
    if halt_share <= avg_hour * (1 + 1e-12):
        if verbose:
            print(f"[gapcal] halt hour 17:00-18:00 carries {halt_share:.4f} vs an average "
                  f"18:00-09:30 hour of {avg_hour:.4f} -- below, left alone")
        return shares, report
    out = np.array(shares, dtype="float64")
    surplus = halt_share - avg_hour
    out[halt] *= avg_hour / halt_share
    out[rest_idx] += surplus * (out[rest_idx] / out[rest_idx].sum())
    report.update(capped=True, surplus=surplus)
    if verbose:
        print(f"[gapcal] halt hour 17:00-18:00 carries {halt_share:.4f} vs an average "
              f"18:00-09:30 hour of {avg_hour:.4f} -- CAPPED, {surplus:.4f} redistributed")
    return out / out.sum(), report


def fit(years=DEFAULT_YEARS, end=None, exclude_events=True, families=None, verbose=True):
    """Fit the gap shape. Returns (shares, meta)."""
    chain = load_chain()
    if not chain:
        raise RuntimeError("no futures chain -- run: python -m rvol.data --chain")
    evcal = None
    if exclude_events:
        try:
            evcal = eventcal.load()
        except FileNotFoundError:
            if verbose:
                print("[gapcal] no event calendar; fitting without exclusion")
    cycles = cached_cycles(years=years, end=end)
    loud, fine, coarse, cov, kept = [], [], [], [], []
    for d0, d1 in cycles:
        # A gap is keyed by the session it lands ON, matching eventcal's component attribution:
        # an after-the-close earnings release contaminates the gap out of D, not D's session.
        if evcal is not None and not evcal.is_clean(d1, "gap", families):
            continue
        got = _pooled_cycle(d0, d1, chain, loud=loud)
        if got is None:
            continue
        fine.append(got["fine"])
        coarse.append(got["coarse"])
        cov.append(got["cov"])
        kept.append((d0, d1))
    if not fine:
        raise RuntimeError("no usable cycles -- is the ES/NQ cache built?")
    f = varcal.winsor_mean(np.vstack(fine))
    c = varcal.winsor_mean(np.vstack(coarse))
    shares = _rescale_to_blocks(f, c)
    shares, halt_report = _apply_halt_cap(shares, verbose=verbose)
    if verbose:
        print(f"[gapcal] {len(kept)} cycles of {len(cycles)} cached, "
              f"mean bucket coverage {np.mean(cov):.1%}")
        for d, u, share, i in sorted(loud, key=lambda z: -z[2])[:5]:
            print(f"[gapcal]   concentrated: {d} {u} bucket {minute_labels()[i]} "
                  f"holds {share:.1%} (kept -- see WARN_BUCKET_SHARE)")
    meta = {
        "cycles": len(kept),
        "cycles_available": len(cycles),
        "first_cycle": str(kept[0][0]),
        "last_cycle": str(kept[-1][1]),
        "mean_coverage": float(np.mean(cov)),
        "event_exclusion": bool(evcal is not None),
        "halt": halt_report,
        # Recomputed FROM the shipped shares, not the coarse fit, because `_apply_halt_cap` runs
        # after the rescale and would otherwise break the nesting identity the validator asserts.
        # The pre-cap coarse fit is kept alongside so the cap's effect stays inspectable.
        "block_shares": [float(shares[k * FINE_PER_BLOCK:(k + 1) * FINE_PER_BLOCK].sum())
                         for k in range(N_BLOCKS)],
        "block_shares_precap": [float(x) for x in c],
        "concentrated": len(loud),
    }
    return shares, meta


# --- artifact ------------------------------------------------------------

def gap_path(underlying, profile="default"):
    stem = underlying if profile == "default" else f"{underlying}.{profile}"
    return config.MODEL_DIR / "variance_calendar" / f"gap_{stem}.json"


def build(years=DEFAULT_YEARS, end=None, exclude_events=True, families=None,
          profile="default", verbose=True):
    """Fit once and write BOTH underlyings' artifacts.

    Like varcal, the shape is pooled and the two files carry identical `shares`/`cum`, differing
    only in provenance -- so every consumer's `gapcal.load(underlying)` keeps working and nothing
    has to know that the fit was pooled.
    """
    shares, meta = fit(years=years, end=end, exclude_events=exclude_events,
                       families=families, verbose=verbose)
    cum = np.concatenate([[0.0], np.cumsum(shares)])
    cum[-1] = 1.0                              # pin against float drift, as varcal does
    written = []
    for u in UNDERLYINGS:
        art = {
            "underlying": u,
            "underlyings": list(UNDERLYINGS),
            "sources": SOURCES,
            "estimator": "per-cycle-fraction/v1",
            "grid_seconds": {"fine": FINE_WIDTH, "coarse": COARSE_WIDTH},
            "gap_minutes": GAP_MINUTES,
            "winsor_pct": varcal.WINSOR_PCT,
            "min_covered": MIN_COVERED,
            "profile": profile,
            "built_utc": dt.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "labels": minute_labels(),
            "shares": [float(x) for x in shares],
            "cum": [float(x) for x in cum],
            **meta,
        }
        p = gap_path(u, profile)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(art))
        tmp.replace(p)
        written.append(p)
    if verbose:
        for p in written:
            print(f"[gapcal] wrote {p}")
    return shares, meta


class GapCalendar:
    """The overnight shape. `phi_gap` is cumulative share of the GAP's variance, not of a session.

    Units are deliberately NOT the same as varcal's. varcal's `phi` is a share of one RTH
    session; this is a share of one overnight gap, and it becomes commensurable with varcal only
    after multiplication by `varcal`'s `overnight_share`. `variance_time` in rvol/features does
    exactly that and is the only place the two are combined.
    """

    def __init__(self, art):
        self.underlying = art["underlying"]
        self.shares = art["shares"]
        self.cum = art["cum"]
        self.profile = art.get("profile", "default")
        self.built_utc = art.get("built_utc")
        self.cycles = art.get("cycles")
        self.estimator = art.get("estimator")
        if len(self.cum) != GAP_MINUTES + 1:
            raise ValueError(f"gap calendar has {len(self.cum)} cum points, "
                             f"want {GAP_MINUTES + 1}")

    @classmethod
    def load(cls, underlying, profile="default"):
        p = gap_path(underlying, profile)
        if not p.exists():
            raise FileNotFoundError(
                f"no gap calendar for {underlying} ({profile}) at {p} -- run: "
                f"python -m pricing.gapcal --build")
        return cls(json.loads(p.read_text()))

    def minutes_into_gap(self, t):
        """Minutes from the preceding 16:00 to ET datetime `t`, or None if `t` is inside RTH.

        The gap that contains an instant is the one STARTING at the most recent 16:00, so a
        02:00 timestamp belongs to the gap that opened at 16:00 the previous calendar day.
        """
        et = t.astimezone(config.ET)
        anchor = dt.combine(et.date(), GAP_START_T, tzinfo=config.ET)
        if et < anchor:
            anchor -= timedelta(days=1)
        mins = (et - anchor).total_seconds() / 60.0
        if mins < 0 or mins > GAP_MINUTES:
            return None                        # inside RTH: varcal's phi owns this instant
        return mins

    def phi_gap(self, t):
        """Cumulative share of the gap's variance realised by ET datetime `t`, in [0, 1].

        Linear within a bucket, exactly as varcal.phi interpolates within a minute.
        """
        mins = self.minutes_into_gap(t)
        if mins is None:
            et = t.astimezone(config.ET)
            return 1.0 if et.time() >= GAP_END_T and et.time() < GAP_START_T else 0.0
        if mins <= 0:
            return 0.0
        if mins >= GAP_MINUTES:
            return 1.0
        i = int(mins)
        frac = mins - i
        return self.cum[i] + frac * (self.cum[i + 1] - self.cum[i])

    def share_between(self, t0, t1):
        """Share of the gap's variance falling in [t0, t1). Both must be in the same gap."""
        return max(0.0, self.phi_gap(t1) - self.phi_gap(t0))

    def hour_shares(self):
        """[(label, minutes, share)] by clock hour -- the shape as a human reads it."""
        out = []
        labs = minute_labels()
        start = 0
        for i in range(1, GAP_MINUTES + 1):
            if i == GAP_MINUTES or labs[i].endswith(":00"):
                out.append((labs[start], i - start, float(sum(self.shares[start:i]))))
                start = i
        return out


def load(underlying, profile="default"):
    return GapCalendar.load(underlying, profile)


# --- cli -----------------------------------------------------------------

def _show(underlying, profile="default"):
    cal = GapCalendar.load(underlying, profile)
    flat = 1.0 / GAP_MINUTES
    print(f"gap calendar {underlying} ({cal.profile})  {cal.cycles} cycles, "
          f"estimator {cal.estimator}, built {cal.built_utc}")
    print(f"\n{'hour':>7} {'mins':>5} {'share':>8} {'x flat minute':>14}")
    for lab, n, sh in cal.hour_shares():
        print(f"{lab:>7} {n:5d} {sh:8.4f} {sh / (n * flat):14.2f}")
    top = sorted(range(GAP_MINUTES), key=lambda i: -cal.shares[i])[:12]
    labs = minute_labels()
    print(f"\nbusiest 12 minutes of the night:")
    for i in top:
        print(f"  {labs[i]}  {cal.shares[i]:.5f}  ({cal.shares[i] / flat:5.2f}x a flat minute)")


def open_staleness(underlying, cycles=None, minutes=5):
    """(gap to first print, gap to first print + `minutes`, ratio) -- mean squared log moves.

    A ratio near 1.0 means the index's opening print is a real level. Materially above 1.0 means
    it is stale: part of the overnight move has not been printed yet and will arrive over the
    following minutes, so anything anchored on that print understates the gap. See the module
    docstring for the measured SPX/NDX split.
    """
    import pyarrow.parquet as pq
    idx_tk = {"SPX": "I:SPX", "NDX": "I:NDX"}[underlying]
    cycles = cycles if cycles is not None else cached_cycles()
    a, b = [], []
    for d0, d1 in cycles:
        p0, p1 = _cache_path(idx_tk, d0), _cache_path(idx_tk, d1)
        if not (p0.exists() and p1.exists()):
            continue
        t0 = pq.read_table(p0); t1 = pq.read_table(p1)
        c0 = np.asarray(t0.column("close"), dtype="float64")
        ts1 = np.asarray(t1.column("t_ms"), dtype="int64")
        c1 = np.asarray(t1.column("close"), dtype="float64")
        if c0.size == 0 or c1.size == 0 or c0[-1] <= 0 or c1[0] <= 0:
            continue
        tgt = int((dt.combine(d1, GAP_END_T, tzinfo=config.ET)
                   + timedelta(minutes=minutes)).timestamp() * 1000)
        i = np.searchsorted(ts1, tgt, side="right") - 1
        if i < 0 or c1[i] <= 0:
            continue
        a.append(math.log(c1[0] / c0[-1]) ** 2)
        b.append(math.log(c1[i] / c0[-1]) ** 2)
    if not a:
        return None
    ma, mb = float(np.mean(a)), float(np.mean(b))
    return ma, mb, (mb / ma if ma > 0 else float("nan"))


def _check(years=DEFAULT_YEARS):
    """Total measured gap variance vs the index's own close-to-open move. Should be ~1.0.

    The only out-of-sample statement this calendar can make about itself: the shape is fitted on
    futures and ETF returns, while the index close-to-open gap is a separate measurement on a
    different instrument. Under a martingale the mean squared gap return equals the mean
    integrated variance over the same window, so the ratio is a real test rather than a
    restatement of the fit.
    """
    import pyarrow.parquet as pq
    chain = load_chain()
    cycles = cached_cycles(years=years)
    print(f"[gapcal] level check over {len(cycles)} cached cycles")
    for und in UNDERLYINGS:
        idx_tk = {"SPX": "I:SPX", "NDX": "I:NDX"}[und]
        meas, gaps = [], []
        for d0, d1 in cycles:
            got = _cycle_returns(und, d0, d1, chain, scale_s=FINE_WIDTH)
            if got is None:
                continue
            r = got[0][np.isfinite(got[0])]
            p0, p1 = _cache_path(idx_tk, d0), _cache_path(idx_tk, d1)
            if r.size == 0 or not (p0.exists() and p1.exists()):
                continue
            c0 = np.asarray(pq.read_table(p0).column("close"), dtype="float64")
            c1 = np.asarray(pq.read_table(p1).column("close"), dtype="float64")
            if c0.size == 0 or c1.size == 0 or c0[-1] <= 0 or c1[0] <= 0:
                continue
            meas.append(float((r ** 2).sum()))
            gaps.append(float(np.log(c1[0] / c0[-1]) ** 2))
        if not meas:
            print(f"  {und}: no usable cycles")
            continue
        m, g = float(np.mean(meas)), float(np.mean(gaps))
        st = open_staleness(und, cycles)
        print(f"  {und}: {len(meas)} cycles   measured {m:.4e}   index gap^2 {g:.4e}   "
              f"ratio {m / g:.3f}")
        if st:
            ma, mb, ratio = st
            verdict = ("index open is STALE, so the gap it anchors is biased LOW and a high "
                       "ratio above is EXPECTED" if ratio > 1.1 else "index open is clean")
            print(f"        open staleness: gap to 09:30 {ma:.4e}, to 09:35 {mb:.4e} "
                  f"({ratio:.2f}x) -- {verdict}")


def _compare(years=DEFAULT_YEARS):
    """The signature plot on the real tape, per underlying. Decides whether 60s is fine enough."""
    cycles = cached_cycles(years=years)[-150:]
    print(f"[gapcal] signature plot over {len(cycles)} cycles -- if RV keeps falling as the scale")
    print("         coarsens, microstructure noise is being read as variance")
    for und in UNDERLYINGS:
        sp = signature_plot(und, cycles)
        base = sp.get(900) or float("nan")
        print(f"\n  {und}:")
        for sc in sorted(sp):
            print(f"    {sc:4d}s  mean gap RV {sp[sc]:.4e}   x{sp[sc] / base:5.3f} of the 900s reading")


def main(argv=None):
    ap = argparse.ArgumentParser(description="overnight (16:00-09:30) variance calendar")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--check", action="store_true", help="measured gap variance vs the index gap")
    ap.add_argument("--compare", action="store_true", help="signature plot by sampling scale")
    ap.add_argument("--underlying", default="SPX")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--years", type=float, default=DEFAULT_YEARS)
    ap.add_argument("--no-exclude-events", action="store_true")
    a = ap.parse_args(argv)
    if a.build:
        build(years=a.years, exclude_events=not a.no_exclude_events, profile=a.profile)
    if a.compare:
        _compare(years=a.years)
    if a.check:
        _check(years=a.years)
    if a.show or not (a.build or a.check or a.compare):
        _show(a.underlying, a.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
