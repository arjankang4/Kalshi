"""Intraday variance calendar: how a trading session's variance is distributed across its minutes.

The hourly Kalshi boards need the variance between *now* and a sub-4pm settlement, but the only
vol observable is backed out of options that expire at 16:00. Splitting one into the other by
clock time assumes variance accrues at a constant rate, and it does not: the first and last half
hours carry far more of it than the middle of the day. Quoting a 12:00 market off a flat split
overstates the variance left in it, and a 15:30 market understates it.

So: measure the shape once from two years of 1-second index prints, store it, and scale by ratio.

    w_h^2 = w_4pm^2 * (Phi(h) - Phi(t)) / (1 - Phi(t))

`Phi` is the cumulative share of the session's variance realised by time t, and `w` is total
stdev (see pricing/bs). Two consequences of writing it as a ratio, both load-bearing:

  * **Overnight variance never enters.** An option quoted at time t already prices only the
    variance from t to its expiry, so the overnight gap is outside both the numerator and the
    denominator. There is no ~70% intraday factor to apply and no risk of applying it twice.
    `overnight_share` is still fitted and stored, because a calendar that cannot say what the gap
    is worth is not much of a calendar, but the hourly path does not read it.
  * **No annualisation, no day count.** Nothing is ever divided by 252 or 365 and multiplied back.

The invariant that falls out: at h == 16:00 the ratio is exactly 1, so the hourly model reproduces
the 4pm vol it was given. pricing/validate_hourly asserts it.

ONE CALENDAR PER INDEX. `SPX.json` and `NDX.json` are fitted separately, each on its own index,
and their `shares` DIFFER -- median ~10% per bucket, up to 80% at the ends. Pooling the
two into a single shared shape was specified, implemented, measured and dropped; see UNDERLYINGS
for the numbers that killed it. Do not read a divergence between the two artifacts as a bug.

The estimator is documented at length under FIT below; the short version is that each day
contributes one vector of fractions summing to 1 and days are averaged with equal weight, rather
than squared returns being averaged and normalised once at the end. That single change, plus
fitting the gap on single nights only, moved the overnight gap from 33.4% (SPX) / 44.8% (NDX) of
the 24h cycle to 19.9% / 24.7%.

BUILD. `--build` writes BOTH underlyings, and with no `--profile` it writes the artifacts the
live pricer loads. It refuses to overwrite an existing default-profile calendar without
`--force`: there is no output-directory flag, nothing schedules a rebuild, and the sample window
slides with today's date, so an unguarded rebuild silently replaces a fitted artifact that
nothing else can reconstruct. To build somewhere else, set `config.MODEL_DIR` before importing
this module and copy `event_calendar.json` into it (pricing/eventcal resolves from the same var).

    python -m pricing.varcal --build --exclude-events --profile shared   # a candidate
    python -m pricing.varcal --compare --underlying SPX   # every variant, one read of the cache
    python -m pricing.varcal --show --underlying SPX
"""
import argparse
import json
import math
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone

import numpy as np

import config
from pricing import eventcal, varbars

# Buckets per RTH session. 390 of them are the clock minutes 09:30..15:59; the 391st, labelled
# '16:00', is the settlement convergence -- see pricing/varbars for what it measures and why.
SESSION_MINUTES = 391
OPEN_T = dtime(9, 30)
CLOSE_T = dtime(16, 0)

# The two grids. The fine one carries the shape, the coarse one carries the level of each block;
# see FIT below. 300 must be a whole multiple of 60 or the blocks do not nest.
FINE_WIDTH = 60
COARSE_WIDTH = 300
FINE_PER_BLOCK = COARSE_WIDTH // FINE_WIDTH             # 5
N_BLOCKS = (SESSION_MINUTES - 1) // FINE_PER_BLOCK      # 78, excluding the terminal bucket

# The underlying whose bars the shape is fitted from: the real settlement index, both sides.
#
# SPX used SPY until 2026-08-21, because I:SPX was NOT_AUTHORIZED and I:GSPC returns empty. The
# substitution was defensible for a *shape* (returns are ratio-invariant to a smooth dividend
# accrual) and never for a level. With the Index entitlement it is simply unnecessary, and
# removing it measurably changed the answer.
#
# Refit over the same 2 years / ~497 sessions, full pipeline (winsoriser + recency ramp):
#
#                       SPY        I:SPX
#     overnight_share   0.743      0.543
#     of a 24h cycle    42.6%      35.2%      (raw, untreated: 31.9%)
#
# SPY's close-to-open gap carries ETF bid-ask bounce, premium/discount swing and four dividend
# ex-dates a year that the index does not have, and all of it was being read as overnight
# variance. The intraday shape moved only at the two ends: the 09:30-10:00 block went 1.97x ->
# 2.34x a flat minute, the mid-session moved under 8%, and the 16:00 bucket went 1.88x -> 0.31x.
#
# That last one looks alarming and is not. The 15:59 MOC ramp -- the feature PROTECT_TAIL exists
# to preserve -- survives in both (SPY 2.69x, I:SPX 2.13x, I:NDX 1.79x). What an index lacks is
# the 16:00 auction *print*, because a computed index has no auction trade; its close is a
# restatement. I:NDX has always behaved this way and has always been the NDX source, so this
# makes SPX consistent with NDX rather than introducing anything new. The last six minutes carry
# 1.50% of the session on I:SPX against a 1.53% flat baseline -- the tail is not flattened.
SOURCE_TICKER = {"SPX": "I:SPX", "NDX": "I:NDX"}

DEFAULT_YEARS = 2

# What `--build` writes. Each is fitted on ITS OWN index, not on a pooled sample.
#
# Pooling the two was specified, built, and then measured and dropped. The idea was sound -- SPX
# and NDX are ~0.9 correlated daily, so averaging two estimates of a shared shape cuts sampling
# noise, and "two years x two indices" reads like four years of data. It is not: the sample is
# ~478 CYCLES, not 956 independent ones, and the noise reduction is nowhere near sqrt(2).
#
# What settled it AT THE TIME was `pricing/validate_varcal` scored against real option boards:
# pooling helps SPX (1.099 -> 1.073) and hurts NDX (1.170 -> 1.199). That gate no longer decides
# anything (it measured FLATNESS against the option market, which was never the right criterion --
# see its docstring), so the decision is restated on the evidence that does stand: NDX's shape is
# genuinely its own and blending imports SPX's. There is no version of the trade where NDX comes
# out ahead on ANY instrument, so the two indices keep their own calendars.
#
# Worth re-measuring against `rvol/accuracy.calendar_test` (realised index variance) before anyone
# revisits pooling. The conclusion is unlikely to move; the reason for it should not rest on a
# retired gate.
UNDERLYINGS = ("SPX", "NDX")

# Winsorise each bucket's series of DAILY FRACTIONS at its own 99th percentile, clipping only
# from above. Fractions are bounded below by zero, so there is nothing to clip at the other end.
#
# This is not the old winsoriser relocated, and the objection recorded against that one does not
# transfer. The old cap was a MAD-derived *scale* applied to raw returns, and a per-bucket
# version of it was measured to destroy 63% of the I:NDX closing minute's variance: in a bucket
# whose returns are mostly ~0, median|r| collapses and takes the cap down with it. A percentile
# of the daily fractions is an order statistic, not a scale estimate. It cannot collapse -- it
# always sits at the 99th of ~479 real observations -- and it clips roughly five days per bucket.
#
# Do not expect this to be the lever on any headline number. Measured on the overnight bucket it
# moves the answer by 0.06pp (24.99% -> 24.93% of the cycle). Per-day normalisation is what
# changed the estimate; this only stops one day from owning one bucket.
WINSOR_PCT = 99.0

# Report-only. A day whose busiest single minute holds more than this much of its own intraday
# variance is printed by the build; it is NOT dropped.
#
# This started life as a tripwire against a broken vendor print, on the theory that per-day
# normalisation makes one bad tick dangerous -- it inflates that day's DENOMINATOR and depresses
# every other bucket's fraction, which a per-bucket p99 clip cannot see. Both halves of that
# theory turned out to be wrong, and the correction is worth keeping written down.
#
# There is no bad-print population in this data. The max-bucket-share distribution over 495
# sessions is smooth and unimodal -- median 7.1%, p95 19.7%, p99 35.7%, max 54.5% -- with no gap
# anywhere to separate "glitch" from "market". Every session at the top is an identifiable event
# that shows up in BOTH indices in the SAME minute and persists rather than reverting:
# 2025-08-22 10:00 Jackson Hole (SPX 54.5%, NDX 45.4%), 2024-09-18 14:00 FOMC (31.2%, 27.1%),
# 2025-04-09 13:19 the tariff pause (28.4%, 26.1%). A cap at 0.5 deleted exactly one session,
# and it was the most informative one in the sample.
#
# And the danger was overstated to begin with: per-day normalisation *is* the protection. Every
# session contributes exactly 1/n to every bucket's mean no matter how violent it was, so a
# wholly corrupted day can move a bucket by 0.2%. That bounded influence is the property the old
# variance-weighted estimator lacked and the reason it needed an 8-sigma return cap at all.
WARN_BUCKET_SHARE = 0.35


# Named fits of the same session. The default profile is the one the pricer loads; `fomc` is a
# separate shape fitted on FOMC days alone, because those days are not a noisy version of a normal
# day -- they are a different shape (see PROFILES below).
#
# Measured, SPX, 2y, x a flat minute:
#
#     block      clean(388)   FOMC(16)   ratio
#     09:30         2.48        1.00      0.42x   <- the market WAITS
#     12:00         0.71        0.26      0.35x
#     14:30         0.53        3.09      5.83x   <- and then it does not
#     15:30         0.81        2.38      2.94x
#
# Total intraday variance is only 1.20x a clean day: FOMC does not add much variance, it RELOCATES
# it from the morning to the afternoon. An unconditional average of the two is wrong for both --
# it over-weights the afternoon on the 97% of days that are clean and under-weights it ~2.7x at
# 14:00 on the 3% that are not. 16 sessions is a small sample and the multipliers are not
# calibrated, but a 5.8x afternoon is not noise.
# `prepromote` is the variance-weighted calendar that was live until 2026-08-23, kept
# scoreable so `validate_varcal` can still compare what shipped against what it replaced.
# `shared` is the CANDIDATE STAGING SLOT -- `build` refuses to overwrite the live default and
# points you here (see the FileExistsError below). It is deliberately EMPTY: the 2026-08-23
# candidate that once sat here was promoted by byte-copy, a comment then called it "a duplicate
# of default", and that stopped being true the moment the all-gaps calendar shipped on
# 2026-08-24 without touching it. A stale artifact in the staging slot reads as a control arm
# and is not one. It was moved to var/model/box-archive/, and that archive was deleted in the
# 2026-09-23 cleanup; `git show a394055:var/model/box-archive/...` still has it.
# `evclean` is written by `pricing/study_event_variance`, not by `build`: every modelled release
# (BLS, claims, FOMC) is SUBTRACTED as a fitted excess-variance kernel rather than excluded. Its
# per-session companion is `event_aware` below.
PROFILES = ("default", "fomc", "shared", "prepromote", "evclean")


def profiles_on_disk(underlying):
    """Every profile with a file for `underlying`, including candidates `build` does not write --
    e.g. `pricing/blendcal`'s `blend`. PROFILES above is the set THIS module builds; the gates must
    be able to score anything that could be promoted."""
    d = config.MODEL_DIR / "variance_calendar"
    out = []
    for p in sorted(d.glob(f"{underlying}*.json")):
        stem = p.name[:-5]
        if stem == underlying:
            out.append("default")
        elif stem.startswith(underlying + "."):
            out.append(stem[len(underlying) + 1:])
    return out


def calendar_path(underlying, profile="default"):
    stem = underlying if profile == "default" else f"{underlying}.{profile}"
    return config.MODEL_DIR / "variance_calendar" / f"{stem}.json"


def minute_labels():
    """The 391 bucket labels, 'HH:MM', bucket i = the move during clock minute 09:30+i."""
    base = datetime.combine(date(2000, 1, 3), OPEN_T)
    return [(base + timedelta(minutes=i)).strftime("%H:%M") for i in range(SESSION_MINUTES)]


# --- fitting -------------------------------------------------------------
#
# THE ESTIMATOR, AND WHY IT CHANGED
#
# The previous fit averaged SQUARED RETURNS per bucket across sessions and normalised once at
# the end. That is a variance-weighted average of days: a handful of violent sessions set the
# shape, and the overnight bucket -- whose distribution across days is savagely right-skewed --
# was set almost entirely by its own tail. It read `overnight_share` at 0.501 for SPX and 0.811
# for NDX, i.e. 33.4% and 44.8% of the 24h cycle, against a per-day MEDIAN of 16.9% and 22.9%
# and a p90 of 58% and 71%.
#
# This fit normalises PER DAY first. Each session contributes one vector of fractions summing to
# 1 and days are then averaged with EQUAL WEIGHT. Note what did and did not do the work on the
# overnight number: the p99 winsoriser moves it by 0.06pp. Per-day normalisation, plus fitting
# the gap on single nights only, moves it from 0.811 to 0.328 on NDX.
#
# This is a DIFFERENT ESTIMATOR, not a bug fix. E[r^2/sum r^2] != E[r^2]/E[sum r^2].
#
# THE COST OF EQUAL WEIGHTING, MEASURED. Arjan's call, made with this number in front of him.
#
# The intraday shape is not independent of the day's volatility. Splitting the 478 sessions at
# the median of their own total variance:
#
#     block          quiet half   loud half   ratio    corr(fraction, log total variance)
#     09:30-10:00      0.2412      0.1935     1.25x          -0.357
#     12:00-14:00      0.1847      0.2127     0.87x          +0.257
#     15:59            0.0080      0.0057     1.40x          -0.118
#
# On a quiet day the open and the MOC ramp own a LARGER FRACTION of a smaller session; on a
# violent day variance spreads into the middle. Equal-weighting days therefore averages toward the
# quiet-day shape. The old option-board flatness gate scored that as a cost:
#
#     shape source     day weighting     SPX      NDX      (flatness, RETIRED as a criterion)
#     shipped (old)    variance          1.141    1.161
#     per-underlying   EQUAL (shipped)   1.199    1.172
#     per-underlying   variance          1.099    1.170
#     pooled SPX+NDX   variance          1.073    1.199
#
# READ THAT TABLE THE OTHER WAY NOW. Equal weighting shipped despite being ~0.06 "worse" on it,
# and the criterion has since been retired: it measured whether the recovered vol curve was FLAT,
# and implied vol is not flat through a session for reasons that have nothing to do with us. The
# table is kept as the record of a knob that was measured and declined -- not as a cost. What
# equal weighting actually buys is on the split-half evidence above: every day contributes 1/n
# regardless of how violent it was, which is the protection the old variance-weighted estimator
# lacked (it read the tail of a savagely right-skewed distribution and produced overnight_share
# 0.811 on NDX).
#
# Do not switch the weighting without saying so.
#
# ONE CALENDAR PER UNDERLYING. Pooling SPX and NDX was specified and then measured and dropped:
# it improves SPX (1.099 -> 1.073) and degrades NDX (1.170 -> 1.199), because NDX's shape is
# genuinely its own and blending imports SPX's. The regularisation argument for pooling is real
# -- averaging two correlated estimates cuts sampling noise -- it simply does not pay for NDX.
#
# TWO ESTIMATORS, TWO DENOMINATORS
#
# Normalising each day over `intraday + overnight` in one denominator would make component-wise
# event exclusion incoherent. `pricing/eventcal` can say that a session's INTRADAY is clean while
# the gap out of it is contaminated (an after-the-close mega-cap earnings release), and dropping
# only the gap is the right response -- but under one shared denominator the "clean" intraday
# fractions would still be depressed by the contaminated gap sitting underneath them.
#
# So the shape and the gap are fitted separately, each with the exclusion that suits it:
#
#     shape     f_b = r_b^2 / sum_intraday r^2          ~478 sessions (intraday clean)
#     overnight f   = r_on^2 / (I + r_on^2)             ~286 single nights (both ends clean)
#
# A day's intraday shape does not depend on how large its overnight gap happened to be, which is
# the right invariance and is also what makes the multi-day-gap problem (see `_select_gaps`)
# confined to the overnight statistic instead of leaking into the shape.
#
# TWO GRIDS
#
# The 1-minute estimate of a single bucket from ~478 samples is noisy -- adjacent minutes differ
# by a median 12% and a p95 of 64%. The 5-minute estimate of a block is not. So the shape is cut
# twice from the same 1-second SOURCE -- once on a 1-MINUTE grid and once on a 5-minute one,
# and never at 1 second (see varbars.grid_returns: last print at or before each edge, then
# log-difference) -- and the fine grid is rescaled to agree with the coarse one
# block by block: the coarse fit supplies the LEVEL of each 5 minutes, the fine fit supplies only
# the shape WITHIN it. That is the whole of the smoothing, and it replaces the centred moving
# median the previous fit used.
#
# Dropping that median was not cosmetic. It was 15 buckets wide and it was erasing a real
# feature: the 15:50 NYSE MOC imbalance publication reads ~3.5x a flat minute in this fit and was
# smoothed to 0.74x in the shipped calendar. The 15:59 ramp likewise strengthens from 2.38x to
# ~2.9x. A fixed-width median cannot tell a spike from noise; the coarse grid can, because a real
# spike survives 5-minute aggregation and noise does not.
#
# Not every spike is real, and the fit does not pretend otherwise: `compare` prints a split-half
# column. 15:50 and 15:59 replicate across halves and have mechanisms. 15:54 reads 2.40x in the
# first half and 1.07x in the second, and should not be believed.


def _day_fractions(u, d, loud=None):
    """One (underlying, date) -> per-grid fraction vectors, intraday totals, open/settle prices.

    Fractions are normalised over the INTRADAY buckets only, which is what keeps the shape
    independent of that day's overnight gap. `None` when the session is unusable.
    """
    loaded = varbars.load_session(varbars.TICKER[u], d)
    if loaded is None:
        return None
    es, close = loaded
    fracs, totals = {}, {}
    for w in (FINE_WIDTH, COARSE_WIDTH):
        rets, open_px, settle_px = varbars.grid_returns(es, close, w)
        sq = rets * rets
        tot = float(sq.sum())
        if tot <= 0:
            return None
        fracs[w], totals[w] = sq / tot, tot
    # Reported, never dropped -- see WARN_BUCKET_SHARE for why the drop was removed.
    if loud is not None and fracs[FINE_WIDTH].max() > WARN_BUCKET_SHARE:
        loud.append((d, u, float(fracs[FINE_WIDTH].max()), int(fracs[FINE_WIDTH].argmax())))
    return {"fracs": fracs, "totals": totals, "open": open_px, "settle": settle_px}


def gather(years=DEFAULT_YEARS, end=None, verbose=True):
    """Decompose the sample once, per underlying: per-day fractions and per-cycle overnight.

    Separated from `fit` so the exclusion study can refit the same sample many ways without
    re-reading two years of Parquet. Reads local disk only -- no vendor call -- which is why
    `compare` is cheap enough to be the default way of deciding anything here.

    Each overnight cycle is keyed by BOTH dates: D because its intraday variance is the
    denominator, and N because `pricing/eventcal` keys a gap by the session it lands on. `fit`
    needs both to apply exclusion correctly. N is always the immediately following cached date,
    so a gap can never span a session that was dropped -- the discipline the old `prev_close =
    None` reset enforced -- and its calendar length is recorded so `_select_gaps` can insist on a
    single night.
    """
    end = end or config.today_et()
    start = end - timedelta(days=int(365.25 * years))
    loud, out = [], {}
    for u in UNDERLYINGS:
        cached = [d for d in varbars.cached_dates(varbars.TICKER[u]) if start <= d <= end]
        if not cached:
            raise FileNotFoundError(
                f"no cached 1-second bars for {u} ({varbars.TICKER[u]}) under {varbars.CACHE} -- "
                f"run: python -m rvol.data --build")
        days = {d: _day_fractions(u, d, loud) for d in cached}
        dates = [d for d in cached if days[d] is not None]

        gap_dates, gap_span, gap_frac = [], [], {FINE_WIDTH: [], COARSE_WIDTH: []}
        for i, d in enumerate(cached[:-1]):
            nxt = cached[i + 1]
            if days[d] is None or days[nxt] is None:
                continue
            on = math.log(days[nxt]["open"] / days[d]["settle"])
            gap_dates.append((d, nxt))
            gap_span.append((nxt - d).days)
            for w in (FINE_WIDTH, COARSE_WIDTH):
                gap_frac[w].append(on * on / (days[d]["totals"][w] + on * on))

        out[u] = {
            "dates": dates,
            "shape": {w: np.array([days[d]["fracs"][w] for d in dates])
                      for w in (FINE_WIDTH, COARSE_WIDTH)},
            "gap_dates": gap_dates,
            "gap_span": np.array(gap_span),
            "gap_frac": {w: np.array(v) for w, v in gap_frac.items()},
            "dropped": len(cached) - len(dates),
            "cached": len(cached),
        }
        if verbose:
            print(f"[varcal] {u} ({varbars.TICKER[u]}): {len(cached)} cached dates "
                  f"{start} -> {end}, {len(dates)} usable, {out[u]['dropped']} dropped "
                  f"(half days, holidays, truncated pulls), {len(gap_dates)} cycles")
    if verbose:
        labs = minute_labels()
        for d, u, sh, i in sorted(loud, key=lambda r: -r[2])[:5]:
            print(f"[varcal]   concentrated: {d} {u} {labs[i]} held {sh:.1%} of the session "
                  f"(kept -- see WARN_BUCKET_SHARE)")
    out["concentrated"] = loud
    return out


def winsor_mean(F):
    """Column-wise p99 clip, then plain mean, renormalised to sum 1.

    `F` is (days, buckets) of per-day fractions. Clipping is one-sided because a fraction cannot
    go below zero, and it is per-bucket because that is where a single day can dominate. The mean
    is unweighted: see the EQUAL WEIGHTING note above for what that costs and why it stands.
    """
    if len(F) == 0:
        raise RuntimeError("no sessions survived selection")
    cap = np.percentile(F, WINSOR_PCT, axis=0)
    m = np.minimum(F, cap).mean(axis=0)
    total = m.sum()
    if total <= 0:
        raise RuntimeError("degenerate calendar: total variance is zero")
    return m / total


def rescale_to_blocks(fine, coarse):
    """Rescale each 5-minute block of `fine` to carry the total `coarse` gives it.

    The terminal bucket is its own singleton block in both grids, so it passes through as the
    coarse grid's value. Afterwards, for every block k:

        sum(out[5k : 5k+5]) == coarse[k]

    which `pricing/validate_varcal.structural_checks` asserts on the shipped artifact to 1e-12.
    That identity is the point: it makes the coarse grid the authority on levels and the fine
    grid the authority on shape within a block.
    """
    if len(fine) != SESSION_MINUTES:
        raise ValueError(f"fine grid has {len(fine)} buckets, want {SESSION_MINUTES}")
    if len(coarse) != N_BLOCKS + 1:
        raise ValueError(f"coarse grid has {len(coarse)} blocks, want {N_BLOCKS + 1}")
    out = np.array(fine, dtype="float64")
    for k in range(N_BLOCKS):
        sl = slice(k * FINE_PER_BLOCK, (k + 1) * FINE_PER_BLOCK)
        tot = out[sl].sum()
        # A block with no observed variance at all cannot be shaped, so it is spread flat rather
        # than left at zero -- zero would put a flat step into phi that theta reads across.
        out[sl] = out[sl] * (coarse[k] / tot) if tot > 0 else coarse[k] / FINE_PER_BLOCK
    out[-1] = coarse[-1]
    return out / out.sum()


def _select_shape(sub, evcal, families, whole_day):
    """Row indices of `sub['shape']` whose session is usable for the intraday shape."""
    if evcal is None:
        return list(range(len(sub["dates"])))
    gap_bad = set()
    if whole_day:
        gap_bad = {d for d, nxt in sub["gap_dates"] if not evcal.is_clean(nxt, "gap", families)}
    return [i for i, d in enumerate(sub["dates"])
            if evcal.is_clean(d, "intraday", families) and d not in gap_bad]


def _select_gaps(sub, evcal, families, span=1):
    """Row indices of `sub['gap_frac']` usable for the overnight statistic.

    Both ends matter for exclusion: D supplies the denominator so its intraday must be clean, and
    the gap itself lands on N, which is where `pricing/eventcal` records an 08:30 CPI print or an
    after-the-close earnings release.

    `span` filters on the gap's CALENDAR length in days and defaults to 1 -- a genuine overnight.
    That filter is not bookkeeping. 105 of 488 cycles in this sample span a weekend or a holiday
    (89 of them three days), and a weekend gap is measurably a different object: it carries
    ~1.4x the variance of a single overnight, enough to lift `overnight_share` from 0.287 to
    0.323 pooled.

    Which is right depends entirely on what reads the number, and the only consumer is
    `rvol/features.variance_time`, which spreads it uniformly over a hardcoded
    `GAP_SECONDS = 17.5 * 3600`. That denominator IS one night. Handing it a weekend-contaminated
    numerator over-states pre-open variance on every Tuesday through Friday, which is most of
    them. Pass `span=None` for every cycle regardless of length; `fit` reports both.

    (Worth noticing in passing: 3.4x the calendar time buys only ~1.4x the variance. Variance
    does not accrue while the market is shut, which is the same reason the hourly path scales by
    a pure ratio and never by clock time.)
    """
    idx = range(len(sub["gap_dates"]))
    if span is not None:
        idx = [i for i in idx if sub["gap_span"][i] == span]
    if evcal is None:
        return list(idx)
    return [i for i in idx
            if evcal.is_clean(sub["gap_dates"][i][0], "intraday", families)
            and evcal.is_clean(sub["gap_dates"][i][1], "gap", families)]


def fit(sub, evcal=None, families=None, whole_day=False, profile="default", verbose=True,
        gap_span=1):
    """One underlying's per-day fractions -> (shares, cum, overnight_share, meta).

    `shares` is the 391-bucket intraday calendar; `overnight_share` is the gap expressed as a
    multiple of one intraday session, the unit the artifact has always used and what
    `session_split` converts back to a share of the 24h cycle.

    `gap_span` selects the POPULATION the overnight statistic is fitted on, and it is a real choice
    rather than bookkeeping -- see `_select_gaps`. `1` is single overnights only; `None` pools every
    cycle including weekends and holidays, which is the population the conventional "25-30% of the
    cycle" figure is quoted on. Whichever is chosen, BOTH are reported, so the artifact always says
    what the other one would have been.
    """
    if profile == "fomc":
        # The complement of the default fit: FOMC sessions only, and no gaps, because an FOMC
        # decision says nothing about that morning's overnight gap. The shape on those days is
        # genuinely different rather than noisier -- 09:30 runs 1.00x against 2.48x on a clean
        # day and 14:30 runs 3.09x against 0.53x. The market waits, and then it does not.
        keep = [i for i, d in enumerate(sub["dates"])
                if "fomc" in evcal.families(d, "intraday")]
        gap_keep, gap_multi = [], []
    else:
        keep = _select_shape(sub, evcal, families, whole_day)
        gap_all = _select_gaps(sub, evcal, families, span=None)
        gap_single = _select_gaps(sub, evcal, families, span=1)
        gap_keep = gap_single if gap_span == 1 else gap_all
        # The complement, purely for reporting: whichever population was NOT fitted.
        gap_multi = (sorted(set(gap_all) - set(gap_single)) if gap_span == 1
                     else sorted(set(gap_single)))

    coarse = winsor_mean(sub["shape"][COARSE_WIDTH][keep])
    shares = rescale_to_blocks(winsor_mean(sub["shape"][FINE_WIDTH][keep]), coarse)

    cum = [0.0]
    for s in shares:
        cum.append(cum[-1] + float(s))
    cum[-1] = 1.0                            # pin the endpoint against float drift

    def _mean_frac(rows):
        a = sub["gap_frac"][COARSE_WIDTH][rows]
        return float(np.minimum(a, np.percentile(a, WINSOR_PCT)).mean())

    on, on_multi = 0.0, None
    if gap_multi:
        fm = _mean_frac(gap_multi)
        on_multi = fm / (1.0 - fm)
    if gap_keep:
        f_on = _mean_frac(gap_keep)
        on = f_on / (1.0 - f_on)

    if verbose:
        fitted = "single-night" if gap_span == 1 else "all-gap"
        other = "weekend/holiday" if gap_span == 1 else "single-night"
        print(f"[varcal]   shape on {len(keep)} sessions, overnight on {len(gap_keep)} "
              f"{fitted} cycles -> {on:.4f} ({on / (1 + on):.1%} of a 24h cycle)")
        if on_multi:
            print(f"[varcal]   the {len(gap_multi)} {other} cycles alone would give "
                  f"{on_multi:.4f} ({on_multi / on:.2f}x) -- not the fitted population")

    return [float(s) for s in shares], cum, on, {
        "block_shares": [float(v) for v in coarse],
        "shape_sessions": len(keep),
        "gap_cycles": len(gap_keep),
        "overnight_multiday": on_multi,
        "gap_cycles_multiday": len(gap_multi),
    }


def build(years=DEFAULT_YEARS, end=None, exclude_events=False, families=None,
          whole_day=False, sample=None, profile="default", underlyings=None, force=False,
          gap_span=1):
    """Fit and write one artifact per underlying. Each is fitted on its own index.

    Refuses to overwrite an existing DEFAULT-profile calendar unless `force=True`. That guard is
    not fussiness. There is no output-directory flag, so the command documented at the top of
    this file writes straight into what the quote loop loads; `end` defaults to today so the
    two-year window slides and a rebuild is never quite the artifact it replaced; this repo is
    not under version control, so the fitted file is the only copy; and `default` and `shared`
    differ in event exclusion, so a `--build` meant as a dry run produces a THIRD calendar that
    matches neither. Named profiles are unguarded -- they are the scratch space.
    """
    targets = list(underlyings or UNDERLYINGS)
    if profile == "default" and not force:
        live = [str(calendar_path(u, profile)) for u in targets
                if calendar_path(u, profile).exists()]
        if live:
            raise FileExistsError(
                "refusing to overwrite the live variance calendar:\n  "
                + "\n  ".join(live)
                + "\n\nBuild a candidate instead:  --build --profile shared\n"
                  "score it on REALISED index variance, which is what decides a promotion:\n"
                  "                              python -m rvol.accuracy --calendar-test\n"
                  "check it is not jagged:       python -m pricing.validate_varcal --profile shared\n"
                  "and promote deliberately.    Pass --force only if that is what you mean.\n\n"
                  "NOTE validate_varcal is a SMOOTHNESS check against the option board, not the\n"
                  "promotion gate. It used to assert flatness; that criterion was retired\n"
                  "2026-08-23 -- see its docstring.")
    if sample is None:
        sample = gather(years=years, end=end)

    evcal = None
    if exclude_events or profile == "fomc":
        evcal = eventcal.load()

    art = None
    for u in targets:
        sub = sample[u]
        shares, cum, on, meta = fit(sub, evcal, families, whole_day, profile,
                                    gap_span=gap_span)
        if profile == "fomc" and meta["shape_sessions"] < 30:
            print(f"[varcal]   WARNING: {meta['shape_sessions']} sessions is a small sample. The "
                  f"SHAPE is clearly different; the multipliers are not calibrated.")
        dates = sub["dates"]
        art = {
            "underlying": u,
            "source_ticker": SOURCE_TICKER[u],
            "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "estimator": "per-day-fraction/v2",
            "day_weighting": "equal",
            "grid_seconds": {"fine": FINE_WIDTH, "coarse": COARSE_WIDTH},
            "winsor_pct": WINSOR_PCT,
            "gap_population": ("single_night" if gap_span == 1 else "all_gaps"),
            "sessions": meta["shape_sessions"],
            "gap_cycles": meta["gap_cycles"],
            "dropped_sessions": sub["dropped"],
            "first_session": str(dates[0]) if dates else None,
            "last_session": str(dates[-1]) if dates else None,
            "overnight_share": on,
            "overnight_span_days": (1 if gap_span == 1 else "all"),
            "overnight_multiday": meta["overnight_multiday"],
            "labels": minute_labels(),
            "shares": shares,
            # The coarse fit, kept so the rescale identity stays checkable on the shipped file
            # rather than only at build time: sum(shares[5k:5k+5]) == block_shares[k].
            "block_shares": meta["block_shares"],
            "cum": cum,
            "event_exclusion": (None if not exclude_events else {
                "mode": "whole_day" if whole_day else "component",
                "families": sorted(families) if families else "all",
                "event_calendar_built": evcal.built_utc if evcal else None}),
            "profile": profile,
            # Each index is anchored at its own first print at or after 09:30:00 ET -- I:SPX
            # first prints at 09:30:01 and I:NDX at 09:30:15 -- while every bucket edge is wall
            # clock. So bucket 0 measures 59s for SPX and 45s for NDX, and the difference lands
            # in the overnight bucket. It is the one bucket whose window length is not identical
            # across underlyings, and it is also the busiest.
            "note_open_anchor": "wall-clock edges; bucket 0 starts at each index's first print",
        }
        p = calendar_path(u, profile)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(art))
        print(f"[varcal]   wrote {p}")
    return art


# --- serving -------------------------------------------------------------

class VarianceCalendar:
    """Read side: cumulative variance share through the session.

    Bucket i covers clock minute [09:30+i, 09:31+i), so `cum[i]` is the share realised as of
    exactly 09:30+i. The 16:00 bucket -- the closing auction -- is therefore realised *at* the
    4pm settlement rather than spread over the minute after it, which is why `phi` clamps to 1.0
    at 16:00:00 rather than interpolating into cum[390]. That is what makes a 16:00 market's
    remaining variance equal the full remaining variance the 4pm option is priced off.
    """

    def __init__(self, art):
        self.underlying = art["underlying"]
        self.shares = art["shares"]
        self.cum = art["cum"]
        self.overnight_share = art["overnight_share"]
        self.sessions = art["sessions"]
        self.built_utc = art.get("built_utc")
        if len(self.cum) != SESSION_MINUTES + 1:
            raise ValueError(f"calendar has {len(self.cum)} cum points, want {SESSION_MINUTES + 1}")

    @classmethod
    def load(cls, underlying, profile="default"):
        p = calendar_path(underlying, profile)
        if not p.exists():
            raise FileNotFoundError(
                f"no variance calendar for {underlying} ({profile}) at {p} -- "
                f"run: python -m pricing.varcal --build --underlying {underlying}"
                + ("" if profile == "default" else f" --profile {profile}"))
        return cls(json.loads(p.read_text()))

    def phi(self, t):
        """Cumulative variance share at ET datetime `t`. 0 before the open, 1 at/after the close."""
        et = t.astimezone(config.ET)
        if et.time() >= CLOSE_T:
            return 1.0
        open_dt = datetime.combine(et.date(), OPEN_T, tzinfo=config.ET)
        elapsed = (et - open_dt).total_seconds() / 60.0
        if elapsed <= 0:
            return 0.0
        i = int(elapsed)
        frac = elapsed - i
        if i >= SESSION_MINUTES - 1:
            return self.cum[SESSION_MINUTES - 1]
        return self.cum[i] + frac * (self.cum[i + 1] - self.cum[i])

    def remaining_ratio(self, now, expiry):
        """Fraction of the variance still ahead at `now` that lands before `expiry`.

        Multiply w_4pm^2 by this to get w_expiry^2. Returns None when there is no variance left
        to apportion -- inside the last seconds before the close, where the denominator collapses
        and the ratio is meaningless rather than merely large.
        """
        p_now = self.phi(now)
        denom = 1.0 - p_now
        if denom <= 1e-9:
            return None
        if expiry <= now:
            return None
        num = self.phi(expiry) - p_now
        return max(0.0, min(1.0, num / denom))

    def block_shares(self, width=30):
        """[(label, minutes, share)] in blocks of `width`.

        The final block is ragged -- 391 minutes is not a multiple of 30 -- so the minute count
        comes back with it. Comparing a 1-minute block's share against a 30-minute flat baseline
        is how the closing auction ends up looking 13x quieter than it is.
        """
        labs = minute_labels()
        out = []
        for start in range(0, SESSION_MINUTES, width):
            block = self.shares[start:start + width]
            out.append((labs[start], len(block), sum(block)))
        return out


def load(underlying, profile="default"):
    return VarianceCalendar.load(underlying, profile)


def load_for_session(underlying, d, evcal=None):
    """The standalone `fomc` profile on FOMC days, else the default. No callers; not wired.

    `event_aware` below is the successor candidate: it adds fitted release kernels to `evclean`
    rather than swapping in a 16-session shape, and it wins on FOMC days out of sample
    (notebook 11). The `fomc` profile fails theta smoothness (11.4x).
    """
    try:
        evcal = evcal or eventcal.load()
        if "fomc" in evcal.families(d, "intraday"):
            return VarianceCalendar.load(underlying, "fomc")
    except FileNotFoundError:
        pass
    return VarianceCalendar.load(underlying, "default")


# Floor on an event day's bucket as a fraction of the clean bucket. The kernels are OLS bin means,
# so a noisy bin can be negative, and a negative bucket would make phi non-monotone.
EVENT_AWARE_FLOOR = 0.25


def kernels_path(underlying):
    return config.MODEL_DIR / "variance_calendar" / f"{underlying}.evkernels.json"


def event_aware(underlying, d, releases=None, base=None, ker=None):
    """Session `d`'s calendar: `evclean` plus the fitted kernels of the releases on `d`.

    shares_d = normalise(evclean.shares + pad(sum_k kernel_k / baseline_total_rth)), where the
    kernels are the index-substrate fits from `pricing/study_event_variance`. Only the families the
    study marked `use` contribute. On a day with none of them, this is exactly `evclean`.

    NOT wired into the live pricer, and neither is `evclean`. Both are candidates. `base` and `ker`
    may be passed in-memory, which is how the out-of-sample score applies year-1 kernels.
    """
    base = base or json.loads(calendar_path(underlying, "evclean").read_text())
    ker = ker or json.loads(kernels_path(underlying).read_text())
    if releases is None:
        releases = eventcal.load_releases().on(d)
    use = set(ker.get("use", ()))
    add = np.zeros(SESSION_MINUTES)
    applied = []
    for f, t in releases:
        key = f"{f}@{t.strftime('%H:%M')}"
        if f in use and key in ker["kernels"]:
            add[:SESSION_MINUTES - 1] += np.asarray(ker["kernels"][key]) / ker["baseline_total_rth"]
            applied.append(key)
    if not applied:
        return VarianceCalendar(base)
    clean = np.asarray(base["shares"])
    sh = np.maximum(clean + add, EVENT_AWARE_FLOOR * clean)
    sh = sh / sh.sum()
    cum = np.concatenate([[0.0], np.cumsum(sh)])
    cum[-1] = 1.0
    art = dict(base, shares=[float(x) for x in sh], cum=[float(x) for x in cum],
               profile="event_aware", session=str(d), releases_applied=applied)
    return VarianceCalendar(art)


# --- cli -----------------------------------------------------------------

def session_split(cal):
    """(intraday share, overnight share) of one full 24h cycle's variance.

    The calendar normalises the 09:30-16:00 session to 1.0 and expresses the gap relative to it,
    so `overnight_share = 0.744` means the gap carries 74.4% of a session. As a share of the full
    cycle that is 0.744 / 1.744 = 42.7%.
    """
    on = cal.overnight_share
    return 1.0 / (1.0 + on), on / (1.0 + on)


def _show(underlying, profile="default"):
    cal = VarianceCalendar.load(underlying, profile)
    print(f"variance calendar {cal.underlying} ({cal.sessions} sessions, built {cal.built_utc})")
    intra, on = session_split(cal)
    print(f"  overnight gap is worth {cal.overnight_share:.3f} of one intraday session's variance")
    print(f"  -> of a full 24h cycle: {intra:6.1%} intraday (6.5h), {on:6.1%} overnight (17.5h)")
    print(f"     per hour, intraday is {(intra / 6.5) / (on / 17.5):.1f}x as variance-dense")
    print(f"  {'block':>7} {'min':>4}  {'share':>7}  {'vs flat':>8}")
    flat = 1.0 / SESSION_MINUTES
    for lab, n, sh in cal.block_shares():
        rel = sh / (n * flat)
        print(f"  {lab:>7} {n:4d}  {sh:7.4f}  {rel:7.2f}x  {'#' * int(round(rel * 10))}")
    print(f"  {'TOTAL':>7}  {sum(cal.shares):7.4f}")


def _block_table(shares, width=30):
    flat = 1.0 / SESSION_MINUTES
    labs = minute_labels()
    return [(labs[st], sum(shares[st:st + width]) / (len(shares[st:st + width]) * flat))
            for st in range(0, SESSION_MINUTES, width)]


def _split_half(sub, keep, width=FINE_WIDTH):
    """Per-bucket fit on each half of the sample, for stability.

    A sharp bucket that replicates across halves has a mechanism; one that does not is noise that
    happened to survive averaging. Measured on this sample: 15:50 (the NYSE MOC imbalance
    publication) and 15:59 (the MOC ramp) replicate; 15:54 reads 2.40x in the first half and
    1.07x in the second and should not be believed.
    """
    h = len(keep) // 2
    return (winsor_mean(sub["shape"][width][keep[:h]]),
            winsor_mean(sub["shape"][width][keep[h:]]))


def compare(underlying="SPX", years=DEFAULT_YEARS, end=None):
    """Fit one underlying every way and print what each choice actually buys.

    One read of the cache, many fits. The families are reported separately because they move the
    overnight number in OPPOSITE directions: FOMC is a 14:00 intraday event, so dropping FOMC
    days removes intraday variance and RAISES the gap's share; CPI/NFP (08:30) and mega-cap
    earnings (after the close) are gap events and lower it. A single net figure hides that.
    """
    sample = gather(years=years, end=end)
    sub = sample[underlying]
    evcal = eventcal.load()
    print(f"[varcal] event calendar built {evcal.built_utc}, window {evcal.window}")

    variants = [
        ("baseline (no exclusion)", None, None, False),
        ("component: FOMC only", {"fomc"}, "component", False),
        ("component: CPI+NFP", {"cpi", "nfp"}, "component", False),
        ("component: earnings", {"earnings"}, "component", False),
        ("component: ALL", None, "component", True),
        ("whole-day: ALL", None, "whole_day", True),
    ]

    rows = []
    for label, fams, mode, use in variants:
        cal = evcal if (use or mode) else None
        shares, _cum, on, meta = fit(sub, cal, fams, whole_day=(mode == "whole_day"),
                                     verbose=False)
        rows.append((label, meta["shape_sessions"], meta["gap_cycles"], on, on / (1.0 + on),
                     shares, meta))

    print(f"\n{underlying} ({SOURCE_TICKER[underlying]}), {len(sub['dates'])} sessions, "
          f"{len(sub['gap_dates'])} cycles")
    print(f"{'variant':<26} {'sess':>5} {'1-nt':>5} {'on_share':>9} {'of cycle':>9} {'vs base':>9}")
    base_on = rows[0][4]
    for label, ns, ng, on, cyc, _sh, _m in rows:
        print(f"{label:<26} {ns:5d} {ng:5d} {on:9.4f} {cyc:9.1%} {cyc - base_on:+9.2%}")

    m = rows[4][6]
    if m["overnight_multiday"]:
        print(f"\nthe {m['gap_cycles_multiday']} weekend/holiday gaps, component: ALL, would give "
              f"{m['overnight_multiday']:.4f} -- {m['overnight_multiday'] / rows[4][3]:.2f}x a "
              f"single night. Excluded because rvol divides by a hardcoded 17.5h.")

    print(f"\nintraday SHAPE (x a flat minute) -- this is what the hourly pricer actually reads")
    base_blocks = _block_table(rows[0][5])
    print(f"{'block':>7} {'baseline':>9} {'comp-ALL':>9} {'whole-day':>10} {'max |d|':>9}")
    worst = 0.0
    for i, (lab, b) in enumerate(base_blocks):
        c = _block_table(rows[4][5])[i][1]
        w = _block_table(rows[5][5])[i][1]
        dmax = max(abs(c - b), abs(w - b))
        worst = max(worst, dmax / b if b else 0.0)
        print(f"{lab:>7} {b:9.2f} {c:9.2f} {w:10.2f} {dmax:9.2f}")
    print(f"\nlargest relative block shift vs baseline: {worst:.1%}")

    keep = _select_shape(sub, evcal, None, False)
    lo, hi = _split_half(sub, keep)
    flat = 1.0 / SESSION_MINUTES
    labs = minute_labels()
    print("\nsplit-half stability of the sharpest buckets (x a flat minute):")
    print(f"  {'bucket':>7} {'first':>7} {'second':>7} {'ratio':>7}")
    for i in sorted(sorted(range(SESSION_MINUTES), key=lambda j: -rows[4][5][j])[:8]):
        fa, fb = lo[i] / flat, hi[i] / flat
        print(f"  {labs[i]:>7} {fa:7.2f} {fb:7.2f} {(fb / fa if fa else 0):7.2f}")
    print("  a bucket whose halves disagree by much more than 2x is noise, not a mechanism.")

    print("\nDECISION GATE: ship the excluded calendar only if the intraday SHAPE moved materially.")
    print("`remaining_ratio` -- the only thing pricing/hourly reads -- depends on the shape alone;")
    print("`overnight_share` is stored but never read by the hourly path (see the module docstring).")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="intraday variance calendar")
    ap.add_argument("--underlying", default="SPX", choices=sorted(SOURCE_TICKER),
                    help="which underlying to --show or --compare; --build writes both")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="fit with and without event exclusion and report the difference")
    ap.add_argument("--exclude-events", action="store_true")
    ap.add_argument("--whole-day", action="store_true",
                    help="with --exclude-events: drop a session from the shape when its gap is "
                         "contaminated too")
    ap.add_argument("--profile", default="default", choices=PROFILES)
    ap.add_argument("--gap-span", default="1", choices=("1", "all"),
                    help="overnight population: 1 = single nights only (default), "
                         "all = every cycle including weekends/holidays")
    ap.add_argument("--years", type=float, default=DEFAULT_YEARS)
    ap.add_argument("--force", action="store_true",
                    help="allow --build to overwrite the live default-profile calendar")
    a = ap.parse_args(argv)
    if a.compare:
        compare(a.underlying, years=a.years)
        return 0
    if a.build:
        build(years=a.years, exclude_events=a.exclude_events,
              whole_day=a.whole_day, profile=a.profile, force=a.force,
              gap_span=(1 if a.gap_span == '1' else None))
    if a.show or not a.build:
        _show(a.underlying, a.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
