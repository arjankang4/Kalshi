"""One session of 1-second index closes, decomposed into variance buckets.

The substrate for `pricing/varcal`. It exists so the variance calendar reads local Parquet
instead of re-pulling two years of vendor minute bars on every build, and so the 1-minute and
5-minute grids are cut from *the same* series -- if they came from two vendor requests at two
granularities, the 5-minute anchoring step would be reconciling two different samples rather
than two views of one.

WHERE THE DATA COMES FROM

`var/rvol/bars/{I_SPX,I_NDX}/dt=YYYY-MM-DD.parquet`, written by `rvol/data.py`: two columns,
`t_ms` (epoch milliseconds) and `close`. 502 sessions, 2024-08-21..2026-08-21, and the two
tickers cover exactly the same dates. Only the close is stored, which is the whole story for an
index -- a computed level has one value per instant, so `o == h == l == c` and reading the range
would report zero.

This module deliberately does NOT own the pull. `rvol.data --build` materialises the cache;
everything here is a reader. `pricing` reaching up into `rvol` for that would invert the
dependency (`rvol/{dataset,live,features}` already import `pricing.varcal`), so the few lines of
Parquet access are repeated here rather than imported.

WHAT A BUCKET IS

Returns are close-to-close on a fixed grid of clock edges from 09:30:00 to 16:00:00, where the
price at an edge is the last print at or before it. Bucket i is the move over
[09:30 + i*width, 09:30 + (i+1)*width). At width=60 that is 390 buckets and at width=300 it is
78, over an identical span -- so the two grids nest exactly, five 1-minute buckets to a
5-minute block, which is what makes the rescale in `varcal` well defined.

THE TERMINAL BUCKET

One bucket is appended: `ln(last print of the session / price at 16:00:00)`. It is not a clock
minute, it is the settlement convergence.

An index is not a traded instrument and has no closing auction print of its own; its official
close is a restatement computed from constituents' auction prices, which arrive over the
following minutes. Measured over 260 sessions: `I:SPX` keeps printing to a median 16:03:43 (11
late bars, max 16:07:34) while `I:NDX` typically stops at 16:00:01. The move from the 16:00:00
tick to the day's last print has rms 2.1e-4 on SPX and 2.6e-4 on NDX -- **15% of the
15:59-16:00 bucket's variance on both**, or about 1.4 SPX points.

Kalshi settles on that official close, and so do the SPXW/NDXP options the ATM vol is backed out
of. The 4pm option's `w` therefore prices variance all the way to the final print, and a
calendar that stops at the 16:00:00 tick under-weights its own terminal bucket and hands the
surplus to every earlier hour. Folding the convergence in keeps `phi(16:00) == 1.0` -- the
bucket is realised *at* settlement, not spread over the minute after it -- so the 16:00 identity
that `pricing/validate_hourly` checks is untouched.

`TAIL_CAP_T` bounds it at 16:15 ET because the vendor's tail is ragged and index-specific:
`I:NDX` has been seen republishing until 17:15, which is long past anything that can be called
a close.

WHEN THERE IS NO LATE PRINT the terminal return is 0.0, and that is honest rather than a fudge:
the last observed price *is* the settlement price, so the move to it is zero. Contrast a missing
mid-session minute, where 0.0 would be a lie -- the grid never produces one, because an edge
with no print of its own carries the last price forward, which yields a zero return in that
bucket and the full move in the next.
"""
import datetime as _dt
from datetime import time as dtime

import numpy as np

import config

CACHE = config.ROOT / "var" / "rvol" / "bars"

# The settlement index, both sides. SPX used SPY until the Index entitlement landed on
# 2026-08-21; see the SOURCE_TICKER note in pricing/varcal for why that mattered for a *shape*
# and not only for a level.
TICKER = {"SPX": "I:SPX", "NDX": "I:NDX"}

OPEN_T = dtime(9, 30)
CLOSE_T = dtime(16, 0)
TAIL_CAP_T = dtime(16, 15)

OPEN_S = OPEN_T.hour * 3600 + OPEN_T.minute * 60
CLOSE_S = CLOSE_T.hour * 3600 + CLOSE_T.minute * 60
TAIL_CAP_S = TAIL_CAP_T.hour * 3600 + TAIL_CAP_T.minute * 60

RTH_MINUTES = (CLOSE_S - OPEN_S) // 60          # 390

# A session must have printed in at least this many distinct RTH minutes to be usable. Half days
# (13:00 close) land around 200 and fall out; a full session that is merely thin does not.
#
# Measured on 101 sampled sessions per ticker: 100 of them have ZERO minutes without a print,
# and the one exception is a half day. This threshold is a tripwire, not a filter that fires in
# normal operation -- which is why it is a count and not a holiday table, the same reasoning as
# varcal.MIN_SESSION_BARS.
MIN_GRID_MINUTES = 380

# Below this a file is a stub rather than a session.
MIN_ROWS = 1000


def cache_path(ticker, d):
    return CACHE / ticker.replace(":", "_") / f"dt={d}.parquet"


def cached_dates(ticker):
    """Sorted [date] present in the cache for `ticker`."""
    p = CACHE / ticker.replace(":", "_")
    if not p.exists():
        return []
    out = []
    for f in p.glob("dt=*.parquet"):
        try:
            out.append(_dt.date.fromisoformat(f.name[3:-8]))
        except ValueError:
            continue
    return sorted(out)


def _et_seconds(t_ms):
    """Epoch-ms -> seconds since ET midnight, vectorised.

    The UTC offset is read once from the first timestamp and applied to the whole array. That is
    safe *here* and would not be in general: a DST transition happens at 02:00 local on a Sunday,
    so it can never fall inside an RTH session. Anything in this repo that spans a whole calendar
    day must go through `config.ET` per timestamp instead -- see the note on `config.ET` about
    the 23- and 25-hour local days.
    """
    first = _dt.datetime.fromtimestamp(int(t_ms[0]) / 1000.0, tz=_dt.timezone.utc)
    off = int(first.astimezone(config.ET).utcoffset().total_seconds())
    return ((t_ms // 1000) + off) % 86400


def load_session(ticker, d):
    """(et_seconds, close) for one session, trimmed to 09:30..TAIL_CAP, or None if unusable.

    Both arrays ascend with time. `None` covers: no cached file, a stub, and a session that did
    not print in `MIN_GRID_MINUTES` distinct RTH minutes.
    """
    import pyarrow.parquet as pq

    p = cache_path(ticker, d)
    if not p.exists():
        return None
    tbl = pq.read_table(p)
    t_ms = np.asarray(tbl.column("t_ms"), dtype="int64")
    close = np.asarray(tbl.column("close"), dtype="float64")
    if len(t_ms) < MIN_ROWS:
        return None

    es = _et_seconds(t_ms)
    keep = (es >= OPEN_S) & (es <= TAIL_CAP_S) & (close > 0)
    es, close = es[keep], close[keep]
    if len(es) < MIN_ROWS:
        return None
    if len(np.unique(es[es < CLOSE_S] // 60)) < MIN_GRID_MINUTES:
        return None
    return es, close


def grid_returns(es, close, width):
    """Log returns on a `width`-second grid, plus the terminal settlement bucket.

    Returns `(rets, open_px, settle_px)`:
      * `rets` has `(CLOSE_S - OPEN_S) // width + 1` entries -- 391 at width=60, 79 at width=300.
        The last is the terminal bucket.
      * `open_px` is the price at the 09:30 edge, i.e. the session's first print. It anchors the
        *incoming* overnight gap.
      * `settle_px` is the last print at or before TAIL_CAP, i.e. where the terminal bucket ends.
        It anchors the *outgoing* gap.

    Those two make the whole 09:30 D -> 09:30 D+1 cycle telescope:
    `sum(rets) + ln(open_{D+1} / settle_D) == ln(open_{D+1} / open_D)`, every part counted once.
    """
    n = (CLOSE_S - OPEN_S) // width
    edges = OPEN_S + width * np.arange(n + 1)
    # Last print at or before each edge. The 09:30 edge is clamped to the first print because an
    # index has no pre-open tick to carry forward -- I:SPX opens at 09:30:01, I:NDX at 09:30:15.
    idx = np.maximum(np.searchsorted(es, edges, side="right") - 1, 0)
    px = close[idx]
    rets = np.log(px[1:] / px[:-1])

    settle_px = float(close[-1])
    term = np.log(settle_px / px[-1]) if es[-1] > CLOSE_S else 0.0
    return np.append(rets, term), float(px[0]), settle_px

