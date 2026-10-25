"""CME futures from Massive: the overnight tape the index and the ETFs cannot provide.

WHY THIS MODULE EXISTS

`pricing/varcal` measures how a session's variance is distributed across its 390 RTH minutes, but
it knows the overnight gap only as a single scalar (`overnight_share`). `rvol/features` then
spreads that lump UNIFORMLY across 17.5 hours, which is certainly wrong -- overnight variance
clusters at the European open and the 08:30 prints. Giving the gap a real minute-wise shape needs a
price series that trades overnight, and until now this repo believed it had none:

  * An index is only computed while its constituents trade. `I:SPX` and `I:NDX` have **no**
    pre-open or overnight bars at all.
  * SPY/QQQ extended hours run **04:00-19:59 ET only**. 20:00-04:00 -- 8 of the 17.5 hours -- is
    genuinely empty, and what coverage exists is thin enough to be dominated by microstructure
    noise (measured: 60s RV over the covered 9.5h reads 1.36x the WHOLE gap's variance, and 10s
    reads 2.78x, falling to 0.76x at 900s -- a textbook noise signature plot).

THE 2026-08-21 "NO FUTURES" PROBE WAS A FALSE NEGATIVE

It is recorded in three places that Massive carries no futures. It does. Re-probed 2026-08-22 and
every endpoint below returns 200 with real data back to 2024. Three independent mistakes, each of
which alone produces "no futures", and each worth knowing because they are all still available to
make:

  1. It requested `/futures/vX/...`. That is a literal placeholder in the docs, not a version. The
     404 is correct and means nothing. The real base is `/futures/v1/`.
  2. It sent `market=futures` to `/v3/reference/tickers`. v3 reference is the EQUITIES namespace
     and returns zero futures no matter what is asked of it. (Same trap as `ES` resolving there to
     Eversource Energy.)
  3. Resolution. `/futures/v1/aggs` accepts ONLY `1sec`/`1min`/`1second`/`1minute`. `1s`, `1m`,
     `second`, `minute`, `S1` and `60` all return **HTTP 400 "Invalid resolution"** -- which, if
     the response body is not read, looks exactly like a dead endpoint. `RESOLUTIONS` below is
     enforced client-side so this cannot recur silently.

DATE FILTERING -- THE ONE THAT WILL COST YOU A CALENDAR

Only `window_start.gte` / `window_start.lt` filter. Measured 2026-08-22, every one of these
returned **HTTP 200 with the most recent session's rows** rather than the range asked for:

    start_date=&end_date=      date=      from=&to=      session_end_date.gte=

That is worse than a 400. A two-year build using `start_date` would silently receive the newest
session repeated for every requested day, and nothing would raise. `aggs` therefore asserts that
the returned min/max timestamps fall inside the requested window -- see `_assert_window`. Do not
remove it, and do not add a convenience `date=` parameter.

Accepted forms on `window_start.*` are epoch **nanoseconds** or ISO-8601; this module uses
nanoseconds throughout. Note the resulting unit spread across one vendor: `/futures/v1/aggs`
`window_start` is NANOSECONDS, `/v2/aggs` `t` is MILLISECONDS, `/v3/quotes` `sip_timestamp` is
nanoseconds. Converting on the wrong assumption lands every bar in 1970 or 56000 AD.

COVERAGE

CME Globex runs 18:00 ET -> 17:00 ET the next day with a maintenance halt 17:00-18:00 ET, so the
futures tape covers the whole overnight gap except that one hour, which is exactly where SPY/QQQ
post-market is used instead. Overnight liquidity is real but nothing like RTH -- measured on
ESU6, 2026-08-20 20:00 -> 08-21 04:00 ET, 1-second bars, as a fraction of the 3600 seconds in
each ET hour:

    20h 50%   21h 48%   22h 33%   23h 26%   00h 28%   01h 25%   02h 30%   03h 42%

10,196 bars over the eight hours SPY has zero. Pre-open 04:00-09:00 ET gives ES ~14.3k bars
against SPY's ~3.7k. So the gap is observable end to end, and `pricing/gapcal` never has to infer
a slice as a residual.

EXPIRED CONTRACTS ARE READABLE, which is what makes a two-year calendar possible at all: ESU4 at
2024-08-22, ESZ4 at 2024-11-14, ESM5 at 2025-05-14 and ESZ5 at 2025-11-13 all return bars.

THE ROLL IS A DISCONTINUITY, NOT VARIANCE

ES and NQ roll quarterly (H/M/U/Z = Mar/Jun/Sep/Dec). A return taken ACROSS a roll boundary is a
pure contract-spread artifact. Measured at the 2024-09 roll: on 2024-09-16 ESU4 closed 5639.00
while ESZ4 closed 5700.50 -- a **61.5 point, 1.09% step** that is carry, not risk. Left in, it
would land in one overnight bucket and dominate it.

`front_month` picks the contract by TRADED VOLUME rather than by days-to-maturity, for two
reasons. It is self-correcting -- the roll is defined as the day volume actually crosses over, not
by a calendar rule that a holiday can shift -- and it does not depend on the contracts endpoint's
metadata, which is NOT uniform across the sample: records dated 2024 carry neither `type` nor
`days_to_maturity`, both of which the 2026 records have. Measured crossover for 2024-09:

    2024-09-12   ESU4 1,233,410   ESZ4    86,557   -> ESU4
    2024-09-16   ESU4   365,541   ESZ4   552,300   -> ESZ4   (the roll)
    2024-09-20   ESU4         0   ESZ4 1,148,724   -> ESZ4

Callers must drop the return spanning a roll rather than measure it. This module reports the
chain; `rvol/data` persists it and `pricing/gapcal` excludes the boundary return.
"""
import datetime
import re
import urllib.parse

import config
from massive.history import MassiveHistoryError, RateLimiter, get_url, AGG_RATE_PER_SEC

# Kalshi underlying -> CME product code. ES is the S&P 500 E-mini, NQ the Nasdaq-100 E-mini --
# the same two indices `varcal` fits, so the gap shape and the intraday shape describe one market.
PRODUCT = {"SPX": "ES", "NDX": "NQ"}

# Local cache ticker per underlying, mirroring rvol/data's {I_SPX,I_NDX,SPY,QQQ} layout.
CACHE_TICKER = {"SPX": "ES", "NDX": "NQ"}

# Enforced client-side: anything else is a 400 whose body must be read to distinguish from a dead
# endpoint. See the module docstring.
RESOLUTIONS = ("1sec", "1min", "1second", "1minute")

# THE CURSOR IS BROKEN. DO NOT PAGINATE.
#
# `/futures/v1/aggs` returns a `next_url` when a response saturates `limit`, and following it
# returns **HTTP 502, deterministically** -- measured 2026-08-22 on ESH5 2025-03-05: page 1 returns
# 200 with 50,000 rows and a cursor, and that cursor 502s on every retry, at limit=50000 and at
# limit=20000 alike. Base64-decoding it shows why it could not have worked anyway:
#
#     ap=1741225356000000000&as=2025-03-06&limit=50000&resolution=...
#
# It carries an "after position" and an "after session" and has **dropped `window_start.lt`** --
# the upper bound of the range that was asked for. Even had it returned 200 it would have walked
# past the end of the requested window.
#
# So `aggs` never follows a cursor. When a response comes back saturated it BISECTS the time range
# and asks again, which needs no vendor state and cannot silently overrun the window. A full ET
# calendar day of ES 1-second bars runs 37k-55k rows, so days above the limit split once.
#
# This is also what made the first two-year build fail at 13%: every day over 50,000 rows -- the
# busy ones, which are exactly the days a variance calendar most needs -- hit the cursor and died.
# The failures were NOT random, and a retry loop was never going to clear them.
PAGE_LIMIT = 50000

# Below this a saturated window is a vendor fault rather than a busy market, and splitting further
# would loop. One second of ES prints cannot fill a 50,000-row page.
MIN_SPLIT_NS = 60 * 1_000_000_000

# CME Globex. 18:00 ET -> 17:00 ET next day, dark 17:00-18:00 ET.
HALT_START = datetime.time(17, 0)
HALT_END = datetime.time(18, 0)

# Quarterly cycle. The year digit is the last digit of the delivery year, which is ambiguous over a
# decade and unambiguous over the two years this repo holds.
MONTH_CODE = {"H": 3, "M": 6, "U": 9, "Z": 12}
_TICKER_RE = re.compile(r"^([A-Z]{2})([HMUZ])(\d)$")

# A correction worth keeping, because the wrong diagnosis was the obvious one. The first two-year
# build failed **33 of its first 250 day-requests with HTTP 502** at 16 req/s across 8 workers, and
# that reads exactly like server-side overload from too much concurrency. It was not. Every one of
# those failures was a day whose 1-second bars exceed the 50,000-row page limit, following the
# broken cursor -- see PAGE_LIMIT. Lowering the rate did nothing; an isolated single request for
# one of those days still 502'd.
#
# The selection bias in that failure mode is the part to remember: the days over 50,000 rows are
# the BUSY ones. A variance calendar built from what survived would have been fitted on the quiet
# sessions and would have understated overnight variance, with nothing anywhere raising.
#
# The rate is therefore modest out of courtesy rather than necessity -- a futures day is ~48,000
# rows against ~22,500 for an RTH index day, and busy days now cost two requests instead of one.
FUTURES_RATE_PER_SEC = 10.0
_limiter = RateLimiter(per_sec=FUTURES_RATE_PER_SEC)

NS_PER_MS = 1_000_000

# A front month trades over a million contracts a session. Anything below this is not a front
# month, and accepting it is how a chain acquires a one-day excursion into a back contract.
#
# This is not hypothetical. `/futures/v1/contracts?date=2026-03-27` returns a list that OMITS
# ESM6 and ESU6 entirely -- it jumps from the already-expired ESH6 straight to ESZ6, ESH7, ESU7.
# ESM6 traded **1,408,742** contracts that day and the aggs endpoint serves them; only the
# REFERENCE endpoint is incomplete. Selecting from that list picked ESZ6 on a volume of **1**,
# putting a spurious two-day roll into the chain and a 9-month basis step into two overnight
# cycles. Hence `quarterly_candidates`, which does not consult the reference endpoint at all.
MIN_FRONT_VOLUME = 1000

# Delivery months of the quarterly cycle, in order.
QUARTERLY_MONTHS = (3, 6, 9, 12)
CODE_FOR_MONTH = {v: k for k, v in MONTH_CODE.items()}


def parse_ticker(ticker):
    """'ESU4' -> ('ES', 9, 2024), or None if it is not a single-delivery contract.

    Calendar spreads come back from the contracts endpoint as 'ESH5-ESM5' and must never be
    treated as an outright -- a spread's price is a few points and its returns are enormous.
    """
    m = _TICKER_RE.match(ticker or "")
    if not m:
        return None
    product, code, year_digit = m.groups()
    # The sample spans 2024-2026; anchor the decade on the current year rather than hardcoding.
    base = datetime.date.today().year
    year = base - (base % 10) + int(year_digit)
    if year < base - 5:
        year += 10
    return product, MONTH_CODE[code], year


def _get(path, **params):
    params = {k: v for k, v in params.items() if v is not None}
    url = f"{config.MASSIVE_REST_BASE}{path}?{urllib.parse.urlencode(params)}"
    page = get_url(url, limiter=_limiter)
    if page.get("status") == "ERROR":
        raise MassiveHistoryError(
            f"futures endpoint returned 200 with an error body: "
            f"{str(page.get('error') or page.get('message'))[:200]}")
    return page


def _assert_window(bars, start_ns, end_ns, ticker):
    """The E9 guard: prove the vendor honoured the range instead of trusting that it did.

    Several plausible date parameters return 200 with the WRONG rows (see the module docstring),
    so a build that filtered on one of them would receive the newest session over and over and
    raise nothing. This turns that into a loud failure at the first request.
    """
    if not bars:
        return
    lo = min(b["window_start"] for b in bars)
    hi = max(b["window_start"] for b in bars)
    if lo < start_ns or hi >= end_ns:
        raise MassiveHistoryError(
            f"{ticker}: vendor ignored the window filter -- asked "
            f"[{start_ns}, {end_ns}), got [{lo}, {hi}]. See the date-filtering note in "
            f"massive/futures.py before 'fixing' this.")


def aggs(ticker, resolution, start_ns, end_ns, limit=PAGE_LIMIT):
    """Bars for `ticker` over [start_ns, end_ns), ascending. Timestamps are NANOSECONDS.

    Each bar is `{ticker, window_start, session_end_date, open, high, low, close, volume,
    transactions, dollar_volume}`. Unlike an index, a future is a traded instrument, so o/h/l/c
    are all real -- but only `close` is used downstream, to keep one estimator across index, ETF
    and futures legs.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(
            f"resolution {resolution!r} is rejected by the vendor with HTTP 400; "
            f"use one of {RESOLUTIONS}")
    start_ns, end_ns = int(start_ns), int(end_ns)
    out = _aggs_range(ticker, resolution, start_ns, end_ns, limit)
    # Deduplicate on the bar's own timestamp: bisected halves share no interval, but a vendor that
    # ever returned a boundary bar twice would put a spurious zero return into a variance bucket.
    seen, uniq = set(), []
    for b in out:
        ws = b.get("window_start")
        if ws is not None and ws not in seen:
            seen.add(ws)
            uniq.append(b)
    uniq.sort(key=lambda b: b["window_start"])
    _assert_window(uniq, start_ns, end_ns, ticker)
    return uniq


def _aggs_range(ticker, resolution, start_ns, end_ns, limit):
    """One request, bisecting the RANGE rather than following the broken cursor."""
    url = (f"{config.MASSIVE_REST_BASE}/futures/v1/aggs/{ticker}?"
           + urllib.parse.urlencode({"resolution": resolution,
                                     "order": "asc",
                                     "sort": "window_start",
                                     "window_start.gte": start_ns,
                                     "window_start.lt": end_ns,
                                     "limit": limit}))
    page = get_url(url, limiter=_limiter)
    if page.get("status") == "ERROR":
        raise MassiveHistoryError(
            f"futures aggs returned 200 with an error body: "
            f"{str(page.get('error') or page.get('message'))[:200]}")
    res = page.get("results") or []
    if len(res) < limit:
        return res
    # Saturated: there is more data in this window than the page can carry. Split rather than
    # accept it -- a truncated window reads as a quiet market, which is the failure a variance
    # calendar is least able to detect.
    if end_ns - start_ns <= MIN_SPLIT_NS:
        raise MassiveHistoryError(
            f"{ticker}: {len(res)} rows in a {(end_ns - start_ns) / 1e9:.0f}s window at the page "
            f"limit and the window cannot be split further")
    mid = start_ns + (end_ns - start_ns) // 2
    return (_aggs_range(ticker, resolution, start_ns, mid, limit)
            + _aggs_range(ticker, resolution, mid, end_ns, limit))


def _et_ns(day, t):
    return int(datetime.datetime.combine(day, t, tzinfo=config.ET).timestamp() * 1e9)


def session_volume(ticker, day):
    """Total RTH volume for one contract on one ET date. The front-month discriminator."""
    bars = aggs(ticker, "1min",
                _et_ns(day, datetime.time(9, 30)),
                _et_ns(day, datetime.time(16, 0)))
    return sum(b.get("volume") or 0 for b in bars)


def third_friday(year, month):
    """The third Friday of a month -- CME's equity-index delivery date."""
    d = datetime.date(year, month, 1)
    fridays = [d + datetime.timedelta(days=i) for i in range(31)
               if (d + datetime.timedelta(days=i)).month == month
               and (d + datetime.timedelta(days=i)).weekday() == 4]
    return fridays[2]


def quarterly_candidates(product_code, day, n=3):
    """The next `n` quarterly contracts delivering on or after `day`, nearest first.

    Derived from the CME calendar rather than from `/futures/v1/contracts`, because that endpoint
    has been observed returning an incomplete list on at least one date -- see MIN_FRONT_VOLUME.
    A deterministic cycle needs no vendor reference at all: ES and NQ deliver on the third Friday
    of March, June, September and December, every quarter, without exception.
    """
    out = []
    year, month = day.year, day.month
    # Rewind to the quarter containing or preceding `day`, then walk forward.
    month = max([m for m in QUARTERLY_MONTHS if m <= month], default=None)
    if month is None:
        year, month = year - 1, 12
    for _ in range(n + 2):
        if third_friday(year, month) >= day:
            out.append((f"{product_code}{CODE_FOR_MONTH[month]}{year % 10}",
                        third_friday(year, month)))
            if len(out) == n:
                break
        i = QUARTERLY_MONTHS.index(month)
        year, month = (year + 1, QUARTERLY_MONTHS[0]) if i == 3 else (year, QUARTERLY_MONTHS[i + 1])
    return out


def front_month(product_code, day, n_candidates=3, incumbent=None):
    """The most-traded outright on ET date `day`, or None if nothing traded.

    By volume, not by calendar -- see the roll discussion in the module docstring. Only the
    `n_candidates` nearest deliveries are probed, because the back months are illiquid enough
    that no reordering is possible and each probe is a request.
    """
    cands = [tk for tk, _e in quarterly_candidates(product_code, day, n=n_candidates)]
    if incumbent and incumbent not in cands:
        cands.insert(0, incumbent)          # always re-probe the sitting contract
    best, best_vol = None, 0
    for tk in cands:
        vol = session_volume(tk, day)
        if vol > best_vol:
            best, best_vol = tk, vol
    # Below the floor nothing traded meaningfully -- a holiday, or a date the vendor has no bars
    # for. Returning None lets the caller carry the incumbent rather than record a phantom roll.
    return best if best_vol >= MIN_FRONT_VOLUME else None


def closes_ms(ticker, start_ns, end_ns, resolution="1sec"):
    """[(epoch_ms, close)] over a window, ascending -- the cache's substrate.

    Milliseconds, not the vendor's nanoseconds, so the futures cache is byte-compatible with the
    index and ETF caches `rvol/data` already writes and every reader downstream stays unchanged.
    Non-positive closes are dropped rather than carried, because a log return is taken of every
    one of these.
    """
    return [(b["window_start"] // NS_PER_MS, b["close"])
            for b in aggs(ticker, resolution, start_ns, end_ns)
            if b.get("close") is not None and b["close"] > 0]
