"""Massive historical options data: NBBO tapes and as-of quotes.

`massive_client` is live-only (last-NBBO, contracts reference, streaming WS).
This is the historical half, and it is deliberately a separate module: the live
client is on the trading hot path and must not grow retry loops and paging state.

Three vendor behaviours drive the shape of this file, all verified against the
account rather than assumed:

  * **`/v3/quotes` returns a `next_url` on the final page.** The
    `massive_client.option_contracts` idiom -- loop until `next_url` is absent --
    never terminates here, and would silently re-page a 2M-quote tape forever.
    Termination is on an empty `results` list.

  * **`next_url` omits `apiKey`.** Every follow-up 401s without re-appending it.

  * **`sip_timestamp` is nanoseconds**, while the Massive *WebSocket* `t` field
    is milliseconds. Same vendor, same concept, different unit. Everything here
    stays int64 ns: at 2026 epoch scale float64 resolves to ~190 ns, so a single
    `ns / 1e9` shifts the microsecond field.

Measured on 2025-08-20 SPXW 0DTE: a near-money contract carries 1.5-2.3M quotes
per session (31-46 pages at limit=50000), a far-OTM one carries 1,255 (one page).
That spread is what the adaptive fetch in storage/massive_backfill exists for.
"""
import calendar
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time as dtime, timedelta, timezone

import config

QUOTE_PAGE_LIMIT = 50_000       # vendor maximum
MAX_RETRIES = 5
# Measured ceiling, not a guess. Sustained req/s by target rate / concurrency,
# all responses 200: 32->29, 64->55, 128->115, 200->190, 400->375, 800->217.
# Throughput collapses past ~375, so that is the real ceiling; 300 leaves room.
RATE_PER_SEC = 300.0
NS = 1_000_000_000

RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 15)       # past the 4pm close: the settling prints land on it


class MassiveHistoryError(RuntimeError):
    pass


# --- nanosecond helpers --------------------------------------------------
# Integer arithmetic only. `calendar.timegm` avoids the local-timezone round trip
# that datetime.timestamp() does on naive values.

def to_ns(dt):
    """tz-aware datetime -> int64 epoch nanoseconds."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime: tz is required for an unambiguous instant")
    utc = dt.astimezone(timezone.utc)
    return calendar.timegm(utc.utctimetuple()) * NS + utc.microsecond * 1000


def from_ns(ns):
    """int64 epoch nanoseconds -> tz-aware UTC datetime (truncating to µs)."""
    secs, rem = divmod(int(ns), NS)
    return datetime.fromtimestamp(secs, tz=timezone.utc).replace(
        microsecond=rem // 1000)


def et_session_ns(d, start=RTH_OPEN, end=RTH_CLOSE):
    """[start, end) of an ET calendar day in ns. DST-correct via config.ET."""
    lo = datetime.combine(d, start, tzinfo=config.ET)
    hi = datetime.combine(d, end, tzinfo=config.ET)
    return to_ns(lo), to_ns(hi)


# --- transport -----------------------------------------------------------

class RateLimiter:
    """Token bucket shared across threads. Massive's budget is separate from
    Kalshi's, so this does not interact with kalshi_client._throttle."""

    def __init__(self, per_sec=RATE_PER_SEC):
        self._lock = threading.Lock()
        self._interval = 1.0 / per_sec
        self._next = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            time.sleep(wait)


_limiter = RateLimiter()


def _with_key(url):
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}apiKey={config.MASSIVE_API_KEY}"


def get_url(url, timeout=60, limiter=None):
    """GET a fully-formed URL, re-appending apiKey and retrying transients."""
    (limiter or _limiter).acquire()
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(_with_key(url), timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read()[:200].decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503, 504):
                last = MassiveHistoryError(f"HTTP {e.code}: {body}")
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise MassiveHistoryError(f"HTTP {e.code}: {body}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = MassiveHistoryError(repr(e)[:200])
            time.sleep(0.5 * (2 ** attempt))
    raise last


def get(path, **params):
    params = {k: v for k, v in params.items() if v is not None}
    return get_url(f"{config.MASSIVE_REST_BASE}{path}?"
                   f"{urllib.parse.urlencode(params)}")


# --- reference -----------------------------------------------------------

def contracts_for_expiry(underlying, expiration_date, expired=True):
    """Every listed contract for one expiry. `expired=true` is required for any
    date in the past. Verified back to 2024-06-03."""
    out = []
    url = (f"{config.MASSIVE_REST_BASE}/v3/reference/options/contracts?"
           + urllib.parse.urlencode({
               "underlying_ticker": underlying,
               "expiration_date": str(expiration_date),
               "expired": "true" if expired else "false",
               "limit": 1000}))
    while url:
        page = get_url(url)
        res = page.get("results") or []
        out.extend(res)
        nxt = page.get("next_url")
        # Reference genuinely stops emitting next_url; guard on empties anyway.
        url = nxt if (nxt and res) else None
    return out


def pm_settled_root(contracts):
    """The PM-settled root for a 0DTE board.

    On the third Friday of each month an underlying lists two roots -- SPX (AM,
    09:30 print) alongside SPXW (PM, 16:00 close), and NDX alongside NDXP. Kalshi
    settles these markets on the 4pm close, so the AM root prices the wrong
    event. `universe.option_root_and_strikes` raises when it sees two roots;
    here we pick deliberately.
    """
    roots = {c["ticker"][2:-15] for c in contracts}
    if not roots:
        return None
    for preferred in ("SPXW", "NDXP"):
        if preferred in roots:
            return preferred
    if len(roots) == 1:
        return roots.pop()
    # Deterministic and loud rather than an arbitrary pick.
    raise MassiveHistoryError(f"ambiguous option roots, no PM-settled match: {roots}")


# --- quotes --------------------------------------------------------------

def quote_pages(option_ticker, start_ns, end_ns, order="asc",
                limit=QUOTE_PAGE_LIMIT, max_pages=None):
    """Yield pages of raw NBBO records, time-ordered.

    Terminates on an empty `results`, NOT on a missing `next_url` -- this
    endpoint returns a cursor even on the final page.
    """
    url = (f"{config.MASSIVE_REST_BASE}/v3/quotes/{option_ticker}?"
           + urllib.parse.urlencode({
               "timestamp.gte": str(int(start_ns)),
               "timestamp.lte": str(int(end_ns)),
               "order": order, "limit": limit}))
    pages = 0
    while url:
        page = get_url(url)
        res = page.get("results") or []
        if not res:
            return
        yield res
        pages += 1
        if max_pages is not None and pages >= max_pages:
            return
        url = page.get("next_url")


def quote_asof(option_ticker, ts_ns, since_ns=None):
    """The prevailing NBBO at `ts_ns`, no earlier than `since_ns`.

    as-of(c, t) = the quote with the greatest (sip_timestamp, sequence_number)
    such that sip_timestamp <= t. `timestamp.lte` is inclusive and the server's
    `desc` sort breaks ties, so one row is exactly right. Measured staleness on a
    liquid 0DTE contract: 17-180 ms.

    **`since_ns` is not optional in practice.** Option contracts are listed days
    ahead of expiry, so an unbounded query on a wing strike that has not quoted
    today happily returns a quote from the day it was listed -- measured p95 of
    7.9 h and max 15 h of staleness, presented as if it were the prevailing
    market. The live pricer never sees such a quote (the WS carries only today's
    ticks), so returning None is both the honest and the faithful answer.
    """
    p = {"limit": 1, "order": "desc", "timestamp.lte": str(int(ts_ns))}
    if since_ns is not None:
        p["timestamp.gte"] = str(int(since_ns))
    page = get(f"/v3/quotes/{option_ticker}", **p)
    res = page.get("results") or []
    return res[0] if res else None


class AsOfIndex:
    """Resolve many instants against an ascending tape in one streaming pass.

    Never materialises the tape: a near-money contract-day is 1.5-2.3M quotes,
    and the whole point of streaming is that only the resolved rows survive.
    Because pages are time-ascending, a partial tape resolves a *prefix* of the
    instants exactly -- which is what makes abandoning a tape mid-way safe.
    """

    def __init__(self, instants_ns):
        self.instants = sorted(int(t) for t in instants_ns)
        self._i = 0
        self._last = None          # most recent quote at or before instants[_i]
        self._out = {}

    def feed(self, page):
        for q in page:
            ts = int(q["sip_timestamp"])
            # Close out every instant this quote has now passed.
            while self._i < len(self.instants) and self.instants[self._i] < ts:
                if self._last is not None:
                    self._out[self.instants[self._i]] = self._last
                self._i += 1
            if self._i >= len(self.instants):
                return False        # nothing further can be resolved
            self._last = q
        return True

    def finish(self):
        """Flush the tail: every remaining instant takes the last quote seen."""
        while self._i < len(self.instants):
            if self._last is not None:
                self._out[self.instants[self._i]] = self._last
            self._i += 1

    @property
    def resolved(self):
        return self._out

    def pending(self):
        return [t for t in self.instants if t not in self._out]


# --- aggregates ----------------------------------------------------------
# Stock and index minute bars, for fitting the intraday variance calendar that
# pricing/varcal builds. Options never enter here: the calendar wants the shape
# of realised variance through a session, which is an underlying property.

AGG_PAGE_LIMIT = 50_000         # vendor maximum; ~128 RTH sessions of 1-minute bars

# Aggregates are a separate rate budget again from the 300 req/s that /v3/quotes sustains.
#
# **This ceiling moved when the Index entitlement was purchased, and the history matters.**
# Measured 2026-08-20, on the pre-purchase key, same endpoint, 24 workers:
#
#     SPY     400 concurrent requests -> 400x HTTP 200. No throttling at all.
#     I:NDX   120 concurrent requests -> 115x HTTP 429, 5x HTTP 200.
#
# The ceiling travelled with the *ticker*, not the endpoint, which is what a partial entitlement
# looks like: I:NDX worked, I:SPX/I:VIX/I:DJI/I:RUT were NOT_AUTHORIZED. 2 req/s kept the index
# side under it.
#
# Re-measured 2026-08-21, after the Index entitlement was added:
#
#     I:SPX   40 day-requests at 16 workers -> 40x HTTP 200, ZERO 429s, 16.0 req/s sustained.
#
# So the index throttle was the entitlement, not a rate policy, and it is gone. 16 req/s is what
# was actually verified rather than the most that might work -- raise it only against a fresh
# measurement, and leave the 2 req/s number above readable, because if the entitlement ever
# lapses the old ceiling is what comes back.
#
# The 429 itself needs no special handling -- `get_url` already retries it with backoff.
AGG_RATE_PER_SEC = 16.0
_agg_limiter = RateLimiter(per_sec=AGG_RATE_PER_SEC)


def _agg_page(url):
    """One aggregates page.

    The body check is belt-and-braces, not an observed failure: every throttle seen on this
    endpoint arrived as a proper HTTP 429 and was absorbed by `get_url`. It is here because a
    200-with-`status:ERROR` body would otherwise be indistinguishable from a final empty page and
    would truncate the sample silently -- an expensive failure to debug for a cheap check.
    """
    page = get_url(url, limiter=_agg_limiter)
    if page.get("status") == "ERROR":
        raise MassiveHistoryError(
            f"aggregates returned 200 with an error body: "
            f"{str(page.get('error') or page.get('message'))[:200]}")
    return page


def aggregate_bars(ticker, multiplier, timespan, start, end, adjusted=True, limit=AGG_PAGE_LIMIT):
    """OHLC bars for `ticker` at any vendor granularity, ascending.

    `start`/`end` are whatever /v2/aggs accepts on this range: ET calendar dates ('2026-08-20')
    or epoch-millisecond integers. Returns raw vendor bars, each `{t, o, h, l, c, v, ...}` where
    **`t` is epoch MILLISECONDS** -- matching the Massive WebSocket, not the nanosecond
    `sip_timestamp` of /v3/quotes. Same vendor, same concept, different unit; converting on the
    wrong assumption lands every bar in 1970 or 56000 AD.

    Coverage facts the caller must handle rather than assume away, all measured:

      * **Stocks come back with extended hours.** SPY returns 855-869 minute bars spanning
        04:00-19:59 ET, not 390, and its second bars span the same window. Those pre/post bars
        are thin -- measured 2026-08-20, SPY second-bar coverage by ET hour, out of 3600
        possible: 04h 255, 05h 79, 06h 95, 07h 421, 08h 769, 09h 2102, then ~3400/h through
        RTH. Anything fitting a session shape must filter to RTH; anything using the pre-open
        must expect to forward-fill it.
      * **Indices have no pre-open bars, but they do run PAST the close, by different
        amounts each.** Measured 2026-08-21 for the 2026-08-20 session: `I:SPX` 395 minute bars
        09:30-16:04, `I:NDX` **466** minute bars 09:30-17:15. Second bars: `I:SPX`
        09:30:01-16:04:57 (~22.5k), `I:NDX` 09:30:15-17:15:59 (~27.9k).

        An older comment here claimed `I:NDX` returns "exactly 391" -- it does not, and did not
        under the pre-refactor code either (verified against the backup). Do not gate on a bar
        count. `bars_by_et_session(rth_only=True)` hard-filters to 09:30-16:00 and is what
        actually protects the variance-calendar fit.

        There are no overnight index bars at all, because an index is only computed while its
        constituents trade -- which is why a pre-open lookback needs SPY/QQQ and not I:SPX.
      * **One RTH day of index second bars fits in a single page.** ~22.5-23k rows against a
        50,000 limit, so a per-day request never pages. Do not infer from that that paging is
        unnecessary in general -- a multi-day second-bar range pages hard.

    Paging follows the /v3/quotes idiom -- terminate on empty `results`, not on a missing
    `next_url` -- because it is the safe superset: /v2/aggs does appear to drop the cursor on
    the final page, but relying on that is the exact assumption that made the quotes loop spin.
    """
    url = (f"{config.MASSIVE_REST_BASE}/v2/aggs/ticker/{ticker}"
           f"/range/{multiplier}/{timespan}/{start}/{end}?"
           + urllib.parse.urlencode({"adjusted": "true" if adjusted else "false",
                                     "sort": "asc", "limit": limit}))
    out = []
    seen = set()
    while url:
        page = _agg_page(url)
        res = page.get("results") or []
        if not res:
            break
        # The cursor restarts the range at the last bar's timestamp, so page boundaries can
        # repeat a bar. Deduping on `t` keeps one row per interval; a duplicated bar would
        # contribute a spurious zero return to that interval's variance bucket.
        for b in res:
            if b["t"] not in seen:
                seen.add(b["t"])
                out.append(b)
        url = page.get("next_url")
    out.sort(key=lambda b: b["t"])
    return out


def second_bars(ticker, start_date, end_date, adjusted=True, limit=AGG_PAGE_LIMIT):
    """1-second OHLC bars over [start_date, end_date], ET calendar dates.

    The realized-vol study's substrate. Note that for an index every bar has o == c == h == l:
    the index is a computed level with one value per second, not a traded instrument with a
    range inside the interval. Only `c` is meaningful, and using `h`/`l` to estimate variance
    would silently read zero.
    """
    return aggregate_bars(ticker, 1, "second", start_date, end_date,
                          adjusted=adjusted, limit=limit)


def bars_by_et_session(bars, rth_only=True):
    """Group minute bars into {ET date: [bar, ...]}, ascending within each session.

    `rth_only` keeps 09:30 <= t <= 16:00 ET inclusive of both ends: the 09:30 bar carries the
    opening auction print and the 16:00 bar carries the closing one, which is the instant every
    market here settles against.
    """
    out = {}
    for b in bars:
        dt = datetime.fromtimestamp(b["t"] / 1000.0, tz=timezone.utc).astimezone(config.ET)
        if rth_only and not (RTH_OPEN <= dt.time() <= dtime(16, 0)):
            continue
        out.setdefault(dt.date(), []).append((dt, b))
    for day in out:
        out[day].sort(key=lambda p: p[0])
    return out
