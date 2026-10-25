"""Index levels from Massive: the real SPX/NDX spot, replacing put-call-parity inference.

Until 2026-08-21 this module could not exist. `I:SPX` was NOT_AUTHORIZED on the key, `I:GSPC`
returned empty, and the only index data of any kind was `I:NDX` minute aggregates -- so spot had
to be inferred from put-call parity off the option board (`pricing/theo.parity_spot`,
`pricing/atmvol.parity_forward`) and SPY was explicitly rejected as a substitute because an ETF
accrues undistributed dividends and bleeds an expense ratio, which makes SPYx10 sawtooth 17-25
points under SPX.

The Index entitlement was purchased on 2026-08-21 and verified the same day: `/v3/snapshot/indices`
returns `"timeframe": "REAL-TIME"` for `I:SPX`, `I:NDX` and `I:VIX`, and `/v2/aggs` serves
1-second index bars back at least a year. So the settlement underlying is now directly observable
and parity becomes the *fallback and cross-check* rather than the source.

Why that distinction is worth keeping rather than deleting parity outright: parity spot is derived
from the same quote snapshot the theos are, so its errors partially cancel against them, while an
index level is truer but arrives on a different feed with its own timestamp. The divergence between
the two is therefore a free health signal on BOTH feeds -- it goes wrong when either does. See
`pricing/atmvol.AtmFit.index_minus_parity`.

WHAT AN INDEX IS NOT
    A computed level, not a traded instrument. Three consequences that have all bitten measurements
    in this repo:

      * Every second bar has o == c == h == l. There is no range inside the interval. Estimating
        variance from h/l silently reads zero.
      * No pre-open or overnight bars exist at all. The index is only computed while constituents
        trade, which is why a pre-open volatility lookback must use SPY/QQQ.
      * High-frequency sampling is biased LOW, badly, because constituents do not all trade every
        second (the Epps effect). Measured 2026-08-20 annualized RTH realized vol: `I:SPX` 6.64% at
        1s vs 7.76% at 30s; `I:NDX` 9.26% at 1s vs 13.76% at 30s -- a 33% underestimate. `SPY` over
        the same session is flat across sampling intervals (8.23% / 7.80%) because it is one traded
        instrument. Sample indices at 30s or coarser.
"""
import datetime
import config
from massive import history as hist

# The settlement underlying for each Kalshi series family. `I:SPX` is the S&P 500 index proper --
# not `I:GSPC` (returns empty) and emphatically not SPY.
INDEX_TICKER = {"SPX": "I:SPX", "NDX": "I:NDX"}

# Default for IndexQuote.is_fresh() when called with no argument. Illustrative value; the live value
# is not published.
DEFAULT_MAX_STALE_S = 5.0

NS = 1_000_000_000


class IndexQuote:
    """One index level with its vendor timestamp. `ts_ns` is epoch NANOSECONDS.

    Note the unit trap this repo has hit twice already: /v3/snapshot `last_updated` and /v3/quotes
    `sip_timestamp` are nanoseconds, while /v2/aggs `t` and the Massive WebSocket `t` are
    milliseconds. Same vendor, same concept, different unit.
    """

    __slots__ = ("underlying", "ticker", "value", "ts_ns", "market_status", "name")

    def __init__(self, underlying, ticker, value, ts_ns, market_status=None, name=None):
        self.underlying = underlying
        self.ticker = ticker
        self.value = value
        self.ts_ns = ts_ns
        self.market_status = market_status
        self.name = name

    def age_s(self, now_ns=None):
        """Seconds since the vendor stamped this level, or None if it carries no timestamp."""
        if self.ts_ns is None:
            return None
        now_ns = now_ns if now_ns is not None else int(datetime.datetime.now(
            datetime.timezone.utc).timestamp() * NS)
        return (now_ns - self.ts_ns) / NS

    def is_fresh(self, max_stale_s=DEFAULT_MAX_STALE_S, now_ns=None):
        age = self.age_s(now_ns)
        return age is not None and 0.0 <= age <= max_stale_s

    def __repr__(self):
        return (f"IndexQuote({self.underlying} {self.value!r} "
                f"age={self.age_s()!r}s status={self.market_status!r})")


def snapshot(underlyings=("SPX", "NDX")):
    """Live index levels: {underlying: IndexQuote}. One request for all of them.

    Batched via `ticker.any_of` deliberately -- the live quote cycle should not pay two round
    trips to learn two numbers it needs simultaneously, and two sequential calls would also give
    the two underlyings different effective timestamps.

    Underlyings the vendor does not return are simply absent from the result rather than present
    with a None value: a missing key forces the caller to decide, where a None invites arithmetic.
    """
    want = {u: INDEX_TICKER[u] for u in underlyings if u in INDEX_TICKER}
    if not want:
        return {}
    # `hist.get` builds the query string itself, so hand it params -- passing a pre-formed
    # `path?a=b` yields a trailing bare '?' and then `&apiKey=`, which is a different URL than
    # intended and only works by the server's tolerance.
    page = hist.get("/v3/snapshot/indices",
                    **{"ticker.any_of": ",".join(sorted(want.values()))})
    by_ticker = {r.get("ticker"): r for r in (page.get("results") or [])}
    out = {}
    for und, tk in want.items():
        r = by_ticker.get(tk)
        if not r or r.get("value") is None:
            continue
        out[und] = IndexQuote(underlying=und, ticker=tk, value=float(r["value"]),
                              ts_ns=r.get("last_updated"),
                              market_status=r.get("market_status"), name=r.get("name"))
    return out


def second_bars(underlying, start_date, end_date):
    """1-second index bars, ET calendar dates. See the module docstring on o == c == h == l."""
    return hist.second_bars(INDEX_TICKER[underlying], start_date, end_date)


def closes_ms(underlying, day):
    """[(epoch_ms, level)] for one ET session, ascending. The realized-vol study's substrate.

    Only `c` is read -- see the module docstring. Zero and negative levels are dropped rather
    than carried, because a log return is taken of every one of these downstream.
    """
    d = str(day)
    return [(b["t"], b["c"]) for b in second_bars(underlying, d, d)
            if b.get("c") is not None and b["c"] > 0]

