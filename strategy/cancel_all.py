"""Cancel every resting order on today's index events -- the shared kill path.

Importable as well as runnable: `run_04` calls `cancel_events()` from its `finally` block, so this
is the kill path rather than a thing you have to remember to type. Orders are good_till_canceled,
so anything left resting stays there until the market closes.

TWO FAILURES THIS FILE HAS NOW HAD, and they are the same failure:

  1. `resting_orders` was unpaged against a `limit` that defaults to 100, so a full board of 180
     orders was cancelled 100-at-a-time and the verifying re-read was truncated the same way.
     Fixed 2026-08-24 in `kalshi/client.resting_orders`.
  2. **This module named only `{series}-{stamp}H1600`, for two series** -- so once the hourly
     ladders were quoted, *every hourly order was invisible to the kill path and would have rested
     until its market closed*. And `cancel_events` did `events[:10]` against a
     `MultipleEventTickerQuery` cap of 10, **silently**, so naming more events would have dropped
     the excess without a word. Fixed 2026-08-25.

The shape of both: a query that quietly answers about less than you asked, on the one path whose
entire job is to leave nothing behind. So the rules here are (a) name every event we could
possibly have quoted, (b) chunk rather than truncate, and (c) verify by re-reading rather than by
trusting the cancel responses.

`/portfolio/orders` has no `series_ticker` filter -- only `MultipleEventTickerQuery` (max 10) and
`market_tickers` -- so the event list is built explicitly and chunked.
"""
import sys

import config
from kalshi import client as k

# Kalshi's MultipleEventTickerQuery cap. Chunk on it; never slice to it.
MAX_EVENTS_PER_QUERY = 10

# Series that list ONE event a day, at the 4pm close.
DAILY_SERIES = ("KXINX", "KXNASDAQ100", "KXINXDUD", "KXNASDAQDUD")
# Series that list an event for every hour. Measured 2026-08-21: every remaining hour is listed
# and active simultaneously and the set only shrinks through the day (10:21 ET showed H1100-H1600,
# 11:49 showed H1200-H1600), so naming all seven always covers whatever is still open.
HOURLY_SERIES = ("KXINXU", "KXNASDAQ100U")
HOURS = (1000, 1100, 1200, 1300, 1400, 1500, 1600)

# Both index families, even though only SPX is quoted today. Cancelling an event we never quoted
# is a harmless no-op that costs one slot in a chunked query; omitting one is orders resting on a
# live market that nothing will pull. That asymmetry is the whole argument.
QUOTED_SERIES = DAILY_SERIES + HOURLY_SERIES


def today_events(daily=DAILY_SERIES, hourly=HOURLY_SERIES, hours=HOURS, date=None):
    """Every event ticker we could hold an order on today. 18 of them by default."""
    d = date or config.today_et_date()
    stamp = f"{d:%y%b%d}".upper()
    out = [f"{s}-{stamp}H1600" for s in daily]
    out += [f"{s}-{stamp}H{h}" for s in hourly for h in hours]
    return out


def _chunks(events, n=MAX_EVENTS_PER_QUERY):
    for i in range(0, len(events), n):
        yield events[i:i + n]


def resting_across(events):
    """Every resting order across `events`, chunked. One flat list."""
    out = []
    for chunk in _chunks(events):
        q = ",".join(chunk)
        out.extend(k.resting_orders(event_ticker=q).get("orders") or [])
    return out


def cancel_events(events, verbose=True):
    """Cancel all resting orders across `events`. Returns the number still resting.

    Verifies by re-reading rather than trusting the cancel responses -- a partial failure that
    reports success is exactly the case this exists to catch. The re-read is chunked the same way
    as the read, because a truncated verification is worse than none: it reports zero.

    No write budget is passed anywhere in this path, deliberately. `kalshi.client.write_bucket`
    records the spend and is allowed to go negative; an order we cannot cancel because our own
    bookkeeping says no is a real position we cannot close.
    """
    if not events:
        return 0
    orders = resting_across(events)
    if verbose:
        print(f"{len(orders)} resting order(s) across {len(events)} event(s) "
              f"in {(len(events) + MAX_EVENTS_PER_QUERY - 1) // MAX_EVENTS_PER_QUERY} quer(ies)")
    failed = 0
    for o in orders:
        oid = o["order_id"]
        try:
            k.cancel_order(oid)
            if verbose:
                px = o.get("price_dollars", o.get("yes_price"))
                print(f"  cancelled {oid}  {o.get('ticker')} {o.get('side')} @ {px}")
        except k.KalshiError as e:
            failed += 1
            print(f"  ! {oid}: {e}", file=sys.stderr)
    left = len(resting_across(events))
    if verbose:
        print(f"remaining resting: {left}" + (f" ({failed} cancel error(s))" if failed else ""))
    return left


def main(argv=None):
    events = today_events()
    print(f"[cancel_all] {len(events)} candidate events: {events[0]} .. {events[-1]}")
    return 0 if cancel_events(events) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
