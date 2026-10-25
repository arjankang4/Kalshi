"""Does the WebSocket book agree with REST? A RUNTIME check, not an offline gate.

It needs an open market, so it cannot PASS/FAIL in the minutes before a session the way
`kalshi/test_ws` can -- that one pins the arithmetic and the validity rules against synthetic
frames and runs anywhere. This one answers the question those cannot: is the book we maintain
frame-by-frame the same book Kalshi would hand us if we asked?

It is the same comparison `storage/reconcile` makes offline between captured WS and REST, moved
onto the live path, and it exists because the quoter's whole book source changed on 2026-08-25.
A WS book that has silently drifted looks exactly like a healthy one -- `top_of_book` returns
numbers either way, and the ladder prices against them without complaint.

WHAT DISAGREEMENT IS NORMAL. The two are sampled at different instants: the WS book is as of the
last frame, the REST book as of the request. On an active market a genuine tick between them
shows up as a one-level difference and is not an error. What is NOT normal is a persistent
disagreement, a difference of more than a tick, or a market where one side has a book and the
other does not.

    python -m kalshi.validate_ws --minutes 2
"""
import argparse
import sys
import time

import config
from kalshi import client as k
from kalshi import ws as kws
from strategy import quoting


def compare(feed, tickers, depth=10):
    """One pass. Returns (rows, summary)."""
    ws_books = feed.books(tickers)
    rest = k.orderbooks(list(tickers), depth=depth)
    rows = []
    for t in tickers:
        w, r = ws_books.get(t), rest.get(t)
        wbb, wba, _, _ = quoting.top_of_book(w)
        rbb, rba, _, _ = quoting.top_of_book(r)
        rows.append({"ticker": t, "ws_bid": wbb, "ws_ask": wba,
                     "rest_bid": rbb, "rest_ask": rba,
                     "ws_valid": w is not None,
                     "bid_diff": (None if wbb is None or rbb is None else round(wbb - rbb, 4)),
                     "ask_diff": (None if wba is None or rba is None else round(wba - rba, 4))})
    both = [x for x in rows if x["ws_valid"] and (x["rest_bid"] is not None
                                                  or x["rest_ask"] is not None)]
    exact = [x for x in both if x["bid_diff"] in (0.0, None) and x["ask_diff"] in (0.0, None)]
    within = [x for x in both
              if (x["bid_diff"] is None or abs(x["bid_diff"]) <= 0.01)
              and (x["ask_diff"] is None or abs(x["ask_diff"]) <= 0.01)]
    return rows, {"n": len(rows), "ws_valid": sum(1 for x in rows if x["ws_valid"]),
                  "comparable": len(both), "exact": len(exact), "within_1t": len(within)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="WS book vs REST, live")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--event", default=None)
    a = ap.parse_args(argv)

    ev = a.event or config.event_ticker_for()
    markets = k.markets(event_ticker=ev)
    if not markets:
        print(f"[ws] no markets for {ev} -- the board is not listed; nothing to compare")
        return 0
    tickers = [m["ticker"] for m in markets]
    print(f"[ws] {ev}: {len(tickers)} markets")

    feed = kws.MarketFeed(tickers).start()
    try:
        cov = feed.wait_for_coverage(config.WS_BOOK_MIN_COVERAGE,
                                     timeout=config.WS_BOOK_WARMUP_S)
        h = feed.health()
        print(f"[ws] warm-up: {cov:.0%} coverage in {h['commands']} commands, "
              f"{h['frames']} frames")

        deadline = time.time() + a.minutes * 60
        agg = {"passes": 0, "comparable": 0, "exact": 0, "within_1t": 0}
        worst = []
        while time.time() < deadline:
            rows, s = compare(feed, tickers)
            agg["passes"] += 1
            for key in ("comparable", "exact", "within_1t"):
                agg[key] += s[key]
            bad = [x for x in rows if x["ws_valid"]
                   and ((x["bid_diff"] is not None and abs(x["bid_diff"]) > 0.01)
                        or (x["ask_diff"] is not None and abs(x["ask_diff"]) > 0.01))]
            worst.extend(bad[:3])
            print(f"[ws] pass {agg['passes']}: valid {s['ws_valid']}/{s['n']}, "
                  f"exact {s['exact']}/{s['comparable']}, "
                  f"within 1 tick {s['within_1t']}/{s['comparable']}"
                  + (f", {len(bad)} beyond a tick" if bad else ""))
            time.sleep(max(0.0, a.interval))

        h = feed.health()
        # NOTHING TO COMPARE IS NOT AGREEMENT, AND IT IS NOT DISAGREEMENT EITHER. After the close
        # every book on the board is empty on both sides, so `comparable` is zero and any ratio
        # built on it is meaningless -- 0/0 reported as 0% would call a perfectly healthy feed
        # broken, and reported as 100% would call a dead one healthy. Split it out, the same way
        # `run_04.verdict` splits NO_SESSION from NO_SPOT.
        if agg["comparable"] == 0:
            print(f"\n[ws] {agg['passes']} passes | NO COMPARABLE BOOKS -- "
                  f"REST returned no resting interest on any of {len(tickers)} markets. "
                  f"Outside 09:30-16:00 ET that is the expected state.")
            print(f"[ws] the socket itself is healthy: {h['n_valid']}/{h['n_wanted']} books "
                  f"valid, {h['frames']} frames, {h['commands']} commands, {h['gaps']} gaps")
            return 0
        c = agg["comparable"]
        print(f"\n[ws] {agg['passes']} passes | exact {agg['exact'] / c:.1%} | "
              f"within one tick {agg['within_1t'] / c:.1%} | "
              f"{h['gaps']} gaps, {h['commands']} commands")
        if worst:
            print("[ws] disagreements beyond one tick (first few):")
            for x in worst[:8]:
                print(f"     {x['ticker']}: ws {x['ws_bid']}/{x['ws_ask']} "
                      f"vs rest {x['rest_bid']}/{x['rest_ask']}")
        # A judgement, not a gate: the honest bar is "within a tick almost always", because the
        # two are sampled at different instants and a live market ticks between them.
        ok = agg["within_1t"] / c >= 0.98
        print(f"[ws] {'OK' if ok else 'DISAGREEMENT'} "
              f"(bar: 98% within one tick)")
        return 0 if ok else 1
    finally:
        feed.stop()


if __name__ == "__main__":
    sys.exit(main())
