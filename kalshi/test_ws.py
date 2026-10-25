"""Pin the WebSocket book: the arithmetic, the validity rules, and the shape the quoter reads.

No socket. Frames are fed straight to `_handle`, which is the whole surface that matters -- the
transport underneath it is `websocket-client` and is not ours to test.

The properties here are the ones that decide whether a quote is priced off a book we actually
know. `book_replay` learned them the expensive way; this is the live twin.

    python -m kalshi.test_ws
"""
import sys

from harness import assert_true, check, finish

from kalshi import ws as W
from strategy.quoting import top_of_book

T1, T2 = "KXINX-26AUG25H1600-B7662", "KXINX-26AUG25H1600-B7675"


class _FakeSocket:
    """Records commands instead of sending them."""

    def __init__(self):
        self.sent = []

    def send(self, body):
        self.sent.append(body)


def feed(tickers=(T1, T2)):
    f = W.MarketFeed(list(tickers))
    f._ws = _FakeSocket()
    f._sids["orderbook_delta"] = 1
    # The pacer sleeps between commands; a test that honoured it would spend real seconds.
    f._next_cmd_t = 0.0
    W.COMMAND_MIN_INTERVAL_S = 0.0
    return f


def snapshot(ticker, yes=(), no=(), seq=1):
    return {"type": "orderbook_snapshot", "seq": seq, "msg": {
        "market_ticker": ticker,
        "yes_dollars_fp": [[p, s] for p, s in yes],
        "no_dollars_fp": [[p, s] for p, s in no]}}


def delta(ticker, side, price, d, seq=2):
    return {"type": "orderbook_delta", "seq": seq, "msg": {
        "market_ticker": ticker, "side": side,
        "price_dollars": price, "delta_fp": d}}


print("a snapshot makes a book, and it reads in the shape the ladder already parses")
f = feed()
f._handle(snapshot(T1, yes=[("0.3900", "40.00"), ("0.4000", "10.00")],
                   no=[("0.5500", "25.00"), ("0.5800", "15.00")]))
e = f.entry(T1)
bb, ba, bsz, asz = top_of_book(e)
check("best bid is the LAST yes level, not the first", bb, 0.40)
check("  with its size", bsz, 10.0)
# A NO bid at 0.58 is a YES offer at 0.42. Getting this inversion wrong silently produces a book
# whose bid is above its ask, which is why it is pinned here as well as in quoting.py.
check("best ask inverts the best NO bid: 1 - 0.58", ba, 0.42)
check("  with its size", asz, 15.0)
assert_true("  and bid < ask, i.e. the inversion is the right way round", bb < ba)

print("\n  levels are ascending, so `ladders` finds the best at the end")
check("yes ladder ascending", [lv[0] for lv in e["yes"]], ["0.3900", "0.4000"])
check("no ladder ascending", [lv[0] for lv in e["no"]], ["0.5500", "0.5800"])

print("\ndeltas move the book")
f._handle(delta(T1, "yes", "0.4100", "20.00", seq=2))
bb, _, bsz, _ = top_of_book(f.entry(T1))
check("a new best bid appears", (bb, bsz), (0.41, 20.0))
f._handle(delta(T1, "yes", "0.4100", "-20.00", seq=3))
bb, _, _, _ = top_of_book(f.entry(T1))
check("and a level taken to zero is REMOVED, not left at 0", bb, 0.40)
check("  the level is gone from the ladder", ["0.4100" in lv for lv in f.entry(T1)["yes"]],
      [False, False])

print("\nan unknown book is None -- never an empty one")
# The distinction is the whole point: `decide_ladder` treats an empty book as a one-sided market
# worth quoting into, and an absent book as nothing to price against.
check("a market we have no snapshot for reads None", f.entry(T2), None)
check("  and it is simply absent from books()", T2 in f.books(), False)
check("  while the known one is present", T1 in f.books(), True)
assert_true("  None is not the same as an empty book",
            f.entry(T2) is None and top_of_book({"yes": [], "no": []}) == (None, None, None, None))

print("\nA SEQUENCE GAP INVALIDATES EVERY BOOK ON THE SUBSCRIPTION")
# `seq` is per SUBSCRIPTION, not per market, so a missing frame could have belonged to any of
# them. Invalidating only the market the next frame mentions would leave the others quotable and
# wrong. Same rule as storage/book_replay.
f = feed()
f._handle(snapshot(T1, yes=[("0.4000", "10.00")], seq=1))
f._handle(snapshot(T2, yes=[("0.6000", "10.00")], seq=2))
check("both books valid", sorted(f.books()), [T1, T2])
n_before = len(f._ws.sent)
f._handle(delta(T1, "yes", "0.4100", "5.00", seq=9))       # 3..8 missing
check("the gap is counted", f.gaps, 1)
check("  BOTH books are invalidated, not just T1", f.books(), {})
assert_true("  and a snapshot was requested to repair them", len(f._ws.sent) > n_before)
check("  the request is get_snapshot, not a resubscribe",
      "get_snapshot" in f._ws.sent[-1], True)

print("\n  a delta arriving on an invalidated book is DISCARDED, not applied to a stale base")
f._handle(delta(T1, "yes", "0.4200", "5.00", seq=10))
check("still invalid", f.entry(T1), None)
print("  ... and a fresh snapshot repairs exactly that market")
f._handle(snapshot(T1, yes=[("0.4300", "7.00")], seq=11))
check("T1 is back", top_of_book(f.entry(T1))[0], 0.43)
check("  T2 is still refused -- it has not been re-snapshotted", f.entry(T2), None)

print("\na NEGATIVE size is proof of a missed frame: refuse the book, do not clamp it")
# `Book.apply` raises rather than flooring at zero, because a floored book looks plausible and is
# wrong. Here that must translate into refusing to quote the market.
f = feed()
f._handle(snapshot(T1, yes=[("0.4000", "10.00")], seq=1))
f._handle(delta(T1, "yes", "0.4000", "-25.00", seq=2))     # more than is there
check("the book is refused", f.entry(T1), None)
check("  and counted as a gap", f.gaps, 1)
assert_true("  a snapshot was requested for it", any("get_snapshot" in s for s in f._ws.sent))

print("\nthe control frames that advance seq but carry no data")
# Counting only data frames reports a phantom gap on every `ok`.
f = feed()
f._handle(snapshot(T1, yes=[("0.4000", "10.00")], seq=1))
f._handle({"type": "ok", "seq": 2, "msg": {}})
f._handle(delta(T1, "yes", "0.4100", "5.00", seq=3))
check("no phantom gap across an `ok`", f.gaps, 0)
check("  and the book survived", top_of_book(f.entry(T1))[0], 0.41)

print("\nfills are drained exactly once")
f = feed()
f._handle({"type": "fill", "msg": {"market_ticker": T1, "count": "10.00"}})
f._handle({"type": "fill", "msg": {"market_ticker": T2, "count": "5.00"}})
check("two fills queued", len(f.drain_fills()), 2)
check("  and the second drain is empty -- each fill applies once", len(f.drain_fills()), 0)

print("\nPOSITIONS -- the frame type is SINGULAR and the field is position_fp")
# Both of these were wrong until 2026-08-25, in the code AND in this test, so the gate was green
# and `_positions` would have stayed empty forever on a real socket. `market_positions` is the
# CHANNEL; the frame's own `type` is `market_position` (yaml: marketPositionPayload).
f = feed()
f._handle({"type": "market_position", "msg": {"market_ticker": T1, "position_fp": "10.00"}})
check("a singular market_position frame lands", f.position(T1), 10.0)
f._handle({"type": "market_position", "msg": {"market_ticker": T1, "position_fp": "-5.00"}})
check("  latest-wins, not a queue", f.position(T1), -5.0)
check("  and it is SIGNED -- negative is long NO", f.position(T1) < 0, True)
check("  a market we never traded is flat, not unknown", f.position(T2), 0.0)

print("  the PLURAL spelling is not a position frame and must not be treated as one")
f2 = feed()
f2._handle({"type": "market_positions", "msg": {"market_ticker": T1, "position_fp": "99.00"}})
check("ignored", f2.position(T1), 0.0)

print("  a fill carries post_position_fp, so position is current with the trade")
# This is the primary source: it cannot be out of step with the fill it arrived with.
f = feed()
f._handle({"type": "fill", "msg": {"market_ticker": T1, "count_fp": "20.00",
                                   "post_position_fp": "20.00", "order_id": "o1"}})
check("position follows the fill", f.position(T1), 20.0)
check("  and the fill is still queued for RestingBook", len(f.drain_fills()), 1)
f._handle({"type": "fill", "msg": {"market_ticker": T1, "count_fp": "20.00",
                                   "post_position_fp": "40.00", "order_id": "o2"}})
check("  a second fill advances it", f.position(T1), 40.0)

print("  malformed values are ignored, not crashed on or zeroed")
f._handle({"type": "market_position", "msg": {"market_ticker": T1, "position_fp": None}})
check("None leaves the last good value", f.position(T1), 40.0)
f._handle({"type": "market_position", "msg": {"market_ticker": T1, "position_fp": "banana"}})
check("  so does junk", f.position(T1), 40.0)
f._handle({"type": "market_position", "msg": {"position_fp": "7.00"}})
check("  a frame with no ticker is dropped", len(f.positions()), 1)

print("\nerror 27 (command rate limit) backs off; a snapshot clears it")
# Unpublished limit, so the only honest response is to stop asking for a while.
f = feed()
check("no backoff to start", f._cmd_backoff, 0.0)
f._handle({"type": "error", "msg": {"code": W.ERR_TOO_MANY_COMMANDS, "msg": "too many"}})
b1 = f._cmd_backoff
assert_true("backoff engaged", b1 >= W.COMMAND_BACKOFF_S, f"{b1}s")
f._handle({"type": "error", "msg": {"code": W.ERR_TOO_MANY_COMMANDS, "msg": "too many"}})
assert_true("  and doubles on a second refusal", f._cmd_backoff > b1, f"{f._cmd_backoff}s")
assert_true("  bounded", f._cmd_backoff <= W.COMMAND_BACKOFF_MAX_S)
f._handle(snapshot(T1, yes=[("0.4000", "10.00")], seq=1))
check("  a snapshot means the stream is answering again -- backoff clears", f._cmd_backoff, 0.0)

print("\nerror 26 (market limit) is surfaced, never silently trimmed")
f = feed()
f._handle({"type": "error", "msg": {"code": W.ERR_MARKET_LIMIT, "msg": "too many markets"}})
check("recorded", f.command_errors[-1]["code"], W.ERR_MARKET_LIMIT)
assert_true("  and no book was quietly dropped to make room", f.tickers == [T1, T2])

print("\na terminal channel error invalidates everything and resubscribes")
f = feed()
f._handle(snapshot(T1, yes=[("0.4000", "10.00")], seq=1))
n = len(f._ws.sent)
f._handle({"type": "error", "msg": {"code": 25, "msg": "subscription buffer overflow"}})
check("books refused", f.books(), {})
check("  reconnect counted", f.reconnects, 1)
assert_true("  and it resubscribed", len(f._ws.sent) > n and "subscribe" in f._ws.sent[-1])

print("\nsnapshot requests are chunked under the unknown per-subscription cap")
f = feed([f"T{i}" for i in range(250)])
f._ws.sent.clear()
sent = f.request_snapshot(f.tickers)
check("250 markets is 3 commands at 100 each", sent, 3)
check("  and no single command names more than the cap", 
      max(len(__import__("json").loads(s)["params"]["market_tickers"]) for s in f._ws.sent),
      W.MAX_SNAPSHOT_REQUEST)

print("\nhealth reports coverage, which is what the loop gates on")
f = feed()
f._handle(snapshot(T1, yes=[("0.4000", "10.00")], seq=1))
h = f.health()
check("one of two books valid", (h["n_valid"], h["n_wanted"]), (1, 2))
check("  coverage", h["coverage"], 0.5)

finish()
