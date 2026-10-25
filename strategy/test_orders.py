"""Pinned behaviour for the resting-order manager. No network.

    python -m strategy.test_orders
"""
import sys

import config
from harness import approx, assert_true, check
from strategy import orders as O
from strategy.quoting import Rung

# The shared list, not a local one -- see harness.py.
from harness import FAILURES as FAILS


# AN EXPLICIT THREE-RUNG LADDER, not `config.QUOTE_TIER_COUNTS`, which may carry a single rung.
# `decide_ladder`, `rung_min_edge_micros` and `tier_offsets` take any number of rungs and the rank
# arithmetic is exactly what these tests exist to pin. Reading the config's tier count here would
# have the ladder tests quietly stop testing the ladder if it were configured to one rung.
LOT = f"{config.LOT_SIZE:.2f}"
C = (LOT, LOT, LOT)
LADDER_N = sum(float(c) for c in C)          # contracts in one side's ladder
TK = O.TICK
FI, BI = O.FRONT_IMPROVE_TICKS, O.BACK_IMPROVE_TICKS
assert FI < BI, "the tests below need a tighter front deadband than back"


def ticks(n):
    """n ticks, as a price increment."""
    return round(n * TK, 4)


class one_ladder_caps:
    """Hold the per-side contract cap and the per-market dollar cap at exactly one 3-rung ladder.

    The caps are designed as one ladder (see `orders.within_risk_limits`): a side quoting its full
    ladder is then AT the cap, which is the case the replace-in-place arithmetic must survive.
    These blocks pin that arithmetic at the design point whatever the configured caps are.
    """

    def __enter__(self):
        self._saved = (config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE,
                       config.MAX_RESTING_ORDER_VALUE_PER_MARKET)
        config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE = LADDER_N
        config.MAX_RESTING_ORDER_VALUE_PER_MARKET = LADDER_N      # two-sided face, in dollars
        return self

    def __exit__(self, *exc):
        (config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE,
         config.MAX_RESTING_ORDER_VALUE_PER_MARKET) = self._saved
        return False


def rungs(prices, counts=None):
    """A decided ladder. `counts` defaults to the live tuple; the trim test passes a graded one.

    The sizes matter and are not decoration: `plan_side` derives its refill budget from the rungs
    it is handed (`budget = sum(r.count)`), so a ladder built with the wrong sizes silently gives
    the top-up a different amount of room than the counts tuple it is measured against.
    """
    cs = counts or C
    return [Rung(i, p, cs[min(i, len(cs) - 1)], 0.01, "t") for i, p in enumerate(prices)]


def seed(book, ticker, side, theo, prices, counts=None):
    st = book.state(ticker)
    st.theo = theo
    counts = counts or C
    st.set_side(side, [{"ticker": ticker, "side": side, "price": p,
                        "count": counts[min(i, len(counts) - 1)], "order_id": f"o{i}"}
                       for i, p in enumerate(prices)])
    return st


def prices(plans):
    return sorted(round(p["price"], 4) for p in plans)


# --- cold start ----------------------------------------------------------
print("cold start")
b = O.RestingBook()
b.state("T").theo = 0.42
c, p, kp = O.plan_side("T", "bid", 0.42, rungs([0.41, 0.40, 0.39]), [], C)
check("places all three", (len(c), len(p), len(kp)), (0, 3, 0))
check("at the ladder prices", prices(p), [0.39, 0.40, 0.41])
check("sized by the ladder", [x["count"] for x in p], list(C))

# --- the deadband: nothing changed, nothing sent -------------------------
print("\nunchanged theo and unchanged ladder")
b = O.RestingBook(); seed(b, "T", "bid", 0.42, [0.41, 0.40, 0.39])
c, p, kp = O.plan_side("T", "bid", 0.42, rungs([0.41, 0.40, 0.39]), b.state("T").bid, C)
check("sends nothing", (len(c), len(p)), (0, 0))
check("keeps all three", len(kp), 3)

# --- our side got BETTER: tiered thresholds ------------------------------
print("\ntheo moved our way (bids improve): the front and back deadbands differ")
LIVE3 = [0.41, 0.40, 0.39]


def shifted(n):
    return [round(x + ticks(n), 4) for x in LIVE3]


b = O.RestingBook(); seed(b, "T", "bid", 0.42, LIVE3)
c, p, kp = O.plan_side("T", "bid", round(0.42 + ticks(FI - 1), 4), rungs(shifted(FI - 1)),
                       b.state("T").bid, C)
check("a gain one tick short of the front deadband moves nothing", len(p), 0)
b = O.RestingBook(); seed(b, "T", "bid", 0.42, LIVE3)
c, p, kp = O.plan_side("T", "bid", round(0.42 + ticks(FI), 4), rungs(shifted(FI)),
                       b.state("T").bid, C)
check("front (FI-tick gain) moves", any(abs(x["price"] - shifted(FI)[0]) < 1e-9 for x in p), True)
check("back two (FI-tick gain) do NOT", len(p), 1)
check("  so two are kept", len(kp), 2)

b = O.RestingBook(); seed(b, "T", "bid", 0.42, LIVE3)
c, p, kp = O.plan_side("T", "bid", round(0.42 + ticks(BI), 4), rungs(shifted(BI)),
                       b.state("T").bid, C)
check("a BI-tick gain moves all three", len(p), 3)

# --- our side got WORSE: sit tight until the edge is negative ------------
print("\ntheo moved against us, edge still positive")
b = O.RestingBook(); seed(b, "T", "bid", 0.42, [0.41, 0.40, 0.39])
c, p, kp = O.plan_side("T", "bid", 0.415, rungs([0.405, 0.395, 0.385]), b.state("T").bid, C)
check("nothing is cancelled", len(c), 0)
check("nothing is placed", len(p), 0)
check("all three kept", len(kp), 3)

print("\ntheo moved through our front bid -> that one is negative edge")
b = O.RestingBook(); seed(b, "T", "bid", 0.42, [0.41, 0.40, 0.39])
c, p, kp = O.plan_side("T", "bid", 0.405, rungs([0.395, 0.385, 0.375]), b.state("T").bid, C)
check("the through-theo order is cancelled", [x["price"] for x in c], [0.41])
check("  for the right reason", c[0]["reason"], "edge_negative")
check("the other two are kept", sorted(x["price"] for x in kp), [0.39, 0.40])

# --- THE BACKFILL the brief describes -----------------------------------
print("\nbackfill: cancel the dead front rung, top up BEHIND the survivors")
check("a third order is sent", len(p), 1)
new = p[0]
check("  it sits behind the worst survivor (0.39)", new["price"] < 0.39, True)
check("  not at the front", new["price"] < 0.40, True)
check("  reason is top_up", new["reason"], "top_up")
check("  and it does not duplicate a live price",
      new["price"] not in [0.39, 0.40], True)
check("  sized from the ladder", new["count"], C[0])
check("  so the side holds exactly one ladder",
      sum(float(x["count"]) for x in kp) + float(new["count"]), LADDER_N)

# THE TRIM, exercised deliberately. Flat sizes make the remainder always a whole rung and the trim
# a no-op -- so it is checked here against a GRADED tuple instead. It is not dead code: it is what
# stops a top-up from refilling at its slot's size when the survivors already hold most of the
# ladder (here 5 + 30 + 30 = 65 contracts against a 50-contract ladder). Losing the assertion with
# the tuple would lose the guard.
GRADED = ("5.00", "15.00", "30.00")
bt = O.RestingBook(); seed(bt, "T", "bid", 0.42, [0.41, 0.40, 0.39], counts=list(GRADED))
ct, pt, kpt = O.plan_side("T", "bid", 0.405, rungs([0.395, 0.385, 0.375], GRADED),
                          bt.state("T").bid, GRADED)
check("  under 5/15/30 the survivors hold 45 of 50",
      sum(float(x["count"]) for x in kpt), 45.0)
check("  so the refill is TRIMMED to the remaining 5, not its slot's 30", pt[0]["count"], "5.00")
check("    and the side lands on exactly one ladder",
      sum(float(x["count"]) for x in kpt) + float(pt[0]["count"]), 50.0)

print("\nsame on the ask side")
b = O.RestingBook(); seed(b, "T", "ask", 0.42, [0.43, 0.44, 0.45])
c, p, kp = O.plan_side("T", "ask", 0.435, rungs([0.445, 0.455, 0.465]), b.state("T").ask, C)
check("the through-theo offer is cancelled", [x["price"] for x in c], [0.43])
check("two kept", sorted(x["price"] for x in kp), [0.44, 0.45])
check("one placed behind the worst (0.45)", p[0]["price"] > 0.45, True)

# --- a rung that would be untradeable is not backfilled -----------------
print("\nbackfill never emits an untradeable or negative-edge price")
b = O.RestingBook(); seed(b, "T", "bid", 0.03, [0.02])
c, p, kp = O.plan_side("T", "bid", 0.03, rungs([0.02, 0.01, 0.00]), b.state("T").bid, C)
check("no rung below MIN_PRICE", all(x["price"] >= config.MIN_PRICE for x in p), True)
check("no rung with non-positive edge",
      all(x["price"] < 0.03 for x in p), True)

# --- risk limits ---------------------------------------------------------
print("\nrisk limits")
b2 = O.RestingBook()
CAP_SIDE = float(config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE)
ok, _ = O.within_risk_limits(b2, [{"ticker": "T", "side": "bid", "count": c_} for c_ in C])
check("the exact ladder fits the per-side cap", ok, True)
ok, _ = O.within_risk_limits(b2, [{"ticker": "T", "side": "bid", "count": f"{CAP_SIDE:.2f}"}])
check("  as does a side filled exactly to the cap", ok, True)
ok, why = O.within_risk_limits(b2, [{"ticker": "T", "side": "bid", "count": f"{CAP_SIDE:.2f}"},
                                    {"ticker": "T", "side": "bid", "count": "1.00"}])
check("one more contract breaches it", ok, False)
with one_ladder_caps():
    ok, why = O.within_risk_limits(b2, [{"ticker": "T", "side": "bid", "count": c_} for c_ in C]
                                   + [{"ticker": "T", "side": "bid", "count": C[0]}])
    check("  with the cap at one ladder, one more rung breaches it", ok, False)
b3 = O.RestingBook()
ok, why = O.within_risk_limits(
    b3, [{"ticker": f"T{i}", "side": "bid", "count": "50.00"} for i in range(200)])
check("the session cap catches a runaway board", ok, False)
check("  and names the limit", any("total" in w for w in why), True)

# The dollar cap is the only limit that binds on PRICE: the same contract count passes at a low
# price and breaches at a high one, while the count caps stay silent.
NV = 80
CNT = f"{CAP_SIDE:.2f}"
assert NV * CAP_SIDE <= config.MAX_RESTING_CONTRACTS_TOTAL
assert NV * CAP_SIDE * 0.90 > config.MAX_RESTING_ORDER_VALUE > NV * CAP_SIDE * 0.10
b4v = O.RestingBook()
cheap = [{"ticker": f"T{i}", "side": "bid", "price": 0.02, "count": CNT} for i in range(NV)]
ok, why = O.within_risk_limits(b4v, cheap)
check(f"{NV * CAP_SIDE:,.0f} contracts at 2c is well inside the dollar cap", ok, True)
rich = [{"ticker": f"T{i}", "side": "bid", "price": 0.90, "count": CNT} for i in range(NV)]
ok, why = O.within_risk_limits(b4v, rich)
check("the SAME contract count at 90c breaches it", ok, False)
check("  and names the dollar limit", any("resting value" in w for w in why), True)
check("  while the count caps stay silent", any("total" in w for w in why), False)

# A resting OFFER is a NO buy: selling YES at 0.90 risks 0.10, not 0.90.
offers = [{"ticker": f"T{i}", "side": "ask", "price": 0.90, "count": CNT} for i in range(NV)]
ok, _ = O.within_risk_limits(b4v, offers)
check("the same orders as OFFERS cost a ninth of that and fit", ok, True)
check("  offer value is (1-p)*count", round(O.order_value("ask", 0.90, "50.00"), 6), 5.0)
check("  bid value is p*count", round(O.order_value("bid", 0.90, "50.00"), 6), 45.0)

# What is already resting counts, not just what we are about to send.
b5v = O.RestingBook()
seed(b5v, "T", "bid", 0.90, [0.90], counts=["50.00"])
check("the book's own resting value is seen", round(b5v.value(), 4), 45.0)

# --- THE 2026-08-24 REVIEW CLUSTER --------------------------------------
#
# Three defects, one code path, and the reason none of them was caught here: every risk-limit
# check above runs against an EMPTY RestingBook, so the replace-in-place arithmetic -- the only
# arithmetic the live loop ever does after its first cycle -- was never exercised.

print("\nrequoting a full side does NOT trip the per-side cap")
_caps = one_ladder_caps().__enter__()
b6 = O.RestingBook()
seed(b6, "T", "bid", round(0.42 + ticks(FI), 4), LIVE3)
check("  the side starts at the cap", b6.contracts("T", "bid"),
      float(config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE))
c6, p6, kp6 = O.plan_side("T", "bid", round(0.42 + ticks(FI), 4), rungs(shifted(FI)),
                          b6.state("T").bid, C)
check("  an FI-tick improvement replaces the front rung", (len(c6), len(p6)), (1, 1))
ok, why = O.within_risk_limits(b6, p6)                       # cancels NOT netted out
check("  ignoring the cancels blocks it (the shipped bug)", ok, False)
ok, why = O.within_risk_limits(b6, p6, c6)                   # cancels netted out
check("  netting them out allows it", ok, True)
check("    and reports no violation", why, [])
_caps.__exit__(None, None, None)

print("\na BLOCKED plan must not send its replacement cancels naked")
mixed = [{"ticker": "T", "side": "bid", "price": 0.41, "count": LOT,
          "reason": f"improve_{FI}t"},
         {"ticker": "T", "side": "bid", "price": 0.39, "count": "50.00",
          "reason": "edge_negative"}]
safe = O.reducing_only(mixed)
check("  only the exposure-reducing cancel survives", [x["reason"] for x in safe],
      ["edge_negative"])

print("\nthe front rung is re-placed after a cancel, not skipped")
# live 0.41/0.40/0.39; theo 0.405 kills 0.41. want[0] must go to the order that is now in front.
b7 = O.RestingBook(); seed(b7, "T", "bid", 0.42, [0.41, 0.40, 0.39])
c7, p7, kp7 = O.plan_side("T", "bid", 0.405, rungs([0.40, 0.39, 0.38]), b7.state("T").bid, C)
check("  the through-theo rung is cancelled", [x["price"] for x in c7], [0.41])
# 0.40 is now rank 0, so it is measured against want[0]=0.40 -- a 0-tick gain, so it is KEPT.
# Under the old index-based matching it was measured against want[1]=0.39, i.e. a target BEHIND
# it, and the ladder lost its front rung for good.
check("  the new front rung is measured against want[0]",
      sorted(x["price"] for x in kp7), [0.39, 0.40])
check("  and the ladder never exceeds one ladder of size",
      sum(float(x["count"]) for x in kp7) + sum(float(x["count"]) for x in p7), LADDER_N)

print("\na FAILED cancel leaves the order visible in the local book")


class _FakeK:
    """Stands in for `kalshi.client`. The first cancel fails, the second succeeds."""

    calls = []

    @staticmethod
    def write_budget():
        return {"max_batch": 20}

    @staticmethod
    def batch_cancel_orders(ids):
        _FakeK.calls.append(list(ids))
        # reduced_by == 0 is how a failed item comes back -- NOT an `error` key.
        return [{"order_id": ids[0], "reduced_by": 0},
                {"order_id": ids[1], "reduced_by": float(C[1])}]

    @staticmethod
    def batch_create_orders(bodies):
        return [{"order_id": f"n{i}"} for i, _ in enumerate(bodies)]

    @staticmethod
    def order_body(t, side, price, count, client_order_id=None):
        return {"ticker": t, "side": side, "price": price, "count": count}

    @staticmethod
    def fp(x, places=4):
        return f"{x:.{places}f}"


b8 = O.RestingBook(); seed(b8, "T", "bid", 0.42, [0.41, 0.40])
to_cancel = [{**o, "reason": "edge_negative"} for o in b8.state("T").bid]
_real_k, O.k = O.k, _FakeK
try:
    r8 = O.apply([], to_cancel, b8, live=True)
finally:
    O.k = _real_k
check("  one cancel confirmed, one not", (r8["cancelled"], len(r8["cancel_errors"])), (1, 1))
check("  the FAILED order is still in the book", [o["price"] for o in b8.state("T").bid],
      [0.41])
# Visible means retryable: the next reconcile plans a cancel for it. Stripping it locally would
# have left it resting on the exchange with nothing in our state pointing at it.
_, retry, _ = O.reconcile(b8, {})
check("    so the next reconcile plans another cancel for it",
      [(x["price"], x["reason"]) for x in retry], [(0.41, "no_longer_quoting")])

# --- apply round-trips the book -----------------------------------------
print("\napply (dry)")
b4 = O.RestingBook(); b4.state("T").theo = 0.42
c, p, kp = O.plan_side("T", "bid", 0.42, rungs([0.41, 0.40, 0.39]), [], C)
r = O.apply(p, c, b4, live=False)
check("reports both legs", (r["cancelled"], r["placed"], r["dry_run"]), (0, 3, True))
check("book holds three", len(b4), 3)
check("contracts sum to the ladder", b4.contracts("T", "bid"), LADDER_N)
st = b4.state("T")
check("state record is the 7-item picture",
      (st.as_record()["ticker"], len(st.as_record()["bid"]), len(st.as_record()["ask"])),
      ("T", 3, 0))

# --- apply under a write budget -----------------------------------------
# Everything below is the 429 path, which until 2026-08-24 did not exist: `apply` sent every
# chunk back-to-back with no accounting and no `try`, so the first rejection unwound the whole
# function after the cancels had gone and before their places had.
from kalshi.client import CANCEL_TOKEN_COST, KalshiError, ORDER_TOKEN_COST, TokenBucket


class _Rejecting:
    """A client whose Nth create batch 429s. Cancels always succeed.

    `bucket` mirrors what the REAL client does and the stub otherwise would not: `_request`
    debits `write_bucket` on every write, which is why `apply` only ever asks (`can_afford`) and
    never charges. A stub that skipped the debit would let `apply` spend an empty bucket forever
    and the budget tests below would pass against a budget that never moved.
    """

    def __init__(self, fail_on=1, bucket=None):
        self.fail_on, self.creates, self.cancels = fail_on, 0, 0
        self.bucket = bucket

    def write_budget(self):
        return {"max_batch": 2}

    def _charge(self, n):
        if self.bucket is not None:
            self.bucket.debit(n)

    def batch_cancel_orders(self, ids):
        self.cancels += 1
        self._charge(CANCEL_TOKEN_COST * len(ids))
        return [{"order_id": i, "reduced_by": 10} for i in ids]

    def batch_create_orders(self, bodies):
        self.creates += 1
        self._charge(ORDER_TOKEN_COST * len(bodies))
        if self.creates == self.fail_on:
            raise KalshiError(429, '{"error":{"code":"too_many_requests"}}')
        return [{"order_id": f"n{self.creates}-{i}"} for i, _ in enumerate(bodies)]

    @staticmethod
    def order_body(t, side, price, count, client_order_id=None):
        return {"ticker": t, "side": side, "price": price, "count": count}

    @staticmethod
    def fp(x, places=4):
        return f"{x:.{places}f}"


def run_apply(fake, place, cancel, book, **kw):
    real, O.k = O.k, fake
    try:
        return O.apply(place, cancel, book, live=True, **kw)
    finally:
        O.k = real


print("\na 429 mid-batch stops cleanly instead of unwinding the cycle")
bk = O.RestingBook()
places = [{"ticker": f"T{i}", "side": "bid", "price": 0.40, "count": "10.00"} for i in range(6)]
fake = _Rejecting(fail_on=2)                     # max_batch 2 -> 3 chunks, the 2nd rejects
r = run_apply(fake, places, [], bk)
check("the first chunk landed", r["placed"], 2)
check("  the 429 is counted as its own thing", r["rate_limited"], 1)
check("  it did not raise", isinstance(r, dict), True)
check("  and the remaining chunk is reported unsent, not silently dropped",
      r["unsent_places"], 2)
check("  the loop stopped rather than burning chunk 3 on a closed door", fake.creates, 2)

print("\na non-429 batch error does not stop the remaining chunks")
# A 500 or a malformed body is about THAT batch. Only a rate limit is about the next one too.
class _Once(_Rejecting):
    def batch_create_orders(self, bodies):
        self.creates += 1
        if self.creates == 1:
            raise KalshiError(500, "boom")
        return [{"order_id": f"n{self.creates}-{i}"} for i, _ in enumerate(bodies)]

bk = O.RestingBook()
r = run_apply(_Once(), places, [], bk)
check("chunks 2 and 3 still went", r["placed"], 4)
check("  and it is not counted as a rate limit", r["rate_limited"], 0)

print("\na place whose paired cancel did NOT land is dropped -- it would double the rung")
class _CancelFails(_Rejecting):
    def batch_cancel_orders(self, ids):
        self.cancels += 1
        # reduced_by 0 is how a failed item comes back; the order is STILL RESTING.
        return [{"order_id": i, "reduced_by": 0} for i in ids]

bk = O.RestingBook()
pid = O._pair_id("T", "bid", 0.40)
cx = {"ticker": "T", "side": "bid", "price": 0.40, "count": "10.00",
      "reason": f"improve_{FI}t", "pair": pid, "order_id": "old-1"}
px = {"ticker": "T", "side": "bid", "price": 0.41, "count": "10.00",
      "reason": "improved", "pair": pid}
r = run_apply(_CancelFails(), [px], [cx], bk)
check("the cancel did not land", r["cancelled"], 0)
check("  so the replacement is NOT placed", r["placed"], 0)
check("  and that is reported", r["orphaned_places"], 1)
check("  leaving exactly one order in the book, not two", len(bk), 0)

print("  ... and when the cancel DOES land, the replacement goes")
bk = O.RestingBook()
r = run_apply(_Rejecting(fail_on=99), [dict(px)], [dict(cx)], bk)
check("cancel confirmed", r["cancelled"], 1)
check("  replacement placed", r["placed"], 1)
check("  none orphaned", r["orphaned_places"], 0)

print("  ... and a dry run is not held to a confirmation that never happened")
bk = O.RestingBook()
r = O.apply([dict(px)], [dict(cx)], bk, live=False)
check("dry run still exercises the ladder", r["placed"], 1)
check("  nothing orphaned", r["orphaned_places"], 0)

print("\nan exhausted budget leaves work UNSENT rather than pushing it into a 429")
bk = O.RestingBook()
empty = TokenBucket(capacity=0.0, refill_rate=0.0)
fake = _Rejecting(fail_on=99)
r = run_apply(fake, places, [], bk, budget=empty)
check("nothing was sent", r["placed"], 0)
check("  the client was never called", fake.creates, 0)
check("  all 6 reported unsent", r["unsent_places"], 6)
check("  and NOT as an error -- we declined, Kalshi did not refuse", len(r["place_errors"]), 0)
check("  nor as a rate limit", r["rate_limited"], 0)

print("  ... a budget that covers two chunks sends two")
bk = O.RestingBook()
b = TokenBucket(capacity=ORDER_TOKEN_COST * 4, refill_rate=0.0)
fake = _Rejecting(fail_on=99, bucket=b)
r = run_apply(fake, places, [], bk, budget=b)
check("4 of 6 placed", r["placed"], 4)
check("  2 unsent", r["unsent_places"], 2)
check("  and the bucket is empty, not negative", b.available(), 0.0)

print("  ... the kill path passes no budget and is never gated")
bk = O.RestingBook()
fake = _Rejecting(fail_on=99)
r = run_apply(fake, places, [], bk, budget=None)
check("all 6 sent with no budget object", r["placed"], 6)

# --- fills reduce the local book ----------------------------------------
# Before 2026-08-25 nothing did this. `RestingBook` was written from create responses and never
# reconciled against the exchange until `cancel_all` at exit, so a filled order lived on as a
# phantom: measured for edge by `plan_side`, counted against the per-side risk cap, and finally
# cancelled by an order_id the exchange had already retired.
print("\na fill reduces the order it names")
bf = O.RestingBook()
bf.state("T").bid = [{"ticker": "T", "side": "bid", "price": 0.40, "count": "50.00",
                      "order_id": "oid-1"},
                     {"ticker": "T", "side": "bid", "price": 0.39, "count": "15.00",
                      "order_id": "oid-2"}]
check("a partial fill decrements",
      (O.apply_fill(bf, {"order_id": "oid-1", "market_ticker": "T", "count_fp": "15.00"}),
       bf.state("T").bid[0]["count"]), (1, "35.00"))
check("  the count stays a fixed-point STRING, as the order body needs",
      isinstance(bf.state("T").bid[0]["count"], str), True)
check("  and the other rung is untouched", bf.state("T").bid[1]["count"], "15.00")

print("  a full fill REMOVES it -- a zero-size resting order is not a thing")
O.apply_fill(bf, {"order_id": "oid-1", "market_ticker": "T", "count_fp": "35.00"})
check("one rung left", [o["order_id"] for o in bf.state("T").bid], ["oid-2"])
check("  and the side's contract count follows", bf.contracts("T", "bid"), 15.0)

print("  an over-fill removes rather than going negative")
O.apply_fill(bf, {"order_id": "oid-2", "market_ticker": "T", "count_fp": "999.00"})
check("side is empty", bf.state("T").bid, [])

print("  matched by order_id, not by price -- a fill is an exchange event and carries the id")
bp = O.RestingBook()
bp.state("T").ask = [{"ticker": "T", "side": "ask", "price": 0.60, "count": "10.00",
                      "order_id": "oid-9"}]
check("finds it on the ASK side too",
      O.apply_fill(bp, {"order_id": "oid-9", "market_ticker": "T", "count_fp": "10.00"}), 1)
check("  and removes it", bp.state("T").ask, [])

print("  a fill we do not recognise is a no-op, not an error")
# Normal for a market we stopped quoting between the fill and the drain, and for anything the
# account trades outside this loop.
check("unknown order_id", O.apply_fill(bf, {"order_id": "nope", "market_ticker": "T",
                                            "count_fp": "1.00"}), 0)
check("  no order_id at all", O.apply_fill(bf, {"market_ticker": "T", "count_fp": "1.00"}), 0)
check("  a zero-count frame", O.apply_fill(bf, {"order_id": "oid-2", "count_fp": "0.00"}), 0)

print("  a fill frees room under the per-side cap, which is the point")
_caps = one_ladder_caps().__enter__()
bc = O.RestingBook()
bc.state("T").theo = 0.42
bc.state("T").bid = [{"ticker": "T", "side": "bid", "price": p, "count": c, "order_id": f"o{i}"}
                     for i, (p, c) in enumerate([(0.41, C[0]), (0.40, C[1]), (0.39, C[2])])]
check("the side starts at the cap", bc.contracts("T", "bid"),
      float(config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE))
top_up_before = O.plan_side("T", "bid", 0.42, rungs([0.41, 0.40, 0.39]), bc.state("T").bid, C)[1]
check("  so nothing can be added", top_up_before, [])
O.apply_fill(bc, {"order_id": "o0", "market_ticker": "T", "count_fp": C[0]})
check("  the fill freed one lot of room", bc.contracts("T", "bid"),
      float(config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE) - float(C[0]))
cn, pl, _ = O.plan_side("T", "bid", 0.42, rungs([0.41, 0.40, 0.39]), bc.state("T").bid, C)
check("  and the ladder refills", len(pl) > 0, True)
# NET THE CANCELS. `held - cancelled + placed` is what will actually rest, and measuring
# `held + placed` double-counts every order being replaced -- the exact arithmetic error that
# `within_risk_limits` shipped with and was fixed for on 2026-08-24. Easy to make twice.
after = (bc.contracts("T", "bid")
         - sum(float(x["count"]) for x in cn)
         + sum(float(x["count"]) for x in pl))
check("    back to exactly one ladder, netting the replacement cancels", after,
      float(config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE))
ok_, viol = O.within_risk_limits(bc, pl, cn)
check("    and the risk gate agrees", (ok_, viol), (True, []))
_caps.__exit__(None, None, None)

# --- the PER-MARKET collateral cap --------------------------------------
# New 2026-08-25. Until then the dollar cap was global only, so nothing stopped one market
# absorbing the whole book -- which mattered much more once the universe went from 30 markets to
# several hundred and one market's share of the plan got correspondingly smaller.
print("\na per-market dollar cap, at its design point of one 3-tier two-sided ladder")
_caps = one_ladder_caps().__enter__()
bm = O.RestingBook()
bm.state("T").bid = [{"ticker": "T", "side": "bid", "price": 0.50, "count": c} for c in C]
bm.state("T").ask = [{"ticker": "T", "side": "ask", "price": 0.50, "count": c} for c in C]
check("a full two-sided ladder is exactly the cap", bm.value("T"),
      config.MAX_RESTING_ORDER_VALUE_PER_MARKET)
check("  and it is allowed", O.within_risk_limits(bm, [], [])[0], True)

print("  the cap holds at ANY fair -- a bid risks p, an ask risks 1-p")
for fair in (0.03, 0.50, 0.97):
    bq = O.RestingBook()
    bq.state("T").bid = [{"ticker": "T", "side": "bid", "price": fair, "count": c} for c in C]
    bq.state("T").ask = [{"ticker": "T", "side": "ask", "price": fair, "count": c} for c in C]
    approx(f"  fair {fair}", bq.value("T"), config.MAX_RESTING_ORDER_VALUE_PER_MARKET, tol=1e-9)

print("  one more rung breaches it")
extra = [{"ticker": "T", "side": "bid", "price": 0.50, "count": C[0]}]
ok, why = O.within_risk_limits(bm, extra, [])
check("blocked", ok, False)
assert_true("  and it names the DOLLAR limit, not just the count",
            any("$" in w for w in why), str(why))

print("  a REAL ladder is CHEAPER than the cap, because we quote with edge")
# The face identity holds only at zero spread. A ladder quoting e cents either side of the theo is
# (p-e)*N + (1-p-e)*N = N - 2eN, so a 1-cent edge on N contracts a side costs N - 0.02N, not N.
# That headroom is not incidental -- it is what lets a market at its cap improve a rung at all,
# because unlike the per-side CONTRACT cap the dollar cap is not price-neutral: moving a bid up a
# tick genuinely does tie up more money.
be = O.RestingBook()
be.state("T").bid = [{"ticker": "T", "side": "bid", "price": 0.49, "count": c} for c in C]
be.state("T").ask = [{"ticker": "T", "side": "ask", "price": 0.51, "count": c} for c in C]
approx("a 1c-edge ladder costs N - 2*0.01*N", be.value("T"), LADDER_N - 2 * 0.01 * LADDER_N,
       tol=1e-9)
check("  inside the cap", O.within_risk_limits(be, [], [])[0], True)

print("  ...so a requote at the cap nets out and is allowed")
# The same arithmetic error the per-side cap shipped with: measuring held+placed instead of
# held-cancelled+placed makes any requote read as a violation, and the caller's response to a
# violation is to send the reducing cancels alone, which unquotes the side.
cx = [{**be.state("T").bid[0], "reason": f"improve_{FI}t"}]
px = [{"ticker": "T", "side": "bid", "price": 0.50, "count": C[0]}]
assert_true("ignoring the cancels over-counts",
            be.value("T") + 0.50 * float(C[0]) > config.MAX_RESTING_ORDER_VALUE_PER_MARKET,
            f"{be.value('T') + 0.50 * float(C[0]):.2f}")
check("  which would block", O.within_risk_limits(be, px, [])[0], False)
check("  netting them out allows it", O.within_risk_limits(be, px, cx)[0], True)

print("  a wing market uses a third of the cap")
bw = O.RestingBook()
bw.state("W").bid = [{"ticker": "W", "side": "bid", "price": 0.50, "count": C[0]}]
bw.state("W").ask = [{"ticker": "W", "side": "ask", "price": 0.50, "count": C[0]}]
check("one rung a side", bw.value("W"), config.MAX_RESTING_ORDER_VALUE_PER_MARKET / 3)

print("  the cap is PER market -- two markets at the cap are both fine")
b2m = O.RestingBook()
for t in ("A", "B"):
    b2m.state(t).bid = [{"ticker": t, "side": "bid", "price": 0.50, "count": c} for c in C]
    b2m.state(t).ask = [{"ticker": t, "side": "ask", "price": 0.50, "count": c} for c in C]
ok, why = O.within_risk_limits(b2m, [], [])
check("both allowed", (ok, b2m.value()), (True, 2 * LADDER_N))
_caps.__exit__(None, None, None)

# --- MarketState carries a net position ---------------------------------
print("\nMarketState holds a signed net position, flat on a cold start")
bp = O.RestingBook()
check("a market we never traded is flat, not unknown", bp.state("X").position, 0.0)
bp.state("X").position = -40.0
check("  and it is signed -- negative is long NO", bp.state("X").position, -40.0)

# --- promotion on cancel -------------------------------------------------
#
# `plan_side` measures a surviving order against the slot it would OCCUPY, not against the slot it
# sits in. So cancelling the front rung immediately re-judges the next one AS a front rung: it is
# compared with want[0] and gets the front deadband instead of want[1] and the back one.
# That is the promotion, and it moves nothing on the exchange -- the survivor keeps its queue
# position at its own price.
#
# The promotion lives in `want[rank]` and the deadband tier, and this block exercises it with no
# knob at all.
print("\npromotion: the survivor is judged in the vacated slot, not its own")

bk = O.RestingBook()
seed(bk, "M", "bid", 0.1850, [0.19, 0.18, 0.17])
# theo 0.1850: 0.19 is negative and must go; 0.18 has 0.5c and survives. want[0] sits FI ticks
# above 0.18, so the promoted 0.18 is measured against it -- an FI-tick gain, which clears the
# FRONT threshold and would NOT clear the back one.
PROMO = [round(0.18 + ticks(FI), 4), 0.18, 0.17]
c, p, kp = O.plan_side("M", "bid", 0.1850, rungs(PROMO),
                       bk.state("M").side("bid"), C)
gone = [round(x["price"], 4) for x in c if x["reason"] == "edge_negative"]
check("only the negative rung is cancelled", gone, [0.19])
moved = {round(x["price"], 4): x["reason"] for x in c if x["reason"] != "edge_negative"}
check("  the promoted rung is judged against want[0] on the FRONT deadband",
      moved.get(0.18), f"improve_{FI}t")
assert_true("  and the third rung, now rank 1, keeps its back deadband",
            0.17 in {round(x["price"], 4) for x in kp}, str(kp))

# Without the promotion the 0.18 order would have been matched to want[1] = 0.18, gained nothing,
# and the touch rung 0.19 would never have been re-placed at all -- the ladder walking backwards.
check("  want[0] is re-placed rather than abandoned",
      PROMO[0] in {round(x["price"], 4) for x in p}, True)


# --- the zombie hold -----------------------------------------------------
#
# THE LOOP. Our own theo wobbles across a boundary, the front rung is cancelled for negative edge
# (correctly), and when the theo comes back `want[0]` returns to the price we just left -- so the
# survivor "gains" a tick and is requoted there. The market never moved; we paid a queue position
# for our own noise. See `config.QUOTE_ZOMBIE_HOLD_S`.
#
# Four properties, each written so it fails if the gate is inverted.
print("\nthe zombie hold: an improve waits, a cancel never does")

T0 = 1_000.0
HOLD = config.QUOTE_ZOMBIE_HOLD_S


def improved(cancels):
    return {round(x["price"], 4) for x in cancels if str(x["reason"]).startswith("improve")}


bk = O.RestingBook()
st = seed(bk, "Z", "bid", 0.1850, [0.18, 0.17, 0.16])
LADDER = rungs([round(x + ticks(FI), 4) for x in (0.18, 0.17, 0.16)])   # each gains FI ticks
UP = round(0.18 + ticks(FI) + 0.005, 4)          # a theo half a tick above want[0]

# 1. cold clock -> the improve fires, which is the pre-2026-08-26 behaviour and the control.
c0, _, _ = O.plan_side("Z", "bid", UP, LADDER, st.side("bid"), C,
                       touch=0.18, zombie_at=None, now=T0)
check("cold clock: the front rung improves", improved(c0), {0.18})

# 2. armed clock, the touch is where it was when we left -> held.
c1, p1, kp1 = O.plan_side("Z", "bid", UP, LADDER, st.side("bid"), C,
                          touch=0.18, zombie_at=T0, zombie_touch=0.18, now=T0 + 1.0)
check("armed, touch unmoved: the same improve is KEPT instead", improved(c1), set())
check("  and it is countable as its own reason",
      sorted(x["reason"] for x in kp1 if x["reason"] == "zombie_hold"), ["zombie_hold"])
check("  nothing is placed in its stead", [x for x in p1 if x.get("pair")], [])

# 3. the touch has MOVED IN since we left -> somebody really did penny us, so the hold lifts.
c2, _, _ = O.plan_side("Z", "bid", UP, LADDER, st.side("bid"), C,
                       touch=0.19, zombie_at=T0, zombie_touch=0.18, now=T0 + 1.0)
check("the touch moved 0.18 -> 0.19 while we were out: the hold lifts", improved(c2), {0.18})

# THE TEST THAT MUST NOT BE "is the touch inside our best resting order". Cancelling our own front
# rung drops our best price by a tick, so on a STATIC book that reading calls it a pennying and
# lifts the hold on exactly the re-entry it exists to block -- measured at 56.6% of them over the
# 2026-08-25 session, against 38.7% of the honest chases. Here our best resting bid is 0.18 and the
# touch is 0.18: unchanged since arming, and it must hold.
c2b, _, _ = O.plan_side("Z", "bid", UP, LADDER, st.side("bid"), C,
                        touch=0.18, zombie_at=T0, zombie_touch=0.18, now=T0 + 1.0)
check("  a static book does NOT lift it, whatever our own best price did", improved(c2b), set())
# And a touch that moved AWAY is not a pennying either -- we are more of the market, not less.
c2c, _, _ = O.plan_side("Z", "bid", UP, LADDER, st.side("bid"), C,
                        touch=0.17, zombie_at=T0, zombie_touch=0.18, now=T0 + 1.0)
check("  a touch that moved AWAY does not lift it", improved(c2c), set())
# No touch recorded at arming -> we cannot tell, so we do not change the policy.
c2d, _, _ = O.plan_side("Z", "bid", UP, LADDER, st.side("bid"), C,
                        touch=0.18, zombie_at=T0, zombie_touch=None, now=T0 + 1.0)
check("  and with no touch on record the hold still applies", improved(c2d), set())

# 4. the floor is untimed: negative edge cancels with the clock armed.
c3, _, _ = O.plan_side("Z", "bid", 0.1650, LADDER, st.side("bid"), C,
                       touch=0.18, zombie_at=T0, zombie_touch=0.18, now=T0 + 1.0)
neg = {round(x["price"], 4) for x in c3 if x["reason"] == "edge_negative"}
assert_true("negative edge is cancelled with the hold armed", {0.18, 0.17} <= neg,
            str(sorted(neg)))

# 5. the clock expires.
c4, _, _ = O.plan_side("Z", "bid", UP, LADDER, st.side("bid"), C,
                       touch=0.18, zombie_at=T0, zombie_touch=0.18, now=T0 + HOLD + 0.1)
check(f"after {HOLD:.0f}s the improve fires again", improved(c4), {0.18})

# The ask mirror, so the sign of the exception is pinned on both sides.
bka = O.RestingBook()
sta = seed(bka, "Z", "ask", 0.1850, [0.19, 0.20, 0.21])
LADDER_A = rungs([round(x - ticks(FI), 4) for x in (0.19, 0.20, 0.21)])
DOWN = round(0.19 - ticks(FI) - 0.005, 4)        # a theo half a tick below want[0]
ca, _, _ = O.plan_side("Z", "ask", DOWN, LADDER_A, sta.side("ask"), C,
                       touch=0.19, zombie_at=T0, zombie_touch=0.19, now=T0 + 1.0)
check("ask side: held with the touch unmoved", improved(ca), set())
ca2, _, _ = O.plan_side("Z", "ask", DOWN, LADDER_A, sta.side("ask"), C,
                        touch=0.18, zombie_at=T0, zombie_touch=0.19, now=T0 + 1.0)
check("  and lifted when the offer moves 0.19 -> 0.18 while we are out", improved(ca2), {0.19})

# ARMING. `apply` is the only place that knows a cancel landed, and only `edge_negative` arms.
print("\narming: only a negative-edge cancel starts the clock")
bk2 = O.RestingBook()
seed(bk2, "A", "bid", 0.1850, [0.19, 0.18])
bk2.state("A").set_touch(0.18, 0.22)
res = O.apply([], [{"ticker": "A", "side": "bid", "price": 0.19, "count": LOT,
                    "order_id": "o0", "reason": "edge_negative"}], bk2, live=False, now=T0)
check("an edge_negative cancel arms the side", bk2.state("A").zombie("bid"), T0)
check("  and stamps the touch we are leaving behind",
      bk2.state("A").zombie_at_touch("bid"), 0.18)
check("  and it is counted", res["zombies_armed"], 1)
check("  the other side is untouched", bk2.state("A").zombie("ask"), None)

bk3 = O.RestingBook()
seed(bk3, "A", "bid", 0.1850, [0.19, 0.18])
O.apply([], [{"ticker": "A", "side": "bid", "price": 0.19, "count": LOT,
              "order_id": "o0", "reason": f"improve_{FI}t", "pair": "x"}], bk3, live=False, now=T0)
check("an improve cancel does NOT arm it", bk3.state("A").zombie("bid"), None)
O.apply([], [{"ticker": "A", "side": "bid", "price": 0.18, "count": LOT,
              "order_id": "o1", "reason": "no_longer_quoting"}], bk3, live=False, now=T0)
check("neither does leaving the market", bk3.state("A").zombie("bid"), None)

# END TO END through `reconcile`, which is the seam the unit tests above skip: `apply` arms the
# clock on MarketState, `reconcile` reads it back off MarketState and hands it to `plan_side`. A
# miswiring here (wrong side, touch not stamped, clock not threaded) passes every test above.
print("\napply -> MarketState -> reconcile: the wiring, not the rule")
bkw = O.RestingBook()
stw = seed(bkw, "W", "bid", 0.1850, [0.19, 0.18, 0.17])
stw.set_touch(0.18, 0.22)                      # the book: 0.18 bid, and it will not move
# Cycle 1's ladder wants exactly what the survivors already are, so the only action is the cancel:
# the improve must not happen in the same cycle that arms the clock, or the test proves nothing.
LAD1 = {("W", "bid"): rungs([0.18, 0.17, 0.16])}
# The theo recovers; want[0] is FI ticks above the surviving 0.18 again.
LAD2 = {("W", "bid"): rungs([round(x + ticks(FI), 4) for x in (0.18, 0.17, 0.16)])}

stw.theo = 0.1850                              # 0.19 is through the theo, 0.18 is not
pl, cn, _ = O.reconcile(bkw, LAD1, now=T0)
check("only the negative rung is cancelled",
      [round(x["price"], 4) for x in cn if x["reason"] == "edge_negative"], [0.19])
O.apply(pl, cn, bkw, live=False, now=T0)
check("  the clock is armed", stw.zombie("bid"), T0)
check("  at the touch we left", stw.zombie_at_touch("bid"), 0.18)

# The theo comes back. Same static book. Without the hold this requotes 0.18 -> want[0].
stw.theo = UP
_, cn2, kp2 = O.reconcile(bkw, LAD2, now=T0 + 2.0)
check("the theo recovers and the requote is HELD",
      [x["reason"] for x in cn2 if str(x["reason"]).startswith("improve")], [])
check("  visible as zombie_hold in the keeps",
      sum(1 for x in kp2 if x["reason"] == "zombie_hold"), 1)

# Somebody posts 0.19. Now it is a real pennying and the hold lifts.
stw.set_touch(0.19, 0.22)
_, cn3, _ = O.reconcile(bkw, LAD2, now=T0 + 3.0)
check("the touch moves in and the requote goes ahead",
      sorted(round(x["price"], 4) for x in cn3 if str(x["reason"]).startswith("improve")), [0.18])

# A ladder carrying no bars at all (the default Rung) must behave exactly as it always did.
bk4 = O.RestingBook()
seed(bk4, "M", "bid", 0.4940, [0.49, 0.48, 0.47])
c5, _, _ = O.plan_side("M", "bid", 0.4940, rungs([0.49, 0.48, 0.47]),
                       bk4.state("M").side("bid"), C)
check("a ladder with no min_edge keeps the pre-2026-08-26 behaviour",
      [x["reason"] for x in c5 if x["reason"] == "edge_negative"], [])

# There is no exit bar above zero edge: a resting order is cancelled for edge only once it is
# through the theo. A test that pinned a deleted mechanism would be a trap for whoever reads it
# next; the reasoning lives in `plan_side`'s cancel branch.

print("\nthe BACKOFF CLOCK is armed by a FILL, on that side only")
BK2 = O.RestingBook()
st = BK2.state("M")
st.set_side("bid", [{"ticker": "M", "side": "bid", "price": 0.48, "count": LOT,
                     "order_id": "b1"}])
st.set_side("ask", [{"ticker": "M", "side": "ask", "price": 0.52, "count": LOT,
                     "order_id": "a1"}])
check("clean to start", (st.last_fill("bid"), st.last_fill("ask")), (None, None))
check("the fill matched",
      O.apply_fill(BK2, {"order_id": "b1", "count_fp": LOT, "market_ticker": "M"},
                   now=100.0), 1)
check("the bid clock is armed", st.last_fill("bid"), 100.0)
check("  the ask is untouched -- flattening is never taxed", st.last_fill("ask"), None)

print("\n  a fill we cannot match arms nothing")
BK3 = O.RestingBook()
BK3.state("M").set_side("bid", [{"ticker": "M", "side": "bid", "price": 0.48,
                                 "count": LOT, "order_id": "b1"}])
check("no match",
      O.apply_fill(BK3, {"order_id": "NOPE", "count_fp": LOT, "market_ticker": "M"},
                   now=100.0), 0)
check("  nothing armed", BK3.state("M").last_fill("bid"), None)

print("\n  a PARTIAL fill arms it too -- we traded, whatever the remainder says")
BK4 = O.RestingBook()
BK4.state("M").set_side("bid", [{"ticker": "M", "side": "bid", "price": 0.48,
                                 "count": LOT, "order_id": "b1"}])
O.apply_fill(BK4, {"order_id": "b1", "count_fp": "5.00", "market_ticker": "M"}, now=42.0)
check("armed on a partial", BK4.state("M").last_fill("bid"), 42.0)
check("  and the remainder is still resting", len(BK4.state("M").side("bid")), 1)

print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
sys.exit(0 if not FAILS else 1)
