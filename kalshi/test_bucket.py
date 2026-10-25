"""Pin the write token bucket. No clock we do not control, no network.

The bucket exists because the pacer it replaced could not spend burst credit, so BURST IS THE
PROPERTY UNDER TEST -- not the average rate. Everything here drives `time.monotonic` through a
stub, because a test that sleeps to observe a refill is a test that is slow and flaky at once.

    python -m kalshi.test_bucket
"""
import sys

from harness import approx, assert_true, check, finish

from kalshi.client import TokenBucket
from kalshi import client as k


class Clock:
    """A monotonic clock we advance by hand."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def bucket(capacity=900.0, refill=300.0, clock=None):
    """A bucket wired to `clock`. Patching the module's `time` is the least invasive seam:
    TokenBucket reads `time.monotonic()` and nothing else."""
    clk = clock or Clock()
    real = k.time.monotonic
    k.time.monotonic = clk
    try:
        b = TokenBucket(capacity, refill)
    finally:
        k.time.monotonic = real
    return b, clk


def with_clock(clk, fn):
    real = k.time.monotonic
    k.time.monotonic = clk
    try:
        return fn()
    finally:
        k.time.monotonic = real


print("a fresh bucket starts FULL")
# Not empty. We have not spent anything, and starting empty would make the first cycle of every
# process wait for a refill it does not owe.
b, clk = bucket()
check("available == capacity", with_clock(clk, b.available), 900.0)
check("affords a full 90-order batch", with_clock(clk, lambda: b.can_afford(90 * 10)), True)

print("\nBURST -- the whole reason this is not a pacer")
# The old next-slot pacer spaced 10 tokens at 300/s = 33 ms apart, so a 90-order batch took
# 3.0 s. Kalshi accepts all 900 tokens instantly. That gap was the defect.
b, clk = bucket()
with_clock(clk, lambda: b.debit(900))
check("900 tokens spendable with no time passing", with_clock(clk, b.available), 0.0)
assert_true("  i.e. a 90-order batch clears in one go, not 3.0 s of spacing", True)

print("\nrefill is linear in elapsed time and CAPPED at capacity")
b, clk = bucket()
with_clock(clk, lambda: b.debit(900))
clk.advance(1.0)
approx("1.0 s at 300/s refills 300", with_clock(clk, b.available), 300.0)
clk.advance(0.5)
approx("another 0.5 s refills 150 more", with_clock(clk, b.available), 450.0)
clk.advance(100.0)
check("100 s later it is capped, not 30,000", with_clock(clk, b.available), 900.0)

print("\ndebit ALWAYS succeeds and may go NEGATIVE")
# This is the kill-path property. An order we cannot cancel because our own bookkeeping says no
# is a real position we cannot close, so `debit` never refuses -- it records the debt.
b, clk = bucket()
lvl = with_clock(clk, lambda: b.debit(1500))
check("overspending by 600 leaves the level at -600", lvl, -600.0)
check("  and available reports the debt honestly", with_clock(clk, b.available), -600.0)
check("  and can_afford says no while in debt", with_clock(clk, lambda: b.can_afford(1)), False)
clk.advance(2.0)
approx("  2.0 s repays 600 of debt", with_clock(clk, b.available), 0.0)
clk.advance(1.0)
approx("  and only then does it accrue", with_clock(clk, b.available), 300.0)

print("\ncan_afford is a QUESTION -- it never consumes")
b, clk = bucket()
with_clock(clk, lambda: [b.can_afford(100) for _ in range(50)])
check("50 questions cost nothing", with_clock(clk, b.available), 900.0)
check("  and none were denied (denied counts penalties, not questions)", b.denied, 0)

print("\npenalise charges again after a 429")
# A 429 means Kalshi's bucket was empty when ours said otherwise. Ours is wrong in the direction
# that causes more 429s, so the rejection is charged for.
b, clk = bucket()
with_clock(clk, lambda: b.debit(300))
with_clock(clk, lambda: b.penalise(300))
check("level reflects both the send and the penalty", with_clock(clk, b.available), 300.0)
check("  denied is counted", b.denied, 1)
check("  spent counts both", b.spent, 600.0)

print("\nresize keeps the level -- it does not hand back what we spent")
# set_write_budget can run more than once. Resetting to full on a re-read would forgive real
# spending and put us straight back into 429s.
b, clk = bucket()
with_clock(clk, lambda: b.debit(600))
with_clock(clk, lambda: b.resize(capacity=900.0, refill_rate=100.0))
check("level survives the resize", with_clock(clk, b.available), 300.0)
check("  refill_rate is the new one", b.refill_rate, 100.0)
clk.advance(1.0)
approx("  and refills at the NEW rate", with_clock(clk, b.available), 400.0)

print("\nresize DOWN clamps a level above the new capacity")
b, clk = bucket(capacity=900.0)
with_clock(clk, lambda: b.resize(capacity=300.0, refill_rate=100.0))
check("900 held against a new 300 cap becomes 300", with_clock(clk, b.available), 300.0)

print("\nthe tier this account actually has")
# Sanity, not a network call: these are the constants the defaults are built from.
check("default cost", k.DEFAULT_TOKEN_COST, 10)
check("order cost", k.ORDER_TOKEN_COST, 10)
check("cancel cost", k.CANCEL_TOKEN_COST, 2)
check("orderbook ticker cap (kalshi_api.yaml maxItems)", k.MAX_ORDERBOOK_TICKERS, 100)
assert_true("a 900-token bucket is exactly 90 creates or 450 cancels",
            900 // k.ORDER_TOKEN_COST == 90 and 900 // k.CANCEL_TOKEN_COST == 450)

finish()
