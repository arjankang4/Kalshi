"""The session loss limit and the halt switch: pinned behaviour.

    python -m strategy.test_risklimit

WHAT THIS IS GUARDING. `strategy/risklimit` is the only thing in the stack that bounds LOSS rather
than exposure, and it is the newest code on the live path -- written 2026-08-25, before the first
`--live` run, so it has no production history at all. Two things in it are easy to get quietly
wrong and expensive to get wrong:

  1. THE SIGN OF THE 'no' LEG. A fill's `outcome_side` is 'no' for both buy-no and sell-yes, and
     both legs print at the same `yes_price_dollars`. Writing it as "bought NO at 1-p" instead of
     "sold YES at p" is off by exactly $1 per contract -- the settlement collateral -- and produces
     a P&L that looks plausible and is wrong by the size of the book.
  2. THE DIRECTION OF A MISSING MARK. A position we could not price must not vanish from the mark,
     because then the P&L IMPROVES every time the pricer fails, and a pricer failing is precisely
     when a risk control needs to be believed.

The zero-P&L identity below is the strongest check here: a fill marked at its own fill price is
worth exactly nothing, on either side, for any size. If that ever moves, the sign is wrong.
"""
import os
import tempfile

import config
from harness import approx, assert_true, check, finish, ok
from strategy import risklimit


def fill(n, p, side, fee="0.00"):
    return {"count_fp": f"{n:.2f}", "yes_price_dollars": f"{p:.4f}",
            "outcome_side": side, "fee_cost": fee}


print("1. THE ZERO IDENTITY -- a fill marked at its own price is worth nothing")
for side in ("yes", "no"):
    for n, p in ((20, 0.40), (137, 0.03), (5, 0.97)):
        s = risklimit.SessionRisk(limit_dollars=1e9)
        s.on_fills([fill(n, p, side)])
        pos = n if side == "yes" else -n
        ok(f"  {side} {n}@{p}: pnl at the fill price", s.mark({"A": pos}, {"A": p}), 0.0,
           tol=1e-9)

print("\n2. THE SIGN -- long gains when the fair rises, short gains when it falls")
s = risklimit.SessionRisk()
s.on_fills([fill(100, 0.40, "yes")])
ok("long 100 @0.40, fair 0.50 -> +10", s.mark({"A": 100.0}, {"A": 0.50}), 10.0, tol=1e-9)
ok("  fair 0.30 -> -10", s.mark({"A": 100.0}, {"A": 0.30}), -10.0, tol=1e-9)
s = risklimit.SessionRisk()
s.on_fills([fill(100, 0.40, "no")])
ok("short 100 @0.40, fair 0.30 -> +10", s.mark({"A": -100.0}, {"A": 0.30}), 10.0, tol=1e-9)
ok("  fair 0.50 -> -10", s.mark({"A": -100.0}, {"A": 0.50}), -10.0, tol=1e-9)
check("  cash from a sell is positive", s.cash, 40.0)

print("\n3. FEES ARE SUBTRACTED, whatever the schedule currently says")
s = risklimit.SessionRisk()
s.on_fills([fill(20, 0.50, "yes", fee="0.35")])
ok("a 35c fee is a 35c loss at the fill price", s.mark({"A": 20.0}, {"A": 0.50}), -0.35, tol=1e-9)
check("  and is recorded separately", round(s.fees, 2), 0.35)

print("\n4. AN UNMARKABLE POSITION DOES NOT IMPROVE THE P&L")
s = risklimit.SessionRisk()
s.on_fills([fill(100, 0.40, "yes")])
s.mark({"A": 100.0}, {"A": 0.20})
worst = s.pnl
s.mark({"A": 100.0}, {"A": None})
check("a market that failed to price keeps its last mark", s.pnl, worst)
check("  and is not counted as unmarked while a last fair exists", s.unmarked, 0)
s2 = risklimit.SessionRisk()
s2.on_fills([fill(100, 0.40, "yes")])
s2.mark({"A": 100.0}, {"A": None})
check("a position never priced at all is COUNTED, not dropped", s2.unmarked, 1)
assert_true("  and its cash still shows the loss", s2.pnl == -40.0, f"{s2.pnl}")

print("\n5. THE LIMIT TRIPS, AND STAYS TRIPPED")
s = risklimit.SessionRisk(limit_dollars=50.0, halt_path=None)
s.on_fills([fill(100, 0.40, "no")])
s.mark({"A": -100.0}, {"A": 0.85})
assert_true("-45 is inside a -50 limit", s.halt_reason() is None, f"pnl={s.pnl}")
s.mark({"A": -100.0}, {"A": 0.95})
assert_true("-55 trips it", bool(s.halt_reason()), f"pnl={s.pnl}")
# STICKY. A halt that un-halts when the mark bounces back is a halt that quotes into the recovery
# of the same disagreement that caused it.
s.mark({"A": 0.0}, {"A": 0.95})
assert_true("  and it does not un-halt when the mark recovers", bool(s.halt_reason()),
            f"pnl={s.pnl}")
check("  worst is kept for the record", s.worst < -50.0, True)

print("\n6. THE LIMIT IS ABSOLUTE-VALUED -- a positive config number still means a LOSS")
s = risklimit.SessionRisk(limit_dollars=-50.0, halt_path=None)
s.on_fills([fill(100, 0.40, "no")])
s.mark({"A": -100.0}, {"A": 0.95})
assert_true("a negative limit is read the same way", bool(s.halt_reason()), f"pnl={s.pnl}")
# THERE IS NO UNLIMITED MODE, and that is the point of pinning it here. `None` means "use the
# configured limit"; a caller that forgets to pass one gets protection, not the absence of it.
s = risklimit.SessionRisk(limit_dollars=None, halt_path=None)
check("limit_dollars=None falls back to config, it does NOT disable the limit",
      s.limit, config.DAILY_LOSS_LIMIT_DOLLARS)
s.on_fills([fill(1000, 0.40, "no")])
s.mark({"A": -1000.0}, {"A": 0.99})
assert_true("  and it still trips", bool(s.halt_reason()), f"{s.pnl}")

print("\n7. THE HALT FILE")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "HALT")
    s = risklimit.SessionRisk(limit_dollars=None, halt_path=p)
    assert_true("absent -> no halt", s.halt_reason() is None)
    open(p, "w").close()
    assert_true("present -> halt", bool(s.halt_reason()))
    os.remove(p)
    assert_true("  removing it does NOT resume this process", bool(s.halt_reason()))
    assert_true("  a fresh process is clear again",
                risklimit.SessionRisk(limit_dollars=None, halt_path=p).halt_reason() is None)

print("\n8. MALFORMED FRAMES ARE COUNTED, NEVER READ AS ZERO")
s = risklimit.SessionRisk()
applied = s.on_fills([{"count_fp": "x", "yes_price_dollars": "0.4", "outcome_side": "yes"},
                      {"count_fp": "10.00", "outcome_side": "yes"},
                      {"count_fp": "10.00", "yes_price_dollars": "0.4"},
                      fill(10, 0.40, "yes")])
check("one of four frames was usable", applied, 1)
check("  the other three are counted", s.skipped, 3)
check("  cash reflects only the usable one", s.cash, -4.0)
# The deprecated `side` field is the fallback, because `outcome_side` is only guaranteed on newer
# frames and a fill we cannot direct is a fill we cannot bound the loss with.
s = risklimit.SessionRisk()
s.on_fills([{"count_fp": "10.00", "yes_price_dollars": "0.40", "side": "no"}])
check("legacy `side` is honoured when outcome_side is absent", s.cash, 4.0)

print("\n9. THE SHIPPED CONFIG IS A REAL LIMIT")
s = risklimit.SessionRisk()
check("the default reads config", s.limit, config.DAILY_LOSS_LIMIT_DOLLARS)
assert_true("  and it is set", s.limit is not None and s.limit > 0, f"{s.limit}")
assert_true("  and it is well inside the resting-collateral cap",
            s.limit <= config.MAX_RESTING_ORDER_VALUE,
            f"{s.limit} vs {config.MAX_RESTING_ORDER_VALUE}")
check("  pnl is None before the first mark, not 0.0", s.pnl, None)

print("\n10. STARTUP RECONCILIATION -- what run_04 does about orders already resting")
# Monkeypatched rather than tested against the exchange: the branch under test is what we DO about
# what came back, and the read itself (`cancel_all.resting_across`) is the same call the kill path
# has used in production since 2026-08-24.
from strategy import cancel_all                                       # noqa: E402
from strategy import run_04_quote_loop as r4                          # noqa: E402

real_resting, real_cancel, real_events = (cancel_all.resting_across, cancel_all.cancel_events,
                                          cancel_all.today_events)
cancel_all.today_events = lambda: ["KXINX-26AUG25H1600"]
try:
    cancel_all.resting_across = lambda _e: []
    check("nothing resting -> nothing to do", r4.startup_reconcile(live=True)["action"], "none")

    cancel_all.resting_across = lambda _e: [{"order_id": "a"}, {"order_id": "b"}]
    cancel_all.cancel_events = lambda _e, **kw: 0
    r = r4.startup_reconcile(live=True)
    check("live + resting -> cancelled", (r["action"], r["found"], r["left"]),
          ("cancelled", 2, 0))
    check("  dry never cancels", r4.startup_reconcile(live=False)["action"], "dry_noop")
    check("  --adopt-resting leaves them", r4.startup_reconcile(live=True, adopt=True)["action"],
          "adopted")

    # THE ONE CASE THAT MUST STOP THE RUN. Quoting on top of orders we could not pull is the exact
    # outcome this function exists to prevent, so it is a SystemExit, not a warning.
    cancel_all.cancel_events = lambda _e, **kw: 1
    try:
        r4.startup_reconcile(live=True)
        check("a cancel that leaves orders resting refuses to start", "returned", "SystemExit")
    except SystemExit as e:
        assert_true("a cancel that leaves orders resting refuses to start",
                    "refusing to quote" in str(e), str(e)[:60])

    # ...and a read that FAILS does not, because refusing to start on a transient
    # /portfolio/orders error is a worse failure than starting.
    def _boom(_e):
        raise RuntimeError("HTTP 503")
    cancel_all.resting_across = _boom
    check("an unreadable startup read runs blind rather than aborting",
          r4.startup_reconcile(live=True)["action"], "blind")
finally:
    cancel_all.resting_across, cancel_all.cancel_events, cancel_all.today_events = (
        real_resting, real_cancel, real_events)

print("\n11. THE SIGNAL PATH -- the stop event exists and the loop waits on it")
src = open(r4.__file__).read()
check("SIGTERM is handled", "signal.SIGTERM" in src, True)
check("  the loop checks the stop event", "not _STOP.is_set()" in src, True)
check("  and sleeps on it rather than time.sleep", "_STOP.wait(max(0.0" in src, True)
check("  a halt forces a non-zero exit even under --no-gate",
      "if risk.halted:\n            exit_code = 1" in src, True)

print("\n12. THE ACCOUNT MARK -- equity vs the day's open, marked to the Kalshi mid")
_tmp = tempfile.mkdtemp()
_real_dir = getattr(config, "RISK_STATE_DIR", None)
config.RISK_STATE_DIR = __import__("pathlib").Path(_tmp)
try:
    def fresh():
        r = risklimit.SessionRisk(limit_dollars=300.0, halt_path=os.path.join(_tmp, "NOHALT"))
        return r
    # Open: $1000 cash, long 100 YES at a 0.40/0.42 book (mid 0.41).
    s = fresh()
    s.mark_account((0.0, 1000.0, {"A": 100.0}), {"A": (0.40, 0.42)}, now_mono=0.0)
    approx("  first mark of the day is flat by definition", s.pnl, 0.0)
    approx("  equity = balance + q * mid", s.equity, 1041.0)

    # THE 2026-08-26 BUG: a RESTART inherits the position. The old fill-cash mark started cash at 0
    # and read the whole position as profit. The account mark reads the persisted open instead.
    s2 = fresh()
    s2.mark_account((0.0, 1000.0, {"A": 100.0}), {"A": (0.40, 0.42)}, now_mono=0.0)
    approx("  a restart does NOT book inherited positions as profit", s2.pnl, 0.0)
    old = risklimit.SessionRisk(limit_dollars=1e9)
    old.mark({"A": 100.0}, {"A": 0.41})
    ok("  (the old fill-cash mark reads the same restart as +$41)", old.pnl, 41.0, tol=1e-9)

    # Mid moves 0.41 -> 0.31: -$10.
    s2.mark_account((1.0, 1000.0, {"A": 100.0}), {"A": (0.30, 0.32)}, now_mono=1.0)
    approx("  a 10c mid drop on 100 long YES is -$10", s2.pnl, -10.0)

    # SETTLEMENT: A settles NO. Position vanishes, balance unchanged (paid 0). No fill frame.
    s2.mark_account((2.0, 1000.0, {}), {}, now_mono=2.0)
    approx("  settling at 0 realises the full -$41 through the balance", s2.pnl, -41.0)

    # SHORT YES = long NO, marked at 1 - mid.
    s3 = fresh()
    s3.mark_account((3.0, 959.0, {"B": -50.0}), {"B": (0.18, 0.22)}, now_mono=3.0)
    approx("  50 NO at a 0.20 YES mid is worth $40", s3.equity - 959.0, 40.0)

    # One-sided books mark conservatively: long YES at the bid, long NO at the ask.
    s4 = fresh()
    s4.mark_account((4.0, 959.0, {"A": 10.0, "B": -10.0}), {"A": (0.35, None), "B": (None, 0.60)},
                    now_mono=4.0)
    approx("  one-sided marks: 10*0.35 + 10*(1-0.60)", s4.equity - 959.0, 7.5)

    # THE LIMIT FIRES on the account number.
    s5 = fresh()
    s5.mark_account((5.0, 700.0, {}), {}, now_mono=5.0)
    assert_true("  -$341 vs the persisted day open trips the $300 limit",
                bool(s5.halt_reason()) and "limit" in s5.halt_reason(), str(s5.halt_reason()))

    # A STALE account read halts once it passes the configured age.
    s6 = fresh()
    s6.mark_account((10.0, 1041.0, {}), {}, now_mono=10.0)
    s6.mark_account(None, {}, now_mono=10.0 + config.RISK_ACCOUNT_STALE_HALT_S + 1)
    assert_true("  a stale account read halts", bool(s6.halt_reason())
                and "stale" in s6.halt_reason(), str(s6.halt_reason()))
    s7 = fresh()
    s7.mark_account((10.0, 1041.0, {}), {}, now_mono=10.0)
    s7.mark_account(None, {}, now_mono=15.0)
    check("  a 5s-old read does not", s7.halt_reason(), None)
finally:
    config.RISK_STATE_DIR = _real_dir


finish()
