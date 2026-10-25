"""Pin the four run_04 verdicts against synthetic stats. No clock, no network.

The whole point of a scheduled check is that its RESULT is legible without anyone reading a log,
so the verdict logic has to be testable without waiting for a market open.

    python -m strategy.test_verdict
"""
import sys
from harness import check

from strategy.run_04_quote_loop import FAIL, NO_SESSION, NO_SPOT, PASS, verdict

# The shared list, not a local one -- see harness.py.
from harness import FAILURES as FAILS


def healthy(**over):
    """A 12-minute run at 2 s that went well: 330 cycles, no errors, spot always, orders placed."""
    return {"cycles": 330, "errors": 0, "no_spot": 0, "placed": 144, "quiet": 170, **over}


EXPECTED = 360.0

print("the healthy run")
v, code, d = verdict(healthy(), EXPECTED, in_session=True)
check("passes", (v, code), (PASS, 0))
check("  with no reasons", d["reasons"], [])
check("  spot rate 100%", round(d["spot_rate"], 3), 1.0)

print("\nspot never resolved -- the split that matters")
v, code, d = verdict(healthy(no_spot=330, placed=0), EXPECTED, in_session=False)
check("outside the session gate -> NO_SESSION, exit 0", (v, code), (NO_SESSION, 0))
check("  reason names it", d["reasons"], ["market not open"])
v, code, d = verdict(healthy(no_spot=330, placed=0), EXPECTED, in_session=True)
check("inside the session gate -> NO_SPOT, exit 1", (v, code), (NO_SPOT, 1))

print("\nthe four ways a run that DID trade the board can be unhealthy")
v, code, d = verdict(healthy(cycles=100), EXPECTED, in_session=True)
check("too few cycles -> FAIL", (v, code), (FAIL, 1))
check("  and says how few", any("100 of an expected 360" in r for r in d["reasons"]), True)

v, code, d = verdict(healthy(cycles=330, errors=60), EXPECTED, in_session=True)
check("error rate over 10% -> FAIL", (v, code), (FAIL, 1))
check("  and says the rate", any("error rate" in r for r in d["reasons"]), True)

v, code, d = verdict(healthy(no_spot=200), EXPECTED, in_session=True)
check("spot on under half the cycles -> FAIL", (v, code), (FAIL, 1))

v, code, d = verdict(healthy(placed=0), EXPECTED, in_session=True)
check("never planned an order -> FAIL", (v, code), (FAIL, 1))
check("  because the ladder was never exercised",
      any("no order was ever planned" in r for r in d["reasons"]), True)

print("\nboundaries")
check("exactly at the cycle floor passes", verdict(healthy(cycles=180), EXPECTED, True)[0], PASS)
check("one cycle under it fails", verdict(healthy(cycles=179), EXPECTED, True)[0], FAIL)
# 33/330 == 10%, and the bar is `>= MAX_ERROR_RATE`, so exactly 10% must FAIL.
check("exactly 10% errors fails", verdict(healthy(errors=33), EXPECTED, True)[0], FAIL)
check("just under 10% passes", verdict(healthy(errors=32), EXPECTED, True)[0], PASS)

print("\ndegenerate input")
v, code, d = verdict({}, EXPECTED, in_session=True)
check("empty stats -> FAIL, not a crash", (v, code), (FAIL, 1))
check("  reason is 'no cycles ran'", d["reasons"], ["no cycles ran"])
v, code, d = verdict(healthy(cycles=0), EXPECTED, in_session=False)
check("zero cycles fails even out of session", (v, code), (FAIL, 1))

print("\nmultiple problems are all reported, not just the first")
v, code, d = verdict(healthy(cycles=50, errors=40, placed=0), EXPECTED, in_session=True)
check("three reasons", len(d["reasons"]) >= 3, True)

# THE PREDICATE THE VERDICT IS GIVEN, which is where the real defect was. `verdict` itself was
# always right; `run_04` fed it `config.in_capture_session()`, which returns False unconditionally
# while CAPTURE_SESSION_ENABLED is off. So a total feed outage in the middle of a trading day was
# classified NO_SESSION and exited 0 -- silent, on the one failure a scheduled check exists to
# catch. A storage flag must not be able to disarm a trading alarm.
print("\nthe session predicate is the trading session, not the capture gate")
import datetime                                                       # noqa: E402

import config                                                         # noqa: E402
from strategy import run_04_quote_loop as r4                          # noqa: E402

ET = config.ET
check("capture gate is off, so it is False mid-session",
      config.in_capture_session(datetime.datetime(2026, 8, 25, 11, 0, tzinfo=ET)), False)
check("  the trading predicate is not",
      config.in_trading_session(datetime.datetime(2026, 8, 25, 11, 0, tzinfo=ET)), True)
check("09:29 ET is out", config.in_trading_session(
    datetime.datetime(2026, 8, 25, 9, 29, tzinfo=ET)), False)
check("09:30 ET is in", config.in_trading_session(
    datetime.datetime(2026, 8, 25, 9, 30, tzinfo=ET)), True)
check("15:59 ET is in", config.in_trading_session(
    datetime.datetime(2026, 8, 25, 15, 59, tzinfo=ET)), True)
# Exclusive at 16:00: KXINX settles there and SPXW stops quoting, so no-spot after the close is
# the market being shut rather than a feed being down.
check("16:00 ET is out", config.in_trading_session(
    datetime.datetime(2026, 8, 25, 16, 0, tzinfo=ET)), False)
check("Saturday is out", config.in_trading_session(
    datetime.datetime(2026, 8, 29, 11, 0, tzinfo=ET)), False)

src = open(r4.__file__).read()
check("run_04 calls in_trading_session", "config.in_trading_session()" in src, True)
check("  and no longer feeds the capture gate to verdict",
      "verdict({**stats, \"cycles\": n}, expected, config.in_capture_session())" in src, False)

# The pairing that matters: mid-session with no spot must be NO_SPOT and exit 1.
v, code, d = verdict(healthy(no_spot=330), EXPECTED,
                     config.in_trading_session(datetime.datetime(2026, 8, 25, 11, 0, tzinfo=ET)))
check("no spot for a whole mid-session run -> NO_SPOT, exit 1", (v, code), (NO_SPOT, 1))
v, code, d = verdict(healthy(no_spot=330), EXPECTED,
                     config.in_trading_session(datetime.datetime(2026, 8, 25, 21, 0, tzinfo=ET)))
check("  the same run after the close -> NO_SESSION, exit 0", (v, code), (NO_SESSION, 0))

print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
sys.exit(0 if not FAILS else 1)
