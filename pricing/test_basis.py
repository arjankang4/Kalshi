"""The fast spot: futures for motion, an index-tick EMA for level, and refusal when unjustified."""
import time

from massive import index as ix
from massive import streams
from pricing import basis as basismod

FAILS = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got={got} want={want}")
    if not ok:
        FAILS.append(label)


def close(label, got, want, tol=1e-6):
    ok = got is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got={got} want={want}+-{tol}")
    if not ok:
        FAILS.append(label)


def assert_true(label, cond):
    check(label, bool(cond), True)


def iq(value, ts_ns, u="SPX"):
    return ix.IndexQuote(u, ix.INDEX_TICKER[u], value, ts_ns)


NS = 1_000_000_000

print("the EMA converges on the basis, and only the INDEX tick advances it")
t = basismod.BasisTracker("SPX", half_life_s=10.0, min_samples=3)
now = 1000.0
for i in range(60):                       # futures 14 points over the index, exactly
    t.observe(7700.0 + i * 0.1 + 14.0, iq(7700.0 + i * 0.1, (i + 1) * NS), now=now + i)
close("basis converges to the true 14.0", t.basis, 14.0, tol=0.01)
check("  one fold per index tick", t._n, 60)

print("\n  a REPEATED index timestamp is the same observation, not a new one")
n_before = t._n
for _ in range(20):
    t.observe(7800.0, iq(7706.0, 60 * NS), now=now + 60)     # same ts_ns 20 times
check("nothing folded", t._n, n_before)

print("\nthe fast spot is the futures level carried onto the index's scale")
s, d = t.spot(7720.0)
close("spot = F - basis", s, 7720.0 - t.basis, tol=1e-9)
check("  and it reports no reason", d["reason"], None)

print("\n  it MOVES with the futures while the index stands still")
a, _ = t.spot(7720.0)
b, _ = t.spot(7725.0)
close("a 5-point futures move is a 5-point spot move", b - a, 5.0, tol=1e-9)

print("\nit REFUSES rather than guessing")
cold = basismod.BasisTracker("SPX", min_samples=15)
s, d = cold.spot(7720.0)
check("no basis yet -> no spot", s, None)
assert_true("  and it says it is warming", d["reason"].startswith("basis_warming"))
for i in range(20):
    cold.observe(7714.0, iq(7700.0, (i + 1) * NS), now=1000.0 + i)
s, d = cold.spot(7720.0)
assert_true("once warm it prices", s is not None)
s, d = cold.spot(None)
check("no futures -> no spot", s, None)
check("  with the honest reason", d["reason"], "no_futures")

print("\nthe break detector: a basis STEP is refused, a basis DRIFT is not")
# Drift: the basis walks 1 point over 200 index ticks. The EMA follows; nothing is refused.
drift = basismod.BasisTracker("SPX", half_life_s=10.0, min_samples=3)
for i in range(200):
    drift.observe(7700.0 + 14.0 + i * 0.005, iq(7700.0, (i + 1) * NS), now=1000.0 + i)
s, d = drift.spot(7715.0)
assert_true("a slow drift still prices", s is not None)
assert_true("  and does not alarm", not d["alarm"])

# Step: the contract rolls and the basis jumps 20 points in one tick.
step = basismod.BasisTracker("SPX", half_life_s=60.0, min_samples=3)
for i in range(60):
    step.observe(7714.0, iq(7700.0, (i + 1) * NS), now=1000.0 + i)
step.observe(7734.0, iq(7700.0, 61 * NS), now=1061.0)        # +20 in one tick
s, d = step.spot(7734.0)
check("a 20-point step refuses the spot", s, None)
assert_true("  and it alarms", d["alarm"])
assert_true("  naming the residual", d["reason"].startswith("basis_break"))

print("\n  the alarm band SCALES with the level, so NDX is not held to SPX's tolerance")
spx = basismod.BasisTracker("SPX", min_samples=3)
ndx = basismod.BasisTracker("NDX", min_samples=3)
for i in range(30):
    spx.observe(7714.0, iq(7700.0, (i + 1) * NS), now=1000.0 + i)
    ndx.observe(29067.0, ix.IndexQuote("NDX", "I:NDX", 29000.0, (i + 1) * NS), now=1000.0 + i)
b_spx, b_ndx = spx.detail()["alarm_points"], ndx.detail()["alarm_points"]
assert_true("NDX's band is wider in points", b_ndx > b_spx)
close("  and equal as a fraction of the level", b_ndx / 29000.0, b_spx / 7700.0, tol=1e-9)

print("\nthe EMA is TIME-weighted, so a dropped tick does not distort it")
# Two trackers see the same two observations; one has a 1s gap, the other 30s. The 30s gap must
# weight the newer observation far more heavily.
fast_gap = basismod.BasisTracker("SPX", half_life_s=10.0, min_samples=1)
slow_gap = basismod.BasisTracker("SPX", half_life_s=10.0, min_samples=1)
fast_gap.observe(7714.0, iq(7700.0, 1 * NS), now=0.0)
slow_gap.observe(7714.0, iq(7700.0, 1 * NS), now=0.0)
fast_gap.observe(7724.0, iq(7700.0, 2 * NS), now=1.0)        # 1s later
slow_gap.observe(7724.0, iq(7700.0, 2 * NS), now=30.0)       # 30s later
assert_true("the long gap moves the EMA further", slow_gap.basis > fast_gap.basis)
assert_true("  and neither overshoots the new observation",
            fast_gap.basis <= 24.0 and slow_gap.basis <= 24.0)

print("\nFuturesFeed parses a Q frame and counts only MID CHANGES")
import threading                                            # noqa: E402
ff = streams.FuturesFeed.__new__(streams.FuturesFeed)      # no network, no front-month lookup
import collections                                          # noqa: E402
ff._by_sym = {"ESU6": "SPX"}
ff._mid, ff._ticks, ff._lock = {}, {}, threading.Lock()
ff._hist = collections.defaultdict(lambda: collections.deque(maxlen=streams.HIST_MAX))
ff.MAX_QUOTE_AGE_S = 3.0
now_ms = int(time.time() * 1000)
changed = ff._on_message({"ev": "Q", "sym": "ESU6", "bp": "7700.00", "ap": "7700.50",
                          "t": now_ms}, time.time_ns())
check("first quote is an update", changed, True)
same = ff._on_message({"ev": "Q", "sym": "ESU6", "bp": "7700.00", "ap": "7700.50",
                       "t": now_ms}, time.time_ns())
check("  an identical mid is NOT", same, False)
moved = ff._on_message({"ev": "Q", "sym": "ESU6", "bp": "7700.25", "ap": "7700.75",
                        "t": now_ms}, time.time_ns())
check("  a moved mid is", moved, True)
check("  tick count counts changes, not frames", ff._ticks["SPX"], 2)
mid, age = ff.mid("SPX")
close("  and the mid is (bid+ask)/2", mid, 7700.5, tol=1e-9)

print("\n  a CROSSED book is not a mid")
before = ff._mid["SPX"][0]
ff._on_message({"ev": "Q", "sym": "ESU6", "bp": "7701.00", "ap": "7699.00", "t": now_ms},
               time.time_ns())
close("the last good mid stands", ff._mid["SPX"][0], before, tol=1e-9)

print("\n  a STALE quote is refused, and so is a future-dated one")
ff._mid["SPX"] = (7700.0, time.time_ns() - 10 * NS)
m, a = ff.mid("SPX")
check("10s old -> refused", m, None)
assert_true("  with the age reported anyway", a is not None and a > 9)
ff._mid["SPX"] = (7700.0, time.time_ns() + 5 * NS)
m, a = ff.mid("SPX")
check("clock skew (negative age) -> refused", m, None)

print("\nmove_scale: the size of the option a resting quote writes over the exposure window")
mf = streams.FuturesFeed.__new__(streams.FuturesFeed)
mf._lock = threading.Lock()
mf._hist = collections.defaultdict(lambda: collections.deque(maxlen=streams.HIST_MAX))
check("too little history -> None, never a guess", mf.move_scale("SPX", 0.1), None)
# A clean random-walk path: 400 ticks, one per 10ms, each +-0.25 points. Realized variance is
# 0.0625 per tick and 100 ticks a second, so sigma is 2.5 points per sqrt(second).
NS_ = 1_000_000_000
px, t0 = 7700.0, 1_000 * NS_
for i in range(400):
    px += 0.25 if i % 2 == 0 else -0.25
    mf._hist["SPX"].append((t0 + i * NS_ // 100, px))
got = mf.move_scale("SPX", 1.0)
close("sigma over 1s is ~2.5 points", got, 2.5, tol=0.15)
close("  and it scales as sqrt(time)", mf.move_scale("SPX", 4.0) / got, 2.0, tol=0.02)
close("  so 100ms is much smaller", mf.move_scale("SPX", 0.1), got * (0.1 ** 0.5), tol=1e-9)
close("  and `sigmas` multiplies it", mf.move_scale("SPX", 1.0, sigmas=2.0), got * 2.0, tol=1e-9)

print("\n  a FLAT market has no adverse move to insure against")
flat = streams.FuturesFeed.__new__(streams.FuturesFeed)
flat._lock = threading.Lock()
flat._hist = collections.defaultdict(lambda: collections.deque(maxlen=streams.HIST_MAX))
for i in range(100):
    flat._hist["SPX"].append((t0 + i * NS_, 7700.0))
check("zero variance -> zero scale", flat.move_scale("SPX", 0.1), 0.0)

print("\n  it reads the VENDOR's clock, not ours -- the question is how fast the market moved")
assert_true("history entries are (vendor_ns, mid)",
            all(isinstance(a, int) and isinstance(b, float) for a, b in mf._hist["SPX"]))

print("\n  and only MID CHANGES enter it")
ff._hist.clear(); ff._mid.clear(); ff._ticks.clear()
now_ms2 = int(time.time() * 1000)
for bp, ap in (("7700.00", "7700.50"), ("7700.00", "7700.50"), ("7700.25", "7700.75")):
    ff._on_message({"ev": "Q", "sym": "ESU6", "bp": bp, "ap": ap, "t": now_ms2}, time.time_ns())
check("three frames, two distinct mids, two history entries", len(ff._hist["SPX"]), 2)

print("\nIndexFeed builds IndexQuotes and converts MILLISECONDS to nanoseconds")
idxf = streams.IndexFeed.__new__(streams.IndexFeed)
idxf._by_tk = {"I:SPX": "SPX"}
idxf._q, idxf._lock = {}, threading.Lock()
idxf.underlyings = ("SPX",)
t_ms = int(time.time() * 1000)
check("a new value is an update",
      idxf._on_message({"ev": "V", "T": "I:SPX", "val": 7661.1, "t": t_ms}, time.time_ns()), True)
check("  the same value again is not",
      idxf._on_message({"ev": "V", "T": "I:SPX", "val": 7661.1, "t": t_ms}, time.time_ns()), False)
q = idxf.snapshot()["SPX"]
check("it is an IndexQuote", type(q).__name__, "IndexQuote")
close("  value carried through", q.value, 7661.1, tol=1e-9)
close("  ms -> ns at the boundary", q.ts_ns, t_ms * 1_000_000, tol=0)
assert_true("  and its age is sane, not off by 10^6", abs(q.age_s()) < 5.0)
check("an underlying never heard from is ABSENT, not None",
      "NDX" in idxf.snapshot(("SPX", "NDX")), False)

print("\nFAIL" if FAILS else "\nPASS")
raise SystemExit(1 if FAILS else 0)
