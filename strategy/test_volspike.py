"""Pin the near-term vol-spike widening (`pricing/volspike`) and, above all, its OFF identity.

THE IDENTITY. With `config.VOLSPIKE_ENABLED = False`, `run_04.build_desired` must return exactly
what it returned before the module existed -- same desired rungs, same notes -- whatever signal is
handed to it, including an extreme one that would pull every market if the flag were on.

Then the function itself: symmetric, never negative, clipped, a pull flag above `s_pull`, inert on
a missing signal. And with the flag ON (in this test only) it widens both sides by the same amount.

    python -m strategy.test_volspike
"""
import math
import sys

import config
from harness import FAILURES, assert_true, check, finish
from pricing import volspike as V
from strategy import band as bandmod, budget
from strategy.run_04_quote_loop import build_desired

F = 7691.0
W = 0.00244


class _EM:
    def __init__(self, m):
        self.m = m

    def get(self, t):
        return self.m.get(t) or {"quotable": False, "reason": "unknown"}


def universe():
    """A 4pm bracket board plus two hourly events -- the same fixture shape as test_pipeline."""
    markets, fairs, em_map, books = [], {}, {}, {}

    def add(ticker, stype, lo, hi, w):
        markets.append({"ticker": ticker, "strike_type": stype,
                        "floor_strike": lo, "cap_strike": hi})
        k = lo if stype != "between" else 0.5 * (lo + hi)
        z = (k - F) / max(1e-9, w * F)
        fair = max(0.01, min(0.99, 0.5 - 0.16 * z))
        fairs[ticker] = {"fair": fair}
        em_map[ticker] = {"quotable": True, "R": 0.30 * max(0.0, 1 - abs(z) / 4) ** 2 + 1e-4,
                          "w": w, "edge_micros": 5_000, "scale_pos": 0.2, "floor_micros": 5_000,
                          "bonus_micros": 2_000,
                          "edge": 0.005, "risk_u": 0.5, "source": "curve", "reason": None}
        books[ticker] = {"yes": [[f"{max(0.01, fair - 0.04):.4f}", "100.00"]],
                         "no": [[f"{max(0.01, 1 - fair - 0.04):.4f}", "100.00"]]}

    for i in range(28):
        lo = F - 350 + 25 * i
        add(f"KXINX-26AUG25H1600-B{i:02d}", "between", lo, lo + 25, W)
    for n, h in enumerate((1200, 1500), start=1):
        for i in range(60):
            add(f"KXINXU-26AUG25H{h}-T{i:02d}", "greater_or_equal", F - 150 + 5 * i, None,
                W * (n / 2) ** 0.5)
    return markets, fairs, em_map, books


def run(markets, fairs, books, em, bands, lean_map, spike_map, **kw):
    d, n = build_desired(markets, fairs, books, em, lean_map, True, bands, spike_map=spike_map,
                         **kw)
    return {k: [(r.price, r.count, r.edge, r.min_edge, r.reason, r.index) for r in v]
            for k, v in d.items()}, n


markets, fairs, em_map, books = universe()
em = _EM(em_map)
bands, _ = bandmod.classify(markets, F, em_map)
counts = {t: b["counts"] for t, b in bands.items()}
theos = {t: d["fair"] for t, d in fairs.items()}
chosen, _ = budget.select_markets(markets, theos, em_map, counts=counts)
sel = [m for m in markets if m["ticker"] in chosen]
# A lean map with real disagreement on some markets, so the identity is tested WITH the lean on.
lean_map = {m["ticker"]: {"p_ref": fairs[m["ticker"]]["fair"],
                          "p_rv": min(0.99, fairs[m["ticker"]]["fair"] + 0.03)}
            for i, m in enumerate(sel) if i % 3 == 0}

print("1. THE OFF IDENTITY -- VOLSPIKE_ENABLED False changes nothing")
check("VOLSPIKE_ENABLED ships False", config.VOLSPIKE_ENABLED, False)
base_d, base_n = run(sel, fairs, books, em, bands, lean_map, None)
assert_true("  the fixture actually quotes", len(base_d) > 20, f"{len(base_d)} sides")
for label, sig in (("no spike_map", None),
                   ("an empty spike_map", {}),
                   ("a quiet signal on every market", 0.0),
                   ("an EXTREME signal on every market (would pull if on)", 5.0),
                   ("a None signal", "none")):
    if sig is None or sig == {}:
        smap = sig
    else:
        smap = {m["ticker"]: {"s": (None if sig == "none" else sig)} for m in sel}
    d, n = run(sel, fairs, books, em, bands, lean_map, smap)
    assert_true(f"  desired bit-identical with {label}", d == base_d,
                f"{len(d)} vs {len(base_d)} sides")
    assert_true(f"  notes bit-identical with {label}", n == base_n)

print("\n2. THE FUNCTION")
adj, det = V.edge_adjust(None)
check("missing signal -> 0", (adj, det["spike_reason"]), (0, "no_signal"))
check("NaN signal -> 0", V.edge_adjust(float("nan"))[0], 0)
p = dict(s0=0.25, k=2.0, s_max=0.5, s_pull=0.6)
check("below s0 -> 0", V.edge_adjust(0.2, **p)[0], 0)
check("a very quiet next minute never TIGHTENS", V.edge_adjust(-3.0, **p)[0], 0)
check("s0 + 0.1 -> 0.2c = 2000 micros", V.edge_adjust(0.35, **p)[0], 2_000)
check("clipped at s_max: 1c", V.edge_adjust(0.75, **p)[0], 10_000)
check("  and no larger at s = 5", V.edge_adjust(5.0, **p)[0], 10_000)
check("pull at s_pull", V.edge_adjust(0.6, **p)[1]["spike_pull"], True)
check("  not below it", V.edge_adjust(0.59, **p)[1]["spike_pull"], False)
prev = -1
mono = True
for i in range(0, 201):
    a = V.edge_adjust(-1.0 + i / 50.0, **p)[0]
    mono &= a >= prev
    prev = a
check("monotone non-decreasing in s", mono, True)
check("signal() = log ratio", round(V.signal(0.2, 0.1), 12), round(math.log(2.0), 12))
for a, b in ((None, 0.1), (0.1, None), (0.0, 0.1), (0.1, -1.0), (float("nan"), 0.1)):
    check(f"signal({a}, {b}) -> None", V.signal(a, b), None)

print("\n3. WITH THE FLAG ON (this test only) -- symmetric widening, and the pull")
_prev = config.VOLSPIKE_ENABLED
try:
    config.VOLSPIKE_ENABLED = True
    s_hot = config.VOLSPIKE_S0 + 0.5 * (config.VOLSPIKE_S_PULL - config.VOLSPIKE_S0)
    want = V.edge_adjust(s_hot)[0]
    assert_true("  the test signal is below the pull", s_hot < config.VOLSPIKE_S_PULL)
    smap = {m["ticker"]: {"s": s_hot} for m in sel}
    d_on, n_on = run(sel, fairs, books, em, bands, lean_map, smap)
    diffs = set()
    for tk, note in base_n.items():
        if "bid_min_edge" in note and "bid_min_edge" in n_on.get(tk, {}):
            diffs.add(n_on[tk]["bid_min_edge"] - note["bid_min_edge"])
            diffs.add(n_on[tk]["ask_min_edge"] - note["ask_min_edge"])
    check("  both sides' posting bar rises by exactly the spike charge", diffs, {want})
    assert_true("  and the board quotes no more sides than before", len(d_on) <= len(base_d),
                f"{len(d_on)} vs {len(base_d)}")
    d_pull, n_pull = run(sel, fairs, books, em, bands, lean_map,
                         {m["ticker"]: {"s": config.VOLSPIKE_S_PULL} for m in sel})
    check("  at s_pull nothing is quoted", len(d_pull), 0)
    check("  and every note says why", {v.get("skip") for v in n_pull.values()}, {"volspike_pull"})
finally:
    config.VOLSPIKE_ENABLED = _prev
check("flag restored", config.VOLSPIKE_ENABLED, False)

print("\n4. THE LIVE SIGNAL -- LeanChain.short_signal")
import datetime
import json
from strategy import leanchain
from rvol import model as rmodel
lc = leanchain.LeanChain(config.OPTION_UNDERLYING)
NOW = datetime.datetime(2026, 8, 21, 11, 49, tzinfo=config.ET)
if not lc.ok:
    print(f"  [skip] LeanChain unavailable ({lc.reason})")
else:
    if not lc.model.short_models:
        check("a production artifact without short_models -> s None, and says why",
              (lc.short_signal(NOW)["s"], lc.short_signal(NOW)["reason"]),
              (None, "no_short_models"))
    cand = config.ROOT / "var" / "backtest" / "rvol" / f"{config.OPTION_UNDERLYING}.short.json"
    if cand.exists():
        lc.model = rmodel.RealizedVolModel(json.loads(cand.read_text()))
        lc.rv.short_forecasts = lambda model, now=None, force=False: {
            "1m": 0.12, "3m": 0.11, "30m": 0.10}
        sig = lc.short_signal(NOW, sigma_impl=0.15)
        check("  s is log(sigma_1m / sigma_30m)", round(sig["s"], 12), round(math.log(1.2), 12))
        check("  s_3m alongside it", round(sig["s_3m"], 12), round(math.log(1.1), 12))
        check("  s_impl is the VRP-carrying comparison, log(sigma_1m / sigma_impl)",
              round(sig["s_impl"], 12), round(math.log(0.12 / 0.15), 12))
        lc.rv.short_forecasts = lambda model, now=None, force=False: None
        check("  no features -> None with a reason", (lc.short_signal(NOW)["s"],
              lc.short_signal(NOW)["reason"]), (None, "no_rv_features"))

        def _boom(model, now=None, force=False):
            raise RuntimeError("buffer")
        lc.rv.short_forecasts = _boom
        check("  and it never raises", lc.short_signal(NOW)["s"], None)

sys.exit(finish())
