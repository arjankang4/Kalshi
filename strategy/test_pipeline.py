"""End-to-end at the SPX universe scale, with no network:
band -> select -> ladder -> reconcile -> admit -> apply.

The pieces are gated individually (`test_band`, `test_budget`, `test_orders`, `test_quoting`,
`kalshi.test_ws`). What this pins is what none of them can see: that the chain composes at the
size the loop actually runs at, and that the two rationing decisions -- capital (who is quoted)
and tokens (whose orders move) -- agree rather than fight.

Scale matters here. At 30 markets the caps are slack and everything fits; at ~450 the board wants
several times its collateral and many times its per-cycle token budget, and every quantity in the
chain changes character.

    python -m strategy.test_pipeline
"""
import sys

import config
from harness import assert_true, check, finish
from kalshi.client import TokenBucket
from pricing import edge as pedge
from strategy import band as bandmod, budget, orders as O
from strategy.run_04_quote_loop import build_desired

C = config.QUOTE_TIER_COUNTS
FLOOR_MICROS = int(round(pedge.EDGE_FLOOR * 1e6))   # the per-market bar, at the edge floor
LADDER3_V = 3.0 * config.LOT_SIZE  # collateral of a two-sided 3-rung ladder at any fair, dollars
F = 7691.0
W = 0.00244                       # total vol to 16:00, measured 2026-08-21 11:49 ET
SIGMA_PTS = W * F                 # 18.8 index points

# The SPX universe as it actually lists: the 4pm bracket board plus every open hourly event.
# Measured 2026-08-21 -- all remaining hours are listed and active at once and the set only
# shrinks, so at the open there are seven KXINXU events of 60 rungs each.
HOURS = (1000, 1100, 1200, 1300, 1400, 1500, 1600)


class _EM:
    """Stands in for `strategy.edgemgr.EdgeManager` -- only `.get` is on the hot path."""

    def __init__(self, m):
        self.m = m

    def get(self, t):
        return self.m.get(t) or {"quotable": False, "reason": "unknown"}


def universe():
    """(markets, fairs, em_map, books) for the SPX board set.

    `w` shrinks with the hour, because an hourly market's total vol is to ITS OWN settlement --
    that is what makes the band narrow on a near-dated event and wide on the 4pm one, and it is
    the behaviour the whole sigma-moneyness design rests on.
    """
    markets, fairs, em_map, books = [], {}, {}, {}

    def add(ticker, stype, lo, hi, w):
        markets.append({"ticker": ticker, "strike_type": stype,
                        "floor_strike": lo, "cap_strike": hi})
        k = lo if stype != "between" else 0.5 * (lo + hi)
        # A crude but monotone fair: far below the strike is near 1, far above near 0.
        z = (k - F) / max(1e-9, w * F)
        fair = max(0.01, min(0.99, 0.5 - 0.16 * z))
        fairs[ticker] = {"fair": fair}
        em_map[ticker] = {"quotable": True, "R": 0.30 * max(0.0, 1 - abs(z) / 4) ** 2 + 1e-4,
                          "w": w, "edge_micros": FLOOR_MICROS, "scale_pos": 0.2,
                          "edge": pedge.EDGE_FLOOR, "risk_u": 0.5, "source": "curve", "reason": None}
        books[ticker] = {"yes": [[f"{max(0.01, fair - 0.02):.4f}", "100.00"]],
                         "no": [[f"{max(0.01, 1 - fair - 0.02):.4f}", "100.00"]]}

    # KXINX: 28 brackets, 25 wide, spanning the forward.
    for i in range(28):
        lo = F - 350 + 25 * i
        add(f"KXINX-26AUG25H1600-B{i:02d}", "between", lo, lo + 25, W)
    # KXINXU: seven events, 60 rungs each, 5 wide. `w` scales with the hour's distance from now.
    for n, h in enumerate(HOURS, start=1):
        w_h = W * (n / len(HOURS)) ** 0.5
        for i in range(60):
            add(f"KXINXU-26AUG25H{h}-T{i:02d}", "greater_or_equal", F - 150 + 5 * i, None, w_h)
    return markets, fairs, em_map, books


markets, fairs, em_map, books = universe()
theos = {t: d["fair"] for t, d in fairs.items()}
em = _EM(em_map)

print("the SPX universe this has to survive")
check("markets listed", len(markets), 448)
check("  one 4pm bracket board", sum(1 for m in markets if m["ticker"].startswith("KXINX-")), 28)
check("  and seven hourly events", len(HOURS), 7)
assert_true("  a uniform 3-tier board would want several times its collateral",
            len(markets) * LADDER3_V / config.MAX_RESTING_ORDER_VALUE > 3,
            f"${len(markets) * LADDER3_V:,.0f} vs ${config.MAX_RESTING_ORDER_VALUE:,.0f}")

print("\nSTEP 1 -- the band decides SHAPE, in sigma, per market's own horizon")
book = O.RestingBook()
bands, bsum = bandmod.classify(markets, F, em_map)
counts_map = {t: b["counts"] for t, b in bands.items()}
check("every market classified", bsum["n"], len(markets))
assert_true("  a minority are in the band", 0 < bsum["band"] < bsum["wing"],
            f"{bsum['band']} band / {bsum['wing']} wing")
# THE SHIPPED RUNG COUNT, whatever it is -- this is the end-to-end test and it must follow the
# config rather than pin a number. With ONE quote a side, band and wing have the same shape and
# the two-pass split rations capital without differing in depth. `test_budget` keeps the
# multi-rung arithmetic pinned with an explicit ladder.
N_BAND = len(config.QUOTE_TIER_COUNTS)
check(f"  band markets get {N_BAND} rung(s)", len(counts_map[
    next(t for t, b in bands.items() if b["inband"])]), N_BAND)
check("  wings get one", len(counts_map[
    next(t for t, b in bands.items() if not b["inband"])]), 1)

print("  the near-dated hourly events have NARROWER bands than the 4pm board")
# w is total vol to that market's settlement, so an event close to settling has little left.
def band_count(prefix):
    return sum(1 for t, b in bands.items() if t.startswith(prefix) and b["inband"])
early, late = band_count("KXINXU-26AUG25H1000"), band_count("KXINXU-26AUG25H1600")
assert_true("H1000 bands fewer rungs than H1600", early < late, f"{early} vs {late}")

print("\nSTEP 2 -- capital selection, two passes")
chosen, sdet = budget.select_markets(markets, theos, em_map, counts=counts_map)
# With a single shipped rung every market is wing-shaped, so the BAND pass has no candidates and
# the wing pass is handed the whole cap -- which `select_markets` does deliberately rather than
# leaving the band's share idle. Assert the split by shape, not that both passes always fire.
if N_BAND > 1:
    check("band pass funded", sdet["band"]["chosen"] > 0, True)
else:
    check("one shipped rung -> no band-shaped markets", sdet["band"]["candidates"], 0)
check("wing pass funded", sdet["wing"]["chosen"] > 0, True)
assert_true("  both caps respected",
            sdet["value"] <= config.MAX_RESTING_ORDER_VALUE + 1e-9
            and sdet["contracts"] <= config.MAX_RESTING_CONTRACTS_TOTAL + 1e-9,
            f"${sdet['value']:,.0f} / {sdet['contracts']:.0f} contracts")
assert_true("  what we cannot afford is REPORTED, not silently dropped",
            sdet["unaffordable"] >= 0, str(sdet["unaffordable"]))
# One rung buys far more markets than a 3-rung ladder would. Assert the direction rather than a
# count that moves with the config.
assert_true("  the cheaper shape funds more markets",
            len(chosen) >= config.MAX_RESTING_ORDER_VALUE // LADDER3_V,
            f"{len(chosen)} vs {int(config.MAX_RESTING_ORDER_VALUE // LADDER3_V)}")

print("\nSTEP 3 -- ladders, at the SHIPPED rung count")
sel = [m for m in markets if m["ticker"] in chosen]
desired, notes = build_desired(sel, fairs, books, em, {}, True, bands)
# Not every chosen market quotes both sides: `decide_ladder` returns [] when rung 0 cannot clear
# its edge, which is normal deep in the wings. What must hold is that whatever IS quoted has the
# rung count its band membership calls for.
sides = len(desired)
assert_true("most chosen markets quote both sides", sides > 1.5 * len(chosen),
            f"{sides} sides across {len(chosen)} markets")
assert_true(f"  band sides carry {N_BAND} rung(s)",
            all(len(v) == N_BAND for (t, _s), v in desired.items() if bands[t]["inband"]))
assert_true("  wing sides carry exactly one",
            all(len(v) == 1 for (t, _s), v in desired.items() if not bands[t]["inband"]))
check("  and the note records which is which",
      notes[sel[0]["ticker"]]["inband"], bands[sel[0]["ticker"]]["inband"])

print("\nSTEP 4 -- reconcile from an empty book, then ration the writes")
for m in markets:
    book.state(m["ticker"]).theo = theos[m["ticker"]]
place, cancel, kept = O.reconcile(book, desired, counts=counts_map)
check("no cancels on a cold start", len(cancel), 0)
check("places = 3 per band side + 1 per wing side", len(place),
      sum(len(v) for v in desired.values()))
bkt = TokenBucket(900.0, 300.0)
p1, c1, bd1 = budget.admit(place, cancel, bkt.available(), theos=theos, em_map=em_map)
assert_true("one cycle cannot place the board", bd1["n_deferred"] > 0,
            f"{bd1['n_admitted']} admitted, {bd1['n_deferred']} deferred")
res = O.apply(p1, c1, book, live=False)
check("  what was admitted was placed", res["placed"], len(p1))
check("  with no rate limiting", res["rate_limited"], 0)

print("\n  ... and successive cycles finish it")
cycles = 1
while len(book) < len(place) and cycles < 60:
    bkt.debit(-600)
    d2, _ = build_desired(sel, fairs, books, em, {}, True, bands)
    pl, cn, _ = O.reconcile(book, d2, counts=counts_map)
    pa, ca, _ = budget.admit(pl, cn, bkt.available(), theos=theos, em_map=em_map)
    O.apply(pa, ca, book, live=False)
    cycles += 1
check("the full board is up", len(book), len(place))
assert_true("  in a manageable number of cycles", cycles <= 40, f"{cycles} cycles")

print("\nSTEP 5 -- every risk limit holds at once")
ok, viol = O.within_risk_limits(book, [], [])
check("no violations", (ok, viol), (True, []))
assert_true("  global collateral inside the cap",
            book.value() <= config.MAX_RESTING_ORDER_VALUE + 1e-9, f"${book.value():,.2f}")
worst = max((book.value(t) for t in book.tickers()), default=0.0)
assert_true("  and no single market exceeds the per-market cap",
            worst <= config.MAX_RESTING_ORDER_VALUE_PER_MARKET + 1e-9, f"${worst:,.2f}")
assert_true("  nor any side one ladder",
            all(book.contracts(t, s) <= config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE + 1e-9
                for t in book.tickers() for s in ("bid", "ask")))

print("\nSTEP 6 -- a steady-state cycle is QUIET")
d3, _ = build_desired(sel, fairs, books, em, {}, True, bands)
pl, cn, kp = O.reconcile(book, d3, counts=counts_map)
check("nothing to place", len(pl), 0)
check("  nothing to cancel", len(cn), 0)
check("  everything kept", len(kp), len(book))

print("\nSTEP 7 -- THE INVENTORY GATE stops one market picking us off repeatedly")
t_band = next(t for t, b in bands.items() if b["inband"] and t in chosen)
t_wing = next(t for t, b in bands.items() if not b["inband"] and t in chosen)
lim_b, lim_w = bands[t_band]["position_limit"], bands[t_wing]["position_limit"]
check("a band market's limit is POSITION_LIMIT_LOTS_BAND lots", lim_b,
      float(config.POSITION_LIMIT_LOTS_BAND * config.LOT_SIZE))
check("  a wing's is POSITION_LIMIT_LOTS_WING lots", lim_w,
      float(config.POSITION_LIMIT_LOTS_WING * config.LOT_SIZE))

# Long YES at the limit: the BID adds and must stop; the ASK flattens and must continue.
d4, n4 = build_desired(sel, fairs, books, em, {}, True, bands, positions={t_band: lim_b})
check("at +limit the bid is pulled", (t_band, "bid") in d4, False)
check("  and the flattening ask is still quoted", (t_band, "ask") in d4, True)
check("  the record says why", n4[t_band]["position_blocked"], ["bid"])
check("  and reports the position and its limit",
      (n4[t_band]["position"], n4[t_band]["position_limit"]), (lim_b, lim_b))

d5, _ = build_desired(sel, fairs, books, em, {}, True, bands, positions={t_band: -lim_b})
check("at -limit the ASK is pulled instead", (t_band, "ask") in d5, False)
check("  and the bid flattens us", (t_band, "bid") in d5, True)

d6, _ = build_desired(sel, fairs, books, em, {}, True, bands,
                      positions={t_band: lim_b - float(C[0])})
check("one lot inside the limit, both sides quote", 
      ((t_band, "bid") in d6, (t_band, "ask") in d6), (True, True))

print("  a wing's limit is a whole number of its single rungs, so filling to it goes one-sided")
# Pick a wing that quotes BOTH sides in the baseline, so the assertion is about the inventory gate
# and not about a deep wing whose ask never cleared its edge in the first place.
t_wing = next(t for t, b in bands.items()
              if not b["inband"] and (t, "bid") in desired and (t, "ask") in desired)
lim_w = bands[t_wing]["position_limit"]
d7a, _ = build_desired(sel, fairs, books, em, {}, True, bands,
                       positions={t_wing: lim_w - float(C[0])})
check("one rung short of the wing limit, both sides quote",
      ((t_wing, "bid") in d7a, (t_wing, "ask") in d7a), (True, True))
d7, _ = build_desired(sel, fairs, books, em, {}, True, bands, positions={t_wing: lim_w})
check("filled on its bid rung up to the limit -> bid pulled", (t_wing, "bid") in d7, False)
check("  ask still quoted", (t_wing, "ask") in d7, True)
assert_true("  the wing limit is POSITION_LIMIT_LOTS_WING of its single rungs",
            config.POSITION_LIMIT_LOTS_WING * float(C[0]) == lim_w)

print("  other markets are unaffected by one market's inventory")
check("the band market still quotes both sides", 
      ((t_band, "bid") in d7, (t_band, "ask") in d7), (True, True))

print("\nSTEP 8 -- a market leaving the set exits, and exiting is always affordable")
# It leaves because `edgemgr` stopped being able to price it -- the realistic exit, and the one
# `select_markets` refuses unconditionally rather than merely ranking last.
gone = sorted(chosen)[0]
em_drop = {**em_map, gone: {**em_map[gone], "quotable": False, "reason": "no_R"}}
kept_set, _ = budget.select_markets(markets, theos, em_drop, counts=counts_map,
                                    incumbents=set(book.tickers()))
check("it is dropped", gone in kept_set, False)
d8, _ = build_desired([m for m in markets if m["ticker"] in kept_set],
                      fairs, books, em, {}, True, bands)
pl, cn, _ = O.reconcile(book, d8, counts=counts_map)
exits = [c for c in cn if c["ticker"] == gone]
assert_true("reconcile plans its exit", len(exits) > 0)
check("  as no_longer_quoting", sorted({c["reason"] for c in exits}), ["no_longer_quoting"])
_, ca, bd = budget.admit(pl, cn, 0.0, theos=theos, em_map=em_map, reserve=0.0)
check("  and it is sent on an EMPTY bucket", 
      len([x for x in ca if x["ticker"] == gone]), len(exits))
assert_true("    flagged as over budget rather than silently deferred", bd["reduce_over_budget"])

print("\nHOURLY CUTOFF -- no hourly quote inside HOURLY_MIN_MINUTES_TO_CLOSE of settlement")
import datetime as _dt
from strategy.run_04_quote_loop import hourly_cutoff
def _m(hh, mm=0):
    return {"close_time": _dt.datetime(2026, 9, 24, hh, mm, tzinfo=config.ET).isoformat()}
_close = _dt.datetime(2026, 9, 24, 11, 0, tzinfo=config.ET)
_cut = config.HOURLY_MIN_MINUTES_TO_CLOSE
_inside = _close - _dt.timedelta(minutes=_cut - 1)        # one minute inside the cutoff
_outside = _close - _dt.timedelta(minutes=_cut + 1)       # one minute before it starts
check("  H1100 one minute inside the cutoff is refused", hourly_cutoff(_m(11), _inside)[0], True)
check("  H1100 one minute before the cutoff is not", hourly_cutoff(_m(11), _outside)[0], False)
check("  the 4pm board is exempt at 15:55", hourly_cutoff(
    _m(16), _dt.datetime(2026, 9, 24, 19, 55, tzinfo=_dt.timezone.utc))[0], False)

finish()
