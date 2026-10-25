"""Pin the write-budget admission policy. No network, no clock, no live bucket.

Two invariants carry the whole module and both are failure modes this repo has already paid for
once, through the risk gate rather than the rate limiter:

  1. an exposure-reducing cancel is NEVER deferred
  2. a replacement cancel is NEVER sent without its place

Everything else here is ordering and arithmetic.

    python -m strategy.test_budget
"""
import sys

from harness import approx, assert_true, check, finish

from kalshi.client import CANCEL_TOKEN_COST, ORDER_TOKEN_COST
import config
from strategy import budget
from strategy.budget import IMPROVE, REDUCE, TOPUP, admit, build_items
from strategy.orders import FRONT_IMPROVE_TICKS, _pair_id

BIG = 10_000.0          # a budget that constrains nothing
LOT = f"{config.LOT_SIZE:.2f}"   # one rung


def cancel(ticker, side="bid", price=0.50, count=LOT, reason="edge_negative", pair=None):
    d = {"ticker": ticker, "side": side, "price": price, "count": count,
         "reason": reason, "order_id": f"oid-{ticker}-{side}-{price}"}
    if pair:
        d["pair"] = pair
    return d


def place(ticker, side="bid", price=0.50, count=LOT, reason="top_up", pair=None):
    d = {"ticker": ticker, "side": side, "price": price, "count": count, "reason": reason}
    if pair:
        d["pair"] = pair
    return d


def improve_pair(ticker, side="bid", old=0.50, new=0.51, count=LOT):
    """What `plan_side` emits when our side got better: a linked cancel + place."""
    pid = _pair_id(ticker, side, old)
    return (cancel(ticker, side, old, count, reason=f"improve_{FRONT_IMPROVE_TICKS}t",
                   pair=pid),
            place(ticker, side, new, count, reason="improved", pair=pid))


print("costs are the account's, per item")
check("a cancel costs 2", CANCEL_TOKEN_COST, 2)
check("a create costs 10", ORDER_TOKEN_COST, 10)
check("so a requote PAIR costs 12", CANCEL_TOKEN_COST + ORDER_TOKEN_COST, 12)

print("\nINVARIANT 1 -- a reducing cancel is never deferred, even at zero budget")
cs = [cancel(f"M{i}") for i in range(20)]
p, c, d = admit([], cs, tokens=0.0, reserve=0.0)
check("all 20 negative-edge cancels admitted on an empty bucket", len(c), 20)
check("  and nothing was placed", len(p), 0)
check("  deferred nothing", d["n_deferred"], 0)
check("  and it says so: the bucket went into debt", d["reduce_over_budget"], True)
check("  tokens_left is negative, honestly", d["tokens_left"] < 0, True)

p, c, d = admit([], [cancel("M0", reason="no_longer_quoting")], tokens=0.0, reserve=0.0)
check("no_longer_quoting is reducing too", len(c), 1)

print("\n  ... and a healthy budget does not flag the alarm")
p, c, d = admit([], cs, tokens=BIG, reserve=0.0)
check("20 cancels x 2 tokens = 40 spent", d["cost"], 40.0)
check("  no alarm", d["reduce_over_budget"], False)

print("\nINVARIANT 2 -- a replacement cancel is never sent without its place")
cx, px = improve_pair("M1")
# 12 tokens is exactly one pair; 11 is one short.
p, c, d = admit([px], [cx], tokens=12.0, reserve=0.0)
check("at exactly the pair's cost, both go", (len(c), len(p)), (1, 1))
p, c, d = admit([px], [cx], tokens=11.0, reserve=0.0)
check("one token short, NEITHER goes", (len(c), len(p)), (0, 0))
check("  the pair is deferred whole", d["n_deferred"], 1)
check("  and its cost is reported", d["deferred_cost"], 12.0)

print("\n  an orphaned replacement cancel is dropped, not sent bare")
# The place was stripped upstream (risk block), or the plan came from an older path. Sending the
# cancel alone would unquote the level to save 10 tokens.
p, c, d = admit([], [cx], tokens=BIG, reserve=0.0)
check("bare improve_* cancel is not admitted", len(c), 0)
check("  and not counted as deferred work either", d["n_deferred"], 0)

print("\n  an orphaned replacement place is dropped too -- it would DOUBLE the rung")
# The order it replaces is still resting, so placing the new one leaves two.
p, c, d = admit([px], [], tokens=BIG, reserve=0.0)
check("bare 'improved' place is not admitted", len(p), 0)

print("\n  a genuine top-up carries no pair and IS admitted alone")
p, c, d = admit([place("M2", reason="top_up")], [], tokens=BIG, reserve=0.0)
check("solo top-up goes", len(p), 1)
check("  costing 10", d["cost"], 10.0)

print("\nPRIORITY -- reduce, then improve, then top-up")
cx, px = improve_pair("M9")
plan_p = [px, place("M8", reason="top_up")]
plan_c = [cx, cancel("M7")]
items = sorted(build_items(plan_p, plan_c), key=budget.Item.key)
check("order of classes", [i.cls for i in items], [REDUCE, IMPROVE, TOPUP])

# Budget for the reduce (2) + exactly one more unit. IMPROVE costs 12, TOPUP costs 10, so a
# budget of 14 can afford the improve OR the topup, and must choose the improve.
p, c, d = admit(plan_p, plan_c, tokens=14.0, reserve=0.0)
check("reduce always in", d["by_class"]["reduce"], 1)
check("  improve wins the last 12 tokens", d["by_class"]["improve"], 1)
check("  top-up is deferred", d["deferred_by_class"]["topup"], 1)
assert_true("  a stale resting price outranks a missing rung",
            d["by_class"]["topup"] == 0)

print("\n  ... and greedy-by-priority refuses the knapsack trade")
# 12 tokens buys one improve, or one topup with 2 left over. A knapsack maximising item count
# would take the topup. Priority order must not.
cx2, px2 = improve_pair("M9")
p, c, d = admit([px2, place("Z1"), place("Z2")], [cx2], tokens=12.0, reserve=0.0)
check("one improve, not one cheaper top-up", (d["by_class"]["improve"], d["by_class"]["topup"]),
      (1, 0))

print("\nCOLD outranks TOP-UP: an unquoted side before a thicker ladder")
# Two solo places. COLD/bid has nothing resting; WARM/bid already shows rungs. Under a budget for
# exactly one place, the unquoted side must win -- it earns nothing and holds no queue position,
# while the other is reaching for an extra rung on a side that is already working.
cold_p = place("COLD", "bid")
warm_p = place("WARM", "bid")
resting = {("WARM", "bid")}
items = sorted(build_items([warm_p, cold_p], [], resting=resting), key=budget.Item.key)
check("the unquoted side is classified cold", [budget.CLASS_NAME[i.cls] for i in items],
      ["cold", "topup"])
p, c, d = admit([warm_p, cold_p], [], tokens=10.0, reserve=0.0, resting=resting)
check("  and it takes the only ten tokens", [x["ticker"] for x in p], ["COLD"])
check("  the thicker ladder waits", d["deferred_by_class"]["topup"], 1)

print("  ... but a LOWER-risk cold side still beats a higher-risk top-up")
# Class dominates urgency, deliberately: being unquoted is a bigger deficit than being one rung
# short, whatever the Greeks say about the two markets.
em_cw = {"COLD": {"risk_u": 0.01}, "WARM": {"risk_u": 0.99}}
items = sorted(build_items([warm_p, cold_p], [], em_map=em_cw, resting=resting),
               key=budget.Item.key)
check("class beats urgency across classes", [i.ticker for i in items], ["COLD", "WARM"])

print("  ... and an omitted `resting` changes nothing")
# The default must reproduce the pre-2026-08-26 behaviour exactly: with no evidence about the
# book, no place may be promoted, or every solo place would outrank every real top-up.
items = sorted(build_items([warm_p, cold_p], []), key=budget.Item.key)
check("no resting set -> every solo place is a top-up",
      sorted({budget.CLASS_NAME[i.cls] for i in items}), ["topup"])

print("  ... and an EMPTY resting set means every side really is cold")
items = sorted(build_items([warm_p, cold_p], [], resting=set()), key=budget.Item.key)
check("empty resting set -> both are cold",
      sorted({budget.CLASS_NAME[i.cls] for i in items}), ["cold"])

print("\nURGENCY ranks on risk_u, so NDX and SPX compete like-for-like")
# Raw R is not comparable between underlyings -- R's delta term is dollars per one-point index
# move, so NDX at ~29,300 carries ~3.8x fewer points for the same proportional move. An NDX
# market at the 99th percentile of its own curve must outrank an SPX market at the 10th, even
# though its raw R is the smaller number.
cn, pn = improve_pair("KXNASDAQ100U-X")
cs, ps = improve_pair("KXINXU-X")
em_u = {"KXNASDAQ100U-X": {"R": 0.030, "risk_u": 0.99},
        "KXINXU-X":       {"R": 0.045, "risk_u": 0.10}}
items = sorted(build_items([pn, ps], [cn, cs], em_map=em_u), key=budget.Item.key)
check("the higher PERCENTILE goes first, not the higher R",
      [i.ticker for i in items], ["KXNASDAQ100U-X", "KXINXU-X"])
# And with no risk_u -- a flat curve under --no-edge -- it falls back to R and ranks the old way.
em_r = {"KXNASDAQ100U-X": {"R": 0.030}, "KXINXU-X": {"R": 0.045}}
items = sorted(build_items([pn, ps], [cn, cs], em_map=em_r), key=budget.Item.key)
check("  falls back to R when there is no percentile",
      [i.ticker for i in items], ["KXINXU-X", "KXNASDAQ100U-X"])

print("\nURGENCY within a class")
print("  reduce: dollars of wrongness = |edge| x contracts, not edge alone")
# Both 1c through a 0.50 theo on the bid side; one is 50 lots, the other 10.
theos = {"BIG": 0.50, "SMALL": 0.50}
big = cancel("BIG", "bid", 0.51, "50.00")
small = cancel("SMALL", "bid", 0.51, "10.00")
items = sorted(build_items([], [small, big], theos=theos), key=budget.Item.key)
check("the 50-lot leaves first", [i.ticker for i in items], ["BIG", "SMALL"])

print("  improve: ranked by the market's R, the same number the edge curve is calibrated on")
c1, p1 = improve_pair("CALM")
c2, p2 = improve_pair("HOT")
em = {"CALM": {"R": 0.001}, "HOT": {"R": 0.25}}
items = sorted(build_items([p1, p2], [c1, c2], em_map=em), key=budget.Item.key)
check("the high-R market is requoted first", [i.ticker for i in items], ["HOT", "CALM"])
# Under a budget for exactly one pair, the calm market is the one that waits.
p, c, d = admit([p1, p2], [c1, c2], tokens=12.0, reserve=0.0, em_map=em)
check("  and it is the one that fits", [x["ticker"] for x in p], ["HOT"])

print("  ties break on ticker, so a deferral pattern is readable across cycles")
ca, pa = improve_pair("AAA")
cb, pb = improve_pair("BBB")
items = sorted(build_items([pb, pa], [cb, ca]), key=budget.Item.key)
check("stable, not dict order", [i.ticker for i in items], ["AAA", "BBB"])

print("\nTHE RESERVE holds tokens back from improves, not from reduces")
cx, px = improve_pair("M1")
p, c, d = admit([px], [cx], tokens=100.0, reserve=95.0)
check("12-token pair does not fit above a 95 reserve of 100", len(p), 0)
p, c, d = admit([px], [cx], tokens=100.0, reserve=88.0)
check("  it fits at a reserve of 88", len(p), 1)
p, c, d = admit([], [cancel("M1")], tokens=100.0, reserve=100.0)
check("a reducing cancel ignores the reserve entirely", len(c), 1)

print("\nthe full board")
# 532 markets x 2 sides x 3 rungs, all wanting a requote, against one cycle's 600 tokens with the
# configured write reserve held back.
RESERVE = config.WRITE_RESERVE_TOKENS
SPENDABLE = 600.0 - RESERVE
N_FIT = int(SPENDABLE // 12)            # whole requote pairs the spendable tokens buy
pairs = [improve_pair(f"KXINXU-{i}", side=s, old=0.50 + 0.01 * r)
         for i in range(532) for s in ("bid", "ask") for r in range(3)]
plan_c = [x[0] for x in pairs]
plan_p = [x[1] for x in pairs]
check("the board wants 3,192 requotes", len(plan_p), 3192)
check("  which is 38,304 tokens", len(plan_p) * 12, 38304)
assert_true("  (the worked example needs a reserve that leaves room for some pairs)",
            0 < N_FIT < 3192, f"N_FIT={N_FIT}")
p, c, d = admit(plan_p, plan_c, tokens=600.0, reserve=RESERVE)
check("  one cycle affords SPENDABLE // 12 of them", len(p), N_FIT)
check("    spending all but the remainder of the spendable tokens", d["cost"], 12.0 * N_FIT)
assert_true("    and the remainder is less than one more pair", SPENDABLE - d["cost"] < 12)
check("    and deferring the rest", d["n_deferred"], 3192 - N_FIT)
assert_true("    the deferral is REPORTED, not silent",
            d["deferred_cost"] > 0 and len(d["deferred_sample"]) == 5)

print("\n  the same board, with 400 negative-edge cancels mixed in")
red = [cancel(f"KXINXU-{i}", reason="edge_negative") for i in range(400)]
p, c, d = admit(plan_p, plan_c + red, tokens=600.0, reserve=RESERVE)
check("all 400 reducing cancels went", d["by_class"]["reduce"], 400)
check("  costing 800 -- more than the whole bucket", d["reduce_cost"], 800.0)
check("  so the bucket is in debt and says so", d["reduce_over_budget"], True)
check("  and NO improve was admitted behind them", d["by_class"]["improve"], 0)
assert_true("  which is the right order: exit the bad risk, defer the penny",
            len(c) == 400 and len(p) == 0)

# =========================================================================
print("\n\nCAPITAL SELECTION -- the collateral caps decide which markets are held")
# =========================================================================
from strategy.budget import ladder_value, moneyness_skip, select_markets

print("\na two-sided ladder costs its FACE at any fair -- p on the bid, 1-p on the ask")
# This is the result that reframes the whole problem: moneyness does NOT make a two-sided market
# cheaper to hold. It only makes it cheaper by quoting FEWER RUNGS, which is what the band does.
# AN EXPLICIT THREE-RUNG LADDER, not `config.QUOTE_TIER_COUNTS`, which may carry a single rung.
# `decide_ladder`, `rung_min_edge_micros` and `tier_offsets` take any number of rungs and the rank
# arithmetic is exactly what these tests exist to pin. Reading the config's tier count here would
# have the ladder tests quietly stop testing the ladder if it were configured to one rung.
BAND, WING = (LOT, LOT, LOT), (LOT,)
BAND_V = 3.0 * config.LOT_SIZE           # face of a two-sided 3-rung ladder, in dollars
WING_V = 1.0 * config.LOT_SIZE           # face of a two-sided single rung
BAND_C = 2.0 * 3 * config.LOT_SIZE       # contracts at rest for a band market (both sides)
WING_C = 2.0 * config.LOT_SIZE
for fair in (0.03, 0.25, 0.50, 0.75, 0.97):
    approx(f"  3-tier at fair {fair}", ladder_value(fair, BAND), BAND_V, tol=1e-9)
    approx(f"  1-tier at fair {fair}", ladder_value(fair, WING), WING_V, tol=1e-9)
assert_true("a wing costs a THIRD of a band market, at every fair",
            abs(ladder_value(0.5, BAND) / ladder_value(0.5, WING) - 3.0) < 1e-9)

print("\nso the caps land on the same number of markets, both ways")
N_BAND = int(config.MAX_RESTING_ORDER_VALUE // BAND_V)    # band markets the value cap affords
N_WING = int(config.MAX_RESTING_ORDER_VALUE // WING_V)
check("value cap / band ladder face = contract cap / band contracts",
      int(config.MAX_RESTING_CONTRACTS_TOTAL // BAND_C), N_BAND)
check("  and likewise for wings", int(config.MAX_RESTING_CONTRACTS_TOTAL // WING_C), N_WING)
assert_true("  a wing is a third of a band market, so at least three times as many fit",
            N_WING >= 3 * N_BAND, f"{N_WING} vs {N_BAND}")
assert_true("  (the worked examples below need a cap that funds tens of band markets)",
            20 <= N_BAND <= 200, f"N_BAND={N_BAND}")


def mkt(t, stype="between"):
    return {"ticker": t, "strike_type": stype}


def board(n, stype="between", base_r=1.0):
    """n markets, R descending, all at a 0.50 fair so every ladder costs the same."""
    ms = [mkt(f"M{i:03d}", stype) for i in range(n)]
    theos = {m["ticker"]: 0.50 for m in ms}
    em = {m["ticker"]: {"R": base_r - i * 0.001, "quotable": True} for i, m in enumerate(ms)}
    return ms, theos, em


print("\nthe full universe is cut to what the capital allows")
ms, theos, em = board(532)
chosen, d = select_markets(ms, theos, em, counts=BAND)
check("532 markets in", d["universe"], 532)
# Uniform counts: every candidate is band-shaped, so there are no wings to reserve for and the
# band pass gets the whole cap rather than BAND_CAPITAL_FRACTION of it.
check("  N_BAND chosen at the band ladder's face each", len(chosen), N_BAND)
check("  and the value is inside the cap", d["value"] <= config.MAX_RESTING_ORDER_VALUE, True)
check("  as are the contracts", d["contracts"] <= config.MAX_RESTING_CONTRACTS_TOTAL, True)
check("  no capital left idle for wings that do not exist", d["value"], N_BAND * BAND_V)
check("  the rest are REPORTED, not silently dropped", d["unaffordable"], 532 - N_BAND)
assert_true("  and they are the lowest-R ones", chosen == {f"M{i:03d}" for i in range(N_BAND)})

print("\nthe moneyness floors are named rules, checkable without R")
FB, FO = config.SELECT_FLOOR_BRACKET, config.SELECT_FLOOR_ONE_SIDED
check("a 4pm bracket AT the bracket floor is skipped", moneyness_skip("between", FB) is not None,
      True)
check("  a cent above it is not", moneyness_skip("between", round(FB + 0.01, 2)), None)
check("an hourly rung AT the one-sided floor is skipped",
      moneyness_skip("greater_or_equal", FO) is not None, True)
check("  a cent above it is not", moneyness_skip("greater_or_equal", round(FO + 0.01, 2)), None)
check("  and at 1 - floor too -- the other wing of the same board",
      moneyness_skip("greater_or_equal", round(1 - FO, 2)) is not None, True)
check("  a cent inside that is not", moneyness_skip("greater_or_equal", round(1 - FO - 0.01, 2)),
      None)
check("  at 50c it is not", moneyness_skip("greater_or_equal", 0.50), None)
# The 4pm board's floor is one-sided: a bracket near 1.0 is not a thing (they sum to 1 across 30
# brackets), so only the low tail is named.
check("a bracket at 97c is NOT skipped by the bracket rule", moneyness_skip("between", 0.97), None)
check("no fair is a skip", moneyness_skip("between", None), "no_fair")

print("\n  and the floors take effect before ranking")
ms = [mkt("DEEP", "greater_or_equal"), mkt("ATM", "greater_or_equal")]
theos = {"DEEP": 0.99, "ATM": 0.50}
em = {"DEEP": {"R": 99.0, "quotable": True}, "ATM": {"R": 0.01, "quotable": True}}
chosen, d = select_markets(ms, theos, em, counts=BAND)
check("the 99% market is excluded despite a huge R", chosen, {"ATM"})
assert_true("  and the reason is recorded", any("one_sided" in kk for kk in d["skipped"]))

print("\na market with no R is not selected -- edgemgr's rule, restated here")
ms, theos, em = board(3)
em["M001"] = {"quotable": True}                     # no R
chosen, d = select_markets(ms, theos, em, counts=BAND)
check("two chosen", len(chosen), 2)
check("  and the no-R one is named", d["skipped"].get("no_R"), 1)
em["M001"] = {"R": 0.5, "quotable": False}
chosen, d = select_markets(ms, theos, em, counts=BAND)
check("not-quotable is excluded too", d["skipped"].get("not_quotable"), 1)

print("\nHYSTERESIS -- an incumbent is not evicted by a hair")
# Without this, R refreshing every EDGE_RECOMPUTE_S flips boundary markets in and out, and each
# flip is a six-order ladder cancelled and re-placed: 72 tokens and every queue position on it.
ms, theos, em = board(N_BAND + 4)
first, _ = select_markets(ms, theos, em, counts=BAND)
check("cold start takes the top N_BAND", len(first), N_BAND)
# Now shuffle R so the incumbent at the boundary drops slightly below a newcomer.
INC, CHAL, NEXT = f"M{N_BAND - 1:03d}", f"M{N_BAND + 3:03d}", f"M{N_BAND:03d}"
r_edge = em[INC]["R"]
em2 = dict(em)
em2[INC] = {"R": r_edge - 0.0005, "quotable": True}     # incumbent, just under
em2[CHAL] = {"R": r_edge, "quotable": True}             # challenger, just over
second, d2 = select_markets(ms, theos, em2, counts=BAND, incumbents=first)
check("the incumbent is retained", INC in second, True)
check("  and the set did not churn", len(first - second), 0)
check("  retained count is reported", d2["retained"], N_BAND)

print("  ... but a genuinely worse market IS dropped")
em3 = dict(em)
em3["M000"] = {"R": 0.0001, "quotable": True}       # incumbent collapses
fourth, d4 = select_markets(ms, theos, em3, counts=BAND, incumbents=first)
check("it falls below the softened bar", "M000" in fourth, False)
check("  and that is reported", d4["dropped_incumbents"], 1)
check("  its slot went to the next-best market", NEXT in fourth, True)

print("\n  a market that is no longer in the universe cannot be retained")
ms_small = [m for m in ms if m["ticker"] != "M000"]
fifth, d5 = select_markets(ms_small, theos, em, counts=BAND, incumbents=first)
check("M000 is gone", "M000" in fifth, False)
check("  still a full board", len(fifth), N_BAND)

print("\nselection and admission rank by the SAME number")
# If they disagreed, a market could be selected as important and then starved of the writes that
# keep its quote current -- chosen and permanently stale.
ms, theos, em = board(5)
chosen, _ = select_markets(ms, theos, em, counts=BAND)
cx, px = improve_pair("M000")
cy, py = improve_pair("M004")
items = sorted(build_items([px, py], [cx, cy], em_map=em), key=budget.Item.key)
check("the highest-R market is requoted first", items[0].ticker, "M000")
assert_true("  and it is also the first selected", "M000" in chosen)

print("\nTWO PASSES -- the band must not eat the money the wings need")
# Band markets are BY CONSTRUCTION the high-R ones, so one undivided R ranking funds them first and
# never reaches a wing. That would silently delete the breadth this whole design buys.
from strategy import band as bandmod

# More band candidates than BAND_CAPITAL_FRACTION of the caps can buy, so the fraction binds.
N_BAND_CANDS = int(config.MAX_RESTING_ORDER_VALUE * config.BAND_CAPITAL_FRACTION // BAND_V) + 7


def mixed(n=200, n_band=N_BAND_CANDS):
    ms, theos, em, counts = [], {}, {}, {}
    for i in range(n):
        t = f"M{i:03d}"
        fair = 0.06 + 0.88 * (i + 0.5) / n
        ms.append({"ticker": t, "strike_type": "greater_or_equal", "floor_strike": 7000 + 5 * i})
        theos[t] = fair
        # R peaks at the money, which is also where the band is -- that is the whole hazard.
        em[t] = {"quotable": True, "R": 0.30 * (1 - abs(fair - 0.5) * 2) ** 2 + 1e-4}
        lo = (n - n_band) // 2
        # BAND/WING explicitly, not `bandmod.counts_for`: the config may give both shapes ONE
        # rung, and the two-pass split this section tests only exists when the two shapes
        # differ. `counts_for`'s own behaviour is `test_band`'s job.
        counts[t] = BAND if lo <= i < lo + n_band else WING
    return ms, theos, em, counts

CAP_V, CAP_C = config.MAX_RESTING_ORDER_VALUE, config.MAX_RESTING_CONTRACTS_TOTAL
FRAC = config.BAND_CAPITAL_FRACTION
NB = int(min(CAP_V * FRAC // BAND_V, CAP_C * FRAC // BAND_C))     # band markets the fraction buys
NW = int(min((CAP_V - NB * BAND_V) // WING_V, (CAP_C - NB * BAND_C) // WING_C))
ms, theos, em, counts = mixed()
chosen, d = select_markets(ms, theos, em, counts=counts)
check("band candidates", d["band"]["candidates"], N_BAND_CANDS)
check("  budgeted to its fraction", d["band"]["budget"], CAP_V * FRAC)
check("  which buys as many band ladders as fit", d["band"]["chosen"], NB)
check("  spending", d["band"]["value"], NB * BAND_V)
assert_true("  the fraction binds -- some band candidates are left unfunded", NB < N_BAND_CANDS)
check("wing candidates", d["wing"]["candidates"], 200 - N_BAND_CANDS)
check("  funded from the remainder", d["wing"]["budget"], CAP_V - NB * BAND_V)
check("  which buys as many single rungs as fit", d["wing"]["chosen"], NW)
check("NB + NW markets in total", len(chosen), NB + NW)
check("  spending", d["value"], NB * BAND_V + NW * WING_V)
check("  contracts", d["contracts"], NB * BAND_C + NW * WING_C)
assert_true("  within a wing of the whole value cap", CAP_V - d["value"] < WING_V)
assert_true("  more than a uniform 3-tier board could afford", len(chosen) > N_BAND,
            f"{len(chosen)} vs {N_BAND}")

print("  ... and every wing chosen really is a wing")
nb = sum(1 for t in chosen if len(counts[t]) > 1)
check("NB band + NW wing", (nb, len(chosen) - nb), (NB, NW))

print("\n  a THIN band hands its unspent money to the wings, not to nobody")
# Late in the day vol decays, the band shrinks, and the capital should buy more breadth.
ms, theos, em, counts = mixed(n=200, n_band=4)
chosen, d = select_markets(ms, theos, em, counts=counts)
NW4 = int(min((CAP_V - 4 * BAND_V) // WING_V, (CAP_C - 4 * BAND_C) // WING_C, 196))
check("4 band markets cost 4 band ladders", d["band"]["value"], 4 * BAND_V)
check("  the wing pass is handed the REST, not the complement of the fraction",
      d["wing"]["budget"], CAP_V - 4 * BAND_V)
assert_true("  which is more than the complement", CAP_V - 4 * BAND_V > CAP_V * (1 - FRAC))
check("  buying every wing it can afford", d["wing"]["chosen"], NW4)
assert_true("  and the cap is still respected", d["value"] <= config.MAX_RESTING_ORDER_VALUE)

print("\n  the two passes do not compete: a wing cannot displace a band market")
ms, theos, em, counts = mixed()
# Give one wing an enormous R. It must not take a band market's slot -- they are not spending the
# same money, and a wing that displaced a band market would get one rung where three were wanted.
em["M000"] = {"quotable": True, "R": 99.0}
chosen2, d2 = select_markets(ms, theos, em, counts=counts)
check("the huge-R wing is chosen", "M000" in chosen2, True)
check("  but the band pass is unchanged", d2["band"]["chosen"], NB)
check("  and it was funded as a WING", len(counts["M000"]), 1)

finish()
