"""Pin the quoting ladder against the worked example and the guards.

    python -m strategy.test_quoting

No pytest dependency -- this has to be runnable in the minutes before a session.
"""
import sys
from harness import check

from strategy.quoting import (
    EDGE_MIN, TICK, Action, decide, should_improve, should_pull, top_of_book,
)

# The shared list, not a local one: `harness`'s helpers append to it, and so do the
# bespoke `.append(...)` calls below. Two lists would mean the local checks vote and
# the shared ones do not, and the gate would print PASS while a helper had failed.
from harness import FAILURES as FAILS


def approx(label, got, want, tol=1e-9):
    ok = got is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got={got} want={want}")
    if not ok:
        FAILS.append(label)


import config  # noqa: E402
from strategy.quoting import EDGE_MIN_MICROS, MICROS, ONE_SIDED_WIDTH  # noqa: E402

BAR = EDGE_MIN_MICROS / MICROS            # the default penny bar, in dollars

# The worked numbers below are computed for the configured QUOTE_OFFSET (2c), the configured
# QUOTE_TIGHTEN_FRACTION (1/2) and ONE_SIDED_WIDTH (8 ticks). Pin those first, so a changed
# constant fails HERE, by name, rather than as a mystery price further down.
check("worked example assumes QUOTE_OFFSET", config.QUOTE_OFFSET, 0.02)
check("worked example assumes QUOTE_TIGHTEN_FRACTION", config.QUOTE_TIGHTEN_FRACTION, 0.5)
check("worked example assumes ONE_SIDED_WIDTH", ONE_SIDED_WIDTH, 8)

print("\nTHE worked example -- 4/12 book, theo 0.10")
# Bid: target = 0.10 - 0.02 = 0.08. Width is 0.08 > 2 ticks and the target would improve the
# touch, so it is capped QUOTE_TIGHTEN_FRACTION of the way in from the bid: 0.04 + 0.08/2 = 0.08.
b = decide("bid", 0.10, 0.04, 0.12)
check("bid kind", b.kind, "improve")
approx("bid price", b.price, 0.08)
approx("bid edge", b.edge, 0.02)
# At a fraction of 1/2 the tighten cap IS the mid, so the cap-at-mid rule (tighten=None) agrees.
b_mid = decide("bid", 0.10, 0.04, 0.12, tighten=None)
approx("the cap-at-mid rule gives the same 0.08 at a fraction of 1/2", b_mid.price, 0.08)

print("\nthe tighten cap at an explicit fraction, so its difference from cap-at-mid stays visible")
# Explicit offset and fraction: this pins the mechanism independently of the configured values.
# 4/12 book, theo 0.10, offset 1c -> target 0.09; a quarter of the way in: 0.04 + 0.08/4 = 0.06.
q = decide("bid", 0.10, 0.04, 0.12, offset=0.01, tighten=0.25)
approx("4/12 theo 10, quarter-way cap -> bid 0.06", q.price, 0.06)
approx("  edge 0.04", q.edge, 0.04)
approx("  while cap-at-mid would have said 0.08",
       decide("bid", 0.10, 0.04, 0.12, offset=0.01, tighten=None).price, 0.08)
# Book 8 at 15, theo 12, offset 1c -> target 11; quarter-way cap 8 + 7/4 = 9.75 -> 9.
w = decide("bid", 0.12, 0.08, 0.15, offset=0.01, tighten=0.25)
approx("8@15 theo 12, quarter-way cap -> bid 0.09", w.price, 0.09)
approx("  and cap-at-mid would have said 0.11",
       decide("bid", 0.12, 0.08, 0.15, offset=0.01, tighten=None).price, 0.11)

# A target already at or behind the touch is untouched by the cap.
narrow = decide("bid", 0.05, 0.04, 0.12)
approx("a target behind the touch is not capped up", narrow.price, 0.04)
# Offer: target = 0.12, the cap would pull it toward 0.08 but max(0.12, 0.08) = 0.12 -- which only
# matches the touch, so it does not improve. Pennying 0.11 earns +0.01, which clears the bar.
# (The cap only ever pushes the offer DOWN when the target sits below it.)
a = decide("ask", 0.10, 0.04, 0.12)
check("ask kind", a.kind, "penny")
approx("ask price", a.price, 0.11)
approx("ask edge", a.edge, 0.01)

print("\nthe offer improves when its target is inside the touch")
# theo 0.09 on a 4/12 book: ask target 0.11, cap -> max(0.11, 0.08) = 0.11, edge +0.02.
a2 = decide("ask", 0.09, 0.04, 0.12)
check("ask on 4/12 theo .09", (a2.kind, a2.price), ("improve", 0.11))

print("\n2c-wide book -- cap does not bind; penny only at or above the bar")
# 0.10/0.12: the bid target is behind the touch, so no improve. Penny 0.11 with theo exactly
# 0.11 + BAR -> edge is EXACTLY the bar, which clears the inclusive test -> penny.
t = decide("bid", round(0.11 + BAR, 6), 0.10, 0.12)
check("2c penny at exactly the bar -> post", (t.kind, t.price), ("penny", 0.11))
# and a hair under the bar -> join instead
t_under = decide("bid", round(0.11 + BAR - 0.0001, 6), 0.10, 0.12)
check("2c penny just under the bar -> join", t_under.kind, "join")
# theo 0.12: penny 0.11 -> edge 0.01, comfortably over the bar, so penny.
t2 = decide("bid", 0.12, 0.10, 0.12)
check("2c penny worth it (price)", t2.price, 0.11)
check("2c penny worth it (kind)", t2.kind in ("improve", "penny"), True)

print("\nnever cross -- 1c-wide book cannot be pennied")
c = decide("bid", 0.99, 0.10, 0.11)
ok = c.price is None or c.price < 0.11
print(f"  {'ok  ' if ok else 'FAIL'} bid never reaches the ask: price={c.price} ask=0.11")
if not ok:
    FAILS.append("no-cross bid")
c2 = decide("ask", 0.01, 0.10, 0.11)
ok2 = c2.price is None or c2.price > 0.10
print(f"  {'ok  ' if ok2 else 'FAIL'} ask never reaches the bid: price={c2.price} bid=0.10")
if not ok2:
    FAILS.append("no-cross ask")

print("\n1c-wide book JOINS the touch -- it does not stop quoting, and it does not take")
# The no-cross checks above pass if we simply refuse to quote, which is NOT the policy: on a
# 1c-wide book where the theo would justify taking, we keep the same passive quote. Pin the kind.
j = decide("bid", 0.99, 0.10, 0.11)
check("1c-wide bid joins its own touch (kind)", j.kind, "join")
check("1c-wide bid joins its own touch (price)", j.price, 0.10)
j2 = decide("ask", 0.01, 0.10, 0.11)
check("1c-wide ask joins its own touch (kind)", j2.kind, "join")
check("1c-wide ask joins its own touch (price)", j2.price, 0.11)

print("\none-sided book is treated as a ONE_SIDED_WIDTH-wide market, not as no width")
# Skipping the tighten cap on a missing side would post the full-edge target -- here 0.10, six
# cents inside the only price anyone is showing. The synthetic far side is 8 ticks away, so the
# cap is 0.04 + 0.08/2 = 0.08.
check("with no book at all the full-edge target would be 0.10",
      decide("bid", 0.12, None, None).price, 0.10)
o = decide("bid", 0.12, 0.04, None)
check("lone bid: capped, not full-edge", o.price, 0.08)
# Lone ask at 0.12: synthetic bid 0.04, same cap.
o2 = decide("bid", 0.12, None, 0.12)
check("lone ask: capped from the synthetic bid", o2.price, 0.08)
# Both touches missing: nothing to anchor to, the plain target stands.
o3 = decide("bid", 0.50, None, None)
check("empty book still quotes the plain target", o3.price, 0.48)
# The synthetic side must also bind the cross guard.
o4 = decide("ask", 0.98, 0.95, None)
check("lone bid: offer never crosses the real bid", o4.price is None or o4.price > 0.95, True)

print("\nnever emit an untradeable price")
# theo 0.00 with an empty bid ladder: target = floor(-0.01) = -0.01. Seen 15x in the first
# dry run before the guard existed.
z = decide("bid", 0.0, None, 0.02)
ok = z.price is None or z.price >= 0.01
print(f"  {'ok  ' if ok else 'FAIL'} theo 0 on empty book -> no negative bid: price={z.price}")
if not ok:
    FAILS.append("negative price")
z2 = decide("ask", 1.0, 0.98, None)
ok2 = z2.price is None or z2.price <= 0.99
print(f"  {'ok  ' if ok2 else 'FAIL'} theo 1 -> no price above 0.99: price={z2.price}")
if not ok2:
    FAILS.append("price above max")

print("\nno positive edge anywhere -> no quote")
check("theo below the bid", decide("bid", 0.02, 0.10, 0.12).kind, "none")

print("\npull rule -- only when alone at our level")
check("stale and alone -> pull", should_pull("bid", 0.05, 0.10, True), True)
check("stale but not alone -> hold", should_pull("bid", 0.05, 0.10, False), False)
check("barely off (inside the bar) -> hold",
      should_pull("bid", round(0.10 - BAR + 0.001, 6), 0.10, True), False)

print("\nimprove rule -- strictly better and worth at least the bar")
check("better by 2c -> amend",
      should_improve("bid", 0.12, 0.08, Action("penny", 0.10, 0.02, "")), True)
_thin = round(BAR - 0.001, 6)
check("better but thin -> hold",
      should_improve("bid", 0.10 + _thin, 0.08, Action("penny", 0.10, _thin, "")), False)
check("worse price -> hold", should_improve("bid", 0.12, 0.10, Action("penny", 0.09, 0.03, "")), False)

print("\ntop_of_book -- NO ladder is a YES offer at 1 - p")
bb, ba, bs, asz = top_of_book({"yes": [["0.03", "50"], ["0.04", "120"]],
                               "no": [["0.85", "10"], ["0.88", "40"]]})
# and the REAL live shape, which nests under orderbook_fp with *_dollars keys
r_bb, r_ba, _, _ = top_of_book({"orderbook_fp": {
    "yes_dollars": [["0.0300", "50.00"], ["0.0400", "120.00"]],
    "no_dollars":  [["0.8500", "10.00"], ["0.8800", "40.00"]]}})
approx("live-shape best bid", r_bb, 0.04)
approx("live-shape best ask", r_ba, 0.12)
# empty yes ladder = no bids at all, must not crash or invent one
e_bb, e_ba, _, _ = top_of_book({"orderbook_fp": {
    "yes_dollars": [], "no_dollars": [["0.9900", "3504.00"]]}})
check("empty yes ladder -> no bid", e_bb, None)
approx("ask from lone no level", e_ba, 0.01)
approx("best bid", bb, 0.04)
approx("best ask", ba, 0.12)     # 1 - 0.88
approx("bid size", bs, 120)
approx("ask size", asz, 40)
check("empty book", top_of_book({}), (None, None, None, None))
check("one-sided book does not crash", decide("bid", 0.10, 0.04, None).kind in
      ("improve", "penny", "join", "none"), True)


# --- the tiered ladder ---------------------------------------------------
#
# THE IDENTITY FIRST. With tiers off and no calibrated edge bar, decide_ladder must be decide()
# wrapped in one Rung -- price, count and edge all identical. This is the item-D equivalent of
# `ratio == 1 at 16:00` for the hourly model: proof the new machinery changed nothing until it
# was asked to. (The old third example, `BLEND_W_SPREAD = 1.0`, named a constant that no longer
# exists -- the two-estimator blend went with the skew model.)
from strategy.quoting import decide_ladder, tier_offsets, ladder_prices_ordered, Rung  # noqa: E402
import config  # noqa: E402

# AN EXPLICIT THREE-RUNG LADDER, not `config.QUOTE_TIER_COUNTS`, which may carry a single rung.
# `decide_ladder`, `rung_min_edge_micros` and `tier_offsets` take any number of rungs and the rank
# arithmetic is exactly what these tests exist to pin. Reading the config's tier count here would
# have the ladder tests quietly stop testing the ladder if it were configured to one rung.
_LOT = f"{config.LOT_SIZE:.2f}"
C = (_LOT, _LOT, _LOT)
_books = [(0.40, 0.44), (None, 0.44), (0.40, None), (None, None),
          (0.01, 0.03), (0.97, 0.99), (0.04, 0.12)]
_theos = [0.005, 0.02, 0.10, 0.42, 0.50, 0.90, 0.995]
_ident = 0
for _side in ("bid", "ask"):
    for _t in _theos:
        for _bb, _ba in _books:
            _a = decide(_side, _t, _bb, _ba)
            _l = decide_ladder(_side, _t, _bb, _ba, C, tiers_enabled=False)
            if _a.price is None:
                check(f"identity none {_side} {_t} {_bb}/{_ba}", _l, [])
            else:
                check(f"identity {_side} {_t} {_bb}/{_ba}",
                      (_l[0].price, _l[0].count, _l[0].edge, len(_l)),
                      (_a.price, C[0], _a.edge, 1))
            _ident += 1
print(f"  (identity checked on {_ident} side/theo/book combinations)")

# Tier offsets step on the EDGE curve's scale, not a percentile rank, between the low and high
# ends of TIER_OFFSET_RANGE.
from strategy.quoting import TIER_OFFSET_RANGE  # noqa: E402
_LO = [lo for lo, _hi in TIER_OFFSET_RANGE]
_HI = [hi for _lo, hi in TIER_OFFSET_RANGE]
check("tier_offsets at scale 0 is the low end", tier_offsets(0.0), _LO)
check("tier_offsets at scale 1 is the high end", tier_offsets(1.0), _HI)
check("tier_offsets clamps above 1", tier_offsets(9.9), _HI)
check("tier_offsets clamps below 0", tier_offsets(-1.0), _LO)
check("tier_offsets tolerates None", tier_offsets(None), _LO)
# The exponential edge curve compresses the bulk, so a typical market rests at the low end and
# only the tail moves out. Worked for the configured range ((1,3),(3,5)): scale 0.035 rounds to
# [1,3], scale 0.263 to [2,4].
check("worked example assumes TIER_OFFSET_RANGE", TIER_OFFSET_RANGE, ((1, 3), (3, 5)))
check("a median-risk market rests at the low end, [1,3]", tier_offsets(0.035), [1, 3])
check("a p99 market rests at [2,4]", tier_offsets(0.263), [2, 4])

# rungs step away from the touch, are tradeable, and never exceed three
for _side in ("bid", "ask"):
    for _t in _theos:
        for _bb, _ba in _books:
            for _u in (0.0, 0.5, 1.0):
                _l = decide_ladder(_side, _t, _bb, _ba, C, scale_pos=_u)
                if not _l:
                    continue
                if not ladder_prices_ordered(_side, _l):
                    FAILS.append(f"ladder unordered {_side} {_t} {_bb}/{_ba} u={_u}")
                for _r in _l:
                    if not (config.MIN_PRICE <= _r.price <= config.MAX_PRICE):
                        FAILS.append(f"rung out of range {_r.price} {_side} {_t}")
                if len(_l) > len(C):
                    FAILS.append(f"too many rungs {_side} {_t}")
                if len({_r.price for _r in _l}) != len(_l):
                    FAILS.append(f"duplicate rung prices {_side} {_t} {_bb}/{_ba} u={_u}")
check("all ladders ordered, in range, deduped, <=3 rungs", True, True)

# a deep rung that would go negative must be DROPPED, not clamped onto MIN_PRICE
_l = decide_ladder("bid", 0.03, 0.01, 0.05, C, scale_pos=1.0)
check("negative back rungs dropped, not clamped to MIN_PRICE",
      all(r.price >= config.MIN_PRICE for r in _l) and
      len({r.price for r in _l}) == len(_l), True)

# sizes follow the configured tiers, in order
_l = decide_ladder("bid", 0.42, 0.40, 0.44, C, scale_pos=1.0)
check("rung sizes follow the configured tiers", [r.count for r in _l], list(C))
check("rung indices are 0,1,2", [r.index for r in _l], [0, 1, 2])

# a raised edge bar suppresses the penny rung it used to take
_hi = decide("bid", 0.115, 0.10, 0.13, min_edge_micros=20_000)
_lo = decide("bid", 0.115, 0.10, 0.13, min_edge_micros=1_000)
check("a higher edge bar changes the rung-0 decision", _hi.kind != _lo.kind or _hi.price != _lo.price, True)


# --- rank-based min_edge, and surviving rung 0 (2026-08-26) --------------
#
# The defect this replaced: back rungs were priced BEHIND rung 0 and the ladder returned [] the
# moment rung 0 declined, so a sub-tick theo move cancelled three orders, two of which still had
# real edge. Measured on B7687, 39.4% of one-sided cycles. See docs/PIPELINE.md 6e.
from strategy.quoting import rung_min_edge_micros, _edge_micros as _em  # noqa: E402

print("\nmin_edge is monotone in rank -- a deeper rung never asks for LESS")
# The median market: the edge floor plus its tier offsets, one cent per offset tick.
from pricing import edge as _pedge  # noqa: E402
FLOOR = int(round(_pedge.EDGE_FLOOR * MICROS))
_offs = tier_offsets(0.035)
check("median market rank 0 is the floor", rung_min_edge_micros(0, FLOOR, 0.035), FLOOR)
check("median market rank 1 = floor + offset[0] cents",
      rung_min_edge_micros(1, FLOOR, 0.035), FLOOR + _offs[0] * 10_000)
check("median market rank 2 = floor + offset[1] cents",
      rung_min_edge_micros(2, FLOOR, 0.035), FLOOR + _offs[1] * 10_000)
# THE INVERSION CHECK, and the reason this is not two flat constants. A market whose calibrated
# bar is 2.27c would, under a fixed rank-1 bar below that, demand LESS at rank 1 than at rank 0.
for _bar in (FLOOR, 12_000, 22_693, 30_000, 100_000):
    for _u in (0.0, 0.035, 0.263, 1.0):
        _reqs = [rung_min_edge_micros(_r, _bar, _u) for _r in range(len(C))]
        if any(b <= a for a, b in zip(_reqs, _reqs[1:])):
            FAILS.append(f"min_edge not increasing: bar={_bar} u={_u} {_reqs}")
        if _reqs[0] != _bar:
            FAILS.append(f"rank 0 must be the calibrated bar: {_reqs[0]} != {_bar}")
check("min_edge strictly increases in rank at every bar and scale", True, True)
check("a 2.27c market does NOT invert at rank 1",
      rung_min_edge_micros(1, 22_693, 0.25) > rung_min_edge_micros(0, 22_693, 0.25), True)

print("\nrung 0 declining no longer empties the ladder")
# The exact measured case: theo 0.4908 against a 0.48/0.49 book -- eight hundredths of a cent
# through the touch. decide() gives up the side; the back rungs must not.
check("decide() still declines it", decide("ask", 0.4908, 0.48, 0.49).price, None)
_d = decide_ladder("ask", 0.4908, 0.48, 0.49, C, min_edge_micros=FLOOR, scale_pos=0.035)
check("  the ladder keeps its back rungs", len(_d), 2)
check("  and none of them is rung 0", [r.index for r in _d], [1, 2])
check("  every surviving rung clears its own bar",
      all(_em("ask", 0.4908, r.price) >= r.min_edge for r in _d), True)
check("  and sits ABOVE the touch it could not join",
      all(r.price > 0.49 for r in _d), True)
# The mirror, on the bid.
_d = decide_ladder("bid", 0.4602, 0.49, 0.51, C, min_edge_micros=FLOOR, scale_pos=0.035)
check("the bid mirror keeps its rungs too", len(_d), 2)
check("  below the touch it could not join", all(r.price < 0.49 for r in _d), True)

print("\ntwo anchors, and each keeps its own bar")
# Behind a live rung 0 the back rungs are TOUCH-anchored and keep the pre-2026-08-26 rule (any
# positive edge) -- deliberately, because touch-anchored prices are sticky and theo-anchored ones
# are not. Measured over the 2026-08-25 session: anchoring everything to the theo tripled the
# per-order replacement rate, 10.5% -> 34.7%. Only the FALLBACK rungs, the ones that exist because
# rung 0 declined, are theo-anchored and held to their rank's bar.
_touch = _fallback = 0
for _side in ("bid", "ask"):
    for _t in _theos:
        for _bb, _ba in _books:
            for _bar in (FLOOR, 22_693):
                for _r in decide_ladder(_side, _t, _bb, _ba, C,
                                        min_edge_micros=_bar, scale_pos=0.035):
                    if _r.min_edge is None:
                        FAILS.append(f"rung {_r.index} has no min_edge {_side} {_t}")
                        continue
                    if _r.index == 0:
                        continue          # decide()'s own, posts on positive edge
                    _e = _em(_side, _t, _r.price)
                    if _r.reason.startswith("tier"):
                        _touch += 1
                        if _e <= 0:
                            FAILS.append(f"touch rung {_r.index} non-positive {_side} {_t}")
                    else:
                        _fallback += 1
                        if _e < _r.min_edge:
                            FAILS.append(f"fallback rung {_r.index} under its bar {_side} {_t}")
check("every rung reports the bar it was placed against", True, True)
check("  touch-anchored back rungs keep the old positive-edge rule", _touch > 0, True)
check("  fallback rungs exist and clear their rank's bar", _fallback > 0, True)

print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
sys.exit(0 if not FAILS else 1)
