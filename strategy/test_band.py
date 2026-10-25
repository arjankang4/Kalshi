"""Pin the quoting band: sigma-moneyness, hysteresis, and what it means on the real boards.

The headline assertion is the one that motivated the design -- that a strike COUNT is not a
constant width. It is checked here against the measured geometry of the two SPX boards rather
than asserted in a comment.

    python -m strategy.test_band
"""
import sys

import config
from harness import approx, assert_true, check, finish

from strategy import band

# Measured 2026-08-21 11:49 ET from data/raw/massive/20260821T154948Z (294 symbols) and
# data/raw/kalshi/20260821T154939Z. These are observations, not choices.
F = 7691.083357158548
W_4PM = 0.002444162443486358
SIGMA_PTS = W_4PM * F                      # 18.8 index points
BRACKET_WIDTH = 25.0                       # KXINX,  29 boundaries 7425..8125
RUNG_WIDTH = 5.0                           # KXINXU, 60 boundaries 7550..7845


def rung(k, ticker="R"):
    return {"ticker": ticker, "strike_type": "greater_or_equal", "floor_strike": k}


def bracket(lo, hi, ticker="B"):
    return {"ticker": ticker, "strike_type": "between", "floor_strike": lo, "cap_strike": hi}


print("one sigma, measured")
approx("w * F", SIGMA_PTS, 18.797, tol=0.01)

print("\nWHY NOT STRIKE COUNTS -- the measurement that decided this")
b3 = 3 * BRACKET_WIDTH
r3 = 3 * RUNG_WIDTH
approx("+-3 KXINX brackets in sigma", b3 / SIGMA_PTS, 3.99, tol=0.01)
approx("+-3 KXINXU rungs in sigma", r3 / SIGMA_PTS, 0.798, tol=0.01)
assert_true("the SAME strike count is 5x wider on one board than the other",
            round((b3 / SIGMA_PTS) / (r3 / SIGMA_PTS), 1) == 5.0,
            f"{(b3 / SIGMA_PTS) / (r3 / SIGMA_PTS):.1f}x")

print("\nthe band is a width in points, and it scales with vol and with the forward")
HALF = config.BAND_SIGMA * SIGMA_PTS          # the default half-width in points
approx("half-width at the default BAND_SIGMA", band.band_width(F, W_4PM), HALF, tol=1e-6)
approx("  at 1 sigma", band.band_width(F, W_4PM, sigma=1.0), SIGMA_PTS, tol=1e-6)
approx("  half the vol, half the band", band.band_width(F, W_4PM / 2), HALF / 2, tol=1e-6)
check("  no forward -> no band", band.band_width(None, W_4PM), None)
check("  no vol -> no band", band.band_width(F, None), None)
check("  a zero vol is not a zero-width band, it is no band", band.band_width(F, 0.0), None)

print("\nan hourly rung near settlement gets almost no band -- which is the point")
# `w` is total vol to THAT market's settlement, so a market minutes from close has almost none.
near = band.band_width(F, W_4PM * 0.1)
assert_true("a 1%-of-a-day horizon has a band a tenth as wide",
            near is not None and abs(near - 0.1 * HALF) < 1e-6, f"{near:.2f} pts")

print("\nmembership")
ok, d = band.in_band(rung(F + 10), F, W_4PM)
check("10 pts out (0.53 sigma) is in", ok, True)
check("  and it reports true sigma-moneyness, not position in the band", d["sigmas"], 0.532)
check("  with the band fraction alongside", d["band_frac"], round(10 / HALF, 3))
assert_true("60 pts out (3.2 sigma) is beyond the default band", 60 > HALF)
check("60 pts out (3.2 sigma) is out", band.in_band(rung(F + 60), F, W_4PM)[0], False)
check("  symmetric below the forward", band.in_band(rung(F - 60), F, W_4PM)[0], False)
check("  exactly on the boundary is IN", band.in_band(rung(F + HALF), F, W_4PM)[0], True)

print("\na bracket is measured from its MIDPOINT, not an edge")
# A `between` market has two boundaries and is not a point on the moneyness axis. The midpoint is
# where its digital is centred and where its Greeks peak.
approx("midpoint of [7675, 7700]", band.market_strike(bracket(7675, 7700)), 7687.5)
check("  the bracket containing F is in", band.in_band(bracket(7675, 7700), F, W_4PM)[0], True)
check("  a bracket 4 brackets away is out", band.in_band(bracket(7775, 7800), F, W_4PM)[0], False)
approx("a one-sided market uses its own boundary",
       band.market_strike(rung(7700)), 7700.0)
approx("  a `less` market uses cap_strike",
       band.market_strike({"strike_type": "less", "cap_strike": 7650}), 7650.0)

print("\nit FAILS CLOSED -- depth is the expensive choice, an unknown does not earn it")
check("no strike -> wing", band.in_band({"strike_type": "greater"}, F, W_4PM)[0], False)
check("unpriceable strike type -> wing",
      band.in_band({"strike_type": "functional"}, F, W_4PM)[0], False)
check("no forward -> wing", band.in_band(rung(F), None, W_4PM)[0], False)
check("no w -> wing", band.in_band(rung(F), F, None)[0], False)
check("  and the reason is named", band.in_band(rung(F), F, None)[1]["reason"], "no_width")

print("\nHYSTERESIS -- an incumbent leaves at a wider distance than it entered")
EXIT = config.BAND_SIGMA * config.BAND_EXIT_MULT   # exit boundary, in sigmas
assert_true("the exit boundary is wider than the entry boundary", EXIT > config.BAND_SIGMA + 0.2)
edge = F + (config.BAND_SIGMA + 0.1) * SIGMA_PTS   # just outside the entry boundary
check("a newcomer just outside stays out", band.in_band(rung(edge), F, W_4PM)[0], False)
check("  an incumbent at the same place is HELD",
      band.in_band(rung(edge), F, W_4PM, incumbent=True)[0], True)
far = F + (EXIT + 0.1) * SIGMA_PTS         # just beyond BAND_SIGMA * BAND_EXIT_MULT
check("  but not indefinitely -- past the exit boundary it goes",
      band.in_band(rung(far), F, W_4PM, incumbent=True)[0], False)
approx("  the exit boundary is BAND_SIGMA * BAND_EXIT_MULT",
       band.in_band(rung(far), F, W_4PM, incumbent=True)[1]["limit"], round(EXIT * SIGMA_PTS, 4),
       tol=1e-4)
check("  and just inside it an incumbent is still held",
      band.in_band(rung(F + (EXIT - 0.1) * SIGMA_PTS), F, W_4PM, incumbent=True)[0], True)

print("\nwhat membership decides")
check("in-band gets the full ladder", band.counts_for(True), config.QUOTE_TIER_COUNTS)
check("  a wing gets its FIRST rung", band.counts_for(False), (config.QUOTE_TIER_COUNTS[0],))
assert_true("  and a wing rung is the same size as a band rung -- the lot is the unit",
            band.counts_for(False)[0] == band.counts_for(True)[0])
check("in-band position limit is POSITION_LIMIT_LOTS_BAND lots", band.position_limit(True),
      float(config.POSITION_LIMIT_LOTS_BAND * config.LOT_SIZE))
check("  a wing is POSITION_LIMIT_LOTS_WING lots", band.position_limit(False),
      float(config.POSITION_LIMIT_LOTS_WING * config.LOT_SIZE))
check("  a rung is one lot", float(band.counts_for(False)[0]), float(config.LOT_SIZE))
assert_true("  so POSITION_LIMIT_LOTS_WING fills of a wing's single rung land exactly ON its limit",
            config.POSITION_LIMIT_LOTS_WING * float(band.counts_for(False)[0])
            == band.position_limit(False))
assert_true("  while a band market can take a full 3-rung fill and still have room",
            3 * float(config.QUOTE_TIER_COUNTS[0]) < band.position_limit(True))

print("\nclassify over a whole board")
# A realistic KXINXU event: 60 rungs, 5 wide, spanning the forward.
ms = [rung(7550 + 5 * i, f"KXINXU-T{7550 + 5 * i}") for i in range(60)]
em = {m["ticker"]: {"w": W_4PM} for m in ms}
cls, summ = band.classify(ms, F, em)
check("every market classified", summ["n"], 60)
expect = sum(1 for m in ms if abs(m["floor_strike"] - F) <= HALF + 1e-9)
check("  band count matches the width", summ["band"], expect)
check("  and the rest are wings", summ["wing"], 60 - expect)
approx_n = 2 * HALF / RUNG_WIDTH          # rungs a band of +-HALF points spans on a 5-wide ladder
assert_true("  ~2*HALF/5 rungs at BAND_SIGMA on a 5-wide ladder",
            abs(summ["band"] - approx_n) <= 2, f"{summ['band']} vs {approx_n:.1f}")
assert_true("  each carries its counts and its limit",
            all(c["counts"] and c["position_limit"] > 0 for c in cls.values()))

print("\n  a market with no w in em_map is a wing, not a crash")
em2 = dict(em); em2[ms[30]["ticker"]] = {}
cls2, summ2 = band.classify(ms, F, em2)
check("one fewer band market", summ2["band"], summ["band"] - 1)
check("  named as no_width", cls2[ms[30]["ticker"]]["reason"], "no_width")

print("\n  incumbency is honoured per market")
edge_t = next(t for t, c in cls.items() if not c["inband"])
cls3, summ3 = band.classify(ms, F, em, incumbents={edge_t})
assert_true("holding a boundary market keeps it in band",
            cls3[edge_t]["inband"] or cls3[edge_t]["distance"] > EXIT * SIGMA_PTS)

print("\nthe 4pm board at the same instant -- coarse brackets, few band markets")
bs = [bracket(7425 + 25 * i, 7450 + 25 * i, f"KXINX-B{i}") for i in range(28)]
emb = {m["ticker"]: {"w": W_4PM} for m in bs}
_, sb = band.classify(bs, F, emb)
assert_true("late-day 4pm board bands only the few brackets around F", 1 <= sb["band"] <= 5,
            f"{sb['band']} of 28")
# At the OPEN there is ~6.5h of variance left rather than ~4.2h, so w is larger and the band
# widens -- this is the behaviour that reproduces "about +-3 brackets" early in the session.
_, sb_open = band.classify(bs, F, {m["ticker"]: {"w": W_4PM * 1.9} for m in bs})
assert_true("  and widens at the open as w grows", sb_open["band"] > sb["band"],
            f"{sb['band']} -> {sb_open['band']}")

finish()
