"""Pin the realized-vol shadow chain: WHICH markets it speaks about, at WHAT horizon, and
against WHAT reference.

Three rules, all 2026-08-25, and each was a way of being silently wrong:

  1. The 4pm board gets NO lean. Its digital is priced off options expiring at that instant, and
     the model's longest fitted horizon is 60 minutes against a 390-minute session.
  2. Every hourly market is priced to ITS OWN settlement. The chain used to price everything to
     16:00, which overstates a 12:00 rung's vol by ~4.4x (measured below).
  3. The comparison is against RAW implied -- no premium centring. That is a deliberate operator
     choice to harvest the variance risk premium, taken with the -9.33-points-from-premium-alone
     measurement in hand. It is a short-vol posture, not a forecasting claim.

    python -m strategy.test_leanchain
"""
import datetime
import math
import sys

import config
from harness import approx, assert_true, check, finish

from strategy import leanchain

ET = config.ET
NOW = datetime.datetime(2026, 8, 21, 11, 49, tzinfo=ET)
SIGMA_RV, SIGMA_IMPL, FWD = 0.10, 0.15, 7691.0
W_IMPL_4PM = 0.00244


def mkt(ticker, hour, k=7700):
    close = datetime.datetime(2026, 8, 21, hour, 0, tzinfo=ET).astimezone(datetime.timezone.utc)
    return {"ticker": ticker, "strike_type": "greater_or_equal", "floor_strike": k,
            "cap_strike": None, "close_time": close.isoformat().replace("+00:00", "Z")}


lc = leanchain.LeanChain(config.OPTION_UNDERLYING)
if not lc.ok:
    print(f"[skip] LeanChain unavailable ({lc.reason})")
    raise SystemExit(0)
cal = lc.cal

# `sigmas()` needs a backfilled index ring buffer -- a network call -- and is not what is under
# test. The estimator produces one session sigma; the question is what the chain does with it.
lc.sigmas = lambda now, quotes, strikes: (
    SIGMA_RV, SIGMA_IMPL,
    {"lean_reason": None, "variance_left": 1.0 - cal.phi(now),
     "w_impl": W_IMPL_4PM, "sigma_impl": SIGMA_IMPL, "forward": FWD})
lc.rv.features = lambda now=None, force=False: {"f_1h": 0.1, "pf_3h": 0.0}
lc.model.predict_for_minutes = lambda minutes, feats, blend=True: (SIGMA_RV, f"stub@{minutes:.0f}")


def run(ms, enabled=True, skew=None):
    return lc.chain(ms, {}, [], NOW, FWD, enabled=enabled, skew_fit=skew)


HOURLY = [mkt(f"KXINXU-H{h}", h) for h in (12, 13, 14, 15)]
FOURPM = mkt("KXINX-H16", 16)

print("RULE 1 -- the 4pm board is never leaned")
out = run(HOURLY + [FOURPM])
check("the 4pm market is refused", out["KXINX-H16"]["p_rv"], None)
check("  and it says why, rather than failing quietly",
      out["KXINX-H16"]["reason"], "4pm_board_implied_only")
check("  no shadow value either -- there is nothing to record",
      out["KXINX-H16"]["p_rv_shadow"], None)
assert_true("  while the hourly markets ARE leaned",
            all(out[m["ticker"]]["p_rv"] is not None for m in HOURLY))

print("\nRULE 2 -- each market is priced to its OWN settlement")
ws = [out[f"KXINXU-H{h}"]["w_rv"] for h in (12, 13, 14, 15)]
assert_true("w rises monotonically with the horizon", all(a < b for a, b in zip(ws, ws[1:])),
            str([round(x, 6) for x in ws]))
r12 = cal.remaining_ratio(NOW, datetime.datetime(2026, 8, 21, 12, 0, tzinfo=ET))
approx("the 12:00 w is w_impl_4pm * sqrt(ratio) * (sigma_rv/sigma_impl)",
       ws[0], W_IMPL_4PM * math.sqrt(r12) * (SIGMA_RV / SIGMA_IMPL), tol=1e-15)
# THE SIZE OF THE BUG THIS REPLACED: pricing a 12:00 rung at the 16:00 horizon.
overstated = 1.0 / math.sqrt(r12)
assert_true("  pricing it at 16:00 would have overstated its vol several-fold",
            overstated > 2.0, f"{overstated:.1f}x")

print("\nRULE 3 -- the reference is RAW implied; there is no premium centring")
from pricing.lean import edge_adjust
row = out["KXINXU-H14"]
p_ref, p_rv = row["p_ref"], row["p_rv"]
assert_true("the reference is the implied chain itself", p_ref is not None)
adj, det = edge_adjust(p_ref, p_rv, "bid")

# WHAT THIS LEAN IS, pinned so it cannot be misread later. sigma_rv/sigma_impl is 0.667 here with
# no information content whatsoever -- it is pure variance risk premium -- and the lean acts on it.
assert_true("a 0.67 realized/implied ratio alone produces a real disagreement",
            abs(det["lean_d"]) > 0.5, f"d = {det['lean_d']:.2f} points")
assert_true("  and the lean acts on it", adj != 0, f"{det['lean_cents']:+.3f}c")
check("  which is a SHORT-VOL posture, not a forecasting claim",
      det["lean_reason"] in ("aggressive", "defensive"), True)

print("  the sign is right: realized BELOW implied sells the upside, buys the downside")
# p_above(K) for K above the forward FALLS when vol falls -> d < 0 -> the ask is aggressive.
# +-1 sigma of the 14:00 market's OWN total vol, not the 4pm board's -- far-out strikes have
# probabilities pinned at 0 or 1 where no vol change can move them.
_sig_pts = out["KXINXU-H14"]["w_impl"] * FWD
up = mkt("UP", 14, k=round(FWD + _sig_pts))
dn = mkt("DN", 14, k=round(FWD - _sig_pts))
o2 = run([up, dn])
_, d_up = edge_adjust(o2["UP"]["p_ref"], o2["UP"]["p_rv"], "ask")
_, d_dn = edge_adjust(o2["DN"]["p_ref"], o2["DN"]["p_rv"], "bid")
check("above the forward the ASK is the aggressive side", d_up["lean_reason"], "aggressive")
check("  below it the BID is", d_dn["lean_reason"], "aggressive")
assert_true("  i.e. sell upside, buy downside -- the premium harvest",
            d_up["lean_d"] < 0 < d_dn["lean_d"],
            f"up d={d_up['lean_d']:.2f}, down d={d_dn['lean_d']:.2f}")

print("\nthe chain goes through the SMILE when one is supplied")
class _Smile:
    ok, flat = True, False
    def __init__(self): self.seen = []
    def p_above_at(self, k, w_atm=None):
        self.seen.append((k, w_atm))
        return 0.42
sm = _Smile()
out_s = run(HOURLY, skew=sm)
assert_true("the smile was consulted", len(sm.seen) > 0, f"{len(sm.seen)} calls")
check("  and its value is what the chain reports", out_s["KXINXU-H14"]["p_rv_shadow"], 0.42)
assert_true("  at the market's own w, not the 4pm one",
            any(abs(w - out["KXINXU-H14"]["w_rv"]) < 1e-12 for _k, w in sm.seen))

print("\n  a FLAT smile falls back to Black-Scholes rather than pricing every strike at ATM")
class _Flat(_Smile):
    flat = True
fl = _Flat()
out_f = run(HOURLY, skew=fl)
check("the flat curve is not consulted", len(fl.seen), 0)
assert_true("  and a probability still comes out",
            out_f["KXINXU-H14"]["p_rv_shadow"] is not None)

print("\ndisabled means inert, not absent")
off = run(HOURLY, enabled=False)
check("p_rv is None", off["KXINXU-H14"]["p_rv"], None)
assert_true("  shadow still recorded", off["KXINXU-H14"]["p_rv_shadow"] is not None)
check("  and the meta says so", off["_meta"]["enabled"], False)

finish()
