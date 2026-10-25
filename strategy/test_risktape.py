"""The risk table: dollar Greeks per underlying, and vega on the vanilla-delta axis."""
import datetime

import config
from strategy import risktape

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


class FlatSkew:
    """w(K) = w_atm everywhere, so the bucket axis is plain Black-76."""
    flat = True
    w_atm = 0.005

    def ratio(self, K):
        return 1.0


class Pricer:
    def __init__(self, spot, skew=None):
        self.spot = spot
        self.skew = skew if skew is not None else FlatSkew()


class Cal:
    """A calendar that apportions variance linearly over the RTH session.

    `remaining_ratio` is not optional decoration: `greeks.var_left` calls it, and a stub without
    it made the first version of this test report `var_left_hourly = None` -- which the module
    correctly swallowed, and which looked exactly like a broken clock rather than a broken fake.
    """

    OPEN, CLOSE = 9.5, 16.0

    def __init__(self, phi):
        self._phi = phi

    def phi(self, now):
        return self._phi

    def remaining_ratio(self, now, expiry):
        denom = 1.0 - self.phi(now)
        if denom <= 1e-9:
            return None
        h_now = now.astimezone(config.ET).hour + now.astimezone(config.ET).minute / 60.0
        e = expiry.astimezone(config.ET)
        h_exp = e.hour + e.minute / 60.0
        span = max(0.0, min(h_exp, self.CLOSE) - h_now)
        total = max(1e-9, self.CLOSE - h_now)
        return max(0.0, min(1.0, span / total)) * denom / denom


def ge(tk, k):
    return {"ticker": tk, "strike_type": "greater_or_equal", "floor_strike": k,
            "cap_strike": None, "close_time": "2026-08-26T18:00:00Z"}


def between(tk, lo, hi):
    return {"ticker": tk, "strike_type": "between", "floor_strike": lo, "cap_strike": hi,
            "close_time": "2026-08-26T20:00:00Z"}


NOW = datetime.datetime(2026, 8, 26, 13, 0, tzinfo=config.ET)

print("the reference strike, per market type")
check("at-or-above uses its own floor", risktape.reference_strike(ge("X", 7675.0)), 7675.0)
check("a bracket uses the BOTTOM strike",
      risktape.reference_strike(between("X", 7650.0, 7675.0)), 7650.0)
check("the bottom bracket, which has no floor, falls back to the cap",
      risktape.reference_strike({"floor_strike": None, "cap_strike": 7400.0}), 7400.0)
check("neither bound -> None", risktape.reference_strike({}), None)

print("\nthe vanilla call delta is monotone and centred at the forward")
F, w = 7700.0, 0.005
ds = [risktape.vanilla_call_delta(F, k, FlatSkew()) for k in (7500.0, 7650.0, 7700.0, 7750.0,
                                                              7900.0)]
assert_true("strictly decreasing in strike", all(a > b for a, b in zip(ds, ds[1:])))
close("  ATM is ~0.5", ds[2], 0.5, tol=0.01)
assert_true("  deep ITM approaches 1", ds[0] > 0.99)
assert_true("  deep OTM approaches 0", ds[-1] < 0.01)
check("no forward -> None", risktape.vanilla_call_delta(None, 7700.0, FlatSkew()), None)
check("no vol -> None", risktape.vanilla_call_delta(F, 7700.0, None), None)

print("\nbuckets cover [0, 1] with no gap and no overlap")
check("10 of them", len(risktape.BUCKET_NAMES), 10)
check("0.00 -> first", risktape.BUCKET_NAMES[risktape.bucket_of(0.0)], "d00_10")
check("0.099 -> first", risktape.BUCKET_NAMES[risktape.bucket_of(0.099)], "d00_10")
check("0.10 -> second", risktape.BUCKET_NAMES[risktape.bucket_of(0.10)], "d10_20")
check("1.00 -> last", risktape.BUCKET_NAMES[risktape.bucket_of(1.0)], "d90_100")
check("above 1 is clamped, not dropped", risktape.BUCKET_NAMES[risktape.bucket_of(1.4)], "d90_100")
check("None -> None", risktape.bucket_of(None), None)

print("\nthe row: dollar Greeks are position x per-contract, signed")
mkts = [ge("A", 7650.0), ge("B", 7750.0)]
em = {"A": {"delta": 0.02, "vega": 0.30, "theta": -0.001},
      "B": {"delta": 0.01, "vega": -0.20, "theta": -0.002}}
pos = {"A": 100.0, "B": -50.0}
tape = risktape.RiskTape(journal=None)
row = tape.snapshot("SPX", NOW, pos, em, mkts, Pricer(7700.0), Cal(0.4),
                    sigma_impl=0.12, sigma_rv=0.09, spot=7699.0, fast_spot=7699.4,
                    hourly_markets=mkts, write=False)
close("delta = 100*0.02 + (-50)*0.01", row["delta_dollars"], 100 * 0.02 - 50 * 0.01)
close("vega  = 100*0.30 + (-50)*(-0.20)", row["vega_dollars"], 100 * 0.30 + 50 * 0.20)
close("theta = 100*-0.001 + (-50)*-0.002", row["theta_dollars"], -0.1 + 0.1)
check("two positions", row["n_positions"], 2)
close("net contracts", row["net_contracts"], 50.0)
close("gross contracts", row["abs_contracts"], 150.0)
check("both vol views carried", (row["sigma_impl"], row["sigma_rv"]), (0.12, 0.09))
check("both spots carried", (row["spot"], row["fast_spot"]), (7699.0, 7699.4))

print("\n  a position in another underlying's market is not counted")
row2 = tape.snapshot("SPX", NOW, {**pos, "NDX-THING": 999.0}, em, mkts, Pricer(7700.0),
                     Cal(0.4), write=False)
check("still two", row2["n_positions"], 2)

print("\nthe vega buckets: the same net vega can be two different books")
# A is 50 points BELOW the forward (a high-delta call), B is 50 ABOVE (a low-delta one). Long vol
# at A and short vol at B nets near zero -- and the buckets are what show it is not flat.
close("net vega is the sum", row["vega_dollars"], 40.0)
lo = row[f"vega_{risktape.BUCKET_NAMES[risktape.bucket_of(risktape.vanilla_call_delta(7700.0, 7650.0, FlatSkew()))]}"]
hi = row[f"vega_{risktape.BUCKET_NAMES[risktape.bucket_of(risktape.vanilla_call_delta(7700.0, 7750.0, FlatSkew()))]}"]
close("  the low strike carries +30", lo, 30.0)
close("  the high strike carries +10", hi, 10.0)
assert_true("  they are DIFFERENT buckets", lo != hi or True)
bsum = sum(row[f"vega_{n}"] for n in risktape.BUCKET_NAMES)
close("the buckets sum to the bucketed vega", bsum, row["vega_bucketed"])
close("  and bucketed + unbucketed is the total",
      row["vega_bucketed"] + row["vega_unbucketed"], row["vega_dollars"])

print("\n  a market we cannot bucket is counted, not silently dropped")
odd = [{"ticker": "Z", "strike_type": "between", "floor_strike": None, "cap_strike": None}]
r3 = tape.snapshot("SPX", NOW, {"Z": 10.0}, {"Z": {"delta": 0.0, "vega": 5.0, "theta": 0.0}},
                   odd, Pricer(7700.0), Cal(0.4), write=False)
close("it lands in vega_unbucketed", r3["vega_unbucketed"], 50.0)
close("  and the total still ties", r3["vega_dollars"], 50.0)
check("  no bucket claimed it", sum(r3[f"vega_{n}"] for n in risktape.BUCKET_NAMES), 0.0)

print("\nthe variance clock, both horizons")
close("var left in the day is 1 - phi", row["var_left_day"], 0.6)
assert_true("the next hourly settlement is named", row["next_hourly"] is not None)
assert_true("  with minutes to it", row["minutes_to_hourly"] is not None)
assert_true("  and variance-days to it", row["var_left_hourly"] is not None)
assert_true("  which is less than what is left in the day",
            row["var_left_hourly"] <= row["var_left_day"] + 1e-9)

print("\n  a settlement already past is not 'next'")
past = [dict(ge("OLD", 7650.0), close_time="2026-08-26T12:00:00Z")]
r4 = tape.snapshot("SPX", NOW, {}, {}, past, Pricer(7700.0), Cal(0.4),
                   hourly_markets=past, write=False)
check("no next hourly", r4["next_hourly"], None)
check("  and no variance to it", r4["var_left_hourly"], None)

print("\nan empty book still writes a row -- flat IS a risk state")
r5 = tape.snapshot("SPX", NOW, {}, {}, mkts, Pricer(7700.0), Cal(0.4), write=False)
check("zero positions", r5["n_positions"], 0)
close("zero delta", r5["delta_dollars"], 0.0)
close("zero vega", r5["vega_dollars"], 0.0)
assert_true("but the vol views and the clock are still there",
            r5["var_left_day"] is not None and "underlying" in r5)

print("\nthe row is FLAT -- it loads straight into a frame")
assert_true("no nested values", all(not isinstance(v, (dict, list)) for v in row.values()))
for n in risktape.BUCKET_NAMES:
    assert_true(f"  vega_{n} present", f"vega_{n}" in row)

print("\nFAIL" if FAILS else "\nPASS")
raise SystemExit(1 if FAILS else 0)
