"""Offline checks for the realized-vol model and the lean. Must PASS before this touches quoting.

Same discipline as pricing/validate_hourly: identities first, then behaviour, then the numbers.
The strongest check here is the lean identity -- when the two vol views agree, the lean must be
EXACTLY the base offset and the quotes must be bit-for-bit what they were before this existed.

    python -m rvol.validate
"""
import math
import sys

import numpy as np

import config
from harness import truthy
from pricing import edge as pedge, lean, varcal
from rvol import features as F

# The shared list, not a local one: `harness`'s helpers append to it, and so do the
# bespoke `.append(...)` calls below. Two lists would mean the local checks vote and
# the shared ones do not, and the gate would print PASS while a helper had failed.
from harness import FAILURES


def ok(label, got, want, tol=None):
    if tol is None:
        good = got == want
    else:
        good = got is not None and want is not None and abs(got - want) <= tol
    print(f"  {'ok  ' if good else 'FAIL'} {label}: got={got!r} want={want!r}"
          + (f" tol={tol}" if tol else ""))
    if not good:
        FAILURES.append(label)
    return good


def check_lean_identity():
    """THE invariant: agreement charges exactly the base offset, on both sides, bit for bit."""
    print("1. LEAN IDENTITY -- equal vols change nothing")
    base = config.QUOTE_OFFSET
    for p in (0.05, 0.5, 0.95):
        for side in ("bid", "ask"):
            off, d = lean.side_offset(p, p, 0.12, 0.12, side)
            ok(f"sigma_rv == sigma_impl, p={p}, {side}", off, base)
            ok(f"  not marked applied ({side}, p={p})", d["lean_applied"], False)
    # and a disagreement smaller than one vol point must also leave it at base
    off, _ = lean.side_offset(0.5, 0.51, 0.12, 0.1205, "bid")
    ok("sub-one-vol-point disagreement leaves base", off, base)


def check_lean_scaling():
    """The specified shape: k vol points -> base * sqrt(1/k), and never wider than base.

    Run against an explicit VRP ratio of 1.0 so the scaling is tested independently of whether
    the premium has been calibrated -- an uncalibrated lean is correctly inert, and testing the
    inert path would prove nothing about the arithmetic.
    """
    print("\n2. LEAN SCALING")
    base = config.QUOTE_OFFSET
    _prev = config.LEAN_VRP_RATIO
    config.LEAN_VRP_RATIO = 1.0
    # p_rv > p_impl favours the bid
    off, d = lean.side_offset(0.50, 0.60, 0.10, 0.12, "bid")
    ok("2 vol points, favoured side = base*sqrt(1/2)", off, base * math.sqrt(0.5), 1e-12)
    ok("  k reported", round(d["lean_k"], 6), 2.0, 1e-9)
    off, _ = lean.side_offset(0.50, 0.60, 0.10, 0.12, "ask")
    ok("2 vol points, other side stays base", off, base)
    off, _ = lean.side_offset(0.50, 0.40, 0.10, 0.12, "ask")
    ok("p_rv < p_impl favours the ask", off, base * math.sqrt(0.5), 1e-12)

    worst = 0.0
    for k in np.linspace(0, 50, 501):
        s_rv = 0.10 + k * config.LEAN_VOL_SCALE
        for side in ("bid", "ask"):
            for p_rv in (0.4, 0.6):
                off, _ = lean.side_offset(0.5, p_rv, 0.10, s_rv, side)
                worst = max(worst, off)
    truthy("offset never exceeds base over k in [0,50]", worst <= base + 1e-15,
           f"max {worst:.6f} vs base {base}")
    config.LEAN_VRP_RATIO = _prev
    off, d = lean.side_offset(0.5, 0.6, 0.10, 0.12, "bid")
    ok("uncalibrated VRP leaves the offset at base", off, base)
    ok("  and says why", d["lean_reason"], "vrp_uncalibrated")


def check_edge_lean():
    """THE identity for the 2026-08-25 mechanism: agreement changes NOTHING, on both sides.

    The lean now moves the required EDGE, not the price. So the invariant is that when the
    realized chain sits exactly on its VRP-centred reference, the adjustment is exactly zero and
    the quote is bit-for-bit what the implied model alone would have produced.
    """
    print("\n3. EDGE LEAN -- agreement changes nothing; disagreement is asymmetric")
    for p in (0.05, 0.5, 0.95):
        for side in ("bid", "ask"):
            adj, d = lean.edge_adjust(p, p, side)
            ok(f"p_rv == p_ref, p={p}, {side} -> zero adjustment", adj, 0)
            ok(f"  and says why ({side}, p={p})", d["lean_reason"], "no_disagreement")

    print("  the worked example: 5 points above implied")
    bid, db = lean.edge_adjust(0.50, 0.55, "bid")
    ask, da = lean.edge_adjust(0.50, 0.55, "ask")
    ok("  bid is the aggressive side", db["lean_reason"], "aggressive")
    ok("  ask is the defensive side", da["lean_reason"], "defensive")
    ok("  bid gives up 0.25c", round(db["lean_cents"], 6), -0.25, 1e-9)
    ok("  ask demands 1.00c more", round(da["lean_cents"], 6), 1.0, 1e-9)
    BASE = 5_000        # the curve's 0.5c floor, in micro-dollars
    ok("  so off a 0.5c base the bid requires 0.25c", (BASE + bid) / 10_000, 0.25, 1e-9)
    ok("  and the ask requires 1.5c", (BASE + ask) / 10_000, 1.5, 1e-9)

    print("  it mirrors on the sign")
    lo, dl = lean.edge_adjust(0.50, 0.45, "ask")
    ok("p_rv BELOW reference makes the ASK aggressive", dl["lean_reason"], "aggressive")
    ok("  by the same amount", round(dl["lean_cents"], 6), -0.25, 1e-9)

    print("  and it is CLAMPED at +-10 points")
    at10, _ = lean.edge_adjust(0.50, 0.60, "ask")
    at50, _ = lean.edge_adjust(0.50, 1.00, "ask")
    ok("10 points is the maximum defensive adjustment", at10 / 10_000, 2.0, 1e-9)
    ok("  and 50 points is no worse", at50, at10)
    ag10, _ = lean.edge_adjust(0.50, 0.60, "bid")
    ok("  the aggressive leg maxes at -0.5c", ag10 / 10_000, -0.5, 1e-9)

    print("  the asymmetry is deliberate: 4x more willing to widen than to tighten")
    truthy("defensive leg is 4x the aggressive leg",
           abs(lean.LEAN_DEFENSIVE_CENTS / lean.LEAN_AGGRESSIVE_CENTS - 4.0) < 1e-12,
           f"{lean.LEAN_DEFENSIVE_CENTS} vs {lean.LEAN_AGGRESSIVE_CENTS}")

    print("  a missing chain is inert, never an error")
    for a, b in ((None, 0.5), (0.5, None), (None, None)):
        adj, d = lean.edge_adjust(a, b, "bid")
        ok(f"  ({a}, {b}) -> 0", adj, 0)
    ok("  and names the reason", lean.edge_adjust(None, 0.5, "bid")[1]["lean_reason"], "no_chain")

    print("  monotone: more disagreement is never less adjustment")
    prev_d, prev_a = -1e9, 1e9
    for pts in [x / 4 for x in range(0, 61)]:
        dfn, _ = lean.edge_adjust(0.50, 0.50 + pts / 100.0, "ask")
        agg, _ = lean.edge_adjust(0.50, 0.50 + pts / 100.0, "bid")
        if dfn < prev_d - 1e-9 or agg > prev_a + 1e-9:
            FAILURES.append("edge lean monotone")
            break
        prev_d, prev_a = dfn, agg
    else:
        print("  ok   monotone in |d| on both sides")


def check_clamp():
    """The safety property: a quote may never cross the realized-vol value."""
    print("\n3. THE CLAMP -- never quote through the realized-vol chain")
    rng = np.random.default_rng(11)
    bad = 0
    for _ in range(20000):
        p_rv = float(rng.uniform(0, 1))
        price = float(rng.uniform(-0.2, 1.2))
        side = "bid" if rng.random() < 0.5 else "ask"
        out, _c = lean.clamp_price(price, p_rv, side)
        if side == "bid" and out > p_rv + 1e-12:
            bad += 1
        if side == "ask" and out < p_rv - 1e-12:
            bad += 1
    ok("20000 randomised prices, none cross p_rv", bad, 0)
    ok("clamp is a no-op without a p_rv", lean.clamp_price(0.42, None, "bid"), (0.42, False))
    ok("clamp is a no-op on a None price", lean.clamp_price(None, 0.5, "bid"), (None, False))


def check_estimator():
    """Averaged RV recovers a known sigma and is invariant to the sub-grid offset."""
    print("\n4. ESTIMATOR -- synthetic GBM with a known sigma")
    rng = np.random.default_rng(7)
    n, sig_ann = 23400, 0.20
    sig_s = sig_ann / math.sqrt(252 * n)
    px = 100 * np.exp(np.cumsum(rng.normal(0, sig_s, n)))
    ts = np.arange(n, dtype="int64") * 1000 + 1_700_000_000_000

    rv, cov = F.averaged_rv(ts, px, ts[0], ts[-1] + 1, scale_s=30)
    est = F.annualized(rv)
    truthy("recovers sigma within 5%", abs(est / sig_ann - 1) < 0.05,
           f"{est:.4f} vs {sig_ann}")
    ok("full coverage on a dense series", round(cov, 2), 1.0, 0.02)

    plain, avg = [], []
    for off in range(0, 30, 3):
        r1, _ = F.averaged_rv(ts, px, ts[0] + off * 1000, ts[-1] + 1, scale_s=30, n_offsets=1)
        r2, _ = F.averaged_rv(ts, px, ts[0] + off * 1000, ts[-1] + 1, scale_s=30)
        plain.append(F.annualized(r1))
        avg.append(F.annualized(r2))
    sp, sa = max(plain) - min(plain), max(avg) - min(avg)
    truthy("sub-grid averaging cuts offset dependence >=5x", sa * 5 <= sp,
           f"plain spread {sp:.5f} vs averaged {sa:.5f}")

    print("\n5. ESTIMATOR -- degenerate inputs return NaN, never a number")
    for label, args in (("empty series", (np.array([], "int64"), np.array([]), 0, 1000)),
                        ("single point", (np.array([1], "int64"), np.array([1.0]), 0, 1000)),
                        ("zero-length window", (ts, px, ts[0], ts[0]))):
        r, _ = F.averaged_rv(*args)
        truthy(f"{label} -> NaN", math.isnan(r), f"got {r!r}")


def check_variance_clock():
    """Variance-time must be positive, monotone, and reproduce the calendar inside RTH."""
    print("\n6. VARIANCE CLOCK")
    import datetime
    cal = varcal.load("SPX")
    d = datetime.date(2026, 8, 20)

    def et(h, m):
        return datetime.datetime(d.year, d.month, d.day, h, m, tzinfo=config.ET)

    v_full = F.variance_time(cal, et(9, 30), et(16, 0))
    ok("full session == 1.0 of a session", round(v_full, 9), 1.0, 1e-6)
    a = F.variance_time(cal, et(10, 0), et(11, 0))
    b = F.variance_time(cal, et(10, 0), et(12, 0))
    truthy("monotone in the end time", b > a, f"{a:.5f} -> {b:.5f}")
    pre = F.variance_time(cal, et(6, 30), et(9, 30))
    truthy("a pre-open window carries positive variance-time", pre > 0, f"{pre:.5f}")
    truthy("pre-open share is bounded by the overnight share",
           pre <= cal.overnight_share + 1e-12,
           f"{pre:.5f} <= {cal.overnight_share:.5f}")
    straddle = F.variance_time(cal, et(8, 30), et(10, 30))
    truthy("a straddling window exceeds its RTH part alone",
           straddle > F.variance_time(cal, et(9, 30), et(10, 30)))

    # THE REGRESSION GUARD. Routing the pre-open through pricing/gapcal must not have moved a
    # single window that lies wholly inside RTH -- those never touch the gap calendar at all. If
    # this fails, the change leaked into the intraday clock and every fitted coefficient, every
    # backtest and the hourly pricer's variance ratio are all affected.
    worst = 0.0
    for h0, m0, h1, m1 in ((9, 30, 16, 0), (10, 0, 11, 0), (12, 15, 15, 45), (15, 0, 16, 0)):
        got = F.variance_time(cal, et(h0, m0), et(h1, m1))
        want = cal.phi(et(h1, m1)) - cal.phi(et(h0, m0))
        worst = max(worst, abs(got - want))
    ok("an all-RTH window is bit-for-bit the calendar alone", worst, 0.0, 1e-15)

    # Additivity across the open: the gap part and the RTH part must sum to the whole.
    lhs = F.variance_time(cal, et(6, 30), et(11, 0))
    rhs = F.variance_time(cal, et(6, 30), et(9, 30)) + F.variance_time(cal, et(9, 30), et(11, 0))
    ok("variance-time is additive across 09:30", lhs, rhs, 1e-12)


def check_gap_clock():
    """The overnight calendar is actually consulted, and it is not the old uniform rule."""
    print("\n6b. GAP CLOCK")
    import datetime
    from pricing import gapcal
    cal = varcal.load("SPX")
    d = datetime.date(2026, 8, 20)

    def et(h, m, day=d):
        return datetime.datetime(day.year, day.month, day.day, h, m, tzinfo=config.ET)

    try:
        g = F.gap_calendar("SPX")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return

    # The whole gap must be worth exactly `overnight_share` of a session -- that is the contract
    # between the two calendars, and the only place their units meet.
    whole = cal.overnight_share * g.share_between(et(16, 0), et(9, 30, d + datetime.timedelta(1)))
    ok("the whole gap is worth overnight_share", whole, cal.overnight_share, 1e-12)

    # And it must NOT reproduce the uniform rule this change replaced. 06:30-09:30 is 180 of the
    # gap's 1050 minutes; if the shape were flat this would be exactly 180/1050 of it. It is the
    # busiest stretch of the night, so it must be materially more.
    pre = F.variance_time(cal, et(6, 30), et(9, 30))
    uniform = cal.overnight_share * (180.0 / gapcal.GAP_MINUTES)
    truthy("06:30-09:30 is not the old uniform apportionment",
           abs(pre - uniform) > 0.05 * uniform,
           f"shaped {pre:.5f} vs uniform {uniform:.5f} "
           f"({pre / uniform:.2f}x)")
    truthy("the pre-open lookback is the BUSY end of the night", pre > uniform,
           f"{pre / uniform:.2f}x a flat stretch of the same length")

    # A 3h lookback taken at the open must be strictly inside the whole gap.
    truthy("a 3h pre-open lookback is a strict subset of the gap",
           0 < pre < cal.overnight_share, f"{pre:.5f} < {cal.overnight_share:.5f}")


def check_clock_fingerprint():
    """The coefficients must have been fitted against the calendars currently on disk.

    This is the check that makes the fingerprint mean something. Before it, the fingerprint was
    RECORDED and never COMPARED -- which buys nothing, because the failure it exists to catch is
    silent by construction: `rvol.validate` replays the metrics stored at fit time rather than
    recomputing them, so a calendar swap leaves every printed number unchanged while every
    coefficient is stale.

    `clock_sha256` covers all three things `features.variance_time` reads: the intraday shape, the
    overnight LEVEL (`overnight_share`), and the overnight shape. Hashing only `shares`, as the two
    older per-calendar fingerprints do, cannot see a level-only change -- and the level is what
    scales the entire pre-open half of the feature set.
    """
    print("\n9. CLOCK FINGERPRINT")
    from rvol import fit as rfit
    from rvol import model as rmodel
    for und in ("SPX", "NDX"):
        try:
            m = rmodel.load(und)
        except FileNotFoundError as e:
            print(f"  SKIP {und}: {e}")
            continue
        import json as _json
        art = _json.loads((config.MODEL_DIR / "rvol" / f"{und}.json").read_text())
        stored = art.get("clock_sha256")
        if stored is None:
            truthy(f"{und} artifact predates clock_sha256 -- refit to make it checkable",
                   False, "no clock_sha256 recorded")
            continue
        current = rfit.clock_fingerprint(und)
        match = stored == current
        note = (f"clock {stored[:16]}" if match else
                f"fitted against {stored[:16]}, disk now {current[:16]} -- REFIT: every "
                f"pre-open feature has moved under these betas")
        truthy(f"{und} coefficients match the calendars on disk", match, note)


def check_model():
    """Coefficients load, predict, and refuse to predict without inputs."""
    print("\n7. MODEL SERVING")
    from rvol import model as rmodel
    for und in ("SPX", "NDX"):
        try:
            m = rmodel.load(und)
        except FileNotFoundError as e:
            print(f"  SKIP {und}: {e}")
            continue
        feats = {"f_3h": 0.12, "f_1h": 0.11, "f_30m": 0.10, "f_15m": 0.10, "f_5m": 0.09,
                 "pf_3h": 0.0}
        p = m.predict("60m", feats)
        truthy(f"{und} 60m predicts a positive sigma", p is not None and p > 0, f"{p}")
        truthy(f"{und} prediction is in a sane range", 0.001 < p < 3.0, f"{p:.4f}")
        for bad in ({**feats, "f_1h": None}, {**feats, "f_1h": 0.0},
                    {**feats, "f_1h": float("nan")}):
            ok(f"{und} refuses a missing/invalid feature", m.predict("60m", bad), None)
        pr, h = m.predict_for_minutes(45, feats, blend=False)
        truthy(f"{und} nearest horizon for 45m is 30m or 60m", h in ("30m", "60m"), h)

        # BLENDED is the live default since 2026-08-25 -- an hourly market at 45 minutes gets half
        # the 30m model and half the 60m one. The property that matters is CONTINUITY: without it
        # a market drifting past the 45-minute midpoint jumps horizon, its sigma steps, and since
        # the lean now moves required edge that step would requote the market for no reason but a
        # labelling boundary.
        p30, _ = m.predict_for_minutes(30, feats)
        p60, _ = m.predict_for_minutes(60, feats)
        pb, hb = m.predict_for_minutes(45, feats)
        truthy(f"{und} 45m blends the two bracketing models", "/" in str(hb), hb)
        truthy(f"{und} and lands between them",
               min(p30, p60) <= pb <= max(p30, p60), f"{p30:.5f} <= {pb:.5f} <= {p60:.5f}")
        truthy(f"{und} at 45m the blend is even", abs(pb - 0.5 * (p30 + p60)) < 1e-12,
               f"{pb:.6f} vs {0.5 * (p30 + p60):.6f}")
        # Continuity across the old nearest-horizon boundary.
        a, _ = m.predict_for_minutes(44.9, feats)
        b, _ = m.predict_for_minutes(45.1, feats)
        truthy(f"{und} is continuous across the 45m midpoint", abs(a - b) < 1e-4,
               f"|{a:.6f} - {b:.6f}| = {abs(a - b):.2e}")
        # Outside the fitted range it clamps rather than extrapolating.
        pl, hl = m.predict_for_minutes(2, feats)
        ph, hh = m.predict_for_minutes(400, feats)
        truthy(f"{und} clamps below the shortest horizon", hl == "7.5m", hl)
        truthy(f"{und} clamps above the longest", hh == "60m", hh)
        # monotone: a uniformly higher vol history must predict a higher vol
        hi = {k: (v * 1.5 if k.startswith("f_") else v) for k, v in feats.items()}
        truthy(f"{und} monotone in its inputs", m.predict("60m", hi) > p)
        for hz in m.horizons():
            met = m.metrics(hz)
            print(f"    {und} {hz}: oosR2 {met['oos_r2']:.4f} (rw {met['oos_r2_rw']:.4f}) "
                  f"QLIKE {met['qlike']:.4f} (rw {met['qlike_rw']:.4f}) "
                  f"| beats baselines: {m.beats_baselines(hz)}")


def check_lean_safety_envelope():
    """The lean is ON since 2026-08-25. This asserts what makes that safe to do.

    It replaces a gate that asserted `LEAN_ENABLED is False`. That assertion was the right one
    while the lean moved the quoted PRICE; once it moves only the required EDGE, "is it off" is
    the wrong question and "how far can it move things, and can it move the wrong things at all"
    is the right one.
    """
    print("\n8. THE LEAN'S SAFETY ENVELOPE (it is ON)")
    truthy("LEAN_ENABLED is on", config.LEAN_ENABLED is True,
           "uncentred: it harvests the variance risk premium -- a short-vol posture, deliberately")

    # 1. It cannot touch the fair value. The only export that reaches the quote path returns an
    #    EDGE adjustment in micro-dollars; there is no theo-shaped return anywhere in it.
    import inspect
    src = inspect.getsource(lean.edge_adjust)
    truthy("edge_adjust returns an EDGE delta, never a price or a theo",
           "p_impl" in src and "return" in src)

    # 2. It is bounded, on both sides, at every disagreement including absurd ones.
    worst_def, worst_agg = 0.0, 0.0
    for pts in range(-200, 201):
        for side in ("bid", "ask"):
            adj, _ = lean.edge_adjust(0.50, 0.50 + pts / 100.0, side)
            worst_def = max(worst_def, adj / 10_000)
            worst_agg = min(worst_agg, adj / 10_000)
    ok("defensive leg never exceeds +2.0c, over +-200 points", round(worst_def, 9), 2.0, 1e-9)
    ok("aggressive leg never exceeds -0.5c", round(worst_agg, 9), -0.5, 1e-9)

    # 3. The required edge it produces can never go negative off the curve's floor. This is the
    #    property that stops the aggressive leg from ever asking us to quote through our own theo.
    floor = int(round(pedge.EDGE_FLOOR * 1e6))
    worst = min(floor + lean.edge_adjust(0.50, 0.50 + p / 100.0, s)[0]
                for p in range(-200, 201) for s in ("bid", "ask"))
    truthy("off the curve's 0.5c floor the leaned edge bottoms at exactly zero, never below",
           worst == 0, f"{worst / 10_000:.4f}c")

    # 4. The tail is bounded by the RISK plumbing, not by this module -- and it must be present.
    truthy("per-market inventory cap exists", config.POSITION_LIMIT_LOTS_BAND > 0
           and config.POSITION_LIMIT_LOTS_WING > 0,
           f"{config.POSITION_LIMIT_LOTS_BAND} fills in-band / "
           f"{config.POSITION_LIMIT_LOTS_WING} on the wings")
    truthy("per-market collateral cap exists",
           config.MAX_RESTING_ORDER_VALUE_PER_MARKET > 0,
           f"${config.MAX_RESTING_ORDER_VALUE_PER_MARKET:.0f}")
    truthy("and a book-wide one", config.MAX_RESTING_ORDER_VALUE > 0,
           f"${config.MAX_RESTING_ORDER_VALUE:,.0f}")


CANDIDATE_DIR = config.ROOT / "var" / "backtest" / "rvol"


def check_short_models():
    """The 3m / 1m models live under `short_models` and can NEVER reach the lean.

    THE IDENTITY. `models` in the candidate artifact must equal production's byte for byte, and
    every `predict_for_minutes` output -- value AND label, blended and nearest -- must be identical
    whether the artifact carries `short_models` or not. That is what keeps the lean, `backtest/rvin`,
    `edgecal/greeks_bt` and the C++ loader unchanged by construction.
    """
    print("\n10. SHORT MODELS (3m / 1m) -- separate key, never in the blend")
    import json as _json
    from pricing import volspike
    from rvol import model as rmodel
    truthy("VOLSPIKE_ENABLED is off", config.VOLSPIKE_ENABLED is False,
           "the near-term widening is journalled, not applied")
    worst = max(volspike.edge_adjust(x / 100.0)[0] for x in range(-300, 301))
    least = min(volspike.edge_adjust(x / 100.0)[0] for x in range(-300, 301))
    ok("volspike never exceeds K * S_MAX", worst,
       int(round(config.VOLSPIKE_K_CENTS * config.VOLSPIKE_S_MAX * 10_000)))
    ok("  and never tightens", least, 0)

    rng = np.random.default_rng(29)
    grid_min = [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 29.9, 30, 30.1, 44.9, 45, 45.1, 59, 60, 61, 120, 400]
    for und in ("SPX", "NDX"):
        prod_p = config.MODEL_DIR / "rvol" / f"{und}.json"
        cand_p = CANDIDATE_DIR / f"{und}.short.json"
        if not prod_p.exists() or not cand_p.exists():
            print(f"  SKIP {und}: {'no production model' if not prod_p.exists() else f'no candidate at {cand_p}'}")
            continue
        prod, cand = _json.loads(prod_p.read_text()), _json.loads(cand_p.read_text())
        truthy(f"{und} candidate `models` is bit-identical to production",
               cand["models"] == prod["models"])
        truthy(f"{und} production carries no short_models (not promoted)",
               "short_models" not in prod)
        mp, mc = rmodel.RealizedVolModel(prod), rmodel.RealizedVolModel(cand)
        stripped = rmodel.RealizedVolModel({k: v for k, v in cand.items() if k != "short_models"})
        ok(f"{und} horizons() unchanged", mc.horizons(), mp.horizons())
        diffs = 0
        n = 0
        for _ in range(200):
            base = float(np.exp(rng.normal(np.log(0.10), 0.5)))
            feats = {f"f_{h}": float(base * np.exp(rng.normal(0, 0.3)))
                     for h in ("3h", "1h", "30m", "15m", "5m", "1m")}
            feats["pf_3h"] = float(rng.uniform(0, 0.6))
            for mins in grid_min:
                for blend in (True, False):
                    a = mp.predict_for_minutes(mins, feats, blend=blend)
                    b = mc.predict_for_minutes(mins, feats, blend=blend)
                    c = stripped.predict_for_minutes(mins, feats, blend=blend)
                    n += 1
                    diffs += (a != b) + (a != c)
        ok(f"{und} every predict_for_minutes output identical ({n} calls, value and label)",
           diffs, 0)
        for mins in (0.5, 1, 3):
            _, lab = mc.predict_for_minutes(mins, feats)
            ok(f"{und} a {mins}m market still clamps to 7.5m, not a short model", lab, "7.5m")

        feats = {"f_3h": 0.12, "f_1h": 0.11, "f_30m": 0.10, "f_15m": 0.10, "f_5m": 0.09,
                 "f_1m": 0.09, "pf_3h": 0.0}
        ok(f"{und} production predict_short is None (no short_models)",
           mp.predict_short("1m", feats), None)
        ok(f"{und} short horizons", mc.short_horizons(), ["1m", "3m"])
        for h in mc.short_horizons():
            sm = mc.short_models[h]
            p = mc.predict_short(h, feats)
            truthy(f"{und} {h} predicts a sane sigma", p is not None and 0.001 < p < 3.0, f"{p}")
            # the served value is the conditional mean: exp(beta'x + s^2/2)
            row = [math.log(feats[c]) for c in sm["features"]] + [feats.get(c, 0.0) for c in sm["extra"]] + [1.0]
            want = math.exp(float(np.dot(row, sm["beta"])) + 0.5 * sm["sigma_resid"] ** 2)
            ok(f"{und} {h} is exp(beta'x + s^2/2)", p, want, 1e-12)
            for bad in ({**feats, "f_5m": None}, {**feats, "f_5m": 0.0},
                        {**feats, "f_5m": float("nan")}):
                ok(f"{und} {h} refuses a missing/invalid feature", mc.predict_short(h, bad), None)
            hi = {k: (v * 1.5 if k.startswith("f_") else v) for k, v in feats.items()}
            truthy(f"{und} {h} monotone in its inputs", mc.predict_short(h, hi) > p)
            truthy(f"{und} {h} target rescaled to the 30s level",
                   sm["target_scale_s"] == 10 and 0.85 < sm["epps_rescale"] < 1.0,
                   f"{sm['target_scale_s']}s, rescale {sm['epps_rescale']:.4f}")
            beats_med = (sm["oos_r2"] > max(sm["oos_r2_rw"], sm["oos_r2_cal"])
                         and sm["qlike"] < min(sm["qlike_rw"], sm["qlike_cal"]))
            beats_srv = (sm["oos_r2"] > max(sm["oos_r2_rw"], sm["oos_r2_cal"])
                         and sm["qlike_served"] < min(sm["qlike_rw"], sm["qlike_cal"]))
            print(f"    {und} {h}: oosR2 {sm['oos_r2']:.4f} (rw {sm['oos_r2_rw']:.4f}, "
                  f"3h {sm['oos_r2_cal']:.4f}) QLIKE {sm['qlike']:.4f} served "
                  f"{sm['qlike_served']:.4f} (rw {sm['qlike_rw']:.4f}, 3h {sm['qlike_cal']:.4f})"
                  f" | beats baselines: median {beats_med}, served {beats_srv}"
                  f" [informational -- a candidate, not a gate]")


def main(argv=None):
    FAILURES.clear()
    check_lean_identity()
    check_lean_scaling()
    check_edge_lean()
    check_clamp()
    check_estimator()
    check_variance_clock()
    check_gap_clock()
    check_model()
    check_lean_safety_envelope()
    check_clock_fingerprint()
    check_short_models()
    print("\nPASS" if not FAILURES else f"\nFAIL ({len(FAILURES)}): {FAILURES}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
