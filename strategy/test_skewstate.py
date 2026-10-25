"""Checks for `strategy.skewstate`.

Straight-line checks, a FAILS list, PASS/FAIL -- same as the other validators in this repo.

The two properties that matter are the ones the module exists for:

  * a single flickering quote must not move the served curve much, and
  * a bucket's history must survive the strike underneath it changing.

Both are checked against measured numbers, not asserted in prose.

    python -m strategy.test_skewstate
"""
import math
import sys
from harness import approx, assert_true, check

from pricing import bs, skew as skewmod
from strategy import skewstate

# The shared list, not a local one: `harness`'s helpers append to it, and so do the
# bespoke `.append(...)` calls below. Two lists would mean the local checks vote and
# the shared ones do not, and the gate would print PASS while a helper had failed.
from harness import FAILURES as FAILS

F0, W0 = 7700.0, 0.0055


def board(F, w_of, n=60, step=5.0, half_spread=0.10, size=10, bump=None):
    """Two-sided board off a per-strike vol function. `bump` corrupts one strike's call quote."""
    strikes = [F + step * i for i in range(-n, n + 1)]
    q = {}
    for k in strikes:
        w = w_of(k)
        for cp in ("C", "P"):
            mid = bs.bs_price(F, k, w, cp)
            if bump and cp == "C" and abs(k - bump[0]) < 1e-9:
                mid = max(mid + bump[1], 0.01)
            bid = mid - half_spread
            q[(cp, k)] = {"bp": bid if bid > 0 else 0.0, "ap": mid + half_spread,
                          "bs": size if bid > 0 else 0, "as": size}
    return q, strikes


def run_checks():
    flat = lambda k: W0                                            # noqa: E731

    print("\n1. Cadence -- it refits on the clock, not on every call")
    st = skewstate.SkewState("SPX")
    rc = st.recompute_s
    q, strikes = board(F0, flat)
    st.observe(q, strikes, F0, W0, now=100.0)
    check("first call fits", st.n_fits, 1)
    st.observe(q, strikes, F0, W0, now=100.0 + 0.5 * rc)
    check("a call inside the window does not refit", st.n_fits, 1)
    st.observe(q, strikes, F0, W0, now=100.0 + rc + 1.0)
    check("a call past the window refits", st.n_fits, 2)
    st.observe(q, strikes, F0, W0, now=100.0 + rc + 1.5, force=True)
    check("force refits regardless", st.n_fits, 3)

    print("\n2. A single sample is not an average -- MIN_SAMPLES fits before a curve is served")
    n_min = skewstate.MIN_SAMPLES
    assert_true("  (MIN_SAMPLES is more than one, or this section is vacuous)", n_min >= 2)
    st2 = skewstate.SkewState("SPX")
    gap = st2.recompute_s + 1.0
    for i in range(n_min - 1):
        c1 = st2.observe(q, strikes, F0, W0, now=i * gap)
    assert_true("MIN_SAMPLES - 1 observations serve a flat curve", c1.flat,
                f"reason={c1.reason} fits={st2.n_fits}")
    c2 = st2.observe(q, strikes, F0, W0, now=(n_min - 1) * gap)
    assert_true("MIN_SAMPLES observations serve a fitted curve", not c2.flat,
                f"reason={c2.reason} buckets={c2.n_buckets}")

    print("\n3. The window prunes")
    st3 = skewstate.SkewState("SPX", window_s=25.0, recompute_s=1.0)
    for i in range(8):
        st3.observe(q, strikes, F0, W0, now=float(i * 5))
    depth_before = max(len(v) for v in st3.queues.values())
    st3.observe(q, strikes, F0, W0, now=200.0)
    depth_after = max((len(v) for v in st3.queues.values()), default=0)
    assert_true("queue is bounded by the window",
                depth_before <= 8 and depth_after <= 2,
                f"before={depth_before} after={depth_after}")

    print("\n4. THE POINT -- one flickering quote barely moves the curve")
    # Build 6 clean observations, then one where a single near-money call is 3 dollars off.
    st4 = skewstate.SkewState("SPX", window_s=40.0, recompute_s=4.0, half_life_s=12.0)
    for i in range(6):
        st4.observe(q, strikes, F0, W0, now=float(i * 5))
    clean = st4.curve
    # Bump a strike the fitter actually SELECTS. Buckets are BUCKET_STEP sigma apart, wider than
    # the 5-point strike spacing here, so most listed strikes are never read and corrupting one
    # of those would prove nothing.
    base = skewmod.fit(q, strikes, F0, W0)
    k_bad = sorted(base.buckets, key=lambda b: abs(b["x"]))[3]["strike"]
    q_bad, _ = board(F0, flat, bump=(k_bad, 3.0))
    smoothed = st4.observe(q_bad, strikes, F0, W0, now=30.0)
    raw_bad = skewmod.fit(q_bad, strikes, F0, W0)

    def worst_move(a, b):
        return max(abs(a.ratio(k) - b.ratio(k)) for k in strikes)

    raw_move = worst_move(raw_bad, skewmod.fit(q, strikes, F0, W0))
    sm_move = worst_move(smoothed, clean)
    assert_true("the raw fit does move on the bad quote", raw_move > 1e-3,
                f"raw moved {raw_move:.5f}")
    assert_true("the smoothed curve moves far less", sm_move < raw_move / 2.0,
                f"smoothed {sm_move:.5f} vs raw {raw_move:.5f} "
                f"({sm_move / raw_move:.1%} of it)" if raw_move else "")

    print("\n5. Bucket history survives the strike underneath it changing")
    st5 = skewstate.SkewState("SPX", window_s=60.0, recompute_s=1.0)
    for i in range(4):
        st5.observe(q, strikes, F0, W0, now=float(i))
    labels_before = set(st5.queues)
    depth_before = {k: len(v) for k, v in st5.queues.items()}
    # Spot moves 40 points: the strike nearest each bucket changes, the labels do not.
    F1 = F0 + 40.0
    q1, s1 = board(F1, flat)
    st5.observe(q1, s1, F1, W0, now=5.0)
    grew = [k for k in labels_before if len(st5.queues.get(k, ())) > depth_before[k]]
    assert_true("labels persist across a spot move",
                len(set(st5.queues) & labels_before) >= 15,
                f"{len(set(st5.queues) & labels_before)} of {len(labels_before)}")
    assert_true("and the existing series were appended to, not restarted",
                len(grew) >= 15, f"{len(grew)} buckets grew")

    print("\n6. Failing soft")
    st6 = skewstate.SkewState("SPX")
    sparse = {("C", F0): {"bp": 1.0, "ap": 2.0, "bs": 5, "as": 5}}
    c = st6.observe(sparse, [F0], F0, W0, now=0.0)
    assert_true("a board it cannot fit yields a flat curve", c.flat, f"reason={c.reason}")
    approx("  and a flat curve prices as flat vol", c.p_above(F0 + 25.0),
           bs.p_above(F0, F0 + 25.0, W0), tol=0.0)
    d = st6.detail()
    assert_true("detail is journalable",
                {"skew_reason", "skew_flat", "n_fits", "queue_depth"} <= set(d))

    print("\n7. A real skew still comes through the smoother")
    def w_skewed(k):
        x = math.log(k / F0) / W0
        return W0 * max(0.35, 1.0 - 0.08 * x)

    st7 = skewstate.SkewState("SPX", recompute_s=1.0)
    qs, ss = board(F0, w_skewed)
    for i in range(6):
        st7.observe(qs, ss, F0, W0, now=float(i))
    cur = st7.curve
    assert_true("smoothed curve is not flat", not cur.flat, f"reason={cur.reason}")
    lo = cur.ratio(F0 * math.exp(-2.0 * W0))
    hi = cur.ratio(F0 * math.exp(+2.0 * W0))
    assert_true("and it still slopes the right way", lo > hi, f"lo={lo:.4f} hi={hi:.4f}")
    inner = [k for k in ss if abs(math.log(k / F0) / W0) <= 2.0]
    worst = max(abs(cur.w_of(k) - w_skewed(k)) / w_skewed(k) for k in inner)
    assert_true("recovers w within 3% inside +-2 sigma after smoothing", worst < 0.03,
                f"worst={worst:.4%}")


def check_live_curve_is_guarded():
    """The 2026-08-24 review finding: the LIVE curve got neither the projection nor the guard.

    `_rebuild` built its `SkewFit` without `strikes=`, so `_projection` short-circuited to empty
    and the corrected digital was served raw; and `skew.ladder_monotone` ran only inside
    `skew.fit`, never on the smoothed object that `validate_theo.resolve_and_price` installs into
    the live pricer. Both are checked here on the object the quote loop actually prices off.
    """
    print("\n8. THE SERVED CURVE CARRIES THE GRID AND PASSES THE GUARD")
    w_sk = lambda k: W0 * (1.0 + 0.30 * max(0.0, -math.log(k / F0) / W0) / 6.0)
    qs, ss = board(F0, w_sk)
    st = skewstate.SkewState("SPX")
    for i in range(skewstate.MIN_SAMPLES + 3):          # enough REFITS, not just calls
        st.observe(qs, ss, F0, W0, now=float(i) * (st.recompute_s + 1.0))
    cur = st.curve
    assert_true("served curve is fitted", cur.ok, f"reason={cur.reason}")
    check("  it carries the listed grid", len(cur.strikes), len(ss))
    ks, adj = cur._projection(W0)
    assert_true("  so the monotone projection is ARMED", bool(ks),
                f"{len(ks)} rungs inside +-{skewmod.PROJECTION_XMAX} sigma")
    assert_true("  the served ladder is non-increasing in K",
                skewmod.ladder_monotone(cur, ss))

    ladder = [cur.p_above(k) for k in ss]
    worst = min((b - a) for a, b in zip(ladder, ladder[1:]))     # <= 0 when non-increasing
    assert_true("  and every bracket it implies is non-negative", worst <= 1e-12,
                f"worst rung {worst:+.3e}")

    # A curve with no grid degrades to the OLD behaviour rather than raising: projection off,
    # guard trivially true. Nothing live takes this path, but the backtest and validators may.
    bare = st._rebuild(F0, W0, 5.0)
    check("no grid -> projection stays off (old behaviour, not an error)",
          bare._projection(W0)[0], ())


def main(argv=None):
    run_checks()
    check_live_curve_is_guarded()
    print(f"\n{'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
