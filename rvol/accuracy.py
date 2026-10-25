"""Is the realized-variance model right, and when it is wrong, is it the MODEL or the CALENDAR?

A FAILED FIRST ATTEMPT, RECORDED SO IT IS NOT REBUILT

The obvious design is to score each forecast twice -- once as an annualized sigma, once converted
back to a total vol over the window by `w = sigma_ann * sqrt(VT / 252)` -- and attribute the
difference to the calendar. It produces exactly zero, every time, at every horizon.

It has to. The target `y_60m` IS `annualized(RV / VT)` for that same window, so
`y * sqrt(VT/252)` reconstructs `sqrt(RV)` identically and the calendar cancels to the last bit.
Both sides were divided by the same VT, so multiplying one back can never reveal anything. This
was measured before it was understood: e_ann and e_tot agreed to four decimals and the "calendar
component" printed +0.0000 for all 78 blocks.

WHAT ACTUALLY SEPARATES THEM

The calendar does not enter as a scale factor on a single forecast -- it enters as the RELATIVE
weighting between the window a feature was measured over and the window the target covers. So a
calendar error is a systematic function of TIME OF DAY, and it shows up inside the annualized
error, not beside it. Two instruments, and the second needs no model at all:

  1. TIME-OF-DAY DECOMPOSITION of the annualized log error. Split `e = log(realized/predicted)`
     into a per-block mean (the clock-shaped part) and the residual within blocks. The clock-shaped
     part is what a mis-stated calendar produces; the within-block scatter is the model. Reported
     as a variance share so the two are comparable.

  2. A MODEL-FREE CALENDAR TEST. Measure each session's realized variance per block directly from
     the bars, normalise per session, average across sessions, and compare against the calendar's
     own `shares` over the same blocks. If the calendar is right this ratio is 1.0 everywhere. It
     involves no forecast, no regression and no fitted parameter, so it cannot be rescued or
     confounded by the model -- which makes it the cleaner of the two, and the one to believe when
     they disagree.

Instrument 1 answers "how much of the model's error is the calendar's fault". Instrument 2 answers
"is the calendar right", full stop. They are different questions and the second is prior.

WHAT THIS IS NOT. Not a significance test. Observations five minutes apart share most of their
windows, so residuals are strongly autocorrelated and any standard error here would be understated
by an unknown factor -- the same reason `rvol/fit` reports no inference. Point estimates plus a
split-half column, and the split-half is the only replication claim being made.
"""
import argparse
import datetime
import json
import math
import sys

import numpy as np

import config
from pricing import varcal
from rvol import dataset as D
from rvol import features as F
from rvol import model as M

TRADING_DAYS = F.TRADING_DAYS


def _variance_time_for(cal, d, minute, horizon_min):
    """Variance-time of the target window [t, t+h), in units where one RTH session == 1.0."""
    t0 = (datetime.datetime.combine(d, F.OPEN_T, tzinfo=config.ET)
          + datetime.timedelta(minutes=minute))
    t1 = t0 + datetime.timedelta(minutes=horizon_min)
    return F.variance_time(cal, t0, t1)


def score(underlying, horizon, split="test"):
    """Per-observation predicted vs realized, in both spaces. Returns a DataFrame.

    Only observations the model would actually have served are scored: `predict` returns None on
    any missing or non-positive feature, and those rows are dropped rather than filled, because a
    filled row measures the filler.
    """
    import pandas as pd
    mdl = M.load(underlying)
    cal = varcal.load(underlying)
    df = D.load(underlying)
    ycol = f"y_{horizon}"
    need = mdl.features + mdl.extra + [ycol]
    df = df.dropna(subset=need).copy()

    # Same temporal split as the fit, by session, so "test" here means what it means there.
    sessions = sorted(df["dt"].unique())
    n_train = int(round(len(sessions) * 0.75))
    train_days = set(sessions[:n_train])
    if split == "test":
        df = df[~df["dt"].isin(train_days)]
    elif split == "train":
        df = df[df["dt"].isin(train_days)]

    hmin = dict(D.HORIZONS)[horizon]
    preds = []
    for rec in df.to_dict("records"):
        p = mdl.predict(horizon, rec)
        preds.append(p if p is not None else np.nan)
    df["pred_ann"] = preds
    df = df[np.isfinite(df["pred_ann"]) & (df["pred_ann"] > 0)].copy()
    df["real_ann"] = df[ycol].astype("float64")

    # Variance-time of each target window, and the two spaces that follow from it.
    vt = [_variance_time_for(cal, datetime.date.fromisoformat(str(r)), int(m), hmin)
          for r, m in zip(df["dt"], df["obs_minute"])]
    df["vt"] = vt
    df = df[df["vt"] > 0].copy()
    # Total vol over the window, for reference only. NOTE these are NOT an independent scoring
    # space: `real_ann` is itself `annualized(RV / vt)`, so `w_real` reconstructs sqrt(RV) exactly
    # and log(w_real/w_pred) is identically log(real_ann/pred_ann). See the module docstring --
    # this was built as a second instrument, measured to be degenerate, and is kept only because a
    # pricer really does perform this multiplication and the magnitudes are useful to see.
    df["w_pred"] = df["pred_ann"] * np.sqrt(df["vt"] / TRADING_DAYS)
    df["w_real"] = df["real_ann"] * np.sqrt(df["vt"] / TRADING_DAYS)
    df["e_ann"] = np.log(df["real_ann"] / df["pred_ann"])
    return df


def _summary(e):
    e = np.asarray(e, dtype="float64")
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "mean": float(e.mean()), "median": float(np.median(e)),
            "sd": float(e.std(ddof=1)) if e.size > 1 else float("nan"),
            "mae": float(np.abs(e).mean()),
            "bias_pct": float(math.expm1(e.mean()) * 100.0)}


def by_minute(df, col="e_ann", width=30):
    """Mean log error by block of the session. The calendar-error detector.

    `width` in minutes. A calendar error is a systematic mean in `e_tot` that tracks the clock and
    is absent from `e_ann`; model noise is neither systematic nor clock-shaped.
    """
    out = []
    for lo in range(0, D.SESSION_MIN, width):
        sub = df[(df["obs_minute"] >= lo) & (df["obs_minute"] < lo + width)]
        if len(sub) == 0:
            continue
        lab = (datetime.datetime.combine(datetime.date(2000, 1, 1), F.OPEN_T)
               + datetime.timedelta(minutes=lo)).strftime("%H:%M")
        out.append({"block": lab, "n": len(sub), "e_ann": float(np.mean(sub["e_ann"]))})
    return out


def decompose(df, width=30):
    """Split the annualized log error into a clock-shaped part and a within-block part.

    The clock-shaped part is what a mis-stated variance calendar produces: a systematic mean that
    depends on WHERE IN THE SESSION the window sat. The within-block part is everything else, which
    is the model. Reported as variance shares so the two are on one scale.

    `between` is the variance of the per-block means weighted by block size -- the standard
    between-group / within-group split. It is descriptive, not a test: with ~750 heavily
    overlapping observations per block the block means are far better determined than the naive
    arithmetic suggests, but not by a factor anyone here can state.
    """
    e = df["e_ann"].to_numpy(dtype="float64")
    blocks = np.asarray((df["obs_minute"] // width).to_numpy(), dtype="int64")
    grand = float(e.mean())
    between = within = 0.0
    means = {}
    for b in np.unique(blocks):
        m = blocks == b
        mu = float(e[m].mean())
        means[int(b)] = mu
        between += m.sum() * (mu - grand) ** 2
        within += float(((e[m] - mu) ** 2).sum())
    n = len(e)
    tot = between + within
    mv = np.array(list(means.values()), dtype="float64")
    return {
        "n": n,
        "overall": _summary(e),
        "clock_shaped": {
            "variance_share": float(between / tot) if tot > 0 else float("nan"),
            "sd_of_block_means": float(mv.std(ddof=1)) if mv.size > 1 else float("nan"),
            "range": (float(mv.min()), float(mv.max())) if mv.size else None,
            "spread_pct": float((math.expm1(mv.max()) - math.expm1(mv.min())) * 100.0)
            if mv.size else float("nan"),
        },
        "model_residual": {
            "variance_share": float(within / tot) if tot > 0 else float("nan"),
            "sd": float(math.sqrt(within / max(n - len(means), 1))),
        },
    }


def calendar_test(underlying, width=30, scale_s=30, max_sessions=None):
    """MODEL-FREE: measured share of session variance per block vs the calendar's share.

    No forecast, no regression, no fitted parameter. Each session's realized variance is measured
    per block straight from the index bars, normalised over that session, and averaged across
    sessions with equal weight -- deliberately the same per-day-fraction estimator `pricing/varcal`
    fits with, so a disagreement is a disagreement about the data and not about the estimator.

    A ratio of 1.0 means the calendar states that block's share correctly. This is the instrument
    to believe when it and the model-based decomposition disagree, because nothing in it can be
    absorbed by a coefficient.
    """
    from rvol import data as rdata
    cal = varcal.load(underlying)
    idx = rdata.SOURCES[underlying]["index"]
    days = rdata.cached_days(idx)
    if max_sessions:
        days = days[-max_sessions:]
    nb = D.SESSION_MIN // width
    acc, used = [], 0
    for d in days:
        ts, px = rdata.load_day(idx, d)
        if ts is None or len(ts) < 1000:
            continue
        v = np.zeros(nb)
        for b in range(nb):
            t0 = (datetime.datetime.combine(d, F.OPEN_T, tzinfo=config.ET)
                  + datetime.timedelta(minutes=b * width))
            t1 = t0 + datetime.timedelta(minutes=width)
            rv, cov = F.averaged_rv(ts, px, F._ms(t0), F._ms(t1), scale_s=scale_s)
            v[b] = rv if np.isfinite(rv) else 0.0
        tot = v.sum()
        if tot <= 0:
            continue
        acc.append(v / tot)
        used += 1
    if not acc:
        return None
    measured = np.vstack(acc).mean(axis=0)
    # The calendar's own share over the same blocks. `shares` has 391 buckets (390 clock minutes
    # plus the terminal settlement convergence); the terminal one belongs to the final block.
    sh = np.asarray(cal.shares, dtype="float64")
    stated = np.array([sh[b * width:(b + 1) * width].sum() for b in range(nb)])
    stated[-1] += sh[D.SESSION_MIN:].sum()
    stated = stated / stated.sum()
    return {"sessions": used, "measured": measured, "stated": stated,
            "ratio": measured / np.where(stated > 0, stated, np.nan), "width": width}


def preopen_split(df):
    """The direct test of the overnight calendar: observations whose features reach before 09:30.

    `pf_3h` is the fraction of the 3-hour lookback's variance that came from the pre-open proxy,
    so pf_3h > 0 marks exactly the rows the gap calendar touches. If the gap shape is right, these
    rows should stop being systematically worse than the rest.
    """
    pre = df[df["pf_3h"] > 0.01]
    rth = df[df["pf_3h"] <= 0.01]
    return {"preopen": _summary(pre["e_ann"]), "preopen_n": int(len(pre)),
            "rth_only": _summary(rth["e_ann"]), "rth_only_n": int(len(rth))}


def split_half(df, col="e_ann"):
    """First vs second half of the test sessions. The only replication claim made here."""
    days = sorted(df["dt"].unique())
    mid = len(days) // 2
    a = df[df["dt"].isin(set(days[:mid]))]
    b = df[df["dt"].isin(set(days[mid:]))]
    return {"first_half": _summary(a[col]), "second_half": _summary(b[col])}


def report(underlying="SPX", split="test", write=True):
    rows = {}
    print(f"\n=== realized-vol accuracy: {underlying} ({split} split) ===")

    ct = calendar_test(underlying)
    if ct:
        r = ct["ratio"]
        labs = [(datetime.datetime.combine(datetime.date(2000, 1, 1), F.OPEN_T)
                 + datetime.timedelta(minutes=i * ct["width"])).strftime("%H:%M")
                for i in range(len(r))]
        print(f"\n--- MODEL-FREE CALENDAR TEST ({ct['sessions']} sessions, "
              f"{ct['width']}-minute blocks) ---")
        print("    measured share of session variance / share the calendar states. 1.00 = right.")
        order = np.argsort(-np.abs(np.log(r)))
        print(f"    {'block':>7} {'measured':>9} {'stated':>8} {'ratio':>7}")
        for i in list(order[:6]):
            print(f"    {labs[i]:>7} {ct['measured'][i]:9.4f} {ct['stated'][i]:8.4f} {r[i]:7.3f}")
        print(f"    worst |log ratio| {np.abs(np.log(r)).max():.3f}   "
              f"rms {np.sqrt((np.log(r) ** 2).mean()):.3f}   "
              f"median ratio {np.median(r):.3f}")
        rows["calendar_test"] = {"sessions": ct["sessions"], "width": ct["width"],
                                 "labels": labs,
                                 "measured": [float(x) for x in ct["measured"]],
                                 "stated": [float(x) for x in ct["stated"]],
                                 "ratio": [float(x) for x in r]}

    for horizon, _mins in D.HORIZONS:
        df = score(underlying, horizon, split=split)
        if len(df) == 0:
            print(f"\n{horizon}: no scorable observations")
            continue
        dec = decompose(df)
        pre = preopen_split(df)
        sh = split_half(df)
        rows[horizon] = {"n": int(len(df)), **dec, "preopen": pre, "split_half": sh,
                         "by_block": by_minute(df)}
        o, c, m = dec["overall"], dec["clock_shaped"], dec["model_residual"]
        print(f"\n{horizon}  ({len(df):,} observations)   "
              f"mean log e {o['mean']:+.4f} ({o['bias_pct']:+.2f}%)  sd {o['sd']:.4f}  "
              f"MAE {o['mae']:.4f}")
        print(f"  CALENDAR (clock-shaped): {c['variance_share']:6.2%} of error variance   "
              f"block means span {c['range'][0]:+.4f} to {c['range'][1]:+.4f} "
              f"({c['spread_pct']:.1f} pct pts of vol)")
        print(f"  MODEL    (within-block): {m['variance_share']:6.2%} of error variance   "
              f"sd {m['sd']:.4f}")
        print(f"  pre-open rows {pre['preopen_n']:,} mean {pre['preopen'].get('mean', float('nan')):+.4f}"
              f"   RTH-only {pre['rth_only_n']:,} mean {pre['rth_only'].get('mean', float('nan')):+.4f}")
        print(f"  split-half: {sh['first_half'].get('mean', float('nan')):+.4f} / "
              f"{sh['second_half'].get('mean', float('nan')):+.4f}")
        worst = sorted(rows[horizon]["by_block"], key=lambda b: -abs(b["e_ann"]))[:4]
        print(f"  worst blocks: " + "  ".join(
            f"{b['block']} {b['e_ann']:+.3f}" for b in worst))
    if write and rows:
        out = config.MODEL_DIR / "rvol" / f"{underlying}.accuracy.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"underlying": underlying, "split": split,
                                   "horizons": rows}, indent=1))
        print(f"\n[rvol.accuracy] wrote {out}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="model-vs-calendar accuracy decomposition")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--underlying", default="SPX")
    ap.add_argument("--split", default="test", choices=("test", "train", "all"))
    a = ap.parse_args(argv)
    report(a.underlying, split=a.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
