"""Near-term vol-spike widening: a SYMMETRIC extra edge charge when the next minute or three is
forecast hotter than the next half hour.

WHAT IT IS. The signal is realized-vs-realized, on one clock:

    s = log(sigma_short / sigma_30m)

where both are `rvol` forecasts (`RealizedVolModel.predict_short` for 1m/3m, `predict("30m")`),
both annualized on the intraday variance clock and both on the 30s-sampling level. It is NOT
compared against implied vol: realized sits systematically below implied (the variance risk
premium, ~0.6x), so `log(sigma_short / sigma_impl)` would carry a ~-0.5 offset from premium alone
and read as a constant -- the flaw that made the lean uncentred. That ratio is computed and
journalled for comparison only (`s_impl`).

WHAT IT DOES. Pure function, mirroring `pricing.lean.edge_adjust`:

    extra_cents = k * clip(s - s0, 0, s_max)          on BOTH sides
    pull        = s_pull is not None and s >= s_pull

It never moves the theo and never moves one side without the other: a spike is a statement about
risk, not about direction. `pull` asks the caller to stop quoting the market this cycle.

GATED BY `config.VOLSPIKE_ENABLED` (False). With it off, `run_04.build_desired` never calls
`edge_adjust` with a signal, so every desired quote is bit-identical to a build without this
module (`strategy.test_volspike` asserts it). The signal itself is still computed and journalled
every cycle (`volspike.jsonl`) so evidence accumulates before anything is enabled.

THE PARAMETERS ARE UNCALIBRATED PLACEHOLDERS (config). The offline evidence -- does `s` predict
the next 1-3 minute index move beyond sigma_30m, and does it predict adverse selection on our
fills -- is Step 5 of the plan and has not been run. Do not enable on these values.
"""
import math

import config

MICROS_PER_CENT = 10_000


def signal(sigma_short, sigma_ref):
    """log(sigma_short / sigma_ref), or None when either is missing or non-positive."""
    try:
        a, b = float(sigma_short), float(sigma_ref)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        return None
    return math.log(a / b)


def edge_adjust(s, s0=None, k=None, s_max=None, s_pull=None):
    """(extra_micros, detail) for a signal `s`. Symmetric: the same charge on bid and ask.

    Returns 0 with a reason whenever there is nothing to say, so a caller can add it
    unconditionally. `extra_micros` is never negative -- a quiet next minute never TIGHTENS.
    """
    s0 = config.VOLSPIKE_S0 if s0 is None else s0
    k = config.VOLSPIKE_K_CENTS if k is None else k
    s_max = config.VOLSPIKE_S_MAX if s_max is None else s_max
    s_pull = config.VOLSPIKE_S_PULL if s_pull is None else s_pull
    d = {"spike_s": None, "spike_cents": 0.0, "spike_pull": False, "spike_reason": None}
    if s is None or not math.isfinite(s):
        d["spike_reason"] = "no_signal"
        return 0, d
    d["spike_s"] = round(float(s), 6)
    if s_pull is not None and s >= s_pull:
        d["spike_pull"] = True
    x = min(max(float(s) - s0, 0.0), s_max)
    if x <= 0.0:
        d["spike_reason"] = "pull" if d["spike_pull"] else "below_s0"
        return 0, d
    cents = k * x
    d["spike_cents"] = round(cents, 4)
    d["spike_reason"] = "pull" if d["spike_pull"] else "widen"
    return int(round(cents * MICROS_PER_CENT)), d
