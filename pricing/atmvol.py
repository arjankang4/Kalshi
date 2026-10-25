"""One option-quote snapshot -> the forward and the ATM total vol to the 4pm expiry.

This is the measurement half of the hourly model. pricing/varcal supplies the shape of the day;
this supplies the level, read off the only instrument that actually trades: the 0DTE SPXW/NDXP
board expiring at 16:00.

Two quantities come out, in this order because the second depends on the first:

  * **The forward `F`**, from put-call parity. At 0DTE with no financing, `C - P = S - K`, so
    every strike implies `S = K + C - P` and the strip implies a whole distribution of them.
  * **The ATM total vol `w = sigma*sqrt(T)`** to 16:00, by inverting Black-Scholes on the
    near-money calls and puts against that forward.

`I:SPX` became entitled on 2026-08-21, so `F` can now be checked against -- and supplied from --
a real index level (massive/index.py); `fit(..., forward=)` takes it. What has NOT changed is why
SPY was never the answer. SPY is a different instrument: it accrues undistributed
dividends between quarterly ex-dates and bleeds an expense ratio, so `SPY x 10` sawtooths against
SPX rather than tracking it. Measured 17-25 SPX points under, about one full KXINX bracket; the
same effect on the entitled `I:NDX / QQQ` pair drifts 41.02-41.21 over two years, which is a 9.8
point average and 86 point worst-case error from a fixed multiplier. Parity is not a fallback
here, it is the right answer.
"""
from statistics import median

import config
from pricing import bs
from pricing.theo import MISSING, quote_value, two_sided as _two_sided

# Strikes averaged into the implied forward. Illustrative value; the live value is not published.
PARITY_STRIKES = 20

# Strikes medianed into ATM vol. Illustrative value; the live value is not published.
ATM_VOL_STRIKES = 4

# GATE: refuse the fit if call and put vols disagree by more than this (relative). Illustrative
# value; the live value is not published.
MAX_CP_VOL_SPREAD = 0.2
# GATE: refuse the fit if near-money vols disperse more than this relative to w. Illustrative value;
# the live value is not published.
MAX_VOL_DISPERSION = 0.15


class AtmFit:
    """Result of one fit. `ok` gates quoting; `reason` says why not."""

    __slots__ = ("forward", "w", "reason", "anchor_strike", "naive_forward",
                 "n_parity", "n_vol", "cp_spread", "dispersion", "w_anchor", "strikes_used",
                 "forward_source", "parity_forward", "index_minus_parity")

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    @property
    def ok(self):
        return self.forward is not None and self.w is not None and self.reason is None

    def as_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __repr__(self):
        if not self.ok:
            return f"<AtmFit unusable: {self.reason}>"
        return (f"<AtmFit F={self.forward:.2f} w={self.w:.5f} "
                f"({self.w * self.forward:.1f}pts) n={self.n_vol} cp={self.cp_spread:.3f}>")


def two_sided(q, require_ask_size=True):
    """Is this quote a genuine two-sided market?  Re-exported from `pricing.theo`.

    The rule and the incident that motivates it live in `theo.two_sided`. It moved there on
    2026-08-24 because it had been written out three times -- here, in `backtest/theo_bt`, and
    NOT in `theo.Pricer.parity_spot`, which is the one the live loop reaches. One definition,
    three callers. The name stays exported here so existing `atmvol.two_sided` callers and the
    `require_ask_size` contract are unchanged.
    """
    return _two_sided(q, require_ask_size=require_ask_size)


def combos(quotes, strikes):
    """[(abs_cp, strike, implied_S, C, P)] over strikes with two genuinely two-sided legs."""
    out = []
    for k in strikes:
        qc, qp = quotes.get(("C", k)), quotes.get(("P", k))
        if not (two_sided(qc) and two_sided(qp)):
            continue
        c, p = quote_value(qc), quote_value(qp)
        if c is MISSING or p is MISSING:
            continue
        out.append((abs(c - p), float(k), float(k) + c - p, c, p))
    return out


def _grid_step(strikes, centre):
    """Spacing of the strike grid near `centre`.

    Local rather than the board-wide minimum: SPX lists 5-wide near the money and up to 200-wide
    in the tails, so a global min describes the near-money grid but a global *mean* would not,
    and either way the weighting below wants the scale where the strikes actually are.
    """
    near = sorted(strikes, key=lambda k: abs(k - centre))[:PARITY_STRIKES]
    near.sort()
    gaps = [b - a for a, b in zip(near, near[1:]) if b > a]
    return min(gaps) if gaps else 1.0


def parity_forward(quotes, strikes):
    """Put-call-parity forward to 16:00, as a strike-distance-weighted average.

    Anchor on the smallest |C - P| -- the most ATM strike, where both legs carry the most time
    value and the least bid/ask noise per dollar -- then average the implied forwards of the 25
    nearest strikes around it.

    Weight is `1/sqrt(max(|K - F0| / step, 1/2))`: nearer strikes count for more, falling off as
    the inverse square root of the distance in grid steps. Distance is measured in *steps* rather
    than points so the same rule travels from SPX's 5-wide grid at 7,700 to NDX's 10-wide at
    29,700 without retuning. The 1/2 floor keeps the strike sitting exactly on F0 from taking an
    infinite weight.
    """
    cs = combos(quotes, strikes)
    if not cs:
        return None, {"reason": "no_two_sided_strikes", "n_parity": 0}
    cs.sort()
    anchor_cp, anchor_k, naive_F = cs[0][0], cs[0][1], cs[0][2]

    near = sorted(cs, key=lambda r: abs(r[1] - naive_F))[:PARITY_STRIKES]
    step = _grid_step([r[1] for r in cs], naive_F)

    num = den = 0.0
    for _, k, implied_s, _, _ in near:
        w = 1.0 / ((max(abs(k - naive_F) / step, 0.5)) ** 0.5)
        num += w * implied_s
        den += w
    if den <= 0:
        return None, {"reason": "no_parity_weight", "n_parity": 0}
    return num / den, {"reason": None, "anchor_strike": anchor_k,
                       "naive_forward": naive_F, "n_parity": len(near),
                       "anchor_cp": anchor_cp}


def delta_band(now=None):
    """(lo, hi) call-delta band: the late band inside ATM_DELTA_LATE_MINUTES of 16:00 ET."""
    if now is not None:
        et = now.astimezone(config.ET)
        close = et.replace(hour=16, minute=0, second=0, microsecond=0)
        if (close - et).total_seconds() <= 60.0 * config.ATM_DELTA_LATE_MINUTES:
            return config.ATM_DELTA_LATE_LO, config.ATM_DELTA_LATE_HI
    return config.ATM_DELTA_LO, config.ATM_DELTA_HI


def atm_total_vol(quotes, strikes, forward, k=ATM_VOL_STRIKES, selection=None, now=None):
    """Median implied total vol over the most-ATM two-sided strikes.

    `selection` (default `config.ATM_VOL_SELECTION`) picks which strikes: "strikes" takes the `k`
    smallest |C-P|; "delta" takes every strike among the PARITY_STRIKES most-ATM whose call delta,
    at its OWN implied vol, lies strictly inside (config.ATM_DELTA_LO, config.ATM_DELTA_HI). Each
    strike's delta comes from its own vol, so the band does not depend on the w it produces.
    With `now`, the band is `delta_band(now)` -- wider in the last hour before the 16:00 expiry.

    Each leg is inverted against the *common weighted forward*, not against its own strike's
    implied forward. That choice is what turns the call/put disagreement into a usable signal:
    solved against a shared F, a C and a P that are individually mispriced in opposite directions
    still recover nearly the same vol (checked with the legs pushed +-20%: 0.00345 recovered
    against 0.00350 true), so a large `cp_spread` means the board is genuinely inconsistent
    rather than merely noisy.
    """
    cs = combos(quotes, strikes)
    if not cs:
        return None, {"reason": "no_two_sided_strikes", "n_vol": 0}
    cs.sort()                                   # by |C - P| -- most ATM first

    by_delta = (selection or config.ATM_VOL_SELECTION) == "delta"
    lo, hi = delta_band(now)
    per_strike, spreads, used = [], [], []
    for _, strike, _, c, p in cs[:PARITY_STRIKES if by_delta else k]:
        wc = bs.implied_w(forward, strike, c, "C")
        wp = bs.implied_w(forward, strike, p, "P")
        both = [x for x in (wc, wp) if x is not None]
        if not both:
            continue
        w_k = sum(both) / len(both)
        if by_delta and not lo < bs.norm_cdf(bs.d1_d2(forward, strike, w_k)[0]) < hi:
            continue
        per_strike.append(w_k)
        used.append(strike)
        if wc is not None and wp is not None:
            mid = 0.5 * (wc + wp)
            spreads.append(abs(wc - wp) / mid if mid > 0 else 0.0)

    if not per_strike:
        return None, {"reason": "no_strike_in_delta_band" if by_delta
                      else "no_invertible_atm_quote", "n_vol": 0}

    w = median(per_strike)
    if w <= 0:
        return None, {"reason": "degenerate_vol", "n_vol": len(per_strike)}
    disp = (max(per_strike) - min(per_strike)) / w if len(per_strike) > 1 else 0.0
    cp = max(spreads) if spreads else 0.0
    return w, {"reason": None, "n_vol": len(per_strike), "cp_spread": cp,
               "dispersion": disp, "w_anchor": per_strike[0], "strikes_used": used}


def fit(quotes, strikes, gate=True, index_forward=None, selection=None, now=None):
    """quotes + strike grid -> AtmFit. The single entry point pricing/hourly uses.

    **The parity forward is the primary and the index is only a fallback.** That is the opposite
    of the daily board (pricing/spot), and the asymmetry is not an oversight -- it was measured.

    The parity forward is BY CONSTRUCTION the forward at which the call-implied and put-implied
    vols agree, which is precisely what `cp_spread` measures and `MAX_CP_VOL_SPREAD` gates. So
    `cp_spread` is really a forward-error detector, and imposing any externally-sourced level
    inflates it. Measured 2026-08-21 on a live SPXW strip, sweeping the forward around parity:

        offset   -2.0    -1.5    -1.0     0.0    +1.0    +1.5    +2.0
        cp_spr  0.360   0.256   0.163   0.004   0.155   0.247   0.352
        ok      False   False    True    True    True    True   False

    The gate's effective tolerance is about +-1.75 points. Two minutes of skew between the option
    snapshot and the index snapshot is worth about that much on a normal day, so substituting the
    index here would spuriously reject healthy boards as `cp_vol_spread` whenever the two feeds
    drifted apart -- converting a priced board into an unpriced one for a reason that has nothing
    to do with the board.

    The index still earns its place: `index_minus_parity` is recorded on every fit as the health
    signal, and `index_forward` is used when parity fails outright, which is the case that
    previously had no answer at all.
    """
    F_par, fd = parity_forward(quotes, strikes)

    if F_par is not None:
        F, src = F_par, "parity"
    elif index_forward is not None:
        # Parity could not price itself; the index at least locates the board. The vol backed out
        # against it is still suspect, so this keeps parity's reason rather than clearing it.
        F, src = index_forward, "index_fallback"
    else:
        return AtmFit(reason=fd["reason"], n_parity=fd.get("n_parity", 0),
                      forward_source=None,
                      index_minus_parity=None, parity_forward=None)

    imp = (index_forward - F_par) if (index_forward is not None and F_par is not None) else None

    w, wd = atm_total_vol(quotes, strikes, F, selection=selection, now=now)
    reason = wd["reason"] or (fd["reason"] if src == "index_fallback" else None)
    out = AtmFit(forward=F, w=w, reason=reason,
                 forward_source=src, parity_forward=F_par, index_minus_parity=imp,
                 anchor_strike=fd.get("anchor_strike"), naive_forward=fd.get("naive_forward"),
                 n_parity=fd.get("n_parity"), n_vol=wd.get("n_vol"),
                 cp_spread=wd.get("cp_spread"), dispersion=wd.get("dispersion"),
                 w_anchor=wd.get("w_anchor"), strikes_used=wd.get("strikes_used"))
    if w is None or not gate:
        return out
    # Gate after recording the numbers, so a rejected fit is still inspectable.
    if out.cp_spread is not None and out.cp_spread > MAX_CP_VOL_SPREAD:
        out.reason = "cp_vol_spread"
    elif out.dispersion is not None and out.dispersion > MAX_VOL_DISPERSION:
        out.reason = "atm_vol_dispersion"
    return out
