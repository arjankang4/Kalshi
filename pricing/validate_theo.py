"""Compare model fair values against live Kalshi quotes. Run before any live order."""
import json

import config
from kalshi import client as k
from massive import client as mv
from massive import index as ix
from pricing import skew as skewmod
from pricing import spot as spotmod
from pricing import hourly
from pricing import theo
from pricing import universe


def price_markets(markets, snapshot, root, expiry, strikes, spot=None):
    quotes = universe.quotes_by_strike(snapshot, root, expiry, strikes)
    pricer = theo.Pricer(quotes, strikes, spot=spot)
    return pricer, {m["ticker"]: pricer.market_fair(m) for m in markets}


def resolve_and_price(markets, snapshot, root, expiry, strikes, index_quote=None,
                      skew_state=None):
    """Price a board off the resolved spot, returning the resolution alongside.

    Returns `(pricer, fairs, resolution)`. The resolution carries BOTH candidate spots and their
    divergence -- see pricing/spot -- so the caller can journal the health signal without
    recomputing anything.

    The parity spot is always computed, even when the index wins, because it is the cross-check.
    That costs one pass over the strike list; pricing the board is the expensive part and only
    happens once, since `theo.Pricer` skips `parity_spot()` entirely when handed a `spot=`.

    `skew_state` is an optional `strategy.skewstate.SkewState` carried ACROSS cycles by the
    quote loop. With it, the curve is refit at most every `SKEW_RECOMPUTE_S` and served as a
    recency-weighted mean over `SKEW_WINDOW_S`; without it the Pricer refits from this one
    snapshot on first use, which is what the backtest and the validators want -- they price
    single instants and have no history to smooth over.

    Smoothing is free here in a way it is not for ATM vol: `pricing/study_skew_cadence` measures
    the skew ratio's structure exponent at H ~ 0.19-0.30, far from the 0.503 of a diffusion, so
    most of its short-horizon motion is the quote noise of two inverted legs rather than
    information. The forward handed to the state is this Pricer's OWN spot, not atmvol's
    separately-derived one -- mixing two forwards would let the ratio absorb the difference
    between them and call it skew (see `skew.atm_vol`).
    """
    pricer, res, _ = _resolve_and_smooth(snapshot, root, expiry, strikes,
                                         index_quote=index_quote, skew_state=skew_state)
    return pricer, {m["ticker"]: pricer.market_fair(m) for m in markets}, res


def _resolve_and_smooth(snapshot, root, expiry, strikes, index_quote=None, skew_state=None):
    """(daily_pricer, spot_resolution, smoothed_curve). The part every board shares.

    Split out 2026-08-25 so several boards can be priced in one cycle without doing this twice.
    BOTH steps here are once-per-cycle facts about the OPTION board, not about any Kalshi board:

      - the spot resolution is a property of the index feed and the option strip;
      - the smoothed skew is a property of the SPXW smile. There is no separate option board at
        15:00 to read a different smile from, so the 4pm brackets and every hourly ladder use the
        same curve.

    `SkewState.observe` in particular MUST run exactly once per cycle. It prunes its window on
    wall time, so observing once per board would multiply the sample count without advancing the
    clock and quietly shorten the effective half-life -- the smoother would report a window it
    does not have.
    """
    quotes = universe.quotes_by_strike(snapshot, root, expiry, strikes)
    probe = theo.Pricer(quotes, strikes)                    # parity, unconditionally
    res = spotmod.resolve(probe.spot, index_quote)
    pricer = (theo.Pricer(quotes, strikes, spot=res.value)
              if res.source == "index" else probe)
    smoothed = None
    if skew_state is not None and pricer.spot:
        w_atm = skewmod.atm_vol(quotes, strikes, pricer.spot)
        if w_atm:
            # Observe unconditionally -- the state prunes on wall time, so skipping the call
            # would leave a queue ageing with nobody noticing it had gone quiet.
            smoothed = skew_state.observe(quotes, strikes, pricer.spot, w_atm)
            # But only PRICE off it when it is a real curve. When the window holds too few
            # buckets to spline (`too_few_smoothed_buckets`) the state returns a FLAT SkewFit,
            # and installing that would throw the smile away and quietly price every strike at
            # the ATM vol -- strictly worse than the per-snapshot fit the Pricer would build for
            # itself. Falling through leaves `Pricer.skew` lazy, which is the old behaviour.
            if smoothed is not None and smoothed.ok:
                pricer.use_skew(smoothed)
            else:
                smoothed = None
    return pricer, res, smoothed


def resolve_and_price_boards(daily_markets, hourly_markets, snapshot, root, expiry, strikes,
                             now=None, calendar=None, index_quote=None, skew_state=None):
    """Price a 4pm bracket board AND any number of hourly ladders in one pass.

    Returns `(pricers, fairs, resolution)` with `pricers = {"daily": ..., "hourly": ...}` and one
    merged `{ticker: (fair, detail)}`. Downstream is genuinely indifferent to which pricer
    produced an entry -- `run_04.build_desired` reads only `detail["fair"]` and
    `detail.get("reason")`.

    ONE HourlyPricer COVERS EVERY HOUR. `market_fair` reads each market's own `close_time` and
    rescales the 16:00 total vol through the variance calendar, so seven open events need one
    instance, not seven. (`pricing/validate_hourly` prices every open event this way already.)

    THE TWO PRICERS CAN DISAGREE ABOUT THE FORWARD, deliberately. The daily pricer takes the
    RESOLVED spot -- the index when it is fresh, because that is what KXINX settles on. The hourly
    pricer takes `atmvol`'s PARITY forward, because `atmvol.fit` treats the index as a fallback
    only: substituting it inflates `cp_spread` and trips the MAX_CP_VOL_SPREAD gate, which is
    measured and documented in that module. The gap between the two is exactly what
    `spot.resolve` already alarms on at INDEX_PARITY_ALARM_PCT, so it is monitored rather than
    hidden. `index_quote`'s level is passed down as atmvol's fallback and nothing more.
    """
    pricer, res, smoothed = _resolve_and_smooth(snapshot, root, expiry, strikes,
                                                index_quote=index_quote, skew_state=skew_state)
    fairs = {m["ticker"]: pricer.market_fair(m) for m in (daily_markets or [])}
    hp = None
    if hourly_markets and calendar is not None and now is not None:
        quotes = universe.quotes_by_strike(snapshot, root, expiry, strikes)
        hp = hourly.HourlyPricer(quotes, strikes, now, calendar,
                                 index_forward=(res.index if res else None))
        if smoothed is not None:
            hp.use_skew(smoothed)
        for m in hourly_markets:
            fairs[m["ticker"]] = hp.market_fair(m)
    return {"daily": pricer, "hourly": hp}, fairs, res


def main():
    mstamp = (config.RAW_DIR / "massive" / "latest").read_text().strip()
    mdir = config.RAW_DIR / "massive" / mstamp
    snap = json.loads((mdir / "latest_quotes.json").read_text())
    tickers = json.loads((mdir / "subscribed_tickers.json").read_text())
    strikes = sorted({int(t[-8:]) / 1000 for t in tickers})
    markets = universe.load_event_markets()

    try:
        iq = ix.snapshot((config.OPTION_UNDERLYING,)).get(config.OPTION_UNDERLYING)
    except Exception as e:                                   # noqa: BLE001
        print(f"[index] snapshot failed ({type(e).__name__}: {str(e)[:80]})")
        iq = None
    pricer, fairs, res = resolve_and_price(markets, snap, "SPXW", config.occ_expiry(),
                                           strikes, index_quote=iq)
    spy = mv.stock_mid(json.loads((mdir / "spy_spot.json").read_text())) * 10
    idx_s = "n/a" if res.index is None else f"{res.index:.2f}"
    par_s = "n/a" if res.parity is None else f"{res.parity:.2f}"
    div_s = "n/a" if res.divergence is None else f"{res.divergence:+.2f} pts ({res.divergence_pct:+.4%})"
    print(f"spot {pricer.spot:.2f} via {res.source}"
          + (f" ({res.reason})" if res.reason else ""))
    print(f"  index {idx_s} | parity {par_s} | divergence {div_s}"
          + ("   <<< ALARM" if res.alarm else ""))
    print(f"  SPYx10 {spy:.1f} | basis {pricer.spot - spy:+.1f} pts")
    print("  NOTE: the cached snapshot and the live index are from different instants; "
          "divergence here is only meaningful on a fresh capture.\n")

    print(f"{'ticker':<32}{'theo':>7}{'k_bid':>7}{'k_ask':>7}{'k_mid':>7}{'diff':>8}{'vol':>8}")
    total = 0.0
    n_priced = 0
    for m in sorted(markets, key=lambda m: m.get("floor_strike") or 0):
        fair, detail = fairs[m["ticker"]]
        kb, ka = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
        kmid = (kb + ka) / 2
        total += fair or 0.0
        n_priced += fair is not None
        fs = "None" if fair is None else f"{fair:.4f}"
        df = "" if fair is None else f"{fair - kmid:+.4f}"
        flag = "  <<" if fair is not None and abs(fair - kmid) > 0.10 else ""
        if fair is None:
            flag = f"  {detail.get('reason') or 'no_fair'}"
        print(f"{m['ticker']:<32}{fs:>7}{kb:>7.2f}{ka:>7.2f}{kmid:>7.3f}{df:>8}"
              f"{float(m['volume_fp']):>8.0f}{flag}")

    # The sum-to-1 check is only meaningful for a mutually-exclusive-and-exhaustive bracket
    # set. A `greater_or_equal` ladder (KXINXU) overlaps by construction -- every rung contains
    # the next -- so summing it would print a number near the rung count and mean nothing.
    exhaustive = {m.get("strike_type") for m in markets} <= {"between", "less", "greater",
                                                             "less_or_equal"}
    if exhaustive:
        print(f"\nsum of fairs = {total:.4f}   "
              f"(mutually exclusive + exhaustive -> should be ~1.00)")
    else:
        print(f"\n{n_priced}/{len(markets)} markets priced   "
              f"(overlapping ladder -- sum-to-1 does not apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
