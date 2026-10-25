"""Maps Kalshi bracket boundaries -> the SPXW option tickers needed to price them."""
import json

import config
from massive import client as mv
from pricing import theo
# massive_history imports only stdlib + config, so this costs the live path nothing.
# massive_client would drag a websocket import into the storage and backtest graphs.
from massive.history import pm_settled_root


def latest_kalshi_dir():
    stamp = (config.RAW_DIR / "kalshi" / "latest").read_text().strip()
    return config.RAW_DIR / "kalshi" / stamp


def load_event_markets(outdir=None):
    outdir = outdir or latest_kalshi_dir()
    target = config.event_ticker_for()
    return json.loads((outdir / f"event_{target}.json").read_text())


def option_root_and_strikes(expiry_date=None, underlying=None):
    """Discover the real strike grid rather than assuming 5-wide holds in the tails.

    On monthly opex an expiry lists two roots -- SPX (AM-settled, the 09:30 print) beside SPXW
    (PM-settled, the 16:00 close), and NDX beside NDXP. Kalshi settles these markets on the 4pm
    close, so the AM root prices the wrong event entirely. Always take the PM root.

    Filtering `contracts` to that root is the load-bearing half, not a tidy-up: a merged grid
    would feed neighbors() and the minimum strike interval with strikes that are not on the
    traded board, and run_02 writes this list to contracts_spx.json which the backtest reads
    back to rebuild the grid.
    """
    # `underlying` is an argument rather than a read of `config.OPTION_UNDERLYING` because
    # `mv.option_contracts` binds that as a DEFAULT ARGUMENT, i.e. at import time -- so a caller
    # that set `config.OPTION_UNDERLYING` to price NDX would silently get the SPX strip back.
    # Two underlyings in one process makes that a live trap rather than a tidiness point.
    contracts = mv.option_contracts(underlying=underlying or config.OPTION_UNDERLYING,
                                    expiration_date=str(expiry_date or config.today_et()))
    root = pm_settled_root(contracts)
    contracts = [c for c in contracts if c["ticker"][2:-15] == root]
    return root, sorted({c["strike_price"] for c in contracts}), contracts


def bracketing_strikes(markets, strikes):
    """The strikes every boundary prices from. These are PINNED for the whole session.

    They already span the entire Kalshi board -- measured 2026-08-20: +-4.6% on SPX, +-4.8% on
    NDX -- so bracket pricing survives a large intraday move without any re-centring. Only the
    parity band below needs to follow spot.
    """
    needed = set()
    for b in theo.boundaries(markets):
        lo, hi = theo.neighbors(b, strikes)
        if lo is None:
            raise RuntimeError(f"no listed strikes bracketing boundary {b}")
        needed.update((lo, hi))
    return needed


def parity_band(strikes, centre, band_pct=None):
    """Listed strikes within +-band_pct of `centre`, for the put-call parity spot.

    Percent, not a strike count, because a fixed count means wildly different coverage per
    board: 50 strikes is +-3.3% on SPX's 5-wide near-money grid but only +-0.85% on NDX, whose
    grid is also 5-wide at the money but sits at ~29,400 instead of ~7,675. Percent travels.
    """
    band_pct = config.QUOTE_BAND_PCT if band_pct is None else band_pct
    if centre is None:
        return set()
    lo, hi = centre * (1 - band_pct), centre * (1 + band_pct)
    return {s for s in strikes if lo <= s <= hi}


def desired_tickers(markets, strikes, root, expiry, spot, band_pct=None,
                    cap=None):
    """Pinned bracketing strikes + a parity band around `spot`, as option tickers.

    Trims the band from the outside in if the total would breach the connection cap -- the
    pinned set is never sacrificed, because losing a bracketing strike silently unprices a
    market whereas a narrower band only weakens the spot estimate.
    """
    cap = mv.QuoteCollector.MAX_TICKERS if cap is None else cap
    pinned = bracketing_strikes(markets, strikes)
    band = parity_band(strikes, spot, band_pct)
    want = pinned | band
    if len(want) * 2 > cap:
        centre = spot if spot is not None else (min(pinned) + max(pinned)) / 2
        extra = sorted(band - pinned, key=lambda s: abs(s - centre))
        room = max(0, cap // 2 - len(pinned))
        want = pinned | set(extra[:room])
    return sorted(want), [mv.occ_ticker(root, expiry, r, k)
                          for k in sorted(want) for r in ("C", "P")]


def build(markets, strikes, root, expiry, pad=3):
    """Strikes needed = each boundary's immediate neighbours, plus a near-money band for parity.

    `pad` extra grid steps on each side of the boundary span give put-call parity enough
    two-sided quotes to anchor spot. Still wanted after `I:SPX` became entitled on 2026-08-21:
    parity is the fallback when the index feed is stale, and the cross-check when it is not.
    """
    bounds = theo.boundaries(markets)
    needed = set()
    for b in bounds:
        lo, hi = theo.neighbors(b, strikes)
        if lo is None:
            raise RuntimeError(f"no listed strikes bracketing boundary {b}")
        needed.update((lo, hi))
    span_lo, span_hi = min(needed), max(needed)
    idx = {k: i for i, k in enumerate(strikes)}
    i_lo, i_hi = idx[span_lo], idx[span_hi]
    needed.update(strikes[max(0, i_lo - pad): i_hi + 1 + pad])

    strikes_needed = sorted(needed)
    tickers = [mv.occ_ticker(root, expiry, right, k)
               for k in strikes_needed for right in ("C", "P")]
    return bounds, strikes_needed, tickers


def quotes_by_strike(snapshot, root, expiry, strikes):
    """Massive snapshot {sym: Q} -> {('C'|'P', strike): Q} for theo.Pricer."""
    out = {}
    for k in strikes:
        for right in ("C", "P"):
            q = snapshot.get(mv.occ_ticker(root, expiry, right, k))
            if q:
                out[(right, k)] = q
    return out
