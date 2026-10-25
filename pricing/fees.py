"""Kalshi trading fees, taken as DATA rather than hardcoded as a rate.

A fee schedule is not a constant.  Kalshi carries `fee_type` and `fee_multiplier` per series
(`GET /series`, and the history via `GET /series/fee_changes?show_historical=true`), plus an
optional per-event override layered on top.  Both already land in our Parquet -- `storage/schemas`
SERIES has `fee_type`/`fee_multiplier`, EVENTS has `fee_type_override`/`fee_multiplier_override`,
populated by `storage/rest_backfill`.  So a backtest can price the fee that was actually in force
on the day it is scoring, and this module is the pure arithmetic that turns those parameters plus
a fill into dollars.

**The finding that matters for our book.**  All six series we trade -- KXINX, KXNASDAQ100, KXINXU,
KXNASDAQ100U, KXINXDUD, KXNASDAQDUD -- are `fee_type = "quadratic"` with `fee_multiplier = 1.0`
(read from our own captured series snapshot, 2026-08-21).  Per the API's own description
(`kalshi_api.yaml`, FeeType): "'quadratic' is described by the General Trading Fees Table,
'quadratic_with_maker_fees' is described by the General Trading Fees Table with maker fees
described in the Maker Fees section."  Only four series exchange-wide carry the maker variant, and
none of them is ours.

So **as the resting maker on our board we pay no trading fee at all**; the fee bites only when we
cross.  That is the single most consequential number for the adds/removes study, whose entire gross
edge is 4.1-4.4 cents per contract -- a maker fee anywhere near that would have flipped its sign.
It does not exist on these series.  This module makes that explicit rather than assumed, so that
the day a series moves to `quadratic_with_maker_fees` the study starts refusing instead of quietly
scoring zero.

**The coefficient is VERIFIED against a real fill.**  It is not in the API -- the API gives only
the multiplier applied to it, and points at kalshi.com/docs/kalshi-fee-schedule.pdf, which sits
behind a JS challenge and could not be fetched.  But the account has one real fill, and it settles
the question outright (2026-07-26, KXMVECROSSCATEGORY, taker, count_fp 2.62, yes 0.3650):

    0.07 * 2.62 * 0.3650 * 0.6350 = 0.042507535   ->  ceil to centicent  ->  0.0426
    Kalshi's own fee_cost on that fill            ->  0.042600           exact match

That pins three things at once: the coefficient is 0.07, the rounding is **up to the centicent**
(rounding up to the nearest *cent* would have charged 0.05, which it did not -- so the widely
repeated "rounds to the next cent" is stale), and the contract count is fractional fixed-point.
`validate_fees.check_against_fills` re-runs that comparison against every fill on the account, so
it tightens on its own as the book trades.
"""
import math

MISSING = None

# Coefficient of the General Trading Fees Table.  Verified to the centicent against a real
# fee_cost -- see the module docstring.  Implies the familiar 1.75c/contract peak at P = 0.50.
QUADRATIC_COEFF = 0.07

# Fees round UP to the nearest centicent, per docs.kalshi.com/getting_started/fee_rounding:
# "Trade fee: fee from the fee model, rounded up to the nearest $0.0001 (centicent)".
#
# The same page describes a second stage -- a "rounding fee" that floors the resulting balance
# change to the member's balance precision, and a rebate refunding accumulated overpayment.  That
# pair is a settlement-time artefact of how a balance is stored, nets to under a cent, and is not
# a per-trade cost.  We record the trade fee only, and deliberately do not model it.
FEE_QUANTUM = 0.0001

QUADRATIC = "quadratic"
QUADRATIC_WITH_MAKER = "quadratic_with_maker_fees"
FLAT = "flat"

KNOWN_FEE_TYPES = frozenset({QUADRATIC, QUADRATIC_WITH_MAKER, FLAT})


# Relative tolerance for "this ratio is really an integer".  See ceil_to.
_CEIL_REL_EPS = 1e-9


def ceil_to(x, quantum=FEE_QUANTUM):
    """Round up to a multiple of `quantum`, with the float slop taken out first.

    A naive `ceil(x / q) * q` promotes a final-ulp error into a whole extra centicent, and it does
    so *asymmetrically*: `0.07*100*0.30*0.70` is exactly 1.47 while `0.07*100*0.70*0.30` is
    1.4700000000000004, so the same fee came out a centicent apart depending on which side of the
    market you named it from.  Since P(1-P) is symmetric, that asymmetry is pure arithmetic noise.

    Rounding the ratio to a fixed number of DECIMALS does not fix it -- the ratio here is ~14700,
    where twelve decimals is far finer than the ulp.  The test has to be relative: if the ratio is
    within a relative epsilon of an integer, it IS that integer.
    """
    if x <= 0.0:
        return 0.0
    r = x / quantum
    nearest = round(r)
    if abs(r - nearest) <= _CEIL_REL_EPS * max(1.0, abs(r)):
        return nearest * quantum
    return math.ceil(r) * quantum


def quadratic_fee(price, contracts, multiplier=1.0, coeff=QUADRATIC_COEFF):
    """multiplier * coeff * C * P * (1 - P), rounded up to the centicent.

    Rounded ONCE for the whole fill, not per contract: the schedule is written in terms of C, and
    rounding per contract would overcharge by up to a centicent times the lot size.
    """
    return ceil_to(multiplier * coeff * contracts * price * (1.0 - price))


def trade_fee(price, contracts, fee_type, fee_multiplier, is_taker):
    """Fee in dollars for one fill, or MISSING when the schedule is not known.

    `price` is the price of the side being charged, in dollars (0..1).  `is_taker` selects the
    side of the fill: on Kalshi both parties trade at the same price, but only some series charge
    the maker at all.

    Returns `(fee, detail)`.  MISSING rather than a guess whenever the parameters name a schedule
    this module cannot price -- an unknown `fee_type`, or a maker on a series that does charge
    makers.  A study that silently scores an unknown fee as zero is a study that will report a
    profit it did not earn.
    """
    d = {"fee_type": fee_type, "fee_multiplier": fee_multiplier, "is_taker": bool(is_taker),
         "coeff": QUADRATIC_COEFF, "side": "taker" if is_taker else "maker",
         "fee": None, "reason": None}

    if fee_type not in KNOWN_FEE_TYPES:
        d["reason"] = "unknown_fee_type"
        return MISSING, d
    if price is None or not (0.0 <= price <= 1.0):
        d["reason"] = "price_out_of_range"
        return MISSING, d
    if fee_multiplier is None:
        d["reason"] = "no_fee_multiplier"
        return MISSING, d
    if contracts is None or contracts <= 0:
        d["fee"] = 0.0
        d["reason"] = "no_contracts"
        return 0.0, d

    if fee_type == QUADRATIC:
        # The General Trading Fees Table has no maker component.  A resting maker on such a
        # series pays nothing; this is the branch our whole book takes.
        fee = quadratic_fee(price, contracts, fee_multiplier) if is_taker else 0.0
        d["fee"] = fee
        return fee, d

    if fee_type == QUADRATIC_WITH_MAKER:
        if is_taker:
            fee = quadratic_fee(price, contracts, fee_multiplier)
            d["fee"] = fee
            return fee, d
        # The Maker Fees section of the PDF is the only place this rate is written down, and the
        # PDF could not be fetched.  Refuse rather than score a maker fill at zero on a series
        # that demonstrably charges makers.
        d["reason"] = "maker_fee_unverified"
        return MISSING, d

    # FLAT -- the Specific Trading Fees Table, a per-series table of flat rates that the
    # multiplier alone does not determine.  None of our six series use it.
    d["reason"] = "flat_rate_unverified"
    return MISSING, d


def resolve_params(series_row, event_row=None):
    """(fee_type, fee_multiplier) for a market, applying the event override if one is set.

    Per the API: "Event fees are an override layered on top of the parent series' fee structure.
    If `fee_type_override` and `fee_multiplier_override` are null, that indicates the override is
    cleared."  The two override independently, so a null type with a non-null multiplier keeps the
    series type and takes the event's multiplier.
    """
    fee_type = (series_row or {}).get("fee_type")
    mult = (series_row or {}).get("fee_multiplier")
    if event_row:
        if event_row.get("fee_type_override") is not None:
            fee_type = event_row["fee_type_override"]
        if event_row.get("fee_multiplier_override") is not None:
            mult = event_row["fee_multiplier_override"]
    return fee_type, mult


def schedule_asof(fee_changes, when):
    """The (fee_type, fee_multiplier) in force at `when`, from a fee-changes timeline.

    `fee_changes` is the `/series/fee_changes?show_historical=true` payload: rows carrying
    `fee_type`, `fee_multiplier` and `scheduled_ts`.  Returns the latest row whose `scheduled_ts`
    is at or before `when`, so a backtest scores each session against the schedule that was
    actually in force rather than against today's.

    Returns `(None, None)` when the timeline starts after `when` -- the honest answer for a
    session predating the earliest recorded schedule, and one the caller must handle rather than
    defaulting to the current one.
    """
    best = None
    for row in fee_changes or ():
        ts = row.get("scheduled_ts")
        if ts is None or ts > when:
            continue
        if best is None or ts > best.get("scheduled_ts"):
            best = row
    if best is None:
        return None, None
    return best.get("fee_type"), best.get("fee_multiplier")
