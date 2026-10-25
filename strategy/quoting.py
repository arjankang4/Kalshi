"""Where to quote, given a theo and a book. Pure functions, no I/O.

The whole point of this module having no I/O is that the rules can be tested against a worked
example before a single order is sent. `decide()` takes numbers and returns an intent; nothing
here talks to Kalshi.

The rule, per side (bid shown; the offer is the mirror with edge = price - theo):

    target = theo -+ QUOTE_OFFSET                       the edge we would like to charge
    if width > 2 ticks:  target = min(target, mid)      wide book: never bid through mid
    edge(p) = theo - p

    1. target improves the book and edge(target) > 0        -> POST target      (improve)
    2. edge(best + tick) >= EDGE_MIN                        -> POST best+tick   (penny)
    3. edge(best) > 0                                       -> POST best        (join)
    4. otherwise                                            -> no quote

The mid cap is what stops us charging too little edge on a wide market. On a 4/12 book with
theo 0.10: target 0.09 caps to mid 0.08, edge +0.02. The *offer* on that same book caps to 0.08,
which is below theo -- negative edge -- so rule 1 fails and we fall through to pennying 0.11 for
+0.01. Both halves of that example are pinned in test_quoting.

Two guards the spec did not name but the ladder needs:

  * **Never cross.** On a 1c-wide book `best_bid + tick == best_ask`, so pennying would quote at
    the far touch and take. Rule 2 is skipped whenever it would reach or cross the other side.
    With post_only=True on the order this is belt and braces, but a post_only rejection is a
    failed quote, not a safe one.
  * **A one-sided book is treated as a `ONE_SIDED_WIDTH`-wide one.** If a touch is missing we
    synthesise it `ONE_SIDED_WIDTH` ticks from the touch that exists, so the tighten cap engages and
    the cross guard binds. Without that, a missing side means no width at all, the cap is skipped,
    and the target goes in at full edge -- several cents inside the only price anyone is showing.
    Less competition should mean giving away LESS, not more. With both
    touches missing there is nothing to anchor to and the plain target stands.
"""
import math
from collections import namedtuple

import config

TICK = 0.01
# Minimum edge for pennying or joining the touch, and the default rung bar. Illustrative value; the
# live value is not published.
EDGE_MIN = 0.01

MICROS = 1_000_000
EDGE_MIN_MICROS = 10_000                # == EDGE_MIN in micro-dollars

# On a one-sided book, the missing side is assumed to be this many ticks away. Illustrative value;
# the live value is not published.
ONE_SIDED_WIDTH = 8

# kind: improve | penny | join | none
Action = namedtuple("Action", "kind price edge reason")

NO_QUOTE = Action("none", None, None, "no_positive_edge")


def _floor_tick(x):
    return math.floor(round(x / TICK, 9)) * TICK


def _ceil_tick(x):
    return math.ceil(round(x / TICK, 9)) * TICK


def _edge(side, theo, price):
    """What we earn per contract if the theo is right. Positive is good for us."""
    return (theo - price) if side == "bid" else (price - theo)


def _edge_micros(side, theo, price):
    """Edge as an integer, in micro-dollars.

    Every threshold comparison goes through this rather than through floats. Measured on the
    real boundary: theo 0.115 against a 0.11 quote gives 0.0050000000000000044 in float, so an
    edge test against a half-cent bar is True by noise alone and the ladder pennies a market it
    should have joined. Integers make the bar mean exactly what it says -- the same reason
    storage/units.py refuses to hold a price as a float.
    """
    t = int(round(theo * MICROS))
    p = int(round(price * MICROS))
    return (t - p) if side == "bid" else (p - t)


def _synth_touches(best_bid, best_ask):
    """(bid, ask) with an absent side imputed `ONE_SIDED_WIDTH` ticks away.

    Factored out of `decide` because `decide_ladder`'s back rungs need the SAME imputed far touch
    for their cross guard. Two copies of this would be two chances to disagree about where a
    missing side sits, and the rungs would then be guarded against a different book from the one
    rung 0 was priced against.
    """
    bid_c, ask_c = best_bid, best_ask
    if best_bid is None and best_ask is not None:
        bid_c = max(config.MIN_PRICE, _floor_tick(best_ask - ONE_SIDED_WIDTH * TICK))
    elif best_ask is None and best_bid is not None:
        ask_c = min(config.MAX_PRICE, _ceil_tick(best_bid + ONE_SIDED_WIDTH * TICK))
    return bid_c, ask_c


def _tradeable(price):
    """Kalshi accepts [MIN_PRICE, MAX_PRICE] only.

    This is not a formality. A worthless bracket has theo 0.00, so the bid target is
    floor(0.00 - 0.01) = -0.01; on a book with no resting bids the empty-side branch will happily
    return that with "+0.01 of edge". Observed 15 times in the first dry run. A price the
    exchange cannot accept is not a quote, so the ladder must never emit one.
    """
    return price is not None and config.MIN_PRICE <= price <= config.MAX_PRICE


def decide(side, theo, best_bid, best_ask, offset=None, min_edge_micros=None, tighten=-1.0):
    """The quoting ladder. Returns an Action; `price` is None when we should not quote.

    `min_edge_micros` is the bar rung 2 (penny) must clear. It defaults to `EDGE_MIN_MICROS`, the
    flat constant, so a caller that does not pass one gets a fixed bar. `pricing/edge` supplies a calibrated value per market instead.

    Note the asymmetry, which predates the edge manager and is kept: PENNY must clear the bar
    because it improves the market and takes on the obligation of being the best price, while JOIN
    only needs positive edge -- holding a join to the same bar would mean quoting almost nothing
    on a tight book.
    """
    if theo is None:
        return Action("none", None, None, "no_theo")
    offset = config.QUOTE_OFFSET if offset is None else offset
    bar = EDGE_MIN_MICROS if min_edge_micros is None else int(min_edge_micros)

    target = (_floor_tick(theo - offset) if side == "bid"
              else _ceil_tick(theo + offset))

    # Wide-market cap: only meaningful when both touches exist.
    #
    # On a wide book the edge-implied target can sit far inside the NBBO, and posting there hands
    # the whole spread away -- nobody was competing at that price. So when we would BE the best
    # bid or offer we tighten only a FRACTION of the way in from the touch we are improving.
    #
    #   book 8 at 15, theo 12, edge 0.8, fraction f  ->  naive target 11
    #                                                    capped at 8 + (15-8)*f, floored to the tick
    #
    # `tighten` defaults to config.QUOTE_TIGHTEN_FRACTION. Pass None for the simpler cap-at-mid
    # rule, kept so the two remain comparable.
    # A ONE-SIDED BOOK IS TREATED AS A `ONE_SIDED_WIDTH`-WIDE ONE. Otherwise a missing side means
    # no width, the cap is skipped entirely and the target goes in at full edge, bounded by nothing
    # but [0.01, 0.99]. That is the opposite of what the cap exists for: the less competition there
    # is on our side, the LESS we should give away.
    #
    # So we synthesise the absent side `ONE_SIDED_WIDTH` ticks from the touch that exists and let the ordinary
    # width logic run. `crosses()` then also binds against the synthetic side, which keeps a quote
    # from wandering to where a real offer would plausibly sit. With BOTH sides absent there is
    # nothing to anchor to and the target stands.
    bid_c, ask_c = _synth_touches(best_bid, best_ask)

    if bid_c is not None and ask_c is not None:
        width = ask_c - bid_c
        frac = config.QUOTE_TIGHTEN_FRACTION if tighten == -1.0 else tighten
        if width > 2 * TICK:
            if frac is None:
                cap = (bid_c + ask_c) / 2.0
            else:
                cap = (bid_c + width * frac if side == "bid"
                       else ask_c - width * frac)
            # min/max, so the cap only ever pulls a target BACK toward the touch; a target already
            # at or behind the touch is untouched.
            target = min(target, cap) if side == "bid" else max(target, cap)
            target = _floor_tick(target) if side == "bid" else _ceil_tick(target)

    # `own` stays the REAL touch -- improve/penny/join are about a book we can actually join, and
    # a synthetic level is not one. `far` uses the imputed side so the cross guard binds.
    own = best_bid if side == "bid" else best_ask
    far = ask_c if side == "bid" else bid_c

    def crosses(p):
        if far is None:
            return False
        return p >= far if side == "bid" else p <= far

    # 1. improve -- strictly better than the touch, positive edge, tradeable, does not cross
    if not _tradeable(target):
        target = None
    if target is not None and own is not None and not crosses(target):
        improves = target > own if side == "bid" else target < own
        em = _edge_micros(side, theo, target)
        if improves and em > 0:
            return Action("improve", round(target, 4), em / MICROS, "target_inside")
    elif target is not None and own is None and not crosses(target):
        em = _edge_micros(side, theo, target)
        if em > 0:
            return Action("improve", round(target, 4), em / MICROS, "empty_side")

    if own is None:
        return NO_QUOTE

    # 2. penny -- one tick inside the touch, but only if it retains at least the edge bar
    pen = round(own + TICK if side == "bid" else own - TICK, 4)
    if _tradeable(pen) and not crosses(pen):
        em = _edge_micros(side, theo, pen)
        if em >= bar:
            return Action("penny", round(pen, 4), em / MICROS, "penny_above_min_edge")

    # 3. join -- passive fallback. Any positive edge is enough: joining does not improve the
    #    market, so holding it to the edge bar would mean quoting almost nothing on a
    #    tight book.
    if _tradeable(own):
        em = _edge_micros(side, theo, own)
        if em > 0:
            return Action("join", round(own, 4), em / MICROS, "join_touch")

    return NO_QUOTE


def should_pull(side, theo, resting_price, alone_at_level):
    """Pull a resting quote that has gone wrong.

    Only when we are alone at our price level: if we are the whole level then our quote *is*
    that level, and leaving it there is offering the market a price we no longer believe. With
    others alongside us, being wrong by less than EDGE_MIN is not worth the queue position.
    """
    if theo is None or resting_price is None:
        return False
    return alone_at_level and _edge_micros(side, theo, resting_price) < -EDGE_MIN_MICROS


def should_improve(side, theo, resting_price, action):
    """Amend an existing quote only for a strictly better price worth more than EDGE_MIN.

    Amending forfeits queue position unless the amendment *decreases size* (Kalshi's rule), so
    an improvement has to clear a real bar rather than chase every tick.
    """
    if action.price is None or resting_price is None:
        return False
    better = (action.price > resting_price if side == "bid"
              else action.price < resting_price)
    return better and action.edge is not None and round(action.edge * MICROS) >= EDGE_MIN_MICROS


class Book:
    """Resting orders keyed (ticker, side).

    Entries are popped on fill, cancel or amend -- an amend is a new order to Kalshi when the
    price changes, so treating it as replace-not-mutate keeps this in step with the exchange.
    """

    def __init__(self):
        self._o = {}

    def get(self, ticker, side):
        return self._o.get((ticker, side))

    def put(self, ticker, side, **fields):
        self._o[(ticker, side)] = {"ticker": ticker, "side": side, **fields}

    def pop(self, ticker, side):
        return self._o.pop((ticker, side), None)

    def all(self):
        return list(self._o.values())

    def __len__(self):
        return len(self._o)


def ladders(book_entry):
    """(yes_levels, no_levels) from a Kalshi orderbook payload, whatever shape it arrives in.

    The live `/markets/orderbooks` response nests under `orderbook_fp` with `yes_dollars` /
    `no_dollars` keys; older/WS forms use bare `yes` / `no`. Verified against a real payload:
    levels are [price_string, size_string], ascending, so the BEST is the LAST entry, and a side
    with no resting interest is an empty list rather than a missing key.
    """
    if not book_entry:
        return [], []
    inner = book_entry.get("orderbook_fp") or book_entry.get("orderbook") or book_entry
    yes = inner.get("yes_dollars", inner.get("yes")) or []
    no = inner.get("no_dollars", inner.get("no")) or []
    return yes, no


def top_of_book(book_entry):
    """(best_bid, best_ask, bid_size, ask_size) in YES dollars.

    Kalshi keeps one ladder per side in that side's own terms: `yes` levels are YES bids, `no`
    levels are NO bids -- and a NO bid at p is a YES *offer* at 1 - p. Getting that inversion
    wrong silently produces a book where the bid is above the ask.
    """
    yes, no = ladders(book_entry)
    bb = ba = bsz = asz = None
    if yes:
        bb, bsz = float(yes[-1][0]), float(yes[-1][1])
    if no:
        ba, asz = round(1.0 - float(no[-1][0]), 4), float(no[-1][1])
    return bb, ba, bsz, asz


# --- tiered orders -------------------------------------------------------
#
# Kalshi is a price-time exchange, so a resting order's value is its queue position and depth
# behind the touch is worth having: the front rung competes for the tight fill, the back rungs
# collect the move that runs through them. Three rungs per side.
#
# Sizes are FLAT (config.LOT_SIZE per rung). The rungs' DISTANCE behind the front is what scales
# with risk: a market in the calm part of the distribution rests closer, one at the top further
# back (TIER_OFFSET_RANGE). `risk_u` is R's percentile rank from `pricing.edge.Curve.risk_u` -- "the same scaled
# percentile logic" as the edge itself.

# `min_edge` is the requirement this rung was placed against, in micro-dollars, and it travels WITH
# the rung because `strategy/orders.plan_side` needs it to decide when to pull the order again. The
# alternative -- threading the bar and scale_pos through `reconcile` -- would let the placement
# threshold and the withdrawal threshold be computed from different inputs, which is the shape of
# bug this ladder already paid for once. Defaulted so a caller that builds a Rung by hand (the
# tests do) still works and simply gets no hysteresis.
Rung = namedtuple("Rung", "index price count edge reason min_edge", defaults=(None,))

# Tunable. Illustrative value; the live value is not published.
TIER_OFFSET_RANGE = ((1, 3), (3, 5))


def tier_offsets(scale_pos):
    """[cents_behind_rung0] for rungs 1 and 2, on the EDGE curve's own scale.

    `scale_pos` is `pricing.edge.Curve.scale_position(R)` -- where R sits on the exponential-then-
    linear edge curve, NOT its percentile rank. Same scale as the edge itself, which is the point:
    a market that earns more edge also rests further back, by the same measure.

    Because the bulk of that curve is exponential, this compresses hard: most markets sit near the
    low end of TIER_OFFSET_RANGE and only the extreme tail reaches the high end. That is inherited
    from the edge curve on purpose -- a market that barely earns more edge should not suddenly rest
    much further back.
    """
    u = 0.0 if scale_pos is None else max(0.0, min(1.0, float(scale_pos)))
    return [int(round(lo + (hi - lo) * u)) for lo, hi in TIER_OFFSET_RANGE]


TICK_MICROS = 10_000                    # one cent, in micro-dollars


def rung_min_edge_micros(rank, min_edge_micros=None, scale_pos=0.0):
    """The edge a rung at RANK must earn, in micro-dollars. Monotone increasing in rank.

    ABSOLUTE, NOT RELATIVE. This is the number that replaced "N cents behind the rung in front":
    an absolute requirement can be evaluated on one rung without reference to any other, which is
    exactly what lets rung 1 survive rung 0 being cancelled.

    BUILT ON THE CALIBRATED EDGE, NOT ON CONSTANTS, and the difference only shows in the tail.
    A flat per-rank constant fails in the tail: on a high-R market the required edge for rung 0 can
    exceed the flat rung-1 bar, so rung 1 would demand LESS edge than rung 0. That inverts the thing
    the ladder is for: a rung further from the theo fills only on a larger adverse move, so it must
    charge more, not less. Adding the `tier_offsets` step to the calibrated edge keeps the order
    right at every point of the distribution.

    `tier_offsets` is reused rather than re-derived because it already scales with `scale_pos` on
    the edge curve's own measure, which is the property we want: a market that earns more edge also
    rests further back, by the same yardstick.
    """
    bar = EDGE_MIN_MICROS if min_edge_micros is None else int(min_edge_micros)
    if rank <= 0:
        return bar
    offs = tier_offsets(scale_pos)
    # Ranks past the calibrated offsets keep stepping by the last one rather than flattening, so a
    # longer ladder never has two ranks wanting the same edge.
    if rank <= len(offs):
        back = offs[rank - 1]
    else:
        back = offs[-1] + (rank - len(offs)) * max(1, offs[-1] - (offs[-2] if len(offs) > 1 else 0))
    return bar + back * TICK_MICROS


def decide_ladder(side, theo, best_bid, best_ask, counts, offset=None,
                  min_edge_micros=None, scale_pos=0.0, tiers_enabled=True, tighten=-1.0):
    """The full per-side order ladder: rung 0 from `decide`, then the tiered rungs behind it.

    THE BACK RUNGS ARE ANCHORED TO THE THEO, NOT TO RUNG 0, and that is the whole change of
    2026-08-26. They used to be priced `tier_offsets` cents BEHIND `act.price` and this function
    returned `[]` the moment rung 0 declined -- "if the front price is not worth quoting, resting
    size behind it is not either". That reasoning is wrong, and the first live session measured
    what it cost. `decide` gives up a whole side once the theo reaches the touch (improve is worse,
    penny is worse, join is negative), and on `B7687` that happened on **39.4% of one-sided cycles**
    -- once with `theo=0.4908` against `best_ask=0.49`, eight hundredths of a cent through. Each
    time it cancelled three orders, two of which still carried 0.9c and 1.9c of edge, and on a
    price-time exchange every one of those cancels forfeits queue position.

    Anchoring to the theo fixes it at the root: when the theo crosses the touch the back rungs step
    OUTWARD instead of vanishing, because their price never referred to the touch in the first
    place. Rung 0 is still `decide`'s, unchanged, so the A/B against the 2026-08-25 session stays
    clean -- what changes is that its decline no longer takes the ladder with it.

    Each rung's requirement is `rung_min_edge_micros(rank)` -- the calibrated per-market edge plus
    this rung's `tier_offsets` step -- so a rung that only fills on a larger adverse move demands
    more edge, not less. See that function for why a flat pair of constants inverts.

    With `tiers_enabled=False` and a `min_edge_micros` of None this returns exactly what `decide`
    returns, wrapped in one Rung. That identity is what `test_quoting` asserts before any
    calibrated number is allowed near it.
    """
    act = decide(side, theo, best_bid, best_ask, offset=offset,
                 min_edge_micros=min_edge_micros, tighten=tighten)

    out = []
    if act.price is not None:
        out.append(Rung(0, act.price, counts[0], act.edge, act.reason,
                        rung_min_edge_micros(0, min_edge_micros, scale_pos)))
    if not tiers_enabled or theo is None:
        return out

    # The far touch for the cross guard, imputed exactly the way `decide` imputes it.
    bid_c, ask_c = _synth_touches(best_bid, best_ask)
    far = ask_c if side == "bid" else bid_c

    # ANCHOR TO RUNG 0 WHILE IT EXISTS, TO THE THEO ONLY WHEN IT DOES NOT -- and that asymmetry is
    # the whole design, arrived at by measurement rather than taste.
    #
    # The first version of this anchored every rung to the theo unconditionally, which looks
    # cleaner and is much worse. Replayed over the 2026-08-25 session through the real
    # `orders.plan_side`, it took the per-order replacement rate from **10.5% to 34.7%** and mean
    # order life from 19s to 6s: `theo +- min_edge` moves on every tick of the theo, so `want[rank]`
    # moves under the deadband on nearly every cycle and the improve rule fires constantly.
    # Rung 0's price comes from the TOUCH, which is sticky -- it only moves when the market moves --
    # and that stickiness is load-bearing on a price-time exchange, where each requote is a queue
    # position given away.
    #
    # So the touch anchor is kept exactly as it was, and the theo anchor is the FALLBACK that
    # applies only when `decide` gave up the side. That is precisely the case this change exists to
    # fix, and nothing else changes.
    tiers = tier_offsets(scale_pos)
    for rank in range(1, len(counts)):
        need = rung_min_edge_micros(rank, min_edge_micros, scale_pos)
        on_touch = bool(out) and out[0].index == 0
        if on_touch:
            back = tiers[min(rank, len(tiers)) - 1]
            px = round(out[0].price - back * TICK if side == "bid"
                       else out[0].price + back * TICK, 4)
        else:
            px = (_floor_tick(theo - need / MICROS) if side == "bid"
                  else _ceil_tick(theo + need / MICROS))
        # WHICHEVER IS FURTHER FROM THE THEO: the absolute floor, or one tick behind the rung in
        # front. Both halves are load-bearing, and the first draft of this had neither.
        #
        # An order's identity on a side IS its price (see strategy/orders), so two rungs at one
        # price are one order and a lost rung -- measured on a 0.13/0.21 book with theo 0.182,
        # where rung 0 pennied to 0.20 and rank 1's floor rounded to 0.20 as well, quietly
        # shipping 40 contracts of a 60-contract ladder.
        #
        # And rung 0's price comes from `decide` -- QUOTE_OFFSET, the tighten cap, the touch --
        # not from this floor, so on a wide book it can land FURTHER out than rank 1's floor. Same
        # book, bid side: rung 0 capped to 0.15 while rank 1's floor was 0.16, i.e. the back rung
        # sitting INSIDE the front one. `ladder_prices_ordered` exists to catch exactly that.
        #
        # Stepping outward only ever increases the edge, so the `need` gate below still holds.
        if out:
            prev = out[-1].price
            px = min(px, prev - TICK) if side == "bid" else max(px, prev + TICK)
        px = round(px, 4)
        # Each rung is gated independently. A 5-cent-behind bid on a 3-cent market is negative and
        # must be DROPPED, not clamped -- clamping would silently stack rungs on MIN_PRICE.
        if not _tradeable(px):
            continue
        # Never cross, on the same imputed book rung 0 was priced against.
        if far is not None and (px >= far if side == "bid" else px <= far):
            continue
        em = _edge_micros(side, theo, px)
        # Behind a live rung 0 the bar is the pre-existing one -- any positive edge -- so this path
        # is byte-for-byte the old ladder and the A/B against the 2026-08-25 session stays clean.
        # The rank bar applies only to the theo-anchored fallback, where there is no rung 0 to
        # inherit a price from and the requirement is the only thing setting the distance.
        if em < (1 if on_touch else need):
            continue
        out.append(Rung(rank, px, counts[rank], em / MICROS,
                        (f"tier{rank}_{tiers[min(rank, len(tiers)) - 1]}c_behind" if on_touch
                         else f"rank{rank}_min_edge_{need / 10_000:.1f}c"),
                        need))
    return out


def ladder_prices_ordered(side, rungs):
    """True when rungs step monotonically AWAY from the touch. A safety assertion, not a filter."""
    px = [r.price for r in rungs]
    if len(px) < 2:
        return True
    return all(a > b for a, b in zip(px, px[1:])) if side == "bid" \
        else all(a < b for a, b in zip(px, px[1:]))
