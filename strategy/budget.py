"""Write-budget admission: which of the planned orders this cycle can actually afford to send.

WHY THIS EXISTS. Kalshi meters writes at 300 tokens/s with a 900-token bucket, and bills per ITEM
in a batch -- 10 tokens per create, 2 per cancel. The SPX board set alone is ~450 markets (30
brackets plus seven hourly events), so a full three-rung two-sided board is ~2,700 orders and
~27,000 tokens: a minute and a half of budget for one placement, against a 2 s cycle. The board is oversubscribed by more than an order of
magnitude and no amount of pacing changes that. Something has to choose.

Before this module the choice was made by the exchange, badly. `orders.apply` sent every chunk
back-to-back with no accounting; the back half of any large refresh 429s by construction
(`kalshi/client.py`, the note above `retryable`), and `orders.py` had no exception handling at
all, so the KalshiError unwound past `apply` into `run_04`'s blanket `except`. By then the cancels
had gone out and their places had not. That is the wipe-the-side failure the 2026-08-24 risk-limit
fix was written to prevent, reached through the rate limiter instead of the risk gate.

THE TWO RULES.

  1. **Exposure-reducing cancels are never deferred.** `edge_negative` and `no_longer_quoting`
     are the two reasons that stand on their own (`orders.REDUCING_CANCEL_REASONS`), they cost
     2 tokens, and they are the whole point of having a budget: the cheapest thing we can do is
     also the most urgent. If they alone exhaust the bucket, that is an alarm to surface, not a
     decision to make -- so they are admitted even into debt, exactly as the kill path is.

  2. **A replacement cancel is admitted only WITH its place.** `plan_side` emits an `improve_*`
     cancel and its `improved` place as a linked pair (`orders._pair_id`). Admitting the cancel
     alone would unquote the level to save 10 tokens, which is the same error as sending naked
     cancels on a risk violation. `reducing_only` states this contract for the risk path; this
     module states it for the budget path, and they are deliberately the same sentence.

TWO RESOURCES, TWO FUNCTIONS. `admit` rations WRITES within one cycle. `select_markets`, at the
bottom of this file, rations CAPITAL across the session -- and that is the constraint that binds
far harder: a two-sided ladder ties up its face value whatever the fair, so `MAX_RESTING_ORDER_VALUE`
buys a few dozen three-rung markets out of a universe of several hundred. It runs TWO PASSES over
that money -- see `select_markets`. They live together
because both must rank by the SAME number (`R`), or a market could be selected as important and
then perpetually starved of the writes that keep its quote current.

WHAT THIS MODULE IS NOT. It does not decide prices or sizes -- `quoting.decide_ladder` and
`edgemgr` own that. It answers only: which markets do we hold at all, and of the orders planned
for them, which go on the wire now.

Pure and testable -- no network, no clock, no module state. `python -m strategy.test_budget`.
"""
import config
from kalshi.client import CANCEL_TOKEN_COST, ORDER_TOKEN_COST
from strategy import band as bandmod
from strategy.orders import REDUCING_CANCEL_REASONS
from strategy.quoting import MICROS, _edge_micros

# Priority classes, lowest number first. These are the tiers, and the ordering is a claim about
# risk rather than about profit:
#
#   REDUCE   an order resting through our own theo is live money bleeding. Cost 2. Unconditional.
#   BACKOFF  a resting order that no longer clears its required edge but is not yet negative.
#            *** NO PRODUCER TODAY. *** `plan_side` deliberately does not back off -- see its
#            module docstring: "When our side got WORSE ... we do nothing until the edge actually
#            goes negative. Requoting there would be paying queue position to reduce a position
#            we still like." The slot is reserved and ranked so that adding a producer is a
#            strategy decision made in one place, not a scheduling accident made here.
#   IMPROVE  our side got better and the resting price is stale. A cancel+place PAIR, cost 12.
#   TOPUP    a rung we want and do not have. A place alone, cost 10.
#
# IMPROVE outranks TOPUP on purpose. A stale resting order is exposure priced wrongly; a missing
# rung is opportunity not taken. This repo's whole policy -- the unconditional negative-edge
# cancel, `reducing_only`, the asymmetric deadband -- consistently prices the first above the
# second, and a scheduler that inverted it under load would quietly reverse that.
#   COLD     the FIRST rung on a side that has no resting order at all. A place alone, cost 10.
#
# COLD outranks TOPUP and is below IMPROVE, added 2026-08-26. A side with nothing resting is not
# quoted: it earns nothing, it cannot be filled, and it holds no queue position to protect. A side
# already showing two rungs and reaching for a third is strictly better off than that. Before this
# the two were the same class and competed only on `R`, so under a tight budget a market could
# thicken an existing ladder while another sat entirely unquoted for cycle after cycle -- and at
# the four-board universe that is the normal case, not an edge one: measured 2026-08-26, a cold
# start plans ~240 orders against a 900-token bucket refilling at 300/s, i.e. 30 orders a second,
# so the first ~8 seconds are pure rationing and the order of that rationing decides which markets
# are quoted at all.
#
# It stays BELOW improve for the same reason topup does: a stale resting order is exposure priced
# wrongly, and fixing that outranks taking new opportunity.
REDUCE, BACKOFF, IMPROVE, COLD, TOPUP = 0, 1, 2, 3, 4

CLASS_NAME = {REDUCE: "reduce", BACKOFF: "backoff", IMPROVE: "improve",
              COLD: "cold", TOPUP: "topup"}


class Item:
    """One admissible unit of work: a solo cancel, a solo place, or a linked cancel+place pair.

    The PAIR is the unit, not the order. That is the whole design: a unit is admitted or deferred
    whole, so there is no code path that can send half of one.
    """

    __slots__ = ("cls", "cancels", "places", "cost", "urgency", "ticker", "side")

    def __init__(self, cls, cancels=(), places=(), urgency=0.0):
        self.cls = cls
        self.cancels = list(cancels)
        self.places = list(places)
        self.cost = (len(self.cancels) * CANCEL_TOKEN_COST
                     + len(self.places) * ORDER_TOKEN_COST)
        self.urgency = float(urgency)
        row = (self.cancels or self.places)[0]
        self.ticker, self.side = row.get("ticker"), row.get("side")

    def key(self):
        """Sort key: class first, then urgency descending, then ticker for a stable order.

        Ties broken by ticker rather than left to dict order, because a plan that reshuffles
        between cycles for no reason makes a deferral pattern impossible to read in a log.
        """
        return (self.cls, -self.urgency, self.ticker or "", self.side or "")

    def record(self):
        return {"cls": CLASS_NAME[self.cls], "ticker": self.ticker, "side": self.side,
                "cost": self.cost, "urgency": round(self.urgency, 4),
                "n_cancel": len(self.cancels), "n_place": len(self.places)}


def _reduce_urgency(c, theos):
    """Dollars of wrongness resting: |edge| x contracts.

    Not just the edge. A rung of 50 contracts priced a cent through the theo is five times the
    problem a rung of 10 is, and when the bucket is short the 50 should leave first. Falls back to
    contract count when there is no theo -- absence of a theo is exactly the case where we most
    want the order gone, so it must not sort last.
    """
    n = float(c.get("count") or 0)
    theo = (theos or {}).get(c.get("ticker"))
    if theo is None or c.get("price") is None:
        return n
    em = _edge_micros(c["side"], theo, c["price"])
    return abs(min(0, em)) / MICROS * n


def _risk_urgency(rows, em_map, theos):
    """`risk_u` for the market -- `R`'s percentile within its own underlying's calibrated curve.

    Using the risk number rather than inventing a second measure of "how much does this market
    matter" is the point: it already combines delta, vega and theta (`greeks.market_greeks`), it
    is already computed every `EDGE_RECOMPUTE_S`, and it is the same number that decided the
    required edge in the first place. A market whose Greeks say it is dangerous is the market
    whose stale quote should be fixed first.

    `risk_u` AND NOT RAW `R`, for the same reason `select_markets` moved 2026-08-26: R's delta
    term is dollars per one-point index move, so an NDX market at ~29,300 carries ~3.8x fewer
    points for the same proportional move and the two underlyings' R distributions are not
    comparable (SPX R_p99 0.0645 against NDX 0.0361). Ranking deferrals by raw R would hand SPX
    the front of the queue on every cold start for a reason that is a unit artifact. Within one
    underlying `risk_u` is a monotone transform of `R`, so a single-board plan defers in exactly
    the order it did before.

    Falls back to R, then to the contracts at stake, so a plan produced with `--no-edge` -- where
    the curve is flat and there is no percentile -- still ranks.
    """
    t = rows[0].get("ticker")
    d = (em_map or {}).get(t) or {}
    u = d.get("risk_u")
    if u is not None:
        return float(u)
    r = d.get("R")
    if r is not None:
        return float(r)
    return sum(float(x.get("count") or 0) for x in rows) / 100.0


def build_items(to_place, to_cancel, theos=None, em_map=None, resting=None):
    """Group a raw plan into admissible units. Returns [Item], unsorted.

    `resting` is the set of `(ticker, side)` that currently have at least one order on the book.
    A solo place into a side NOT in that set is the first rung on an unquoted side and is
    classified COLD rather than TOPUP. Omitting it is safe and reproduces the pre-2026-08-26
    behaviour exactly -- every solo place is then a TOPUP -- which is what the pure ladder tests
    and the backtest pass.
    """
    places_by_pair, solo_places = {}, []
    for p in to_place:
        pid = p.get("pair")
        if pid:
            places_by_pair.setdefault(pid, []).append(p)
        else:
            solo_places.append(p)

    items, paired_pids = [], set()
    for c in to_cancel:
        if c.get("reason") in REDUCING_CANCEL_REASONS:
            items.append(Item(REDUCE, cancels=[c], urgency=_reduce_urgency(c, theos)))
            continue
        pid = c.get("pair")
        mates = places_by_pair.get(pid) or []
        if not mates:
            # A replacement cancel whose place vanished -- risk-blocked upstream, or a plan built
            # by an older path. It is NOT admitted as a bare cancel; that would unquote the level.
            # Dropping it here is safe: the order stays resting and is re-planned next cycle.
            continue
        paired_pids.add(pid)
        rows = [c] + mates
        items.append(Item(IMPROVE, cancels=[c], places=mates,
                          urgency=_risk_urgency(rows, em_map, theos)))

    # NOTE what is deliberately absent here: a fallback that admits an orphaned paired PLACE --
    # one whose cancel disappeared -- as a solo. Placing it without the cancel would double the
    # rung, because the order it was meant to replace is still resting. Such a place is carried
    # only in `places_by_pair`, never in `solo_places`, so leaving it unclaimed drops it. Same
    # reasoning as the orphaned cancel above, opposite side.
    del paired_pids
    # `None` means the caller did not say, and the honest reading of that is TOPUP -- claiming a
    # side is cold on no evidence would promote every solo place above every real top-up. An
    # EMPTY set is different and means what it says: nothing is resting anywhere, so every solo
    # place is a first rung.
    known = resting is not None
    resting = set(resting or ())
    for p in solo_places:
        cold = known and (p.get("ticker"), p.get("side")) not in resting
        items.append(Item(COLD if cold else TOPUP, places=[p],
                          urgency=_risk_urgency([p], em_map, theos)))
    return items


def admit(to_place, to_cancel, tokens, theos=None, em_map=None, reserve=None, resting=None):
    """(place, cancel, detail) -- the subset of the plan this cycle can afford, in priority order.

    `tokens` is what `kalshi.client.write_bucket.available()` says right now. `reserve` holds back
    a floor for the reducing cancels of the NEXT cycle, because the one thing worse than deferring
    a requote is arriving at a negative-edge cancel with an empty bucket.

    REDUCE items are admitted before the reserve is honoured and regardless of the balance --
    rule 1. Everything else spends what is left above the reserve, in `Item.key()` order, and a
    unit that does not fit is deferred WHOLE.

    Note this is greedy in priority order, not a knapsack. Deliberately: a knapsack would admit
    three cheap low-urgency items over one expensive urgent one, which is precisely the trade this
    module exists to refuse.
    """
    reserve = config.WRITE_RESERVE_TOKENS if reserve is None else reserve
    items = sorted(build_items(to_place, to_cancel, theos, em_map, resting=resting),
                   key=Item.key)

    place, cancel, admitted, deferred = [], [], [], []
    budget = float(tokens)
    spent_reduce = 0.0

    for it in items:
        if it.cls == REDUCE:
            cancel.extend(it.cancels)
            admitted.append(it)
            budget -= it.cost
            spent_reduce += it.cost
            continue
        if it.cost > budget - reserve:
            deferred.append(it)
            continue
        budget -= it.cost
        cancel.extend(it.cancels)
        place.extend(it.places)
        admitted.append(it)

    detail = {
        "tokens_in": round(float(tokens), 1),
        "tokens_left": round(budget, 1),
        "reserve": reserve,
        "cost": round(float(tokens) - budget, 1),
        "reduce_cost": round(spent_reduce, 1),
        # The alarm: reducing cancels alone outran the bucket. Nothing was refused -- they went --
        # but we are now in debt to the exchange and the next 429 is ours.
        "reduce_over_budget": spent_reduce > float(tokens),
        "n_admitted": len(admitted),
        "n_deferred": len(deferred),
        "deferred_cost": round(sum(i.cost for i in deferred), 1),
        "by_class": {CLASS_NAME[c]: sum(1 for i in admitted if i.cls == c)
                     for c in (REDUCE, BACKOFF, IMPROVE, COLD, TOPUP)},
        "deferred_by_class": {CLASS_NAME[c]: sum(1 for i in deferred if i.cls == c)
                              for c in (REDUCE, BACKOFF, IMPROVE, COLD, TOPUP)},
        # What we gave up first, so a log reader can see whether the budget is merely tight or
        # structurally short. Silent truncation reads as "we did everything".
        "deferred_sample": [i.record() for i in deferred[:5]],
    }
    return place, cancel, detail


# =========================================================================
# CAPITAL SELECTION -- which markets get a ladder at all.
#
# This is the OTHER scarce resource, and it binds roughly seventeen times harder than the write
# budget above. `admit` rations the orders we send in one cycle; this rations the board we are
# willing to hold at all. Both live here because they are the same shape of decision -- rank by
# what matters, spend until the resource runs out, and say what was given up -- and because they
# must rank by the SAME number, or a market could be selected as important and then perpetually
# starved of the writes that would keep its quote current.
#
# The arithmetic: a two-sided ladder ties up its full face value at ANY fair, because a bid at p
# risks p and an ask at p risks 1-p. So MAX_RESTING_ORDER_VALUE / (face value per market) is how
# many markets can be held, and it is a small fraction of the several-hundred-market universe.
# =========================================================================

# Strike types that resolve to a single boundary -- "K or above", "K or below". The whole hourly
# ladder (KXINXU, KXNASDAQ100U) is `greater_or_equal`. `between` is the 4pm bracket board.
# Keys match `pricing.theo.STRIKE_TYPE_RULE`; anything absent from both sets is unpriceable
# upstream and never reaches here.
ONE_SIDED_TYPES = frozenset(("greater", "greater_or_equal", "less", "less_or_equal"))
BRACKET_TYPES = frozenset(("between",))


def ladder_value(fair, counts=None, sides=("bid", "ask")):
    """Collateral one market's ladder ties up, in dollars.

    Delegates to `orders.order_value` rather than restating `p` vs `1-p`: that asymmetry is the
    single most repeated source of error in this codebase's risk arithmetic, and it belongs in
    one place.
    """
    from strategy.orders import order_value
    counts = counts or config.QUOTE_TIER_COUNTS
    return sum(order_value(s, fair, c) for s in sides for c in counts)


def moneyness_skip(strike_type, fair):
    """The operator-named floor: a reason string when this market is too far from the money, else
    None.

    Deliberately expressed on the FAIR rather than on R, even though R mostly agrees. Two reasons:
    R is unavailable for a market the edge manager could not price, and a stated boundary is
    auditable in a way an emergent one is not -- "we do not quote below this fair" is a sentence an
    operator can check, "it ranked 47th by R" is not.
    """
    if fair is None:
        return "no_fair"
    if strike_type in BRACKET_TYPES:
        if fair <= config.SELECT_FLOOR_BRACKET:
            return f"bracket_fair<={config.SELECT_FLOOR_BRACKET}"
        return None
    if strike_type in ONE_SIDED_TYPES:
        lo = config.SELECT_FLOOR_ONE_SIDED
        if fair <= lo or fair >= 1.0 - lo:
            return f"one_sided_fair_outside_{lo}-{1 - lo}"
        return None
    return None


def _fill(cands, incumbents, max_value, max_contracts, hysteresis):
    """Greedily take from an R-ranked candidate list until either cap binds.

    `cands` is [(R, ticker, value, contracts)], already sorted best-first. Returns
    (chosen, value, contracts, cut_R, keep_R, n_retained).

    The cut is computed on a pure R ranking that IGNORES incumbency, and only then is the softened
    bar applied -- otherwise the incumbents would set their own boundary and could never be
    displaced by a genuinely better market.
    """
    cut_r, val, n = 0.0, 0.0, 0.0
    for r, _t, v, c in cands:
        if val + v > max_value or n + c > max_contracts:
            break
        val += v
        n += c
        cut_r = r
    keep_r = cut_r * float(hysteresis)

    chosen, value, contracts = [], 0.0, 0.0

    def _take(t, v, c):
        nonlocal value, contracts
        if value + v > max_value or contracts + c > max_contracts:
            return False
        chosen.append(t)
        value += v
        contracts += c
        return True

    for r, t, v, c in cands:
        if t in incumbents and r >= keep_r:
            _take(t, v, c)
    n_retained = len(chosen)
    held = set(chosen)
    for r, t, v, c in cands:
        if t in held:
            continue
        if not _take(t, v, c):
            break
    return chosen, value, contracts, cut_r, keep_r, n_retained


def select_markets(markets, theos, em_map, counts=None, incumbents=(),
                   max_value=None, max_contracts=None, hysteresis=None,
                   band_fraction=None):
    """(chosen_tickers, detail) -- the markets we can afford to hold a ladder in.

    Ranked by `risk_u`, the percentile of `R` within the market's OWN underlying's calibrated
    curve. A market whose Greeks say its price matters is the market worth spending capital on --
    and `risk_u` is how that question is asked once more than one underlying is on the board,
    because raw `R` is not comparable between them. See the long note at the ranking key. Within a
    single underlying the two orderings are identical, so nothing about a one-board session
    changed when this moved on 2026-08-26.

    `R` remains what `admit` ranks requotes by and what the edge curve is calibrated against;
    only the cross-market capital ordering reads `risk_u`. The `cut_R` / `keep_R` figures in the
    returned detail are therefore in `risk_u` units too.

    TWO PASSES, and the split is the whole point. `counts` is a `{ticker: counts}` map (see
    `strategy/band`): a band market quotes three rungs a side, a wing market quotes one and costs a
    third as much. Band markets are BY CONSTRUCTION the high-R ones,
    so a single undivided R ranking funds them first and exhausts the capital before reaching a
    single wing -- and the 1-tier breadth this design exists to buy would never be funded at all.
    So band markets get `band_fraction` of the caps and wings get whatever is left, including
    anything the band pass did not spend.

    HYSTERESIS, because the naive version churns. R is recomputed every `EDGE_RECOMPUTE_S`, and a
    market sitting near the cut would flip in and out of the set on every recompute. Each flip is a
    ladder cancelled and re-placed -- write tokens, and worse, every queue position on it, on a
    price-time exchange where queue position is most of what a resting order is worth. So an
    incumbent is retained while its R holds at `hysteresis` of the marginal entrant's, and only
    genuinely better markets displace it. The two passes carry their own boundaries: a band market
    is not displaced by a wing market, because they are not competing for the same money.

    Returns the ticker set, not a plan. `build_desired` does the pricing; this only says who.
    """
    max_value = config.MAX_RESTING_ORDER_VALUE if max_value is None else max_value
    max_contracts = (config.MAX_RESTING_CONTRACTS_TOTAL if max_contracts is None
                     else max_contracts)
    hysteresis = config.SELECT_HYSTERESIS if hysteresis is None else hysteresis
    band_fraction = config.BAND_CAPITAL_FRACTION if band_fraction is None else band_fraction
    incumbents = set(incumbents)

    band_c, wing_c, skipped = [], [], {}
    for m in markets:
        t = m.get("ticker")
        fair = (theos or {}).get(t)
        why = moneyness_skip(m.get("strike_type"), fair)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        d = (em_map or {}).get(t) or {}
        if not d.get("quotable", True):
            skipped["not_quotable"] = skipped.get("not_quotable", 0) + 1
            continue
        r = d.get("R")
        if r is None:
            # A market with no R is not quoted at all -- edgemgr's rule since 2026-08-23, and it
            # is repeated here so selection never hands one downstream to be dropped silently.
            skipped["no_R"] = skipped.get("no_R", 0) + 1
            continue
        # THE RANKING KEY IS `risk_u`, NOT RAW `R`, AND ONLY BECAUSE THERE ARE TWO UNDERLYINGS.
        #
        # `R` is dollars per contract, but its DELTA term is dollars per one-point move in the
        # index, so the same proportional move is worth about 3.8x fewer points on NDX at ~29,300
        # than on SPX at ~7,700. Vega and theta carry no such factor. The result is that the two
        # calibrated curves live on different scales -- measured 2026-08-26, SPX R_p99 0.0645 /
        # R_max 0.2849 against NDX R_p99 0.0361 / R_max 0.0950 -- and a single raw-R ranking is
        # not a comparison at all, it is a unit artifact. Measured on a four-board dry session
        # the same day: NDX took 52 of 17,331 quoted rungs, 0.3%, while listing 2,430 of the
        # 2,820 markets.
        #
        # `risk_u` is R's percentile WITHIN ITS OWN UNDERLYING'S CURVE (`pricing/edge.risk_u`),
        # so it says "how much does this market's price matter, for this board" -- which is what
        # the raw-R ranking was always reaching for. Two properties make the swap safe:
        #
        #   1. WITHIN one underlying `risk_u` is a monotone transform of `R`, so a single-board
        #      session ranks EXACTLY as it did before. This is not a behaviour change for SPX.
        #   2. It is unitless and bounded, so the comparison across underlyings is like-for-like
        #      rather than an accident of where the index happens to trade.
        #
        # `R` is still what `admit` ranks requotes by and still what the curve is calibrated
        # against; only the cross-market CAPITAL ordering moves. Fall back to R when a market has
        # no `risk_u` -- a flat curve produces none, and under `--no-edge` every market is flat,
        # which is the pre-calibration ordering and the right one there.
        rank = d.get("risk_u")
        rank = float(r) if rank is None else float(rank)
        cn = bandmod.resolve_counts(counts, t)
        row = (rank, t, ladder_value(fair, cn), 2.0 * sum(float(c) for c in cn))
        (band_c if len(cn) > 1 else wing_c).append(row)

    band_c.sort(key=lambda x: (-x[0], x[1]))
    wing_c.sort(key=lambda x: (-x[0], x[1]))

    # The fraction exists to RESERVE money for wings, so with no wing candidates there is nothing
    # to reserve it for and the band pass gets the whole cap. Without this guard a board with
    # uniform counts -- which is what every pure ladder test passes, and what a `--no-band` run
    # would produce -- would leave 40% of the capital idle and quote a board 40% smaller than it
    # could afford, silently.
    #
    # The opposite case needs no guard: a thin band late in the day, when vol has decayed and
    # little is near the money, underspends its fraction and the wing pass is handed
    # `max_value - b_val` rather than the complement of the fraction.
    bv = max_value * band_fraction if wing_c else max_value
    bc = max_contracts * band_fraction if wing_c else max_contracts
    b_sel, b_val, b_con, b_cut, b_keep, b_ret = _fill(
        band_c, incumbents, bv, bc, hysteresis)
    # The wing pass gets everything the band pass did not spend, not just the complement of the
    # fraction -- a thin band (late in the day, when vol has decayed and few markets are near the
    # money) should buy more breadth, not leave the money idle.
    w_sel, w_val, w_con, w_cut, w_keep, w_ret = _fill(
        wing_c, incumbents, max_value - b_val, max_contracts - b_con, hysteresis)

    chosen = set(b_sel) | set(w_sel)
    detail = {"universe": len(markets), "candidates": len(band_c) + len(wing_c),
              "chosen": len(chosen),
              "band": {"candidates": len(band_c), "chosen": len(b_sel), "retained": b_ret,
                       "value": round(b_val, 2), "budget": round(bv, 2),
                       "cut_R": round(b_cut, 6), "keep_R": round(b_keep, 6),
                       "unaffordable": max(0, len(band_c) - len(b_sel))},
              "wing": {"candidates": len(wing_c), "chosen": len(w_sel), "retained": w_ret,
                       "value": round(w_val, 2), "budget": round(max_value - b_val, 2),
                       "cut_R": round(w_cut, 6), "keep_R": round(w_keep, 6),
                       "unaffordable": max(0, len(wing_c) - len(w_sel))},
              "retained": b_ret + w_ret, "added": len(chosen) - (b_ret + w_ret),
              "dropped_incumbents": len(incumbents - chosen),
              "value": round(b_val + w_val, 2), "max_value": max_value,
              "contracts": b_con + w_con, "max_contracts": max_contracts,
              # Every market we are NOT quoting, by reason. Silent truncation of a several-hundred
              # market universe down to a few dozen would read as "we quoted the board".
              "skipped": skipped,
              "unaffordable": max(0, len(band_c) + len(wing_c) - len(chosen))}
    return chosen, detail
