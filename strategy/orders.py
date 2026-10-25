"""Resting-order manager: per (market, side) ladder state, and the smallest change that is right.

THE STATE. One `MarketState` per Kalshi market: the theo we last priced it at, plus up to three
live orders on each side -- seven things, which is the whole picture for that market. Orders are
held best-first, and their identity is their PRICE, not a slot index: a rung that survives keeps
its order id, its size and, crucially, its queue position.

THE POLICY. Kalshi is price-time and any price amendment forfeits queue position (only a size
DECREASE preserves it). So the default is to leave an order alone, and there are exactly three
reasons to touch one:

  1. **Negative edge -- cancel, always, no threshold.** We are quoting through our own theo. This
     is not a "has it drifted enough" question.
  2. **Our side got better -- improve, on a tiered threshold.** When the theo moves our way our
     resting orders are too passive. The FRONT order moves if it can gain >= 1 tick and still meet
     its edge requirement; the back two move only for >= 2 ticks, because they are further from
     the action and their queue position is worth more than a penny of price. And for
     `config.QUOTE_ZOMBIE_HOLD_S` after a negative-edge cancel on the side, it does not move at
     all unless somebody has posted inside us -- see `zombie_held`.
  3. **We are short a level -- top up.** If a cancel left us with two orders where we want three,
     the new one goes BEHIND the worst survivor rather than re-deriving the whole ladder. That is
     the difference between adding an order and throwing two good queue positions away.

When our side got WORSE -- the theo moved against us, so our resting order is now more aggressive
than we would newly quote -- we do nothing until the edge actually goes negative. Requoting there
would be paying queue position to reduce a position we still like.
"""
import time
import uuid

import config
from kalshi import client as k
# Imported by NAME, not reached through `k`. These are protocol constants and a type, not
# client behaviour, so they must survive the module swap the order tests use as their seam
# (`test_orders._FakeK`) -- a stub that had to restate the token costs to be installed could
# restate them wrongly, and a stub that had to restate the exception class could not be
# caught by the handler that exists to catch it.
from kalshi.client import CANCEL_TOKEN_COST, KalshiError, ORDER_TOKEN_COST
from strategy import band as bandmod
from strategy.quoting import MICROS, TICK, _edge_micros, _tradeable

PLACE, CANCEL, KEEP = "place", "cancel", "keep"

# Tunable. Illustrative value; the live value is not published.
FRONT_IMPROVE_TICKS = 2
# Requote deadband for the back rungs. Illustrative value; the live value is not published.
BACK_IMPROVE_TICKS = 3


def _best_first(side, orders):
    return sorted(orders, key=lambda o: (-o["price"] if side == "bid" else o["price"]))


def _pair_id(ticker, side, price):
    """The key linking a replacement cancel to its place. See the call site in `plan_side`."""
    return f"{ticker}|{side}|{float(price):.4f}"


def order_value(side, price, count):
    """Dollars of collateral one resting order ties up.

    Not `price * count` on both sides. Kalshi collateralises a binary at what the order could
    LOSE, and a resting offer is a NO buy in disguise: selling YES at 0.95 risks 0.05, not 0.95.
    Using price on both sides would read a book that is mostly offers as many times more
    expensive than it is, and a book that is mostly deep bids as cheaper.
    """
    if price is None or count is None:
        return 0.0
    p = float(price)
    return (p if side == "bid" else 1.0 - p) * float(count)


class MarketState:
    """Theo plus the live ladder on each side, for one market."""

    __slots__ = ("ticker", "theo", "bid", "ask", "updated_ns", "position",
                 "fill_bid", "fill_ask",
                 "best_bid", "best_ask", "zombie_bid", "zombie_ask",
                 "zombie_bid_touch", "zombie_ask_touch")

    def __init__(self, ticker):
        self.ticker = ticker
        self.theo = None
        self.bid = []
        self.ask = []
        # When this side last traded, monotonic. Feeds `pricing/lean.backoff_cents`, which taxes
        # the ENTRY bar for a side we have just been filled on so one adverse move cannot be sold
        # to five times in ten seconds. None means clean.
        self.fill_bid = None
        self.fill_ask = None
        self.updated_ns = None
        # The touch the ladder was last priced against, stamped by `run_04.cycle` in the same pass
        # that stamps `theo`. Both must come from one place: the zombie hold's exception asks
        # whether somebody is inside us, and answering it off a different book from the one that
        # produced `want` would lift the hold for a market state that never existed.
        self.best_bid = None
        self.best_ask = None
        # Monotonic seconds at which we last cancelled an order on this side for NEGATIVE EDGE.
        # None means never. See `config.QUOTE_ZOMBIE_HOLD_S` for what it suppresses and why it is
        # on the entry rather than the exit.
        self.zombie_bid = None
        self.zombie_ask = None
        # Where the touch stood when that clock was armed. The hold's exception is "the touch has
        # MOVED since we left", and it needs a before to compare against.
        self.zombie_bid_touch = None
        self.zombie_ask_touch = None
        # NET contracts held, signed: positive is long YES, negative is long NO. Fed from the WS
        # `fill` frames' `post_position_fp` and refreshed by `market_position` frames -- see
        # `kalshi/ws.MarketFeed`. 0.0 rather than None on a cold start, because a market we have
        # never traded is flat, and an inventory gate that read "unknown" as "refuse" would refuse
        # the entire board on the first cycle.
        self.position = 0.0

    def side(self, s):
        return self.bid if s == "bid" else self.ask

    def touch(self, s):
        return self.best_bid if s == "bid" else self.best_ask

    def set_touch(self, best_bid, best_ask):
        self.best_bid, self.best_ask = best_bid, best_ask

    def last_fill(self, s):
        return self.fill_bid if s == "bid" else self.fill_ask

    def mark_fill(self, s, now):
        if s == "bid":
            self.fill_bid = now
        else:
            self.fill_ask = now

    def zombie(self, s):
        return self.zombie_bid if s == "bid" else self.zombie_ask

    def zombie_at_touch(self, s):
        return self.zombie_bid_touch if s == "bid" else self.zombie_ask_touch

    def arm_zombie(self, s, now):
        """Start the hold, and remember where the touch stood when we left."""
        if s == "bid":
            self.zombie_bid, self.zombie_bid_touch = now, self.best_bid
        else:
            self.zombie_ask, self.zombie_ask_touch = now, self.best_ask

    def set_side(self, s, orders):
        if s == "bid":
            self.bid = orders
        else:
            self.ask = orders

    def contracts(self, s=None):
        sides = ("bid", "ask") if s is None else (s,)
        return sum(float(o.get("count") or 0) for x in sides for o in self.side(x))

    def value(self, s=None):
        sides = ("bid", "ask") if s is None else (s,)
        return sum(order_value(x, o.get("price"), o.get("count"))
                   for x in sides for o in self.side(x))

    def as_record(self):
        return {"ticker": self.ticker, "theo": self.theo,
                "bid": [(o["price"], o["count"]) for o in _best_first("bid", self.bid)],
                "ask": [(o["price"], o["count"]) for o in _best_first("ask", self.ask)]}


class RestingBook:
    """Every market's ladder state."""

    def __init__(self):
        self._m = {}

    def state(self, ticker):
        if ticker not in self._m:
            self._m[ticker] = MarketState(ticker)
        return self._m[ticker]

    def resting_tickers(self):
        """Markets that actually hold an order.

        NOT `tickers()`, which returns every market a `MarketState` was ever created for -- and
        `run_04.cycle` creates one for EVERY market on the board each cycle when it stamps the
        theo and the touch. Measured 2026-08-26: `tickers()` returned 1,440 while about forty held
        orders, which had the fast guard re-pricing the whole board at 200Hz (23.1ms a tick
        against a 5ms budget) and stealing enough CPU to push the main cycle's p95 to 1,444ms.
        """
        return [t for t, st in self._m.items() if st.bid or st.ask]

    def resting_sides(self):
        """{(ticker, side)} that currently hold at least one order.

        What `budget.admit` needs to tell a first rung on an unquoted side from a third rung on a
        side already showing two. Cheap: the book is a few hundred entries and this runs once a
        cycle.
        """
        return {(t, s) for t in self.resting_tickers() for s in ("bid", "ask")
                if self._m[t].side(s)}

    def tickers(self):
        return list(self._m)

    def all_orders(self):
        return [o for st in self._m.values() for s in ("bid", "ask") for o in st.side(s)]

    def contracts(self, ticker=None, side=None):
        if ticker is not None:
            return self._m[ticker].contracts(side) if ticker in self._m else 0.0
        return sum(st.contracts(side) for st in self._m.values())

    def value(self, ticker=None, side=None):
        if ticker is not None:
            return self._m[ticker].value(side) if ticker in self._m else 0.0
        return sum(st.value(side) for st in self._m.values())

    def __len__(self):
        return len(self.all_orders())


def zombie_held(side, touch, zombie_at, zombie_touch, now, hold_s=None):
    """True when an IMPROVE on this side must wait because we just left a level here.

    The full argument is in `config.QUOTE_ZOMBIE_HOLD_S`. In one line: after cancelling an order
    for negative edge, our own theo wobbling back across the same boundary would otherwise requote
    the survivor a tick better and pay a queue position for a market that never moved.

    THE EXCEPTION IS THAT THE TOUCH HAS *MOVED* SINCE WE LEFT -- not that the touch is currently
    inside our best resting order, which is what this asked first and is measurably wrong.
    Cancelling our own front rung is itself what makes the touch look "inside" us: our best
    resting price drops a tick the moment the order leaves, so a static book reads as having
    pennied us. Measured over the 2026-08-25 session, that test lifted the hold on **56.6% of the
    re-entries it exists to block** and on only 38.7% of the honest chases -- it selected FOR the
    pathology. Comparing the touch against where it stood when the clock was armed asks the
    question the operator actually posed: did somebody come in, or did only our own theo move?

    Returns False whenever it cannot tell (no clock, no cancel on record, hold disabled, no touch
    recorded at arming), because the pre-2026-08-26 behaviour is "improve" and an unknown must not
    silently change the policy.
    """
    if zombie_at is None or now is None:
        return False
    hold = config.QUOTE_ZOMBIE_HOLD_S if hold_s is None else hold_s
    if hold <= 0 or (now - zombie_at) >= hold:
        return False
    if touch is not None and zombie_touch is not None:
        moved_in = (touch > zombie_touch + 1e-9) if side == "bid" \
            else (touch < zombie_touch - 1e-9)
        if moved_in:
            return False
    return True


def plan_side(ticker, side, theo, rungs, live, counts,
              front_ticks=FRONT_IMPROVE_TICKS, back_ticks=BACK_IMPROVE_TICKS,
              touch=None, zombie_at=None, zombie_touch=None, now=None):
    """(cancels, places, keeps) for one (market, side).

    `rungs` is the freshly-decided ladder (best-first); `live` is what we already have out.
    Returns plans only -- nothing is sent here, so the whole policy is testable without a network.

    `touch`, `zombie_at`, `zombie_touch` and `now` are the zombie hold's inputs and are ARGUMENTS
    rather than
    lookups so this stays a pure function: `reconcile` reads them off `MarketState` and the replay
    harness drives them off journal timestamps. Omitting them disables the hold, which is the
    pre-2026-08-26 behaviour and what the pure ladder tests want.
    """
    cancels, places, keeps = [], [], []
    live = _best_first(side, live)
    want = [r.price for r in rungs]
    # Evaluated once for the side, not per order: `live` and the touch do not change inside the
    # loop, and the hold is a property of the (market, side), not of any one rung.
    held_back = zombie_held(side, touch, zombie_at, zombie_touch, now)

    # `rank` is the order's position among the ones that will REMAIN, not its position in the
    # live list, and the difference is a real bug that shipped. Enumerating `live` matched
    # want[i] to the i-th LIVE order, so a front rung cancelled for negative edge left the
    # second-best order measured against want[1] and the third against want[2] -- want[0], the
    # rung that competes for the touch, was then never placed at all. The ladder walked
    # backwards a rung every time the front was cancelled, and the top-up below refilled at the
    # BACK, so it never walked forward again. The deadband tier follows rank for the same
    # reason: whichever order is now in front is the one worth a single tick.
    rank = 0
    for o in live:
        e = _edge_micros(side, theo, o["price"]) if theo is not None else -1
        if e <= 0:
            # Reason 1: through our own theo. UNCONDITIONAL and untimed -- we never hold an order
            # we believe is losing money, and no hysteresis is allowed above this line.
            #
            # No exit bar above zero: tighter exits were tried and rejected on measured replacement
            # rate and markouts. Adverse selection on a resting quote is answered by a better entry
            # price, or by not quoting that market, not by a tighter exit.
            #
            # Rank is NOT advanced -- this order is leaving, so it does not consume a slot.
            cancels.append({**o, "reason": "edge_negative"})
            continue
        # `rank` is the slot this order would take IF it survives, so an order behind one that is
        # leaving is measured against the vacated slot's target and deadband rather than its own.
        # THAT IS THE PROMOTION: cancel the front rung and the next one is immediately judged as a
        # front rung. Nothing moves on the exchange -- the survivor keeps its queue position at its
        # own price, which is the entire point of promoting instead of re-deriving the ladder.
        target = want[rank] if rank < len(want) else None
        rank += 1
        if target is None:
            keeps.append({**o, "reason": "no_target_for_slot"})
            continue
        # Reason 2: our side got better. `gain` is positive only when the new target is MORE
        # aggressive than what we have out; a target behind our resting price means the theo moved
        # against us, and we sit tight until the edge actually goes negative.
        gain = (target - o["price"]) if side == "bid" else (o["price"] - target)
        thresh = (front_ticks if rank == 1 else back_ticks) * TICK
        if gain >= thresh - 1e-9 and held_back:
            # THE ZOMBIE HOLD. We would improve, and we are inside the window that follows a
            # negative-edge cancel on this side. Sit still instead: the order we are holding
            # carries MORE edge than the one we would newly quote (that is what `gain` being
            # positive means), and it already has queue position. Costs fill rate, never money.
            keeps.append({**o, "reason": "zombie_hold"})
            continue
        if gain >= thresh - 1e-9:
            # `pair` links this cancel to the place that replaces it, and the link has to be
            # EXPLICIT rather than inferred from list position. `strategy/budget` must be able to
            # admit or defer the two together: sending the cancel without its place is not
            # caution, it is unquoting the level -- the same failure `reducing_only` exists to
            # prevent on the risk path, arriving instead through the write budget.
            #
            # The key is (ticker, side, the price being replaced), which is unique because an
            # order's identity on a side IS its price -- see the module docstring. No counter,
            # so it survives being planned and admitted in separate passes.
            pid = _pair_id(ticker, side, o["price"])
            cancels.append({**o, "reason": f"improve_{gain / TICK:.0f}t", "pair": pid})
            places.append({"ticker": ticker, "side": side, "price": target,
                           "count": o["count"], "reason": "improved", "pair": pid})
        else:
            keeps.append({**o, "reason": "within_deadband" if gain < thresh else "no_gain"})

    # Reason 3: top up to a full ladder, BEHIND the worst thing we are keeping.
    #
    # "Behind", not "at the ladder price", is the point. If a cancel left two good orders with
    # queue position, re-deriving all three from the fresh ladder would throw both away to gain a
    # penny on each. Adding one at the back keeps them and restores the depth.
    held = [kp["price"] for kp in keeps] + [pl["price"] for pl in places]
    step = _tier_step(want)

    # The top-up is sized against what is LEFT of the decided ladder, not against the slot it
    # happens to land in. A survivor keeps the size it was placed with, so after the front rung
    # is cancelled the side holds 20 + 50 = 70 of an 80-contract ladder while `slot` reads 2 and
    # would have asked for another 50 -- 120 contracts against a
    # MAX_RESTING_CONTRACTS_PER_MARKET_SIDE of exactly 80. The cap is one ladder, so the refill
    # is one ladder minus what is already out. On a cold start the budget is the whole ladder and
    # every rung gets its full size, which is the unchanged case.
    budget = sum(float(r.count) for r in rungs)
    on_side = sum(float(kp.get("count") or 0) for kp in keeps) + \
              sum(float(pl.get("count") or 0) for pl in places)

    while len(held) < len(want):
        slot = len(held)
        remaining = budget - on_side
        if remaining <= 0:
            break                               # the ladder's size is already resting
        if not held:
            px = want[slot]                     # cold start: use the ladder as decided
        else:
            back = min(held) if side == "bid" else max(held)
            px = round(back - step if side == "bid" else back + step, 4)
        if not _tradeable(px) or _edge_micros(side, theo, px) <= 0:
            break                               # nothing further back can be better; stop
        if any(abs(px - h) < 1e-9 for h in held):
            break
        # Counts are Kalshi FIXED-POINT STRINGS ("10.00"), and they go into the order body
        # verbatim -- so the untrimmed case must hand back the exact element from `counts`, not a
        # float that happens to equal it. Only a trimmed rung is re-formatted, and in the same
        # 2dp spelling `config.ORDER_COUNT` uses.
        full = counts[min(slot, len(counts) - 1)]
        n = float(full)
        if n <= remaining + 1e-9:
            cnt = full
        else:
            n = remaining
            cnt = f"{n:.2f}"
        places.append({"ticker": ticker, "side": side, "price": px, "count": cnt,
                       "reason": "cold_start" if slot == 0 else "top_up"})
        held.append(px)
        on_side += n
    return cancels, places, keeps


def _tier_step(want):
    """Spacing to use when adding behind the survivors: the ladder's own gap, else one tick."""
    if len(want) >= 2:
        gaps = [abs(a - b) for a, b in zip(want, want[1:])]
        gaps = [g for g in gaps if g > 1e-9]
        if gaps:
            return min(gaps)
    return TICK


def apply_fill(book, fill, now=None):
    """Reduce the resting order a `fill` frame names. Returns 1 if it matched, 0 if not.

    THE HOLE THIS CLOSES. `RestingBook` is written from create responses (`apply._record`) and,
    until 2026-08-25, was never reconciled against the exchange during a session -- `cancel_all`
    at exit was the first and only re-read. So a filled order stayed in local state forever: it
    kept being measured for edge by `plan_side`, kept consuming its share of
    `MAX_RESTING_CONTRACTS_PER_MARKET_SIDE`, kept blocking the top-up that should have replaced
    it, and would eventually be cancelled by an order_id the exchange had already retired.

    MATCHED BY `order_id`, NOT BY PRICE. Price is an order's identity within a side for planning
    purposes (see the module docstring), but a fill is an exchange event and carries the id, which
    is unambiguous. A partial fill decrements; a full one removes.

    A fill that matches nothing returns 0 rather than raising. That is the normal case for a
    market we stopped quoting between the fill and the drain, and for anything the account trades
    outside this loop -- neither is an error here.
    """
    oid = fill.get("order_id")
    if not oid:
        return 0
    now = time.monotonic() if now is None else now
    try:
        n = float(fill.get("count_fp") or 0)
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0

    tickers = [fill["market_ticker"]] if fill.get("market_ticker") else book.tickers()
    for t in tickers:
        st = book.state(t)
        for side in ("bid", "ask"):
            rows = st.side(side)
            for o in rows:
                if o.get("order_id") != oid:
                    continue
                left = float(o.get("count") or 0) - n
                if left > 1e-9:
                    # Kalshi carries counts as fixed-point STRINGS and they go into an order body
                    # verbatim, so the remainder is re-spelled in the same 2dp form the ladder
                    # uses rather than left as a float.
                    o["count"] = f"{left:.2f}"
                else:
                    st.set_side(side, [x for x in rows if x is not o])
                # THE BACKOFF CLOCK, armed here and only here: this is the one place that knows a
                # fill actually landed on a side we were quoting. It arms on a PARTIAL fill too --
                # we traded, whatever the remainder says. A fill we cannot match (a market we
                # stopped quoting, anything traded outside this loop) arms nothing, which is
                # right: there is no resting quote of ours to back off from.
                st.mark_fill(side, now)
                return 1
    return 0


def reconcile(book, desired, counts=None, now=None):
    """(to_place, to_cancel, kept) across every market. `desired` is {(ticker, side): [Rung]}.

    `counts` is either one tuple for the whole board or a `{ticker: counts}` map -- since
    2026-08-25 a market outside the quoting band gets one rung and a market inside it gets three,
    so the sizes are a per-market property. Note the rung COUNT is carried by `desired` (which
    `decide_ladder` already truncated); `counts` only supplies the size for each slot, and the two
    must agree or the top-up below refills a wing market to three rungs.

    `now` is a `time.monotonic()` reading -- the zombie hold's clock -- and is read once for the
    whole board so every market is judged against the same instant. Leave it None outside tests
    and the replay harness.
    """
    to_place, to_cancel, kept = [], [], []
    seen = set()
    now = time.monotonic() if now is None else now
    for (ticker, side), rungs in desired.items():
        st = book.state(ticker)
        seen.add((ticker, side))
        c, p, kp = plan_side(ticker, side, st.theo, rungs, st.side(side),
                             bandmod.resolve_counts(counts, ticker),
                             touch=st.touch(side), zombie_at=st.zombie(side),
                             zombie_touch=st.zombie_at_touch(side), now=now)
        to_cancel.extend(c)
        to_place.extend(p)
        kept.extend(kp)
    # Anything held on a market/side we no longer want to quote at all.
    for ticker in book.tickers():
        st = book.state(ticker)
        for side in ("bid", "ask"):
            if (ticker, side) not in seen:
                for o in st.side(side):
                    to_cancel.append({**o, "reason": "no_longer_quoting"})
    return to_place, to_cancel, kept


# Cancels that stand on their own. Every other cancel exists only so a matching place can take
# its slot, so sending it without that place is not "being careful" -- it is unquoting a side.
REDUCING_CANCEL_REASONS = frozenset(("edge_negative", "no_longer_quoting"))


def reducing_only(to_cancel):
    """The subset of a cancel plan that is safe to send when the places are BLOCKED.

    `apply` takes cancels and places together and sends the cancels first, so a caller that
    zeroes `to_place` on a risk violation and passes the plan through unchanged sends every
    cancel naked -- including the `improve_*` ones whose only purpose was to be replaced one tick
    better. On a tripped limit that empties the side instead of protecting it. These two reasons
    reduce exposure and are correct to send either way; the rest wait for the next cycle, when
    the order they wanted to replace is still resting and still earning.
    """
    return [c for c in to_cancel if c.get("reason") in REDUCING_CANCEL_REASONS]


def within_risk_limits(book, to_place, to_cancel=()):
    """(ok, violations) -- checked BEFORE anything is sent.

    `to_cancel` is NOT optional in spirit: `apply` sends the cancels first, so what will actually
    rest is `held - cancelled + placed`, and omitting the cancels double-counts every order being
    REPLACED. That is not an edge case. `MAX_RESTING_CONTRACTS_PER_MARKET_SIDE` is exactly one
    3-tier ladder (3 x LOT_SIZE), so a side quoting its full ladder is always AT the cap -- and any
    requote at all then reads as cap + one rung and blocks. Reproduced: one improved front
    rung was enough to block the places while `run_04` still sent the cancels, wiping the side.

    NOTE WHAT THE FIRST CHECK COUNTS: `book.contracts(ticker, side)` is RESTING ORDERS, never
    position. The inventory limit is a different gate in a different place -- `band.position_limit`
    (POSITION_LIMIT_LOTS_BAND / _WING lots) applied in `run_04.build_desired`,
    which stops quoting the side that would ADD to a position already at its limit. This is a cap
    on the book we SHOW; that is a cap on the inventory we CARRY. A cap breach here is never an
    inventory refusal, and reading it as one would hide a ladder that has grown a fourth rung.

    The default is empty so existing callers keep working, and it is the CONSERVATIVE direction --
    it can only over-count, never under-count.
    """
    problems = []

    def _by_side(rows):
        out = {}
        for r in rows:
            key = (r["ticker"], r["side"])
            out[key] = out.get(key, 0.0) + float(r.get("count") or 0)
        return out

    add_side, drop_side = _by_side(to_place), _by_side(to_cancel)
    for key in set(add_side) | set(drop_side):
        t, s = key
        after = book.contracts(t, s) - drop_side.get(key, 0.0) + add_side.get(key, 0.0)
        if after > config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE + 1e-9:
            problems.append(f"{t}/{s}: {after:.0f} > "
                            f"{config.MAX_RESTING_CONTRACTS_PER_MARKET_SIDE}")

    # PER-MARKET COLLATERAL. Without it nothing stops one market absorbing the whole book.
    # MAX_RESTING_ORDER_VALUE_PER_MARKET is exactly one 3-tier two-sided ladder (a bid at p and an
    # ask at p together risk $1 per contract); a 1-tier wing market uses a third of it.
    #
    # `RestingBook.value(ticker)` has existed since the risk limits landed and had no caller.
    # NETTING THE CANCELS is not optional here for the same reason it is not optional above: a
    # market at its cap that requotes would otherwise read as cap + replacement and block, and the
    # caller's response to a block is to send the reducing cancels alone -- which unquotes it.
    def _by_market(rows):
        out = {}
        for r in rows:
            out[r["ticker"]] = out.get(r["ticker"], 0.0) + order_value(
                r["side"], r.get("price"), r.get("count"))
        return out

    add_mkt, drop_mkt = _by_market(to_place), _by_market(to_cancel)
    for t in set(add_mkt) | set(drop_mkt):
        after = book.value(t) - drop_mkt.get(t, 0.0) + add_mkt.get(t, 0.0)
        if after > config.MAX_RESTING_ORDER_VALUE_PER_MARKET + 1e-9:
            problems.append(f"{t}: ${after:,.2f} > "
                            f"${config.MAX_RESTING_ORDER_VALUE_PER_MARKET:,.2f}")

    total = (book.contracts()
             - sum(float(c.get("count") or 0) for c in to_cancel)
             + sum(float(p["count"]) for p in to_place))
    if total > config.MAX_RESTING_CONTRACTS_TOTAL + 1e-9:
        problems.append(f"total {total:.0f} > {config.MAX_RESTING_CONTRACTS_TOTAL}")
    # The only limit that binds on PRICE rather than size. A count cap alone cannot see the
    # difference between 5,000 contracts resting at a penny and 5,000 resting at 90c.
    value = (book.value()
             - sum(order_value(c["side"], c.get("price"), c.get("count")) for c in to_cancel)
             + sum(order_value(p["side"], p.get("price"), p.get("count")) for p in to_place))
    if value > config.MAX_RESTING_ORDER_VALUE + 1e-9:
        problems.append(f"resting value ${value:,.2f} > ${config.MAX_RESTING_ORDER_VALUE:,.2f}")
    return (not problems), problems


def apply(to_place, to_cancel, book, live=False, max_batch=None, budget=None, now=None):
    """Send the plan. Cancels first, then places.

    Cancels lead deliberately: placing first would briefly hold both the old and the new order and
    could breach the per-market-side cap that `within_risk_limits` just checked.

    `budget` is `kalshi.client.write_bucket` (or None to send unmetered, which is what the kill
    path wants). It is a BACKSTOP, not the policy: `strategy/budget.admit` has already chosen what
    fits, and this only catches the case where the bucket drained between the choosing and the
    sending. When it fires, the remaining chunks are LEFT UNSENT rather than pushed into a 429.

    THE FAILURE THIS FUNCTION USED TO HAVE. Every chunk went back-to-back with no accounting and
    no `try`, so a 429 on chunk 3 raised out of `apply`, past `cycle`, into `run_04`'s blanket
    `except Exception` -- after the cancels had gone and before their places had. The side was
    left thin, and the exception was indistinguishable from any other in the cycle stats. At the
    ~180 orders that path was written for it took a burst to trigger; at the 3,192 of the full
    universe it is arithmetic (`client.py`: 1,800 tokens against a 900-token bucket, so the back
    half 429s by construction).

    So: every batch call is caught, a 429 is counted as its own thing, and -- the part that
    matters -- **a place whose paired cancel did not land is dropped**. Otherwise the order it was
    meant to replace is still resting and we have just doubled the rung.
    """
    out = {"cancelled": 0, "placed": 0, "cancel_errors": [], "place_errors": [],
           "dry_run": not live, "rate_limited": 0, "unsent_cancels": 0, "unsent_places": 0,
           "orphaned_places": 0, "zombies_armed": 0, "cancel_gone": 0}
    now = time.monotonic() if now is None else now
    mb = max_batch or (k.write_budget().get("max_batch") or 20)

    # `_send` returns one of four statuses, and the distinction between the first two is the whole
    # reason it is not a bool. A DECLINED chunk never left this process; a REJECTED one reached
    # Kalshi and came back 429. Counting them the same way either double-counts the rejected chunk
    # as also unsent, or loses the declined one entirely -- and the resulting number is the only
    # evidence a log reader has for whether the budget is tight or the exchange is angry.
    OK, DECLINED, REJECTED, FAILED = "ok", "declined", "rejected", "failed"

    def _send(fn, arg, cost, kind):
        """(results, status). Never raises."""
        if budget is not None and not budget.can_afford(cost):
            # Not an error: we chose not to spend into debt. Nothing is recorded in `*_errors`,
            # and that absence is what distinguishes "we declined" from "Kalshi refused".
            # Debt is reserved for the kill path, which passes no budget at all.
            return [], DECLINED
        try:
            return fn(arg), OK
        except KalshiError as e:
            # THE BODY IS KEPT. Until 2026-08-26 this recorded `f"HTTP {e.status}"` and threw the
            # response away, and nothing printed even that -- so a whole live session could send
            # sixty orders a cycle, have every batch refused, and report `placed: 0, errors: 1`
            # with no way to learn why. `KalshiError` has carried `.body` all along.
            out[f"{kind}_errors"].append({"batch_error": f"HTTP {e.status}", "n": len(arg),
                                          "status": e.status, "body": str(e.body)[:400]})
            if e.status == 429:
                out["rate_limited"] += 1
                # A 429 will not clear inside this cycle -- the bucket refills at 300 tok/s
                # against a 2 s cycle -- so stop rather than burn the rest on a closed door.
                return [], REJECTED
            # Any other status is about THIS batch: a malformed body, a 500 on one shard. The
            # remaining chunks are unrelated and are still worth attempting.
            return [], FAILED
        except Exception as e:                               # noqa: BLE001
            out[f"{kind}_errors"].append({"batch_error": f"{type(e).__name__}: {str(e)[:80]}",
                                          "n": len(arg)})
            return [], FAILED

    # Which cancels actually landed decides which orders leave the LOCAL book. Stripping every
    # requested cancel unconditionally -- what this did until 2026-08-24 -- orphans an order that
    # is still resting on the exchange: our state says the level is free, `reconcile` never plans
    # another cancel for it because it can no longer see it, and it sits there until
    # `cancel_all`. Which is exactly the order most likely to have failed a cancel, since a
    # partial fill is one of the ways a cancel comes back with reduced_by == 0.
    ids = [c.get("order_id") for c in to_cancel if c.get("order_id")]
    done_ids = set()
    if live and ids:
        for i in range(0, len(ids), mb):
            chunk = ids[i:i + mb]
            results, status = _send(k.batch_cancel_orders, chunk,
                                    CANCEL_TOKEN_COST * len(chunk), "cancel")
            # Results come back positionally, as they do for creates. `order_id` off the result
            # is preferred when the API supplies it; the zip is the fallback, and a short result
            # list leaves the tail unconfirmed rather than silently confirmed.
            for oid, res in zip(chunk, results):
                # `reduced_by` is the confirmation, NOT the absence of an error -- a failed item
                # comes back with reduced_by == 0.
                err = res.get("error") or {}
                gone = (err.get("code") if isinstance(err, dict) else None) == "not_found"
                if gone:
                    # NOT FOUND MEANS THE EXCHANGE DOES NOT HAVE IT, which is the opposite of a
                    # failure to cancel -- it is already off the book, filled or cancelled
                    # elsewhere. Until 2026-08-26 this fell into the error branch and the order
                    # stayed in the LOCAL book forever: `reconcile` re-planned the same cancel
                    # every cycle (measured live, a steady ~7 per cycle), the phantom kept eating
                    # the per-market side cap and the capital budget, and `within_risk_limits`
                    # started blocking real orders against resting value that did not exist.
                    # So: drop it locally, but do NOT count it as `cancelled` -- nothing was
                    # reduced, and the count is what a reader uses to judge churn.
                    done_ids.add(oid)
                    out["cancel_gone"] += 1
                elif res.get("error") or not float(res.get("reduced_by") or 0):
                    out["cancel_errors"].append({"order_id": oid, **res})
                else:
                    done_ids.add(res.get("order_id") or oid)
                    out["cancelled"] += 1
            if results and len(results) != len(chunk):
                out["cancel_errors"].append(
                    {"batch_shape": f"{len(results)} results for {len(chunk)} cancels"})
            if status in (DECLINED, REJECTED):
                # Count what never reached Kalshi, so a short cycle is legible as a budget
                # decision rather than as a plan that was never that big. A DECLINED chunk is
                # itself unsent; a REJECTED one was sent and is already in `cancel_errors`.
                out["unsent_cancels"] += len(ids) - (i if status == DECLINED else i + len(chunk))
                break

    for c in to_cancel:
        if live and c.get("order_id") not in done_ids:
            continue                     # still resting on the exchange -- keep it visible
        st = book.state(c["ticker"])
        st.set_side(c["side"], [o for o in st.side(c["side"])
                                if abs(o["price"] - c["price"]) > 1e-9])
        # ARM THE ZOMBIE HOLD, here and only here: this is the one point that knows a cancel
        # actually landed, and it covers the dry path too. `edge_negative` ONLY. An `improve_*`
        # cancel is a deliberate replacement and an arming there would turn the hold into a
        # 30-second governor on every improve -- a much larger change than this is, and one that
        # would arrive as a side effect rather than as a measurement. `no_longer_quoting` means we
        # are leaving the market, and coming back is a cold start, which the hold does not gate.
        if c.get("reason") == "edge_negative":
            st.arm_zombie(c["side"], now)
            out["zombies_armed"] += 1
    if not live:
        out["cancelled"] = len(to_cancel)

    # A place exists to take the slot its paired cancel freed. If that cancel did not land -- the
    # budget ran out mid-chunk, a 429 stopped the loop, or the item came back with reduced_by 0
    # because it had partly filled -- then the old order is STILL RESTING and placing the new one
    # doubles the rung. Two orders where the risk gate approved one, and the local book would show
    # only the new one, so `reconcile` could not see the other to cancel it either.
    #
    # In a dry run nothing was sent, so nothing was confirmed; the pairing check would drop every
    # place and the ladder would never be exercised. Live only.
    cancelled_pairs = {c.get("pair") for c in to_cancel
                       if c.get("pair") and (not live or c.get("order_id") in done_ids)}
    survivors = []
    for p in to_place:
        if p.get("pair") and p["pair"] not in cancelled_pairs:
            out["orphaned_places"] += 1
            continue
        survivors.append(p)
    to_place = survivors

    bodies, meta = [], []
    for p in to_place:
        p["client_order_id"] = str(uuid.uuid4())
        bodies.append(k.order_body(p["ticker"], p["side"], k.fp(p["price"]), p["count"],
                                   client_order_id=p["client_order_id"]))
        meta.append(p)

    def _record(p, order_id=None, dry=False):
        st = book.state(p["ticker"])
        st.set_side(p["side"], st.side(p["side"]) + [
            {"ticker": p["ticker"], "side": p["side"], "price": p["price"],
             "count": p["count"], "order_id": order_id,
             "client_order_id": p["client_order_id"], "dry_run": dry}])

    if live and bodies:
        for i in range(0, len(bodies), mb):
            chunk, chunk_meta = bodies[i:i + mb], meta[i:i + mb]
            results, status = _send(k.batch_create_orders, chunk,
                                    ORDER_TOKEN_COST * len(chunk), "place")
            for p, res in zip(chunk_meta, results):
                if res.get("error"):
                    out["place_errors"].append({**p, "error": res["error"]})
                    continue
                _record(p, order_id=res.get("order_id"))
                out["placed"] += 1
            if results and len(results) != len(chunk):
                out["place_errors"].append(
                    {"batch_shape": f"{len(results)} results for {len(chunk)} orders"})
            if status in (DECLINED, REJECTED):
                out["unsent_places"] += len(bodies) - (i if status == DECLINED else i + len(chunk))
                break
    elif bodies:
        for p in meta:
            _record(p, dry=True)
        out["placed"] = len(bodies)
    return out
