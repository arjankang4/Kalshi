"""Append-only record of every quoting decision and the book that produced it.

This is the substrate for two studies:

  * **What we quoted** -- ticker, side, our theo at that instant, the price we sent, the edge we
    thought we were charging, and which rule in the ladder fired. Shaped so it lines up with the
    REST trade tape (`storage.schemas.TRADES`) on ticker / price / count / time, with the
    decision fields added.
  * **Where we would have sat in the queue** -- the full book at decision time, plus size ahead
    of us at our own price level. `kalshi.client.queue_positions` is ground truth once we are
    live; this is what stands in for it in a dry run, and what cross-checks it afterwards.

JSONL, one record per decision, fsync'd per batch: a crash mid-session should cost the last
write, not the file. Conform to Parquet later -- `ORDERBOOK_SNAPSHOTS` already has a `rest_poll`
snapshot_kind for exactly this shape.
"""
import json
import os
import time
from datetime import datetime, timezone

import config

JOURNAL_DIR = config.ROOT / "var" / "journal"


def _now_ns():
    return time.time_ns()


def size_ahead(book_entry, side, price):
    """Contracts resting at our price on our side -- i.e. how many are ahead of us if we join.

    Kalshi's book gives one ladder per side in that side's own terms: `yes` levels are YES bids,
    `no` levels are NO bids, which are YES offers at 1 - p. Comparing in cents avoids matching
    0.08 against 0.08000000000000002.
    """
    if not book_entry or price is None:
        return None
    from strategy.quoting import ladders
    yes, no = ladders(book_entry)
    levels = yes if side == "bid" else no
    want = int(round((price if side == "bid" else 1.0 - price) * 100))
    for lvl in levels:
        if int(round(float(lvl[0]) * 100)) == want:
            return float(lvl[1])
    return 0.0


class Journal:
    FSYNC_EVERY = 50

    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.n = 0

    def write(self, rec):
        self._fh.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
        self.n += 1
        # The docstring promises durability, so honour it. A quoting process that is killed
        # mid-session must not lose the record of what it decided -- that record is the only
        # evidence of what we would have done. Batched so it is not an fsync per line.
        if self.n % self.FSYNC_EVERY == 0:
            self.flush()

    def flush(self):
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        self.flush()
        self._fh.close()

    def decision(self, *, ticker, side, action, theo_detail, book_entry, best_bid, best_ask,
                 bid_size, ask_size, resting=None, live=False, event=None, spot=None,
                 spot_detail=None, lean_detail=None, order_id=None, client_order_id=None,
                 count=config.ORDER_COUNT):
        """One decision. Records what we saw, what we decided, and why."""
        px = action.price
        rec = {
            "v": 1,
            "ts_ns": _now_ns(),
            "ts": datetime.now(timezone.utc).isoformat(),
            "live": live,
            "event_ticker": event,
            "ticker": ticker,
            "side": side,                       # bid | ask, in YES terms
            "count": count,

            # --- the decision ---
            "action": action.kind,              # improve | penny | join | none
            "rule": action.reason,
            "price": px,
            "edge": action.edge,

            # --- what we believed ---
            "theo": (theo_detail or {}).get("fair"),
            "theo_detail": theo_detail,         # bracket_fair() output: probs, terms, use_* flags
            "spot": spot,
            # Which source spot came from and how far the other one was. `spot` stays top-level
            # and unqualified so the existing backtest/notebook columns keep working; this is the
            # health signal on both feeds -- see pricing/spot.
            "spot_detail": spot_detail,
            # What the realized-vol model said and what the lean did with it. Present even when
            # LEAN_ENABLED is False, so a session's worth of would-be leans is observable before
            # any of them is acted on.
            "lean_detail": lean_detail,

            # --- what the market looked like ---
            "best_bid": best_bid, "best_ask": best_ask,
            "bid_size": bid_size, "ask_size": ask_size,
            "spread": (None if best_bid is None or best_ask is None
                       else round(best_ask - best_bid, 4)),
            "book": book_entry,                 # full depth as returned by /markets/orderbooks

            # --- queue substrate ---
            # Size resting at the price we chose, on our side. If we join, this is how many are
            # ahead of us; if we improve or penny, it is 0 by construction and we are first.
            "size_ahead": size_ahead(book_entry, side, px),

            # --- our state ---
            "resting_price": (resting or {}).get("price"),
            "resting_order_id": (resting or {}).get("order_id"),
            # The join key to orders_iter*.json and, once live, to the client_order_id echoed
            # back on our own orderbook_delta frames.
            "client_order_id": client_order_id,
            "order_id": order_id,
        }
        self.write(rec)
        return rec


def open_journal(stamp, name="decisions.jsonl"):
    return Journal(JOURNAL_DIR / f"run={stamp}" / name)
