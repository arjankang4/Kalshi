"""WebSocket frames -> the REST-canonical schema.

REST is canonical. Every transform here moves WebSocket data toward the REST
shape; none moves REST toward WebSocket. Where REST and WebSocket disagree and
REST looks wrong, that is an escalation, not a licence to reconcile in the
convenient direction.

The traps this module exists to absorb, all verified against the specs:

  * `market_ticker` (WS) is `ticker` (REST).
  * WS trades have no `created_time`; it is synthesized from `ts_ms`. The
    deprecated integer-seconds `ts` is asserted-then-dropped, never used as a
    fallback -- second resolution would silently reorder same-second trades
    against a microsecond REST tape.
  * `orderbook_delta.ts` is an RFC3339 *string* while `trade.ts` and `ticker.ts`
    are integer *seconds*. Same field name, different JSON type, same protocol:
    dispatch on channel, never on field name.
  * WS omits `yes_dollars_fp` entirely when a side is empty; REST emits `[]`.
    Absent and empty are the same fact here, and both mean zero levels -- but
    "zero levels" and "never observed" must stay distinguishable, which is what
    the snapshot header's level counts are for.
  * In a price level `["0.1500","100.00"]`, element 1 is a *quantity*. Running it
    through the price converter is a silent 100x error.
"""
import hashlib
from datetime import datetime, timezone

import config
from storage.units import to_cc, to_micros

SCHEMA_VERSION = config.SCHEMA_VERSION


def _us(ts_ms):
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _ns_to_dt(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _rfc3339(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class ConformanceError(ValueError):
    pass


def delta_source_ts(msg):
    """Resolution order for orderbook_delta, whose ts fields are both optional.

    Returns (source_ts, ts_quality).
    """
    if msg.get("ts_ms") is not None:
        return _us(msg["ts_ms"]), "exchange_ms"
    if msg.get("ts"):
        # RFC3339 string on THIS channel only.
        return _rfc3339(msg["ts"]), "exchange_s"
    return None, "client_recv"


def resolve_basis(price_micros, side, use_yes_price, market_type, notional_micros):
    """Bring a WebSocket price onto the REST-native (no-leg) basis.

    REST quotes `no_dollars` in no-leg pricing. When a subscription set
    `use_yes_price=true`, the WS no side arrives in yes-leg pricing and must be
    inverted back, using the market's own `notional_value_dollars` -- NOT a
    hardcoded 1.0, which is right for every binary today and silently wrong the
    first time a non-unit-notional market lists.

    Scalar markets are left alone: their yes/no relationship is not 1 - p, so
    the inversion would be meaningless. They are flagged instead.
    """
    if not use_yes_price or side != "no":
        return price_micros, "no_leg_native"
    if market_type != "binary":
        return price_micros, "yes_leg"
    if not notional_micros:
        raise ConformanceError(
            "use_yes_price=true but notional_value_dollars is unknown; refusing "
            "to guess the inversion constant")
    return notional_micros - price_micros, "no_leg_native"


def trade(frame, recv_ns, sid, ingest_ts):
    """WS trade msg -> the REST Trade schema.

    Note the completeness caveat this encodes: WS trade frames carry no `seq`,
    so gaps on that channel are undetectable. `source` records that, and dedup
    must never let a ws_trade shadow a REST trade.
    """
    m = frame["msg"]
    ts_ms = m.get("ts_ms")
    if ts_ms is None:
        raise ConformanceError("WS trade has no ts_ms; cannot synthesize created_time")
    if m.get("ts") is not None and int(m["ts"]) != ts_ms // 1000:
        raise ConformanceError(
            f"WS trade {m.get('trade_id')}: ts={m['ts']} disagrees with "
            f"ts_ms={ts_ms}; upstream timestamps are inconsistent")
    created = _us(ts_ms)
    return {
        "trade_id": m["trade_id"],
        "ticker": m["market_ticker"],                 # RENAME
        "event_ticker": None,
        "series_ticker": m["market_ticker"].split("-")[0],
        "market_id": None,                            # not present on WS trade
        "yes_price_micros": to_micros(m["yes_price_dollars"]),
        "no_price_micros": to_micros(m["no_price_dollars"]),
        "count_cc": to_cc(m["count_fp"]),
        "taker_outcome_side": m.get("taker_outcome_side"),
        "taker_book_side": m.get("taker_book_side"),
        "taker_side_legacy": m.get("taker_side"),
        "is_block_trade": bool(m.get("is_block_trade")),
        "created_time": created,                      # SYNTHESIZED from ts_ms
        "source_ts": created,
        "ingest_ts": ingest_ts,
        "ts_quality": "exchange_ms",                  # REST carries us; WS only ms
        "source": "ws_trade",
        "ws_sid": sid,
        "schema_version": SCHEMA_VERSION,
    }


def ticker(frame, recv_ns, sid, capture_session_id, ingest_ts):
    """WS ticker msg -> the ticker_updates table.

    Note `ts` here is an integer *seconds* value, while the identically named
    field on orderbook_delta is an RFC3339 string. Verified live on both. This
    function only ever reads ts_ms, so the ambiguity cannot leak in.
    """
    m = frame["msg"]
    ts_ms = m.get("ts_ms")
    if ts_ms is None:
        raise ConformanceError("WS ticker has no ts_ms")
    tkr = m["market_ticker"]
    return {
        "ticker": tkr,                                # RENAME from market_ticker
        "market_id": m.get("market_id") or None,
        "series_ticker": tkr.split("-")[0],
        "price_micros": to_micros(m.get("price_dollars")),
        "yes_bid_micros": to_micros(m.get("yes_bid_dollars")),
        "yes_ask_micros": to_micros(m.get("yes_ask_dollars")),
        "yes_bid_size_cc": to_cc(m.get("yes_bid_size_fp")),
        "yes_ask_size_cc": to_cc(m.get("yes_ask_size_fp")),
        "last_trade_size_cc": to_cc(m.get("last_trade_size_fp")),
        "volume_cc": to_cc(m.get("volume_fp")),
        "open_interest_cc": to_cc(m.get("open_interest_fp")),
        "dollar_volume": m.get("dollar_volume"),
        "dollar_open_interest": m.get("dollar_open_interest"),
        "capture_session_id": capture_session_id,
        "sid": sid,
        "source_ts": _us(ts_ms),
        "recv_ts": _ns_to_dt(recv_ns),
        "ingest_ts": ingest_ts,
        "ts_quality": "exchange_ms",
        "source": "ws_ticker",
        "schema_version": SCHEMA_VERSION,
    }


def _levels(raw, side, use_yes_price, market_type, notional):
    """[[price_str, size_str], ...] -> normalized level dicts, best first.

    Both `yes_dollars` and `no_dollars` are *bid* stacks (Kalshi returns no
    asks), so best = highest price on each side.
    """
    out = []
    for pair in raw or []:
        px = to_micros(pair[0])
        sz = to_cc(pair[1])                            # element 1 is a QUANTITY
        px, basis = resolve_basis(px, side, use_yes_price, market_type, notional)
        out.append({"price_micros": px, "size_cc": sz, "price_basis": basis})
    out.sort(key=lambda r: -r["price_micros"])
    for i, r in enumerate(out):
        r["level_ix"] = i
    return out


def snapshot(frame, recv_ns, capture_session_id, epoch, sub, market_meta, ingest_ts):
    """WS orderbook_snapshot -> (header row, level rows).

    A deterministic snapshot_id, never a uuid4: promotion must be byte-
    reproducible, and a random id would change the file on every rerun.
    """
    m = frame["msg"]
    ticker = m["market_ticker"]
    meta = market_meta.get(ticker) or {}
    use_yes = bool(sub.get("use_yes_price"))
    mtype = meta.get("market_type") or "binary"
    notional = meta.get("notional_value_micros")

    yes = _levels(m.get("yes_dollars_fp"), "yes", use_yes, mtype, notional)
    no = _levels(m.get("no_dollars_fp"), "no", use_yes, mtype, notional)

    epoch_id = (epoch or {}).get("book_epoch_id")
    seq = frame.get("seq")
    sid = frame.get("sid")
    snapshot_id = hashlib.sha256(
        f"{epoch_id}|{seq}".encode()).hexdigest()[:32]
    recv_ts = _ns_to_dt(recv_ns)
    basis = (yes + no)[0]["price_basis"] if (yes or no) else "no_leg_native"

    header = {
        "snapshot_id": snapshot_id,
        "ticker": ticker,
        "market_id": m.get("market_id"),
        "series_ticker": ticker.split("-")[0],
        "event_ticker": meta.get("event_ticker"),
        "snapshot_kind": "ws_subscribe",
        "source": "ws_orderbook_snapshot",
        # 0 means "observed empty". A missing key upstream and an empty array
        # are the same fact; "never observed" is the absence of this row.
        "yes_levels": len(yes), "no_levels": len(no),
        "yes_total_cc": sum(r["size_cc"] for r in yes),
        "no_total_cc": sum(r["size_cc"] for r in no),
        "best_yes_price_micros": yes[0]["price_micros"] if yes else None,
        "best_yes_size_cc": yes[0]["size_cc"] if yes else None,
        "best_no_price_micros": no[0]["price_micros"] if no else None,
        "best_no_size_cc": no[0]["size_cc"] if no else None,
        "price_basis": basis,
        "use_yes_price_raw": use_yes,
        "notional_micros": notional,
        "capture_session_id": capture_session_id,
        "sid": sid, "seq": seq,
        "book_epoch_id": epoch_id,
        "epoch_seq_start": (epoch or {}).get("seq_start"),
        "source_ts": None,        # no orderbook payload carries one, REST or WS
        "recv_ts": recv_ts,
        "ingest_ts": ingest_ts,
        "ts_quality": "client_recv",
        "schema_version": SCHEMA_VERSION,
    }
    levels = [{
        "snapshot_id": snapshot_id, "ticker": ticker, "side": side,
        "level_ix": r["level_ix"], "price_micros": r["price_micros"],
        "size_cc": r["size_cc"], "price_basis": r["price_basis"],
        "recv_ts": recv_ts, "book_epoch_id": epoch_id,
        "schema_version": SCHEMA_VERSION,
    } for side, rows in (("yes", yes), ("no", no)) for r in rows]
    return header, levels


def delta(frame, recv_ns, capture_session_id, epoch, sub, market_meta, ingest_ts):
    """WS orderbook_delta -> the deltas table."""
    m = frame["msg"]
    ticker = m["market_ticker"]
    meta = market_meta.get(ticker) or {}
    use_yes = bool(sub.get("use_yes_price"))
    side = m["side"]
    px, basis = resolve_basis(
        to_micros(m["price_dollars"]), side, use_yes,
        meta.get("market_type") or "binary", meta.get("notional_value_micros"))
    src_ts, quality = delta_source_ts(m)
    seq = frame.get("seq")
    start = (epoch or {}).get("seq_start")
    return {
        "ticker": ticker,                              # RENAME from market_ticker
        "market_id": m.get("market_id"),
        "series_ticker": ticker.split("-")[0],
        "side": side,
        "price_micros": px,
        "delta_cc": to_cc(m["delta_fp"]),              # signed
        "price_basis": basis,
        "use_yes_price_raw": use_yes,
        "capture_session_id": capture_session_id,
        "sid": frame.get("sid"), "seq": seq,
        "book_epoch_id": (epoch or {}).get("book_epoch_id"),
        "epoch_seq_start": start,
        "seq_in_epoch": (seq - start) if (seq is not None and start is not None) else None,
        "source_ts": src_ts,
        "recv_ts": _ns_to_dt(recv_ns),
        "ingest_ts": ingest_ts,
        "ts_quality": quality,
        "client_order_id": m.get("client_order_id"),
        "subaccount": m.get("subaccount"),
        "source": "ws_orderbook_delta",
        "schema_version": SCHEMA_VERSION,
    }


def rest_orderbook(ticker, body, recv_ts, market_meta, ingest_ts):
    """REST /markets/orderbooks entry -> the same snapshot tables.

    REST is already on the canonical basis, so nothing is inverted here. Note
    the single-market REST response body does not contain the ticker at all --
    it has to be carried from the request.
    """
    ob = body.get("orderbook_fp") or body
    meta = market_meta.get(ticker) or {}
    yes = _levels(ob.get("yes_dollars"), "yes", False, "binary", None)
    no = _levels(ob.get("no_dollars"), "no", False, "binary", None)
    snapshot_id = hashlib.sha256(
        f"rest|{ticker}|{int(recv_ts.timestamp()*1e9)}".encode()).hexdigest()[:32]
    header = {
        "snapshot_id": snapshot_id, "ticker": ticker, "market_id": None,
        "series_ticker": ticker.split("-")[0], "event_ticker": meta.get("event_ticker"),
        "snapshot_kind": "rest_poll", "source": "rest_orderbooks",
        "yes_levels": len(yes), "no_levels": len(no),
        "yes_total_cc": sum(r["size_cc"] for r in yes),
        "no_total_cc": sum(r["size_cc"] for r in no),
        "best_yes_price_micros": yes[0]["price_micros"] if yes else None,
        "best_yes_size_cc": yes[0]["size_cc"] if yes else None,
        "best_no_price_micros": no[0]["price_micros"] if no else None,
        "best_no_size_cc": no[0]["size_cc"] if no else None,
        "price_basis": "no_leg_native", "use_yes_price_raw": None,
        "notional_micros": meta.get("notional_value_micros"),
        "capture_session_id": None, "sid": None, "seq": None,
        "book_epoch_id": None, "epoch_seq_start": None,
        "source_ts": None, "recv_ts": recv_ts, "ingest_ts": ingest_ts,
        "ts_quality": "client_recv", "schema_version": SCHEMA_VERSION,
    }
    levels = [{
        "snapshot_id": snapshot_id, "ticker": ticker, "side": side,
        "level_ix": r["level_ix"], "price_micros": r["price_micros"],
        "size_cc": r["size_cc"], "price_basis": r["price_basis"],
        "recv_ts": recv_ts, "book_epoch_id": None,
        "schema_version": SCHEMA_VERSION,
    } for side, rows in (("yes", yes), ("no", no)) for r in rows]
    return header, levels
