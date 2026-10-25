"""Declared Parquet schemas. Never inferred.

An inferred schema differs between days whenever a column happens to be all-null
that day, and Athena then fails the union with HIVE_BAD_DATA. Every schema here
is explicit and versioned.

Conventions:
  * pa.string(), not large_string -- both write BYTE_ARRAY, 32-bit offsets suffice.
  * No pa.dictionary() in the file schema. Dictionary encoding is a *physical*
    Parquet encoding selected via use_dictionary= on the writer; an Arrow
    dictionary type only adds a hint Athena ignores.
  * timestamp("us", tz="UTC") -> TIMESTAMP(MICROS, isAdjustedToUTC) -> Athena
    timestamp. Not "ns", which invites a Glue type mismatch.
  * Partition keys are path-encoded only, never columns in the file.
  * Every fact row carries source, source_ts, ingest_ts, ts_quality, schema_version.
"""
import pyarrow as pa

import config

TS = pa.timestamp("us", tz="UTC")
STR = pa.string()
I64 = pa.int64()
I32 = pa.int32()
I16 = pa.int16()
BOOL = pa.bool_()
F64 = pa.float64()
DEC = pa.decimal128(38, 10)

_UNITS = {b"kalshi.price_unit": b"micro_dollars_1e-6",
          b"kalshi.count_unit": b"centi_contracts_1e-2",
          b"kalshi.schema_version": str(config.SCHEMA_VERSION).encode()}

PRICE_RANGE = pa.struct([("start_micros", I64), ("end_micros", I64),
                         ("step_micros", I64)])
SETTLEMENT_SOURCE = pa.struct([("name", STR), ("url", STR)])


TRADES = pa.schema([
    pa.field("trade_id", STR, nullable=False),
    pa.field("ticker", STR, nullable=False),        # REST name; WS says market_ticker
    pa.field("event_ticker", STR),
    pa.field("series_ticker", STR),                 # leading sort key
    pa.field("market_id", STR),                     # WS-only uuid; NULL for REST rows
    pa.field("yes_price_micros", I64, nullable=False),
    pa.field("no_price_micros", I64, nullable=False),
    pa.field("count_cc", I64, nullable=False),
    pa.field("taker_outcome_side", STR, nullable=False),
    pa.field("taker_book_side", STR, nullable=False),
    # Deprecated upstream but still required in the response; kept verbatim so it
    # can be *proven* equal to taker_outcome_side rather than assumed.
    pa.field("taker_side_legacy", STR),
    pa.field("is_block_trade", BOOL, nullable=False),
    pa.field("created_time", TS, nullable=False),
    pa.field("source_ts", TS, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),    # exchange_us | exchange_ms
    pa.field("source", STR, nullable=False),        # rest_trades|rest_historical_trades|ws_trade
    pa.field("ws_sid", I32),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

MARKETS = pa.schema([
    pa.field("ticker", STR, nullable=False),
    pa.field("event_ticker", STR, nullable=False),
    pa.field("series_ticker", STR),
    pa.field("market_type", STR),                   # binary | scalar
    pa.field("exchange_index", I32),
    pa.field("yes_sub_title", STR),
    pa.field("no_sub_title", STR),
    pa.field("created_time", TS),
    pa.field("updated_time", TS),
    pa.field("open_time", TS),
    pa.field("close_time", TS),
    pa.field("expected_expiration_time", TS),
    pa.field("latest_expiration_time", TS),
    pa.field("settlement_ts", TS),
    pa.field("settlement_timer_seconds", I32),
    # Response vocabulary VERBATIM. It differs from the query-filter vocabulary;
    # amending it to match would be amending REST.
    pa.field("status", STR),
    pa.field("status_query_group", STR),            # derived bridge; 'paused' unreachable
    pa.field("result", STR),
    pa.field("can_close_early", BOOL),
    pa.field("is_provisional", BOOL),
    pa.field("yes_bid_micros", I64),
    pa.field("yes_bid_size_cc", I64),
    pa.field("yes_ask_micros", I64),
    pa.field("yes_ask_size_cc", I64),
    pa.field("no_bid_micros", I64),
    pa.field("no_ask_micros", I64),
    pa.field("last_price_micros", I64),
    pa.field("previous_yes_bid_micros", I64),
    pa.field("previous_yes_ask_micros", I64),
    pa.field("previous_price_micros", I64),
    pa.field("volume_cc", I64),
    pa.field("volume_24h_cc", I64),
    pa.field("open_interest_cc", I64),
    pa.field("notional_value_micros", I64),         # the use_yes_price inversion constant
    pa.field("settlement_value_micros", I64),
    pa.field("strike_type", STR),
    pa.field("floor_strike", DEC),                  # index level, not a price
    pa.field("cap_strike", DEC),
    pa.field("functional_strike", STR),
    pa.field("custom_strike_json", STR),            # canonical JSON, sorted keys
    pa.field("expiration_value", STR),
    pa.field("price_level_structure", STR),
    pa.field("price_ranges", pa.list_(PRICE_RANGE)),
    pa.field("min_tick_micros", I64),               # derived: min(step)
    pa.field("rules_primary", STR),
    pa.field("rules_secondary", STR),
    pa.field("mve_collection_ticker", STR),
    # liquidity_dollars is documented as always "0.0000" -- dropped deliberately.
    pa.field("row_hash", STR, nullable=False),      # SCD change detector
    pa.field("observed_ts", TS, nullable=False),
    pa.field("source_ts", TS),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),
    pa.field("source", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

EVENTS = pa.schema([
    pa.field("event_ticker", STR, nullable=False),
    pa.field("series_ticker", STR, nullable=False),
    pa.field("title", STR),
    pa.field("sub_title", STR),
    pa.field("category_legacy", STR),               # deprecated; series-level is canonical
    pa.field("collateral_return_type", STR),
    pa.field("mutually_exclusive", BOOL),
    pa.field("available_on_brokers", BOOL),
    pa.field("strike_date", TS),                    # XOR with strike_period
    pa.field("strike_period", STR),
    pa.field("settlement_sources", pa.list_(SETTLEMENT_SOURCE)),
    pa.field("product_metadata_json", STR),
    pa.field("fee_type_override", STR),
    pa.field("fee_multiplier_override", F64),       # a multiplier, not money: float is right
    pa.field("exchange_index", I32),
    pa.field("last_updated_ts", TS),
    pa.field("row_hash", STR, nullable=False),
    pa.field("observed_ts", TS, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),
    pa.field("source", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

SERIES = pa.schema([
    pa.field("ticker", STR, nullable=False),
    pa.field("title", STR),
    pa.field("category", STR, nullable=False),      # Financials | Crypto | ...
    pa.field("frequency", STR),
    pa.field("tags", pa.list_(STR)),
    pa.field("settlement_sources", pa.list_(SETTLEMENT_SOURCE)),
    pa.field("contract_url", STR),
    pa.field("contract_terms_url", STR),
    pa.field("fee_type", STR),
    pa.field("fee_multiplier", F64),
    pa.field("additional_prohibitions", pa.list_(STR)),
    pa.field("product_metadata_json", STR),
    pa.field("volume_cc", I64),
    pa.field("exchange_index", I32),
    pa.field("last_updated_ts", TS),
    pa.field("row_hash", STR, nullable=False),
    pa.field("observed_ts", TS, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),
    pa.field("source", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

# Live and historical candlestick endpoints disagree on field names (volume vs
# volume_fp, open vs open_dollars, and historical lacks min/max). Both are REST,
# so "never amend REST" gives no tiebreak: take the live vocabulary as the
# superset and mark which variant a row came from.
CANDLESTICKS = pa.schema([
    pa.field("ticker", STR, nullable=False),
    pa.field("series_ticker", STR),
    pa.field("event_ticker", STR),
    pa.field("interval_min", I32, nullable=False),  # 1 | 60 | 1440
    pa.field("end_period_ts", TS, nullable=False),
    pa.field("start_period_ts", TS),
    pa.field("price_open_micros", I64),
    pa.field("price_high_micros", I64),
    pa.field("price_low_micros", I64),
    pa.field("price_close_micros", I64),
    pa.field("price_mean_micros", I64),
    pa.field("price_previous_micros", I64),
    pa.field("price_min_micros", I64),              # live variant only
    pa.field("price_max_micros", I64),              # live variant only
    pa.field("yes_bid_open_micros", I64),
    pa.field("yes_bid_high_micros", I64),
    pa.field("yes_bid_low_micros", I64),
    pa.field("yes_bid_close_micros", I64),
    pa.field("yes_ask_open_micros", I64),
    pa.field("yes_ask_high_micros", I64),
    pa.field("yes_ask_low_micros", I64),
    pa.field("yes_ask_close_micros", I64),
    pa.field("volume_cc", I64),
    pa.field("open_interest_cc", I64),
    pa.field("trade_count", I64),                   # derived rows only
    pa.field("vwap_micros", I64),                   # derived rows only
    pa.field("candle_api_variant", STR, nullable=False),  # live|historical|derived_from_trades
    pa.field("source", STR, nullable=False),
    pa.field("source_ts", TS, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

ORDERBOOK_DELTAS = pa.schema([
    pa.field("ticker", STR, nullable=False),        # WS market_ticker -> REST name
    pa.field("market_id", STR),
    pa.field("series_ticker", STR),
    pa.field("side", STR, nullable=False),          # yes | no
    pa.field("price_micros", I64, nullable=False),
    pa.field("delta_cc", I64, nullable=False),      # signed
    pa.field("price_basis", STR, nullable=False),   # no_leg_native (REST) | yes_leg
    pa.field("use_yes_price_raw", BOOL),
    # The validity model. No gap_before flag: a gap terminates an epoch, so by
    # construction no row inside an epoch is preceded by one.
    pa.field("capture_session_id", STR, nullable=False),
    pa.field("sid", I32, nullable=False),
    pa.field("seq", I64, nullable=False),
    pa.field("book_epoch_id", STR, nullable=False),
    pa.field("epoch_seq_start", I64),
    pa.field("seq_in_epoch", I64),
    pa.field("source_ts", TS),                      # ts/ts_ms are optional on this msg
    pa.field("recv_ts", TS, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),    # exchange_ms|exchange_s|client_recv
    pa.field("client_order_id", STR),               # set only when WE caused the change
    pa.field("subaccount", I32),
    pa.field("source", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

# Split header/levels deliberately. WS omits yes_dollars_fp entirely when a side
# is empty; REST emits []. In one flat table both are zero rows -- so "empty" and
# "never observed" become indistinguishable. The counts below make that structural.
ORDERBOOK_SNAPSHOTS = pa.schema([
    pa.field("snapshot_id", STR, nullable=False),   # deterministic hash, never a uuid4
    pa.field("ticker", STR, nullable=False),
    pa.field("market_id", STR),
    pa.field("series_ticker", STR),
    pa.field("event_ticker", STR),
    pa.field("snapshot_kind", STR, nullable=False), # ws_subscribe|rest_poll|materialized
    pa.field("source", STR, nullable=False),
    pa.field("yes_levels", I16, nullable=False),    # 0 == observed empty; never NULL
    pa.field("no_levels", I16, nullable=False),
    pa.field("yes_total_cc", I64, nullable=False),
    pa.field("no_total_cc", I64, nullable=False),
    pa.field("best_yes_price_micros", I64),         # NULL iff that side has 0 levels
    pa.field("best_yes_size_cc", I64),
    pa.field("best_no_price_micros", I64),
    pa.field("best_no_size_cc", I64),
    pa.field("price_basis", STR, nullable=False),
    pa.field("use_yes_price_raw", BOOL),
    pa.field("notional_micros", I64),
    pa.field("capture_session_id", STR),
    pa.field("sid", I32),
    pa.field("seq", I64),
    pa.field("book_epoch_id", STR),
    pa.field("epoch_seq_start", I64),
    pa.field("source_ts", TS),                      # always NULL: no payload carries one
    pa.field("recv_ts", TS, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

ORDERBOOK_SNAPSHOT_LEVELS = pa.schema([
    pa.field("snapshot_id", STR, nullable=False),
    pa.field("ticker", STR, nullable=False),        # denormalized: avoids a join
    pa.field("side", STR, nullable=False),
    pa.field("level_ix", I16, nullable=False),      # 0 = best, ascending away
    pa.field("price_micros", I64, nullable=False),
    pa.field("size_cc", I64, nullable=False),
    pa.field("price_basis", STR, nullable=False),
    pa.field("recv_ts", TS, nullable=False),        # denormalized for time pruning
    pa.field("book_epoch_id", STR),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

BOOK_EPOCHS = pa.schema([
    pa.field("book_epoch_id", STR, nullable=False),
    pa.field("capture_session_id", STR, nullable=False),
    pa.field("sid", I32, nullable=False),
    pa.field("channel", STR, nullable=False),
    pa.field("n_markets", I32),
    pa.field("use_yes_price_raw", BOOL),
    pa.field("price_basis_stored", STR),
    pa.field("seq_start", I64),
    pa.field("seq_end", I64),
    pa.field("frames_expected", I64),
    pa.field("frames_delta", I64),
    pa.field("frames_snapshot", I64),
    pa.field("frames_control", I64),                # ok/unsubscribed CONSUME seq
    pa.field("frames_seen", I64),
    pa.field("is_contiguous", BOOL, nullable=False),
    pa.field("opened_with_snapshot", BOOL, nullable=False),   # reconstructability gate
    pa.field("open_reason", STR),
    pa.field("close_reason", STR),
    pa.field("close_error_code", I32),
    pa.field("start_recv_ts", TS),
    pa.field("end_recv_ts", TS),
    pa.field("gap_seq_lo", I64),
    pa.field("gap_seq_hi", I64),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

# Per-market top-of-book and volume snapshots. No `seq` on this channel, so
# there is no gap detection: treat it as a sampled signal, not a complete tape.
# Its value is as an independent check that a book reconstructed from deltas
# agrees with Kalshi's own top-of-book.
TICKER_UPDATES = pa.schema([
    pa.field("ticker", STR, nullable=False),        # WS market_ticker -> REST name
    pa.field("market_id", STR),
    pa.field("series_ticker", STR),
    pa.field("price_micros", I64),                  # last traded
    pa.field("yes_bid_micros", I64),
    pa.field("yes_ask_micros", I64),
    pa.field("yes_bid_size_cc", I64),
    pa.field("yes_ask_size_cc", I64),
    pa.field("last_trade_size_cc", I64),
    pa.field("volume_cc", I64),
    pa.field("open_interest_cc", I64),
    # These two are already integers on the wire, in whole dollars.
    pa.field("dollar_volume", I64),
    pa.field("dollar_open_interest", I64),
    pa.field("capture_session_id", STR),
    pa.field("sid", I32),
    pa.field("source_ts", TS, nullable=False),      # from ts_ms
    pa.field("recv_ts", TS, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),    # exchange_ms
    pa.field("source", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

MARKET_ID_XREF = pa.schema([
    pa.field("market_id", STR, nullable=False),     # WS-only; REST exposes no id
    pa.field("ticker", STR, nullable=False),
    pa.field("first_seen_ts", TS, nullable=False),
    pa.field("last_seen_ts", TS, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)


# --- Massive option data -------------------------------------------------
# Not Kalshi. Prices here are per-share option premiums, still stored as int64
# micro-dollars so the "never hold a price as a float" rule holds across vendors.
# Sizes are WHOLE CONTRACTS, not centi-contracts -- Massive and Kalshi disagree
# on this and mixing them is a silent 100x. Hence the explicit `_contracts`
# suffix rather than the `_cc` used everywhere else.

OPTION_CONTRACTS = pa.schema([
    pa.field("option_ticker", STR, nullable=False),  # O:SPXW250820C06350000
    pa.field("underlying", STR, nullable=False),     # SPX | NDX
    pa.field("root", STR, nullable=False),           # SPXW | NDXP
    pa.field("expiration_date", STR, nullable=False),
    pa.field("contract_type", STR, nullable=False),  # call | put
    pa.field("strike", DEC, nullable=False),
    pa.field("shares_per_contract", I32),
    pa.field("primary_exchange", STR),
    pa.field("exercise_style", STR),
    pa.field("first_seen_ts", TS, nullable=False),
    pa.field("source", STR, nullable=False),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)

# One row per (option_ticker, asof_ts) -- NOT per trade leg. A single instant is
# shared by every Kalshi market that traded then, and one contract commonly backs
# several boundaries, so keying on the leg would duplicate the same NBBO many
# times over. The notebook joins trades -> boundary -> strike -> here.
OPTION_QUOTES_ASOF = pa.schema([
    pa.field("option_ticker", STR, nullable=False),
    pa.field("underlying", STR, nullable=False),
    pa.field("root", STR, nullable=False),
    pa.field("expiration_date", STR, nullable=False),
    pa.field("contract_type", STR, nullable=False),
    pa.field("strike", DEC, nullable=False),
    pa.field("asof_ts_ns", I64, nullable=False),     # the Kalshi instant
    pa.field("asof_ts", TS, nullable=False),
    # False = the contract never quoted at or before this instant. Distinct from
    # a quote with no bid, which is a real 0.00 x 0 and prices to zero. Conflating
    # the two is what theo.py:37-49 warns nulls half of all 0DTE quotes.
    pa.field("has_quote", BOOL, nullable=False),
    pa.field("had_bid", BOOL),
    pa.field("had_ask", BOOL),
    pa.field("quote_ts_ns", I64),                    # sip_timestamp, nanoseconds
    pa.field("quote_ts", TS),
    pa.field("quote_age_ns", I64),                   # asof - quote; measures clock skew
    pa.field("bid_micros", I64),
    pa.field("bid_size_contracts", I64),
    pa.field("ask_micros", I64),
    pa.field("ask_size_contracts", I64),
    pa.field("bid_exchange", I32),
    pa.field("ask_exchange", I32),
    pa.field("sequence_number", I64),
    pa.field("fetch_mode", STR, nullable=False),     # tape | asof
    pa.field("source", STR, nullable=False),
    pa.field("source_ts", TS),
    pa.field("ingest_ts", TS, nullable=False),
    pa.field("ts_quality", STR, nullable=False),
    pa.field("schema_version", I16, nullable=False),
], metadata=_UNITS)


TABLES = {
    "option_contracts": OPTION_CONTRACTS,
    "option_quotes_asof": OPTION_QUOTES_ASOF,
    "trades": TRADES,
    "markets": MARKETS,
    "events": EVENTS,
    "series": SERIES,
    "candlesticks": CANDLESTICKS,
    "orderbook_deltas": ORDERBOOK_DELTAS,
    "orderbook_snapshots": ORDERBOOK_SNAPSHOTS,
    "orderbook_snapshot_levels": ORDERBOOK_SNAPSHOT_LEVELS,
    "book_epochs": BOOK_EPOCHS,
    "ticker_updates": TICKER_UPDATES,
    "market_id_xref": MARKET_ID_XREF,
}

# Leading component drives row-group min/max pruning, which is what replaces
# partitioning by series (1,172 series x 365 days = 428k mostly-empty partitions).
SORT_KEYS = {
    "option_contracts": ["underlying", "expiration_date", "strike", "contract_type"],
    "option_quotes_asof": ["underlying", "option_ticker", "asof_ts_ns"],
    "trades": ["series_ticker", "ticker", "created_time", "trade_id"],
    "markets": ["series_ticker", "ticker", "observed_ts"],
    "events": ["series_ticker", "event_ticker", "observed_ts"],
    "series": ["category", "ticker", "observed_ts"],
    "candlesticks": ["series_ticker", "ticker", "interval_min", "end_period_ts"],
    "orderbook_deltas": ["ticker", "capture_session_id", "sid", "seq"],
    "orderbook_snapshots": ["ticker", "recv_ts", "snapshot_id"],
    "orderbook_snapshot_levels": ["ticker", "recv_ts", "snapshot_id", "side", "level_ix"],
    "book_epochs": ["capture_session_id", "sid", "seq_start"],
    "ticker_updates": ["ticker", "source_ts", "recv_ts"],
    "market_id_xref": ["ticker", "market_id"],
}

# Natural key for dedup during the sorted merge.
NATURAL_KEYS = {
    "option_contracts": ["option_ticker"],
    "option_quotes_asof": ["option_ticker", "asof_ts_ns"],
    "trades": ["trade_id"],
    "markets": ["ticker", "row_hash"],
    "events": ["event_ticker", "row_hash"],
    "series": ["ticker", "row_hash"],
    "candlesticks": ["ticker", "interval_min", "end_period_ts", "candle_api_variant"],
    "orderbook_deltas": ["capture_session_id", "sid", "seq"],
    "orderbook_snapshots": ["snapshot_id"],
    "orderbook_snapshot_levels": ["snapshot_id", "side", "level_ix"],
    "book_epochs": ["book_epoch_id"],
    # A market can legitimately emit two ticker frames in the same millisecond
    # with different content, so recv_ts is part of the key.
    "ticker_updates": ["ticker", "source_ts", "recv_ts"],
    "market_id_xref": ["market_id", "ticker"],
}


# Scoped study tables: identical schemas, separate partitions.
#
# The index 4pm-close backfill writes here rather than into `trades`, and that is
# not cosmetic. `already_done()` keys resume on (table, dt) alone, so a scoped
# run writing `trades` would make the full exchange-wide sweep skip every day it
# touched -- leaving those partitions holding six series forever, with a manifest
# asserting they are complete. Separate tables make the two runs incapable of
# interfering.
SCOPED_SUFFIX = "_index_close"

for _t in ("trades", "markets"):
    _alias = _t + SCOPED_SUFFIX
    TABLES[_alias] = TABLES[_t]
    SORT_KEYS[_alias] = SORT_KEYS[_t]
    NATURAL_KEYS[_alias] = NATURAL_KEYS[_t]
del _t, _alias


def dict_columns(table):
    """Low-cardinality string columns worth dictionary-encoding physically."""
    schema = TABLES[table]
    skip = {"trade_id", "snapshot_id", "row_hash", "rules_primary", "rules_secondary",
            "custom_strike_json", "product_metadata_json", "title", "sub_title",
            "yes_sub_title", "no_sub_title", "contract_url", "contract_terms_url"}
    return [f.name for f in schema
            if pa.types.is_string(f.type) and f.name not in skip]
