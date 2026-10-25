# Storage layer

Moves Kalshi market data from the wire to long-term Parquet, at bounded cost.

```
WebSocket ──► local gzipped segments ──► <S3_BUCKET_WS> ──► <S3_BUCKET_HIST>
             (roll 15 min / 64 MB,        (raw, expires  7d)             (Parquet, Glacier IR 365d, gone 1095d)
              sync to disk every 20s)              │                              ▲
REST ──────────────────────────────────────────────┴──── promote at T+3 ──────────┘
                                                          4-day overlap
```

## Capture scope

**Channels:** `orderbook_delta`, `trade`, `ticker`, `fill` — one subscribe command, one sid
per channel, all four written to a single segment stream (`channel=all`). Frames carry only
`sid`, so the `sid_channel` control records are what let promotion route them. `fill` is our
own executions on the authenticated connection, not public market data.

**Markets:** index/ETF **4pm close** markets only, ~522 across six series — `KXINX` and
`KXNASDAQ100` (range brackets), `KXINXU` and `KXNASDAQ100U` (above/below), `KXINXDUD` and
`KXNASDAQDUD` (up/down). The hourly S&P markets were dropped deliberately: pricing them
needs a vol model, and the current theo prices probabilities off SPX option quotes, which
only lines up with the 4pm close.

Kalshi lists **no Dow or ETF close market** — `KXDJI` is an hourly one-off and there is no
SPY/QQQ/IWM/DIA close series. `discover_close_series()` sweeps index-like Financials series
at startup and at day rollover and logs any that gain a 4pm event, so a Dow or ETF listing
gets picked up without a code change.

## The constraint that shapes everything

**Kalshi's REST API has no historical order book endpoint.** `/markets/{ticker}/orderbook`
and `/markets/orderbooks` are current-state only — no timestamp params, no archive variant.
Book history exists only if the recorder captures it live. Trades, candles and metadata are
all re-fetchable; book data is not. That asymmetry is why the recorder gets `Restart=always`,
why a dropped frame writes a marker instead of vanishing, and why the local delete is
strictly conditional on S3 confirming the bytes.

## Operating it

```bash
systemctl status  kalshi-ws-recorder          # continuous capture
systemctl list-timers 'kalshi*'               # flush / seal / promote / monitor
python -m storage.monitor                     # health + leak audit, exit 2 = critical
python -m storage.monitor --json              # risk-agent payload
python -m storage.setup_buckets               # audit lifecycle (--apply to fix)
python -m storage.reconcile fixtures          # WS vs REST conformance
tail -f var/log/{ws-recorder,flush,backfill}.log
```

| Unit | Cadence | Does |
|---|---|---|
| `kalshi-ws-recorder.service` | gated, see below | capture → local segments |
| `kalshi-ws-flush.timer` | every 15 min | upload, verify, delete local |
| `kalshi-ws-seal.timer` | 23:59 **America/New_York** | final flush + day manifest |
| `kalshi-ws-promote.timer` | 01:30 ET | promote D−3 → Parquet |
| `kalshi-storage-monitor.timer` | every 15 min | health + leak audit |
| `kalshi-rest-backfill.service` | one-shot, resumable | 1y history |

## The capture gate

Live capture is gated in `config.py`. **It is off right now** -- turned off
2026-08-22, after the SPX/NDX capture it was enabled for completed (2026-08-20 and
08-21 are the two full sessions on tape).

```python
CAPTURE_SESSION_ENABLED = False     # master switch
CAPTURE_SESSION_OPEN    = (9, 30)   # ET, inclusive
CAPTURE_SESSION_CLOSE   = (16, 15)  # ET, exclusive
```

The gate is weekdays-only as well as time-boxed, and exchange holidays are
deliberately not modelled -- on a holiday the recorder records an empty tape,
which costs nothing and is obvious in the data, whereas a stale hardcoded holiday
table would silently skip a real trading day.

With the gate closed the recorder process stays up and idles: no subscription, no REST
calls, no segments. Turning capture on is a flag flip plus a restart, not a redeploy:

```bash
systemctl restart kalshi-ws-recorder          # after editing CAPTURE_SESSION_ENABLED
grep '^\[gate\]' var/log/ws-recorder.log      # what the gate decided and why
python -m storage.ws_recorder --ignore-gate   # one-off run, ignores flag and window
```

Two things that are easy to get wrong here:

- **Only the recorder is gated.** `flush`, `seal`, `promote` and `monitor` keep running on
  their timers. That is deliberate — it is what drains capture already sitting on local
  disk. They just find nothing new to do, so a closed gate does not strand bytes.
- **The close is 16:15, not 16:00.** Trading stops at the 4pm close but the final book and
  trade frames land right on it, and a hard `16:00:00` cut clips exactly the frames these
  markets settle against.

Exchange holidays are not modelled. On a holiday the recorder subscribes and records an
empty tape, which costs nothing and is obvious in the data. A hardcoded holiday table fails
the other way: it silently skips a real trading day, and book data has no backfill path.

## Things that will bite you, recorded so they don't again

- **Both buckets are versioned.** A lifecycle rule that only sets `Expiration` deletes
  nothing — it writes a delete marker and leaves the bytes as a noncurrent version, billed
  forever. Every retention rule needs `NoncurrentVersionExpiration`,
  `ExpiredObjectDeleteMarker` and `AbortIncompleteMultipartUpload` beside it. Incomplete
  multipart parts are billed *and* invisible to `ListObjects`.
- **`/tmp` and `/dev/shm` are tmpfs — they are RAM.** Scratch goes in `var/scratch`.
  Spilling to `/tmp` spends the memory budget the spill exists to protect.
- **`seq` is per-subscription, not per-market**, and `ok` / `unsubscribed` control frames
  *consume* sequence numbers. Counting only data frames reports a phantom gap on every
  control response. `sid` is also only unique within one connection — after a reconnect,
  session B reuses `sid=1`, so all accounting is keyed by `(session, sid)`.
- **`orderbook_delta.ts` is an RFC3339 string; `trade.ts` and `ticker.ts` are integer
  seconds.** Same field name, different JSON type, same protocol. Dispatch on channel.
- **WS omits `yes_dollars_fp` entirely when a side is empty; REST sends `[]`.** Both mean
  zero levels — but "zero levels" and "never observed" must stay distinguishable, which is
  what the snapshot header's `yes_levels`/`no_levels` counts are for. This is the same trap
  as the Massive `bp`/`bs` note in the top-level README.
- **In a price level `["0.1500","100.00"]`, element 1 is a quantity.** Running it through
  the price converter is a silent 100× error.
- **Only `orderbook_delta` carries `seq`.** `trade` and `ticker` frames have none, so gaps
  on those channels are undetectable and a book epoch over them would assert a contiguity
  guarantee that does not exist — epochs are opened for the book sid only. REST remains the
  only authoritative trade source; a `ws_trade` must never shadow a REST trade in dedup.
- **`ticker.ts` is an integer (seconds); `orderbook_delta.ts` is an RFC3339 string.** Both
  verified live on the same connection. `conform` only ever reads `ts_ms`, so the ambiguity
  cannot leak in.
- **`use_yes_price` inverts using the market's `notional_value_dollars`, not 1.0.** Right
  for every binary today, silently wrong the first time a non-unit-notional market lists.
  Kalshi intends to flip this flag's default and then remove it, so it is recorded per
  segment or historical no-side prices become uninterpretable.
- **Prices are int64 micro-dollars, sizes int64 centi-contracts.** Kalshi emits up to 6dp
  and has `deci_cent`/`centi_cent` markets, so integer *cents* would collapse distinct
  price levels. Conversion hard-fails rather than rounding. Never float, anywhere.
- **Never a fixed UTC offset for ET.** `config.ET` is `ZoneInfo("America/New_York")`. And
  never assume a day is 86400 s: 2026-11-01 is 25 hours, 2026-03-08 is 23.
- **Never infer a Parquet schema.** A day where a nullable column happens to be all-null
  infers a different type, and Athena fails the union with `HIVE_BAD_DATA`. Same reason
  there is no Glue crawler: a crawler can silently rewrite a column's type.
- **Digests canonicalize by schema type, not Python type.** `fee_multiplier` arrives as the
  JSON integer `1`, lands in a float64 column and reads back `1.0`; `decimal128` pads scale.
  A naive `str()` digest reports corruption on a perfectly correct file.

## Determinism

Promotion is a pure function of (sorted input set, code version): a rerun writes
byte-identical output, so re-running is a no-op rather than a duplication risk. What
protects that: `ingest_ts` is stamped at capture and carried through, never recomputed;
run identity and `peak_rss_kb` live in the sidecar manifest, never in a column;
`snapshot_id` is a hash, never `uuid4`; row groups are bounded by row *count*, not bytes;
every input list and spill bucket is sorted.

## Book validity

A gap does not set a flag — it *terminates a book epoch* and opens a new one. So:

> A reconstructed book is valid iff every delta shares one `book_epoch_id` with its
> anchoring snapshot, and that epoch has `opened_with_snapshot` and `is_contiguous`.

The expensive check happens once, at write time, by the process that observed the stream.
`orderbook_snapshots_v` and `orderbook_deltas_v` expose it as `book_valid`. An epoch that
fails is a known-unreconstructable interval: the answer there is *unknown*, never *empty book*.

## Athena

Database `kalshi`, partition projection (no crawler). Fact tables partition by `dt`
(ET date); `events`/`series`/`market_id_xref` by `ym`, because 365 files of 50 KB is worse
than 12 files of 2 MB and Glacier IR bills a 128 KB minimum per object.

`series_ticker` is deliberately **not** a partition key — 1,172 series × 365 days is 428k
mostly-empty partitions. It is the leading sort column instead, so row-group min/max
statistics prune it for free.

Query the `_v` views, not the raw tables: they expose dollars via `DECIMAL` division
(never `/1e6`, which is float and reintroduces exactly what micros prevents).

```sql
SELECT ticker, created_time, yes_price_dollars, count_contracts
FROM   kalshi.trades_v
WHERE  dt = '2026-08-20' AND series_ticker = 'KXINX'
ORDER  BY created_time;
```

Always predicate on `dt`/`ym`: without one, projection enumerates the whole range.
