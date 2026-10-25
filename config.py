"""Shared configuration: credentials, endpoints, paths, and every tunable of the quoting engine.

Secrets are read from .env.kalshi / .env.massive (see the .example files) and never inlined here.
Tuned values in this public copy are ILLUSTRATIVE: the live values are private.
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# (RUNS_DIR removed 2026-08-24 -- data/runs/ was run_03's per-cycle capture. The quote loop
#  journals to var/journal/ via strategy/journal.py and nothing read RUNS_DIR.)

# Kalshi settles on the *Eastern* close. This must be a real zone, not a fixed
# offset: a hardcoded UTC-4 is correct only during EDT and silently misfiles every
# 23:00-00:00 timestamp for the five months of EST. The storage layer derives its
# dt= partitions and its 23:59 seal from this, so the error would be permanent.
# Note also that 2026-11-01 is a 25-hour local day and 2026-03-08 a 23-hour one --
# never assume a day is 86400 seconds.
ET = ZoneInfo("America/New_York")


def _load_env(path):
    """Minimal .env parser (python-dotenv is not installed)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_kalshi = _load_env(ROOT / ".env.kalshi")
_massive = _load_env(ROOT / ".env.massive")

KALSHI_API_BASE = _kalshi["KALSHI_API_BASE"].rstrip("/")
KALSHI_WS_BASE = _kalshi.get("KALSHI_WS_BASE", "")
KALSHI_KEY_ID = _kalshi["KALSHI_API_KEY_ID"]
KALSHI_PRIVATE_KEY_PATH = _kalshi["KALSHI_PRIVATE_KEY_PATH"]

MASSIVE_API_KEY = _massive["MASSIVE_API_KEY"]
MASSIVE_REST_BASE = "https://api.massive.com"
# Real-time cluster. The `delayed.` host authenticates fine but lags 900s --
# verified empirically. Do not switch it without re-checking frame timestamps.
MASSIVE_WS_OPTIONS = "wss://socket.massive.com/options"
# VERIFIED ON THIS ACCOUNT 2026-08-26, against a docstring in `strategy/deltaadj` that had claimed
# since it was written that no futures WebSocket existed here. Both connect, authenticate with our
# key, and stream -- and all THREE sockets were open in one process simultaneously with no
# `max_connections` refusal, so the one-per-account limit is on the OPTIONS socket alone.
#
# Measured, 60s side by side:            REST (what we ran)      WebSocket
#   futures age                          MED 320ms / p95 787ms   MED  73.6ms / p95 95.5ms
#   futures message rate                 4/s                     114-220/s
#   index age                            MED 711ms / p95 1472ms  MED  21.3ms / p95 31.6ms
#   per-read TCP+TLS handshake           40.4ms, every read      103ms ONCE, at connect
#
# The index still publishes on a one-second grid -- that is the vendor recomputing it, not a read
# we can hurry -- but 690ms of the 711ms we used to see was our own REST polling. See
# `massive/streams`.
MASSIVE_WS_FUTURES = "wss://socket.massive.com/futures"
MASSIVE_WS_INDICES = "wss://socket.massive.com/indices"

# Target market
#
# THE BOARDS THE QUOTER PRICES, keyed by UNDERLYING.
#
# `QUOTE_BOARDS` is the whole definition: {underlying: (daily series, hourly series)}. Underlying
# is the right key because everything a board needs that is not the market list is a fact about
# the underlying and never about the hour -- the option strip (SPXW vs NDXP), the variance
# calendar, the realized-vol model, the smile, and the futures contract the theo is carried
# forward on (ES vs NQ). `strategy/run_04_quote_loop.Board` is one of these.
#
# NDX ADDED 2026-08-26. It needed no new modelling: the variance calendar, the gap calendar, the
# realized-vol model, the edge curve and the NQ front-month resolution all already existed and
# load clean. What it needed was wiring -- per-underlying edge manager, lean chain, smile and
# delta adjuster, a second `/markets` request per series, and a spot resolution of its own.
#
# NOT a capacity increase. Selection is capital-bound at roughly 41 ladders on
# MAX_RESTING_ORDER_VALUE, so NDX DISPLACES SPX rather than adding to it: on measured live
# boards NDX takes the majority of the slots (69 of 125 at 11:00, 94 of 125 at 15:30) because
# vega and theta are not unit-scaled by the index level and NDX lists ~2,000 quotable markets
# against SPX's ~330. Raising the cap is a separate, deliberate decision.
QUOTE_BOARDS = {
    "SPX": (("KXINX",), ("KXINXU",)),
    "NDX": (("KXNASDAQ100",), ("KXNASDAQ100U",)),
}

# Derived, and kept because `discover_boards`, `cancel_all` and the validators all want the flat
# tuple rather than the mapping. Sorted so the order is stable across runs.
QUOTE_DAILY_SERIES = tuple(sd for _u, (d, _h) in sorted(QUOTE_BOARDS.items()) for sd in d)
QUOTE_HOURLY_SERIES = tuple(sh for _u, (_d, h) in sorted(QUOTE_BOARDS.items()) for sh in h)

# THE HOURLY BOARD IS BACK ON as of 2026-08-26, having been emptied for the first `--live`
# session on 2026-08-25 so that session tested one new thing rather than two. It prices, it
# passes `validate_hourly` and `validate_boards`, and it ran clean for 45 minutes of dry quoting
# on a live board (390 markets, 0 errors, 0 skipped cycles).
#
# An empty tuple remains a clean no-op, not a special case: `discover_boards` iterates it, and
# `validate_theo.resolve_and_price_boards` only builds a HourlyPricer when `hourly_markets` is
# non-empty. To pull one board, empty its half of `QUOTE_BOARDS`; to pull an underlying, remove
# its key. `strategy/cancel_all` deliberately names every series regardless -- cancelling an
# event we never quoted is free, and it is the wrong direction to be economical in.

SERIES_TICKER = "KXINX"
KALSHI_CATEGORY = "Financials"  # exact category string; "Finance" is not one
OPTION_UNDERLYING = "SPX"       # contracts API key; tickers come back as SPXW
STOCK_SYMBOL = "SPY"

# Flat half-width, bid = theo - offset and ask = theo + offset. Only with `--edge` off. Illustrative
# value; the live value is not published.
QUOTE_OFFSET = 0.02
# Half-width, as a fraction of spot, of the option strikes that feed put-call parity. Illustrative
# value; the live value is not published.
QUOTE_BAND_PCT = 0.02
# Past this age the index level is not used as spot; fall back to parity. Illustrative value; the
# live value is not published.
INDEX_MAX_STALE_S = 5.0

# If no option quote at all has arrived for this long, the socket is dead and the cycle is refused.
# Illustrative value; the live value is not published.
MASSIVE_MAX_QUOTE_AGE_S = 10.0

# Alarm (not a gate) when |index - parity| exceeds this fraction of spot. Illustrative value; the
# live value is not published.
INDEX_PARITY_ALARM_PCT = 0.001

# One "vol point" of RV/IV disagreement. Illustrative value; the live value is not published.
LEAN_VOL_SCALE = 0.02

# Typical sigma_rv / sigma_impl -- the variance risk premium. Only the DEVIATION from this is
# information; the level itself is a constant and belongs in QUOTE_OFFSET. Measured live on one
# quiet SPX board 2026-08-21: 4.79% realized against 7.87% implied, a ratio of 0.61 -- one
# observation, not a calibration. None keeps the lean inert.
# NOT ON THE LIVE PATH since 2026-08-25. The lean compares against RAW implied and deliberately
# harvests the variance risk premium -- an operator decision taken with the -9.33-points-from-
# premium-alone measurement in hand; see `strategy/leanchain.chain` rule 3. This constant survives
# only as a parameter of the retired `pricing.lean.side_offset` arm, which `backtest/analysis`
# still sweeps. Setting it changes nothing about a live quote.
LEAN_VRP_RATIO = None

# Realized vs implied vol for the RV lean is recomputed this often. Illustrative value; the live
# value is not published.
LEAN_RECOMPUTE_S = 3.0

# The RV lean changes required EDGE per side (never the theo or the price). It is uncentred: it
# deliberately harvests the variance risk premium.
LEAN_ENABLED = True

# --- near-term vol-spike widening (pricing/volspike) ----------------------
# s = log(sigma_1m / sigma_30m), both rvol forecasts on the same clock. extra edge on BOTH sides =
# VOLSPIKE_K_CENTS * clip(s - VOLSPIKE_S0, 0, VOLSPIKE_S_MAX) cents; s >= VOLSPIKE_S_PULL pulls the
# market for the cycle. OFF: with it off the desired quotes are bit-identical (strategy.test_volspike),
# but the signal is still computed and journalled every cycle to volspike.jsonl.
#
# THE FOUR NUMBERS ARE UNCALIBRATED PLACEHOLDERS, set only so the function has a sane shape. The
# signal is NARROW: on the 126 held-out sessions (candidate short models, 2026-09-29) s_1m has
# median -0.075 / -0.091 (SPX / NDX; not zero -- the 1m forecast runs below the 30m one), 95th
# pct 0.005 / 0.015, 99.5th 0.077 / 0.097, 99.9th 0.109 / 0.129. So S0 sits near the 95th pct,
# S_PULL near the 99.5-99.9th, and K is sized so the clip at S_MAX is 1c. Calibrating them against
# adverse selection is Step 5 of the 1m/3m plan and has not been done.
VOLSPIKE_ENABLED = False
VOLSPIKE_HORIZON = "1m"
VOLSPIKE_S0 = 0.01
VOLSPIKE_K_CENTS = 10.0
VOLSPIKE_S_MAX = 0.10
VOLSPIKE_S_PULL = 0.10

# The smile curve is refit this often. Illustrative value; the live value is not published.
SKEW_RECOMPUTE_S = 3.0
# The served curve is a recency-weighted mean of the fits from this many seconds. Illustrative
# value; the live value is not published.
SKEW_WINDOW_S = 60.0
# Half-life of that recency weighting. Illustrative value; the live value is not published.
SKEW_HALF_LIFE_S = 15.0

# Which strikes ATM vol is the median of: "strikes" = the ATM_VOL_STRIKES smallest |C-P|, "delta" =
# every strike with call delta strictly inside ATM_DELTA_LO..HI (no minimum count), widening to
# ATM_DELTA_LATE_LO..HI in the last ATM_DELTA_LATE_MINUTES when the caller passes `now`.
ATM_VOL_SELECTION = "strikes"
# Lower call-delta edge of the "delta" band. Illustrative value; the live value is not published.
ATM_DELTA_LO = 0.35
# Upper call-delta edge of the "delta" band. Illustrative value; the live value is not published.
ATM_DELTA_HI = 0.65
# Tunable. Illustrative value; the live value is not published.
ATM_DELTA_LATE_MINUTES = 45.0
# Lower call-delta edge of the late band. Illustrative value; the live value is not published.
ATM_DELTA_LATE_LO = 0.25
# Upper call-delta edge of the late band. Illustrative value; the live value is not published.
ATM_DELTA_LATE_HI = 0.75

# Legacy single-order size. It is also the default `count` of strategy/journal.py. Illustrative
# value; the live value is not published.
ORDER_COUNT = '5.00'

# Contracts per rung. Position limits are counted in lots. Illustrative value; the live value is not
# published.
LOT_SIZE = 10
# Size of each rung, one entry per rung. One entry means one quote per side. Illustrative value; the
# live value is not published.
QUOTE_TIER_COUNTS = ('10.00',)

# Master switch for the back rungs (inert while QUOTE_TIER_COUNTS has one entry).
TIERS_ENABLED = True

# Use the calibrated R -> edge curve. When on, no R means no quote.
EDGE_MANAGER_ENABLED = True

# The dollar Greeks, and the required edge built from them, are recomputed this often. Illustrative
# value; the live value is not published.
EDGE_RECOMPUTE_S = 20.0

# Tunable. Illustrative value; the live value is not published.
QUOTE_TIGHTEN_FRACTION = 0.5

# Tunable. Illustrative value; the live value is not published.
MIN_MINUTES_TO_CLOSE = 5.0

# Stop quoting an hourly market this many minutes before its settlement. Illustrative value; the
# live value is not published.
HOURLY_MIN_MINUTES_TO_CLOSE = 15.0

# --- requote deadband ----------------------------------------------------
# The values in force live in `strategy/orders.py` (FRONT_IMPROVE_TICKS, BACK_IMPROVE_TICKS), next
# to the code that reads them, along with the reasoning about queue position. `REQUOTE_TICKS` is a
# DERIVED view of them, resolved lazily to break the import cycle (strategy.orders imports config),
# so there is one source of truth and this reports it.
def __getattr__(name):
    if name == "REQUOTE_TICKS":
        from strategy import orders
        return (orders.FRONT_IMPROVE_TICKS, orders.BACK_IMPROVE_TICKS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Resting contracts allowed on one side of one market. Illustrative value; the live value is not
# published.
MAX_RESTING_CONTRACTS_PER_MARKET_SIDE = 30
# Resting contracts allowed across all markets. Illustrative value; the live value is not published.
MAX_RESTING_CONTRACTS_TOTAL = 4000
# Total collateral allowed at rest, in dollars. Illustrative value; the live value is not published.
MAX_RESTING_ORDER_VALUE = 2000.0

# Collateral allowed at rest in one market, in dollars. Illustrative value; the live value is not
# published.
MAX_RESTING_ORDER_VALUE_PER_MARKET = 30.0

MIN_PRICE = 0.01
MAX_PRICE = 0.99

# Halt the session once account equity (marked to the Kalshi mid) is down this much on the day.
# Illustrative value; the live value is not published.
DAILY_LOSS_LIMIT_DOLLARS = 500.0

# `touch var/HALT` from any shell stops the quoter within one cycle and takes the cancel path.
# A file rather than a signal or a config flag: the operator is in a different shell, needs no PID,
# and it still works on a process too wedged to answer anything else. Removing the file does not
# resume -- resuming is a decision made by starting the process again.
HALT_FILE = ROOT / "var" / "HALT"

# Account balance and positions are polled from REST this often for the loss limit. Illustrative
# value; the live value is not published.
RISK_ACCOUNT_POLL_S = 2.0
# Halt if the account snapshot is older than this. Illustrative value; the live value is not
# published.
RISK_ACCOUNT_STALE_HALT_S = 20.0
RISK_STATE_DIR = ROOT / "var" / "risk"


# Net position limit per in-band market, in lots. At the limit, quote only the side that flattens
# the position. Illustrative value; the live value is not published.
POSITION_LIMIT_LOTS_BAND = 4
# The same limit for wing markets. Illustrative value; the live value is not published.
POSITION_LIMIT_LOTS_WING = 2


# After leaving a price level, do not re-improve back onto it for this many seconds. Illustrative
# value; the live value is not published.
QUOTE_ZOMBIE_HOLD_S = 20.0


# Write tokens held back from quoting so cancels always have budget. Illustrative value; the live
# value is not published.
WRITE_RESERVE_TOKENS = 100.0


# Share of the account's REST read budget given to the quoter. Illustrative value; the live value is
# not published.
READ_SHARE_TRADING = 0.6
# Share given to the WebSocket recorder's market resolver. Illustrative value; the live value is not
# published.
READ_SHARE_CAPTURE = 0.3


# Share of markets that need a Kalshi WebSocket book before quoting starts. Illustrative value; the
# live value is not published.
WS_BOOK_MIN_COVERAGE = 0.8
# Microprice size exponent. An inline literal, not a named constant. Illustrative value; the live
# value is not published.
WS_BOOK_WARMUP_S = 20.0


def today_et():
    return datetime.now(ET).date()


def run_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def event_ticker_for(date=None):
    """KXINX-26AUG19H1600 -- the 4pm ET close event for a given date."""
    d = date or today_et()
    return f"{SERIES_TICKER}-{d:%y%b%d}".upper() + "H1600"


def occ_expiry(date=None):
    return f"{(date or today_et()):%y%m%d}"


# --- storage layer ---------------------------------------------------------
# Short-term raw WebSocket capture; lifecycle expires objects after 7 days.
S3_BUCKET_WS = os.environ.get("KALSHI_S3_BUCKET_WS", "example-kalshi-ws-capture")
# Long-term Parquet; Glacier IR at 365d, expiry at 1095d.
S3_BUCKET_HIST = os.environ.get("KALSHI_S3_BUCKET_HIST", "example-kalshi-historical")
AWS_REGION = "us-east-2"        # no region is configured on the box; pin it explicitly

SCHEMA_VERSION = 1
S3_PREFIX_WS = "ws"                     # raw jsonl.gz
S3_PREFIX_PARQUET = f"parquet/v{SCHEMA_VERSION}"
S3_PREFIX_RAW_ARCHIVE = f"raw/v{SCHEMA_VERSION}"
S3_PREFIX_MANIFEST = f"_manifests/v{SCHEMA_VERSION}"

WS_DIR = DATA_DIR / "ws"                # local capture buffer
SCRATCH_DIR = ROOT / "var" / "scratch"  # NEVER /tmp -- that is a tmpfs, i.e. RAM
# Fitted model artifacts read by the LIVE pricer: the intraday variance calendar that scales an
# ATM vol to a sub-4pm horizon. Deliberately not var/backtest (that is study cache, and this is a
# trading dependency) and not var/scratch (that is external-sort spill, and gets unlinked).
MODEL_DIR = ROOT / "var" / "model"

SEGMENT_ROLL_SECONDS = 15 * 60
SEGMENT_ROLL_BYTES = 64 * 1024 * 1024

# Canonical units -- prices as int64 micro-dollars, sizes as int64 centi-contracts -- are owned
# by `storage/units.py` (MICROS / CENTI), which is what every conversion actually calls. The
# duplicate scale constants that sat here were read by nothing; removed 2026-08-24.

# The markets recorded live. KXINX is the daily 4pm S&P range; KXINXU is the
# hourly above/below. Resolved to concrete event tickers at runtime -- the day's
# KXINX event is often still `unopened` at recorder start, so a startup-only
# market list would silently miss it.
# The markets recorded live: index/ETF **4pm close** markets only.
#
# The hourly S&P markets were dropped deliberately -- pricing them needs a vol
# model, and the current theo prices probabilities off SPX option quotes, which
# only lines up with the 4pm close.
#
# Kalshi lists no Dow or ETF close market today: KXDJI is an hourly one-off and
# there is no SPY/QQQ/IWM/DIA close series. So the universe is S&P and Nasdaq.
# Discovery (below) picks up a Dow or ETF listing automatically if one appears.
CAPTURE_CLOSE_SUFFIX = "H1600"          # 4pm ET close

# The digital is -dC/dK including the smile slope, not N(d2).
DIGITAL_SLOPE_TERM = True
CAPTURE_SERIES = (
    "KXINX",          # S&P 500 close range brackets      (30 markets)
    "KXNASDAQ100",    # Nasdaq-100 close range brackets   (30)
    "KXINXU",         # S&P 500 close above/below         (60)
    "KXNASDAQ100U",   # Nasdaq-100 close above/below     (400)
    "KXINXDUD",       # S&P 500 close up/down              (1)
    "KXNASDAQDUD",    # Nasdaq-100 close up/down           (1)
)
# Substrings that mark a Financials series as an index/ETF underlying. Used by
# the daily discovery sweep so a newly listed Dow or ETF close market is picked
# up without a code change.
CAPTURE_INDEX_PATTERNS = ("INX", "NASDAQ", "NDQ", "DJI", "DJX", "RUT",
                          "SPX", "SPY", "QQQ", "IWM", "DIA")

# orderbook_delta is the volume driver and the only unrecoverable one.
# trade maps almost 1:1 onto the REST Trade schema, so it is the conformance
# check. ticker gives an independent read on top-of-book to validate a book
# reconstructed from deltas.
# `fill` is our OWN executions, not public data -- it needs the authenticated connection the
# recorder already has. Verified the combined subscribe frame is accepted alongside the public
# channels, and the frame handler keeps non-book channels out of epoch accounting
# (ws_recorder: `sid_channel.get(sid) not in (None, "orderbook_delta")`). Scoped to
# market_tickers like the others, so it reports fills on the capture universe -- a superset of
# what we quote.
# --- the quoter's own tape -----------------------------------------------
#
# `run_04` tees its Kalshi socket's raw frames into the same rolling segments `ws_recorder`
# writes. It exists because the recorder has been gated off since 2026-08-22
# (`CAPTURE_SESSION_ENABLED = False`) and the last order-book tape in S3 is 2026-08-21, so both
# live trading sessions have no book behind them -- and order book data has no REST backfill path.
# `/portfolio/fills` and the decision journal say what we quoted and what filled; only the book
# says whether the fill was any good.
#
# ITS OWN CHANNEL, deliberately. Two writers sharing `channel=all` would have a replay apply the
# same delta twice -- same markets, same exchange, two sockets -- and a book built from duplicated
# deltas is wrong without looking wrong. `flush_to_s3` globs every channel, so the existing flush,
# promote and seal timers pick this up unchanged.
#
# Free in the currency that is scarce: Kalshi meters COMMANDS, not frames, and these frames are
# already arriving for the quoter's own book. The cost is disk -- budget several hundred MB a day
# gzipped at the four-board universe, against the ~99 MB/day the 522-market capture produced on
# 2026-08-21 -- and `var/journal` is already unpruned beside it (docs/PIPELINE.md §8 C2).
QUOTE_TEE_ENABLED = True
QUOTE_TEE_CHANNEL = "quoter"

# The SPOT tape: futures mid changes and index publications, in their own channel so a replay can
# never mix them with the Kalshi book. Written because the 2026-08-26 post-mortem could not
# decompose a fill's loss into `delta * dF` -- both feeds were live all afternoon and neither was
# recorded, leaving the 2s quote cycle as the finest spot series on disk.
SPOT_TEE_CHANNEL = "spot"

# The time a quote is exposed before it can be pulled (order-path latency). Illustrative value; the
# live value is not published.
QUOTE_EXPOSURE_WINDOW_S = 0.1

# Size of move, in sigmas of the exposure window, that counts as adverse. Illustrative value; the
# live value is not published.
ADVERSE_MOVE_SIGMAS = 1.5

# Pull or skip a quote when spot has moved adversely within the exposure window.
ADVERSE_GATE_ENABLED = True

CAPTURE_CHANNELS = ("orderbook_delta", "trade", "ticker", "fill")
CAPTURE_RESOLVE_SECONDS = 300


def today_et_date():
    """ET calendar date -- the dt= partition key."""
    return datetime.now(ET).date()


def et_day_bounds(d):
    """UTC [start, end) for an ET calendar day. DST-correct: this is 23, 24 or 25 hours."""
    from datetime import time as _t, timedelta as _td
    start = datetime.combine(d, _t(0, 0), tzinfo=ET)
    end = datetime.combine(d + _td(days=1), _t(0, 0), tzinfo=ET)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


# --- capture session gate --------------------------------------------------
# Live WebSocket capture is gated on the ET regular-session window. While the
# gate is closed the recorder holds no subscription and writes no segments, but
# the process stays up -- so switching capture on is a flag flip plus a
# `systemctl restart kalshi-ws-recorder`, not a redeploy.
#
# ON as of 2026-08-20 13:52 ET. This is the first live capture session: the
# recorder has never run inside a trading window before, so every minute from here
# is order-book history that has no other source -- Kalshi exposes no historical
# orderbook endpoint, and the queue-position study depends on this tape.
# Nothing else in the pipeline is gated -- flush, seal, promote and monitor keep
# running on their timers regardless.
# Turned off 2026-08-22: the SPX/NDX hourly + 4pm capture this was enabled for is
# complete (2026-08-20 and 08-21 are the two full sessions on tape), and the
# backtesting work downstream reads from S3, not from a live feed. The recorder
# process stays up and idles so the storage timers and monitor keep their
# continuity; the gate simply never opens. Flip back to True + restart the unit
# to resume -- nothing else needs changing.
CAPTURE_SESSION_ENABLED = False
CAPTURE_SESSION_OPEN = (9, 30)      # ET, inclusive -- NYSE regular open
# Exclusive. 16:15 rather than 16:00 on purpose: trading stops at the 4pm close
# but the last book and trade frames land right on it, and a hard 16:00:00 cut
# clips exactly the frames these markets settle against.
CAPTURE_SESSION_CLOSE = (16, 15)
CAPTURE_GATE_POLL_SECONDS = 30      # how often a closed gate re-checks itself


def in_capture_session(now=None):
    """True when live WebSocket capture is allowed.

    Exchange holidays are deliberately not modelled: on a holiday the recorder
    subscribes and records an empty tape, which costs nothing and is obvious in
    the data. A stale hardcoded holiday table fails the other way -- it silently
    skips a real trading day, and order book data has no REST backfill path.
    """
    if not CAPTURE_SESSION_ENABLED:
        return False
    now = now or datetime.now(ET)
    if now.weekday() >= 5:              # these markets do not list on weekends
        return False
    return CAPTURE_SESSION_OPEN <= (now.hour, now.minute) < CAPTURE_SESSION_CLOSE


# The quoter's own session window. SEPARATE FROM THE CAPTURE GATE ON PURPOSE, and this was a live
# defect: `run_04.verdict` split NO_SPOT (exit 1, alarms) from NO_SESSION (exit 0, silent) on
# `in_capture_session()`, which returns False unconditionally while CAPTURE_SESSION_ENABLED is off.
# That flag was flipped off 2026-08-22 for a storage reason with no trading intent, and the effect
# was that a total feed outage in the middle of a trading day reported NO_SESSION and exited 0.
# A storage decision must not be able to disarm a trading alarm, so the two are now two predicates.
#
# 16:00 exclusive, not 16:15: KXINX settles at 16:00 and SPXW stops quoting there, so a no-spot
# reading after the close is the market being shut, not a feed being down.
TRADING_SESSION_OPEN = (9, 30)
TRADING_SESSION_CLOSE = (16, 0)


def in_trading_session(now=None):
    """True when the boards we quote are open. Holidays deliberately not modelled -- see above:
    a holiday reports NO_SPOT and alarms once, which is the cheap direction to be wrong in."""
    now = now or datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return TRADING_SESSION_OPEN <= (now.hour, now.minute) < TRADING_SESSION_CLOSE


def capture_gate_reason(now=None):
    """Why the gate is closed right now, phrased for the recorder log."""
    if not CAPTURE_SESSION_ENABLED:
        return "CAPTURE_SESSION_ENABLED=False"
    now = now or datetime.now(ET)
    if now.weekday() >= 5:
        return f"weekend ({now:%a})"
    return (f"outside {CAPTURE_SESSION_OPEN[0]:02d}:{CAPTURE_SESSION_OPEN[1]:02d}"
            f"-{CAPTURE_SESSION_CLOSE[0]:02d}:{CAPTURE_SESSION_CLOSE[1]:02d} ET"
            f" (now {now:%H:%M})")


# Tunable. Illustrative value; the live value is not published.
SELECT_FLOOR_ONE_SIDED = 0.04
# Tunable. Illustrative value; the live value is not published.
SELECT_FLOOR_BRACKET = 0.02

# Hysteresis on selection, so a market does not flap in and out. Illustrative value; the live value
# is not published.
SELECT_HYSTERESIS = 0.8


# The quoting band is +/- this many sigmas of the remaining move around the forward. Illustrative
# value; the live value is not published.
BAND_SIGMA = 2.5

# Hysteresis: a market leaves the band only at BAND_SIGMA x this. Illustrative value; the live value
# is not published.
BAND_EXIT_MULT = 1.5

# Share of the collateral budget reserved for in-band markets. Illustrative value; the live value is
# not published.
BAND_CAPITAL_FRACTION = 0.5
