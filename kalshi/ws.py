"""Kalshi market data over WebSocket: a live order book, our fills, and our positions.

WHY THIS EXISTS -- AND IT IS A BUDGET ARGUMENT, NOT A LATENCY ONE.

Kalshi meters REST reads at 300 tokens/s per API key, 10 tokens per request. It does not meter
WebSocket frames at all: the token bucket in the OpenAPI spec is defined over HTTP `method`+`path`
pairs (`kalshi_api.yaml`, `EndpointTokenCost`), `/account/limits` exposes exactly two buckets --
`read` and `write` -- and the AsyncAPI spec mentions no tokens anywhere. Its only limit is error
27, *"the subscription exceeded its COMMAND rate limit"*: commands sent, not frames received.

Our own capture proves the economics. `var/log/ws-recorder.log`, 2026-08-21: **four subscribe
commands covering 532 markets, then 3,085,362 frames, zero resubscribes.** At REST prices those
frames would be 30 million tokens against a budget of 300 a second.

So the full SPX/NDX universe -- 522-532 markets -- can carry a live book here for the cost of a
handful of commands, where `/markets/orderbooks` would cost 60 tokens every poll and still only
be as fresh as the poll interval.

WHAT IT DOES NOT DO. It is not a second recorder. `storage/ws_recorder` owns capture-to-disk and
is a different job with different failure modes (a dropped frame there is permanent data loss; a
dropped frame here is a book we refuse to quote until it resyncs). The two share the ONE thing
that must not be written twice -- the delta arithmetic -- by importing `storage.book_replay.Book`
rather than reimplementing it. This repo has already paid twice for the same computation existing
in two places: `theo.parity_spot` against `atmvol.two_sided`, and `skew.ladder_monotone` running
inside `skew.fit` but not on the smoothed curve the live pricer actually installed.

VALIDITY IS THE POINT, exactly as in `book_replay`. A book is quotable only if a snapshot has been
seen for it AND no sequence gap has occurred since. `seq` is per SUBSCRIPTION, not per market, so
one gap invalidates every book on that subscription until each is re-snapshotted. `entry()`
returns None for an invalid book and NEVER a stale or half-applied one -- a pricer treats a
present book as current, so serving a doubtful one is worse than serving nothing.

    feed = MarketFeed(tickers).start()
    books = feed.books()          # {ticker: <shape quoting.top_of_book understands>}
    for f in feed.drain_fills():  # our own executions, since the last drain
        ...
"""
import json
import threading
import time
import urllib.parse

import config
from kalshi import client as k
from storage.book_replay import Book
from storage.units import to_cc, to_micros

# `fill` and `market_positions` are subscribed alongside the book because they are the two things
# the quote loop currently CANNOT see. `strategy/orders.RestingBook` is seeded from create
# responses and never cross-checked against the exchange until `cancel_all` at exit, so a fill
# leaves a phantom order that `plan_side` keeps measuring as if it were still resting. Both
# channels are free here and neither has a REST equivalent that would be.
CHANNELS = ("orderbook_delta", "fill", "market_positions")

# Which of them are scoped to MARKETS rather than to the account. `fill` and `market_positions`
# take an OPTIONAL market filter and deliver everything when it is omitted, so they must be
# subscribed without one -- see the note in `_subscribe` for what filtering them cost.
MARKET_SCOPED_CHANNELS = ("orderbook_delta",)

# Frame types that consume a sequence number. Control responses advance `seq` too, so counting
# only data frames reports a phantom gap on every `ok`. Taken from `storage.book_replay`, where
# the same mistake was already made and fixed.
SEQ_BEARING = {"orderbook_delta", "orderbook_snapshot", "ok", "unsubscribed"}

# Terminal channel errors: the subscription is gone and must be rebuilt. `storage/ws_recorder`
# handles the same three.
TERMINAL_ERRORS = {10, 17, 25}

# Error 26 is "adding markets would exceed the per-subscription market limit"; 27 is "the
# subscription exceeded its command rate limit". NEITHER IS QUANTIFIED ANYWHERE -- not in the
# vendored AsyncAPI spec, not in any doc on this box. What we know empirically is that 532 markets
# on one subscription is accepted. So the pacer below is deliberately conservative, and both codes
# back off rather than retrying into the same wall.
ERR_MARKET_LIMIT, ERR_TOO_MANY_COMMANDS = 26, 27

# The command pacer. Nothing in this codebase has ever paced WS commands, because nothing has ever
# sent them in anger: the recorder subscribes once a session and adds markets every 5 minutes at
# most. A quoter that resyncs per-market on every gap is a different traffic shape entirely, and a
# gap storm becoming a command storm is exactly how you meet error 27 for the first time in
# production. One command in flight, spaced, with an exponential back-off on 27.
COMMAND_MIN_INTERVAL_S = 0.25
COMMAND_BACKOFF_S = 2.0
COMMAND_BACKOFF_MAX_S = 30.0

# Kalshi's own per-subscription cap is unknown; this is ours, and it is under the 532 we have seen
# accepted so that a resync request can never itself be the thing that trips error 26.
MAX_SNAPSHOT_REQUEST = 100

# Markets per `subscribe` / `add_markets` command. There is no documented ceiling on the command
# itself, so this is set by MEASUREMENT rather than by spec: the largest `market_tickers` list
# this repo has ever had accepted is 532, and the four-board universe is 3,290. 500 sits just
# under the known-good size and costs 7 commands at startup rather than one of unknown validity.
#
# Not smaller, because commands are the scarce currency here -- error 27 is a command-RATE
# refusal, not a size one -- and `_send` paces them at COMMAND_MIN_INTERVAL_S. At 100 the same
# universe would be 33 paced commands, i.e. ~8 s before the book is fully subscribed.
MAX_SUBSCRIBE_MARKETS = 500


def _fp(micros):
    """int64 micro-dollars -> the fixed-point dollar string Kalshi and `quoting.ladders` use."""
    return f"{micros / 1_000_000:.4f}"


def _cc(centi):
    return f"{centi / 100:.2f}"


class MarketFeed:
    """One authenticated socket, one book per market, plus fills and positions.

    Threading follows `massive.client.QuoteCollector` rather than the recorder's asyncio: the
    quote loop's `cycle()` is synchronous and already drives one feed this way, and two feeds with
    two concurrency models in one loop is a cost with no benefit.
    """

    def __init__(self, tickers, channels=CHANNELS, tee=None):
        self.tickers = list(tickers)
        self.channels = tuple(channels)
        # OPTIONAL TAPE. `storage.feed_tee.FeedTee`, or anything with `offer(recv_ns, frame)`.
        # The feed does not own it, does not start it and does not stop it -- the quote loop does,
        # because the tape's lifetime is the session's and not this socket's (a reconnect must not
        # roll a segment). All this class owes it is every frame, once, without ever waiting on it.
        self.tee = tee
        self._books = {}                    # ticker -> Book
        self._valid = {}                    # ticker -> bool (snapshot seen, no gap since)
        self._fills = []                    # drained by the caller
        self._positions = {}                # ticker -> latest position frame
        self._lock = threading.Lock()
        self._ws = None
        self._thread = None
        self._stop = threading.Event()

        self._cmd_lock = threading.Lock()
        self._cmd_id = 0
        self._next_cmd_t = 0.0
        self._cmd_backoff = 0.0
        self._sids = {}                     # channel -> sid

        self.seq = None
        self.gaps = 0
        self.frames = 0
        self.reconnects = 0
        self.commands = 0
        self.command_errors = []
        self.pump_error = None
        self.last_frame_t = None
        self.started_t = None

    # -- lifecycle ---------------------------------------------------------
    def start(self, attempts=4, backoff=3.0):
        import websocket                    # imported here so `--no-ws` never needs the dep
        base = urllib.parse.urlparse(config.KALSHI_WS_BASE)
        for attempt in range(attempts):
            try:
                # Signed exactly as `storage/ws_recorder` signs it: `_auth_headers` builds the
                # RSA-PSS headers and -- this is the part that makes the whole module free --
                # never calls `_throttle`. The handshake is not a metered request.
                hdrs = k._auth_headers("GET", base.path)
                hdrs.pop("Content-Type", None)
                # `suppress_origin=True` IS LOAD-BEARING. `websocket-client` sends an `Origin:`
                # header on every handshake by default, and Kalshi's edge answers a WebSocket
                # upgrade carrying one with **403 Forbidden** -- before authentication, so the
                # signature never gets looked at and the failure reads like a credential problem.
                # Measured 2026-08-25: identical headers, identical signing path, 403 with the
                # Origin and an immediate `subscribed` without it.
                #
                # `storage/ws_recorder` never hit this because `websockets.asyncio` sends no
                # Origin unless asked. Two libraries, one endpoint, and only one of them works out
                # of the box -- worth the four lines to say so.
                self._ws = websocket.create_connection(
                    config.KALSHI_WS_BASE, timeout=20, suppress_origin=True,
                    header=[f"{h}: {v}" for h, v in hdrs.items()])
                break
            except Exception:                                # noqa: BLE001
                self._close_socket()
                if attempt == attempts - 1:
                    raise
                time.sleep(backoff)

        self._ws.settimeout(1.0)
        self.started_t = time.monotonic()
        # THE PUMP STARTS BEFORE THE SUBSCRIBE, and the order matters as of 2026-08-26. A
        # chunked subscribe has to read the first chunk's `subscribed` ack to learn the sid the
        # remaining chunks add markets to, and that ack is recorded by the pump. Subscribing
        # first would deadlock `_await_sid` against a reader that has not been started.
        # Starting the pump early is free: `recv` simply times out until frames arrive.
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        self._subscribe(self.tickers)
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._close_socket()
        return self

    def _close_socket(self):
        try:
            if self._ws:
                self._ws.close()
        except Exception:                                    # noqa: BLE001
            pass
        self._ws = None

    @property
    def alive(self):
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    # -- commands ----------------------------------------------------------
    def _send(self, payload):
        """Send one command, paced. Returns its id, or None if the socket is gone.

        The pacing is the only backpressure that exists on this side of the connection, and it is
        ours alone -- Kalshi tells us we exceeded a command rate only by refusing (error 27).
        """
        with self._cmd_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_cmd_t - now)
            if wait:
                time.sleep(wait)
            self._cmd_id += 1
            payload["id"] = self._cmd_id
            try:
                self._ws.send(json.dumps(payload, separators=(",", ":")))
            except Exception as e:                           # noqa: BLE001
                self.command_errors.append(f"send: {type(e).__name__}: {str(e)[:60]}")
                return None
            self.commands += 1
            self._next_cmd_t = (time.monotonic()
                                + COMMAND_MIN_INTERVAL_S + self._cmd_backoff)
            return self._cmd_id

    def _subscribe(self, tickers, chunk=None, sid_timeout=10.0):
        """Subscribe to `tickers`, in chunks, keeping ONE sid per channel.

        CHUNKED SINCE 2026-08-26, when the universe went from one underlying to four boards. This
        used to send the whole list in a single `subscribe` command; the largest list that had
        ever been accepted was 532 markets, and the four-board universe is 3,290. Whether a
        command that size is accepted is not something to discover with real orders resting.

        The first chunk is a real `subscribe` and establishes the sids. Every chunk after it is
        `update_subscription` / `add_markets` against the sid the first one returned, NOT another
        `subscribe` -- a second subscribe would open a second subscription with its own sid, and
        `request_snapshot` reads exactly one sid per channel, so half the board would silently
        stop being re-snapshotted after a gap.

        That means waiting for the first ack before the rest can be sent, which is why this takes
        a timeout. If the sid never arrives we fall back to sending the remainder as plain
        `subscribe` commands: a duplicated sid is a degraded snapshot path, and no subscription at
        all is a blind book.
        """
        want = list(tickers)
        if not want:
            return None
        chunk = chunk or MAX_SUBSCRIBE_MARKETS

        # THE ACCOUNT CHANNELS TAKE NO MARKET FILTER, and getting this wrong cost a live session.
        #
        # `fill` and `market_positions` are scoped to the ACCOUNT: the spec says the market
        # specification is "optional ... omit to receive all your fills". Passing them a ticker
        # list FILTERS them. When the subscribe was one unchunked command that was harmless
        # because the list was the whole board; the moment it was chunked (2026-08-26) the account
        # channels silently became "the first 500 tickers only", and every fill outside that
        # window stopped arriving.
        #
        # Measured, live, the same day: the loop reported **0 fills** across a whole session while
        # `/portfolio/fills` showed a full page of them and five open positions worth $63.78. So
        # `RestingBook` believed orders were still resting after they had traded, `reconcile`
        # re-planned a cancel for each one every cycle (the steady ~7 `not_found` per cycle),
        # `risk.on_fills` never fired -- the loss limit was blind while real money moved -- and
        # the inventory gate saw a flat book it did not have.
        #
        # So they are subscribed WITHOUT a market list, which is both the fix and strictly better
        # than the pre-chunking behaviour: fills on a market we stopped quoting now arrive too.
        market_scoped = [c for c in self.channels if c in MARKET_SCOPED_CHANNELS]
        account_scoped = [c for c in self.channels if c not in MARKET_SCOPED_CHANNELS]

        first = None
        if account_scoped:
            first = self._send({"cmd": "subscribe",
                                "params": {"channels": account_scoped}})
        if not market_scoped:
            return first

        cmd_id = self._send({"cmd": "subscribe", "params": {
            "channels": market_scoped,
            "market_tickers": want[:chunk],
        }})
        first = first or cmd_id
        rest = want[chunk:]
        if not rest:
            return first
        # Every chunk after the first is `add_markets` against the sid the first one returned, NOT
        # another `subscribe` -- a second subscribe would open a second subscription with its own
        # sid, and `request_snapshot` reads exactly one sid per channel, so half the board would
        # silently stop being re-snapshotted after a gap. That means waiting for the first ack,
        # which is why the pump starts before this runs.
        sid = self._await_sid(market_scoped[0], timeout=sid_timeout)
        for i in range(0, len(rest), chunk):
            batch = rest[i:i + chunk]
            if sid is None:
                self.command_errors.append(
                    f"subscribe: no sid after {sid_timeout:.0f}s, "
                    f"adding {len(batch)} markets as a second subscription")
                self._send({"cmd": "subscribe", "params": {
                    "channels": market_scoped, "market_tickers": batch}})
            else:
                self._send({"cmd": "update_subscription", "params": {
                    "sids": [sid], "action": "add_markets", "market_tickers": batch}})
        return first

    def _await_sid(self, channel, timeout=10.0):
        """Block until the pump records this channel's sid, or give up. None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sid = self._sids.get(channel)
            if sid is not None:
                return sid
            if self._stop.is_set() or self._ws is None:
                return None
            time.sleep(0.05)
        return self._sids.get(channel)

    def request_snapshot(self, tickers):
        """`update_subscription` / `get_snapshot` -- re-snapshot markets WITHOUT resubscribing.

        Cheaper than a resubscribe in the one currency that is scarce here: commands. The spec
        says `get_snapshot` "returns an `orderbook_snapshot` for the requested `market_tickers`
        without modifying the subscription", so a gap costs one command per chunk rather than an
        unsubscribe plus a subscribe plus a full 532-market snapshot burst.
        """
        sid = self._sids.get("orderbook_delta")
        if sid is None:
            return 0
        sent = 0
        want = list(tickers)
        for i in range(0, len(want), MAX_SNAPSHOT_REQUEST):
            if self._send({"cmd": "update_subscription", "params": {
                    "sids": [sid], "action": "get_snapshot",
                    "market_tickers": want[i:i + MAX_SNAPSHOT_REQUEST]}}):
                sent += 1
        return sent

    # -- the pump ----------------------------------------------------------
    def _pump(self):
        import websocket
        while not self._stop.is_set():
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:                           # noqa: BLE001
                self.pump_error = f"{type(e).__name__}: {str(e)[:80]}"
                return
            if not raw:
                continue
            # Stamped at RECEIPT, before parsing, because it is the only timestamp in the record
            # that is ours: the exchange's own `ts` is when it decided, not when we learned.
            recv_ns = time.time_ns()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            self.frames += 1
            self.last_frame_t = time.monotonic()
            if self.tee is not None:
                # BEFORE `_handle`, and outside its try. Two reasons: a frame that makes the
                # handler throw is exactly the frame worth having on the tape, and `offer` is a
                # bounded put that cannot block or raise, so there is nothing here for the
                # handler's exception guard to catch.
                self.tee.offer(recv_ns, msg)
            try:
                self._handle(msg)
            except Exception as e:                           # noqa: BLE001
                # One malformed frame must not take the feed down; it invalidates its own book
                # through the delta arithmetic and the next snapshot repairs it.
                self.pump_error = f"handle: {type(e).__name__}: {str(e)[:80]}"

    def _handle(self, msg):
        typ = msg.get("type")
        if typ == "subscribed":
            p = msg.get("msg") or {}
            self._sids[p.get("channel")] = p.get("sid")
            return
        if typ == "error":
            self._on_error(msg)
            return

        seq = msg.get("seq")
        if typ in SEQ_BEARING and seq is not None:
            if self.seq is not None and seq != self.seq + 1:
                # A gap on the SUBSCRIPTION invalidates every book on it, not just the market the
                # missing frame belonged to -- we do not know which market it was. Same rule as
                # `storage/book_replay`, same reason.
                self.gaps += 1
                with self._lock:
                    for t in self._valid:
                        self._valid[t] = False
                self.request_snapshot(list(self._books))
            self.seq = seq

        body = msg.get("msg") or {}
        if typ == "orderbook_snapshot":
            self._on_snapshot(body, seq)
        elif typ == "orderbook_delta":
            self._on_delta(body)
        elif typ == "fill":
            with self._lock:
                self._fills.append(body)
            # A fill carries `post_position_fp` -- our net position in that market AFTER it -- so
            # the position is current the instant we learn of the trade, without waiting for the
            # market_position channel to catch up. This is the primary source precisely because it
            # cannot be out of step with the fill it arrived with.
            self._set_position(body.get("market_ticker"), body.get("post_position_fp"))
        elif typ == "market_position":
            # SINGULAR. `market_positions` is the CHANNEL name; the frame's own `type` is
            # `market_position` (kalshi_websocket_api.yaml, marketPositionPayload). Matching the
            # plural here -- which this did until 2026-08-25 -- silently drops every position
            # frame, and the unit test pinned the same wrong spelling so the gate stayed green.
            # The field is `position_fp`, a fixed-point NET position, not `position`.
            self._set_position(body.get("market_ticker"), body.get("position_fp"))

    def _on_error(self, msg):
        body = msg.get("msg") or msg
        code = body.get("code")
        self.command_errors.append({"code": code, "msg": str(body.get("msg"))[:80]})
        if code == ERR_TOO_MANY_COMMANDS:
            # Back off multiplicatively. The limit is unpublished, so the only way to find it is
            # to stop asking for a while -- retrying at the same cadence just meets it again.
            with self._cmd_lock:
                self._cmd_backoff = min(COMMAND_BACKOFF_MAX_S,
                                        max(COMMAND_BACKOFF_S, self._cmd_backoff * 2))
        elif code == ERR_MARKET_LIMIT:
            # We asked for more markets than one subscription holds. There is no recovery here
            # that this module can perform honestly -- the universe has to shrink or be split
            # across subscriptions -- so it is surfaced and nothing is silently dropped.
            pass
        elif code in TERMINAL_ERRORS:
            with self._lock:
                for t in self._valid:
                    self._valid[t] = False
            self.reconnects += 1
            self._subscribe(self.tickers)

    def _on_snapshot(self, body, seq):
        t = body.get("market_ticker")
        if not t:
            return
        with self._lock:
            b = self._books.setdefault(t, Book())
            b.load(body, time.time_ns(), seq)
            self._valid[t] = True
            # A snapshot repays the command backoff: the stream is answering us again.
            self._cmd_backoff = 0.0

    def _on_delta(self, body):
        t = body.get("market_ticker")
        with self._lock:
            b = self._books.get(t)
            if b is None or not self._valid.get(t):
                # No snapshot yet, or invalidated by a gap. Applying a delta to an unknown base
                # would produce a book that looks fine and is wrong.
                return
            try:
                b.apply(body)
            except ValueError:
                # `Book.apply` raises rather than clamping a negative size, which is arithmetic
                # proof of a frame we never saw. Refuse the book and ask for a fresh snapshot.
                self._valid[t] = False
                self.gaps += 1
                self.request_snapshot([t])

    # -- what the quote loop reads ----------------------------------------
    def entry(self, ticker):
        """One market's book in the shape `strategy.quoting.top_of_book` already parses, or None.

        Levels are `[price_string, size_string]` ASCENDING, so the best is the LAST -- that is
        what `quoting.ladders` documents against a real `/markets/orderbooks` payload, and
        matching it exactly is why `build_desired` needs no change to move from REST to WS.

        None means "we do not know", never "empty". An empty book and an invalidated one are
        different answers and the ladder must not confuse them.
        """
        with self._lock:
            if not self._valid.get(ticker):
                return None
            b = self._books.get(ticker)
            if b is None:
                return None
            return {"yes": [[_fp(p), _cc(s)] for p, s in sorted(b.yes.items())],
                    "no": [[_fp(p), _cc(s)] for p, s in sorted(b.no.items())]}

    def books(self, tickers=None):
        """{ticker: entry} for every market whose book is currently trustworthy."""
        want = self.tickers if tickers is None else tickers
        out = {}
        for t in want:
            e = self.entry(t)
            if e is not None:
                out[t] = e
        return out

    def drain_fills(self):
        """Our executions since the last call. Draining is deliberate: each fill must be applied
        to `RestingBook` exactly once."""
        with self._lock:
            out, self._fills = self._fills, []
        return out

    def _set_position(self, ticker, value_fp):
        """Record a net position from a fixed-point string. Ignores absent/unparseable values.

        SIGNED, and the sign is the whole contract: positive is long YES, negative is long NO
        (i.e. short YES). A bid ADDS to it and an ask REDUCES it, which is what the inventory gate
        in `run_04.build_desired` depends on.
        """
        if not ticker or value_fp is None:
            return
        try:
            n = float(value_fp)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._positions[ticker] = n

    def seed_positions(self, positions):
        """Prime the position map from a REST read, for markets the socket has not spoken about.

        `market_positions` is a DELTA channel -- "updates sent when your position changes" -- so a
        position opened before this process started is invisible to it until it next moves. On a
        cold start that is harmless (we are flat). After a restart mid-session it is not: the
        inventory gate would read flat, keep quoting the side that ADDS, and the loss limit would
        mark a position it does not know it has.

        Only fills gaps. A ticker the socket has already reported wins, because that value came
        from the exchange after this connection opened and the REST read may predate it.
        """
        with self._lock:
            for tk, n in (positions or {}).items():
                if tk not in self._positions:
                    self._positions[tk] = float(n)
            return len(self._positions)

    def positions(self):
        """{ticker: net contracts}, signed. Absent means we have never held that market."""
        with self._lock:
            return dict(self._positions)

    def position(self, ticker):
        """Net contracts in one market, 0.0 if we have never held it.

        0.0 for unknown is correct here rather than None: a market we have never traded is flat,
        and an inventory gate that treated "unknown" as "refuse to quote" would refuse the whole
        board on a cold start.
        """
        with self._lock:
            return self._positions.get(ticker, 0.0)

    def health(self, now=None):
        t = time.monotonic() if now is None else now
        with self._lock:
            n_valid = sum(1 for v in self._valid.values() if v)
        return {"alive": self.alive,
                "pump_error": self.pump_error,
                "frame_age_s": (None if self.last_frame_t is None else t - self.last_frame_t),
                "n_books": len(self._books),
                "n_valid": n_valid,
                "n_wanted": len(self.tickers),
                "coverage": (n_valid / len(self.tickers)) if self.tickers else 0.0,
                "seq": self.seq, "gaps": self.gaps, "frames": self.frames,
                "commands": self.commands, "reconnects": self.reconnects,
                "command_backoff_s": self._cmd_backoff,
                "command_errors": self.command_errors[-3:]}

    def wait_for_coverage(self, fraction=0.8, timeout=20.0, poll=0.25):
        """Block until `fraction` of the wanted markets have a valid book, or time out.

        Returns the coverage reached. The caller decides what is enough -- a quoter that refused
        to start below 100% would never start, because a 532-market snapshot burst always has a
        tail.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cov = self.health()["coverage"]
            if cov >= fraction:
                return cov
            time.sleep(poll)
        return self.health()["coverage"]
