"""Live futures and index over the Massive WebSockets, and the basis that joins them.

WHY THIS REPLACES POLLING. Measured 2026-08-26, side by side in one 60-second window:

                        REST (what we ran)        WebSocket (this module)
    futures age         MED 320ms / p95 787ms     MED  73.6ms / p95 95.5ms
    futures rate        4/s (FUTURES_POLL_S)      114-220/s
    index age           MED 711ms / p95 1472ms    MED  21.3ms / p95 31.6ms
    per-read handshake  TCP 18.4 + TLS 22.1ms     ONCE, at connect

`massive/history.get` runs on `urllib.request.urlopen`, which sends `Connection: close`, so every
futures poll and every index snapshot paid a fresh TCP+TLS handshake -- 40.4ms, 29% of a 139ms
poll, again every 250ms. A WebSocket handshakes once (measured 103ms) and then delivers frames
with no further setup: 22,561 frames over 90s, 0 transport errors, 0 reconnects, an amortised
0.0046ms of setup per message.

THE TWO FEEDS ARE NOT THE SAME SHAPE, and this is the fact the design turns on:

  * The INDEX publishes on a hard one-second grid and EVERY publication is a new value --
    measured 58 messages in 60s, gaps p05 965ms / MED 1002ms / p95 1044ms, and **0% consecutive
    repeats**. It is low-rate and 100% information. Its one-second grid is the vendor's; the
    690ms of further lag we used to see on top of it was our own REST polling.

  * The FUTURES publish on every change to bid price, ask price, bid size OR ask size --
    measured 6,868 messages in 60s (114/s) carrying only **11 distinct mids**, i.e. **95.8%
    consecutive repeats**. They are not a higher sampling rate of a moving number; they are
    event-driven immediacy. When the price moves you know inside a p95 of 72ms instead of waiting
    out a one-second index grid.

That asymmetry is why `FuturesFeed` notifies only on an actual MID CHANGE. Waking a consumer on
every frame would be 114 wake-ups a second to learn nothing 96% of the time.

UNITS. The WebSocket `t` is MILLISECONDS while `/v3/snapshot`'s `last_updated` is NANOSECONDS --
same vendor, same concept, different unit, and this repo has been caught by it twice already (see
`massive/index.IndexQuote`). Everything here converts to nanoseconds at the boundary so that
`IndexQuote` and `pricing/spot.resolve` are untouched.

CONCURRENCY. Verified 2026-08-26: the options, futures and indices sockets were open in one
process simultaneously with no `max_connections` refusal. The one-connection-per-account limit
documented in `massive.client.QuoteCollector` is on the OPTIONS socket specifically.
"""
import collections
import json
import math
import threading
import time

import config
from massive import futures as fut
from massive import index as ix

FUTURES_WS = getattr(config, "MASSIVE_WS_FUTURES", "wss://socket.massive.com/futures")
INDICES_WS = getattr(config, "MASSIVE_WS_INDICES", "wss://socket.massive.com/indices")

NS = 1_000_000_000

# Mid changes retained per underlying for `FuturesFeed.move_scale`. At a measured 10-20 distinct
# mids a second this is roughly a minute of history -- long enough to estimate a short-horizon
# volatility, short enough to follow a regime change within about a minute.
HIST_MAX = 1200

# Minimum history before the move-scale estimate behind the adverse gate is trusted. Illustrative
# value; the live value is not published.
MIN_SCALE_TICKS = 20


class MassiveStreamError(RuntimeError):
    pass


class _Stream:
    """Connect, auth, subscribe, pump, reconnect. The half both feeds share."""

    RECONNECT_BACKOFF_S = 2.0
    MAX_BACKOFF_S = 30.0

    def __init__(self, url, subs, name, tee=None):
        self.url = url
        self.subs = list(subs)
        self.name = name
        # OPTIONAL TAPE, and it records UPDATES, not frames -- see `_pump`. `storage.feed_tee` or
        # anything with `offer(recv_ns, frame)`. The stream neither starts nor stops it; the quote
        # loop owns its lifetime, because the tape's span is the session's and not a socket's (a
        # reconnect must not roll a segment).
        self.tee = tee
        self._ws = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.frames = 0
        self.updates = 0            # frames that actually changed something
        self.reconnects = 0
        self.errors = 0
        self.last_error = None
        self.last_frame_t = None
        self.connected_at = None

    # -- lifecycle --

    def start(self, timeout=15.0):
        """Connect and begin pumping. Returns self; never raises."""
        try:
            self._connect()
        except Exception as e:                               # noqa: BLE001
            self.last_error = f"connect: {type(e).__name__}: {str(e)[:120]}"
            self.errors += 1
            return self
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _connect(self):
        import websocket
        ws = websocket.create_connection(self.url, timeout=15)
        ws.recv()                                            # {"ev":"status","status":"connected"}
        ws.send(json.dumps({"action": "auth", "params": config.MASSIVE_API_KEY}))
        auth = ws.recv()
        if "auth_success" not in str(auth):
            try:
                ws.close()
            except Exception:                                # noqa: BLE001
                pass
            raise MassiveStreamError(f"{self.name} auth failed: {str(auth)[:160]}")
        for s in self.subs:
            ws.send(json.dumps({"action": "subscribe", "params": s}))
        ws.settimeout(1.0)
        self._ws = ws
        self.connected_at = time.monotonic()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._close()
        return self

    def _close(self):
        try:
            if self._ws:
                self._ws.close()
        except Exception:                                    # noqa: BLE001
            pass
        self._ws = None

    @property
    def ok(self):
        return bool(self._thread and self._thread.is_alive() and self._ws is not None)

    # -- the pump --

    def _pump(self):
        import websocket
        backoff = self.RECONNECT_BACKOFF_S
        while not self._stop.is_set():
            if self._ws is None:
                # RECONNECT RESUBSCRIBES, because a subscription lives on the connection. A
                # reconnect that forgot to resubscribe would look perfectly healthy -- socket
                # open, no error -- and deliver nothing, which is the worst shape a feed failure
                # can take.
                try:
                    self._connect()
                    self.reconnects += 1
                    backoff = self.RECONNECT_BACKOFF_S
                except Exception as e:                       # noqa: BLE001
                    self.errors += 1
                    self.last_error = f"reconnect: {type(e).__name__}: {str(e)[:100]}"
                    self._stop.wait(backoff)
                    backoff = min(self.MAX_BACKOFF_S, backoff * 2)
                    continue
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:                           # noqa: BLE001
                self.errors += 1
                self.last_error = f"recv: {type(e).__name__}: {str(e)[:100]}"
                self._close()
                continue
            recv_ns = time.time_ns()
            try:
                msgs = json.loads(raw)
            except ValueError:
                continue
            self.last_frame_t = time.monotonic()
            for m in (msgs if isinstance(msgs, list) else [msgs]):
                if m.get("ev") == "status":
                    if m.get("status") not in ("success", "connected", "auth_success"):
                        self.last_error = f"status: {str(m)[:120]}"
                    continue
                self.frames += 1
                try:
                    if self._on_message(m, recv_ns):
                        self.updates += 1
                        # ONLY WHEN SOMETHING CHANGED. 95.8% of futures frames repeat the mid
                        # (measured 2026-08-26: 6,868 messages carrying 11 distinct mids in 60s),
                        # so taping every frame would be 25x the volume for the same information
                        # and would put a disk writer behind a 220/s socket for nothing. The
                        # index is the opposite -- every push is a new value -- so it tapes in
                        # full. Between them these reconstruct spot at any instant, which is
                        # exactly what the 2026-08-26 post-mortem could not do: `FuturesFeed` ran
                        # all afternoon at 114-220 msg/s and nothing wrote it down, so the
                        # decomposition of a loss into `delta * dF` could not be measured below
                        # the 2s quote cycle.
                        if self.tee is not None:
                            self.tee.offer(recv_ns, m)
                except Exception as e:                       # noqa: BLE001
                    self.errors += 1
                    self.last_error = f"handle: {type(e).__name__}: {str(e)[:100]}"

    def _on_message(self, msg, recv_ns):
        raise NotImplementedError

    def health(self, now=None):
        now = now or time.monotonic()
        return {"name": self.name, "ok": self.ok, "frames": self.frames,
                "updates": self.updates, "reconnects": self.reconnects,
                "errors": self.errors, "last_error": self.last_error,
                "silent_s": (None if self.last_frame_t is None
                             else round(now - self.last_frame_t, 3))}


class FuturesFeed(_Stream):
    """Top-of-book mid per underlying, off `Q.<contract>`.

    `mid(underlying)` keeps `deltaadj.FuturesTicker.mid`'s contract exactly -- `(mid, age)` or
    `(None, age|None)`, age measured on the VENDOR's stamp -- so `DeltaAdjuster` is unchanged.

    The front month is resolved once at `start()`, by traded VOLUME rather than by calendar (see
    `massive.futures.front_month` on the roll), and the same resolution the polling ticker used.
    """

    MAX_QUOTE_AGE_S = getattr(config, "FUTURES_MAX_STALE_S", 2.0)

    def __init__(self, underlyings=("SPX", "NDX"), day=None, contracts=None, tee=None):
        self.underlyings = tuple(underlyings)
        self.contract = {}          # underlying -> e.g. "ESU6"
        self._by_sym = {}           # "ESU6" -> underlying
        self.resolve_error = {}
        day = day or config.today_et_date()
        for u in self.underlyings:
            product = fut.PRODUCT.get(u, u)
            tk = (contracts or {}).get(u)
            if tk is None:
                try:
                    tk = fut.front_month(product, day)
                except Exception as e:                       # noqa: BLE001
                    self.resolve_error[u] = f"{type(e).__name__}: {str(e)[:90]}"
                    tk = None
            if tk:
                self.contract[u] = tk
                self._by_sym[tk] = u
        self._mid = {}              # underlying -> (mid, ts_ns)
        self._ticks = {}            # underlying -> count of MID CHANGES
        # A short history of MID CHANGES per underlying, for `move_scale`. Only changes go in --
        # 95.8% of frames repeat the mid, and a history padded with repeats would understate the
        # volatility it exists to measure. Bounded by count, not by time, so a quiet market cannot
        # grow it and a busy one cannot either.
        self._hist = collections.defaultdict(lambda: collections.deque(maxlen=HIST_MAX))
        subs = [f"Q.{t}" for t in self.contract.values()]
        super().__init__(FUTURES_WS, subs, "futures", tee=tee)

    def _on_message(self, m, recv_ns):
        u = self._by_sym.get(m.get("sym"))
        if u is None:
            return False
        bp, ap = m.get("bp"), m.get("ap")
        if bp is None or ap is None:
            return False
        bp, ap = float(bp), float(ap)
        if ap < bp:
            return False                                     # crossed: not a mid
        mid = (bp + ap) / 2.0
        # WS `t` is MILLISECONDS. See the module docstring on the unit trap.
        ts_ns = int(m["t"]) * 1_000_000 if m.get("t") else recv_ns
        with self._lock:
            prev = self._mid.get(u)
            self._mid[u] = (mid, ts_ns)
            # 95.8% of frames repeat the mid (measured). Only a CHANGE is an update, and only a
            # change is worth waking a consumer for.
            if prev is not None and prev[0] == mid:
                return False
            self._ticks[u] = self._ticks.get(u, 0) + 1
            self._hist[u].append((ts_ns, mid))
        return True

    def mid(self, underlying, max_age_s=None):
        """(mid, age_s) or (None, age_s|None). Age is on the VENDOR's stamp, as before."""
        max_age_s = self.MAX_QUOTE_AGE_S if max_age_s is None else max_age_s
        with self._lock:
            got = self._mid.get(underlying)
        if got is None:
            return None, None
        mid, ts_ns = got
        age = (time.time_ns() - ts_ns) / NS
        # A negative age is clock skew, refused exactly as the polling ticker refused it.
        if age < 0 or age > max_age_s:
            return None, age
        return mid, age

    def move_scale(self, underlying, horizon_s, sigmas=1.0, min_ticks=MIN_SCALE_TICKS):
        """Expected adverse futures move over `horizon_s`, in index points. None if unknown.

        `sigmas * sigma * sqrt(horizon)`, with sigma estimated from the realized variance of the
        recent MID CHANGES. This is the size of the free option we write to anyone who sees the
        futures before we do: over the ~85ms between a tick moving and our cancel landing, a
        market with dollar delta `d` can move `d * move_scale` against a resting quote, and if that
        exceeds the edge we charged, the quote is mispriced by construction.
        USES THE VENDOR'S TIMESTAMPS, not receipt: the question is how fast the market moved, not
        how fast we heard about it. Returns None rather than a guess when the history is too thin,
        because a gate built on three ticks would refuse or admit essentially at random.
        """
        with self._lock:
            h = list(self._hist.get(underlying) or ())
        if len(h) < min_ticks:
            return None
        span_s = (h[-1][0] - h[0][0]) / NS
        if span_s <= 0:
            return None
        # Realized variance per second from the increments, then scaled by sqrt(time). Sum of
        # squared increments over the span is the standard estimator and needs no even spacing,
        # which matters here because mid changes are event-timed, not clocked.
        ss = sum((b[1] - a[1]) ** 2 for a, b in zip(h, h[1:]))
        var_per_s = ss / span_s
        if var_per_s <= 0:
            return 0.0
        return sigmas * math.sqrt(var_per_s * horizon_s)

    def detail(self):
        out = {u: {"contract": self.contract.get(u), "ticks": self._ticks.get(u, 0)}
               for u in self.underlyings}
        for u in self.underlyings:
            mid, age = self.mid(u)
            out[u]["mid"] = mid
            out[u]["age_s"] = None if age is None else round(age, 4)
        return {**self.health(), "by_underlying": out, "resolve_error": self.resolve_error}


class IndexFeed(_Stream):
    """Index level per underlying, off `V.I:SPX`, as `massive.index.IndexQuote` objects.

    `snapshot()` matches `massive.index.snapshot`'s return shape, so `pricing/spot.resolve`, the
    hourly pricer's fallback and the quote loop's `_index_quotes` all take it unchanged.

    ONE PER SECOND IS THE VENDOR'S GRID, not a defect and not something a faster reader can fix:
    the index is recomputed from constituent trades once a second. What WebSocket delivery buys is
    the 690ms we used to add on top of it by polling REST.
    """

    def __init__(self, underlyings=("SPX", "NDX"), tee=None):
        self.underlyings = tuple(underlyings)
        self._tickers = {u: ix.INDEX_TICKER[u] for u in underlyings if u in ix.INDEX_TICKER}
        self._by_tk = {v: k for k, v in self._tickers.items()}
        self._q = {}
        subs = [f"V.{t}" for t in self._tickers.values()]
        super().__init__(INDICES_WS, subs, "index", tee=tee)

    def _on_message(self, m, recv_ns):
        # `V` carries `T` (ticker) and `val`; the `A` aggregate carries `sym`. Accept either.
        tk = m.get("T") or m.get("sym")
        u = self._by_tk.get(tk)
        val = m.get("val")
        if u is None or val is None:
            return False
        ts_ns = int(m["t"]) * 1_000_000 if m.get("t") else recv_ns
        with self._lock:
            prev = self._q.get(u)
            self._q[u] = ix.IndexQuote(u, tk, float(val), ts_ns, market_status="open")
        return prev is None or prev.value != float(val)

    def snapshot(self, underlyings=None):
        """{underlying: IndexQuote}. An underlying we have not heard from is ABSENT, not None."""
        want = tuple(underlyings) if underlyings else self.underlyings
        with self._lock:
            return {u: self._q[u] for u in want if u in self._q}

    def detail(self):
        snap = self.snapshot()
        return {**self.health(),
                "by_underlying": {u: {"value": q.value, "age_s": round(q.age_s() or 0.0, 4)}
                                  for u, q in snap.items()}}
