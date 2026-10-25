"""Massive market-data client: REST reference/quotes + options Quotes WebSocket collector."""
import collections
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import websocket

import config


class MassiveError(RuntimeError):
    pass


def rest(path, **params):
    params = {k: v for k, v in params.items() if v is not None}
    params["apiKey"] = config.MASSIVE_API_KEY
    url = f"{config.MASSIVE_REST_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise MassiveError(f"HTTP {e.code}: {e.read()[:300].decode('utf8', 'ignore')}") from None


def _rest_url(url):
    sep = "&" if "?" in url else "?"
    with urllib.request.urlopen(f"{url}{sep}apiKey={config.MASSIVE_API_KEY}", timeout=30) as resp:
        return json.loads(resp.read())


def stock_nbbo(symbol=config.STOCK_SYMBOL):
    """Real-time NBBO for the underlying stock. Fields: p/s = bid px/size, P/S = ask px/size."""
    return rest(f"/v2/last/nbbo/{symbol}")


def stock_mid(payload):
    r = payload["results"]
    return (r["p"] + r["P"]) / 2.0


def option_contracts(underlying=config.OPTION_UNDERLYING, expiration_date=None):
    """All contracts for an expiry. Returns the raw `results` list (SPX -> SPXW tickers)."""
    out = []
    page = rest("/v3/reference/options/contracts", underlying_ticker=underlying,
                expiration_date=expiration_date, limit=1000)
    while True:
        out.extend(page.get("results") or [])
        nxt = page.get("next_url")
        if not nxt:
            return out
        page = _rest_url(nxt)


def occ_ticker(root, expiry_yymmdd, right, strike):
    """O:SPXW260819C07700000 -- strike is 8 digits of strike*1000."""
    return f"O:{root}{expiry_yymmdd}{right}{round(strike * 1000):08d}"


# Keep the tail only: enough to diagnose a subscribe failure, bounded against a
# session's worth of per-ticker acks.
STATUS_LOG_MAX = 2000


class QuoteCollector:
    """Subscribes to Q.<ticker> for a list of option tickers and keeps the latest quote per symbol.

    Every raw frame is teed to `raw_path` as JSONL so the on-the-wire format stays inspectable.
    """

    # Massive caps a connection at 1000 subscribed tickers and permits ONE options connection
    # per account, so the subscription is a budget to be managed, not a set-and-forget.
    MAX_TICKERS = 1000
    # Minimum seconds between re-centring the option strike subscription on spot. Illustrative
    # value; the live value is not published.
    RECENTER_MIN_INTERVAL = 3.0

    def __init__(self, tickers, raw_path=None, url=config.MASSIVE_WS_OPTIONS):
        self.tickers = list(tickers)
        self.url = url
        self.raw_path = raw_path
        self.latest = {}          # sym -> latest Q dict
        self.frame_count = 0
        self.msg_count = 0
        self._lock = threading.Lock()
        # Separate from _lock: _pump holds _lock while draining frames, and a re-centre must be
        # able to send without waiting on it. Sends and receives are on the same socket from
        # different threads, so sends are serialised among themselves.
        self._send_lock = threading.Lock()
        self._subscribed = set(self.tickers)
        self._last_recenter = 0.0
        self.recenter_count = 0
        self._stop = threading.Event()
        self._raw_fh = None
        self._ws = None
        self._thread = None
        # Bounded. Massive acks subscribe/unsubscribe PER TICKER, so a quoter that re-centres
        # every 5s accumulates them fast: measured ~220 acks from a single 210-ticker subscribe,
        # projecting to 194k dicts (~49 MB) per session swapping 20 tickers per re-centre, or
        # ~243 MB at 100. Nothing reclaims them -- the list stays reachable for the collector's
        # lifetime, so Python never collects it. Was harmless in the old two-cycle run_03, but
        # cycles; a genuine leak in the long-running quoter this class is headed for. The tail
        # is what has diagnostic value anyway.
        self.status_log = collections.deque(maxlen=STATUS_LOG_MAX)
        self.status_dropped = 0
        # Feed health. `_pump` used to `break` out of its loop on any socket exception and the
        # thread simply ended -- no flag, no log line, no reconnect. `self.latest` then froze at
        # whatever it last held, and since a present quote is treated as current the quoter went
        # on pricing and posting off a dead feed indefinitely. (Spot did NOT catch it: the index
        # arrives on a separate REST call and stayed fresh.) These four are what makes the death
        # observable, and `health()` is what a caller gates on.
        self.pump_error = None       # str, set when the pump exits abnormally
        self.pump_exited_at = None   # time.monotonic() of that exit
        self.last_msg_t = None       # time.monotonic() of the last QUOTE frame, not any frame
        self.reconnects = 0

    # -- lifecycle --
    def start(self, attempts=6, backoff=5.0):
        """Connect, auth, subscribe.

        The account permits only one concurrent options connection, and the server takes a
        few seconds to reap a closed one -- so `max_connections` is retried rather than fatal.
        """
        if self.raw_path:
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            self._raw_fh = open(self.raw_path, "a")

        for attempt in range(attempts):
            try:
                self._connect_and_auth()
                break
            except MassiveError as e:
                self._close_socket()
                if "max_connections" not in str(e) or attempt == attempts - 1:
                    self._close_raw()
                    raise
                time.sleep(backoff)
            except Exception:
                self._close_socket()
                self._close_raw()
                raise

        self._ws.settimeout(1.0)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _connect_and_auth(self):
        self._ws = websocket.create_connection(self.url, timeout=20)
        self.status_log.append(json.loads(self._ws.recv()))

        self._ws.send(json.dumps({"action": "auth", "params": config.MASSIVE_API_KEY}))
        auth = json.loads(self._ws.recv())
        self.status_log.append(auth)
        if not any(m.get("status") == "auth_success" for m in auth):
            raise MassiveError(f"auth failed: {auth}")

        # One frame, comma-joined -- exactly {"action":"subscribe","params":"Q.O:...,Q.O:..."}
        params = ",".join(f"Q.{t}" for t in self.tickers)
        self._ws.send(json.dumps({"action": "subscribe", "params": params}))

    def _close_socket(self):
        try:
            self._ws and self._ws.close()
        except Exception:
            pass
        self._ws = None

    def _close_raw(self):
        try:
            self._raw_fh and self._raw_fh.close()
        except Exception:
            pass
        self._raw_fh = None

    def _pump(self):
        while not self._stop.is_set():
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:                           # noqa: BLE001
                # Record it. This used to be a bare `break`, which is the difference between a
                # feed that is down and a feed that is quiet -- and the quoter cannot tell those
                # apart from `latest` alone, because a stale quote and a current one are the
                # same dict.
                self.pump_error = f"{type(e).__name__}: {str(e)[:200]}"
                self.pump_exited_at = time.monotonic()
                break
            if not raw:
                continue
            if self._raw_fh:
                self._raw_fh.write(raw if raw.endswith("\n") else raw + "\n")
            try:
                msgs = json.loads(raw)
            except json.JSONDecodeError:
                continue
            with self._lock:
                self.frame_count += 1
                for m in msgs:
                    self.msg_count += 1
                    if m.get("ev") != "Q":
                        if len(self.status_log) == self.status_log.maxlen:
                            self.status_dropped += 1
                        self.status_log.append(m)
                        continue
                    sym = m.get("sym")
                    prev = self.latest.get(sym)
                    # Sequence numbers are per-symbol and increasing; ignore out-of-order frames.
                    if prev is None or m.get("q", 0) >= prev.get("q", 0):
                        self.latest[sym] = m
                    self.last_msg_t = time.monotonic()

    # -- health --

    @property
    def alive(self):
        """Is the pump thread still reading the socket?"""
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def health(self, now=None):
        """What a caller must check before pricing off `snapshot()`.

        `quote_age_s` is measured from the last QUOTE frame, not from any frame: heartbeats and
        subscribe acks keep arriving on a socket whose data has gone quiet, so ageing on all
        traffic would report a healthy feed that is delivering nothing. None means no quote has
        ever arrived, which is the warm-up state and is not the same as stale.
        """
        t = time.monotonic() if now is None else now
        return {"alive": self.alive,
                "pump_error": self.pump_error,
                "quote_age_s": (None if self.last_msg_t is None else t - self.last_msg_t),
                "n_symbols": len(self.latest),
                "reconnects": self.reconnects,
                "frames": self.frame_count}

    def reconnect(self, attempts=6, backoff=5.0):
        """Rebuild the socket and resubscribe to whatever is currently subscribed.

        **Clears `latest`.** A reconnect means an unknown gap, and every cached quote is now of
        unknown age. Serving them would be the exact failure `unsubscribe` already refuses to
        allow: "a silently stale quote is worse than a missing one -- the pricer treats a present
        quote as current and would build a theo on it." A cleared cache prices nothing until the
        symbols tick again, which is the honest state.

        The account permits ONE concurrent options connection and the server takes a few seconds
        to reap a closed one, so the `max_connections` retry in `start()` applies here too.
        """
        self._close_socket()
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=5)
        self._stop = threading.Event()
        with self._lock:
            self.latest.clear()

        self.tickers = sorted(self._subscribed) or self.tickers
        for attempt in range(attempts):
            try:
                self._connect_and_auth()
                break
            except MassiveError as e:
                self._close_socket()
                if "max_connections" not in str(e) or attempt == attempts - 1:
                    self.pump_error = f"reconnect failed: {str(e)[:160]}"
                    return False
                time.sleep(backoff)
            except Exception as e:                           # noqa: BLE001
                self._close_socket()
                self.pump_error = f"reconnect failed: {type(e).__name__}: {str(e)[:160]}"
                return False

        self._ws.settimeout(1.0)
        self.pump_error = None
        self.pump_exited_at = None
        self.last_msg_t = None
        self.reconnects += 1
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return True

    # -- dynamic subscription --

    def _send(self, action, tickers):
        if not tickers:
            return 0
        params = ",".join(f"Q.{t}" for t in tickers)
        with self._send_lock:
            self._ws.send(json.dumps({"action": action, "params": params}))
        return len(tickers)

    def subscribe(self, tickers):
        new = [t for t in tickers if t not in self._subscribed]
        if not new:
            return 0
        if len(self._subscribed) + len(new) > self.MAX_TICKERS:
            raise MassiveError(
                f"subscribe would reach {len(self._subscribed) + len(new)} tickers, "
                f"over the {self.MAX_TICKERS} cap")
        self._send("subscribe", new)
        self._subscribed.update(new)
        return len(new)

    def unsubscribe(self, tickers):
        gone = [t for t in tickers if t in self._subscribed]
        if not gone:
            return 0
        self._send("unsubscribe", gone)
        self._subscribed.difference_update(gone)
        # Evict the cached quotes too. A symbol we no longer receive updates for would otherwise
        # keep serving its last value forever, and a silently stale quote is worse than a missing
        # one: the pricer treats a present quote as current and would build a theo on it.
        with self._lock:
            for t in gone:
                self.latest.pop(t, None)
        return len(gone)

    def recenter(self, want, force=False):
        """Move the subscription to `want`. Returns (added, removed), or None if rate-limited.

        Rate-limited rather than queued: if the caller asks again inside the interval the right
        answer is "not yet", not "do it twice". Removals go first so a re-centre can never
        transiently exceed the cap.

        **Newly added symbols arrive blind.** There is no snapshot-on-subscribe: a symbol has no
        quote until it next ticks. Measured on the live socket -- 15 s after re-adding 8
        near-money 0DTE strikes, only 2 had quoted. So a re-centre is not free, and the interval
        exists as much to bound that blind window as to bound socket traffic. The pinned
        bracketing strikes are never dropped precisely so this never costs us a priced market.
        """
        now = time.monotonic()
        if not force and (now - self._last_recenter) < self.RECENTER_MIN_INTERVAL:
            return None
        want = set(want)
        if len(want) > self.MAX_TICKERS:
            raise MassiveError(f"{len(want)} tickers requested, over the {self.MAX_TICKERS} cap")
        removed = self.unsubscribe(self._subscribed - want)
        added = self.subscribe(want - self._subscribed)
        self._last_recenter = now
        self.tickers = sorted(self._subscribed)
        if added or removed:
            self.recenter_count += 1
        return added, removed

    @property
    def subscribed(self):
        return set(self._subscribed)

    def snapshot(self):
        with self._lock:
            return dict(self.latest)

    def wait_for_coverage(self, min_symbols, timeout=20.0, poll=0.25):
        """Block until at least `min_symbols` distinct symbols have quoted, or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.latest) >= min_symbols:
                    return True
            time.sleep(poll)
        return False

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._close_socket()
        self._close_raw()
        return self
