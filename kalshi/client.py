"""Kalshi Trade API v2 client: RSA-PSS request signing + market data / order helpers."""
import base64
import http.client
import json
import threading
import time
import urllib.parse

import config
from kalshi import auth


def _private_key():
    return auth.load_private_key(config.KALSHI_PRIVATE_KEY_PATH)


def _sign(timestamp_ms, method, path):
    """RSA-PSS-SHA256 over `{ts}{METHOD}{path}`. `path` includes /trade-api/v2, excludes query."""
    return auth.sign(_private_key(), timestamp_ms, method, path)


def _auth_headers(method, path):
    return auth.auth_headers(config.KALSHI_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH, method, path)


# Kalshi throttles reads per API key; sweeping every series trips 429 immediately.
# Token bucket + retry keeps a full-exchange sweep inside the limit.
#
# READS AND WRITES ARE METERED SEPARATELY BY KALSHI, and until 2026-08-21 this client ran both
# through one bucket -- so an order queued behind whatever market data happened to be in flight,
# for no exchange-side reason. Two buckets now. The write budget is the one that matters for a
# tiered quoter: three rungs x two sides x ~30 markets is up to 180 orders, which through one
# 8 req/s bucket is ~22 seconds of pure sleep, longer than the edge cycle it has to fit inside.
# Every order in a batch is billed separately: 10 tokens per create, 2 per cancel. The largest
# batch the exchange ACCEPTS is a separate limit from what the bucket can pay for, and it is
# smaller -- 30 on this account, measured 2026-08-26, against the 90 the capacity implies. The
# OpenAPI spec carries no `maxItems` on either batch array, so it is measured rather than read.
# See `_max_batch`.
ORDER_TOKEN_COST = 10
CANCEL_TOKEN_COST = 2

# The account's default, for any endpoint not overridden. Confirmed against
# GET /account/endpoint_costs on 2026-08-24: `default_cost: 10`, and `/markets/orderbooks` is
# NOT in the override list -- so a 100-ticker book fetch costs 10 tokens, the same as a
# 1-ticker one. Only the two BATCHED order endpoints are billed per item (kalshi_api.yaml:1066,
# :1103); every other endpoint note in the spec reads "N tokens per request".
DEFAULT_TOKEN_COST = 10

# How many times to sample /account/limits before believing it. Shared by both budget setters
# because the hazard is one hazard: the ALB fronts backends that disagree about our tier, so a
# single reading of EITHER bucket is a coin flip. See `set_read_budget` for the measurement.
TIER_SAMPLES = 3

_RATE_PER_SEC = 8.0             # reads, in REQUESTS per second
_MAX_RETRIES = 6
_rate_lock = threading.Lock()
_next_slot = [0.0]

# Writes are metered in TOKENS, not requests, and a batch is billed per item -- 10 tokens per
# order created, 2 per order cancelled. So a batch of 90 creates is one HTTP request and 900
# tokens, and throttling it as "one request" would blow straight through the bucket. This is a
# token bucket at the account's own refill rate, charged by cost.
#
# Measured on this account 2026-08-21 and re-measured 2026-08-24: usage_tier 'advanced', write
# bucket_capacity 900, refill_rate 300/s. That is 30 orders/s sustained -- and 30 is also the
# largest batch the endpoint accepts, measured 2026-08-26. Do NOT read 90 out of the capacity.
_WRITE_TOKENS_PER_SEC = 300.0
_WRITE_CAPACITY = 900.0

_WRITE_METHODS = ("POST", "DELETE", "PUT", "PATCH")


class TokenBucket:
    """Kalshi's own accounting, mirrored locally: `capacity` tokens, refilling at `refill_rate`/s.

    THE POINT OF THIS CLASS IS THE BURST CREDIT, and that is precisely what the pacer it replaces
    could not do. `_throttle`'s write half was a next-slot pacer -- it spaced every order
    10/300 = 33 ms apart and took 3.0 s to push a 90-order batch that Kalshi would have accepted
    instantly, because a pacer has no notion of capacity ALREADY ACCUMULATED. That is why the
    pacing was removed on 2026-08-23 rather than fixed. A bucket is the fix: idle for three
    seconds and you may spend 900 tokens in one go, exactly as the exchange allows.

    IT NEVER BLOCKS AND IT NEVER REFUSES. `debit` always succeeds and is allowed to drive the
    level NEGATIVE. Two reasons, and both are load-bearing:

      1. The kill path (`strategy/cancel_all`) must not be gated by our own bookkeeping. An
         order we cannot cancel because we ran out of imaginary tokens is a real position we
         cannot close. It sends; the bucket records the debt; the debt refills off.
      2. A refusal inside the transport is a silent behaviour change at a layer that cannot know
         what it is dropping. Deciding WHAT to give up is the scheduler's job -- it is the only
         thing that knows an `edge_negative` cancel outranks a penny improvement.

    So callers ASK (`available`, `can_afford`) and the transport CHARGES (`debit`). Charging in
    exactly one place is what keeps the mirror honest; a scheduler that both reserved and sent
    would double-count every order.
    """

    __slots__ = ("capacity", "refill_rate", "_level", "_t", "_lock", "spent", "denied")

    def __init__(self, capacity, refill_rate):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._level = float(capacity)          # start full: we have not spent anything yet
        self._t = time.monotonic()
        self._lock = threading.Lock()
        self.spent = 0.0                       # cumulative, for the cycle record
        self.denied = 0                        # times can_afford said no

    def _refill(self, now):
        """Advance the level to `now`. Caller holds the lock."""
        if now > self._t:
            self._level = min(self.capacity, self._level + (now - self._t) * self.refill_rate)
            self._t = now
        return self._level

    def available(self):
        """Tokens spendable right now. Negative when we are in debt to the exchange."""
        with self._lock:
            return self._refill(time.monotonic())

    def can_afford(self, tokens):
        return self.available() >= float(tokens)

    def debit(self, tokens):
        """Charge `tokens`. Always succeeds; may go negative. Returns the resulting level."""
        n = float(tokens)
        with self._lock:
            self._refill(time.monotonic())
            self._level -= n
            self.spent += n
            return self._level

    def resize(self, capacity=None, refill_rate=None):
        """Re-point at a freshly read tier. Keeps the current level, clamped to the new capacity.

        Keeping the level matters: `set_write_budget` runs at startup and could run again after a
        tier change, and resetting to full there would hand back tokens we have already spent.
        """
        with self._lock:
            self._refill(time.monotonic())
            if capacity is not None:
                self.capacity = float(capacity)
                self._level = min(self._level, self.capacity)
            if refill_rate is not None:
                self.refill_rate = float(refill_rate)

    def penalise(self, tokens):
        """Charge without a request behind it, after a 429.

        A 429 means Kalshi's bucket was empty when ours said otherwise, so ours is wrong in the
        optimistic direction -- the one that produces more 429s. Charging for the rejection pulls
        the mirror back towards the truth instead of leaving it confidently ahead.
        """
        self.denied += 1
        return self.debit(tokens)

    def detail(self):
        return {"available": round(self.available(), 1), "capacity": self.capacity,
                "refill_rate": self.refill_rate, "spent": round(self.spent, 1),
                "denied": self.denied}


# The account's write budget, mirrored. Sized from /account/limits by `set_write_budget`; the
# defaults are this account's measured tier so a process that never calls it is still roughly
# right rather than unbounded.
write_bucket = TokenBucket(_WRITE_CAPACITY, _WRITE_TOKENS_PER_SEC)


def _throttle(method="GET", tokens=None):
    """Wait for read capacity. Writes are still NOT DELAYED here -- see below.

    THE WRITE PACING WAS REMOVED 2026-08-23, deliberately, and it stays removed. It cost
    10 tokens / 300 tok/s = 33 ms of spacing between orders and 3.0 s for a 90-order batch, and
    it was never a token bucket -- it was a next-slot pacer with no burst credit, so it could not
    spend the account's 900-token capacity even though Kalshi allows exactly that instantly. An
    order should reach the exchange as fast as the socket allows; the exchange is the thing
    entitled to say no.

    What 2026-08-24 adds is ACCOUNTING, not delay. `write_bucket` (a real TokenBucket, with the
    burst credit the old pacer lacked) is debited by `_request` on every write, and the SCHEDULER
    reads it to decide what to send. Nothing sleeps on it and nothing is refused by it, so the
    property above is intact: an admitted order still reaches the socket immediately, and the
    kill path is never gated by our own bookkeeping.

    What we still rely on: Kalshi's own 429, surfaced immediately rather than retried.
    `_request` refuses to retry ANY write on ANY status, so a rate-limited order raises
    KalshiError(429) at the caller instead of being silently re-sent. That pairing is the whole
    safety argument -- removing the pacing without it would have turned every over-budget burst
    into a double-send path.

    `tokens` is the write cost, priced by the caller (`orders.py` prices a batch per item). It is
    not used for pacing; `_request` charges it to `write_bucket`.
    """
    if method.upper() in _WRITE_METHODS:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = max(0.0, _next_slot[0] - now)
        _next_slot[0] = max(now, _next_slot[0]) + 1.0 / _RATE_PER_SEC
    if wait:
        time.sleep(wait)


def set_write_budget(refill_rate=None, samples=TIER_SAMPLES):
    """Point `write_bucket` at the account's real write budget. Returns the budget dict.

    SAMPLED `samples` TIMES AND MINIMISED, for the same reason `set_read_budget` does it: the ALB
    fronts backends that disagree about our tier. Measured 2026-08-21, grant present throughout,
    back-to-back calls returned `basic` (write 100 tok/s) and `advanced` (300) alternating --
    5 of 8 advanced. A single reading is a coin flip, and a bucket sized for advanced on a
    backend that thinks we are basic 429s by construction. Re-measured 2026-08-24: 5/5 advanced,
    900/300 on both buckets. One clean reading does not retire a flap that has been observed, so
    the minimum stays.

    This used to sample once and set `_WRITE_TOKENS_PER_SEC`, which nothing read. Now it sizes a
    real bucket. `max_batch = bucket_capacity // ORDER_TOKEN_COST` is unchanged and still the
    other load-bearing output: `strategy/orders.py` chunks creates and cancels at that size, and
    exceeding it is a rejected batch rather than a slow one.

    Never raises -- a quoter that cannot read its own limits keeps the conservative default.
    """
    global _WRITE_TOKENS_PER_SEC, _WRITE_CAPACITY
    obs, b = [], {}
    if refill_rate is None:
        for _ in range(max(1, samples)):
            b = write_budget()
            obs.append({"usage_tier": b.get("usage_tier"), "refill_rate": b.get("refill_rate"),
                        "bucket_capacity": b.get("bucket_capacity"), "error": b.get("error")})
        rates = [o["refill_rate"] for o in obs if o.get("refill_rate")]
        caps = [o["bucket_capacity"] for o in obs if o.get("bucket_capacity")]
        r = min(rates) if rates else None
        cap = min(caps) if caps else None
        # The batch size must follow the MINIMISED figures, not the last sample's -- they are what
        # a rejected batch is measured against.
        if cap:
            b = {**b, "bucket_capacity": cap, "refill_rate": r,
                 "max_batch": _max_batch(cap, r)}
    else:
        b = write_budget()
        r, cap = float(refill_rate), b.get("bucket_capacity")

    if r:
        _WRITE_TOKENS_PER_SEC = float(r)
    if cap:
        _WRITE_CAPACITY = float(cap)
    write_bucket.resize(capacity=cap or None, refill_rate=r or None)
    return {**b, "observations": obs, "tiers_seen": sorted({o.get("usage_tier") for o in obs
                                                            if o.get("usage_tier")}),
            "bucket": write_bucket.detail()}


def endpoint_costs():
    """GET /account/endpoint_costs -> {'default_cost': int, 'endpoint_costs': [...]}.

    The authority on what a request actually costs, and the answer to a question the OpenAPI
    prose only half states: an endpoint absent from `endpoint_costs` is billed `default_cost`
    PER REQUEST. That is how we know `/markets/orderbooks` costs 10 tokens whether it carries
    one ticker or its `maxItems` limit of 100.

    Never raises -- this is diagnostic, and nothing should refuse to quote over it.
    """
    try:
        return _request("GET", "/account/endpoint_costs", signed=True)
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}


# The read bucket's conservative default. `_RATE_PER_SEC = 8.0` is a floor chosen when the tier
# was unknown; advanced allows 300 tok/s at 10 tokens per read, i.e. 30 req/s. The trading path
# used to leave it at 8, paying 125 ms of slot spacing on every markets()/orderbooks() call for
# headroom it already owned. `set_read_budget` raises it, but never past this.
#
# WHY A CAP AT ALL, and why it is not 30: `_RATE_PER_SEC` is a module global PER PROCESS, while
# Kalshi meters PER API KEY. Two processes each sized for the full account limit is 2x the rate
# against one bucket, which is exactly how the 365-day backfill produced HTTP 429s through all
# six retries while another job ran. Until the budget is shared across processes, each one takes
# a fraction.
READ_RATE_CAP = 20.0
READ_TIER_SAMPLES = TIER_SAMPLES


def set_read_budget(refill_rate=None, cap=READ_RATE_CAP, samples=READ_TIER_SAMPLES, share=1.0):
    """Point the read bucket at the account's real refill rate. Returns the observations.

    `share` is this process's slice of the account, in [0, 1] -- see `config.READ_SHARE_*`. The
    limit is per API KEY and `_RATE_PER_SEC` is per PROCESS, so two contenders each sized for the
    whole account make twice the rate against one bucket. `cap` remains as a backstop for callers
    that pass no share at all.

    Samples `/account/limits` `samples` times and takes the MINIMUM, because the ALB fronts
    backends that disagree: back-to-back calls have returned `basic` (read 200 tok/s) and
    `advanced` (300) alternating, with the grant present throughout. A client that reads its
    limit once gets a coin flip, and a burst sized for advanced 429s on a backend that thinks we
    are basic. The minimum is the only reading that is safe under both.

    Never raises. A quoter that cannot read its own limits should keep the conservative default,
    not refuse to quote.
    """
    global _RATE_PER_SEC
    obs = []
    if refill_rate is None:
        for _ in range(max(1, samples)):
            try:
                lim = _request("GET", "/account/limits", signed=True)
                obs.append({"usage_tier": lim.get("usage_tier"),
                            "read_refill_rate": float((lim.get("read") or {}).get("refill_rate") or 0)})
            except Exception as e:                               # noqa: BLE001
                obs.append({"error": f"{type(e).__name__}: {str(e)[:80]}"})
        rates = [o["read_refill_rate"] for o in obs if o.get("read_refill_rate")]
        r = min(rates) if rates else None
    else:
        r = float(refill_rate) * ORDER_TOKEN_COST

    before = _RATE_PER_SEC
    if r:
        # 10 tokens per read, then 10% of headroom left on the table for retries, then this
        # process's share of what remains.
        want = min(cap, r / DEFAULT_TOKEN_COST * 0.9 * float(share))
        # `max(before, ...)` is the FLOOR that keeps a process from collapsing below the
        # conservative 8 req/s default on a bad reading -- but it must not apply to an explicit
        # share, whose whole job may be to ask for LESS than the default. A caller that says
        # "take a fifth of the account" and silently gets 8 req/s is not sharing anything.
        _RATE_PER_SEC = want if float(share) < 1.0 else max(before, want)
    return {"observations": obs, "min_read_refill_rate": r,
            "req_per_s_before": before, "req_per_s_after": _RATE_PER_SEC,
            "cap": cap, "share": float(share),
            "tiers_seen": sorted({o.get("usage_tier") for o in obs if o.get("usage_tier")})}


_BATCH_FALLBACK = 20


def _max_batch(capacity, refill):
    """Orders per batched create/cancel request.

    MEASURED 2026-08-26, and it is NOT `bucket_capacity // ORDER_TOKEN_COST`. That derivation
    said 90 on this account; the exchange refuses anything over **30** with a whole-request
    400 `too_many_orders_in_batch`, placing nothing. Probed with deliberately invalid tickers so
    the size check answers before the order check and nothing can rest: 30 accepted (30 per-order
    results back), 31, 32, 33, 34, 35, 45, 60 and 90 all refused for size.

    30 is exactly `refill_rate // ORDER_TOKEN_COST` = 300/10, which matches the spec's "the
    maximum batch size scales with your tier's write budget". Both figures are taken here and the
    smaller wins, so this can only ever be conservative if the rule is something else again --
    and being conservative is free, because the 10-tokens-per-ORDER billing means splitting a
    batch costs nothing but one extra HTTP round trip.

    This had never bitten because it had never been reached: the 2026-08-25 live session ran a
    30-market board and its largest batch was 14 orders. The four-board universe plans ~60 a
    cycle, so the first live session on it placed NOTHING for 43 cycles -- and said so only as
    `placed: 0, errors: 1`, because the response body was being discarded (see `orders.apply`).
    """
    figures = [f for f in (capacity, refill) if f]
    if not figures:
        return _BATCH_FALLBACK
    return max(1, int(min(figures) // ORDER_TOKEN_COST))


def write_budget():
    """{'max_batch': int, ...} from GET /account/limits, with a conservative fallback.

    Never raises: a quoter that cannot read its own rate limits should fall back to small batches,
    not refuse to quote.
    """
    try:
        lim = _request("GET", "/account/limits", signed=True)
        cap = float((lim.get("write") or {}).get("bucket_capacity") or 0)
        refill = float((lim.get("write") or {}).get("refill_rate") or 0)
        return {"usage_tier": lim.get("usage_tier"),
                "bucket_capacity": cap, "refill_rate": refill,
                "write_req_per_s": refill / ORDER_TOKEN_COST if refill else None,
                "max_batch": _max_batch(cap, refill)}
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:80]}", "max_batch": _BATCH_FALLBACK}


# --- transport ------------------------------------------------------------
# One keep-alive connection PER THREAD. `urllib.request.urlopen` sends
# `Connection: close`, so every call paid a fresh TCP + TLS handshake: measured
# against external-api.kalshi.com, 27.6 ms per request on a new connection vs
# 15.8 ms on a reused one, with the handshake alone 8.1 ms (29% of the request).
# Per-thread and not shared: an http.client connection carries one in-flight
# request at a time, and the backfill runs one ET day per worker thread.
_local = threading.local()


def _connection(scheme, netloc, timeout):
    """This thread's connection to `netloc`, plus whether its socket was already open.

    The flag is what makes a transparent retry safe to reason about: a request on
    an already-open socket can fail because the server closed an idle connection,
    which is a transport artefact and not an answer about the request.
    """
    key = (scheme, netloc)
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "key", None) == key:
        try:
            # Honour a per-call timeout without churning the connection.
            conn.timeout = timeout
            if conn.sock is not None:
                conn.sock.settimeout(timeout)
            return conn, conn.sock is not None
        except OSError:
            # Cached socket is already dead (EBADF and friends). Rebuilding here
            # rather than letting it raise keeps the failure out of the caller's
            # lap -- it is our own pooling artefact, not an answer from Kalshi.
            pass
    _drop_connection()
    cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    conn = cls(netloc, timeout=timeout)
    _local.conn, _local.key = conn, key
    return conn, False


def _drop_connection():
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except OSError:
            pass
    _local.conn = None
    _local.key = None


class KalshiError(RuntimeError):
    def __init__(self, status, body):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:400]}")


def _request(method, path, params=None, body=None, signed=False, timeout=20, tokens=None):
    """`path` is relative to /trade-api/v2, e.g. '/markets'."""
    base = urllib.parse.urlparse(config.KALSHI_API_BASE)
    full_path = base.path + path
    target = full_path
    if params:
        pairs = params if isinstance(params, list) else list(params.items())
        target += "?" + urllib.parse.urlencode([(k, v) for k, v in pairs if v is not None])

    data = json.dumps(body).encode() if body is not None else None
    method = method.upper()

    is_write = method in _WRITE_METHODS
    # Priced by the caller where the cost is per-item (the two batched order endpoints); the
    # account's default otherwise. Measured 2026-08-24 via GET /account/endpoint_costs:
    # default_cost 10, and the only write overrides that matter here are the two DELETEs at 2.
    cost = float(tokens) if tokens is not None else float(DEFAULT_TOKEN_COST)

    for attempt in range(_MAX_RETRIES):
        _throttle(method, tokens)
        # Charged BEFORE the send, not after: a request that reaches Kalshi has spent its tokens
        # whether or not we get to read the response. Writes never retry, so this runs once.
        if is_write:
            write_bucket.debit(cost)
        # Signature is timestamped, so headers must be rebuilt on every retry.
        headers = _auth_headers(method, full_path) if signed else {"Content-Type": "application/json"}
        conn, reused = _connection(base.scheme, base.netloc, timeout)
        try:
            conn.request(method, target, body=data, headers=headers)
            resp = conn.getresponse()
            # Drain before the next request: a partly-read response leaves the
            # connection out of sync and the following read returns this body.
            payload = resp.read()
            if resp.will_close:
                _drop_connection()
        except (http.client.HTTPException, OSError):
            # Reaching here on a socket we did not just open means the server
            # dropped an idle keep-alive connection, so the request was never
            # answered. Safe to repeat for a GET; a POST is ambiguous -- it may
            # have been received -- and repeating one could double-send an order.
            _drop_connection()
            if reused and method == "GET" and attempt < _MAX_RETRIES - 1:
                continue
            raise

        if 200 <= resp.status < 300:
            return json.loads(payload)

        detail = payload.decode("utf8", "ignore")
        # NEVER retry a write, on any status. The comment here used to say exactly that while the
        # code did the opposite: `resp.status == 429 or (resp.status >= 500 and method == "GET")`
        # binds the GET guard to the 500 clause only, so a 429 on an order create WAS retried, up
        # to six times. That is the double-send this line exists to prevent, and it mattered the
        # moment the write pacing came out -- an unthrottled 180-order refresh is 1800 tokens
        # against a 900-token bucket, so the back half 429s by construction.
        #
        # A 429 on a write is therefore surfaced immediately. The caller sees KalshiError(429) and
        # can decide; silently re-sending an order is never the right default.
        retryable = method not in _WRITE_METHODS and (resp.status == 429 or resp.status >= 500)
        if is_write and resp.status == 429:
            # Our mirror said we could afford this and Kalshi disagreed, so the mirror is wrong
            # in the optimistic direction -- the direction that produces more 429s. Charge again
            # to pull it back. This is the only feedback the exchange gives us about the drift.
            write_bucket.penalise(cost)
        if not retryable or attempt == _MAX_RETRIES - 1:
            raise KalshiError(resp.status, detail)
        time.sleep(0.4 * (2 ** attempt))


# --- market data (unauthenticated) ---

def get(path, **params):
    return _request("GET", path, params=params)


def paged(path, key, signed=False, **params):
    """Follow the `cursor` pagination until exhausted, returning one list.

    Convenient, but the whole result is held in memory. Some Kalshi series are
    enormous -- KXBTC alone lists ~297k markets from its 15-minute contracts --
    so bulk callers should use `paged_iter` instead.
    """
    return list(paged_iter(path, key, signed=signed, **params))


def paged_iter(path, key, signed=False, **params):
    """Yield items page by page. Peak memory is one page, not one series.

    `signed` is an explicit keyword, not part of **params: folded into params it would be sent
    as a query STRING and the request would go out unsigned, which the portfolio endpoints
    reject with 401.
    """
    cursor = None
    while True:
        page = _request("GET", path, params={**params, "cursor": cursor}, signed=signed)
        yield from (page.get(key) or [])
        cursor = page.get("cursor")
        if not cursor:
            return


def series_by_category(category):
    return _request("GET", "/series", params={"category": category}).get("series") or []


def markets(**params):
    return paged("/markets", "markets", limit=1000, **params)


def orderbook(ticker, depth=None):
    return _request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})


MAX_ORDERBOOK_TICKERS = 100


def orderbooks(tickers, depth=None):
    """All books, chunked -> {ticker: orderbook}.

    `tickers` must go over the wire as repeated params, not a comma-joined string: the
    comma form is validated as a single oversized ticker and rejected. ~40ms for 30 markets
    versus ~3.7s looping the single-market endpoint, which matters because this is the
    pre-trade snapshot.

    CHUNKED AT 100 SINCE 2026-08-24, because the spec caps it there and nothing respected the
    cap. `kalshi_api.yaml`'s `/markets/orderbooks` block carries `maxItems: 100` on `tickers` --
    the note at :56 above about "no `maxItems` on either batch array" is about the two BATCHED
    ORDER arrays and does not apply here. The live loop passed its whole ticker list unchunked
    (`run_04_quote_loop.py`), which was fine at ~30 markets and is a rejected request at the
    522-532 the full SPX/NDX universe lists.

    The chunking is nearly free, and this is the one number that makes the REST book path
    affordable at all: `/markets/orderbooks` is absent from GET /account/endpoint_costs, so it
    is billed the default 10 tokens PER REQUEST regardless of how many tickers it carries.
    532 markets is 6 requests and 60 tokens against a 300 tok/s read budget.
    """
    tickers = list(tickers)
    out = {}
    for i in range(0, len(tickers), MAX_ORDERBOOK_TICKERS):
        params = [("tickers", t) for t in tickers[i:i + MAX_ORDERBOOK_TICKERS]]
        if depth is not None:
            params.append(("depth", depth))
        resp = _request("GET", "/markets/orderbooks", params=params)
        out.update({b["ticker"]: b for b in (resp.get("orderbooks") or [])})
    return out


# --- portfolio (signed) ---

def balance():
    return _request("GET", "/portfolio/balance", signed=True)


def resting_orders(**params):
    """Every resting order, ALL PAGES, as `{"orders": [...], "cursor": None}`.

    Paged since 2026-08-24, and this is a kill-path bug, not a tidiness one. `LimitQuery`
    defaults to **100**; a full board is three rungs x two sides x ~30 markets = up to 180
    orders. The single-page form returned the first 100, so `strategy.cancel_all` -- which
    `run_04` calls from its `finally` block, and which an operator runs before 16:00 -- cancelled
    100 of 180 and left the rest resting on a live market until it closed. It even re-read to
    verify, and the re-read was truncated the same way, so the count it printed as "remaining"
    was itself wrong.

    The RESPONSE SHAPE is deliberately unchanged -- a dict with an `orders` key -- because all
    three callers do `.get("orders") or []`, and the one that does not
    (`total_resting_order_value`) already carries a comment about exactly that trap. `cursor` is
    None because there is never a next page by the time this returns.

    `limit` is raised to the documented maximum of 1000, so the common case is still one request.
    """
    return {"orders": paged("/portfolio/orders", "orders", signed=True,
                            **{"status": "resting", "limit": 1000, **params}),
            "cursor": None}


def create_order(ticker, side, price, count=config.ORDER_COUNT, client_order_id=None,
                 time_in_force="good_till_canceled", post_only=True,
                 self_trade_prevention_type="maker"):
    """side: 'bid' = buy YES, 'ask' = sell YES. price/count are fixed-point strings.

    `self_trade_prevention_type` must be one of the API's enum ['taker_at_cross', 'maker'] and is
    required. It previously read "cancel_resting", which is in neither the V2 nor the legacy
    enum -- and since _request never retries a POST, every order would have 400'd. 'maker' keeps
    our resting side, which is the right choice for a quoting book.

    `post_only=True` is load-bearing: it guarantees we never cross and become the taker.
    """
    body = {
        "ticker": ticker,
        "side": side,
        "count": count,
        "price": price,
        "time_in_force": time_in_force,
        "self_trade_prevention_type": self_trade_prevention_type,
        "post_only": post_only,
    }
    if client_order_id:
        body["client_order_id"] = client_order_id
    return _request("POST", "/portfolio/events/orders", body=body, signed=True)


def batch_create_orders(orders):
    """POST /portfolio/events/orders/batched. `orders` is a list of create-order bodies.

    Returns the raw per-order result list. **Partial failure is normal**: each element carries its
    own `error`, and a batch does not fail as a unit. A caller that checks only the HTTP status
    will happily believe it has 180 resting orders when it has 140.

    Billed 10 tokens per order, so the throttle is charged for the whole batch rather than for one
    request -- see `_throttle`. Split anything larger than `write_budget()['max_batch']`
    (90 on this account) before calling.
    """
    if not orders:
        return []
    resp = _request("POST", "/portfolio/events/orders/batched",
                    body={"orders": list(orders)}, signed=True,
                    tokens=ORDER_TOKEN_COST * len(orders))
    return resp.get("orders") or []


def batch_cancel_orders(order_ids):
    """DELETE /portfolio/events/orders/batched. Billed 2 tokens per order.

    Per-item results carry `reduced_by` -- the contracts actually cancelled, **zero when that
    item errored** -- so `reduced_by` is the honest confirmation, not the absence of an error.
    """
    if not order_ids:
        return []
    resp = _request("DELETE", "/portfolio/events/orders/batched",
                    body={"orders": [{"order_id": o} for o in order_ids]}, signed=True,
                    tokens=CANCEL_TOKEN_COST * len(order_ids))
    return resp.get("orders") or []


def open_positions(strict=False):
    """{ticker: net contracts}, signed, for every market we currently hold. {} on failure.

    The REST counterpart to the `market_positions` WebSocket channel, which is a DELTA feed and
    therefore says nothing about a position opened before the socket connected. Used to seed
    `kalshi.ws.MarketFeed` at startup so a mid-session restart does not begin flat when it is not.

    `position_fp` is the signed fixed-point contract count -- positive is long YES, negative is
    short. Never raises: a quoter that cannot read its own positions should say so and start
    conservatively, not refuse to start.
    """
    out = {}
    try:
        cursor = None
        while True:
            params = {"limit": 200, "count_filter": "position"}
            if cursor:
                params["cursor"] = cursor
            resp = _request("GET", "/portfolio/positions", params=params, signed=True)
            for row in resp.get("market_positions") or []:
                n = row.get("position_fp", row.get("position"))
                if n is None:
                    continue
                n = float(n)
                if n:
                    out[row["ticker"]] = n
            cursor = resp.get("cursor")
            if not cursor:
                return out
    except Exception:                                        # noqa: BLE001
        # `strict` is for the loss limit: there, "could not read" and "flat" must not be the
        # same answer, because reading a failure as flat marks every position at zero.
        if strict:
            raise
        return out


def total_resting_order_value(**params):
    """Dollars tied up in resting orders, summed from /portfolio/orders. None on failure.

    NOT from `/portfolio/summary/total_resting_order_value`. That endpoint exists and is the
    obvious choice, but it is **FCM-members-only** and returns
    `403 permission_denied` on this account -- the spec says so itself ("only intended for use by
    FCM members (rare)"). Verified 2026-08-21.

    So it is summed from the resting-order list, which does work. Worst-case exposure on a YES
    binary is price x count for a bid; an ask's exposure is (1 - price) x count, because that is
    what it costs if it settles against us.
    """
    try:
        # `resting_orders` returns the RAW response dict, not a list -- iterating it directly
        # walks the keys ('cursor', 'orders') and silently sums zero. Every other caller in the
        # tree does `.get("orders") or []`; so does this one.
        rows = resting_orders(**params).get("orders") or []
    except Exception:                                        # noqa: BLE001
        return None
    total = 0.0
    for o in rows:
        try:
            px = float(o.get("yes_price_dollars") or o.get("price") or 0)
            n = float(o.get("remaining_count") or o.get("count") or 0)
        except (TypeError, ValueError):
            continue
        if o.get("side") == "ask":
            px = 1.0 - px
        total += px * n
    return total


def order_body(ticker, side, price, count, client_order_id=None,
               time_in_force="good_till_canceled", post_only=True,
               self_trade_prevention_type="maker"):
    """One create-order body, for either the single or the batched endpoint.

    Same shape both ways, which is the point: `create_order` and `batch_create_orders` must not
    drift. `self_trade_prevention_type` must be 'maker' or 'taker_at_cross' -- 'cancel_resting'
    is in neither enum and 400'd every order it was ever sent with.
    """
    body = {"ticker": ticker, "side": side, "count": count, "price": price,
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
            "post_only": post_only}
    if client_order_id:
        body["client_order_id"] = client_order_id
    return body


def cancel_order(order_id):
    """Single cancel. 2 tokens, not the default 10 -- confirmed against GET /account/endpoint_costs
    (`DELETE /portfolio/events/orders/:order_id`), which matters because `strategy/cancel_all`
    loops this over the whole board and would otherwise overstate its cost 5x."""
    return _request("DELETE", f"/portfolio/events/orders/{order_id}", signed=True,
                    tokens=CANCEL_TOKEN_COST)


def fp(x, places=4):
    """Float -> Kalshi fixed-point dollar string."""
    return f"{x:.{places}f}"


def fills(**params):
    """Our own executions, ALL pages.

    The record is close to the public trade tape -- trade_id, ticker, count_fp,
    yes/no_price_dollars, created_time -- plus order_id, is_taker and fee_cost, which the public
    tape does not carry. Filterable by ticker, order_id, min_ts, max_ts.

    Paginated deliberately: the endpoint is cursor-based, so a single-page fetch silently
    truncates a session with more fills than the default limit -- and a truncated fill record is
    indistinguishable from a quiet session.
    """
    return paged("/portfolio/fills", "fills", signed=True, **params)


def queue_positions(market_tickers=None, event_ticker=None):
    """Kalshi's own price-time-priority queue position for each resting order.

    This is ground truth for "where were we in the queue", so the study does not have to infer
    it from book depth. Returns nothing when we have no resting orders -- so it is silent in a
    dry run, and only becomes useful once we are actually quoting.
    """
    p = {}
    if market_tickers:
        p["market_tickers"] = ",".join(market_tickers) if not isinstance(market_tickers, str) else market_tickers
    if event_ticker:
        p["event_ticker"] = event_ticker
    return _request("GET", "/portfolio/orders/queue_positions", params=p, signed=True)
