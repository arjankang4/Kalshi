"""Where does order-send latency actually go? Massive pull -> theo -> live Kalshi order.

Scrap harness. Mirrors the structure of `kalshi/client.py` but instruments every leg, so the
answer is a breakdown rather than one number. Changes no production file.

It sends TWO real orders, which is the whole point -- a signed GET (what `strategy/latency.py`
measures) does not include matching-engine accept time, so it cannot answer this:

  order 1  production path, Basic tier   -- urllib, one fresh TCP+TLS per call, 8 req/s throttle
  <upgrade POST /account/api_usage_level/upgrade>
  order 2  best case, Advanced tier      -- pre-warmed pooled connection, throttle at real limit

Both are 1 lot at $0.01 with post_only=True on a market bid far above that, cancelled
immediately. Capital at risk is one cent.

Order 1 is not merely a measurement: the upgrade endpoint requires that at least one of the
last 100 orders was created via API, so it 403s until an order exists. Order 1 unlocks order 2.

Those two orders differ in three ways at once (tier, throttle, connection reuse), so they bound
the problem but do not decompose it. `--matrix` does the decomposition separately, using signed
GET /portfolio/balance -- identical signing and TLS, no order -- which is safe to repeat.

    python -m strategy.latency_order --dry-run     # everything except the two POSTs
    python -m strategy.latency_order               # live: 2 orders, ~$0.01 at risk
    python -m strategy.latency_order --matrix-only # transport decomposition, sends nothing
"""
import argparse
import http.client
import json
import socket
import ssl
import statistics
import sys
import time
import urllib.parse
import uuid
from datetime import datetime, timezone

import config
from kalshi import client as k

OUT_DIR = config.ROOT / "var" / "latency"

# 1c bid on a market whose best bid is far above it. post_only means we can never take, and the
# order is cancelled on the next line, so this cannot fill outside a total market collapse.
TEST_PRICE = "0.0100"
TEST_COUNT = "1.00"          # deliberately NOT config.ORDER_COUNT (10)
MIN_SAFE_BID = 0.05          # refuse to trade a market whose bid is anywhere near our price

_BASE = urllib.parse.urlparse(config.KALSHI_API_BASE)
_HOST = _BASE.hostname
_PREFIX = _BASE.path                      # /trade-api/v2 -- part of the signed string


def _ms(a, b):
    return round((b - a) * 1000, 3)


class Instrumented:
    """One HTTPS connection with every leg timed separately.

    Deliberately not `requests`/`urllib`: both hide the boundary between DNS, TCP, TLS and the
    server's own think time, and that boundary is the entire question here. Signing is imported
    from kalshi.client rather than reimplemented, so the auth path cannot silently diverge.
    """

    def __init__(self, host=_HOST, port=443):
        self.host, self.port = host, port
        self.sock = None
        self.connect_t = {}

    def connect(self):
        t0 = time.perf_counter()
        ai = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)
        t1 = time.perf_counter()
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Nagle would coalesce a small POST body with its headers and wait. Off, as any order
        # path must be -- and it makes the measurement reflect the socket we would actually use.
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        raw.settimeout(20)
        raw.connect(ai[0][4])
        t2 = time.perf_counter()
        ctx = ssl.create_default_context()
        self.sock = ctx.wrap_socket(raw, server_hostname=self.host)
        t3 = time.perf_counter()
        self.connect_t = {"dns_ms": _ms(t0, t1), "tcp_ms": _ms(t1, t2), "tls_ms": _ms(t2, t3),
                          "connect_total_ms": _ms(t0, t3),
                          "tls_version": self.sock.version(), "peer": ai[0][4][0]}
        return self.connect_t

    def request(self, method, path, body=None, signed=True):
        """`path` is relative to /trade-api/v2, matching kalshi.client._request."""
        if self.sock is None:
            self.connect()
        method = method.upper()
        data = json.dumps(body).encode() if body is not None else b""

        t0 = time.perf_counter()
        headers = (k._auth_headers(method, _PREFIX + path) if signed
                   else {"Content-Type": "application/json"})
        t1 = time.perf_counter()

        lines = [f"{method} {_PREFIX + path} HTTP/1.1", f"Host: {self.host}",
                 "Connection: keep-alive", "Accept: application/json"]
        if data:
            lines.append(f"Content-Length: {len(data)}")
        lines += [f"{h}: {v}" for h, v in headers.items()]
        wire = ("\r\n".join(lines) + "\r\n\r\n").encode() + data

        self.sock.sendall(wire)
        t2 = time.perf_counter()
        resp = http.client.HTTPResponse(self.sock, method=method)
        resp.begin()                       # returns once status line + headers have landed
        t3 = time.perf_counter()
        payload = resp.read()              # must drain fully or the connection is unusable
        t4 = time.perf_counter()

        timing = {"sign_ms": _ms(t0, t1), "send_ms": _ms(t1, t2),
                  "ttfb_ms": _ms(t2, t3), "body_ms": _ms(t3, t4),
                  "request_total_ms": _ms(t0, t4), "status": resp.status}
        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            parsed = {"_raw": payload[:400].decode("utf8", "ignore")}
        return parsed, timing

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


# --- preflight -------------------------------------------------------------------

def account_limits():
    lim = k._request("GET", "/account/limits", signed=True)
    cost = 10.0                                  # default_cost from /account/endpoint_costs
    return {
        "usage_tier": lim.get("usage_tier"),
        "grants": lim.get("grants") or [],
        "read_refill": lim["read"]["refill_rate"],
        "write_refill": lim["write"]["refill_rate"],
        "read_req_per_s": lim["read"]["refill_rate"] / cost,
        "write_req_per_s": lim["write"]["refill_rate"] / cost,
        "client_rate_per_s": k._RATE_PER_SEC,
    }


def pick_market(event=None):
    """Most liquid market whose best YES bid is far enough above 1c to be unfillable.

    Reads the book, not /markets: the list endpoint returns bid/ask as null for this series,
    and `orderbook_fp` levels ASCEND, so the best bid is the LAST entry, not the first.
    """
    ev = event or config.event_ticker_for()
    ms = k._request("GET", "/markets", params={"event_ticker": ev, "limit": 1000}).get("markets") or []
    active = [m for m in ms if m.get("status") == "active"]
    if not active:
        raise SystemExit(f"[abort] no active markets in {ev}")
    books = k.orderbooks([m["ticker"] for m in active], depth=5)

    best = []
    for tk, b in books.items():
        yes = (b.get("orderbook_fp") or {}).get("yes_dollars") or []
        if not yes:
            continue
        price, size = float(yes[-1][0]), float(yes[-1][1])
        if price >= MIN_SAFE_BID:
            best.append((price, size, tk))
    if not best:
        raise SystemExit(f"[abort] no market in {ev} has a YES bid >= ${MIN_SAFE_BID:.2f}; "
                         f"a 1c order would be too close to the touch to be safe")
    best.sort(reverse=True)
    price, size, tk = best[0]
    print(f"[market] {tk}  best YES bid ${price:.2f} x {size:.0f}  "
          f"-- our ${float(TEST_PRICE):.2f} bid sits ${price - float(TEST_PRICE):.2f} below it")
    return tk, price


# --- the Massive question --------------------------------------------------------

def massive_legs(collector=None):
    """Time every way we get pricing data, so 'is it Massive?' has a numeric answer.

    The distinction that matters: the live quote path reads options quotes from a WebSocket
    that is already open, which is a dict copy. The REST calls are the slow ones, and they are
    slow because api.massive.com is not in this region.
    """
    from massive import client as mv
    from massive import index as ix
    out = {}

    if collector is not None:
        t0 = time.perf_counter()
        snap = collector.snapshot()
        out["massive_ws_snapshot_ms"] = _ms(t0, time.perf_counter())
        out["massive_ws_symbols"] = len(snap)

    t0 = time.perf_counter()
    try:
        c = mv.option_contracts()   # full listed chain -- startup cost, NOT per-cycle
        out["massive_contracts_ms"] = _ms(t0, time.perf_counter())
        out["massive_contracts_n"] = len(c)
    except Exception as e:                                   # noqa: BLE001
        out["massive_contracts_ms"] = _ms(t0, time.perf_counter())
        out["massive_contracts_err"] = f"{type(e).__name__}: {str(e)[:120]}"

    for label, fn in (("massive_spy_nbbo", mv.stock_nbbo), ("massive_index", ix.snapshot)):
        times = []
        for _ in range(3):                                   # each opens a fresh connection
            t0 = time.perf_counter()
            try:
                fn()
                times.append(_ms(t0, time.perf_counter()))
            except Exception as e:                           # noqa: BLE001
                out[f"{label}_err"] = f"{type(e).__name__}: {str(e)[:120]}"
                break
        if times:
            out[f"{label}_ms"] = round(statistics.median(times), 2)
    return out


# --- transport decomposition (sends no orders, so it is safe to repeat) -----------

def transport_matrix(n=5):
    """Cold vs warm, on a signed GET. Same signing and TLS as an order, no order.

    This is what separates "the network is slow" from "we are sleeping in our own token
    bucket". The two live orders cannot do it: they differ in three variables at once.
    """
    rows = []
    for _ in range(n):
        c = Instrumented()
        ct = c.connect()
        _, rt = c.request("GET", "/portfolio/balance")
        rows.append({"mode": "cold", **ct, **rt,
                     "wall_ms": round(ct["connect_total_ms"] + rt["request_total_ms"], 3)})
        for _ in range(3):                       # same socket, handshake already paid
            _, rt2 = c.request("GET", "/portfolio/balance")
            rows.append({"mode": "warm", **rt2, "wall_ms": rt2["request_total_ms"]})
        c.close()
    return rows


def throttle_cost():
    """How long our own 8 req/s bucket makes an order wait behind two market-data calls.

    This mirrors the quote cycle: markets() then orderbooks() then the order. Kalshi meters
    reads and writes in *separate* buckets, so on the exchange side this contention does not
    exist -- it is entirely ours.
    """
    k._next_slot[0] = 0.0                        # start from an undrained bucket
    k._throttle()                                # stands in for markets()
    k._throttle()                                # stands in for orderbooks()
    t0 = time.perf_counter()
    k._throttle()                                # the order pays this before touching a socket
    return _ms(t0, time.perf_counter())


# --- the two live orders ---------------------------------------------------------

def _order_body(ticker):
    return {"ticker": ticker, "side": "bid", "count": TEST_COUNT, "price": TEST_PRICE,
            "time_in_force": "good_till_canceled", "self_trade_prevention_type": "maker",
            "post_only": True, "client_order_id": str(uuid.uuid4())}


def _order_id(resp):
    """V2 returns order_id FLAT. The nested `resp['order']['order_id']` read is the legacy
    shape and silently yields None, which loses the handle to a live resting order."""
    return resp.get("order_id") or (resp.get("order") or {}).get("order_id")


def order_prod_path(ticker):
    """Order 1: exactly what production does today -- urllib, fresh TLS, throttle engaged,
    issued after a market-data pair so the bucket is drained the way a real cycle drains it."""
    k._RATE_PER_SEC = 8.0
    k._next_slot[0] = 0.0
    ev = config.event_ticker_for()
    t_pre = time.perf_counter()
    ms = k._request("GET", "/markets", params={"event_ticker": ev, "limit": 1000}).get("markets") or []
    k.orderbooks([m["ticker"] for m in ms][:30], depth=5)
    pre_ms = _ms(t_pre, time.perf_counter())

    body = _order_body(ticker)
    t0 = time.perf_counter()
    resp = k._request("POST", "/portfolio/events/orders", body=body, signed=True)
    total = _ms(t0, time.perf_counter())
    return resp, {"mode": "prod_path_urllib", "tier_rate_per_s": k._RATE_PER_SEC,
                  "preceding_marketdata_ms": pre_ms, "order_total_ms": total}


def order_warm_pooled(ticker, conn):
    """Order 2: connection already open and handshaken, no client-side throttle.
    The timer starts at signing, so the 7.6 ms TLS handshake is genuinely not in it."""
    body = _order_body(ticker)
    t0 = time.perf_counter()
    resp, rt = conn.request("POST", "/portfolio/events/orders", body=body)
    total = _ms(t0, time.perf_counter())
    return resp, {"mode": "warm_pooled_no_throttle", **rt, "order_total_ms": total}


def cancel(order_id, conn=None):
    if not order_id:
        return {"cancelled": False, "reason": "no order_id"}
    t0 = time.perf_counter()
    try:
        if conn is not None:
            resp, rt = conn.request("DELETE", f"/portfolio/events/orders/{order_id}")
            return {"cancelled": rt["status"] in (200, 201), "status": rt["status"],
                    "cancel_ms": _ms(t0, time.perf_counter())}
        k.cancel_order(order_id)
        return {"cancelled": True, "cancel_ms": _ms(t0, time.perf_counter())}
    except Exception as e:                                   # noqa: BLE001
        return {"cancelled": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}


def sweep_resting():
    """Last line of defence. Orders are GTC: anything left behind rests until it fills."""
    try:
        r = k.resting_orders()
        orders = r.get("orders") or []
        for o in orders:
            oid = o.get("order_id")
            print(f"[sweep] cancelling orphan {oid} {o.get('ticker')} @ {o.get('yes_price')}")
            try:
                k.cancel_order(oid)
            except Exception as e:                           # noqa: BLE001
                print(f"[sweep] FAILED to cancel {oid}: {e}", file=sys.stderr)
        return len(orders)
    except Exception as e:                                   # noqa: BLE001
        print(f"[sweep] could not list resting orders: {e}", file=sys.stderr)
        return -1


def upgrade_tier():
    """POST /account/api_usage_level/upgrade.

    Answers **201 with an empty body**, which `kalshi.client._request` cannot survive --
    it calls json.loads() on the response unconditionally and raises JSONDecodeError on b"".
    The grant is created regardless, so the exception is purely a client-side parse bug and
    reads as a failed upgrade when it is not. Sent over the instrumented connection, which
    treats an empty payload as {}.
    """
    c = Instrumented()
    try:
        c.connect()
        t0 = time.perf_counter()
        resp, rt = c.request("POST", "/account/api_usage_level/upgrade")
        return {"upgraded": rt["status"] in (200, 201), "status": rt["status"],
                "body": str(resp)[:200], "upgrade_ms": _ms(t0, time.perf_counter())}
    except Exception as e:                                   # noqa: BLE001
        return {"upgraded": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        c.close()


# --- run -------------------------------------------------------------------------

def run(dry_run=False, matrix_only=False, with_ws=False, event=None, skip_order1=False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = OUT_DIR / f"run={stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    rec = {"stamp": stamp, "host": _HOST, "dry_run": dry_run}

    print(f"\n{'='*74}\n  ORDER-PATH LATENCY  {stamp}  ->  {_HOST}\n{'='*74}")

    rec["limits_before"] = account_limits()
    lb = rec["limits_before"]
    print(f"[account] tier={lb['usage_tier']} grants={len(lb['grants'])} | "
          f"exchange allows read {lb['read_req_per_s']:.0f}/s write {lb['write_req_per_s']:.0f}/s "
          f"| our client is hardcoded to {lb['client_rate_per_s']:.0f}/s")

    print("\n-- Massive: how long does pulling pricing data actually take? --")
    collector = None
    if with_ws:
        from massive import client as mv
        from pricing import universe
        collector = mv.QuoteCollector([])       # caller must subscribe; kept opt-in
        collector.start()
    try:
        rec["massive"] = massive_legs(collector)
    finally:
        if collector is not None:
            collector.stop()
    for kk, v in sorted(rec["massive"].items()):
        print(f"  {kk:<28} {v}")

    print("\n-- Kalshi transport: cold vs warm (signed GET, no order) --")
    rows = transport_matrix()
    rec["transport"] = rows
    cold = [r for r in rows if r["mode"] == "cold"]
    warm = [r for r in rows if r["mode"] == "warm"]

    def med(rs, key):
        v = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return round(statistics.median(v), 2) if v else None

    rec["transport_summary"] = {
        "cold_dns_ms": med(cold, "dns_ms"), "cold_tcp_ms": med(cold, "tcp_ms"),
        "cold_tls_ms": med(cold, "tls_ms"), "cold_sign_ms": med(cold, "sign_ms"),
        "cold_ttfb_ms": med(cold, "ttfb_ms"), "cold_wall_ms": med(cold, "wall_ms"),
        "warm_sign_ms": med(warm, "sign_ms"), "warm_ttfb_ms": med(warm, "ttfb_ms"),
        "warm_wall_ms": med(warm, "wall_ms"),
        "tls_version": cold[0].get("tls_version") if cold else None,
    }
    ts = rec["transport_summary"]
    print(f"  cold: dns {ts['cold_dns_ms']} + tcp {ts['cold_tcp_ms']} + TLS {ts['cold_tls_ms']} "
          f"+ sign {ts['cold_sign_ms']} + ttfb {ts['cold_ttfb_ms']} = {ts['cold_wall_ms']} ms")
    print(f"  warm: sign {ts['warm_sign_ms']} + ttfb {ts['warm_ttfb_ms']} = {ts['warm_wall_ms']} ms"
          f"   ({ts['tls_version']})")
    if ts["cold_wall_ms"] and ts["warm_wall_ms"]:
        print(f"  -> connection reuse is worth {ts['cold_wall_ms'] - ts['warm_wall_ms']:.2f} ms "
              f"per call, of which {ts['cold_tls_ms']:.2f} ms is the TLS handshake")

    rec["throttle_wait_ms"] = throttle_cost()
    print(f"\n-- Our own token bucket --\n  an order issued after markets()+orderbooks() sleeps "
          f"{rec['throttle_wait_ms']:.1f} ms in _throttle() before it opens a socket")

    if matrix_only:
        _write(rec, outdir)
        return rec

    ticker, best_bid = pick_market(event)
    rec["ticker"], rec["best_bid"] = ticker, best_bid

    if dry_run:
        print("\n[dry-run] stopping before the two POSTs. Nothing was sent.")
        _write(rec, outdir)
        return rec

    # ---- order 1: production path, Basic tier ----
    oid1 = None
    if skip_order1:
        print("\n-- ORDER 1  SKIPPED (already sent this session) --")
        rec["order_1"] = {"skipped": True}
    else:
      try:
        print(f"\n-- ORDER 1  production path, tier={lb['usage_tier']} --")
        resp1, t1 = order_prod_path(ticker)
        oid1 = _order_id(resp1)
        t1["order_id"] = oid1
        print(f"  market data first: {t1['preceding_marketdata_ms']:.1f} ms")
        print(f"  ORDER ACK: {t1['order_total_ms']:.1f} ms   order_id={oid1}")
        if not oid1:
            print("  [warn] order_id is None -- resting order with no handle", file=sys.stderr)
        rec["order_1"] = t1
      finally:
        rec["cancel_1"] = cancel(oid1)
        print(f"  cancel: {rec.get('cancel_1')}")

    # ---- upgrade ----
    print("\n-- TIER UPGRADE --")
    if any(g.get("level") == "advanced" for g in lb["grants"]):
        rec["upgrade"] = {"skipped": True, "reason": "advanced grant already present"}
        print("  advanced grant already present; not re-requesting")
    else:
        rec["upgrade"] = upgrade_tier()
    print(f"  {rec['upgrade']}")
    rec["limits_after"] = account_limits()
    la = rec["limits_after"]
    print(f"[account] tier={la['usage_tier']} grants={len(la['grants'])} | "
          f"read {la['read_req_per_s']:.0f}/s write {la['write_req_per_s']:.0f}/s")
    try:
        from storage.rest_backfill import raise_rate_limit
        raise_rate_limit()                    # the fix that already exists and is never called
    except Exception as e:                    # noqa: BLE001
        print(f"[rate] raise_rate_limit failed ({e}); lifting manually")
        k._RATE_PER_SEC = max(8.0, la["read_req_per_s"] * 0.9)
    rec["client_rate_after"] = k._RATE_PER_SEC
    print(f"[rate] client bucket now {k._RATE_PER_SEC:.1f} req/s")

    # ---- order 2: warm pooled, throttle lifted, Advanced tier ----
    conn, oid2 = Instrumented(), None
    try:
        warm_ct = conn.connect()               # handshake paid BEFORE the timer starts
        conn.request("GET", "/portfolio/balance")   # ensure the socket is hot
        print(f"\n-- ORDER 2  warm pooled, tier={la['usage_tier']} --")
        print(f"  (connection opened ahead of time: {warm_ct['connect_total_ms']:.1f} ms, "
              f"not counted below)")
        resp2, t2 = order_warm_pooled(ticker, conn)
        oid2 = _order_id(resp2)
        t2["order_id"] = oid2
        print(f"  ORDER ACK: {t2['order_total_ms']:.1f} ms  "
              f"(sign {t2['sign_ms']} + send {t2['send_ms']} + ttfb {t2['ttfb_ms']})  "
              f"status={t2['status']} order_id={oid2}")
        if not oid2:
            print(f"  [warn] no order_id; body={str(resp2)[:200]}", file=sys.stderr)
        rec["order_2"] = t2
    finally:
        rec["cancel_2"] = cancel(oid2, conn)
        print(f"  cancel: {rec.get('cancel_2')}")
        conn.close()

    rec["orphans_swept"] = sweep_resting()
    _report(rec)
    _write(rec, outdir)
    return rec


def _write(rec, outdir):
    p = outdir / "order_latency.json"
    p.write_text(json.dumps(rec, indent=2, default=str))
    print(f"\n[out] {p}")


def _report(rec):
    o1 = rec.get("order_1", {})
    o2 = rec.get("order_2", {})
    ts = rec.get("transport_summary", {})
    mv_ = rec.get("massive", {})
    print(f"\n{'='*74}\n  WHERE THE TIME GOES\n{'='*74}")
    print(f"{'leg':<44}{'ms':>10}")
    line = lambda n, v: print(f"{n:<44}{v:>10}") if v is not None else None
    line("Massive: WS snapshot (what quoting reads)", mv_.get("massive_ws_snapshot_ms"))
    line("Massive: REST options contracts", mv_.get("massive_contracts_ms"))
    line("Massive: REST SPY NBBO", mv_.get("massive_spy_nbbo_ms"))
    line("Massive: REST index snapshot", mv_.get("massive_index_ms"))
    print("-" * 54)
    line("Kalshi: our own throttle sleep", rec.get("throttle_wait_ms"))
    line("Kalshi: TCP handshake", ts.get("cold_tcp_ms"))
    line("Kalshi: TLS handshake", ts.get("cold_tls_ms"))
    line("Kalshi: RSA-PSS sign", ts.get("cold_sign_ms"))
    line("Kalshi: server think + first byte (warm)", ts.get("warm_ttfb_ms"))
    print("-" * 54)
    line("ORDER 1  production path, Basic", o1.get("order_total_ms"))
    line("ORDER 2  warm pooled, Advanced", o2.get("order_total_ms"))
    a, b = o1.get("order_total_ms"), o2.get("order_total_ms")
    if a and b:
        print(f"\n  order 2 is {a - b:+.1f} ms vs order 1  ({(1 - b/a) * 100:.0f}% faster)")
        print("  NOTE: three variables changed at once (tier, throttle, connection reuse).")
        print("  The transport rows above are what attribute it.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="everything except the two POSTs")
    ap.add_argument("--matrix-only", action="store_true", help="transport only; sends nothing")
    ap.add_argument("--with-ws", action="store_true",
                    help="also open the Massive options socket (conflicts with run_02/run_04)")
    ap.add_argument("--skip-order1", action="store_true",
                    help="order 1 already sent; go straight to upgrade + order 2")
    ap.add_argument("--event", default=None)
    a = ap.parse_args(argv)
    run(dry_run=a.dry_run, matrix_only=a.matrix_only, with_ws=a.with_ws, event=a.event,
        skip_order1=a.skip_order1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
