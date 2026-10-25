"""Measure the quote->theo->order chain in real time, and what the market did while we thought.

The question this answers is "how fast are we relative to everyone else", so it records two
different things and they are not interchangeable:

  * **Our latency** -- how long each stage of our own loop takes.
  * **The market's motion during it** -- how many order-book updates landed on our own markets
    between the quote we priced off and the moment we would have acted. That number, not our
    millisecond count, is what says whether we were late: 3 ms is slow if the book moved twice
    in it, and 300 ms is fine if nothing happened.

Sends nothing. The order path is probed with a signed GET (`/portfolio/balance`) as a stand-in
for the round trip, which is honest about what it is -- a real order ack includes matching-engine
work this does not. It is labelled `kalshi_rtt_ms`, never `order_latency`.

    python -m strategy.latency --minutes 1
    python -m strategy.latency --minutes 5 --interval 0.5
"""
import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from websockets.asyncio.client import connect

import config
from kalshi import client as k
from massive import client as mv
from pricing import theo, universe, validate_theo

OUT_DIR = config.ROOT / "var" / "latency"


class DeltaCounter:
    """Counts Kalshi book updates per market on its own connection.

    Separate from the recorder deliberately: the recorder writes to disk for later, and reading
    its tape would tell us what happened minutes ago. This needs the count *while we decide*.
    """

    def __init__(self, tickers):
        self.tickers = list(tickers)
        self.n = 0
        self.per_market = {}
        self.events = []                  # (recv_ns, ticker) -- trimmed, see _trim
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
        self._thread.start()
        return self

    async def _run(self):
        url = config.KALSHI_WS_BASE
        hdrs = k._auth_headers("GET", urlparse(url).path)
        hdrs.pop("Content-Type", None)
        try:
            async with connect(url, additional_headers=hdrs, open_timeout=15,
                               ping_interval=20, ping_timeout=30, max_size=None) as ws:
                await ws.send(json.dumps({"id": 1, "cmd": "subscribe", "params": {
                    "channels": ["orderbook_delta"], "send_initial_snapshot": False,
                    "market_tickers": self.tickers, "use_yes_price": False}}))
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1)
                    except asyncio.TimeoutError:
                        continue
                    d = json.loads(raw)
                    if d.get("type") != "orderbook_delta":
                        continue
                    tk = (d.get("msg") or {}).get("market_ticker")
                    with self._lock:
                        self.n += 1
                        self.per_market[tk] = self.per_market.get(tk, 0) + 1
                        self.events.append((time.time_ns(), tk))
                        self._trim()
        except Exception as e:                       # noqa: BLE001
            print(f"[delta] stream ended: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)

    def _trim(self, keep_ns=120 * 10**9):
        """Bounded by time, not count: this runs for a whole session in a long-lived quoter."""
        cut = time.time_ns() - keep_ns
        while self.events and self.events[0][0] < cut:
            self.events.pop(0)

    def since(self, t0_ns):
        with self._lock:
            ev = [e for e in self.events if e[0] >= t0_ns]
        return len(ev), len({t for _, t in ev})

    def stop(self):
        self._stop.set()
        return self


def _quote_age_ms(snapshot, now_ms):
    """Age of the quotes we priced off. Massive `Q.t` is MILLISECONDS (the REST sip_timestamp
    is nanoseconds -- same vendor, same concept, different unit)."""
    ages = [now_ms - q["t"] for q in snapshot.values() if q.get("t")]
    return ages


def run(minutes, interval, event=None):
    event = event or config.event_ticker_for()
    markets = k.markets(event_ticker=event)
    root, all_strikes, _ = universe.option_root_and_strikes()
    expiry = config.occ_expiry()
    bounds = theo.boundaries(markets)
    seed = (min(bounds) + max(bounds)) / 2 if bounds else None
    _, tickers = universe.desired_tickers(markets, all_strikes, root, expiry, seed)

    stamp = config.run_stamp()
    outdir = OUT_DIR / f"run={stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    fh = open(outdir / "latency.jsonl", "a")

    print(f"[latency] {event}: {len(markets)} markets, {len(tickers)} option tickers")
    col = mv.QuoteCollector(tickers).start()
    dc = DeltaCounter([m["ticker"] for m in markets]).start()
    col.wait_for_coverage(max(4, len(tickers) // 4), timeout=30)

    deadline = time.time() + minutes * 60
    rows = []
    try:
        while time.time() < deadline:
            cyc = {"ts": datetime.now(timezone.utc).isoformat()}
            t_start = time.time_ns()

            t0 = time.perf_counter()
            snap = col.snapshot()
            cyc["snapshot_ms"] = round((time.perf_counter() - t0) * 1000, 3)

            now_ms = time.time() * 1000
            ages = _quote_age_ms(snap, now_ms)
            if ages:
                ages.sort()
                cyc["n_quotes"] = len(ages)
                cyc["quote_age_ms_p50"] = round(ages[len(ages) // 2], 1)
                cyc["quote_age_ms_p95"] = round(ages[int(len(ages) * .95)], 1)
                cyc["quote_age_ms_min"] = round(ages[0], 1)

            t0 = time.perf_counter()
            pricer, fairs = validate_theo.price_markets(markets, snap, root, expiry, all_strikes)
            cyc["theo_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            cyc["spot"] = pricer.spot
            cyc["n_priced"] = sum(1 for v in fairs.values() if v[0] is not None)

            t0 = time.perf_counter()
            try:
                k._request("GET", "/portfolio/balance", signed=True)
                cyc["kalshi_rtt_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            except Exception as e:                   # noqa: BLE001
                cyc["kalshi_rtt_err"] = type(e).__name__

            # The decisive number: how much did OUR markets move while we did all of the above?
            n_deltas, n_markets_moved = dc.since(t_start)
            cyc["deltas_during"] = n_deltas
            cyc["markets_moved_during"] = n_markets_moved
            cyc["decide_to_send_ms"] = round((time.time_ns() - t_start) / 1e6, 2)

            fh.write(json.dumps(cyc) + "\n")
            fh.flush()
            rows.append(cyc)
            time.sleep(max(0, interval - (time.time_ns() - t_start) / 1e9))
    finally:
        col.stop()
        dc.stop()
        fh.close()

    _summary(rows, dc, outdir)
    return rows


def _summary(rows, dc, outdir):
    def col(name):
        return [r[name] for r in rows if isinstance(r.get(name), (int, float))]

    print(f"\n[latency] {len(rows)} cycles -> {outdir/'latency.jsonl'}")
    print(f"{'metric':<26}{'p50':>10}{'p95':>10}{'max':>10}")
    for name in ("quote_age_ms_p50", "theo_ms", "kalshi_rtt_ms", "decide_to_send_ms",
                 "deltas_during", "markets_moved_during"):
        v = sorted(col(name))
        if not v:
            continue
        p50 = v[len(v) // 2]
        p95 = v[int(len(v) * .95)] if len(v) > 1 else v[0]
        print(f"{name:<26}{p50:>10.2f}{p95:>10.2f}{max(v):>10.2f}")
    tot = dc.n
    print(f"\n[latency] total book updates on our markets during the run: {tot:,}")
    if rows:
        span = len(rows) * (rows[0].get('decide_to_send_ms', 0) / 1000 or 1)
        print(f"[latency] our own loop ran {len(rows)} times in that window")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between cycles")
    ap.add_argument("--event", default=None)
    a = ap.parse_args(argv)
    run(a.minutes, a.interval, a.event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
