"""Kalshi WebSocket recorder: live frames -> gzipped local segments.

Captured verbatim. Conformance to the REST schema happens later, in promotion,
so a mapping bug is repairable by re-running against the raw -- which is the
only reason the raw is worth keeping.

Two invariants the rest of the pipeline depends on:

  1. Nothing is dropped silently. A frame lost to backpressure writes a marker,
     because a silent drop is indistinguishable from a quiet market.
  2. Sequence gaps close a book epoch and open a new one. Order book validity is
     then a per-epoch property rather than something a query has to re-derive.

Usage:  python -m storage.ws_recorder [--duration SECONDS] [--dry-run]
                                      [--ignore-gate]

Capture is gated on config.CAPTURE_SESSION_ENABLED and the ET session window.
With the gate closed this process stays up and idles: no subscription, no REST
calls, no segments. --ignore-gate forces capture for a one-off test run.
"""
import argparse
import asyncio
import contextlib
import json
import signal
import sys
import time
import uuid
from urllib.parse import urlparse

from websockets.asyncio.client import connect

import config
from kalshi import client as k
from storage import paths
from storage.segments import Segment

QUEUE_MAX = 20_000          # ~5 MB of frames; backpressure long before RAM matters
# How long teardown waits for the writer to persist frames the reader already accepted.
# The writer's own loop condition (`... or not self.queue.empty()`) was written to drain, and it
# never got to: the teardown cancelled every task at once, so a queue holding up to QUEUE_MAX
# frames was discarded on EVERY close -- deadline, SIGTERM and each of the day's reconnects. The
# frames were not merely lost: nothing cleared the queue either, so whatever survived was written
# by the NEXT connection's writer, stamped with the NEXT session id, into the NEXT segment.
# 5s is generous against a queue the writer empties at socket speed and short enough that SIGTERM
# still feels immediate.
DRAIN_TIMEOUT_S = 5.0
RECONNECT_BASE = 1.0
RECONNECT_MAX = 30.0

# Frame types that carry a sid+seq on the orderbook_delta stream. `ok` and
# `unsubscribed` CONSUME sequence numbers, so contiguity accounting that counts
# only data frames reports a phantom gap on every control response.
SEQ_BEARING = {"orderbook_delta", "orderbook_snapshot", "ok", "unsubscribed"}


def discover_close_series():
    """Index/ETF series that list a 4pm-close event.

    Configured series are the fast path. This sweep exists so a newly listed Dow
    or ETF close market is picked up without a code change -- the user asked for
    "any other index/etf close price at 4pm market", and today Kalshi lists none
    beyond S&P and Nasdaq. Run at startup and at ET day rollover, not every tick.
    """
    try:
        allser = k.paged("/series", "series", limit=200)
    except Exception as e:                           # noqa: BLE001
        print(f"[discover] series sweep failed ({type(e).__name__}); "
              f"falling back to the configured list", file=sys.stderr)
        return list(config.CAPTURE_SERIES), []
    cand = [s["ticker"] for s in allser
            if s.get("category") == "Financials"
            and any(p in s["ticker"] for p in config.CAPTURE_INDEX_PATTERNS)]

    stamp = f"{config.today_et_date():%y%b%d}".upper()
    suffix = f"-{stamp}{config.CAPTURE_CLOSE_SUFFIX}"
    extra = []
    for st in cand:
        if st in config.CAPTURE_SERIES:
            continue
        for status in ("open", "unopened"):
            try:
                evs = k.paged("/events", "events", limit=200,
                              series_ticker=st, status=status)
            except Exception:                        # noqa: BLE001
                continue
            if any(e["event_ticker"].endswith(suffix) for e in evs):
                extra.append(st)
                break
    if extra:
        print(f"[discover] index/ETF series with a {config.CAPTURE_CLOSE_SUFFIX} "
              f"event that are NOT in CAPTURE_SERIES: {extra}", flush=True)
    return list(config.CAPTURE_SERIES) + extra, extra


def resolve_target_markets(series=None):
    """Today's 4pm-close markets across the capture series.

    Resolved live and re-run on a timer: a market can list intraday, and an
    event that is still `unopened` at recorder start still returns its markets.
    """
    d = config.today_et_date()
    stamp = f"{d:%y%b%d}".upper()
    series = series or list(config.CAPTURE_SERIES)
    tickers, per_event = [], {}
    for st in series:
        ev = f"{st}-{stamp}{config.CAPTURE_CLOSE_SUFFIX}"
        try:
            ms = k.paged("/markets", "markets", limit=1000, event_ticker=ev)
        except Exception as e:                       # noqa: BLE001
            per_event[ev] = f"error: {type(e).__name__}"
            continue
        per_event[ev] = len(ms)
        tickers.extend(m["ticker"] for m in ms)
    return sorted(set(tickers)), per_event


class Recorder:
    def __init__(self, channels=None, use_yes_price=False, dry_run=False,
                 ignore_gate=False):
        self.channels = list(channels or config.CAPTURE_CHANNELS)
        # Segments carry every channel in one stream. Frames are tagged by type,
        # and the sid->channel binding is recorded from the `subscribed` frames,
        # so promotion can route without a separate file per channel.
        self.channel = "all"
        self.use_yes_price = use_yes_price
        self.dry_run = dry_run
        self.ignore_gate = ignore_gate
        self.session_id = uuid.uuid4().hex
        self.dt = str(config.today_et_date())
        self.capture_day = self.dt        # ET day the current subscription covers

        self.queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self.segment = None
        self.seg_no = 0
        self.stop = asyncio.Event()

        # All four are CONNECTION-scoped: sid restarts at 1 on every socket, so they must be
        # cleared by _new_connection() before subscribing. See its docstring.
        self.conn_no = 0
        self.subscribed = {}        # sid -> {"channel","markets"}
        self.sid_channel = {}       # sid -> channel name
        self.epoch = {}             # sid -> epoch state
        self.last_seq = {}          # sid -> last seen seq
        self.markets = []
        self.series = list(config.CAPTURE_SERIES)
        self.cmd_id = 0

        self.stats = {"frames": 0, "dropped": 0, "gaps": 0, "reconnects": 0,
                      "segments": 0, "resubscribes": 0, "dropped_on_close": 0}

    async def _sleep(self, seconds):
        """Sleep that wakes on shutdown. A plain asyncio.sleep here would hold
        the whole gather open for a full resolve interval after SIGTERM."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        return self.stop.is_set()

    # --- segments --------------------------------------------------------

    def _sub_state(self):
        return {"channels": self.channels, "use_yes_price": self.use_yes_price,
                "n_markets": len(self.markets),
                "series": self.series,
                "sids": {str(s): v.get("channel") for s, v in self.subscribed.items()}}

    def _open_segment(self):
        # Re-evaluate the ET date every roll. Pinning it at startup would file a
        # long-running process's segments under the day it booted.
        self.dt = str(config.today_et_date())
        self.seg_no += 1
        self.segment = Segment(self.dt, self.channel, self.session_id,
                               self.seg_no, self._sub_state())
        self.stats["segments"] += 1
        # A segment must be self-describing: data frames carry only `sid`, so
        # without the subscription state a reader cannot attribute them.
        self.segment.write({"v": 1, "session": self.session_id,
                            "recv_ns": time.time_ns(), "kind": "segment_header",
                            "sub": self._sub_state(), "dt": self.dt})

    def _close_segment(self):
        if self.segment is None:
            return None
        meta = self.segment.close()
        print(f"[seg] {self.segment.path.name} lines={meta['line_count']} "
              f"plain={meta['uncompressed_bytes']}B gz={meta['compressed_bytes']}B "
              f"ratio={meta['uncompressed_bytes']/max(1,meta['compressed_bytes']):.1f}x "
              f"dropped={meta['dropped_frames']}", flush=True)
        self.segment = None
        return meta

    # --- epochs ----------------------------------------------------------

    def _open_epoch(self, sid, seq, reason):
        eid = uuid.uuid4().hex
        self.epoch[sid] = {"book_epoch_id": eid, "seq_start": seq,
                           "open_reason": reason, "opened_with_snapshot": False,
                           "start_ns": time.time_ns()}
        if self.segment:
            self.segment.epochs_opened.append(eid)
            self.segment.write({"v": 1, "kind": "epoch_open", "sid": sid,
                                "session": self.session_id,
                                "book_epoch_id": eid, "seq_start": seq,
                                "open_reason": reason, "recv_ns": time.time_ns()})
        return eid

    def _close_epoch(self, sid, reason, gap_lo=None, gap_hi=None, code=None):
        e = self.epoch.pop(sid, None)
        if not e:
            return
        rec = {"v": 1, "kind": "epoch_close", "sid": sid,
               "session": self.session_id,
               "book_epoch_id": e["book_epoch_id"], "seq_start": e["seq_start"],
               "seq_end": self.last_seq.get(sid), "close_reason": reason,
               "opened_with_snapshot": e["opened_with_snapshot"],
               "gap_seq_lo": gap_lo, "gap_seq_hi": gap_hi,
               "close_error_code": code, "recv_ns": time.time_ns()}
        if self.segment:
            self.segment.epochs_closed.append(e["book_epoch_id"])
            self.segment.write(rec)

    # --- socket ----------------------------------------------------------

    async def _send(self, ws, payload):
        self.cmd_id += 1
        payload["id"] = self.cmd_id
        body = json.dumps(payload, separators=(",", ":"))
        await ws.send(body)
        # Outbound commands are part of the record: they are what explains a sid
        # appearing, a market being added, or a resubscribe after a gap.
        if self.segment:
            self.segment.write({"v": 1, "kind": "command", "session": self.session_id,
                                "recv_ns": time.time_ns(), "frame": payload})
        return self.cmd_id

    async def _subscribe(self, ws, tickers):
        await self._send(ws, {"cmd": "subscribe", "params": {
            # One command, several channels -> one sid per channel.
            "channels": self.channels,
            "send_initial_snapshot": True,
            "market_tickers": tickers,
            # Recorded explicitly. Kalshi intends to flip this default and then
            # remove the flag; without knowing which basis a frame used, the
            # no-side prices become uninterpretable after the fact.
            "use_yes_price": self.use_yes_price,
        }})

    async def reader(self, ws):
        while not self.stop.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            recv_ns = time.time_ns()
            self.stats["frames"] += 1
            try:
                self.queue.put_nowait((recv_ns, msg))
            except asyncio.QueueFull:
                # Never a silent discard.
                self.stats["dropped"] += 1
                if self.segment:
                    self.segment.dropped += 1
                    self.segment.write({"v": 1, "kind": "drop_marker",
                                        "session": self.session_id,
                                        "recv_ns": recv_ns,
                                        "reason": "queue_full",
                                        "queue_max": QUEUE_MAX})

    async def writer(self, ws):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                recv_ns, msg = await asyncio.wait_for(self.queue.get(), timeout=1)
            except asyncio.TimeoutError:
                if self.segment:
                    self.segment.sync()
                    if self.segment.should_roll():
                        self._roll()
                continue

            try:
                d = json.loads(msg)
            except Exception:                        # noqa: BLE001
                if self.segment:
                    self.segment.write({"v": 1, "kind": "unparseable",
                                        "recv_ns": recv_ns, "raw": msg[:2000]})
                continue

            ftype = d.get("type")
            sid = d.get("sid")
            seq = d.get("seq")

            await self._track(ws, d, ftype, sid, seq, recv_ns)

            if self.segment:
                self.segment.write({"v": 1, "session": self.session_id,
                                    "recv_ns": recv_ns, "frame": d})
                self.segment.note_frame(recv_ns, sid, seq, ftype)
                self.segment.sync()
                if self.segment.should_roll():
                    self._roll()

    async def _track(self, ws, d, ftype, sid, seq, recv_ns):
        """Subscription bookkeeping and gap detection. Cheap; runs per frame."""
        if ftype == "subscribed":
            m = d.get("msg", {})
            new_sid = m.get("sid")
            ch = m.get("channel")
            self.subscribed[new_sid] = {"channel": ch, "markets": len(self.markets)}
            self.sid_channel[new_sid] = ch
            if self.segment:
                self.segment.subscribe_state = self._sub_state()
                # Data frames carry only `sid`. Without this record, promotion
                # cannot tell which channel a frame came from.
                self.segment.write({"v": 1, "kind": "sid_channel",
                                    "session": self.session_id,
                                    "sid": new_sid, "channel": ch,
                                    "recv_ns": time.time_ns()})
            # Only orderbook_delta carries `seq`. trade and ticker have no
            # sequence number at all, so a book epoch over them would assert a
            # contiguity guarantee that does not exist.
            if ch == "orderbook_delta":
                self._open_epoch(new_sid, None, "subscribe")
            print(f"[sub] sid={new_sid} channel={ch} "
                  f"markets={len(self.markets)}", flush=True)
            return

        if ftype == "error":
            code = (d.get("msg") or {}).get("code")
            print(f"[err] code={code} {(d.get('msg') or {}).get('msg')!r}", flush=True)
            # 10 channel error, 17 internal, 25 subscription buffer overflow are
            # terminal per the spec: the subscription is dead until resubscribed.
            if code in (10, 17, 25):
                for s in list(self.epoch):
                    self._close_epoch(s, f"error_{code}", code=code)
                await self._resubscribe(ws, f"resubscribe_after_error_{code}")
            return

        if seq is None or sid is None:
            return
        if self.sid_channel.get(sid) not in (None, "orderbook_delta"):
            # A seq on a non-book channel would be unexpected; record it rather
            # than folding it into book-epoch accounting.
            return

        prev = self.last_seq.get(sid)
        if prev is not None and seq != prev + 1:
            self.stats["gaps"] += 1
            print(f"[GAP] sid={sid} expected {prev+1} got {seq}", flush=True)
            self._close_epoch(sid, "seq_gap", gap_lo=prev + 1, gap_hi=seq - 1)
            self._open_epoch(sid, seq, "resubscribe_after_gap")
            await self._resubscribe(ws, "gap")
        self.last_seq[sid] = seq

        e = self.epoch.get(sid)
        if e is not None:
            if e["seq_start"] is None:
                e["seq_start"] = seq
            if ftype == "orderbook_snapshot":
                e["opened_with_snapshot"] = True

    def _new_connection(self):
        """Drop all per-connection state before subscribing on a fresh socket.

        `sid` is only unique WITHIN a connection -- Kalshi restarts numbering at 1 every time --
        so `last_seq`, `epoch`, `sid_channel` and `subscribed` are connection-scoped even though
        they are keyed by sid alone. Carrying them across a reconnect makes the new connection's
        `sid=1, seq=1` look like a gap after the old connection's last seq, and the recorder
        reacts by closing a bogus epoch and force-resubscribing.

        Observed in var/log/ws-recorder.log:

            [sub] sid=1 channel=orderbook_delta markets=522
            [GAP] sid=1 expected 524 got 1      <- 523 = 522 snapshots + 1 control frame
            [err] code=7 'Unknown subscription ID'

        The cost is not just noise: the spurious epoch closes with opened_with_snapshot=False,
        which marks that window unreconstructable and fails the day's promotion. It fires on the
        first frames of every session, because the nightly gate close tears the socket down.

        segments.py:81-87 fixed this same class of bug for segment accounting by keying on
        session+sid; this is the recorder's own copy of that state.
        """
        self.conn_no += 1
        if self.last_seq or self.epoch or self.subscribed:
            print(f"[conn] #{self.conn_no}: clearing {len(self.subscribed)} sub / "
                  f"{len(self.epoch)} epoch / {len(self.last_seq)} seq entries carried from "
                  f"the previous connection", flush=True)
        self.subscribed.clear()
        self.sid_channel.clear()
        self.last_seq.clear()
        self.epoch.clear()

    async def _resubscribe(self, ws, why):
        """A fresh subscription yields a new sid and a fresh snapshot, which is
        what makes the new epoch reconstructable."""
        self.stats["resubscribes"] += 1
        if self.subscribed:
            with contextlib.suppress(Exception):
                await self._send(ws, {"cmd": "unsubscribe",
                                      "params": {"sids": list(self.subscribed)}})
        self.subscribed.clear()
        self.sid_channel.clear()
        self.last_seq.clear()
        if self.markets:
            await self._subscribe(ws, self.markets)

    def _roll(self):
        self._close_segment()
        self._open_segment()

    async def resolver(self, ws):
        """Re-resolve the universe and add newly-opened markets in place."""
        while not self.stop.is_set():
            if await self._sleep(config.CAPTURE_RESOLVE_SECONDS):
                return
            today = str(config.today_et_date())
            loop = asyncio.get_running_loop()
            if today != self.capture_day:
                self.series, _x = await loop.run_in_executor(
                    None, discover_close_series)
            new, per_event = await loop.run_in_executor(
                None, resolve_target_markets, self.series)
            if not new:
                continue

            if today != self.capture_day:
                # Day rollover. Yesterday's markets have settled; keeping them
                # subscribed would accumulate dead tickers indefinitely. A full
                # resubscribe also starts a clean epoch on the new day.
                print(f"[resolve] ET day {self.capture_day} -> {today}, "
                      f"resubscribing to {len(new)} markets", flush=True)
                self.capture_day = today
                self.markets = new
                self._roll()
                await self._resubscribe(ws, "day_rollover")
                continue

            added = sorted(set(new) - set(self.markets))
            if added and self.subscribed:
                sid = next(iter(self.subscribed))
                await self._send(ws, {"cmd": "update_subscription", "params": {
                    "sid": sid, "action": "add_markets", "market_tickers": added}})
                self.markets = sorted(set(self.markets) | set(added))
                print(f"[resolve] +{len(added)} markets -> {len(self.markets)} "
                      f"({', '.join(added[:3])}{'...' if len(added) > 3 else ''})",
                      flush=True)

    # --- capture gate ----------------------------------------------------

    def _gate_open(self):
        return self.ignore_gate or config.in_capture_session()

    async def _gate_watch(self):
        """Return when the capture window closes.

        run() waits FIRST_COMPLETED, so this task completing tears the
        subscription down through exactly the same path as a disconnect --
        epochs closed, segment sealed -- rather than leaving a half-written
        segment for the flush to pick up.
        """
        while not self.stop.is_set():
            if not self._gate_open():
                print(f"[gate] window closed ({config.capture_gate_reason()})"
                      f" -- disconnecting", flush=True)
                return
            if await self._sleep(config.CAPTURE_GATE_POLL_SECONDS):
                return

    def _print_universe(self, per_event):
        print(f"[universe] {len(self.markets)} markets", flush=True)
        for ev, n in per_event.items():
            print(f"    {ev:<28} {n}", flush=True)

    async def run(self, duration=None):
        url = config.KALSHI_WS_BASE
        path = urlparse(url).path
        backoff = RECONNECT_BASE
        # Started when capture actually begins, not at process launch. Universe resolution is
        # ~20 s of REST calls, so a deadline set here would already have expired by the time the
        # socket opens -- `--duration 25` then produced a clean-looking run with ZERO frames,
        # because _deadline returned instantly and FIRST_COMPLETED tore the reader down before
        # its first recv. A pre-flight check that silently captures nothing is worse than none.
        deadline = None

        self.capture_day = str(config.today_et_date())
        loop = asyncio.get_running_loop()
        resolved_day = None
        gate_logged = False

        if self.dry_run:
            # Resolution is deliberately NOT gated: --dry-run opens no socket and
            # writes nothing, so it stays usable for calibration while capture is
            # switched off.
            self.series, _extra = await loop.run_in_executor(
                None, discover_close_series)
            self.markets, per_event = await loop.run_in_executor(
                None, resolve_target_markets, self.series)
            self._print_universe(per_event)
            print("[dry-run] resolved only; not connecting")
            return 0

        while not self.stop.is_set():
            if deadline and time.time() >= deadline:
                break

            if not self._gate_open():
                # Idle in-process rather than exiting: systemd would restart a
                # clean exit every RestartSec, and staying up means enabling
                # capture is a config flip plus a restart.
                if not gate_logged:
                    print(f"[gate] capture closed: {config.capture_gate_reason()}"
                          f" -- idle, no subscription and no segments written",
                          flush=True)
                    gate_logged = True
                if await self._sleep(config.CAPTURE_GATE_POLL_SECONDS):
                    break
                continue
            if gate_logged:
                print("[gate] capture open -- connecting", flush=True)
                gate_logged = False

            # Resolve on the first open tick and once per ET day thereafter. Held
            # until the gate opens so a closed gate costs no REST quota, which the
            # backfill is competing for.
            today = str(config.today_et_date())
            if resolved_day != today:
                self.series, _extra = await loop.run_in_executor(
                    None, discover_close_series)
                self.markets, per_event = await loop.run_in_executor(
                    None, resolve_target_markets, self.series)
                self._print_universe(per_event)
                resolved_day = today
                self.capture_day = today
            if not self.markets:
                # Was `return 1`. Under the gate an empty universe is a transient
                # state -- the day's event is often still `unopened` at 09:30 --
                # so retry instead of handing systemd a crash loop.
                print("no markets resolved -- nothing to record yet",
                      file=sys.stderr, flush=True)
                resolved_day = None
                if await self._sleep(config.CAPTURE_GATE_POLL_SECONDS):
                    break
                continue

            if duration and deadline is None:
                deadline = time.time() + duration

            try:
                hdrs = k._auth_headers("GET", path)
                hdrs.pop("Content-Type", None)
                async with connect(url, additional_headers=hdrs,
                                   open_timeout=15, ping_interval=20,
                                   ping_timeout=30, max_size=None) as ws:
                    backoff = RECONNECT_BASE
                    self._new_connection()
                    self._open_segment()
                    await self._subscribe(ws, self.markets)
                    tasks = [asyncio.create_task(self.reader(ws)),
                             asyncio.create_task(self.writer(ws)),
                             asyncio.create_task(self.resolver(ws)),
                             asyncio.create_task(self._gate_watch())]
                    if deadline:
                        tasks.append(asyncio.create_task(
                            self._deadline(deadline)))
                    writer_task = tasks[1]
                    done = ()
                    try:
                        # FIRST_COMPLETED, not gather: when the deadline fires or
                        # any task exits, the others are cancelled at once rather
                        # than the process waiting out their timers.
                        done, _pending = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        # ORDER MATTERS. Stop taking NEW frames first, let the
                        # writer persist the ones already accepted, and only then
                        # cancel it. Cancelling all five together -- what this did
                        # until 2026-08-24 -- threw the queue away at every close.
                        others = [t for t in tasks if t is not writer_task]
                        for t in others:
                            t.cancel()
                        await asyncio.gather(*others, return_exceptions=True)
                        await self._drain_queue()
                        writer_task.cancel()
                        await asyncio.gather(writer_task, return_exceptions=True)
                        for s in list(self.epoch):
                            self._close_epoch(s, "ws_close")
                        self._close_segment()

                    # A task that ended in an EXCEPTION is a dropped connection,
                    # not a clean shutdown, and it has to reach the handler below.
                    #
                    # `ws.recv()` raising ConnectionClosed propagates out of the
                    # reader, but `gather(..., return_exceptions=True)` in the
                    # teardown consumed it, so the `async with` exited normally
                    # and the outer `while` reconnected on the spot: no
                    # `reconnects` increment, no backoff, and a server refusing
                    # the socket became a hot loop that reported a clean run. The
                    # deadline and a `stop`-driven exit leave no exception here,
                    # so they still fall through and end the run.
                    _exc = next((t.exception() for t in done
                                 if not t.cancelled() and t.exception() is not None),
                                None)
                    if _exc is not None:
                        raise _exc
            except Exception as e:                   # noqa: BLE001
                self.stats["reconnects"] += 1
                for s in list(self.epoch):
                    self._close_epoch(s, "ws_close")
                self._close_segment()
                print(f"[reconnect] {type(e).__name__}: {str(e)[:160]} "
                      f"-- retry in {backoff:.0f}s", file=sys.stderr, flush=True)
                if self.stop.is_set() or (deadline and time.time() >= deadline):
                    break
                await asyncio.sleep(backoff)
                backoff = min(RECONNECT_MAX, backoff * 2)

        print(f"[done] {json.dumps(self.stats)}", flush=True)
        return 0

    async def _drain_queue(self, timeout=DRAIN_TIMEOUT_S):
        """Let the writer finish the queue, then EMPTY whatever is left. Returns frames lost.

        The emptying is as important as the waiting. `self.queue` outlives one connection, so a
        residue left behind is picked up by the next connection's writer and written into the
        next segment under the next session id -- frames that arrived before a reconnect,
        recorded as if they arrived after it, with a `sid`/`seq` pair belonging to a subscription
        that no longer exists. A replay then applies them to the wrong ladder. Losing them
        loudly, with a count in `stats`, is strictly better than keeping them in the wrong place.
        """
        deadline = time.time() + timeout
        while not self.queue.empty() and time.time() < deadline:
            await asyncio.sleep(0.05)
        left = self.queue.qsize()
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if left:
            self.stats["dropped_on_close"] += left
            print(f"[drain] {left} frame(s) still queued after {timeout:.0f}s -- discarded "
                  f"rather than carried into the next segment", file=sys.stderr, flush=True)
        return left

    async def _deadline(self, when):
        await asyncio.sleep(max(0, when - time.time()))
        self.stop.set()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=None,
                    help="seconds to record, then exit (calibration runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the universe and exit without connecting")
    ap.add_argument("--ignore-gate", action="store_true",
                    help="capture regardless of CAPTURE_SESSION_ENABLED and the "
                         "ET session window (one-off test runs)")
    a = ap.parse_args(argv)

    # Take a NAMED SHARE of the account's read budget, not the whole thing. The WebSocket stream
    # itself costs nothing -- subscribe once, receive forever -- but the 5-minute market resolver
    # (`_resolve`, via `k.paged`) is REST, and Kalshi meters reads per API KEY while
    # `client._RATE_PER_SEC` is per PROCESS. This module never called `set_read_budget` at all, so
    # it sat at the conservative 8 req/s default and contended with a quote loop that had raised
    # itself to 20. Two contenders, one bucket, and the 429s land on whichever asks next.
    #
    # A share rather than a `pgrep` guard because this process is `Restart=always` and is meant to
    # be up for the whole session: a guard naming it would make the 09:33 quote run skip on
    # exactly the day capture is turned back on. See config.READ_SHARE_CAPTURE.
    reads = k.set_read_budget(share=config.READ_SHARE_CAPTURE)
    print(f"[budget] read {reads['req_per_s_before']:.1f} -> {reads['req_per_s_after']:.1f} req/s "
          f"(share {reads['share']:.0%}), tiers seen {reads['tiers_seen'] or 'none'}", flush=True)

    rec = Recorder(dry_run=a.dry_run, ignore_gate=a.ignore_gate)

    async def _run():
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(s, rec.stop.set)
        return await rec.run(duration=a.duration)

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
