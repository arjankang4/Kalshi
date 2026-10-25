"""Tee the quote loop's own Kalshi socket to the rolling capture segments.

WHY THIS EXISTS. `storage/ws_recorder` is the general capture path and it has been gated off
since 2026-08-22 (`config.CAPTURE_SESSION_ENABLED = False`), so the last order-book tape in S3 is
2026-08-21 -- and order book data has no REST backfill path, so a day not captured is a day gone.
That includes both live trading sessions. We can reconstruct from `/portfolio/fills` and the
decision journal exactly what we quoted and what filled; we cannot reconstruct the BOOK at the
moment of a fill, which is the markout, which is the whole question a market maker asks about its
own trades.

WHY IT TEES RATHER THAN CAPTURES. `run_04` already holds an authenticated Kalshi socket
subscribed to `orderbook_delta` over every market it quotes. Those frames are already arriving and
already free -- Kalshi meters commands, not frames. Writing them costs one more consumer of a
stream we are reading anyway: no second connection, no second subscription, no extra tokens, and
by construction the tape covers exactly the markets we traded rather than a universe chosen
separately from the one we quoted.

WHAT IT DELIBERATELY IS NOT. It is not a replacement for `ws_recorder`. It sees only what the
quoter subscribed to, it starts and stops with the quoter, and it carries none of the recorder's
epoch machinery -- gap tracking lives in `MarketFeed` and is reported through `health()`, not
written into the tape. `book_replay` reads `rec["frame"]` and ignores records without one, so the
absence of epoch records costs a reader the ability to tell a gap-crossed book from a clean one.
Read the manifest's `dropped_frames` and the feed's own gap count before trusting a replayed book.

ITS OWN CHANNEL, AND THAT IS LOAD-BEARING. Segments go to `dt=<et>/channel=quoter/`, never
`channel=all`. If both writers shared a directory a replay would apply the same delta twice --
same markets, same exchange, two sockets -- and a book built from duplicated deltas is wrong
without ever looking wrong. `flush_to_s3` globs `dt=*/*/*.jsonl.gz`, so the existing flush,
promote and seal timers pick this up with no change; `book_replay` takes a `channel` argument.

NEVER BLOCK THE PUMP. `offer` puts on a bounded queue and returns immediately; a writer thread
drains it. If the queue is full the frame is DROPPED and counted, because the alternative is
back-pressuring the socket that prices the book we are quoting into. A dropped frame costs a
research question; a stalled pump costs money.
"""
import json
import queue
import threading
import time
import uuid

import config
from storage import segments

# Frames the writer may fall behind by before it starts dropping. At the measured four-board rate
# (~30k frames/minute, i.e. ~500/s) this is about twenty seconds of slack, which is far more than
# a gzip flush needs and small enough that a genuinely stuck writer is bounded rather than eating
# the heap.
QUEUE_MAX = 10000

# One drop marker per contiguous run of drops, not one per dropped frame: a marker per frame would
# turn the failure into its own flood, and the count is what a reader needs.
DROP_MARKER_MIN_INTERVAL_S = 5.0


class FeedTee:
    """Rolling segment writer fed from a `MarketFeed`'s raw frames."""

    def __init__(self, dt=None, channel=None, subscribe_state=None, queue_max=QUEUE_MAX):
        self.dt = str(dt or config.today_et_date())
        self.channel = channel or getattr(config, "QUOTE_TEE_CHANNEL", "quoter")
        self.session_id = uuid.uuid4().hex[:8]
        self.subscribe_state = subscribe_state or {}
        self._q = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._thread = None
        self.segment = None
        self.seq_no = 0
        self.frames = 0
        self.dropped = 0
        self.segments_closed = 0
        self.bytes_written = 0
        self.error = None
        self._last_drop_marker = 0.0

    # -- lifecycle --

    def start(self):
        """Open the first segment and begin draining. Returns self; never raises."""
        try:
            self._open_segment()
        except Exception as e:                               # noqa: BLE001
            self.error = f"open: {type(e).__name__}: {str(e)[:100]}"
            return self
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        return self

    @property
    def ok(self):
        return bool(self._thread and self._thread.is_alive() and self.error is None)

    def stop(self):
        """Drain what is queued, close the segment and write its manifest."""
        self._stop.set()
        if self._thread:
            # Long enough to finish a segment close (a sha256 over the plaintext) but bounded --
            # the exit path also has orders to cancel and must not wait on a writer.
            self._thread.join(timeout=15)
        self._close_segment()
        return self

    # -- the hot side, called from the socket pump --

    def offer(self, recv_ns, frame):
        """Queue one parsed frame. Never blocks, never raises, returns whether it was taken."""
        try:
            self._q.put_nowait((recv_ns, frame))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    # -- the writer thread --

    def _drain(self):
        while not (self._stop.is_set() and self._q.empty()):
            try:
                recv_ns, frame = self._q.get(timeout=0.5)
            except queue.Empty:
                self._maybe_roll()
                continue
            try:
                self._write(recv_ns, frame)
            except Exception as e:                           # noqa: BLE001
                # A write failure must not take the session down and must not spin: record it,
                # stop writing, and let the loop keep quoting without a tape.
                self.error = f"write: {type(e).__name__}: {str(e)[:100]}"
                return
            self._maybe_roll()

    def _write(self, recv_ns, frame):
        if self.segment is None:
            return
        if self.dropped and (time.monotonic() - self._last_drop_marker) > DROP_MARKER_MIN_INTERVAL_S:
            self.segment.write({"v": 1, "kind": "drop_marker", "session": self.session_id,
                                "recv_ns": time.time_ns(), "dropped_total": self.dropped})
            self.segment.dropped = self.dropped
            self._last_drop_marker = time.monotonic()
        self.segment.write({"v": 1, "session": self.session_id,
                            "recv_ns": recv_ns, "frame": frame})
        self.frames += 1
        # The same sid/seq accounting the recorder keeps, so the manifest can be read the same
        # way. `type` is the frame's own; a control frame carries no sid and `note_frame` says so.
        msg = frame if isinstance(frame, dict) else {}
        self.segment.note_frame(recv_ns, msg.get("sid"), msg.get("seq"), msg.get("type"))
        self.segment.sync()

    def _maybe_roll(self):
        if self.segment is not None and self.segment.should_roll():
            self._close_segment()
            self._open_segment()

    def _open_segment(self):
        self.segment = segments.Segment(self.dt, self.channel, self.session_id,
                                        self.seq_no, self.subscribe_state)
        self.segment.write({"v": 1, "session": self.session_id, "recv_ns": time.time_ns(),
                            "kind": "segment_header", "sub": self.subscribe_state,
                            "dt": self.dt, "writer": "run_04_quote_loop"})
        self.seq_no += 1

    def _close_segment(self):
        if self.segment is None:
            return None
        try:
            self.segment.dropped = self.dropped
            meta = self.segment.close()
            self.segments_closed += 1
            self.bytes_written += meta.get("compressed_bytes") or 0
            return meta
        except Exception as e:                               # noqa: BLE001
            self.error = self.error or f"close: {type(e).__name__}: {str(e)[:100]}"
            return None
        finally:
            self.segment = None

    def detail(self):
        return {"ok": self.ok, "dt": self.dt, "channel": self.channel,
                "session": self.session_id, "frames": self.frames,
                "dropped": self.dropped, "queued": self._q.qsize(),
                "segments": self.segments_closed, "gz_bytes": self.bytes_written,
                "error": self.error}
