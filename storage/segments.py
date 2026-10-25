"""Gzipped rolling segment writer.

Bytes go straight from the socket into a GzipFile over a buffered handle. No
frame is ever retained after it is written, so the writer's footprint is its
buffers, not the day.
"""
import gzip
import json
import time
import zlib

import config
from storage import paths, s3util

# How often buffered frames are pushed to disk. Bounds crash loss to seconds
# rather than to the segment roll interval.
SYNC_SECONDS = 20


class Segment:
    """One rolling capture file plus the accounting needed to trust it later."""

    def __init__(self, dt, channel, session_id, seq_no, subscribe_state):
        self.dt = dt
        self.channel = channel
        self.session_id = session_id
        self.seq_no = seq_no
        self.path = paths.segment_path(dt, channel, session_id, seq_no)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._raw = open(self.path, "wb", buffering=1 << 20)
        # mtime=0 keeps the gzip container byte-stable for identical input; the
        # manifest hashes the plaintext regardless, but there is no reason to
        # introduce avoidable nondeterminism.
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="wb", compresslevel=6, mtime=0)

        self.opened_at = time.time()
        self.lines = 0
        self.plain_bytes = 0
        self.dropped = 0
        self.first_recv_ns = None
        self.last_recv_ns = None
        # {sid: {"first_seq","last_seq","n_delta","n_snapshot","n_control"}}
        self.sids = {}
        self.epochs_opened = []
        self.epochs_closed = []
        self.subscribe_state = subscribe_state
        self.closed = False
        self.last_sync = self.opened_at

    def write(self, obj):
        line = json.dumps(obj, separators=(",", ":")).encode() + b"\n"
        self._gz.write(line)
        self.lines += 1
        self.plain_bytes += len(line)

    def sync(self, now=None):
        """Push buffered frames to disk.

        Without this, gzip's internal buffer plus a 1 MB file buffer can hold
        everything in memory for the whole 15-minute segment on a quiet market:
        a crash would lose frames the writer had already accepted, and the
        partial file is unreadable for monitoring. Z_SYNC_FLUSH ends the current
        deflate block so the bytes on disk are a valid gzip stream; the cost is
        a slightly worse ratio at each flush point, which at this cadence is noise.
        """
        now = now or time.time()
        if self.closed or now - self.last_sync < SYNC_SECONDS:
            return False
        self._gz.flush(zlib.Z_SYNC_FLUSH)
        self._raw.flush()
        self.last_sync = now
        return True

    def note_frame(self, recv_ns, sid, seq, ftype):
        if self.first_recv_ns is None:
            self.first_recv_ns = recv_ns
        self.last_recv_ns = recv_ns
        if sid is None:
            return
        # Keyed by session too: `sid` is only unique within one WebSocket
        # connection, so after a reconnect session B reuses sid=1 and a
        # sid-only key merges two unrelated streams into one phantom gap.
        s = self.sids.setdefault(
            f"{self.session_id}:{sid}",
            {"first_seq": seq, "last_seq": seq, "n_delta": 0, "n_snapshot": 0, "n_control": 0},
        )
        if seq is not None:
            if s["first_seq"] is None or seq < s["first_seq"]:
                s["first_seq"] = seq
            if s["last_seq"] is None or seq > s["last_seq"]:
                s["last_seq"] = seq
        if ftype == "orderbook_delta":
            s["n_delta"] += 1
        elif ftype == "orderbook_snapshot":
            s["n_snapshot"] += 1
        else:
            s["n_control"] += 1

    def should_roll(self, now=None):
        now = now or time.time()
        if self._raw.tell() >= config.SEGMENT_ROLL_BYTES:
            return True
        if now - self.opened_at < config.SEGMENT_ROLL_SECONDS:
            return False
        # Don't roll a segment that holds nothing but its own header. Overnight
        # these markets are silent, and rolling on the timer regardless would
        # emit ~90 empty objects a night for no recoverable content.
        return self.lines > 1

    def close(self):
        """Close and write the sidecar manifest. Returns the manifest dict."""
        if self.closed:
            return self.meta
        self._gz.close()
        self._raw.close()
        self.closed = True

        plaintext_sha256, plain_bytes = s3util.sha256_gz_plaintext(self.path)
        self.meta = {
            "segment_id": f"{self.session_id}-{self.seq_no:06d}",
            "capture_session_id": self.session_id,
            "capture_date_et": self.dt,
            "channel": self.channel,
            "line_count": self.lines,
            "uncompressed_bytes": plain_bytes,
            "compressed_bytes": self.path.stat().st_size,
            "plaintext_sha256": plaintext_sha256,
            "first_recv_ns": self.first_recv_ns,
            "last_recv_ns": self.last_recv_ns,
            "sids": self.sids,
            "dropped_frames": self.dropped,
            "subscribe_state": self.subscribe_state,
            "opened_at": self.opened_at,
            "closed_at": time.time(),
            "epochs_opened": self.epochs_opened,
            "epochs_closed": self.epochs_closed,
        }
        # Guard against a silently truncated write: the accounting the promotion
        # job trusts must agree with what is actually on disk.
        if plain_bytes != self.plain_bytes:
            self.meta["byte_count_mismatch"] = {
                "counted": self.plain_bytes, "on_disk": plain_bytes}
        paths.segment_meta_path(self.path).write_text(json.dumps(self.meta, indent=1))
        return self.meta
