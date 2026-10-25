"""The quoter's tape: it writes what the recorder writes, and it never blocks the pump."""
import gzip
import json
import shutil
import tempfile
import time
from pathlib import Path

import config

_TMP = Path(tempfile.mkdtemp(prefix="feedtee-"))
_REAL_WS_DIR = config.WS_DIR
config.WS_DIR = _TMP                     # before the writer resolves any path

from storage import book_replay, feed_tee, paths   # noqa: E402
from storage import s3util   # noqa: E402

# Offline: the tape under test is local, so the S3 half of replay sees an empty bucket.
s3util.list_keys = lambda bucket, prefix: []

FAILS = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got={got} want={want}")
    if not ok:
        FAILS.append(label)


def assert_true(label, cond):
    check(label, bool(cond), True)


def _frames(n, sid=1, start_seq=1):
    """`n` orderbook_delta frames on one market, sequential."""
    return [{"type": "orderbook_delta", "sid": sid, "seq": start_seq + i,
             "msg": {"market_ticker": "KXINX-TEST-B1", "price": 50, "delta": 1,
                     "side": "yes"}}
            for i in range(n)]


def _drain(tee, timeout=5.0):
    """Wait for the writer thread to catch up."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and tee._q.qsize():
        time.sleep(0.02)


print("the tape lands where the recorder's does, in its own channel")
dt = "2026-08-26"
tee = feed_tee.FeedTee(dt=dt, subscribe_state={"n_markets": 3}).start()
assert_true("the writer came up", tee.ok)
check("its own channel, never 'all'", tee.channel, "quoter")
for i, f in enumerate(_frames(50)):
    tee.offer(1_000_000_000 + i, f)
_drain(tee)
tee.stop()

seg_dir = config.WS_DIR / f"dt={dt}" / "channel=quoter"
segs = sorted(seg_dir.glob("seg-*.jsonl.gz"))
check("one segment written", len(segs), 1)
check("  and it is named the way the recorder names one",
      bool(paths.parse_segment_name(segs[0].name)), True)
meta_p = paths.segment_meta_path(segs[0])
check("  with a sidecar manifest beside it", meta_p.exists(), True)
meta = json.loads(meta_p.read_text())
check("  header + 50 frames", meta["line_count"], 51)
check("  nothing dropped", meta["dropped_frames"], 0)
check("  and the manifest's byte count matches the disk",
      "byte_count_mismatch" not in meta, True)

print("\nevery line is the recorder's record shape")
lines = [json.loads(x) for x in gzip.open(segs[0], "rt")]
check("the first line is a segment header", lines[0]["kind"], "segment_header")
body = lines[1:]
check("every other line carries a frame", all("frame" in r for r in body), True)
check("  and our own receipt stamp, not the exchange's",
      all(isinstance(r["recv_ns"], int) for r in body), True)
check("  the frame is the parsed message, untouched",
      body[0]["frame"]["msg"]["market_ticker"], "KXINX-TEST-B1")
check("  the writer identifies itself, so a mixed directory is diagnosable",
      lines[0]["writer"], "run_04_quote_loop")

print("\nthe sid/seq accounting the manifest is trusted for")
sids = meta["sids"]
check("one sid tracked", len(sids), 1)
only = next(iter(sids.values()))
check("  first seq", only["first_seq"], 1)
check("  last seq", only["last_seq"], 50)
check("  all counted as deltas", only["n_delta"], 50)

print("\nbook_replay reads it, and ONLY when asked for this channel")
res = book_replay.replay(dt, channel="quoter")
n = res.pop("_frames")
check("the quoter channel replays the frames", n, 50)
res_all = book_replay.replay(dt, channel="all")
check("  and 'all' sees nothing -- the two tapes never merge",
      res_all.pop("_frames"), 0)

print("\noffer NEVER blocks: a full queue drops and counts, it does not wait")
# No writer thread at all, so nothing drains: the queue fills and every further offer is refused
# immediately. This is the property that protects the pump -- the socket that prices the book we
# quote must never wait on a disk writer.
stuck = feed_tee.FeedTee(dt=dt, queue_max=4)     # deliberately NOT started
t0 = time.monotonic()
taken = sum(1 for i in range(200) if stuck.offer(i, {"type": "x"}))
elapsed = time.monotonic() - t0
check("only the queue's worth was taken", taken, 4)
check("  the rest were dropped and counted", stuck.dropped, 196)
assert_true("  and 200 offers took under a tenth of a second", elapsed < 0.1)

print("\na drop is recorded IN the tape, so a replay knows the day is incomplete")
tee2 = feed_tee.FeedTee(dt=dt, queue_max=2).start()
tee2.dropped = 7                                  # as if the pump had outrun the writer
tee2._last_drop_marker = 0.0
tee2.offer(1, _frames(1)[0])
_drain(tee2)
tee2.stop()
segs2 = sorted((config.WS_DIR / f"dt={dt}" / "channel=quoter").glob("seg-*.jsonl.gz"))
newest = max(segs2, key=lambda p: p.stat().st_mtime)
kinds = [json.loads(x).get("kind") for x in gzip.open(newest, "rt")]
assert_true("a drop_marker was written", "drop_marker" in kinds)
m2 = json.loads(paths.segment_meta_path(newest).read_text())
check("  and the manifest carries the count", m2["dropped_frames"], 7)

print("\nstop() is safe to call twice, and on a writer that never started")
tee2.stop()
never = feed_tee.FeedTee(dt=dt)
never.stop()
assert_true("no writer, no crash", never.error is None)

config.WS_DIR = _REAL_WS_DIR
shutil.rmtree(_TMP, ignore_errors=True)

print("\nFAIL" if FAILS else "\nPASS")
raise SystemExit(1 if FAILS else 0)
