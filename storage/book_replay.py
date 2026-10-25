"""Rebuild a Kalshi order book from the captured tape, and say whether it is trustworthy.

Kalshi sends a full `orderbook_snapshot` per market when you subscribe -- and then nothing but
deltas. Measured on 2026-08-20: 1,504 snapshots against 1,052,753 deltas, and every snapshot sits
in a subscribe burst. So the snapshot alone answers "what was the book at 09:30"; any other
instant has to be replayed. That is what this module is for.

**Validity is not optional.** A reconstructed book is meaningful only if

  1. a snapshot for that market was observed at or before the target time, and
  2. no sequence gap occurred on that subscription between the snapshot and the target.

`seq` is per-subscription, not per-market, so a gap anywhere on the stream invalidates every
market's book until the next snapshot. When either condition fails the answer is **unknown**,
never an empty or stale book -- that distinction is the whole point of the epoch machinery in
ws_recorder, and it is preserved here.

Units follow the house convention: prices int64 micro-dollars, sizes int64 centi-contracts.

    python -m storage.book_replay --dt 2026-08-20 --ticker KXINX-26AUG20H1600-B7662 --at 15:59
    python -m storage.book_replay --dt 2026-08-20 --at 15:59 --all
"""
import argparse
import gzip
import io
import json
import sys
from datetime import datetime, time as dtime, timezone

import config
from storage import paths, s3util
from storage.units import to_cc, to_micros

# Frame types that consume a sequence number on the orderbook_delta subscription. `ok` and
# `unsubscribed` are control responses but still advance `seq`, so counting only data frames
# reports a phantom gap on every one of them.
SEQ_BEARING = {"orderbook_delta", "orderbook_snapshot", "ok", "unsubscribed"}


class Book:
    """One market's resting interest. Both ladders are BID stacks in their own side's terms:
    a `no` level at price p is a YES offer at 1 - p."""

    __slots__ = ("yes", "no", "snapshot_ns", "snapshot_seq", "n_deltas")

    def __init__(self):
        self.yes, self.no = {}, {}
        self.snapshot_ns = self.snapshot_seq = None
        self.n_deltas = 0

    def load(self, msg, recv_ns, seq):
        self.yes, self.no = {}, {}
        for side, key in (("yes", "yes_dollars_fp"), ("no", "no_dollars_fp")):
            # Absent means the side is genuinely empty -- WS omits the key entirely rather
            # than sending []. Treating absent as "unknown" would be wrong.
            for px, sz in (msg.get(key) or []):
                getattr(self, side)[to_micros(px)] = to_cc(sz)
        self.snapshot_ns, self.snapshot_seq, self.n_deltas = recv_ns, seq, 0

    def apply(self, msg):
        side = getattr(self, msg["side"])
        p = to_micros(msg["price_dollars"])
        new = side.get(p, 0) + to_cc(msg["delta_fp"])
        if new < 0:
            # Never clamp. A negative resting size is arithmetic proof of a frame we did not
            # see -- i.e. the gap detector missed one -- and silently flooring it at zero would
            # turn a known-bad book into a plausible-looking one.
            raise ValueError(f"negative size {new} at {p} on {msg['side']} "
                             f"({msg.get('market_ticker')}) -- a delta was missed")
        if new == 0:
            side.pop(p, None)
        else:
            side[p] = new
        self.n_deltas += 1

    def best(self):
        """(best_bid, best_ask) in YES micro-dollars. The `no` ladder inverts."""
        bb = max(self.yes) if self.yes else None
        ba = (1_000_000 - max(self.no)) if self.no else None
        return bb, ba

    def as_dict(self):
        return {"yes": dict(sorted(self.yes.items())), "no": dict(sorted(self.no.items()))}


def _segments(dt, local_first=True, channel=None):
    """Segments for an ET day in CAPTURE-TIME order. Local disk first, then S3.

    Ordered by time, never by name. A segment is `seg-<session8>-<nnnnnn>` and the session id is
    random hex, so sorting by filename interleaves sessions arbitrarily -- a test capture taken
    at 20:24 sorted ahead of the real 17:56 session and the replay stopped on its first frame,
    reporting zero books. Deltas are only meaningful applied in the order they were observed.
    """
    d = str(dt)
    # `channel` because there are now two writers with two universes: `ws_recorder` writes
    # `channel=all` and `run_04`'s tee writes `channel=quoter` (see `storage/feed_tee`). They must
    # never be replayed together -- the same market from two sockets means the same delta applied
    # twice, and a book built that way is wrong without looking wrong.
    ch = channel or "all"
    ddir = config.WS_DIR / f"dt={d}" / f"channel={ch}"
    local = sorted(ddir.glob("seg-*.jsonl.gz"), key=lambda p: p.stat().st_mtime) \
        if local_first and ddir.exists() else []
    if local:
        return [("local", p) for p in local]
    for bucket, prefix in paths.raw_day_locations(d, ch):
        rows = [r for r in s3util.list_keys(bucket, prefix) if r[0].endswith(".jsonl.gz")]
        if rows:
            return [("s3", (bucket, k)) for k in _ordered_keys(rows, bucket)]
    return []


def _ordered_keys(rows, bucket):
    """Segment keys in capture-time order, from the sidecars rather than S3 metadata.

    LastModified is NOT capture time: any copy rewrites it. The us-east-1 -> us-east-2
    migration re-uploaded every sealed day in one burst, collapsing 59 segments of
    2026-08-20 into 5 distinct timestamps four seconds apart, at which point the sort
    degenerated to filename order -- which put a segment captured 16 hours later first
    and made the replay stop on its first out-of-range frame, reporting zero books.

    `first_recv_ns` in the `.meta.json` sidecar is stamped at capture and survives being
    copied. Sessions are ordered by their earliest frame; within a session the segment
    sequence number is authoritative by construction, which also places the control-only
    stubs that carry no `first_recv_ns` at all.
    """
    sess_first, seg_meta = {}, {}
    for key, _, last_mod in rows:
        m = paths.SEG_RE.match(key.split("/")[-1])
        if not m:
            continue
        sess, n = m.group("sess"), int(m.group("n"))
        first = None
        try:
            doc = json.loads(s3util.open_stream(bucket, key + ".meta.json").read())
            first = doc.get("first_recv_ns")
        except Exception:                              # noqa: BLE001
            pass                                       # no sidecar -> fall back below
        seg_meta[key] = (sess, n)
        if first is not None:
            prev = sess_first.get(sess)
            sess_first[sess] = first if prev is None else min(prev, first)
        # last_mod is the last resort, and only for a session with no usable sidecar
        sess_first.setdefault(sess, None)
        if sess_first[sess] is None:
            sess_first[sess] = int(last_mod.timestamp() * 1e9)

    return sorted(seg_meta, key=lambda k: (sess_first[seg_meta[k][0]], seg_meta[k][1]))


def _lines(kind, ref):
    raw = (ref.read_bytes() if kind == "local"
           else s3util.open_stream(*ref).read())      # ref is (bucket, key) for s3
    for line in gzip.GzipFile(fileobj=io.BytesIO(raw)):
        yield line


def replay(dt, until_ns=None, tickers=None, verbose=False, channel=None):
    """Replay the day's tape. Returns {ticker: (Book, reason)} where reason is None if valid.

    One pass over the tape serves every market at once -- the deltas are interleaved, so
    filtering to a single ticker still costs a full scan.

    `channel` selects the WRITER, and it matters because there are two. `None`/"all" is
    `ws_recorder`'s capture; "quoter" is the tape `run_04` tees off its own socket
    (`storage/feed_tee`), which covers exactly the markets that session quoted and nothing else.
    Never replay both together: the same market from two sockets is the same delta twice.
    """
    want = set(tickers) if tickers else None
    books, invalid = {}, {}
    last_seq, gap_after = {}, {}          # per sid
    n_frames = 0

    for kind, ref in _segments(dt, channel=channel):
        for line in _lines(kind, ref):
            rec = json.loads(line)
            f = rec.get("frame")
            if not f:
                continue
            ftype, sid, seq = f.get("type"), f.get("sid"), f.get("seq")
            recv_ns = rec.get("recv_ns")
            if until_ns is not None and recv_ns and recv_ns > until_ns:
                return _finish(books, invalid, want, n_frames)

            # Contiguity is a property of the SUBSCRIPTION, so track it before filtering by
            # ticker: a gap on this sid invalidates every market riding on it.
            if seq is not None and sid is not None and ftype in SEQ_BEARING:
                prev = last_seq.get(sid)
                if prev is not None and seq != prev + 1:
                    gap_after[sid] = recv_ns
                last_seq[sid] = seq

            if ftype not in ("orderbook_snapshot", "orderbook_delta"):
                continue
            msg = f.get("msg") or {}
            tk = msg.get("market_ticker")
            if not tk or (want and tk not in want):
                continue
            n_frames += 1
            b = books.setdefault(tk, Book())
            try:
                if ftype == "orderbook_snapshot":
                    b.load(msg, recv_ns, seq)
                    # A fresh snapshot re-anchors this market: any earlier gap is now moot.
                    invalid.pop(tk, None)
                    gap_after.pop(sid, None)
                else:
                    if b.snapshot_ns is None:
                        invalid.setdefault(tk, "no_snapshot_before_target")
                        continue
                    if sid in gap_after:
                        invalid.setdefault(tk, "seq_gap_since_snapshot")
                    b.apply(msg)
            except ValueError as e:
                invalid[tk] = f"negative_size: {e}"
    return _finish(books, invalid, want, n_frames)


def _finish(books, invalid, want, n_frames):
    out = {}
    for tk, b in books.items():
        reason = invalid.get(tk)
        if reason is None and b.snapshot_ns is None:
            reason = "no_snapshot_before_target"
        out[tk] = (b, reason)
    for tk in (want or ()):
        out.setdefault(tk, (Book(), "never_observed"))
    out["_frames"] = n_frames
    return out


def verify(dt, at="15:30", limit=None, channel=None):
    """Cross-check the replayed book against Kalshi's own top-of-book from the `ticker` channel.

    Compared at the TICKER FRAME'S OWN timestamp, not at a common cutoff. `ticker` is a sampled
    channel with no sequence number, so at any fixed wall time it lags the delta stream -- on
    2026-08-20 the last ticker before 15:30 was 831 ms stale, and comparing there showed 9 of 522
    markets off by a cent. Replayed to each frame's own instant, all of them matched exactly.
    A mismatch here is a real reconstruction bug; a mismatch at a fixed cutoff is just sampling.
    """
    until = at_ns(dt, at)
    ticks = {}
    for kind, ref in _segments(dt, channel=channel):
        stop = False
        for line in _lines(kind, ref):
            r = json.loads(line)
            f = r.get("frame")
            if not f or f.get("type") != "ticker":
                continue
            if r["recv_ns"] > until:
                stop = True
                break
            m = f.get("msg") or {}
            if m.get("market_ticker"):
                ticks[m["market_ticker"]] = (m, r["recv_ns"])
        if stop:
            break

    items = sorted(ticks.items())[:limit] if limit else sorted(ticks.items())
    agree = dis = 0
    bad = []
    for tk, (m, ns) in items:
        res = replay(dt, until_ns=ns, tickers=[tk])
        res.pop("_frames", None)
        b, reason = res[tk]
        if reason is not None:
            continue
        bb, ba = b.best()
        rb = None if bb is None else round(bb / 1e6, 2)
        ra = None if ba is None else round(ba / 1e6, 2)
        # Kalshi spells an empty side as 0.00 bid / 1.00 ask; the replay spells it None.
        tb = None if float(m["yes_bid_dollars"]) == 0.0 else round(float(m["yes_bid_dollars"]), 2)
        ta = None if float(m["yes_ask_dollars"]) == 1.0 else round(float(m["yes_ask_dollars"]), 2)
        if (rb, ra) == (tb, ta):
            agree += 1
        else:
            dis += 1
            bad.append((tk, (rb, ra), (tb, ta)))
    return agree, dis, bad


def at_ns(dt, hhmm):
    """'15:59' on an ET date -> epoch nanoseconds."""
    h, m = (int(x) for x in hhmm.split(":"))
    return int(datetime.combine(dt, dtime(h, m), tzinfo=config.ET)
               .astimezone(timezone.utc).timestamp() * 1_000_000_000)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", required=True)
    ap.add_argument("--at", default="15:59", help="ET wall time, e.g. 15:59")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--all", action="store_true", help="report completeness across every market")
    ap.add_argument("--verify", type=int, nargs="?", const=25, default=None,
                    metavar="N", help="cross-check N markets against the ticker channel")
    ap.add_argument("--channel", default="all",
                    help="which writer's tape: 'all' = storage/ws_recorder, "
                         "'quoter' = run_04's tee of its own socket. Never mix them -- the same "
                         "market from two sockets is the same delta applied twice.")
    a = ap.parse_args(argv)

    d = datetime.fromisoformat(a.dt).date()
    if a.verify is not None:
        agree, dis, bad = verify(d, a.at, limit=a.verify, channel=a.channel)
        print(f"[verify] dt={d} at={a.at} ET -- replayed book vs Kalshi ticker top-of-book")
        print(f"[verify] agree={agree} disagree={dis}")
        for tk, r, t in bad[:5]:
            print(f"    MISMATCH {tk:<34} replay={r} kalshi={t}")
        return 0 if dis == 0 else 1
    until = at_ns(d, a.at)
    res = replay(d, until_ns=until, tickers=[a.ticker] if a.ticker else None,
                 channel=a.channel)
    n_frames = res.pop("_frames")

    ok = {t: b for t, (b, r) in res.items() if r is None}
    bad = {t: r for t, (b, r) in res.items() if r is not None}
    print(f"[replay] dt={d} at={a.at} ET  channel={a.channel}  "
          f"frames_applied={n_frames:,}")
    print(f"[replay] books COMPLETE: {len(ok)}   incomplete: {len(bad)}")
    if bad:
        import collections
        for reason, cnt in collections.Counter(bad.values()).most_common():
            print(f"    {cnt:>5} x {reason}")
    if a.ticker:
        b, r = res[a.ticker]
        bb, ba = b.best()
        print(f"\n{a.ticker}")
        print(f"  valid={r is None}{'' if r is None else '  reason=' + r}")
        print(f"  snapshot_seq={b.snapshot_seq}  deltas_applied={b.n_deltas}")
        print(f"  best_bid={None if bb is None else bb/1e6:.4f}" if bb is not None else "  best_bid=None")
        print(f"  best_ask={None if ba is None else ba/1e6:.4f}" if ba is not None else "  best_ask=None")
        print(f"  levels: yes={len(b.yes)} no={len(b.no)}")
    elif a.all and ok:
        s = sorted(ok.items())[:3]
        for t, b in s:
            bb, ba = b.best()
            print(f"    {t:<34} bid={'-' if bb is None else f'{bb/1e6:.2f}'} "
                  f"ask={'-' if ba is None else f'{ba/1e6:.2f}'} deltas={b.n_deltas}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
