"""Single source of truth for local paths and S3 keys.

Every other module derives its locations from here. Key layout is duplicated
nowhere else, so a change to the partition scheme is a one-file change.
"""
import re

import config

# --- local ---------------------------------------------------------------

def segment_dir(dt, channel):
    """Local directory holding one ET day's segments for one channel."""
    return config.WS_DIR / f"dt={dt}" / f"channel={channel}"


def segment_path(dt, channel, session_id, seq_no):
    return segment_dir(dt, channel) / f"seg-{session_id[:8]}-{seq_no:06d}.jsonl.gz"


# The inverse of `segment_path`'s filename, so a key can be parsed back into
# (session, sequence). It belongs here for the reason this module exists: the
# layout is defined in exactly one place.
#
# `storage/book_replay._ordered_keys` has referenced `paths.SEG_RE` since it was
# written and it was NEVER DEFINED -- so every S3-backed replay raised
# AttributeError on its first key. Local replay is the path that gets exercised
# day to day (`_segments` returns local segments before it ever looks at S3), so
# the break stayed invisible until a day had aged off local disk.
#
# `session_id` is `uuid.uuid4().hex` truncated to 8 (`ws_recorder`), hence 8 hex
# digits; the sequence is `%06d` but is not bounded to six, since a long capture
# day rolls past 999,999 segments in principle and a regex that silently stopped
# matching there would be the same class of bug.
SEG_RE = re.compile(r"^seg-(?P<sess>[0-9a-f]{8})-(?P<n>\d{6,})\.jsonl\.gz$")


def parse_segment_name(name):
    """`seg-<sess8>-<nnnnnn>.jsonl.gz` -> (session8, seq_no), or None."""
    m = SEG_RE.match(name)
    return (m.group("sess"), int(m.group("n"))) if m else None


def segment_meta_path(seg_path):
    """Sidecar manifest written beside a closed segment."""
    return seg_path.with_name(seg_path.name + ".meta.json")


def scratch_dir(job, dt):
    """Spill space for the external sort. Never /tmp -- that is a tmpfs."""
    return config.SCRATCH_DIR / job / f"dt={dt}"


# --- S3: short-term raw bucket -------------------------------------------

def ws_key(dt, channel, filename):
    return f"{config.S3_PREFIX_WS}/dt={dt}/channel={channel}/{filename}"


def ws_prefix(dt, channel=None):
    p = f"{config.S3_PREFIX_WS}/dt={dt}/"
    return p + f"channel={channel}/" if channel else p


def day_manifest_key(dt):
    """The 23:59 seal. Its presence is what makes a day promotable."""
    return f"{config.S3_PREFIX_WS}/_manifest/dt={dt}/manifest.json"


# --- S3: long-term parquet bucket ----------------------------------------

# Fact tables partition daily; low-volume dimensions partition monthly, because
# 365 files of 50 KB is worse than 12 files of 2 MB in every dimension -- and
# Glacier IR bills a 128 KB minimum per object.
# option_contracts is one small board per expiry (~400-1000 rows); daily would be
# 250 files well under the 128 KB Glacier IR minimum.
MONTHLY_TABLES = frozenset({"events", "series", "market_id_xref",
                            "option_contracts"})


def partition_kv(table, dt):
    """(key, value) for the table's partition. dt is a datetime.date."""
    if table in MONTHLY_TABLES:
        return "ym", f"{dt:%Y-%m}"
    return "dt", f"{dt:%Y-%m-%d}"


# `trades` is the one table two independent producers write: promote_ws from the
# WebSocket tape, rest_backfill from the REST tape. They shared a single key and
# silently destroyed each other -- promote overwrote unconditionally, and the
# backfill's already_done() saw promote's manifest and skipped the day entirely.
# A second partition key keeps both, and the `source` column already in the trades
# schema says which is which. One Athena query still reads across both.
SOURCED_TABLES = frozenset({"trades"})

# The closed set of producers for a sourced table. Glue partition projection uses an
# enum over this, so a value missing here is a partition Athena will never look at.
PRODUCERS = frozenset({"ws", "rest"})


def _src_element(table, src):
    """`src=<producer>/` for a table with more than one writer, else empty.

    Strict in both directions on purpose: omitting src for a sourced table is the
    exact mistake that cost a tape, and passing one for a single-writer table means
    the caller believes something about the layout that is not true.
    """
    if table in SOURCED_TABLES:
        if not src:
            raise ValueError(
                f"{table!r} is written by more than one producer; pass src= "
                f"(one of: ws, rest)")
        return f"src={src}/"
    if src:
        raise ValueError(f"{table!r} has a single producer; src= is not meaningful")
    return ""


def table_prefix(table):
    return f"{config.S3_PREFIX_PARQUET}/{table}/"


def partition_prefix(table, dt, src=None):
    k, v = partition_kv(table, dt)
    return f"{table_prefix(table)}{k}={v}/{_src_element(table, src)}"


def parquet_key(table, dt, part=0, src=None):
    return f"{partition_prefix(table, dt, src)}part-{part:05d}.snappy.parquet"


def tmp_parquet_key(table, dt, input_set_digest, src=None):
    """Written first, verified, then CopyObject'd to the final key."""
    return f"{partition_prefix(table, dt, src)}_tmp/{input_set_digest}.parquet"


def promotion_manifest_key(table, dt, src=None):
    k, v = partition_kv(table, dt)
    return (f"{config.S3_PREFIX_MANIFEST}/{table}/{k}={v}/"
            f"{_src_element(table, src)}manifest.json")


def seal_day_locations(dt):
    """(bucket, key) pairs that may hold a day's seal manifest, preferred first.

    The seal is the only record that a capture day was complete, and it was
    written to the short-term bucket alone -- whose `expireAfter7` lifecycle rule
    carries `Filter: {}`, so it reaps ws/_manifest/ exactly like the segments it
    describes. Every seal-scoped check treats a failed GET as "not a captured
    day", so at day 7 the audit trail would not break loudly, it would report a
    clean bill of health for days it could no longer see.

    Same shape and same key in both buckets, so the archived copy is a drop-in.
    """
    key = day_manifest_key(dt)
    return ((config.S3_BUCKET_WS, key), (config.S3_BUCKET_HIST, key))


def raw_day_locations(dt, channel=None):
    """(bucket, prefix) pairs holding a day's raw capture, most-preferred first.

    flush_to_s3 writes every segment to BOTH: the short-term bucket, which expires at
    7 days, and the permanent archive in the historical bucket. Nothing is moved at
    day 7 because nothing needs to be -- the archived copy has been there since the
    15-minute flush that produced it.

    Reading prefers the short-term copy and falls back to the archive. Without the
    fallback the archive was write-only: every reader hardcoded the short-term bucket,
    so a day older than 7 days was unreachable even though its bytes were safe.
    """
    ch = f"channel={channel}/" if channel else ""
    return ((config.S3_BUCKET_WS, f"{config.S3_PREFIX_WS}/dt={dt}/{ch}"),
            (config.S3_BUCKET_HIST, f"{config.S3_PREFIX_RAW_ARCHIVE}/dt={dt}/{ch}"))


def raw_archive_key(dt, channel, filename):
    """Raw jsonl.gz copied to the historical bucket so conformance stays
    re-derivable after the 7-day short-term expiry."""
    return f"{config.S3_PREFIX_RAW_ARCHIVE}/dt={dt}/channel={channel}/{filename}"
