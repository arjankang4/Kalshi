"""Memory-bounded Parquet writing.

Peak resident memory is a function of the batch and row-group sizes, never of
the input size. That property is the whole point: this runs beside a live
trading loop, and a converter whose footprint scales with a busy day is a
converter that eventually takes the trading process down with it.

What is deliberately NOT used here, each being O(input) in RSS:
    pandas.read_json           a 400 MB gzipped day expands to 8-15 GB
    pyarrow.json.read_json     returns one Table, and infers schema per file
    pa.Table.from_pylist       holds every dict at once
    Table.sort_by on a day     makes peak a function of input size
"""
import hashlib
import json
import resource
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq

BATCH_ROWS = 65_536          # Arrow build unit
ROW_GROUP_ROWS = 1_000_000   # ~128 MB for a NARROW schema; see ROW_GROUP_BYTES
DATA_PAGE_BYTES = 1 << 20
DICT_PAGE_LIMIT = 2 << 20
ARROW_BUDGET = 900 << 20     # hard backstop on pyarrow-allocated bytes

# The row bound alone is not enough. "~128 MB for these schemas" holds for the
# narrow tables (trades is 19 columns) but not for markets, which is 54 columns
# at ~2.24 KB/row: 1M buffered rows is ~2.24 GB, so ARROW_BUDGET fired long
# before the row-count flush was ever reached. Flushing on whichever bound is
# hit first makes the row group a function of BYTES, which is what both Athena
# and the memory ceiling actually care about.
ROW_GROUP_BYTES = 256 << 20

# 32 vCPUs would otherwise mean 32 compression threads per writer.
pa.set_cpu_count(4)
pa.set_io_thread_count(2)
try:
    pa.jemalloc_set_decay_ms(0)   # return freed arenas promptly so RSS is honest
except Exception:                 # noqa: BLE001
    pass


def peak_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def to_batches(rows, schema, size=BATCH_ROWS):
    """Stream dicts -> RecordBatches, column-wise.

    Peak is one batch of Python objects, released as soon as the batch is built.
    """
    fields = list(schema)
    cols = {f.name: [] for f in fields}
    n = 0
    for row in rows:
        for f in fields:
            cols[f.name].append(row.get(f.name))
        n += 1
        if n == size:
            yield pa.RecordBatch.from_arrays(
                [pa.array(cols[f.name], type=f.type) for f in fields], schema=schema)
            for v in cols.values():
                v.clear()
            n = 0
    if n:
        yield pa.RecordBatch.from_arrays(
            [pa.array(cols[f.name], type=f.type) for f in fields], schema=schema)


def _normalizer(field):
    """Canonicalize a value the way the *schema* will store it, not the way
    Python happens to type it.

    This matters more than it looks. `fee_multiplier` arrives from Kalshi as the
    JSON integer 1, lands in a float64 column, and reads back as 1.0 -- so a
    naive str() digest reports corruption on a file that is perfectly correct.
    decimal128 has the same problem in reverse: Decimal("8074.9999") is stored at
    scale 10 and reads back as Decimal("8074.9999000000").
    """
    t = field.type
    if pa.types.is_boolean(t):
        return lambda v: "\x00" if v is None else ("T" if v else "F")
    if pa.types.is_floating(t):
        return lambda v: "\x00" if v is None else repr(float(v))
    if pa.types.is_integer(t):
        return lambda v: "\x00" if v is None else str(int(v))
    if pa.types.is_decimal(t):
        return lambda v: "\x00" if v is None else _dec(v)
    if pa.types.is_timestamp(t):
        return lambda v: "\x00" if v is None else _tstr(v)
    if pa.types.is_nested(t):
        return lambda v: "\x00" if v is None else json.dumps(
            _deep(v), sort_keys=True, separators=(",", ":"))
    return lambda v: "\x00" if v is None else str(v)


def _dec(v):
    d = Decimal(str(v)).normalize()
    # normalize() renders integral values in exponent form (1E+2); undo that.
    return format(d, "f")


def _tstr(v):
    if isinstance(v, (int, float)):
        return str(int(v))
    s = v.isoformat()
    return s.replace("+00:00", "Z")


def _deep(v):
    if isinstance(v, dict):
        return {k: _deep(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep(x) for x in v]
    if isinstance(v, Decimal):
        return _dec(v)
    if isinstance(v, float):
        return repr(v)
    if hasattr(v, "isoformat"):
        return _tstr(v)
    return v


class RowDigest:
    """Two digests over a canonical per-row serialization.

    unordered  XOR of per-row SHA-256 -- order-independent, so it catches loss,
               duplication and value corruption but is blind to reordering.
    ordered    rolling SHA-256 in write order -- catches reordering.

    Together they prove a Parquet file holds the same multiset of rows, in the
    same order, as what the conformance layer produced. That is a much stronger
    statement than a row count, for one extra streaming pass.
    """

    def __init__(self, schema):
        self.names = [f.name for f in schema]
        self.norm = [_normalizer(f) for f in schema]
        self.unordered = bytearray(32)
        self.ordered = hashlib.sha256()
        self.rows = 0

    def add_row(self, row):
        blob = "\x1f".join(fn(row.get(n)) for n, fn in zip(self.names, self.norm))
        h = hashlib.sha256(blob.encode()).digest()
        for i in range(32):
            self.unordered[i] ^= h[i]
        self.ordered.update(h)
        self.rows += 1

    def add_batch(self, batch):
        for row in batch.to_pylist():
            self.add_row(row)

    def result(self):
        return {"unordered_xor": bytes(self.unordered).hex(),
                "ordered_sha256": self.ordered.hexdigest(),
                "rows": self.rows}


class BoundedWriter:
    """Accumulates RecordBatches to an exact row-group size, then flushes."""

    def __init__(self, path, schema, dict_cols=None, sort_key=None):
        self.schema = schema
        self.path = path
        kw = dict(
            compression="snappy",
            version="2.6",
            data_page_version="2.0",
            write_statistics=True,
            data_page_size=DATA_PAGE_BYTES,
            dictionary_pagesize_limit=DICT_PAGE_LIMIT,
            use_dictionary=dict_cols or True,
            write_page_index=True,      # page-level min/max for finer skipping
            store_schema=False,         # drop the ARROW:schema blob Athena ignores
        )
        if sort_key:
            names = [f.name for f in schema]
            kw["sorting_columns"] = [
                pq.SortingColumn(names.index(c)) for c in sort_key if c in names]
        try:
            self._w = pq.ParquetWriter(str(path), schema, **kw)
        except TypeError:
            # Older pyarrow: drop the optional niceties rather than hard-failing.
            for opt in ("sorting_columns", "write_page_index", "store_schema"):
                kw.pop(opt, None)
            self._w = pq.ParquetWriter(str(path), schema, **kw)
        self._buf, self._rows = [], 0
        self.total_rows = 0

    def write_batch(self, batch):
        # Never writer.write_batch() directly: pyarrow would make each batch its
        # own row group, giving ~15 tiny row groups per row-group's worth of data
        # and destroying both compression ratio and Athena's ability to skip.
        self._buf.append(batch)
        self._rows += batch.num_rows
        self.total_rows += batch.num_rows
        if self._rows >= ROW_GROUP_ROWS or pa.total_allocated_bytes() >= ROW_GROUP_BYTES:
            self._flush()
        alloc = pa.total_allocated_bytes()
        if alloc > ARROW_BUDGET:
            raise MemoryError(
                f"arrow pool {alloc/1e6:.0f} MB exceeds budget "
                f"{ARROW_BUDGET/1e6:.0f} MB writing {self.path.name}")

    def _flush(self):
        if not self._rows:
            return
        # from_batches is zero-copy -- it wraps the batches as chunks rather than
        # concatenating, which is why buffering them costs only the batches.
        tbl = pa.Table.from_batches(self._buf, self.schema)
        self._w.write_table(tbl, row_group_size=ROW_GROUP_ROWS)
        self._buf.clear()
        self._rows = 0
        del tbl
        pa.default_memory_pool().release_unused()

    def close(self):
        self._flush()
        self._w.close()
        return self.total_rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_rows(path, table_name, rows, schemas_mod):
    """Convenience: stream dicts to one Parquet file, returning the digests."""
    schema = schemas_mod.TABLES[table_name]
    dig = RowDigest(schema)
    with BoundedWriter(path, schema,
                       dict_cols=schemas_mod.dict_columns(table_name),
                       sort_key=schemas_mod.SORT_KEYS[table_name]) as w:
        for batch in to_batches(_tee(rows, dig), schema):
            w.write_batch(batch)
    return dig.result()


def _tee(rows, digest):
    for r in rows:
        digest.add_row(r)
        yield r


def verify_file(path, schema, expected):
    """Re-read with bounded memory and recompute the digests.

    Equality proves the file holds exactly the rows that were handed to the
    writer, in order -- not merely the same number of them.
    """
    pf = pq.ParquetFile(str(path))
    dig = RowDigest(schema)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS):
        dig.add_batch(batch)
    got = dig.result()
    ok = (got["unordered_xor"] == expected["unordered_xor"]
          and got["ordered_sha256"] == expected["ordered_sha256"]
          and got["rows"] == expected["rows"])
    return ok, got, {"num_rows": pf.metadata.num_rows,
                     "row_groups": pf.metadata.num_row_groups,
                     "bytes": path.stat().st_size}


# --- reading a partition back out of S3 ------------------------------------

_S3FS = None


def s3_filesystem():
    """Arrow's own S3 filesystem, built once and reused.

    Deliberately not boto3: see read_s3_table for why the bytes must never
    reach Arrow through a Python file object.
    """
    global _S3FS
    if _S3FS is None:
        import pyarrow.fs as pafs
        import config
        _S3FS = pafs.S3FileSystem(region=config.AWS_REGION)
    return _S3FS


def read_s3_table(bucket, key, columns=None):
    """One Parquet partition from S3, or None if the key does not exist.

    **Never hand Arrow an io.BytesIO.** `pq.read_table(io.BytesIO(body))` reads
    correctly and then destroys the process during interpreter teardown -- either
    SIGABRT ("terminate called without an active exception", exit 134) or an
    unkillable hang that ignores SIGTERM. Measured on this box, pyarrow 21.0.0:
    the in-memory form failed 3/3 runs, reading the same bytes from a path passed
    3/3. It is not boto3's doing and it is not a leaked HTTP connection -- reading
    a purely local file through BytesIO reproduces it with boto3 never imported,
    and closing the stream first does not help. pa.BufferReader and pa.py_buffer
    fail the same way.

    That mattered more than a crash usually does, because the work finishes
    *before* the teardown: a job could upload correct output and still be marked
    failed. Both prior call sites -- backtest/data.py and massive_backfill.py --
    were the in-memory form, so "read a partition back" was a process-killing
    operation across the whole repo, which is exactly what a study harness does
    in a loop.

    Arrow's own filesystem hands the reader a native file and sidesteps it
    entirely, with the bonus that `columns` becomes a range request rather than a
    full download.
    """
    try:
        f = s3_filesystem().open_input_file(f"{bucket}/{key}")
    except (FileNotFoundError, OSError):
        return None
    return pq.read_table(f, columns=columns)
