"""Glue database + Athena external tables over the historical bucket.

Partition projection, not a crawler. Three reasons, in order of weight:

  1. A crawler re-infers column types from data. On a day where a nullable column
     happens to be entirely null it can rewrite the table definition, and the
     next query fails on the union. We have declared, versioned schemas --
     schema discovery here is a hazard, not a feature.
  2. Projection computes partition locations arithmetically, so there is no
     GetPartitions call at plan time and nothing to throttle.
  3. A partition is queryable the instant the object lands. No MSCK REPAIR.

`series_ticker` is deliberately NOT a partition key: 1,172 series x 365 days is
428k mostly-empty partitions and thousands of sub-100KB files. It is instead the
leading sort column, so row-group min/max statistics prune it for free.

Usage:  python -m storage.glue_setup [--apply]
"""
import argparse
import sys

import pyarrow as pa

import config
from storage import paths, s3util, schemas

DATABASE = "kalshi"
ATHENA_RESULTS = f"s3://{config.S3_BUCKET_HIST}/athena-results/"
BACKFILL_START = "2025-08-01"      # not 1970: projection LISTs every prefix in range


def hive_type(t):
    if pa.types.is_string(t):
        return "string"
    if pa.types.is_int64(t):
        return "bigint"
    if pa.types.is_int32(t):
        return "int"
    if pa.types.is_int16(t):
        return "smallint"
    if pa.types.is_boolean(t):
        return "boolean"
    if pa.types.is_floating(t):
        return "double"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_decimal(t):
        return f"decimal({t.precision},{t.scale})"
    if pa.types.is_list(t):
        return f"array<{hive_type(t.value_type)}>"
    if pa.types.is_struct(t):
        inner = ",".join(f"{f.name}:{hive_type(f.type)}" for f in t)
        return f"struct<{inner}>"
    raise TypeError(f"no Hive mapping for {t}")


def ddl(table):
    schema = schemas.TABLES[table]
    cols = ",\n  ".join(f"`{f.name}` {hive_type(f.type)}" for f in schema)
    pkey, _ = paths.partition_kv(table, __import__("datetime").date(2026, 1, 1))
    loc = f"s3://{config.S3_BUCKET_HIST}/{paths.table_prefix(table)}"
    monthly = pkey == "ym"
    proj = {
        "projection.enabled": "true",
        f"projection.{pkey}.type": "date",
        f"projection.{pkey}.format": "yyyy-MM" if monthly else "yyyy-MM-dd",
        f"projection.{pkey}.range": f"{BACKFILL_START[:7] if monthly else BACKFILL_START},NOW",
        f"projection.{pkey}.interval": "1",
        f"projection.{pkey}.interval.unit": "MONTHS" if monthly else "DAYS",
        "storage.location.template": loc + f"{pkey}=${{{pkey}}}/",
        "classification": "parquet",
        "parquet.compression": "SNAPPY",
    }
    # A table with more than one producer partitions by src as well, so both tapes
    # live under one table and a single query still reads across them. enum rather
    # than injected: the set of producers is closed, and projection will not LIST a
    # prefix it was not told about.
    pkeys = [pkey]
    if table in paths.SOURCED_TABLES:
        pkeys.append("src")
        proj["projection.src.type"] = "enum"
        proj["projection.src.values"] = ",".join(sorted(paths.PRODUCERS))
        proj["storage.location.template"] = loc + f"{pkey}=${{{pkey}}}/src=${{src}}/"

    props = ",\n  ".join(f"'{k}' = '{v}'" for k, v in proj.items())
    partitioned_by = ", ".join(f"`{k}` string" for k in pkeys)
    return f"""CREATE EXTERNAL TABLE IF NOT EXISTS `{DATABASE}`.`{table}` (
  {cols}
)
PARTITIONED BY ({partitioned_by})
STORED AS PARQUET
LOCATION '{loc}'
TBLPROPERTIES (
  {props}
)"""


# Presentation views. Never expose raw micro-dollars to ad-hoc SQL, and never
# divide by 1e6 -- that is a float literal and reintroduces exactly the error
# integer micros exists to prevent. DECIMAL division stays exact.
VIEWS = {
    "trades_v": """CREATE OR REPLACE VIEW trades_v AS
SELECT t.*,
       CAST(yes_price_micros AS DECIMAL(18,6)) / DECIMAL '1000000' AS yes_price_dollars,
       CAST(no_price_micros  AS DECIMAL(18,6)) / DECIMAL '1000000' AS no_price_dollars,
       CAST(count_cc         AS DECIMAL(18,2)) / DECIMAL '100'     AS count_contracts
FROM trades t""",

    # Book state with validity as a column rather than a footgun. A caller who
    # ignores book_valid gets an explicit flag, not a plausible-looking empty book.
    "orderbook_snapshots_v": """CREATE OR REPLACE VIEW orderbook_snapshots_v AS
SELECT s.*,
       COALESCE(e.opened_with_snapshot AND e.is_contiguous, false) AS book_valid,
       CAST(best_yes_price_micros AS DECIMAL(18,6)) / DECIMAL '1000000' AS best_yes_dollars,
       CAST(best_no_price_micros  AS DECIMAL(18,6)) / DECIMAL '1000000' AS best_no_dollars
FROM orderbook_snapshots s
LEFT JOIN book_epochs e ON s.book_epoch_id = e.book_epoch_id""",

    "orderbook_deltas_v": """CREATE OR REPLACE VIEW orderbook_deltas_v AS
SELECT d.*,
       COALESCE(e.opened_with_snapshot AND e.is_contiguous, false) AS book_valid,
       CAST(price_micros AS DECIMAL(18,6)) / DECIMAL '1000000' AS price_dollars,
       CAST(delta_cc     AS DECIMAL(18,2)) / DECIMAL '100'     AS delta_contracts
FROM orderbook_deltas d
LEFT JOIN book_epochs e ON d.book_epoch_id = e.book_epoch_id""",

    "ticker_updates_v": """CREATE OR REPLACE VIEW ticker_updates_v AS
SELECT t.*,
       CAST(price_micros   AS DECIMAL(18,6)) / DECIMAL '1000000' AS price_dollars,
       CAST(yes_bid_micros AS DECIMAL(18,6)) / DECIMAL '1000000' AS yes_bid_dollars,
       CAST(yes_ask_micros AS DECIMAL(18,6)) / DECIMAL '1000000' AS yes_ask_dollars,
       CAST(volume_cc      AS DECIMAL(18,2)) / DECIMAL '100'     AS volume,
       CAST(open_interest_cc AS DECIMAL(18,2)) / DECIMAL '100'   AS open_interest
FROM ticker_updates t""",

    "markets_v": """CREATE OR REPLACE VIEW markets_v AS
SELECT m.*,
       CAST(last_price_micros     AS DECIMAL(18,6)) / DECIMAL '1000000' AS last_price_dollars,
       CAST(yes_bid_micros        AS DECIMAL(18,6)) / DECIMAL '1000000' AS yes_bid_dollars,
       CAST(yes_ask_micros        AS DECIMAL(18,6)) / DECIMAL '1000000' AS yes_ask_dollars,
       CAST(notional_value_micros AS DECIMAL(18,6)) / DECIMAL '1000000' AS notional_dollars,
       CAST(open_interest_cc      AS DECIMAL(18,2)) / DECIMAL '100'     AS open_interest,
       CAST(volume_cc             AS DECIMAL(18,2)) / DECIMAL '100'     AS volume
FROM markets m""",
}


def run_query(sql, wait=True, database=None):
    a = s3util._session.client("athena", region_name=config.AWS_REGION) \
        if s3util._session else None
    if a is None:
        s3util.client()
        a = s3util._session.client("athena", region_name=config.AWS_REGION)
    kw = {"QueryString": sql,
          "ResultConfiguration": {"OutputLocation": ATHENA_RESULTS}}
    if database:
        # CREATE VIEW is rejected outright without a database context, even when
        # every name in the statement is fully qualified.
        kw["QueryExecutionContext"] = {"Database": database}
    r = a.start_query_execution(**kw)
    qid = r["QueryExecutionId"]
    if not wait:
        return qid, None
    import time
    while True:
        st = a.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if st["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return qid, st
        time.sleep(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--print", action="store_true", help="dump the DDL and exit")
    a = ap.parse_args(argv)

    stmts = [f"CREATE DATABASE IF NOT EXISTS `{DATABASE}`"]
    stmts += [ddl(t) for t in schemas.TABLES]
    stmts += list(VIEWS.values())

    if a.print or not a.apply:
        for s in stmts:
            print(s + ";\n")
        if not a.apply:
            print("-- dry run; pass --apply to execute", file=sys.stderr)
        return 0

    rc = 0
    for sql in stmts:
        is_view = "VIEW" in sql.split("\n")[0]
        name = (sql.split("VIEW ")[1].split()[0] if is_view
                else (sql.split("`")[1] if "`" in sql else sql[:40]))
        if "CREATE EXTERNAL TABLE" in sql:
            name = sql.split("`.`")[1].split("`")[0]
        qid, st = run_query(sql, database=DATABASE if is_view else None)
        state = st["State"]
        if state != "SUCCEEDED":
            rc = 1
            print(f"  FAIL {name}: {st.get('StateChangeReason','')[:160]}")
        else:
            print(f"  ok   {name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
