"""Kalshi storage layer: WebSocket capture, S3 transfer, Parquet conversion.

Stage map:
    ws_recorder  -> local gzipped segments
    flush_to_s3  -> s3://<S3_BUCKET_WS>            (raw, expires at 7d)
    promote_ws   -> s3://<S3_BUCKET_HIST> (Parquet, T+3)
    rest_backfill-> s3://<S3_BUCKET_HIST> (Parquet, one-shot)

REST is canonical. WebSocket data is amended to match REST, never the reverse.
"""
