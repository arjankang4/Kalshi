"""Market data from the Massive vendor: live quotes and historical NBBO.

`client` is the live path (REST last-quote + options WebSocket) and stays lean because it runs
inside the quote cycle. `history` carries the paging and retry machinery for backfills. The
split is deliberate -- see massive/history.py.
"""
