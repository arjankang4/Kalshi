"""Vendor option quotes -> fair value for a Kalshi bracket.

`theo` is the model, `universe` maps Kalshi boundaries onto the option tickers needed to price
them, and `validate` checks the result against live Kalshi quotes before any order is sent.
"""
