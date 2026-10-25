"""Realized-volatility model: measure what vol IS doing, to lean the quotes the implied-vol
model produces.

`pricing/hourly` prices the hourly boards off an ATM implied vol scaled through the variance
calendar. That is a forward-looking number from the option market and it says nothing about what
the index is actually doing right now. This package supplies the second opinion, and the quoting
path uses the DISAGREEMENT between the two to decide how much edge to charge -- never to move the
theo itself.

    data      pull and cache 1-second bars
    features  averaged realized variance, calendar-relative, feature/target construction
    fit       three log-HAR models (60m / 30m / 15m horizons)
    model     the serving side, mirroring pricing.varcal's load/serve split
    live      the 5-second ring buffer the quote loop reads
    validate  must PASS before any of this touches quoting
"""
