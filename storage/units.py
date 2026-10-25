"""Canonical numeric units.

Kalshi sends every price and size as a fixed-point *string*, never a number.
Prices are dollars with up to six decimals, and `price_level_structure` admits
`deci_cent` and `centi_cent` markets -- so the house "integer cents" convention
is lossy here: it would collapse a hundred distinct price levels into one.

    price -> int64 micro-dollars   (1e-6 USD, exactly the API's stated max)
    size  -> int64 centi-contracts (1e-2, exactly FixedPointCount's 2dp)

The convention's intent -- never hold a price as a float -- is preserved. Only
the scale changes. Conversion hard-fails rather than rounding, so if Kalshi ever
widens precision it surfaces as an error instead of silent truncation.
"""
from decimal import Decimal, InvalidOperation

MICROS = 6
CENTI = 2


class PrecisionError(ValueError):
    """Kalshi emitted more precision than the canonical unit can hold."""


def to_micros(s):
    """Fixed-point dollar string -> int64 micro-dollars."""
    if s is None or s == "":
        return None
    try:
        q = Decimal(s).scaleb(MICROS)
    except InvalidOperation:
        raise PrecisionError(f"not a decimal: {s!r}") from None
    if q != q.to_integral_value():
        raise PrecisionError(
            f"price {s!r} needs more than {MICROS}dp; canonical unit would truncate")
    return int(q)


def to_cc(s):
    """Fixed-point count string -> int64 centi-contracts."""
    if s is None or s == "":
        return None
    try:
        q = Decimal(s).scaleb(CENTI)
    except InvalidOperation:
        raise PrecisionError(f"not a decimal: {s!r}") from None
    if q != q.to_integral_value():
        raise PrecisionError(
            f"count {s!r} needs more than {CENTI}dp; canonical unit would truncate")
    return int(q)


def to_strike(v):
    """floor_strike / cap_strike are index levels, not prices -- JSON numbers.

    Real data has 8074.9999, which is not exact in float64, so these are parsed
    as Decimal and stored as decimal128(38,10).
    """
    if v is None:
        return None
    return Decimal(str(v))
