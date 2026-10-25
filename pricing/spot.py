"""Resolve the spot price for an underlying: real index level, with put-call parity as fallback.

Until 2026-08-21 there was no choice to make -- `I:SPX` was not entitled, so spot was whatever
put-call parity said (`theo.Pricer.parity_spot`, `atmvol.parity_forward`). With the Index
entitlement the index is the truer number, but it is not strictly better in every respect, and
that is why this is a module rather than a one-line swap:

  * **The index is truer.** It is the thing KXINX actually settles on. Parity infers it from
    option quotes and inherits every defect of those quotes -- most sharply the deep-ITM call
    with no bid, which prices to 0.00, wins the min |C-P| anchor and implies S = its own strike.
    Measured in the backtest: spot 7387.5 on a day whose true spot was ~7706.

  * **Parity is more internally consistent.** It is derived from the SAME quote snapshot the
    theos are, so a stale or skewed option board moves parity and the theos together and the
    errors partially cancel. An index level is correct but arrives on a different feed with its
    own clock, so adopting it introduces a skew that did not previously exist. This is the same
    hazard that made the one live hourly-model check meaningless at 219s of skew.

So: index when it is fresh, parity when it is not, and record BOTH every time along with their
divergence. That divergence is the single best health signal available on either feed -- it goes
wrong when the option board thins out AND when the index feed stalls, and it costs one
subtraction.
"""
import config

MISSING = None


class SpotResolution:
    """What spot is, where it came from, and how much the two sources disagreed.

    `value` is the number to price with. `source` is 'index' or 'parity' or None. `divergence`
    is `index - parity` in points and is None unless both were available -- which is exactly
    when it is meaningful.
    """

    __slots__ = ("value", "source", "index", "parity", "index_age_s",
                 "market_status", "divergence", "divergence_pct", "alarm", "reason")

    def __init__(self, value=None, source=None, index=None, parity=None, index_age_s=None,
                 market_status=None, reason=None):
        self.value = value
        self.source = source
        self.index = index
        self.parity = parity
        self.index_age_s = index_age_s
        self.market_status = market_status
        self.reason = reason
        self.divergence = None
        self.divergence_pct = None
        self.alarm = False
        if index is not None and parity is not None and parity > 0:
            self.divergence = index - parity
            self.divergence_pct = self.divergence / parity
            self.alarm = abs(self.divergence_pct) > config.INDEX_PARITY_ALARM_PCT

    def as_dict(self):
        """Flat record for the journal and the cycle state. Keys are stable."""
        return {"spot": self.value, "spot_source": self.source,
                "spot_index": self.index, "spot_parity": self.parity,
                "index_age_s": self.index_age_s, "index_market_status": self.market_status,
                "spot_divergence": self.divergence, "spot_divergence_pct": self.divergence_pct,
                "spot_alarm": self.alarm, "spot_reason": self.reason}

    def __repr__(self):
        return (f"SpotResolution({self.value!r} from {self.source!r}, "
                f"div={self.divergence!r}, alarm={self.alarm})")


def resolve(parity_spot, index_quote=None, max_stale_s=None):
    """(parity spot, live IndexQuote|None) -> SpotResolution.

    Pure: takes the two numbers, not the feeds. `index_quote` is a `massive.index.IndexQuote`
    or None when the snapshot failed or omitted this underlying.

    The freshness test is on the vendor's own timestamp, not on when we made the call, so a
    served-from-cache response cannot masquerade as live.
    """
    max_stale_s = config.INDEX_MAX_STALE_S if max_stale_s is None else max_stale_s

    idx_val = age = status = None
    if index_quote is not None:
        idx_val, status = index_quote.value, index_quote.market_status
        age = index_quote.age_s()

    if index_quote is not None and index_quote.is_fresh(max_stale_s):
        return SpotResolution(value=idx_val, source="index", index=idx_val, parity=parity_spot,
                              index_age_s=age, market_status=status)

    if index_quote is None:
        reason = "no_index_quote"
    elif age is None:
        reason = "index_no_timestamp"
    elif age < 0:
        # A negative age means our clock and the vendor's disagree. Treating it as fresh would
        # hide a clock skew that also corrupts every other age check in the process.
        reason = "index_timestamp_in_future"
    else:
        reason = f"index_stale_{age:.1f}s"

    if parity_spot is not None:
        return SpotResolution(value=parity_spot, source="parity", index=idx_val,
                              parity=parity_spot, index_age_s=age, market_status=status,
                              reason=reason)

    return SpotResolution(value=None, source=None, index=idx_val, parity=None,
                          index_age_s=age, market_status=status,
                          reason=f"{reason}+no_parity")
