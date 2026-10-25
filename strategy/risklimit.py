"""Session loss limit and halt switch -- the two controls this stack did not have.

WHY THIS EXISTS. Every limit in `config` before this one bounds EXPOSURE: contracts per side,
contracts in total, collateral per market, collateral across the book, net inventory in lots. Not
one of them bounds LOSS. A model that is wrong in a consistent direction fills inside every one of
those caps, all day, and no code path in this repo would have noticed -- there was no P&L
accumulator, no drawdown figure and no halt path anywhere (grepped 2026-08-25, before the first
`--live` run). Exposure caps answer "how much can be at risk at one instant"; they say nothing
about how many times we can lose it.

THE P&L IS MARK-TO-MARKET, NOT REALISED, and that is the point. A market maker's realised cash
flow is meaningless mid-session: every fill hands us a position whose value is exactly what the
model says it is, so a book that has "made" $400 of realised spread while accumulating a position
worth -$900 is losing. So:

    pnl = cash + sum_over_markets(net_position * fair)

`cash` accumulates from fill frames; `net_position` comes from the exchange (`MarketFeed.positions`,
fed by `post_position_fp` on every fill and by the `market_position` channel), and `fair` is the
same theo the quoter is resting on. Marking to our own model is the honest choice here even though
it is self-referential: it is the number the position was taken at, and marking to the Kalshi mid
instead would book the very disagreement we are being paid for as a loss the instant we fill.

SIGN CONVENTION -- everything is in SIGNED YES CONTRACTS. A fill carries `outcome_side`, which is
'yes' for buy-yes and sell-no, and 'no' for buy-no and sell-yes. Both sides of a trade print at the
same `yes_price_dollars`, so:

    outcome_side 'yes':  position += n,  cash -= n * p
    outcome_side 'no':   position -= n,  cash += n * p

The 'no' leg is written as "sold YES at p" rather than "bought NO at 1-p" on purpose. The two
differ by a constant $1 per contract -- the collateral returned at settlement -- and only the first
convention makes `cash + position * p` converge to the realised result as p goes to its 0/1
outcome. Getting this backwards produces a P&L that looks fine and is off by the size of the book.

**SUPERSEDED FOR THE LOSS LIMIT, 2026-09-23: the limit now reads ACCOUNT EQUITY MARKED TO THE
KALSHI MID** (`mark_account`, below). The fill-cash mark above stays for the record, but it is not
what halts. On 2026-08-26 it read +$1,744.50 while the account was -$899.05 against its 09:30
open, and the reason is structural, not a sign slip:

  * `cash` starts at 0 in every process, while the position map is SEEDED from REST at startup
    (`run_04`, "SEED THE POSITION MAP"). After a mid-session restart every inherited position was
    therefore marked at its full value with no cost against it -- free profit on paper.
  * An hourly board SETTLES mid-session. The payout is credited to the balance and the position
    disappears, but no fill frame arrives, so `cash` never saw it.
  * Marking to our own theo booked our disagreement with the market as profit.

`mark_account` sidesteps all three by reading what the exchange says we own:

    equity = balance + sum_{q>0} q * mid  +  sum_{q<0} |q| * (1 - mid)
    pnl    = equity - equity at the day's FIRST mark (persisted, so a restart does not reset it)

`balance` is Kalshi's cash balance, which does NOT hold back collateral for resting orders --
measured 2026-08-25: with 12 orders resting, `balance` fell 3500.0011 -> 3489.6011, exactly the
$10.40 of `portfolio_value` that had filled. So resting orders need no term. A long-NO position is
marked at 1 - mid because that is what it is: |q| NO contracts, each worth 1 - (YES mid). A deposit
or withdrawal mid-session moves `balance` and will read as P&L -- do not move money while it runs.

THE HALT FILE is a file, not a flag or a signal, because the operator using it is in a different
shell from the process and possibly on a different day. `touch var/HALT` works with no IPC, needs
no PID, survives a wedged loop and is visible to anyone who lists the directory. Removing it does
not resume -- resuming is a decision that should be made by starting the process again.

Neither control cancels anything itself. Both set the loop's stop event and let the existing
`finally` block run `strategy.cancel_all`, which is the one kill path in this repo that verifies by
re-reading. A second cancel path would be a second thing to keep correct.
"""
import datetime
import json
import os
import threading
import time

import config


def _f(x, default=0.0):
    """Kalshi carries counts and prices as fixed-point STRINGS. Absent is not zero-with-a-warning
    here: an unparseable field means we do not know what happened, and this module's job is to be
    conservative, so it is surfaced by the caller counting `skipped` rather than silently read
    as 0."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class SessionRisk:
    """Cash from fills, marked against live positions and fairs, plus the halt file.

    Pure of I/O except the halt-file `stat`. Nothing here sends, cancels or prices.
    """

    def __init__(self, limit_dollars=None, halt_path=None):
        # `None` means "use the configured limit", NOT "no limit" -- there is deliberately no way
        # to construct an unlimited one. The argument exists so the gate can pin the arithmetic at
        # a convenient number, and the one direction a risk control must never be wrong in is the
        # direction where a caller that forgot to pass anything gets no protection.
        self.limit = (config.DAILY_LOSS_LIMIT_DOLLARS if limit_dollars is None
                      else limit_dollars)
        self.halt_path = config.HALT_FILE if halt_path is None else halt_path
        self.cash = 0.0
        self.fees = 0.0
        self.fills = 0
        self.skipped = 0          # fill frames we could not read; never treated as $0
        self.contracts = 0.0      # gross traded, for the session record
        self.pnl = None           # None until the first mark -- not 0.0, which reads as flat
        self.marked = 0
        self.worst = 0.0
        self.halted = None        # the reason string, once tripped; sticky

    # --- accumulation -----------------------------------------------------

    def on_fills(self, fills):
        """Fold a batch of `fill` frames into cash. Returns the number applied.

        Fees are subtracted whatever the schedule says. `pricing/validate_fees` measured the maker
        fee at zero on a real fill, but a fee model is exactly the thing that changes without
        asking, and a loss limit that trusts a constant is not a loss limit.
        """
        n_ok = 0
        for fl in fills or []:
            n = _f(fl.get("count_fp"), -1.0)
            p = _f(fl.get("yes_price_dollars"), -1.0)
            side = (fl.get("outcome_side") or fl.get("side") or "").lower()
            if n <= 0 or p < 0 or side not in ("yes", "no"):
                self.skipped += 1
                continue
            self.cash += (n * p) if side == "no" else -(n * p)
            fee = _f(fl.get("fee_cost"))
            self.fees += fee
            self.cash -= fee
            self.contracts += n
            self.fills += 1
            n_ok += 1
        return n_ok

    def mark(self, positions, fairs):
        """Re-mark the book. `positions` is {ticker: signed yes contracts}, `fairs` {ticker: p}.

        A position in a market we could not price this cycle is marked at its LAST known fair if we
        have one, and otherwise counted in `unmarked` rather than dropped. Dropping it would make
        the P&L improve every time the pricer failed, which is the direction a risk control must
        never be wrong in.
        """
        if not hasattr(self, "_last_fair"):
            self._last_fair = {}
        value, unmarked = 0.0, 0
        for t, q in (positions or {}).items():
            if not q:
                continue
            p = (fairs or {}).get(t)
            if p is None:
                p = self._last_fair.get(t)
            if p is None:
                unmarked += 1
                continue
            self._last_fair[t] = p
            value += q * p
        for t, p in (fairs or {}).items():
            if p is not None:
                self._last_fair[t] = p
        self.pnl = self.cash + value
        self.marked += 1
        self.worst = min(self.worst, self.pnl)
        self.unmarked = unmarked
        return self.pnl

    # --- the two stop conditions -----------------------------------------

    def halt_reason(self):
        """The reason to stop, or None. Sticky: once tripped it stays tripped.

        Checked before pricing, so a halt works on a cycle the feed gate would have skipped.
        """
        if self.halted:
            return self.halted
        try:
            if self.halt_path and os.path.exists(self.halt_path):
                self.halted = f"halt file present: {self.halt_path}"
                return self.halted
        except OSError:
            # An unreadable halt path is not a reason to keep trading, but it is also not proof of
            # an operator decision. Say nothing and let the loss limit do its job.
            pass
        stale = getattr(self, "stale_s", None)
        if stale is not None and stale > config.RISK_ACCOUNT_STALE_HALT_S:
            self.halted = (f"account mark is {stale:.0f}s stale (> "
                           f"{config.RISK_ACCOUNT_STALE_HALT_S:g}s) -- the loss limit is blind")
            return self.halted
        if self.limit is not None and self.pnl is not None and self.pnl <= -abs(self.limit):
            self.halted = (f"session loss {self.pnl:+.2f} through the "
                           f"{-abs(self.limit):+.2f} limit")
            return self.halted
        return None

    # --- the account mark (what the loss limit reads) ----------------------

    def mark_account(self, snap, books, fairs=None, now_mono=None):
        """Mark ACCOUNT equity to the Kalshi mid and set `pnl` against the day's opening equity.

        `snap` is `AccountPoller.latest()` -- (mono_t, balance_dollars, {ticker: signed yes}) or
        None. `books` is {ticker: (best_bid, best_ask)} in YES dollars. `fairs` {ticker: theo} is
        the last-resort mark for a market with no book at all (closed, not yet settled).

        Returns the pnl, or None when there is no usable snapshot -- in which case `pnl` keeps its
        previous value and `stale_s` says how old the last good one is; `halt_reason` stops the
        loop once that passes RISK_ACCOUNT_STALE_HALT_S.
        """
        now_mono = time.monotonic() if now_mono is None else now_mono
        if not hasattr(self, "_last_mark"):
            self._last_mark, self.unmarked, self.stale_s = {}, 0, None
        if snap is None:
            self.stale_s = None if not hasattr(self, "_snap_t") else now_mono - self._snap_t
            return None
        t, balance, positions = snap
        self._snap_t = t
        self.stale_s = now_mono - t
        value, unmarked, how = 0.0, 0, {}
        for tk, q in (positions or {}).items():
            if not q:
                continue
            bb, ba = (books or {}).get(tk, (None, None))
            m = None
            if bb is not None and ba is not None and 0 < bb < ba < 1:
                m, src = (bb + ba) / 2.0, "mid"
            elif q > 0 and bb is not None and bb > 0:
                m, src = bb, "bid"                # long YES, one-sided: what we could sell at
            elif q < 0 and ba is not None and ba < 1:
                m, src = ba, "ask"                # long NO, one-sided: what buying back costs
            elif tk in self._last_mark:
                m, src = self._last_mark[tk], "last"
            elif fairs and fairs.get(tk) is not None:
                m, src = float(fairs[tk]), "theo"
            if m is None:
                unmarked += 1
                continue
            if src in ("mid", "bid", "ask"):
                self._last_mark[tk] = m
            how[src] = how.get(src, 0) + 1
            value += q * m if q > 0 else (-q) * (1.0 - m)
        # An unmarked position counts at ZERO -- its cost already left the balance, so this errs
        # toward a LOSS, the direction a risk control may be wrong in.
        equity = float(balance) + value
        self.equity = equity
        self.unmarked = unmarked
        self.mark_sources = how
        base = self.opening_equity(equity)
        self.pnl = equity - base
        self.marked += 1
        self.worst = min(self.worst, self.pnl)
        return self.pnl

    def opening_equity(self, equity_now):
        """The day's first marked equity, persisted to RISK_STATE_DIR so every restart shares it."""
        if getattr(self, "_base", None) is not None:
            return self._base
        day = datetime.datetime.now(config.ET).date().isoformat()
        d = getattr(config, "RISK_STATE_DIR", config.ROOT / "var" / "risk")
        p = d / f"opening_equity_{day}.json"
        try:
            self._base = float(json.loads(p.read_text())["equity"])
            self.base_source = str(p)
        except (OSError, ValueError, KeyError):
            self._base = float(equity_now)
            self.base_source = "this process (first mark of the day)"
            try:
                d.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps({"equity": self._base, "day": day,
                                           "written_utc": datetime.datetime.now(
                                               datetime.timezone.utc).isoformat()}))
                tmp.replace(p)
            except OSError:
                pass
        return self._base

    def summary(self):
        return {"fills": self.fills, "skipped_fills": self.skipped,
                "contracts": round(self.contracts, 2),
                "cash": round(self.cash, 4), "fees": round(self.fees, 4),
                "pnl": None if self.pnl is None else round(self.pnl, 4),
                "pnl_basis": "account equity vs day open, marked to Kalshi mid",
                "equity": getattr(self, "equity", None),
                "opening_equity": getattr(self, "_base", None),
                "unmarked": getattr(self, "unmarked", None),
                "mark_sources": getattr(self, "mark_sources", None),
                "worst": round(self.worst, 4),
                "limit": self.limit, "halted": self.halted}


class AccountPoller:
    """Reads the balance and every open position from REST every RISK_ACCOUNT_POLL_S seconds.

    Both come from REST in the same poll so they describe the same instant to within one request,
    which the WS position map cannot promise against a REST balance. Settlements, fees and fills
    placed by an earlier process are all inside `balance` already. A failed read leaves the last
    good snapshot in place and ages it; `SessionRisk.halt_reason` acts on the age.
    """

    def __init__(self, interval_s=None, client=None):
        self.interval_s = config.RISK_ACCOUNT_POLL_S if interval_s is None else interval_s
        if client is None:
            from kalshi import client as client
        self.client = client
        self._snap = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.errors = 0
        self.last_error = None
        self._th = None

    def poll_once(self):
        try:
            b = self.client.balance()
            bal = float(b["balance_dollars"]) if "balance_dollars" in b else b["balance"] / 100.0
            pos = self.client.open_positions(strict=True)
        except Exception as e:                               # noqa: BLE001
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {str(e)[:80]}"
            return False
        with self._lock:
            self._snap = (time.monotonic(), bal, pos)
        return True

    def latest(self):
        with self._lock:
            return self._snap

    def start(self):
        self.poll_once()
        self._th = threading.Thread(target=self._run, name="account-poller", daemon=True)
        self._th.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self.poll_once()

    def stop(self):
        self._stop.set()
