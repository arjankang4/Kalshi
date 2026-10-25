"""Stage 2: capture raw Massive options Quotes (and SPY spot) for the strikes KXINX needs."""
import json
import time

import config
from massive import client as mv
from pricing import universe


def main(duration=60.0):
    stamp = config.run_stamp()
    outdir = config.RAW_DIR / "massive" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    expiry = config.occ_expiry()
    print(f"run {stamp}  |  expiry {expiry}  |  -> {outdir}")

    markets = universe.load_event_markets()
    root, strikes, contracts = universe.option_root_and_strikes()
    (outdir / "contracts_spx.json").write_text(json.dumps(contracts, indent=2))
    print(f"{len(contracts)} contracts, root {root}, {len(strikes)} strikes "
          f"({strikes[0]}..{strikes[-1]})")

    bounds, needed, tickers = universe.build(markets, strikes, root, expiry)
    print(f"{len(markets)} markets -> {len(bounds)} boundaries ({bounds[0]}..{bounds[-1]}) "
          f"-> {len(needed)} strikes -> {len(tickers)} option tickers")
    assert len(tickers) <= 1000, "exceeds Massive's 1000-contract-per-connection cap"
    (outdir / "subscribed_tickers.json").write_text(json.dumps(tickers, indent=2))

    spy = mv.stock_nbbo()
    (outdir / "spy_spot.json").write_text(json.dumps(spy, indent=2))
    print(f"SPY NBBO mid {mv.stock_mid(spy):.4f}  (x10 = {mv.stock_mid(spy)*10:.1f})")

    raw_path = outdir / "quotes.jsonl"
    col = mv.QuoteCollector(tickers, raw_path=raw_path).start()
    print(f"\nsubscribed; collecting {duration:.0f}s -> {raw_path.name}")
    got = col.wait_for_coverage(len(tickers) // 2, timeout=25)
    print(f"coverage {len(col.snapshot())}/{len(tickers)} symbols after warmup (half-covered: {got})")
    time.sleep(max(0.0, duration - 25))
    snap = col.snapshot()
    col.stop()

    (outdir / "latest_quotes.json").write_text(json.dumps(snap, indent=2))
    print(f"\n{col.frame_count} frames / {col.msg_count} messages / {len(snap)} symbols quoted")

    lag = [time.time() * 1000 - q["t"] for q in snap.values()]
    lag.sort()
    print(f"quote age ms: p50 {lag[len(lag)//2]:.0f}  p95 {lag[int(len(lag)*.95)]:.0f}  "
          f"max {lag[-1]:.0f}   <-- must be seconds, not ~900000 (delayed feed)")

    print("\nsample raw Q frames:")
    for sym in sorted(snap)[:3]:
        print("  " + json.dumps(snap[sym]))

    (config.RAW_DIR / "massive" / "latest").write_text(stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
