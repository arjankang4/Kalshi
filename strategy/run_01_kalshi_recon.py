"""Stage 1: dump Kalshi's Financials markets for today, verbatim, and locate the S&P range event."""
import json
import sys
from datetime import datetime, time as dtime, timedelta

import config
from kalshi import client as k


def dump(outdir, name, obj):
    path = outdir / name
    path.write_text(json.dumps(obj, indent=2))
    return path


def main():
    stamp = config.run_stamp()
    today = config.today_et()
    outdir = config.RAW_DIR / "kalshi" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"run {stamp}  |  ET date {today}  |  -> {outdir}")

    # -- 1. every series in the Financials category (Kalshi's exact category string) --
    series = k.series_by_category(config.KALSHI_CATEGORY)
    dump(outdir, "series_financials.json", series)
    print(f"\n{len(series)} series in category {config.KALSHI_CATEGORY!r}")

    # -- 2. today's open markets, saved exactly as returned --
    # One paginated sweep bounded by close_time, then filter to the Financials series locally.
    # Far cheaper than a call per series, which trips Kalshi's read throttle.
    fin = {s["ticker"]: s for s in series}
    day_start = datetime.combine(today, dtime(0, 0), tzinfo=config.ET)
    open_today = k.markets(
        status="open",
        min_close_ts=int(day_start.timestamp()),
        max_close_ts=int((day_start + timedelta(days=1)).timestamp()),
    )
    dump(outdir, "markets_open_today_all_categories.json", open_today)
    print(f"{len(open_today)} open markets exchange-wide closing today")

    index, all_today = {}, []
    by_series = {}
    for m in open_today:
        by_series.setdefault(m["event_ticker"].split("-")[0], []).append(m)

    for st, todays in sorted(by_series.items()):
        if st not in fin:
            continue
        s = fin[st]
        dump(outdir, f"markets_{st}.json", todays)
        events = sorted({m["event_ticker"] for m in todays})
        index[st] = {"title": s.get("title"), "markets_today": len(todays), "events": events}
        all_today.extend(todays)
        print(f"  {st:<24} {len(todays):>4} markets  {len(events)} event(s)  {s.get('title','')[:52]}")

    dump(outdir, "index.json", {"run": stamp, "date": str(today), "series": index})
    print(f"\ntotal Financials markets closing today: {len(all_today)}")

    # -- 3. the target: S&P price range today at 4pm --
    target = config.event_ticker_for(today)
    tgt = [m for m in all_today if m["event_ticker"] == target]
    if not tgt:
        print(f"\n!! {target} not found among today's open markets", file=sys.stderr)
        return 1

    dump(outdir, f"event_{target}.json", tgt)
    print(f"\n=== {target}  ({len(tgt)} markets) ===")
    print(f"{'ticker':<34}{'type':<9}{'floor':>10}{'cap':>10}  {'yes_bid':>8}{'yes_ask':>8}{'vol':>9}")
    for m in sorted(tgt, key=lambda m: m.get("floor_strike") or 0):
        print(f"{m['ticker']:<34}{m.get('strike_type',''):<9}"
              f"{m.get('floor_strike') or '':>10}{m.get('cap_strike') or '':>10}  "
              f"{m['yes_bid_dollars']:>8}{m['yes_ask_dollars']:>8}{float(m['volume_fp']):>9.0f}")

    # -- 4. orderbook snapshot per market, verbatim --
    books = {m["ticker"]: k.orderbook(m["ticker"]) for m in tgt}
    dump(outdir, f"orderbooks_{target}.json", books)
    print(f"\nsaved {len(books)} orderbooks")
    sample = next(iter(books.items()))
    print(f"orderbook shape, {sample[0]}:\n{json.dumps(sample[1])[:220]}")

    (config.RAW_DIR / "kalshi" / "latest").write_text(stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
