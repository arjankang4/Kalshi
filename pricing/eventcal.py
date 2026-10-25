"""Scheduled macro and earnings events: two historical artifacts for the variance calendar.

    event_calendar.json    which sessions were contaminated, and in which component; the live
                           calendar fit EXCLUDES them (`build`, below)
    release_calendar.json  every modelled release with its time; `pricing/study_event_variance`
                           SUBTRACTS a fitted excess-variance kernel per family instead (`build_releases`)

`pricing/varcal` fits the average session's variance shape. An FOMC decision, a CPI print or an
Apple earnings release is not the average session, and leaving them in means the calendar prices
every quiet Tuesday as though it were partly an FOMC day.

WHAT THIS IS NOT. It is not a forward event calendar and does not predict anything: both artifacts
are historical. `varcal.event_aware` is a candidate, not wired; a live event-aware pricer would
also need a forward schedule.

THE COMPONENT MATTERS MORE THAN THE DATE, and this is the part that is easy to get wrong. The
variance calendar has two separable pieces: the intraday shape and the overnight gap. An event
lands in exactly one of them, decided by the clock:

    release >= 16:00 ET on D   ->  the gap INTO the next session   (mega-cap earnings)
    release <  09:30 ET on D   ->  the gap INTO D                  (CPI, NFP at 08:30)
    09:30 <= release < 16:00   ->  D's intraday session            (FOMC at 14:00)

So the three event families push in OPPOSITE directions. Dropping FOMC days removes intraday
variance and therefore RAISES the measured overnight share; dropping CPI/NFP/earnings days removes
gap variance and LOWERS it. A single "excluded vs not" number hides that, which is why `build`
reports the families separately.

None of the timings are hardcoded. BLS publishes the release time in the schedule row, and SEC
EDGAR stamps every filing with an `acceptanceDateTime`, so the component is derived per event.

SOURCES -- all free, all keyless, all verified reachable 2026-08-21:

    FOMC       federalreserve.gov/monetarypolicy/fomccalendars.htm
               The statement link `/monetary{YYYYMMDD}a...` carries the exact decision date, which
               is more reliable than parsing the "28-29" date range out of the panel.
    CPI/NFP    bls.gov/schedule/{year}/home.htm -- one page per year, with the release TIME.
               NOT bls.gov/schedule/news_release/cpi.htm: that is the forward schedule and yields
               about four dates. `bls.gov/schedule/archives/*` 404s.
    Earnings   data.sec.gov/submissions/CIK{10}.json, 8-K **Item 2.02** ("Results of Operations").
               Item 2.02 is the earnings release itself, so its acceptance timestamp IS the event.
    Members    slickcharts.com/{sp500,nasdaq100} for current constituents and weights.

BUILD ONCE, THEN READ THE ARTIFACT. These are historical facts and the sources are HTML that will
drift. `var/model/event_calendar.json` is the contract; nothing downstream re-scrapes.

    python -m pricing.eventcal --build
    python -m pricing.eventcal --show
    python -m pricing.eventcal --build-releases
"""
import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import config

# SEC requires a descriptive User-Agent with contact details on every automated request.
USER_AGENT = os.environ.get("EDGAR_USER_AGENT", "Kalshi Research contact@example.com")

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_YEAR_URL = "https://www.bls.gov/schedule/{year}/home.htm"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SLICKCHARTS_URL = {"SPX": "https://www.slickcharts.com/sp500",
                   "NDX": "https://www.slickcharts.com/nasdaq100"}

# The two BLS releases EXCLUDED from the live calendar fit. The other BLS families (PPI, JOLTS,
# import prices, ECI, productivity) live in the timed release calendar below; the joint
# excess-variance study finds none of them measurable (pricing/study_event_variance).
BLS_RELEASES = {"Consumer Price Index": "cpi", "Employment Situation": "nfp"}

TOP_N = 10

OPEN_T = datetime.time(9, 30)
CLOSE_T = datetime.time(16, 0)

# SEC asks for <= 10 requests/second. Well under it, and the whole sweep is ~20 tickers.
SEC_WORKERS = 4


def calendar_path():
    return config.MODEL_DIR / "event_calendar.json"


def _get(url, accept=None):
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _text(url):
    return _get(url).decode("utf-8", "replace")


def _json(url):
    return json.loads(_get(url))


# --- component attribution ----------------------------------------------

def component_for(dt_et):
    """(component, session_date) for an event at ET datetime `dt_et`.

    'gap' means the close-to-open move landing on `session_date`; 'intraday' means that session's
    09:30-16:00. An after-close release therefore attaches to the NEXT calendar day, and the
    caller reconciles that to the next *trading* day -- this function deliberately does not know
    the exchange calendar, so it never silently invents a session.
    """
    t = dt_et.time()
    if t >= CLOSE_T:
        return "gap", dt_et.date() + datetime.timedelta(days=1)
    if t < OPEN_T:
        return "gap", dt_et.date()
    return "intraday", dt_et.date()


# --- FOMC ----------------------------------------------------------------

def pull_fomc(start_date, end_date):
    """[(date, 'fomc', 'intraday')] from the Fed's own calendar page.

    Statement links are the anchor. The FOMC statement is released at 14:00 ET on the final day of
    a meeting, so every one of these is an intraday event -- which is why dropping FOMC days RAISES
    the measured overnight share rather than lowering it.
    """
    html = _text(FOMC_URL)
    # Anchor on the "Statement:" label, NOT on every /monetary{date}a link on the page. The page
    # also carries notation votes -- 2025-08-22 links the "Statement on Longer-Run Goals and
    # Monetary Policy Strategy", which is a framework document, not a rate decision at 14:00.
    # Matching the bare link pattern picked it up and would have excluded a clean session.
    out, seen = [], set()
    for m in re.finditer(r'<strong>\s*Statement:\s*</strong>', html):
        nxt = re.search(r'/monetary(\d{8})a', html[m.end():m.end() + 400])
        if not nxt:
            continue
        s = nxt.group(1)
        d = datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        if start_date <= d <= end_date and d not in seen:
            seen.add(d)
            out.append((d, "fomc", "intraday"))
    return sorted(out)


# --- BLS -----------------------------------------------------------------

_BLS_ROW = re.compile(
    r'<td class="date-cell"><p>(?P<date>[^<]+)</p></td>\s*'
    r'<td class="time-cell"><p>(?P<time>[^<]+)</p></td>\s*'
    r'<td class="desc-cell"><p><strong>(?P<name>[^<]+)</strong>', re.S)


def _bls_rows(start_date, end_date):
    """(date, time | None, name) for every row of the BLS year schedules inside the window.

    Holidays carry a blank time cell, so they come back with `time=None` rather than being dropped:
    the jobless-claims derivation needs them.
    """
    for year in range(start_date.year, end_date.year + 1):
        try:
            html = _text(BLS_YEAR_URL.format(year=year))
        except urllib.error.HTTPError as e:
            print(f"[eventcal]   BLS {year}: HTTP {e.code}, skipped", file=sys.stderr)
            continue
        for m in _BLS_ROW.finditer(html):
            try:
                d = datetime.datetime.strptime(
                    m.group("date").split(", ", 1)[1].strip(), "%B %d, %Y").date()
            except (ValueError, IndexError):
                continue
            if not (start_date <= d <= end_date):
                continue
            try:
                t = datetime.datetime.strptime(m.group("time").strip(), "%I:%M %p").time()
            except ValueError:
                t = None
            yield d, t, m.group("name").strip()


def pull_bls(start_date, end_date):
    """[(date, tag, component)] for CPI and the Employment Situation (NFP).

    The release TIME comes out of the row rather than being assumed, so a schedule change -- BLS
    has moved release times before -- reclassifies the event instead of silently misattributing it.
    """
    out = []
    for d, t, name in _bls_rows(start_date, end_date):
        tag = BLS_RELEASES.get(name)
        if tag is None or t is None:
            continue
        comp, sess = component_for(datetime.datetime.combine(d, t))
        out.append((sess, tag, comp))
    return sorted(set(out))


# --- index constituents --------------------------------------------------

_SLICK_ROW = re.compile(r'/symbol/([A-Z.\-]+)"[^>]*>[^<]*</a></td>\s*<td[^>]*>([\d.]+)')


def top_constituents(underlying, n=TOP_N):
    """[(ticker, weight_pct)] -- the current top `n` by index weight.

    CURRENT weights, applied to a two-year window. Historical index weights are not freely
    available, and for mega-caps the approximation is a good one: the names at the top of both
    indices have been the same throughout the sample. It is still an approximation and is recorded
    as one in the artifact.
    """
    html = _text(SLICKCHARTS_URL[underlying])
    rows = _SLICK_ROW.findall(html)
    return [(t, float(w)) for t, w in rows[:n]]


# --- earnings ------------------------------------------------------------

def _cik_map():
    return {v["ticker"]: int(v["cik_str"]) for v in _json(SEC_TICKERS_URL).values()}


def _earnings_for_cik(ticker, cik, start_date, end_date):
    """8-K Item 2.02 events for one issuer, component derived from the acceptance timestamp."""
    try:
        sub = _json(SEC_SUBMISSIONS_URL.format(cik=cik))
    except Exception as e:                                   # noqa: BLE001
        print(f"[eventcal]   {ticker}: {type(e).__name__}, skipped", file=sys.stderr)
        return []
    r = sub.get("filings", {}).get("recent", {})
    out = []
    for form, items, acc in zip(r.get("form", []), r.get("items", []),
                                r.get("acceptanceDateTime", [])):
        if form != "8-K" or "2.02" not in (items or ""):
            continue
        try:
            dt = datetime.datetime.fromisoformat(str(acc).replace("Z", "+00:00"))
        except ValueError:
            continue
        et = dt.astimezone(config.ET)
        comp, sess = component_for(et.replace(tzinfo=None))
        if start_date <= sess <= end_date:
            out.append((sess, f"earnings:{ticker}", comp))
    return out


def pull_earnings(tickers, start_date, end_date):
    """[(date, 'earnings:TICKER', component)] for a set of tickers.

    Item 2.02 is "Results of Operations and Financial Condition" -- the earnings release itself --
    so the filing's acceptance time is the event time, not a proxy for it. Measured: every AAPL
    release in the sample was accepted at 16:30 ET, i.e. after the close, landing in the NEXT
    session's gap.
    """
    cik = _cik_map()
    todo = [(t, cik[t]) for t in tickers if t in cik]
    missing = [t for t in tickers if t not in cik]
    if missing:
        print(f"[eventcal]   no CIK for {missing}", file=sys.stderr)
    out = []
    with ThreadPoolExecutor(max_workers=SEC_WORKERS) as pool:
        for res in pool.map(lambda p: _earnings_for_cik(p[0], p[1], start_date, end_date), todo):
            out.extend(res)
    return sorted(set(out))


# --- build ---------------------------------------------------------------

def build(years=2, end=None, top_n=TOP_N):
    end = end or config.today_et()
    start = end - datetime.timedelta(days=int(365.25 * years))
    print(f"[eventcal] window {start} -> {end}")

    members = {u: top_constituents(u, top_n) for u in ("SPX", "NDX")}
    for u, rows in members.items():
        print(f"[eventcal]   {u} top {top_n}: " + ", ".join(f"{t}({w:.2f}%)" for t, w in rows))
    # GOOGL/GOOG and any other dual-class pair are one issuer reporting once; the CIK map collapses
    # them, and the set here keeps the ticker list from double-counting the request.
    tickers = sorted({t for rows in members.values() for t, _ in rows})

    print(f"[eventcal] pulling FOMC ...")
    ev = list(pull_fomc(start, end))
    print(f"[eventcal]   {len(ev)} FOMC decisions")

    print(f"[eventcal] pulling BLS (CPI, Employment Situation) ...")
    bls = pull_bls(start, end)
    print(f"[eventcal]   {sum(1 for _, t, _ in bls if t == 'cpi')} CPI, "
          f"{sum(1 for _, t, _ in bls if t == 'nfp')} NFP")
    ev += bls

    print(f"[eventcal] pulling earnings for {len(tickers)} issuers ...")
    earn = pull_earnings(tickers, start, end)
    print(f"[eventcal]   {len(earn)} Item 2.02 releases")
    ev += earn

    events = {}
    for d, tag, comp in sorted(ev):
        events.setdefault(str(d), {}).setdefault(comp, [])
        if tag not in events[str(d)][comp]:
            events[str(d)][comp].append(tag)

    art = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {"start": str(start), "end": str(end)},
        "constituents": {u: [{"ticker": t, "weight_pct": w} for t, w in rows]
                         for u, rows in members.items()},
        "constituents_note": ("CURRENT weights applied to the whole window; historical index "
                              "weights are not freely available. Good for mega-caps, an "
                              "approximation nonetheless."),
        "sources": {"fomc": FOMC_URL, "bls": BLS_YEAR_URL,
                    "earnings": SEC_SUBMISSIONS_URL, "members": SLICKCHARTS_URL},
        "n_events": len(ev),
        "events": events,
    }
    out = calendar_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=1))
    print(f"[eventcal] {len(events)} contaminated dates -> {out}")
    return art


# --- serving -------------------------------------------------------------

class EventCalendar:
    """Read side: which dates are contaminated, and in which component."""

    def __init__(self, art):
        self.built_utc = art.get("built_utc")
        self.window = art.get("window", {})
        self.constituents = art.get("constituents", {})
        self.events = {datetime.date.fromisoformat(k): v for k, v in art["events"].items()}

    @classmethod
    def load(cls):
        p = calendar_path()
        if not p.exists():
            raise FileNotFoundError(
                f"no event calendar at {p} -- run: python -m pricing.eventcal --build")
        return cls(json.loads(p.read_text()))

    def tags(self, d, component=None):
        """Tags on date `d`, optionally restricted to 'gap' or 'intraday'."""
        rec = self.events.get(d, {})
        if component is not None:
            return list(rec.get(component, ()))
        return sorted({t for v in rec.values() for t in v})

    def families(self, d, component=None):
        """{'fomc','cpi','nfp','earnings'} present on `d`."""
        return {t.split(":", 1)[0] for t in self.tags(d, component)}

    def is_clean(self, d, component, families=None):
        """True when `d`'s `component` carries no event from `families` (default: any)."""
        f = self.families(d, component)
        return not (f if families is None else (f & set(families)))


def load():
    return EventCalendar.load()


# --- release calendar (timed, for the event-variance model) ---------------
#
# A SEPARATE artifact from `event_calendar.json`, on purpose. `varcal` excludes whatever that file
# lists, so adding JOLTS (10:00) or weekly claims to it would silently change the live default on
# its next `--build`: every JOLTS day would leave the shape fit and every Thursday the gap fit.
# The release calendar is read only by `pricing/study_event_variance` and the candidate profiles.

# BLS row name -> family. Real Earnings is omitted: it prints in the same minute as CPI on every
# date, so no regression can tell the two apart and CPI's kernel already carries it.
BLS_TIMED = {
    "Consumer Price Index": "cpi",
    "Employment Situation": "nfp",
    "Producer Price Index": "ppi",
    "Job Openings and Labor Turnover Survey": "jolts",
    "U.S. Import and Export Price Indexes": "impx",
    "Employment Cost Index": "eci",
    "Productivity and Costs (P)": "prod",
    "Productivity and Costs (R)": "prod",
}

FOMC_T = datetime.time(14, 0)
CLAIMS_T = datetime.time(8, 30)


def release_calendar_path():
    return config.MODEL_DIR / "release_calendar.json"


def claims_dates(start_date, end_date, holidays):
    """Weekly jobless claims: Thursdays at 08:30, the Wednesday before when Thursday is a holiday.

    DERIVED, not scraped -- DOL has no keyless historical schedule. The 2025-10-01..11-12 federal
    shutdown suspended the national release; those weeks are kept here and are expected to FAIL the
    study's spike check, which is what that check is for.
    """
    out = []
    d = start_date + datetime.timedelta(days=(3 - start_date.weekday()) % 7)
    while d <= end_date:
        out.append(d - datetime.timedelta(days=1) if d in holidays else d)
        d += datetime.timedelta(days=7)
    return out


def build_releases(years=2, end=None):
    """Write `release_calendar.json`: one row per (date, family, time_et)."""
    end = end or config.today_et()
    start = end - datetime.timedelta(days=int(365.25 * years))
    print(f"[eventcal] releases, window {start} -> {end}")
    rows, holidays = set(), set()
    for d, t, name in _bls_rows(start, end):
        if t is None:
            holidays.add(d)
            continue
        fam = BLS_TIMED.get(name)
        if fam is not None:
            rows.add((d, fam, t))
    for d, _, _ in pull_fomc(start, end):
        rows.add((d, "fomc", FOMC_T))
    for d in claims_dates(start, end, holidays):
        rows.add((d, "claims", CLAIMS_T))
    rel = [{"date": str(d), "family": f, "time_et": t.strftime("%H:%M")}
           for d, f, t in sorted(rows)]
    art = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {"start": str(start), "end": str(end)},
        "sources": {"bls": BLS_YEAR_URL, "fomc": FOMC_URL,
                    "claims": "derived: Thursday 08:30, Wednesday in a BLS-listed holiday week"},
        "bls_names": BLS_TIMED,
        "holidays": sorted(str(d) for d in holidays),
        "n_releases": len(rel),
        "releases": rel,
    }
    out = release_calendar_path()
    out.write_text(json.dumps(art, indent=1))
    from collections import Counter
    print(f"[eventcal]   {dict(Counter(r['family'] for r in rel))} -> {out}")
    return art


class ReleaseCalendar:
    """Read side of `release_calendar.json`."""

    def __init__(self, art):
        self.built_utc = art.get("built_utc")
        self.window = art.get("window", {})
        self.rows = [(datetime.date.fromisoformat(r["date"]), r["family"],
                      datetime.time.fromisoformat(r["time_et"])) for r in art["releases"]]
        self.by_date = {}
        for d, f, t in self.rows:
            self.by_date.setdefault(d, []).append((f, t))

    @classmethod
    def load(cls):
        p = release_calendar_path()
        if not p.exists():
            raise FileNotFoundError(
                f"no release calendar at {p} -- run: python -m pricing.eventcal --build-releases")
        return cls(json.loads(p.read_text()))

    def on(self, d):
        """[(family, time_et)] released on `d`."""
        return list(self.by_date.get(d, ()))

    def families(self):
        return sorted({f for _, f, _ in self.rows})


def load_releases():
    return ReleaseCalendar.load()


def main(argv=None):
    ap = argparse.ArgumentParser(description="scheduled macro + earnings event calendar")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--years", type=float, default=2)
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--build-releases", action="store_true")
    ap.add_argument("--show-releases", action="store_true")
    a = ap.parse_args(argv)
    if a.build_releases or a.show_releases:
        if a.build_releases:
            build_releases(years=a.years)
        rc = ReleaseCalendar.load()
        from collections import Counter
        print(f"release calendar built {rc.built_utc}, window {rc.window}")
        for (f, t), n in sorted(Counter((f, t.strftime("%H:%M")) for _, f, t in rc.rows).items()):
            print(f"  {f:>7} {t}  {n:4d}")
        return 0
    if a.build:
        build(years=a.years, top_n=a.top)
    if a.show or not a.build:
        cal = EventCalendar.load()
        print(f"event calendar built {cal.built_utc}, window {cal.window}")
        from collections import Counter
        cf = Counter()
        for d in cal.events:
            for comp in ("gap", "intraday"):
                for fam in cal.families(d, comp):
                    cf[(comp, fam)] += 1
        print(f"  {'component':>9} {'family':>10} {'dates':>6}")
        for (comp, fam), n in sorted(cf.items()):
            print(f"  {comp:>9} {fam:>10} {n:6d}")
        print(f"  {len(cal.events)} contaminated dates total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
