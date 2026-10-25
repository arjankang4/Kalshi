# Kalshi — SPX / NDX index-market quoting engine (Python)

A market-making engine for Kalshi's S&P 500 and Nasdaq-100 event contracts (the 4pm close boards
`KXINX` / `KXNASDAQ100` and the hourly boards `KXINXU` / `KXNASDAQ100U`). It prices every binary off
the live 0DTE index option board, sizes the edge it requires from each contract's dollar Greeks, and
maintains three-rung, two-sided ladders on the Kalshi book under hard risk limits.

This is the **reference implementation**. Two sibling repos build on it:

| repo | what it is |
|---|---|
| **Kalshi** (this repo) | the Python engine: market data → fair value → required edge → ladders → orders, plus the capture/storage pipeline |
| [**Kalshi_Prod**](https://github.com/arjankang4/Kalshi_Prod) | a C++20 port of the whole live path, held **bit-identical** to this Python by a golden-vector and replay-parity gate |
| [**Kalshi_BTC**](https://github.com/arjankang4/Kalshi_BTC) | the same quoting rules applied to Kalshi's BTC 15-minute and hourly markets, with its own spot, vol and venue layers |

> **What is not published.** This is a public copy of a private repository. The research and
> backtests, the fitted model artifacts (`var/model/`: variance calendars, edge-curve anchors,
> realized-vol fits), the market-data datasets, and the live tuned parameters are private. Every
> tuned value in `config.py` and in module constants is **illustrative** and marked as such.
> Comments occasionally cite a research module or notebook that is not included. Running the
> engine needs Kalshi and Massive API credentials (see `.env.*.example`) and those model artifacts.

## How a quote is made

```
Massive options WS ──► per-contract microprice ──► smile fit (OTM legs, sigma-moneyness)
Massive index / futures ──► spot, parity forward, futures delta between snapshots
                                   │
                                   ▼
        P(S > K) = N(d2) − F·φ(d1)·dw/dK        (pricing/skew.py, pricing/theo.py)
        hourly boards: same, scaled through an intraday variance calendar (pricing/hourly.py)
                                   │
                                   ▼
   dollar Greeks (delta, vega, theta) → R → required edge from a calibrated curve (pricing/edge.py)
   ± realized-vs-implied vol lean on the required edge (pricing/lean.py, rvol/)
                                   │
                                   ▼
   ladder: improve → penny → join, up to three rungs a side, each gated on its own edge
   (strategy/quoting.py) → requote deadband that only moves an order when it is worth the
   queue position (strategy/orders.py) → signed REST orders (kalshi/client.py)
                                   │
                                   ▼
   risk: daily loss halt on account equity, position limits, resting-collateral budget,
   kill switch (`touch var/HALT`), cancel-on-exit (strategy/risklimit.py, strategy/budget.py)
```

A few of the ideas the code is built around:

- **A binary is the strike-derivative of the call curve**, `P(S > K) = −dC/dK`. With a
  strike-dependent vol the chain rule leaves a smile-slope term, `−F·φ(d1)·dw/dK`, that a naive
  "read the vol off the smile and plug it into N(d2)" drops. The pricer carries it.
- **Only out-of-the-money legs are inverted.** An in-the-money option's time value is a small
  difference of large numbers and inherits the whole quote width.
- **Black–Scholes is parameterised by total vol** `w = σ√T`, so no year-fraction convention leaks in.
  The 4pm board settles when the options expire; hourly boards need a variance calendar.
- **Greeks are in dollars per contract and never pass through a probability clamp.** For a binary,
  gamma is vega rescaled, so the risk number uses delta, vega and theta only.
- **Kalshi is price-time and any price amendment forfeits queue position**, so the requote policy
  is a deadband: an order is touched only for a reason worth its place in the queue.

## Layout

| directory | duty |
|---|---|
| `kalshi/` | signed access to the Kalshi Trade API v2 (RSA-PSS signing, REST, WebSocket book) |
| `massive/` | vendor market data: options WS, index and futures REST, historical NBBO |
| `pricing/` | quotes → fair value: `bs`, `skew`, `theo`, `universe`, `spot`, `basis`; hourly: `varcal`, `gapcal`, `atmvol`, `hourly`; quoting inputs: `greeks`, `edge`, `lean`, `fees`. `validate_theo` is the runtime pricing entry point |
| `strategy/` | the live session (`run_04_quote_loop`) and the live state `pricing/` deliberately holds none of: `skewstate`, `edgemgr`, `leanchain`, `deltaadj`, `fastcancel`, `orders`, `band`, `budget`, `risklimit`, `risktape` |
| `rvol/` | realized-vol model: data, features, dataset, fit, live feed |
| `edgecal/` | loader for the edge-curve anchors (the calibration itself is private) |
| `storage/` | WebSocket capture → S3 → Parquet, with conformance, replay and monitoring; see `storage/README.md` |
| `harness.py` | the small pass/fail harness every `test_*` module uses |

## Running

Python 3.12; dependencies in `requirements.txt` (`pip install -r requirements.txt`).
Run everything as modules from the repo root.

```bash
cp .env.kalshi.example .env.kalshi && cp .env.massive.example .env.massive   # then fill in

# unit tests (no network, no model artifacts needed)
for f in $(find . -name 'test_*.py'); do python -m $(echo ${f#./} | sed 's#/#.#g; s#\.py$##'); done

python -m strategy.run_04_quote_loop --minutes 5 --tiers --edge   # dry run: price, plan, send nothing
python -m strategy.run_04_quote_loop --live                       # sends real orders
python -m strategy.cancel_all                                     # cancel everything resting
touch var/HALT                                                    # kill switch, honoured within one cycle
```

The quote loop is dry unless `--live` is passed. SIGTERM/SIGINT and `var/HALT` all take the same
cancel-on-exit path.
