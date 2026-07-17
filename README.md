# Trader-Bot — `H4-Donchian-Trend`

A conservative, **positive-skew trend-following forex bot** built for a $5,000
retail account. It trades major FX pairs on the 4-hour timeframe, entering on
Donchian breakouts aligned with the daily trend and managing every position with
ATR-based stops, a partial take-profit, a chandelier trailing stop, and
portfolio-level circuit breakers.

The strategy was masterminded with the **Fable** model and is specified in full
in [`docs/STRATEGY.md`](docs/STRATEGY.md). Every rule in that document is
implemented here and covered by tests.

> **⚠️ Honesty first.** Retail forex is hard and most bots lose money. This
> project is engineered around *risk control and consistency*, not get-rich
> promises. The backtest below runs on **synthetic data** (no broker key
> required) which trends more cleanly than real markets and therefore
> **overstates** live performance. Treat the synthetic results as a
> demonstration of the system's *mechanics and risk behaviour*, not a forecast.
> Realistic expectations are 8–20%/yr with 10–15% drawdowns (see spec §10).
> **Nothing here is financial advice. Trade a practice account first.**

---

## The edge in one paragraph

FX markets trend because monetary-policy divergence and slow institutional flows
push prices in one direction for months, and the large players driving those
flows aren't trying to maximize short-term profit. We capture that with a
**breakout entry** (price makes a 20-bar extreme) but only **in the direction of
the daily EMA-50 regime**, filtered for trend strength (ADX) and sane volatility
(ATR band). We lose small and often (hard 2×ATR stops, breakeven moves, time
stops) and win big and rarely (a partial take-profit plus a 3×ATR trailing stop
that lets winners run). The real edge is **cost control + risk management**, sized
so the inevitable 10-trade losing streak costs ~5% of the account, not the account.

---

## Repository layout

```
config.yaml               All strategy parameters (1:1 with docs/STRATEGY.md)
docs/STRATEGY.md          Full strategy specification (the source of truth)
src/trader_bot/
  indicators.py           EMA, ATR/ADX (Wilder), Donchian, rolling median
  instruments.py          Pip size & pip-value maths (incl. JPY pairs)
  config.py               Typed config loader
  data.py                 CSV loader + reproducible synthetic data generator
  features.py             Enrich H4 bars + no-lookahead D1 regime mapping
  models.py               Position / Signal / Side / state enums
  strategy.py             Entry signals + per-bar exit state machine
  risk.py                 Sizing, exposure/correlation limits, circuit breakers
  backtest.py             Portfolio event-driven backtester (spread+slippage)
  metrics.py              Performance metrics & reporting
  broker.py               Broker interface + PaperBroker + OANDA v20 adapter
  live.py                 Live / dry-run runner (one pass per H4 close)
scripts/
  run_backtest.py         Run a single backtest, write results/
  robustness.py           Run many random histories -> consistency distribution
tests/                    25 unit/integration tests
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Backtest on 8 years of synthetic data (no broker needed)
python scripts/run_backtest.py

# 2) Consistency check across many random market histories
python scripts/robustness.py --runs 30

# 3) Dry-run the live decision loop (places no orders)
python -m trader_bot.live            # from src/, or: python src/trader_bot/live.py

# 4) Run the tests
python -m pytest -q
```

To backtest on **real** data, drop OANDA/Dukascopy H4 CSVs into
`data/EUR_USD_H4.csv` etc. (columns: `time,open,high,low,close`, UTC) and run:

```bash
python scripts/run_backtest.py --data-dir data
```

## Backtest results (synthetic, illustrative)

8 years, 4 pairs, $5,000 start, spread + 0.5-pip slippage modeled, stop-fills-first:

```
Starting equity   : $5,000.00
Final equity      : $19,776.90
CAGR              : +18.85%
Max drawdown      : 5.70%
Profit factor     : 2.43
Expectancy        : +0.663 R / trade
Trades            : 422   (win rate 54.5%)
```

**Consistency** across 30 independent random 8-year histories:

```
Total return %   median +369%   [p10 +274%, p90 +476%]
CAGR %           median +21.4%  [p10 +18.0%, p90 +24.6%]
Max drawdown %   median   4.5%  [worst observed 7.5%]
Profitable histories : 100%     Profit factor > 1.25 : 100%
```

Again — synthetic markets trend cleaner than reality, so live win rate and
returns will be **lower** and drawdowns **higher** than these figures. The point
of the robustness harness is that the system stays positive and well-behaved
across *many* different randomly-generated histories rather than one lucky one.

## Going live (do this carefully)

1. Open an **OANDA practice** account; put the token/account id in `.env`
   (copy from `.env.example` — never commit real keys).
2. Validate on real historical CSVs against the go-live gate in spec §10
   (profit factor ≥ 1.25, DD ≤ 20%, walk-forward positive).
3. Schedule the runner once per H4 close (UTC), e.g. cron:
   ```
   1 0,4,8,12,16,20 * * 1-5  cd /path/to/Trader-Bot && python -m trader_bot.live --broker oanda --live
   ```
   Omit `--live` to keep it in dry-run (logs intended actions only).
4. Run on the practice account for **≥ 3 months** before risking real money.

## Design principles

- **Closed-bar decisions only** — no repainting, no intrabar signal peeking.
- **No lookahead** — the D1 regime filter is mapped so each H4 bar sees only
  fully-completed daily bars (tested in `test_backtest.py`).
- **Backtest == live** — the runner reuses the exact `strategy.py` logic.
- **One place for risk** — all sizing/limits/breakers live in `risk.py`.
- **Conservative fills** — spread + slippage on every fill; stop wins ambiguous bars.

## Roadmap / honest limitations

- Synthetic data flatters trend-following; wire real H4 history before trusting numbers.
- `OandaBroker.modify_stop` needs the stop-order id tracked before live trailing.
- News-filter feed is a config hook; connect an economic-calendar API for production.
- Add walk-forward optimization and Monte-Carlo trade-shuffling to the go-live gate.
- Consider v2: opposite-signal reversals, pyramiding, USD/CAD as a 5th pair.

## License

Provided as-is for educational purposes. Trading involves substantial risk of
loss. You are responsible for your own capital and decisions.
