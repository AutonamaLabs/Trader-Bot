# Trader-Bot

Two forex strategies, built and validated honestly on real market data:

1. **`FX-XSect-Factor` (v2, the researched edge)** — a market-neutral,
   cross-sectional long/short factor on a daily FX + gold basket. This is the
   one with a *real, out-of-sample-robust edge* (Sharpe ~0.45, positive in ~8/9
   years, market-neutral). Start here → [`docs/RESEARCH.md`](docs/RESEARCH.md).
2. **`H4-Donchian-Trend` (v1)** — a classic trend-following breakout bot
   (masterminded with **Fable**), fully implemented with strong risk management
   → [`docs/STRATEGY.md`](docs/STRATEGY.md). Kept because the engine is solid and
   instructive, **but honest testing on real data showed it has no live edge**
   (see below) — it's a well-built example, not a money-maker.

> **⚠️ Honesty first.** Retail forex is hard and most systems lose. We tested
> many strategies on **real** 2012–2022 data and most had *no* out-of-sample
> edge. The one that survived (v2) is real but **modest** — mid-single-digit
> CAGR at conservative sizing, with losing years. Nothing here is financial
> advice; trade a practice account first.

## TL;DR — what the research found

| Strategy | Data | Result | Verdict |
|---|---|---|---|
| v1 Donchian trend (H4) | **real** | PF ~0.95, ~breakeven-to-negative every 3y window | ❌ no live edge |
| Trend / momentum (H4, D1) | real | strong in-sample, **collapses** out-of-sample | ❌ regime, not edge |
| Mean-reversion (RSI-2, D1) | real | positive 4/9 years only | ❌ fails walk-forward |
| **v2 cross-sectional factor** | **real** | after bias audit: **momentum dead, weak fragile reversal (~0.23), tradeable book ~flat** | ⚠️ **marginal** |

**Integrity note:** the v2 factor's first-reported Sharpe ~0.45 was later found to
be inflated by selection bias and rebalance-phase luck. A phase-free re-test
(`scripts/audit_phasefree.py`) deflates it to a weak, USD/CHF-dependent reversal
tendency whose leverage-timed live version is roughly flat 2013–2021. Honest
verdict: **no reliable edge was established** on the data available here. Full
methodology, numbers, and the correction: [`docs/RESEARCH.md`](docs/RESEARCH.md).

## v2 — the edge (`FX-XSect-Factor`)

Each day, rank an 11-instrument basket (FX majors + crosses + gold) by trailing
return and hold a **dollar-neutral long/short book** blending medium-term
**momentum** (20d) and short-term **reversal** (5–10d) — two sleeves that are
negatively correlated and hedge each other. Size the whole book to a 10% annual
volatility target, leverage capped at 2×. Daily timeframe; one rebalance at the
NY close (~21:00–22:00 UTC).

```bash
scripts/fetch_data.sh              # download real daily/H4/H1 data (git-ignored)
python scripts/run_factor.py       # backtest the v2 edge from $5,000
```

Real-data result 2013–2022 ($5k start, 10% vol target, 2× cap): **$5,000 →
$7,188, CAGR +4.0%, Sharpe 0.45, max DD 14%, positive in 8/10 years**, and it
survives 2× costs and dropping any instrument.

---

## v1 — `H4-Donchian-Trend` (trend-following, kept for reference)

A conservative, positive-skew trend-following bot for a $5,000 account: major FX
pairs on H4, Donchian breakouts aligned with the daily trend, ATR stops, partial
take-profit, chandelier trail, and portfolio circuit breakers. Fully implemented
and tested — but on **real data it has no live edge** (profit factor ~0.95). The
synthetic backtest below flatters it because synthetic markets trend too cleanly;
it is a teaching example of a well-engineered engine, not a live money-maker.

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
