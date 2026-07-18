# Pepperstone MT5 demo forward-test — setup

Forward-testing runs on **your own Windows machine** (MetaTrader 5's Python API is
local + Windows; this repo's cloud environment cannot reach the broker). The bot
computes signals from **your adjusted daily data** and uses MT5 only to execute.

> **Honest expectation (per the survivorship-free validation + Fable review):**
> live CAGR **3–7% (~5%)**, Sharpe **0.3–0.5**, max drawdown **20–30%**. This may
> not beat SPY buy-and-hold risk-adjusted; the value is drawdown control and
> diversification. The demo is an **execution audit** (does the plumbing work?),
> not proof of edge — 6–12 months of demo barely moves the needle statistically.

## 1. Accounts & install
1. Open a **Pepperstone demo** account (MT5). Note login, password, server
   (e.g. `Pepperstone-Demo`).
2. Install the MT5 **terminal** and log into the demo.
3. In the terminal, find the exact **US share CFD** symbol names (e.g. `AAPL.US`)
   — Market Watch → right-click → Symbols → Stocks.
4. `pip install MetaTrader5 pandas numpy pyyaml`

## 2. Data (critical — Fable's #1 rule)
Compute signals from **dividend/split-ADJUSTED** daily bars, never the broker's
unadjusted CFD feed. Keep `data/swing_stocks/` (or `data/universe_broad/`)
refreshed daily from your adjusted source (the repo's `fetch` scripts pull a
static snapshot; for live use, wire a daily updater).

## 3. Configure & run
Edit `SYMBOL_MAP_EXAMPLE` in `scripts/mt5_runner.py` to map each data ticker to
your broker's CFD symbol. Then, **once per trading day after the US cash close**:

```bash
python scripts/mt5_runner.py                      # DRY-RUN: prints intended actions
python scripts/mt5_runner.py --live --login 123 --password ... --server Pepperstone-Demo
```

Schedule it with Windows Task Scheduler ~10 min after the 16:00 ET close. Signals
use the completed daily bar; orders fill at the next session's open.

## 4. Which sleeves to run on a CFD broker
- **Mean-reversion sleeve: YES** — ~3.6-day holds, overnight financing is
  negligible. This is the default (`use_trend_sleeve=False`).
- **Trend sleeve: NO on CFDs** — it holds for months; long US-share-CFD financing
  (~7–8%/yr) would eat most of the edge. Run it only on a cash-equity broker
  (e.g. IBKR) or not at all. Enable with `--use-trend` only if you've modeled
  financing.

## 5. What to LOG and check during the demo (the real deliverables)
1. **Fill slippage:** demo fill price vs. the official next-day open, per trade —
   the single most valuable number the demo produces (real fills are worse).
2. **Signal parity:** confirm the symbols the bot picked match a same-day
   recompute on your adjusted data (catches feed/adjustment drift).
3. **Gap-through-stop behaviour** on any down-gap open.
4. **Within-sleeve clustering:** on market-dip days many RSI<10 names fire at
   once — verify the 10-position / 10%-risk caps actually bind.

## 6. Firm-ups to run in parallel (these make the real go-live decision)
1. **Point-in-time S&P 500 constituent backtest, 2000–2024** (kills survivorship
   *and* the liquidity-filter lookahead; spans 2008/2020/2022 bears). Decisive.
2. **Model CFD financing + commissions** in the backtest before trusting demo P&L.
3. **Alpha-vs-SPY regression** — confirm there's timing alpha, not just diluted beta.

Only after (1)–(3) look good should real money follow the demo.
