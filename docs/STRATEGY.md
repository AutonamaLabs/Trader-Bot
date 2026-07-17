# Trader-Bot Strategy Specification v1.0

**Strategy name:** `H4-Donchian-Trend` — Multi-timeframe trend-following breakout with ATR risk management
**Account:** $5,000 USD retail, OANDA-style REST broker, spread-only pricing assumed
**Status:** Backtest-first. No live deployment until backtest + walk-forward criteria (Section 10) are met.

> Masterminded with the Fable model; implemented in this repository. This file is the single source of truth — every value maps to `config.yaml`.

---

## 1. Core Thesis

**Primary edge: time-series momentum (trend-following) on major FX pairs, entered via volatility breakout, held with an asymmetric trailing exit.**

Why this edge and not another:

- **It persists for structural reasons.** FX trends are driven by multi-month monetary-policy divergence (rate-hiking vs rate-cutting central banks), current-account flows, and the behavioral under-reaction of large, slow-moving participants (corporates hedging, central banks smoothing, asset managers rebalancing). These actors are not profit-maximizing on the trend timescale, so the inefficiency isn't arbitraged away. Time-series momentum is one of the few effects documented across decades and asset classes (Moskowitz/Ooi/Pedersen 2012 and the entire CTA industry).
- **It has positive skew.** Trend-following loses small and often, wins big and rarely. For a small account, positive skew + hard stops is far safer than mean-reversion's negative skew (win often, occasionally get destroyed). A $5k account cannot survive a fat left tail; it can survive a 40% win rate.
- **It is simple enough to be robust.** Donchian breakouts have ~3 real parameters. Fewer parameters = less curve-fitting = smaller gap between backtest and live.

What we deliberately do **not** do: no martingale, no grid, no averaging down, no counter-trend scalping, no news trading. One edge, executed consistently.

---

## 2. Instruments

Ranked by spread, liquidity, trend cleanliness, and USD-account pip-value simplicity.

| Rank | Pair | Typical spread | Trade? | Notes |
|---|---|---|---|---|
| 1 | EUR/USD | 0.6–1.0 | **Yes** | Deepest liquidity, cleanest fills |
| 2 | USD/JPY | 0.7–1.2 | **Yes** | Strong policy-divergence trender |
| 3 | GBP/USD | 0.9–1.5 | **Yes** | Trends well; correlated with EUR/USD |
| 4 | AUD/USD | 0.9–1.4 | **Yes** | Commodity/China beta; diversifies EUR block |
| 5 | USD/CAD | 1.2–1.8 | Phase 2 | Oil-driven; add after live validation |
| 6 | USD/CHF | 1.2–2.0 | No | ~-0.95 correlated to EUR/USD |
| 7 | EUR/GBP | 1.0–1.8 | No | Ranges chronically |
| 8 | NZD/USD | 1.5–2.5 | No | Redundant with AUD/USD, wider spread |

**Trade universe v1: EUR/USD, GBP/USD, USD/JPY, AUD/USD.**

---

## 3. Timeframes

- **Signal timeframe: H4** (UTC-aligned 00/04/08/12/16/20). All decisions on **closed** bars.
- **Higher-timeframe filter: D1** (regime direction & slope).
- H4 is slow enough that a 1-pip spread is <4% of a ~30–60 pip stop, fast enough for 3–6 trades/week across 4 pairs.

---

## 4. Trading Hours / Sessions

Positions are **held around the clock** (stops always working). Only **entries** are session-gated:

- **Entry window: 07:00–17:00 UTC** (London open → NY morning).
- **No new entries:** 20:00–01:00 UTC (rollover), Friday after 15:00 UTC, Sunday open until 01:00 UTC Monday, Dec 20–Jan 3, US Thanksgiving Thu/Fri, and inside the news blackout (Section 9).
- Open positions are never force-closed for session reasons; weekend gap risk is bounded by position sizing.

---

## 5. Indicators

| Indicator | TF | Params | Purpose |
|---|---|---|---|
| Donchian Channel | H4 | 20 (excl. current bar) | Entry trigger: breakout of ~3.3-day extreme |
| EMA | D1 | 50 | Regime/trend direction filter |
| ATR (Wilder) | H4 | 14 | Stops, trail, vol filter, sizing |
| ADX (Wilder) | H4 | 14 | Trend-strength filter |
| ATR median | H4 | 100 | Volatility-regime baseline |

No RSI/MACD/Bollinger — every added indicator is another curve-fitting axis.

---

## 6. Entry Rules

Evaluated once per H4 close, per pair. `close_D1` = last **completed** daily bar (no lookahead).

**Common filters (ALL must hold):**
```
F1: no open position in this pair
F2: open_positions_total < MAX_CONCURRENT (3)
F3: currency_exposure_ok(pair)
F4: entry_window: 07:00 <= utc_hour(next_bar_open) < 17:00
F5: not friday_after_1500 and not holiday_blackout and not news_blackout
F6: ADX_H4(14) > 20
F7: 0.6*ATR_median100 <= ATR_H4(14) <= 2.5*ATR_median100
F8: spread <= MAX_SPREAD[pair] and spread < 5% of stop distance
F9: daily_loss_ok and weekly_loss_ok
```

**LONG:** `close_H4 > donchian_high_20` AND `close_D1 > EMA50_D1` AND `EMA50_D1 rising (vs 5 days ago)`
**SHORT:** mirror.
Enter at market on the next H4 bar open. Failed-filter signals are discarded, not queued.

---

## 7. Exit Rules

`R = 2.0 × ATR_at_entry` (in price). Stops are resting broker-side orders.

1. **Initial stop:** 2.0×ATR from entry.
2. **Partial TP:** at +1.5R, close 50% at market.
3. **Breakeven:** on partial fill, move stop to entry + 0.1R.
4. **Chandelier trail (remainder):** on each H4 close, `stop = max(stop, highest_high − 3.0×ATR)` (long) / mirror (short). Ratchets only in favour.
5. **Time stop:** if after 30 H4 bars the trade never reached +1R, close at market.
6. **Regime-flip exit:** if D1 filter flips against the position for 2 consecutive daily closes, close.
7. No opposite-signal reversal in v1.

Ambiguous bar (touches both stop and target): **stop fills first** (conservative).

---

## 8. Risk Management

- **Per-trade risk: 0.5% of equity** (never >1.0%). A 10-loss streak draws down ~5%.
- **Sizing:** `units = floor( equity*0.005 / (stop_pips * pip_value) )`, capped at notional ≤ 4× equity.
- **Max concurrent: 3** (aggregate open risk ≤ 1.5%). **Max per pair: 1. Max per currency: 2.**
- **Correlation:** EUR/USD & GBP/USD never held same-direction together.
- **Circuit breakers:** daily loss ≥2% → pause to next day; weekly ≥4% → pause to Monday; drawdown from HWM ≥15% → **halt, human review**.
- **Sizing equity floored at HWM×0.85** (anti-death-spiral).

---

## 9. Filters To Avoid Bad Trades

1. **Trend filter** — D1 EMA50 level + slope.
2. **Trend-strength** — ADX(14) > 20.
3. **Volatility** — ATR within [0.6×, 2.5×] of its 100-bar median.
4. **Spread** — reject if spread > per-pair cap or > 5% of stop distance.
5. **News** — block entries 60 min before → 30 min after high-impact events for either currency; always block FOMC, NFP, US CPI, ECB/BoE/BoJ/RBA decisions. Open positions are not closed for news.
6. **Rollover/Friday/holiday windows** (Section 4).

---

## 10. Expected Profile (honest)

| Metric | Expectation |
|---|---|
| Win rate | 35–45% |
| Avg winner | ~+1.8R blended |
| Avg loser | ~−0.85R |
| Expectancy | ~+0.15 to +0.30R / trade |
| Frequency | 3–6 entries/week (~15–25/month) |
| Return goal | **8–20%/year** on conservative sizing |
| Max drawdown | **Plan for 10–15%**; losing streaks of 8–12 trades are certain |

**Go-live gate:** ≥8y H4 data (incl. 2019 chop + 2022 trend), spread + 0.5-pip slippage, stop-fills-first; require profit factor ≥1.25, max DD ≤20%, positive expectancy in ≥3 of 4 pairs, walk-forward (3y train/1y test rolling) not collapsing to ≤0; then ≥3 months on a practice account before real funds.

---

## 11. Parameter Table

See `config.yaml` — every key there corresponds 1:1 to a value in this spec.
