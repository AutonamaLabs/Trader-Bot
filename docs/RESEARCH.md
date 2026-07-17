# Research log: hunting a real, out-of-sample edge

This documents the honest search for a profitable, *consistent* edge on real FX
data — what failed, what survived, and the strategy we ended up with. The
guiding rule: **an edge must survive out-of-sample and across instruments, or it
isn't an edge.**

## Data

Real MetaTrader H1/H4/D1 history (2012–2022) for 11 instruments — EUR/USD,
GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, EUR/JPY, GBP/JPY, EUR/GBP, AUD/JPY,
XAU/USD — from the public [ejtraderLabs/historical-data](https://github.com/ejtraderLabs/historical-data)
mirror. Reproduce with `scripts/fetch_data.sh`. Prices are point-scaled and
auto-normalised on load; costs (spread + slippage) are charged on every fill.

## Methodology

- **In-sample / out-of-sample split:** optimise on 2013–2018, validate untouched
  on 2019–2022.
- **Cross-instrument pooling:** a parameter set must work across a *majority* of
  the 11 instruments (breadth), not one lucky symbol.
- **Walk-forward:** the finalist is checked year-by-year (2013–2021) with **no
  per-year fitting** — it must be positive in most years.
- **Cost stress:** re-run at 2× costs; an edge that dies at 2× costs is too thin.

## What we tested and what happened

### 1. Single-instrument directional / mean-reversion (H4 and D1)

| Family | Timeframe | In-sample | Out-of-sample | Verdict |
|---|---|---|---|---|
| Donchian breakout / Turtle | H4 | +0.06R, PF 1.10 | **−0.04R, PF 0.92** | dies OOS |
| EMA crossover | D1 | +0.42R, PF 1.52 | **−0.66R, PF 0.26** | severe overfit |
| Time-series momentum | D1 | +0.12R | **−0.28R** | dies OOS |
| Donchian + chandelier (bot v1) | H4 | ~0R | **−0.05R** | no edge |
| Bollinger mean-reversion | D1 | ~0R | +0.04R (thin) | regime-dependent |
| RSI-2 mean-reversion | D1 | ~0R | +0.02R, PF 1.15 | **failed walk-forward** |

The RSI-2 mean-reversion looked promising in the single OOS split, but
walk-forward exposed it: **positive in only 4/9 years and 5/11 instruments** — a
coin flip. Trend/momentum was strong pre-2018 and collapsed 2019–2022: a *regime*,
not an edge. **Conclusion: no simple outright edge on these majors survives.**
This is expected — liquid FX arbitrages these away.

### 2. Cross-sectional (market-neutral) factors — the survivor

Instead of betting on direction, rank the basket each day and hold a
**dollar-neutral long/short book**. Being long-and-short nets out the common USD
move and isolates *relative value*, which is more robust.

| Sleeve | Lookback / Hold | Sharpe | Positive years |
|---|---|---|---|
| Momentum (long winners / short losers) | 20d / 20d | 0.30 | **8/9** |
| Momentum | 20d / 20d, k=2 | 0.29 | 6/9 |
| Reversal (long losers / short winners) | 20d / 5d | 0.15 | 7/9 |
| Reversal | 10d / 5d | 0.10 | 5/9 |

Momentum and reversal are **negatively correlated (−0.61)** — they hedge each
other. Blending them at equal risk gives the finalist.

## The edge (Trader-Bot v2)

**FX cross-sectional factor ensemble** — a dollar-neutral daily long/short book
blending medium-term momentum and short-term reversal across an 11-instrument
basket.

| Metric (2013–2022, net of costs) | Ensemble | Vol-targeted 10% (2× lev cap) |
|---|---|---|
| Sharpe | **0.44** | 0.45 |
| CAGR | +2.4% | **+4.0%** |
| Max drawdown | **8.4%** | 14.2% |
| Positive years | **8/9** | 8/10 |
| Survives 2× costs | yes (Sharpe 0.29) | yes |
| Stable to dropping any instrument | yes | yes |

Why it's credible: it is **market-neutral (regime-independent), positive across
most years and instruments, robust to costs and basket choice, and uses no
per-period fitting.** It is *not* a magic money machine — Sharpe ~0.45 and a
mid-single-digit CAGR at conservative sizing is a genuine but *modest* edge.
That is what a real, honest FX edge looks like.

## Specification

- **Strategy:** dollar-neutral cross-sectional long/short. Each day, rank the
  basket by trailing return; go long the top *k* and short the bottom *k*,
  inverse-vol weighted. Blend four sleeves (2× momentum 20/20, reversal 20/5,
  reversal 10/5) at equal risk. Size the whole book to a 10% annual vol target,
  leverage capped at 2×.
- **Instruments:** EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF,
  EUR/JPY, GBP/JPY, EUR/GBP, AUD/JPY, XAU/USD (robust to subsets).
- **Timeframe:** daily bars.
- **Trading hours:** one decision per day at the daily close (NY 17:00 ET ≈
  21:00–22:00 UTC). Momentum leg rebalances every 20 trading days, the reversal
  legs every 5 — so turnover and costs stay low.

Run it: `python scripts/run_factor.py`. Reproduce the search:
`python scripts/research.py --tf d1`, `python scripts/walkforward.py`,
`python scripts/cross_sectional.py --sweep`.

## ⚠️ Audit & correction (important)

A subsequent quant review (and `scripts/audit_phasefree.py`) found the
originally-reported ensemble Sharpe **~0.45 was inflated by two biases**:

1. **Selection bias** — the four sleeves were chosen from a full-sample grid
   sweep (`cross_sectional.py --sweep`).
2. **Rebalance-phase luck** — a 20-day-hold block has 20 possible start offsets.
   Across all 20, the momentum sleeve's Sharpe averages **−0.15** (range −0.50 to
   +0.30); the reported result had landed on a lucky phase.

Rebuilding every sleeve **phase-free** (overlapping daily tranches = average of
all offsets) gives the honest picture:

| Sleeve (phase-free) | Sharpe | Positive years |
|---|---|---|
| Momentum 20/20 | **−0.19** | 3/9 |
| Reversal 20/5 | **+0.23** | 7/9 |
| Reversal 10/5 | +0.05 | 3/9 |

So **cross-sectional momentum does not work** on this FX basket once phase luck is
removed, and blending it with reversal (they are −0.78 correlated) cancels the
edge (ensemble ≈ +0.06). The only survivor is a **weak short-term reversal**
tendency (Sharpe ~0.23) — and it is **fragile**: dropping USD/CHF alone cuts it
to +0.07, and the leverage-timed, tradeable version is roughly flat over
2013–2021. **Net: the v2 edge is marginal at best, not the 4%/yr originally
shown.** This is the same hard lesson as v1, found one level deeper — and the
reason the audit (phase-free construction + drop-one + truly-unseen data) is now
mandatory before believing any backtest here.

Legitimate paths that could produce a *real* edge (not yet built — blocked on
data in this environment): a **carry sleeve** from policy-rate differentials
(needs FRED/rate data), universe breadth, currency-netting to cut financing
drag, and — above all — validating a frozen system on **2023–2025** data.

## Honest limitations

- Sharpe ~0.45 is real but modest; expect losing years (2017 −12.5% at 10% vol).
- Vol-targeting can raise leverage into a low-vol lull right before a shock — the
  2× cap limits but does not remove this.
- Broker-time vs UTC offset in the source data slightly affects the daily-bar
  boundary; align to your broker's daily close for live use.
- Live execution needs a dollar-neutral, low-turnover rebalancer and borrow for
  shorts (all instruments here are CFD/margin-shortable at retail FX brokers).
- Next steps: add a carry sleeve (needs rate data), a value sleeve (PPP/real-rate),
  and de-correlate further; each additional robust, low-correlation sleeve is the
  legitimate way to push Sharpe up.
