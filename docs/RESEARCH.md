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

## Follow-up: "go faster" and "cross-sectional on fresh data" (both negative)

Two further avenues were tested rigorously; both failed.

**Intraday / H1 (go faster).** Phase-free cross-sectional reversal at hourly
horizons, net of a realistic 1.2bp/side cost (`scripts/intraday.py`): Sharpe
**−9.4** at 1h hold, still negative at 12h. Retail transaction costs annihilate
fast strategies. Confirmed dead.

**Cross-sectional G10 currency factor on ECB daily data 1999–2026**
(`scripts/cross_ccy.py`, an independent and *current* dataset — AUD/CAD/CHF/GBP/
JPY/NOK/NZD/SEK vs EUR). This is the cleanest out-of-sample test available:

| Sleeve (phase-free) | 2013–2019 | 2020–2026 (unseen) | 2023–2026 (unseen) |
|---|---|---|---|
| Reversal 20/5 | Sharpe +0.44 (5/7 yrs) | **+0.01 (3/7)** | **−0.02 (2/4)** |
| Reversal 10/5 | −0.16 | +0.11 | +0.18 (1/4) |
| Momentum 12-month | −0.31 | ~0.00 | −0.01 |

The reversal edge that made 2013–2019 look good **decayed to ~zero in
2020–2026** — the classic signature of a data-mined pattern, not a persistent
edge. Momentum is inconsistent across eras. **On real data through 2026, across
timeframes, instruments, and both single-name and cross-sectional structures, no
technical edge survives out-of-sample at retail costs.**

**Carry** — the one structurally-motivated factor not yet tested — needs policy-
/interest-rate data, which is blocked in this environment (FRED and broker APIs
are unreachable; only raw GitHub files are). It remains the single most promising
untested idea and is the reason a real broker/data feed is required to go
further.

## BREAKTHROUGH: a real edge exists — but in EQUITIES, not FX (TPS-5)

After FX/crypto/gold all failed, the strategy that finally survives is a
**trend-pullback swing on equity indices and stocks** (Fable's "TPS-5",
`scripts/swing_tps5.py` / `swing_portfolio.py`): buy an RSI(3) dip while price is
above its 200-day MA and volatility is calm; exit into strength (RSI>65), on a
regime break (Close<SMA200), a 3×ATR stop, or a 10-day time stop. Long-only.

**Survivorship-free proof it's a genuine signal** — single index ETFs, no
universe selection:

| Instrument | Win rate | Profit factor | Per-trade | Span |
|---|---|---|---|---|
| SPY | 73% | 1.77 | +0.15R | 2000–2026 |
| QQQ | 71% | 1.82 | +0.14R | 1999–2019 |

SPY is positive in every era including 2020–2026. This RSI-pullback effect on
equity indices is documented since the 1990s and is driven by institutional
dip-buying / vol-selling flow — it does **not** exist on FX majors (which is
exactly why every FX test failed).

**Scaled across a 48 liquid large-cap universe** (`swing_portfolio.py`, one
compounding account, ≤10 concurrent, 1% risk/trade, 20% notional cap, 2004–2024):

| | Value |
|---|---|
| $5,000 → | **$40,827** (20.9y) |
| CAGR | **+10.6%** |
| Sharpe | 0.80 |
| Max drawdown | 26% |
| Monthly mean / median | +0.89% / +1.09% |
| Positive months | 66% |

### Honest caveats (these matter a lot)
1. **Survivorship bias:** the 48 names are *today's* winners; buying dips in
   stocks that became mega-caps flatters the result. The true edge is lower than
   +10.6%/yr. The survivorship-free SPY/QQQ test is the trustworthy core.
2. **Regime-dependent:** disjoint sub-periods — 2004–10 Sharpe 0.52, **2011–17
   Sharpe 1.57 (+23.7%/yr)**, **2018–24 Sharpe 0.30 (+3.4%/yr)**. It shines in
   steady bull markets and struggles in crash/chop (2008, 2020, 2022). The great
   full-sample number is carried by 2011–2017.
3. **Saturates ~1%/month:** raising per-trade risk 1%→3% does *not* raise
   returns (drawdown cap binds) — CAGR tops out ~10–11%. **This cannot be levered
   to 10%/month without abandoning the risk controls that keep it alive.**

**Bottom line:** a genuine, survivable swing edge worth ~high-single-digit to
low-double-digit %/yr at ~25% max drawdown, best in bull regimes — real, and the
first thing in this whole search that survives out-of-sample. Just not 10%/month.

### Quality tuning — raising profit factor (`scripts/swing_sweep.py`)

A grid sweep over entry depth / stop / exit / down-streak (ranked by profit
factor, judged by *consistency* across the grid, not a single cell) found two
principled, monotonic improvements — deeper oversold entry and exit at the mean:

| Config | Portfolio PF | Sharpe | Max DD | SPY PF | QQQ PF |
|---|---|---|---|---|---|
| Original (RSI<15, exit RSI>65) | 1.26 | 0.80 | 26% | 1.77 | 1.82 |
| **Tuned (RSI<10, exit RSI>50)** | **1.44** | **0.96** | **16%** | **2.04** | **2.12** |

Confirmed on the survivorship-free indices (SPY/QQQ PF ~1.8 → ~2.0+), so it's a
real quality gain, not a stock-selection artifact. Sized up to match the old
return, the tuned config gives **CAGR ~11.9%, Sharpe 0.96, PF 1.40, DD ~22%** —
strictly better than the original on every axis. The down-streak filter added
nothing (redundant with deep RSI). These are now the canonical defaults.

Remaining soft spot: the 2018–2024 sub-period is still weak (PF ~1.12) — a
crash-regime/exposure overlay is the next lever, but the blanket SPY>200MA gate
cut drawdown without lifting PF, so a more surgical filter is needed.

### Exit management: trailing / scale-in / scale-out (`scripts/swing_exits.py`)

Because TPS-5 is **mean-reversion**, trend-following exit tools were tested and
mostly backfire — the edge is the snap-back to the mean, so holding for a "runner"
gives profit back:

| Variant | PF | Win% | Sharpe | DD | verdict |
|---|---|---|---|---|---|
| base (exit at mean, RSI>50) | 1.44 | 67% | **0.98** | 15% | — |
| trailing stop (let winners run) | 1.30 | 49% | 0.71 | 20% | **hurts** |
| trail after target | 1.32 | 50% | 0.74 | 21% | **hurts** |
| scale-OUT (partial + trail rest) | 1.37 | 65% | 0.87 | 16% | slightly hurts |
| **scale-IN on further weakness** | **1.46** | 68% | 0.97 | **14%** | small help |

Answers to the three exit questions:
1. **Trailing stops to let winners run — no.** Avg win rises ($49→$100+) but win
   rate collapses (67%→50%); net PF and Sharpe fall, drawdown rises. Mean-reversion
   winners don't trend; the tight exit-at-the-mean is already near-optimal.
2. **Scale in as a trade goes your way — no** (that adds at *worse* reversion
   odds). But scaling in on **further weakness** (deeper oversold, one add at
   RSI<5) is the mean-reversion-consistent version and marginally helps:
   PF 1.44→1.46, CAGR +0.7pts, DD 15%→14%.
3. **Scale out (bank part, let rest run) — slightly no**; the trailed remainder
   gives back gains (Sharpe 0.98→0.87).

Takeaway: trailing/pyramiding/partials are a *trend-following* toolkit; on a
mean-reversion system they subtract. "Letting winners run" requires a different
(trend/breakout) strategy — which we already showed is dead on the instruments
and timeframes available here.

## Two-sleeve system: mean-reversion + TREND (`scripts/trend_sleeve.py`)

"Letting winners run" is a *trend-following* tool, so we built the trend sleeve
it belongs to and paired it with mean-reversion. Trend sleeve = long-only
Donchian breakout in a confirmed uptrend, ridden with a wide chandelier ATR
trailing stop (here the trail *helps* — positive skew, 43% win rate, 2.7:1
winners):

| Sleeve (2004–2024, 48 stocks) | PF | Win% | CAGR | Sharpe | Max DD |
|---|---|---|---|---|---|
| Trend (Donchian-200, 3×ATR stop, 6×ATR trail) | **2.00** | 43% | +19.0% | 0.98 | 28% |
| Mean-reversion (TPS-5, tuned + scale-in) | 1.47 | 68% | +9.4% | 0.98 | 16% |
| **50/50 combined book** | — | — | **+14.0%** | **1.12** | **16%** |

The combined Sharpe (1.12) beats either sleeve alone: they are only +0.43
correlated and complementary across regimes — trend is strongest in **2018–2024
(Sharpe 0.83)** exactly where mean-reversion is weakest (0.30), and vice-versa in
choppy years. The trend sleeve is positive in every sub-period (2004–10 Sharpe
0.77, 2011–17 1.29, 2018–24 0.83).

**Caveat (important):** both sleeves run on today's 48 large-cap winners, so
**survivorship bias inflates results — and it inflates the TREND sleeve more**
(riding multi-year winners like AAPL/NVDA is exactly what a winners-only universe
supplies). Mean-reversion was validated survivorship-free on SPY/QQQ; the trend
sleeve has no equivalent bias-free check yet (indices don't "break out" like
single stocks). Treat the trend numbers as an upper bound until re-run on a
survivorship-free universe incl. delisted names. Both sleeves are also long-only
equities — a market crash hits both (trailing/fast exits cap it at ~16–28% DD).

## Survivorship-bias reckoning (`scripts/validate_universe.py`)

The two-sleeve numbers above use survivor universes (today's live tickers), which
flatters everything. Re-tested on a genuinely **survivorship-free** universe (680
names incl. delisted, point-in-time; teddykoker dataset) over the same 2013–2018
window:

| Universe (2013–2018) | Combo Sharpe | Combo CAGR | Max DD |
|---|---|---|---|
| 48 winners | 1.76 | +25% | 10% |
| 110 broad survivors | 1.69 | +28% | 14% |
| Survivorship-FREE, all 545 | **0.34** | +4% | 21% |
| Survivorship-FREE, liquid ≥ $20 | 0.39 | +5% | — |
| Survivorship-FREE, liquid ≥ $50 | **0.58** | +8.6% | — |

**Conclusion: survivorship bias inflated the Sharpe from ~0.5 to ~1.7.** The edge
is real but MUCH smaller than the survivor backtests implied — on bias-free,
liquid, quality names it's ~Sharpe 0.5–0.6 / ~8% CAGR, not the headline 1.1. The
strategy genuinely works better on higher-quality (higher-priced, liquid) names,
and catches falling-knives/delistings in the full junk-inclusive universe. Any
live expectation must be set from the survivorship-free liquid number (~0.5
Sharpe), not the survivor backtest. Caveat on the caveat: this bias-free window is
only ~4 years (2013–2018, a bull market); a point-in-time S&P 500 constituent
test would sharpen it further.

## Point-in-time S&P 500 validation (2000-2024)

The 2013-2018 survivorship-free result above is the strongest evidence so far
that the MR sleeve is real, but a Fable review flagged the obvious gap: it's a
single ~4-year window that never touches a real bear market, and it uses
*today's* index membership even for the survivorship-free universe (a name
that IPO'd in 2019 shouldn't be tradable in 2013). This section re-tests
2000–2024 — spanning the 2008 GFC, the 2020 COVID crash, and the 2022 bear —
using **point-in-time S&P 500 constituents**: a symbol is only a legal NEW
entry on dates it was an actual index member, never in hindsight.

**Sourcing (sandbox network only reaches raw.githubusercontent.com / github.com
git clone; Yahoo, Stooq, FRED, Quandl/Nasdaq Data Link all return 403):**

- **Membership** — [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents),
  a daily point-in-time constituent list, 1996-01-02 to present, one row per
  membership-change date. Reachable directly via raw.githubusercontent.com.
  This is genuine, not reconstructed — 1,006 distinct tickers were S&P 500
  members at some point between 2000 and 2024.
- **Prices** — no free, fully delisted-inclusive daily-OHLC source for ~1,000
  tickers back to 2000 was reachable (Quandl's WIKI Prices, the standard free
  answer to this exact problem, was killed in 2018 and its raw CSV isn't
  mirrored anywhere reachable here). The best available compromise, spliced
  from two sources per symbol:
  - **2000 → 2017-11-10:** a GitHub mirror of the Kaggle "Huge Stock Market
    Dataset" ([neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset](https://github.com/neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset),
    itself sourced from Stooq's bulk EOD archive). Split-adjusted, not
    dividend-adjusted, ~7,000 tickers, **and it does carry many since-delisted
    names** (e.g. Wachovia-era ticker WB, Circuit-City-era ticker CC) that a
    current-constituents-only source would miss.
  - **2017-11-10 → 2024-12-06:** `data/universe_broad/` (already in this repo,
    dividend+split adjusted), for the subset of names still trading under the
    same ticker.
  - Where both exist, the pre-2017 leg is rescaled by one constant factor so
    the switchover day carries ~0% fabricated return (like a back-adjusted
    continuous futures contract) — see `scripts/prepare_pit_universe.py`.
  - **710 of the 1,006 point-in-time tickers** were recoverable this way; 296
    were not (mostly bankruptcy-suffix tickers like AAMRQ/ABKFQ/BTUUQ/CCTYQ —
    genuine total losses the Stooq mirror itself doesn't carry either — plus
    names that took their current ticker only after the 2017 cutoff, e.g.
    BKNG, CARR, AMCR).
  - **Known limitation — ticker recycling:** a free, symbol-only dataset can't
    fully rule out a ticker being reused by an unrelated company decades
    later (WB and CC above are themselves examples: today's WB is Weibo, not
    Wachovia; today's CC is a different, more recent listing). This affects
    an unknown but likely small share of the 710 matched names.

Reproduce: `scripts/fetch_pit_universe.sh` → `scripts/prepare_pit_universe.py`
→ `scripts/validate_pit_universe.py --risk 0.01` (adds a point-in-time
membership gate on top of `scripts/swing_sweep.py`'s engine; open positions
are allowed to run to their normal exit even if the name leaves the index
mid-trade, and a position is force-closed at its last available price if the
underlying data source's coverage ends mid-trade — only 11 of 5,703 trades,
0.2%, hit that branch).

**MR sleeve only** (risk_per_trade=0.01, `rsi_lo=10, sl_atr=3.0, rsi_exit=50`
— the trend sleeve isn't included here; it has no equivalent point-in-time
check yet):

| Period | CAGR | Sharpe | Max DD | Profit factor | Win rate | Trades |
|---|---|---|---|---|---|---|
| **Full 2000–2024** | **+9.3%** | **0.76** | 25% | 1.22 | 63% | 5,703 |
| 2000–2007 | +13.2% | 0.98 | 25% | 1.28 | 64% | 2,324 |
| 2008–2009 (GFC) | +5.4% | 0.58 | 14% | 1.24 | 62% | 310 |
| 2010–2019 | +11.0% | 0.79 | 25% | 1.24 | 63% | 2,632 |
| 2020 (COVID) | −0.1% | 0.01 | 6% | 0.99 | 55% | 65 |
| 2021–2022 (bear) | +2.4% | 0.41 | 6% | 1.16 | 57% | 183 |
| 2023–2024 | +0.8% | 0.17 | 7% | 1.05 | 60% | 189 |

**This still overstates the edge — and we can prove it.** Re-running the exact
same point-in-time universe restricted to the original teddykoker window
(2013-06-01 → 2018-02-27) gives MR-only Sharpe **0.90**, CAGR +13.8%. But the
true survivorship-free (teddykoker, genuinely delisted-inclusive) universe on
that identical window gives MR-only Sharpe **0.25**, CAGR +2.9% (re-run here
for a clean apples-to-apples MR-only number; the previously-published 0.34 was
the 50/50 trend+MR combo, which is higher because the two sleeves diversify
each other). That's a >3x Sharpe gap on the *same dates, same strategy,
same point-in-time membership gating* — the only difference is the price
source. **Conclusion: fixing the membership-timing lookahead is necessary but
not sufficient.** The Stooq/HSMD mirror is missing exactly the true
bankruptcies (the Q-suffix tickers) that the CRSP-style teddykoker dataset
keeps, so this composite universe is still meaningfully survivorship-biased —
just less obviously than a naive "today's constituents" backtest.

**Reading the table honestly, then:**

- Treat the 2000–2024 numbers above as an **upper bound**, not the true
  answer. The true Sharpe is very likely closer to the previously-established
  survivorship-free range (0.25–0.58) than to 0.76, and given the demonstrated
  gap, plausibly below the low end of that range for the post-2018 years
  specifically.
- The edge's shape across regimes is nonetheless informative even at reduced
  confidence: it works reasonably well pre-2018 in this (upper-bound) test,
  including a positive Sharpe through the 2008 GFC, and **degrades sharply
  from 2020 onward** — COVID is a wash (Sharpe 0.01) and 2021–2024 stay weak
  (0.17–0.41) even before the residual survivorship-bias correction above is
  applied. Once that correction is applied, 2020–2024 performance is a
  realistic candidate for **net-negative**, not just weak.
- The 2020–2024 readings carry a second, compounding caveat: only 103 of the
  710 priced tickers (the ones present in `data/universe_broad/`) have any
  price data past 2017-11-10, so those sub-periods trade a much narrower,
  more concentrated, large-cap-survivor-tilted universe than 2000–2017 does —
  both less diversified and more exposed to the exact bias this section set
  out to remove.
- A liquidity cut (drop symbols with all-time median close < $20, mirroring
  the earlier ≥$20/≥$50 cuts) does **not** rescue this: full-period Sharpe
  0.73 (barely changed), but 2008–2009 Sharpe drops to 0.19 and 2023–2024
  turns negative (Sharpe −0.28, PF 0.88) — unlike the 2013-2018 study, a
  liquidity filter does not reliably improve results here.

**Bottom line: the edge does not clearly survive point-in-time validation
across real bear markets.** It looks fine through the 2008 GFC in this
upper-bound test, but the already-weak 2020–2024 numbers are almost certainly
overstated by residual survivorship bias in the only reachable free price
source, and the true recent performance is a plausible candidate for flat or
negative. This does not overturn the 2013-2018 survivorship-free finding — it
sharpens it in the direction that finding already pointed: real-account
expectations should anchor on the low end of the 0.25–0.58 Sharpe range
established earlier, not on the more optimistic multi-decade number in the
table above. A genuinely delisted-inclusive, dividend-adjusted daily-OHLC
dataset for the full ~1,000-name universe back to 2000 (CRSP, Norgate, or
equivalent paid data) is needed to close this out properly; it was not
obtainable for free in this sandbox.

## Crypto funding-rate arbitrage — feasibility and backtest

A fundamentally different edge type was requested: **market-neutral** crypto
cash-and-carry (long spot BTC/ETH, short the same-notional perpetual future,
collect the funding payment). Unlike everything above, this does not bet on
price direction — so it was worth testing on its own terms, but "market-neutral"
turns out **not** to mean "risk-free," which is the main finding.

### Data

Exchange REST APIs (Binance `fapi`, Bybit, OKX, Deribit, CoinGecko) all
returned `403`/connection-refused from this sandbox on a fresh direct test —
confirming the standing constraint that only `raw.githubusercontent.com` (and,
it turns out, the agent's own web-search/fetch tools) are reachable. Real
historical funding-rate data was located and verified on the public GitHub
mirror
[supervik/historical-funding-rates-fetcher](https://github.com/supervik/historical-funding-rates-fetcher),
which scraped each exchange's own funding-history endpoint:

- **BTC-USDT and ETH-USDT funding-rate history**, 8h settlements, **Binance,
  Bybit, Gate.io**, 2020-01-01 → 2023-12-31 (Bybit from 2021-01-01) —
  `scripts/fetch_funding_data.sh` → `data/funding/`.
- **Spot daily close**, reused from `data/swing/BTCUSD.csv` / `ETHUSD.csv`
  (already in-repo). Also used as a *proxy* for perp mark price for the
  liquidation-risk check, since no separate perp OHLC series is reachable
  here — flagged explicitly below, and it likely **understates** true risk
  (real perp prices can gap further from spot in a basis blowout).

This is a real 4-year sample spanning very different regimes: the 2020 COVID
crash, the 2020–21 bull run, the 2022 bear market (Terra/LUNA May 2022, FTX
collapse Nov 2022), and the 2023 recovery — exactly the regime diversity this
project's methodology requires. `scripts/funding_arb.py` runs the full
analysis; `scripts/fetch_funding_data.sh` reproduces the data pull.

### Gross funding — the headline number, before any cost

| | 2020 | 2021 | 2022 | 2023 | Full sample (gross, ann.) |
|---|---|---|---|---|---|
| BTC (Binance) | +17.2% | +30.6% | +4.2% | +7.9% | **+15.0%** |
| ETH (Binance) | +27.4% | +37.5% | +0.8% | +8.3% | **+18.5%** |

Consistent across Binance/Bybit/Gate.io (13.7–18.5% annualized gross). This
roughly matches the ~11% "baseline" figure quoted going in — funding is real
income, most of the time. But note the collapse in 2022: **the bear-market
year paid ~10x less than the bull-market year**, and 12–19% of individual
8-hourly settlements were outright **negative** (a cost, not income) depending
on exchange/coin. Funding is not a fixed rate; it is itself a volatile,
regime-dependent quantity.

### Net-of-cost backtest (`scripts/funding_arb.py`)

Costs modelled: 10bp spot taker fee, 5bp perp taker fee, 2bp half-spread per
leg — a realistic round-trip (open spot + open perp + close spot + close
perp) costs **0.38%** of notional.

| Strategy | BTC (Binance) CAGR | Sharpe* | MaxDD |
|---|---|---|---|
| **[A] Static** — enter once, hold the whole 4y, exit once | **+16.0%** | 10.8 | 1.5% |
| **[B] Gated** — exit when trailing 3d funding turns negative (no lookahead) | +8.5% | 3.7 | 7.8% |
| **[C] Cross-exchange spread capture** — long the cheap venue's perp, short the rich one's, trailing signal | **−27.8%** | −10.9 | 62% |

*Sharpe here is computed only on the realized funding P&L stream (the
whole point of the trade is that price risk is hedged out) — it is **not**
comparable to a normal strategy Sharpe, because it excludes the tail/margin
risk that is real and is analyzed separately below. Treat these Sharpe
numbers as "how smooth funding income looks day to day," not "how safe this
trade is."

Two results are the important, somewhat counter-intuitive ones:

1. **Trying to be clever about avoiding negative-funding stretches loses
   money.** [B] underperforms simply buying-and-holding-the-carry [A] in
   *every* coin/exchange tested (CAGR roughly halved, drawdown up 5–15x) —
   negative-funding periods in this sample are short, scattered blips, not
   sustained regimes, so a trailing-signal exit whipsaws in and out and pays
   the 0.38% round-trip cost repeatedly for little benefit. The cheapest,
   best-performing version of this trade is the boring one: put it on and
   leave it on.
2. **Cross-exchange funding-spread capture — the strategy web research
   flagged at up to ~29% annualized — is net NEGATIVE once real costs are
   included**, and this holds at every rebalance speed tested (1-day lookback
   through 90-day): CAGR ranges from **−42% (1-day signal) up to roughly flat
   at best (~60–90 day signal), never durably positive**. Decomposing it: the
   *gross* captured spread between exchanges averages only ~7.5%/yr, but a
   responsive signal switches which exchange to be long/short on **~40% of
   days**, and each switch costs ~0.28% (two perp legs, open+close) — turnover
   alone costs ~40%/yr at a 3-day lookback, dwarfing the tiny spread being
   chased. This is a direct, data-backed rebuttal of the optimistic headline
   number: **the ~29% figure is a gross, pre-cost artifact; after realistic
   transaction costs it does not clear zero in this sample**, consistent with
   the ~40%-of-opportunities-net-positive caveat flagged going in — if
   anything, this test found conditions worse than that.

### Liquidation risk — the part "market-neutral" doesn't mean "risk-free"

The short perp leg needs margin, and margin can be wiped out by a sharp
**rally** (a crash *helps* a short perp — the danger is the price running
away from you, not crashing). Using spot price as a mark-price proxy (real
perp prices can gap further in a basis blowout, so this is a floor on the
risk, not a ceiling), the worst adverse move against a short-perp margin
position in this real 2020–2023 sample:

| Rebalance discipline | BTC worst weekly rally | ETH worst weekly rally |
|---|---|---|
| Weekly (margin topped up every 7 days) | 34.0% | **67.8%** |
| Never rebalanced across the sample | 1,259% | 4,251% |

The ETH number is not a data artifact — it is the real **Jan 2-9, 2021 move**
(ETH $730 → $1,225, +68% in one week; BTC did $24.6k → $40.8k, +66%, the same
week). At standard retail leverage:

| Leverage on perp leg | Initial margin | Survives BTC weekly-rebalanced? | Survives ETH weekly-rebalanced? |
|---|---|---|---|
| 2× | 50% | **yes** | **no — liquidated** |
| 3× | 33% | no — liquidated | no — liquidated |
| 5×+ | ≤20% | no — liquidated | no — liquidated |

**Only fully (or near-fully) margining the perp leg (≈1×, i.e. posting margin
roughly equal to the spot notional) survives every historical week in this
sample without a margin call.** That is the honest risk-adjusted picture:
funding arb is delta-hedged against *slow* price drift, but a real single
week of the kind that has actually happened twice in this 4-year sample can
liquidate a leveraged short-perp leg even while the paired spot position is
fine — because the two legs typically sit in separate wallets/accounts with
separate margin, not one netted portfolio, at most retail venues. This is the
same failure mode that hit real funds running basis trades in 2022 (the
funding compression in that table above coincided with exactly this kind of
dislocation risk). **This is not a hypothetical caveat — it is a real,
recurring feature of this exact 4-year sample.**

### Verdict: does it beat a bank savings rate?

| Version of the trade | Realistic net return | Risk |
|---|---|---|
| Fully collateralized (≈1× on perp leg, no liquidation risk found in-sample) | **~half the gross rate** once you account for capital tied up as idle margin buffer, i.e. roughly **7–10%/yr** on total capital deployed, with 2022-style years as low as ~2–4% | Genuinely low *liquidation* risk (survives every week tested); still carries counterparty/exchange risk (hacks, freezes, insolvency — not FDIC-insured), and funding itself can go structurally negative for extended stretches in future bear markets |
| Moderately levered (2–3× on perp leg, the "attractive" version) | Headline ~14–20%/yr gross-ish, but **would have been liquidated at 3×+ leverage during a real week in this sample (Jan 2021)**, and even 2× fails on ETH | Not survivable at realistic historical stress with meaningful leverage — this is the version that "looks like free money" and is the one that actually blows up |
| Naive active timing / cross-exchange spread-chasing | **Negative** net of turnover costs | Loses money AND carries the same margin risk, for nothing |

Against today's ~4–5% bank savings benchmark: **the fully-collateralized,
survivable version of this trade is a real, modest, genuine edge that clears
the bar most years (~7–10%/yr vs 4–5%) — but by a much smaller margin than
the headline 11–29% figures suggest, it is not risk-free, it had at least one
year (2022) where it barely cleared or matched the bank rate, and any
leverage aggressive enough to meaningfully beat the bank rate by a wide margin
would have been liquidated at least once in this exact real 4-year sample.**
Same conclusion this project keeps re-deriving: the honest, survivable version
of an edge is real but smaller and more fragile than the number that gets
quoted first.

### Honest limitations

- Perp mark price is proxied by spot close (no separate perp price series
  reachable here); real basis-blowout risk during panics is therefore
  understated, not overstated.
- Only Binance/Bybit/Gate.io and only BTC/ETH were available; smaller-cap
  perps (which the original research pointers suggested for larger funding
  spreads) could not be tested — this sandbox could not reach a source for
  them.
- 4 years (2020–2023) is a genuinely diverse but still limited sample; it
  contains exactly one violent multi-week rally in each coin, which is enough
  to show the liquidation risk is real but not enough to bound its worst case.
- Exchange counterparty/custody risk (the single biggest real-world driver of
  crypto-carry-fund losses in 2022, e.g. FTX) is not modelled at all here —
  it is a separate, non-market risk on top of everything above.

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

## Diversified multi-asset trend-following (CTA-style)

The FX-only trend attempts above (v1 Donchian, v2 momentum) all died — but that
tested breakout/momentum on 2-3 correlated FX pairs, which is not how real CTAs
(AHL, Winton, etc.) make money. Hurst/Ooi/Pedersen (AQR, 2017, "A Century of
Evidence on Trend-Following Investing") show a **diversified** time-series
momentum book — long/short every market by its own trailing trend, vol-scaled so
no market dominates, averaged across dozens of weakly-correlated markets and
FOUR asset classes — was positive in every decade back to 1880 across 67
markets. The claimed mechanism is **breadth**, not any one market's edge. This
section builds the real test: `scripts/trend_multi_asset.py`.

### Data (`scripts/fetch_multi_asset.sh`)

Same discipline as every other dataset here: WebSearch for candidate GitHub
mirrors, `curl -sS -o /dev/null -w "%{http_code}"` to verify every URL before
trusting it (Yahoo/Stooq/FRED/broker APIs are still 403 in this sandbox —
re-verified, not just assumed). **37 markets, 4 asset classes:**

| Asset class | Markets (n) | Source | Span |
|---|---|---|---|
| FX | AUD/CAD/CHF/GBP/JPY/NOK/NZD/SEK vs EUR + UUP (USD index) (9) | ECB daily reference rates (`data/research/ecb_daily.csv`, already in-repo) + scienclick/stocks | 1999–2026 (UUP: 2007–2017) |
| Equity | SPY, QQQ, IWM, DIA, EFA, EEM, EWJ, EWG, EWU, EWQ, EWY (11) | already-in-repo SPY/QQQ + scienclick/stocks (Kaggle "Huge Stock Market Dataset" mirror, same family as `fetch_pit_universe.sh`'s source) | SPY 2000–2025; rest 2005–2017 |
| Commodity | XAUUSD, WTI, Brent, natural gas, silver, copper, corn, wheat, soybeans, sugar, coffee, cotton (12) | already-in-repo XAUUSD + `datasets/oil-prices`, `datasets/natural-gas` (live to 2026) + scienclick/stocks | WTI 1986–2026, NatGas 1997–2026, rest 2005/2007–2017 |
| Bond | TLT, IEF, SHY, BWX, IGOV (5) | scienclick/stocks | **2005–2017 only — no reachable source extends bonds past 2017 from this sandbox** (verified: FRED, Treasury, Stooq all 403; no working single-country Bund/Gilt/JGB CSV found on GitHub either) |

**Honest caveat on breadth:** most of the equity/commodity/bond legs are ETF
proxies from a dataset frozen at 2017-11-10, not continuously-adjusted futures
(what real CTAs trade) — no roll yield, real financing cost, or futures margin
efficiency is captured, and many "markets" here are correlated developed-market
cap-weighted ETFs (SPY/QQQ/IWM/DIA/EFA/EWG/EWJ/EWU/EWQ all move together in a
crash) rather than truly independent bets. 37 nominal markets is genuine
progress over 2-3 FX pairs, but it is **not** AQR's 67-market, futures-native,
137-year universe — this is a partial, proxy-based replication, reported as
such.

### Methodology

Per market: `signal = sign(trailing 12-month return)`; `weight = signal ×
(target_vol / ex-ante vol)` (EWM vol, 60-day halflife, annualised) — every
market individually scaled to the same vol, so averaging is automatically
equal-risk. Two-level combination: equal-weight markets within an asset class,
then equal-weight the four (or three) class indices — equal risk **within and
across** asset classes, as specified. Rebalanced monthly, weight decided at
month-end close applied starting the *next* trading day only (no lookahead).
Costs: 1bp FX, 2bp equity/bond, 6bp commodity (round-trip, charged on every
rebalance-day weight change) — cheap for futures-like FX, wider for ETP
proxies. One data-cleaning note: WTI printed **-$36.98 on 2020-04-20** (real
contract-expiry mechanics, not a market move a trend system should "catch") —
treated as a bad print (carried forward from the last valid price), not
clipped to a small positive number (which would fake an enormous return the
day price recovered).

Because the bond data stops in 2017, two books are reported: **CORE** (all 4
classes, common window bond-limited to 2005–2017) and **EXTENDED** (drop
bonds, common window 1999–2026 via ECB FX + SPY + WTI/Brent/NatGas). Robustness
checked across 12-month vs 1/3/12-month-blend signals × monthly vs weekly
rebalance (4 combinations, all reported, no cherry-picking) on the EXTENDED
book.

### Results

| Book | Window | Raw Sharpe | Vol-targeted 12% CAGR | maxDD |
|---|---|---|---|---|
| CORE (4 classes, incl. bonds) | 2005–2017 (12.7y) | **0.09** | +0.54% | 19.4% |
| EXTENDED (3 classes, no bonds) | 1999–2026 (27.0y) | **0.24** | +2.38% | 21.6% |

Per-class Sharpe (EXTENDED): equity **0.35**, commodity 0.07, fx 0.06 — nearly
all the edge is coming from the equity trend leg (consistent with the
already-validated equity trend sleeve elsewhere in this repo), FX and
commodities are close to flat. CORE book adds bonds at Sharpe **-0.24** (a drag,
not a diversifier, over 2005–2017). Asset-class correlations are genuinely low
(0.01–0.22 pairwise) — the diversification premise holds structurally — but
weak per-class Sharpes mean low correlation isn't rescuing the total.

**Robustness (EXTENDED book, all 4 lookback×rebalance combos, unscaled):**

| Lookback | Rebalance | Sharpe |
|---|---|---|
| 12-month | Monthly (selected) | 0.24 |
| 12-month | Weekly | 0.14 |
| 1/3/12-month blend | Monthly | 0.16 |
| 1/3/12-month blend | Weekly | 0.08 |

All four are positive but modest — no wild swings from the parameter choice
(not an overfit-looking result), but also no combination gets anywhere close to
AQR's reported Sharpe.

**Crisis / sub-period behaviour (EXTENDED book):**

| Period | Return | Sharpe | maxDD | Verdict |
|---|---|---|---|---|
| 2008 GFC (Jun'07–Jun'09) | +2.2% | **+0.31** | 6.9% | crisis alpha confirmed — trend caught the slow multi-month grind down |
| 2020 COVID (Jan–Jun'20) | **-4.8%** | **-0.34** | 6.2% | **failed** — the crash was too fast (V-shaped); by the time 12-month momentum flipped short, the rebound had already started, and the strategy got whipsawed on both legs |
| 2022 stock+bond selloff | +0.8% | +0.14 | 6.2% | modestly positive — trend caught the slower 2022 grind |

This is an important, non-cherry-picked nuance on trend-following's "crisis
alpha" reputation: it worked in the two *slow* crises (2008, 2022) and failed
in the one *fast* one (2020) — exactly the documented real-world criticism of
trend systems (many actual CTAs also had a mixed 2020 despite their crisis-alpha
brand). Correlation to a naive 60/40 SPY/TLT benchmark over the CORE window is
**-0.03** — genuinely uncorrelated, which is the one part of the CTA thesis that
holds up cleanly here.

### Verdict: does it clear a 4-5% bank-rate bar?

**No.** Even the best-looking configuration (EXTENDED book, vol-targeted to
12% annualised, 2× leverage cap) produces CAGR **+2.38%** at Sharpe **0.24** —
below a plain savings account, and increasing leverage further to chase a
higher CAGR just scales the 21.6% drawdown proportionally (a weak-Sharpe book
does not become a strong one by adding leverage). The CORE (bond-inclusive) book
is worse still: CAGR +0.54%, Sharpe 0.09 — essentially noise.

This is also **weaker than the already-validated equity two-sleeve system**
in this repo (survivorship-free: Sharpe 0.34–0.58, CAGR 4–8.6%, PF ~1.19–1.47)
on every axis. The honest reasons: (1) the equity/commodity/bond legs here are
mostly ETF proxies frozen at 2017, not continuously-linked futures, so real
CTA cost/roll efficiencies aren't captured; (2) 37 correlated-ETF markets is not
67 genuinely-independent futures markets — the AQR breadth thesis needs more,
and more truly independent, markets than this sandbox's reachable data sources
supply; (3) a plain 12-month-momentum-with-inverse-vol-sizing signal is the
simplest form of the strategy — no attempt was made to squeeze more out of it
(matching the "don't overfit" constraint), so this is a lower bound on how well
a *carefully engineered* CTA implementation could do, not a ceiling.

**Bottom line: real breadth (4 asset classes, uncorrelated legs, -0.03
correlation to 60/40) was assembled and honestly tested, and the classic
"trend does well in slow crashes" pattern shows up in 2008/2022 — but the
risk-adjusted return (Sharpe 0.09-0.24) is too weak, after realistic costs and
with the data actually reachable from this sandbox, to beat a savings account
or to rival this repo's existing validated equity edge. If a live futures/data
feed with a genuinely broad (60+), long-history, continuously-rolled contract
universe becomes available, this methodology is worth re-testing — the
structural low-correlation result is real and the method is directionally
correct; the data available here is simply too narrow and too proxy-heavy to
prove the full CTA case.
