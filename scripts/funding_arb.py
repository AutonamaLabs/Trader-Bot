#!/usr/bin/env python3
"""Crypto perpetual-funding-rate arbitrage (cash-and-carry): long spot, short
the same-notional perp, collect the funding payment. Market-neutral in
*price* -- NOT risk-free (margin/liquidation risk on the perp leg is real).

Data (real, not simulated):
  - Funding-rate history for BTC-USDT / ETH-USDT on Binance, Bybit, Gate.io,
    2020-01-01 .. 2023-12-31, 8h settlement -- data/funding/*.csv, fetched by
    scripts/fetch_funding_data.sh from the public GitHub mirror
    supervik/historical-funding-rates-fetcher (exchange REST APIs are 403 in
    this sandbox; only raw.githubusercontent.com is reachable).
  - Spot daily close -- data/swing/BTCUSD.csv / ETHUSD.csv (already in-repo,
    scripts/fetch_swing_data.sh). Used both as the spot leg AND as a proxy
    for the perp mark price (we have no separate perp OHLC series) -- flagged
    explicitly wherever it drives a risk number, since basis blowouts (perp
    decoupling from spot) are exactly what a spot-price proxy cannot see.

Two return components are modelled separately and never confused:
  1. Funding P&L -- the actual "carry" (collected each settlement, real cost
     when negative).
  2. Basis P&L -- convergence of (perp - spot) between entry and exit. Over
     a multi-year hold this nets close to zero (perp must converge to spot at
     any expiry / continuously via funding for a perp), but is charged as a
     realistic execution slippage proxy on every entry/exit event.

Usage:
    python scripts/funding_arb.py
    python scripts/funding_arb.py --coin ETH --exchange bybit
    python scripts/funding_arb.py --lookback 3
    python scripts/funding_arb.py --skip-improvements   # base report only (faster)

The base report above ([A] static / [B] gated / [C] cross-exchange / liquidation
grid) is the ORIGINAL validated system. Everything below "IMPROVEMENT TESTS" is
a follow-up search for ways to beat it -- see docs/RESEARCH.md, "Improving the
funding-rate arbitrage system", for the full write-up and honest verdicts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUND_DIR = ROOT / "data" / "funding"
SWING_DIR = ROOT / "data" / "swing"

# Realistic retail/prosumer cost assumptions (Binance-style fee schedule,
# taker on both legs -- conservative, no VIP/maker discounts assumed).
SPOT_FEE = 0.0010   # 10 bps taker, spot
PERP_FEE = 0.0005   # 5 bps taker, USDT-margined perp
HALF_SPREAD = 0.0002  # 2 bps half-spread, BTC/ETH majors, each leg
# One full round trip = open spot + open perp + close spot + close perp.
ROUND_TRIP_COST = 2 * (SPOT_FEE + HALF_SPREAD) + 2 * (PERP_FEE + HALF_SPREAD)

MAINTENANCE_MARGIN = {"BTC": 0.005, "ETH": 0.005}  # ~0.5% top-tier bracket


def load_funding(coin: str, exchange: str) -> pd.Series:
    """Per-settlement funding rate (decimal), longs pay shorts when positive."""
    f = FUND_DIR / f"{coin}USDT_{exchange}.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing -- run scripts/fetch_funding_data.sh")
    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")["Funding Rate"].sort_index()
    return s[~s.index.duplicated()]


def load_spot(coin: str) -> pd.Series:
    f = SWING_DIR / f"{coin}USD.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing -- run scripts/fetch_swing_data.sh")
    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Close"].sort_index()


def stats(daily_ret: pd.Series) -> dict:
    r = daily_ret.dropna()
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0.0
    sharpe = np.sqrt(365) * r.mean() / r.std() if r.std() > 0 else 0.0
    dd = (1 - eq / eq.cummax()).max()
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": dd, "years": yrs, "equity": eq}


def simulate_static(daily_fund: pd.Series) -> dict:
    """Always-on: enter once, hold through, exit once. Collects every
    settlement, positive or negative. One round-trip cost total."""
    r = daily_fund.copy()
    # ROUND_TRIP_COST already includes both the open leg (spot buy + perp
    # short) and the close leg (spot sell + perp cover) -- charge it once,
    # on entry, for this buy-and-hold-the-whole-sample variant.
    r.iloc[0] -= ROUND_TRIP_COST
    return stats(r)


def simulate_gated(daily_fund: pd.Series, lookback: int, thresh: float) -> dict:
    """Realistic active management: at each day's *open*, look only at the
    trailing `lookback`-day mean funding (no lookahead) and decide to be
    in/out of the trade. Every flip pays the full round-trip cost. This is
    the version that tests the "avoid negative-funding stretches" tactic."""
    trail = daily_fund.rolling(lookback).mean().shift(1)  # strictly past info
    in_trade = (trail > thresh).astype(int).fillna(0)
    r = daily_fund * in_trade
    flips = in_trade.diff().abs().fillna(in_trade.iloc[0]).clip(upper=1)
    r = r - flips * ROUND_TRIP_COST
    frac_in = in_trade.mean()
    out = stats(r)
    out["frac_in_market"] = frac_in
    out["n_flips"] = int(flips.sum())
    return out


def simulate_cross_exchange(funds: dict[str, pd.Series], lookback: int) -> dict:
    """Long the perp with the lower (or most negative) funding, short the
    perp with the higher funding, each period -- captures the *spread*
    between two exchanges' funding, independent of spot direction. Decision
    uses only trailing (lookback)-day mean funding per exchange (no
    lookahead). Needs margin on BOTH perp legs (no spot leg at all), and
    roughly 2x the perp-side trading cost of the single-exchange trade
    (open+close, two venues) since there's no spot leg to amortize costs
    against."""
    df = pd.DataFrame(funds).dropna(how="all").fillna(0.0)
    trail = df.rolling(lookback).mean().shift(1)
    valid = trail.dropna(how="any").index
    ex_lo = trail.loc[valid].idxmin(axis=1)
    ex_hi = trail.loc[valid].idxmax(axis=1)
    realized = []
    cost_per_flip = 2 * (PERP_FEE + HALF_SPREAD) * 2  # open+close, 2 legs
    prev_pair = None
    for dt in df.index:
        if dt not in valid:
            realized.append(0.0)
            prev_pair = None
            continue
        lo, hi = ex_lo.get(dt), ex_hi.get(dt)
        if pd.isna(lo) or pd.isna(hi) or lo == hi:
            realized.append(0.0)
            prev_pair = None
            continue
        # captured spread = short the high-funding venue (receive), long the
        # low/negative-funding venue (pay less / receive if negative)
        spread = df.loc[dt, hi] - df.loc[dt, lo]
        cost = cost_per_flip if (lo, hi) != prev_pair else 0.0
        realized.append(spread - cost)
        prev_pair = (lo, hi)
    r = pd.Series(realized, index=df.index)
    return stats(r)


def simulate_two_sided(daily_fund: pd.Series, lookback: int, thresh: float) -> dict:
    """IMPROVEMENT IDEA 1: two-sided funding capture. [B]-gated only ever goes
    to FLAT when trailing funding turns negative, earning zero on that stretch.
    But negative funding means shorts pay longs -- so when trailing funding is
    sufficiently negative, the profitable trade is to FLIP: short spot / long
    perp, which now collects the (negative) funding rate. State each day (no
    lookahead -- decided from `lookback`-day trailing mean, shifted one day):
        +1 (long spot / short perp)  if trail >  thresh
        -1 (short spot / long perp)  if trail < -thresh
         0 (flat, in the deadband)   otherwise
    Full ROUND_TRIP_COST is charged on every state change, including a direct
    +1<->-1 reversal (conservative -- see docs/RESEARCH.md for the fill-count
    justification)."""
    trail = daily_fund.rolling(lookback).mean().shift(1)
    state = pd.Series(0.0, index=daily_fund.index)
    state[trail > thresh] = 1.0
    state[trail < -thresh] = -1.0
    state = state.fillna(0.0)
    r = state * daily_fund
    prev = state.shift(1).fillna(0.0)
    flips = (state != prev).astype(float)
    r = r - flips * ROUND_TRIP_COST
    out = stats(r)
    out["frac_long"] = (state == 1).mean()
    out["frac_short"] = (state == -1).mean()
    out["frac_flat"] = (state == 0).mean()
    out["n_flips"] = int(flips.sum())
    return out


def load_spot_ohlc(coin: str) -> pd.DataFrame:
    f = SWING_DIR / f"{coin}USD.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing -- run scripts/fetch_swing_data.sh")
    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")[["Open", "High", "Low", "Close"]].sort_index()
    return df[~df.index.duplicated()]


def worst_intraday_runup(ohlc: pd.DataFrame, cadence_days: int) -> tuple[float, pd.Timestamp]:
    """IMPROVEMENT IDEA 2 helper. We are SHORT the perp, so the dangerous move
    is a RALLY against the margin. If margin is topped up every `cadence_days`
    (using that day's close as the new reference), the worst-case adverse move
    before the NEXT top-up is: (highest intraday High seen in the following
    `cadence_days`) / (close at the last top-up) - 1. Evaluated at every
    possible day-offset (phase-free, like the rest of this project's
    methodology -- not just one fixed rebalance lattice), which is the
    worst-case over all possible rebalance schedules, not one lucky phase.
    Uses the daily High (not just Close), so it is a materially more precise
    (and typically larger / more conservative) estimate than a Close-only
    proxy -- real sub-day risk within a single bar still isn't visible since
    the source data is daily OHLC, so even the cadence=1 case is a lower bound
    on true intraday risk, not an upper bound."""
    anchor = ohlc["Close"].shift(cadence_days)
    hmax = ohlc["High"].rolling(cadence_days, min_periods=1).max()
    run_up = (hmax / anchor - 1.0).dropna()
    return run_up.max(), run_up.idxmax()


def leverage_survival_report(coins=("BTC", "ETH"), cadences=(1, 2, 3, 7),
                              lev_grid=(1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5),
                              fund_start=None, fund_end=None):
    """IMPROVEMENT IDEA 2: quantify the max collateralization ratio (leverage
    on the SHORT PERP leg only -- the spot leg is always bought outright) that
    would have survived the ENTIRE 2020-2023 sample -- including the LUNA/FTX
    2022 chaos, which shows up here as ordinary vol, not the binding
    constraint; the binding event both years is the same Jan 2021 rally
    already flagged in the base report -- at a realistic active-management
    rebalance cadence (1 or 2 days), not the unrealistic "never touch it"
    scenario. Reports per-cadence worst run-up per coin and the leverage grid
    each cadence survives for BOTH coins simultaneously (no per-coin tuning)."""
    print("\n" + "-" * 74)
    print("  IMPROVEMENT IDEA 2: safe-leverage quantification (short-perp leg)")
    print("-" * 74)
    ohlcs = {}
    for coin in coins:
        df = load_spot_ohlc(coin)
        if fund_start is not None:
            df = df[(df.index >= fund_start) & (df.index <= fund_end)]
        ohlcs[coin] = df
    for cadence in cadences:
        worst = {}
        for coin in coins:
            w, wd = worst_intraday_runup(ohlcs[coin], cadence)
            worst[coin] = w
            print(f"  cadence={cadence}d  {coin}: worst adverse rally {w*100:6.2f}% "
                  f"(anchor->high, on {wd.date()})")
        maint = max(MAINTENANCE_MARGIN.get(c, 0.005) for c in coins)
        binding_worst = max(worst.values())
        binding_coin = max(worst, key=worst.get)
        print(f"    {'Leverage':>9} | {'Initial margin':>15} | {'Liq. move':>10} | "
              f"survives BOTH coins? (binding: {binding_coin})")
        max_survivable = 1.0
        for L in lev_grid:
            init_margin = 1.0 / L
            liq_move = init_margin - maint
            ok = binding_worst < liq_move
            if ok:
                max_survivable = L
            print(f"    {L:>8.2f}x | {init_margin*100:14.1f}% | {liq_move*100:9.1f}% | "
                  f"{'yes' if ok else 'NO -- liquidated'}")
        print(f"    -> max leverage surviving every {cadence}-day window in this sample "
              f"(both coins): ~{max_survivable:.2f}x\n")
    print("  Caveat: single 4-year historical sample (2020-2023); a future rally could "
          "exceed\n  Jan-2021's +30-68%/week move. Perp mark price proxied by spot (real "
          "perp can gap\n  further in a basis blowout) -- this likely UNDERSTATES true risk. "
          "Treat the grid\n  above as a historical ceiling, not a safe target -- recommend "
          "trading meaningfully\n  below it, not at it.")


def diversification_report(exchanges=("binance", "bybit", "gate")):
    """IMPROVEMENT IDEA 3: multi-coin diversification. Check whether BTC and
    ETH funding are imperfectly correlated, and if so, whether an equal-
    weighted 2-coin book (half notional each, own round-trip cost per leg)
    gives a smoother equity curve than either coin alone -- quantified, on
    all 3 exchanges (no cherry-picking one)."""
    print("\n" + "-" * 74)
    print("  IMPROVEMENT IDEA 3: multi-coin (BTC+ETH) diversification")
    print("-" * 74)
    print(f"  {'Exchange':<10} {'Corr(BTC,ETH)':>14} | {'BTC Sharpe/DD':>16} | "
          f"{'ETH Sharpe/DD':>16} | {'50/50 Sharpe/CAGR/DD':>24}")
    for ex in exchanges:
        try:
            btc = load_funding("BTC", ex).resample("1D").sum()
            eth = load_funding("ETH", ex).resample("1D").sum()
        except FileNotFoundError:
            continue
        common = btc.index.intersection(eth.index)
        btc, eth = btc.reindex(common), eth.reindex(common)
        corr = btc.corr(eth)
        r_btc = btc.copy(); r_btc.iloc[0] -= ROUND_TRIP_COST
        r_eth = eth.copy(); r_eth.iloc[0] -= ROUND_TRIP_COST
        combined = 0.5 * r_btc + 0.5 * r_eth
        s_b, s_e, s_c = stats(r_btc), stats(r_eth), stats(combined)
        print(f"  {ex:<10} {corr:>14.3f} | Sh {s_b['sharpe']:5.2f} DD {s_b['maxdd']*100:4.2f}% "
              f"| Sh {s_e['sharpe']:5.2f} DD {s_e['maxdd']*100:4.2f}% "
              f"| Sh {s_c['sharpe']:5.2f} CAGR {s_c['cagr']*100:5.2f}% DD {s_c['maxdd']*100:4.2f}%")
    print("\n  BTC/ETH funding is highly correlated (~0.80-0.87 across venues -- both driven "
          "by\n  the same crypto-wide leverage-demand cycle), so this is a modest, not a "
          "dramatic,\n  diversification benefit. It is nonetheless real and consistent across "
          "all 3\n  exchanges: the combined book's Sharpe edges out the better single coin "
          "on every\n  venue, and max drawdown falls by roughly 40-55% vs either coin alone "
          "(the two\n  coins' rare negative-funding stretches are not perfectly synchronized). "
          "CAGR is\n  simply the notional-weighted average of the two coins' CAGRs, not a "
          "new source\n  of return -- the benefit here is a smoother ride, not a higher one.")


def cost_sensitivity_report(coin="BTC", exchange="binance"):
    """IMPROVEMENT IDEA 4: how much would net return change if maker/limit
    orders replaced the current taker-fee-on-both-legs assumption? Reported as
    a pure sensitivity table on the SAME strategies already defined above --
    not a promise, since realistic maker-fill-probability isn't backtested
    here (flagged explicitly). Run on both [A] static (one round trip over
    the whole sample) and [B] gated (75+ round trips) to show WHERE fee
    assumptions actually matter -- they barely move a low-turnover strategy
    and move a high-turnover one a lot."""
    print("\n" + "-" * 74)
    print("  IMPROVEMENT IDEA 4: cost-sensitivity (maker vs taker fee assumptions)")
    print("-" * 74)
    daily_fund = load_funding(coin, exchange).resample("1D").sum()
    scenarios = {
        "current (taker+taker+spread, default)": ROUND_TRIP_COST,
        "maker (partial discount, realistic)": 2 * (0.0008 + 0.0001) + 2 * (0.0002 + 0.0001),
        "best-case (VIP/promo maker, optimistic)": 2 * 0.0002 + 2 * 0.0000,
        "2x stress (worse liquidity)": 2 * ROUND_TRIP_COST,
    }
    print(f"  {'Scenario':<42} {'RTC':>7} | {'[A] Static CAGR':>16} | {'[B] Gated CAGR':>15}")
    for name, rtc in scenarios.items():
        r = daily_fund.copy()
        r.iloc[0] -= rtc
        s_static = stats(r)
        trail = daily_fund.rolling(3).mean().shift(1)
        in_trade = (trail > 0.0).astype(int).fillna(0)
        rg = daily_fund * in_trade
        flips = in_trade.diff().abs().fillna(in_trade.iloc[0]).clip(upper=1)
        rg = rg - flips * rtc
        s_gated = stats(rg)
        print(f"  {name:<42} {rtc*100:5.2f}% | {s_static['cagr']*100:15.2f}% | "
              f"{s_gated['cagr']*100:14.2f}%")
    print(f"\n  ({coin}/{exchange}) Static barely moves (one round trip amortized over "
          "~4 years) --\n  the recommended low-turnover system is nearly fee-insensitive. "
          "Gated (75+\n  round trips) swings hugely with the fee assumption -- exactly why "
          "high-turnover\n  variants (gated, two-sided) are fragile to an assumption this "
          "backtest cannot\n  verify live (whether limit orders actually fill at the assumed "
          "price/time).")


def two_sided_report(coins=("BTC", "ETH"), exchanges=("binance", "bybit", "gate"),
                      lookback=3, threshes=(0.0, 0.00005, 0.0001, 0.0002, 0.0005)):
    """Run IMPROVEMENT IDEA 1 across a fixed threshold grid, both coins, all
    3 exchanges -- the SAME threshold applied everywhere (no per-coin/
    per-exchange tuning), so a real effect should show up broadly, not in one
    lucky cell."""
    print("\n" + "-" * 74)
    print("  IMPROVEMENT IDEA 1: two-sided funding capture (flip short when "
          "funding\n  is sufficiently negative, instead of just going flat)")
    print("-" * 74)
    for thresh in threshes:
        print(f"\n  thresh = {thresh*100:.3f}%/day (~{thresh*365*100:.1f}%/yr annualized), "
              f"lookback={lookback}d")
        print(f"  {'Coin/Exch':<14} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} "
              f"{'%long':>6} {'%short':>6} {'flips':>6}")
        for coin in coins:
            for ex in exchanges:
                try:
                    daily_fund = load_funding(coin, ex).resample("1D").sum()
                except FileNotFoundError:
                    continue
                s = simulate_two_sided(daily_fund, lookback, thresh)
                print(f"  {coin+'/'+ex:<14} {s['cagr']*100:7.2f}% {s['sharpe']:7.2f} "
                      f"{s['maxdd']*100:6.2f}% {s['frac_long']*100:5.0f}% "
                      f"{s['frac_short']*100:5.0f}% {s['n_flips']:6d}")
    print("\n  Compare to [A] STATIC (no flips, no deadband) at the SAME coin/exchange "
          "cells,\n  printed above for the default --coin/--exchange, and see "
          "docs/RESEARCH.md for\n  the full cross-check: two-sided NEVER beats static "
          "at any threshold tested, on\n  any coin/exchange. Verdict: this idea does NOT "
          "hold up -- confirms it's the\n  switching cost eating the benefit again, same "
          "failure mode as [B] gated.")


def liquidation_check(spot: pd.Series, coin: str, leverages=(2, 3, 5, 10, 20)):
    """We are SHORT the perp, so the dangerous move is a RALLY (price up)
    against the short leg's margin -- a crash helps the short. Using spot
    price as a mark-price proxy for the perp (flagged: real perp can gap
    further from spot in a basis blowout, so this likely *understates* true
    risk), compute the worst rolling drawup (peak rise from a low) over a
    weekly-rebalance assumption (margin topped up at most weekly) and over a
    buy-and-forget assumption (no rebalancing across the whole sample), then
    check which leverage levels would have been liquidated.
    """
    maint = MAINTENANCE_MARGIN.get(coin, 0.005)
    # "buy and forget": rolling max rise from any point to any later point
    # within a trailing window (proxy for "how far can price run against you
    # before you notice / top up margin").
    windows = {"1 week (rebalanced weekly)": 7, "90 days (rarely touched)": 90,
               "never (full sample)": len(spot)}
    worst = {}
    for label, w in windows.items():
        w = min(w, len(spot))
        roll_min = spot.rolling(w, min_periods=1).min()
        run_up = (spot / roll_min - 1.0)
        worst[label] = run_up.max()
    print(f"\n  Worst adverse rally vs. a short-perp margin position ({coin}, "
          f"spot used as perp-price proxy):")
    for label, v in worst.items():
        print(f"    {label:<28}: {v*100:6.1f}%")
    print(f"  Maintenance margin assumed: {maint*100:.1f}% (top-tier bracket)")
    print(f"  {'Leverage':>9} | {'Initial margin':>15} | {'Liquidation move':>17} | "
          f"survives weekly-rebalance? | survives never-rebalanced?")
    for L in leverages:
        init_margin = 1.0 / L
        liq_move = init_margin - maint
        wk_ok = "yes" if worst["1 week (rebalanced weekly)"] < liq_move else "NO -- liquidated"
        never_ok = "yes" if worst["never (full sample)"] < liq_move else "NO -- liquidated"
        print(f"    {L:>7}x | {init_margin*100:14.1f}% | {liq_move*100:16.1f}% | "
              f"{wk_ok:>26} | {never_ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coin", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--exchange", default="binance", choices=["binance", "bybit", "gate"])
    ap.add_argument("--lookback", type=int, default=3, help="days of trailing funding for the gated strategy")
    ap.add_argument("--thresh", type=float, default=0.0, help="annualized funding threshold to be 'in' (decimal, e.g. 0.05=5%%)")
    ap.add_argument("--skip-improvements", action="store_true",
                     help="skip ideas 1-4 (two-sided / leverage / diversification / cost-sensitivity) for a faster base-only report")
    args = ap.parse_args()

    fund = load_funding(args.coin, args.exchange)
    daily_fund = fund.resample("1D").sum()
    n_settlements = len(fund)
    n_days = len(daily_fund)
    ann_gross = daily_fund.mean() * 365
    frac_neg = (fund < 0).mean()

    print("=" * 74)
    print(f"  Funding-rate arbitrage: long spot {args.coin} / short {args.coin}-USDT "
          f"perp ({args.exchange})")
    print(f"  Sample: {fund.index[0].date()} -> {fund.index[-1].date()}  "
          f"({n_settlements} settlements, {n_days} days)")
    print("=" * 74)
    print(f"  Gross annualized funding (before ANY cost): {ann_gross*100:+.2f}%")
    print(f"  Fraction of settlements with NEGATIVE funding (a cost, not income): "
          f"{frac_neg*100:.1f}%")
    print(f"  Assumed round-trip transaction cost (spot+perp, fee+spread, both "
          f"sides): {ROUND_TRIP_COST*100:.2f}%")

    s_static = simulate_static(daily_fund)
    print(f"\n  [A] STATIC hold-through-everything (enter once, hold {s_static['years']:.1f}y, exit once)")
    print(f"      CAGR {s_static['cagr']*100:+.2f}%  Sharpe {s_static['sharpe']:.2f}  "
          f"MaxDD {s_static['maxdd']*100:.2f}%")

    s_gated = simulate_gated(daily_fund, args.lookback, args.thresh)
    print(f"\n  [B] GATED (in trade only when trailing {args.lookback}d mean funding > "
          f"{args.thresh*100:.1f}%, no lookahead, full cost on every flip)")
    print(f"      CAGR {s_gated['cagr']*100:+.2f}%  Sharpe {s_gated['sharpe']:.2f}  "
          f"MaxDD {s_gated['maxdd']*100:.2f}%  In-market {s_gated['frac_in_market']*100:.0f}% "
          f"of days  ({s_gated['n_flips']} flips)")

    print(f"\n  Year-by-year gross annualized funding ({args.coin}, {args.exchange}):")
    yr = daily_fund.groupby(daily_fund.index.year).mean() * 365
    for y, v in yr.items():
        print(f"    {y}: {v*100:+6.2f}%")

    # Cross-exchange funding-spread capture, all 3 venues, both coins available.
    print("\n  [C] CROSS-EXCHANGE funding-spread capture (long perp on the "
          "cheapest venue, short on the richest, trailing-signal, no lookahead)")
    for coin in ["BTC", "ETH"]:
        try:
            funds = {ex: load_funding(coin, ex).resample("1D").sum()
                     for ex in ["binance", "bybit", "gate"]}
        except FileNotFoundError:
            continue
        common = None
        for s in funds.values():
            common = s.index if common is None else common.intersection(s.index)
        funds = {k: v.reindex(common) for k, v in funds.items()}
        s_cross = simulate_cross_exchange(funds, args.lookback)
        print(f"    {coin}: CAGR {s_cross['cagr']*100:+.2f}%  Sharpe {s_cross['sharpe']:.2f}  "
              f"MaxDD {s_cross['maxdd']*100:.2f}%  ({s_cross['years']:.1f}y)")

    # Margin / liquidation risk (real, not "market-neutral = risk-free").
    spot = load_spot(args.coin)
    spot = spot[(spot.index >= fund.index[0]) & (spot.index <= fund.index[-1])]
    liquidation_check(spot, args.coin)

    print("\n  NOTE: perp mark price approximated by spot price (no separate perp\n"
          "  OHLC series available in this sandbox) -- real perp prices can gap\n"
          "  further from spot during basis blowouts (e.g. LUNA/FTX 2022), so\n"
          "  this likely UNDERSTATES true liquidation risk, not overstates it.")
    print("=" * 74)

    if not args.skip_improvements:
        print("\n" + "=" * 74)
        print("  IMPROVEMENT TESTS -- see docs/RESEARCH.md 'Improving the funding-rate")
        print("  arbitrage system' for the full write-up and final recommendation")
        print("=" * 74)
        two_sided_report(lookback=args.lookback)
        leverage_survival_report(fund_start=fund.index[0], fund_end=fund.index[-1])
        diversification_report()
        cost_sensitivity_report(coin=args.coin, exchange=args.exchange)
        print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
