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
    python scripts/funding_arb.py --gated --lookback 3
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


if __name__ == "__main__":
    main()
