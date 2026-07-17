#!/usr/bin/env python3
"""Year-by-year COMPOUNDING table for the v2 FX factor at a $5,000 start.

Shows, for each calendar year, the starting balance, the year's return, the
profit in dollars, and the compounded end balance — i.e. what the account
actually does over time as gains reinvest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot import research as R
from trader_bot.factor import FactorStrategy

BASKET = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
          "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "XAUUSD"]


def build_panel():
    closes = {}
    for inst in BASKET:
        try:
            closes[inst] = R.load(inst, "d1")["close"]
        except FileNotFoundError:
            pass
    return pd.DataFrame(closes).sort_index().ffill().dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_equity", type=float, default=5000.0)
    ap.add_argument("--target_vol", type=float, default=0.10)
    ap.add_argument("--max_lev", type=float, default=2.0)
    args = ap.parse_args()

    panel = build_panel()
    strat = FactorStrategy(target_vol=args.target_vol, max_leverage=args.max_lev)
    equity, net, _ = strat.vol_scaled_equity(panel, args.start_equity)
    net = net.dropna()

    # Full calendar years only (drop the 2012 stub and partial 2022).
    yearly_ret = net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly_ret = yearly_ret[(yearly_ret.index >= 2013) & (yearly_ret.index <= 2021)]

    print("=" * 64)
    print(f"  v2 FX Factor — compounding from ${args.start_equity:,.0f}  "
          f"(vol {args.target_vol:.0%}, {args.max_lev:g}x cap)")
    print("=" * 64)
    print(f"  {'Year':<6}{'Start $':>12}{'Return':>9}{'Profit $':>12}{'End $':>12}")
    print("-" * 64)
    bal = args.start_equity
    for y, r in yearly_ret.items():
        start = bal
        bal = bal * (1 + r)
        print(f"  {int(y):<6}{start:>12,.0f}{r*100:>8.1f}%{bal-start:>+12,.0f}{bal:>12,.0f}")
    print("-" * 64)
    n_years = len(yearly_ret)
    cagr = (bal / args.start_equity) ** (1 / n_years) - 1
    best = yearly_ret.max() * 100
    worst = yearly_ret.min() * 100
    print(f"  {n_years} full years: ${args.start_equity:,.0f} -> ${bal:,.0f}"
          f"   ({(bal/args.start_equity-1)*100:+.0f}% total, {cagr*100:+.1f}%/yr compounded)")
    print(f"  Avg year: {yearly_ret.mean()*100:+.1f}%   Best: {best:+.1f}%   "
          f"Worst: {worst:+.1f}%   Positive: {(yearly_ret>0).sum()}/{n_years}")
    print(f"  Typical $ profit/year at this size: "
          f"~${args.start_equity*yearly_ret.mean():,.0f} (grows as it compounds)")
    print("=" * 64)


if __name__ == "__main__":
    main()
