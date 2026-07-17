#!/usr/bin/env python3
"""Backtest the FX cross-sectional factor strategy (v2) on real daily data,
sized for a $5,000 account with volatility targeting.

    python scripts/run_factor.py
    python scripts/run_factor.py --target_vol 0.15 --max_lev 3
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
from trader_bot.factor import FactorStrategy, performance, per_year

BASKET = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
          "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "XAUUSD"]


def build_panel(tf="d1"):
    closes = {}
    for inst in BASKET:
        try:
            closes[inst] = R.load(inst, tf)["close"]
        except FileNotFoundError:
            pass
    return pd.DataFrame(closes).sort_index().ffill().dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_equity", type=float, default=5000.0)
    ap.add_argument("--target_vol", type=float, default=0.10)
    ap.add_argument("--max_lev", type=float, default=2.0)
    args = ap.parse_args()

    panel = build_panel("d1")
    strat = FactorStrategy(target_vol=args.target_vol, max_leverage=args.max_lev)
    equity, net, lev = strat.vol_scaled_equity(panel, args.start_equity)
    m = performance(equity, net)

    print("=" * 60)
    print("  FX Cross-Sectional Factor (v2) — Real-Data Backtest")
    print("=" * 60)
    print(f"  Basket            : {len(panel.columns)} instruments (daily)")
    print(f"  Period            : {equity.dropna().index[0].date()} -> {equity.dropna().index[-1].date()}")
    print(f"  Vol target / maxLev: {args.target_vol:.0%} / {args.max_lev:g}x")
    print(f"  Starting equity   : ${args.start_equity:,.2f}")
    print(f"  Final equity      : ${m['final']:,.2f}")
    print(f"  CAGR              : {m['cagr']*100:+.2f}%")
    print(f"  Realised vol      : {m['vol']*100:.1f}%")
    print(f"  Max drawdown      : {m['max_dd']*100:.1f}%")
    print(f"  Sharpe            : {m['sharpe']:.2f}")
    print("-" * 60)
    print("  Return by year:")
    yr = per_year(net)
    for y, v in yr.items():
        bar = "+" if v > 0 else "-"
        print(f"    {int(y)}: {v*100:+6.1f}%  {bar}")
    posy = int((yr[yr.index >= 2013] > 0).sum())
    print(f"  Positive years    : {posy}/{len(yr[yr.index >= 2013])}")
    print("=" * 60)

    book = strat.target_book_today(panel, m["final"])
    active = book[book.abs() > 1]
    print("\n  Today's target book ($ notional, +long / -short):")
    for inst, notional in active.items():
        print(f"    {inst:<8} {notional:>+12,.0f}")

    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    equity.to_csv(outdir / "factor_equity.csv", header=["equity"])
    print(f"\n  Wrote {outdir/'factor_equity.csv'}")


if __name__ == "__main__":
    main()
