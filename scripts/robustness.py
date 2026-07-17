#!/usr/bin/env python3
"""Consistency / robustness harness.

Runs the backtest across many independent synthetic histories (different RNG
seeds) and reports the DISTRIBUTION of outcomes. A single good backtest can be
luck; a strategy that stays positive across dozens of randomly-generated market
histories is showing genuine, repeatable edge — that is what "consistent" means.

Usage:
    python scripts/robustness.py --runs 40 --years 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot.backtest import Backtester
from trader_bot.config import load_config
from trader_bot.data import generate_synthetic
from trader_bot.features import build_features
from trader_bot.metrics import compute_metrics


def one_run(cfg, years: float, seed: int) -> dict:
    data = {p: build_features(generate_synthetic(p, years=years, seed=seed), cfg)
            for p in cfg.pairs}
    result = Backtester(cfg, data).run()
    m = compute_metrics(result)
    return {
        "seed": seed,
        "return_pct": m.total_return_pct,
        "cagr_pct": m.cagr_pct,
        "max_dd_pct": m.max_drawdown_pct,
        "profit_factor": m.profit_factor,
        "expectancy_r": m.expectancy_r,
        "win_rate": m.win_rate * 100,
        "sharpe": m.sharpe,
        "trades": m.n_trades,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = []
    for i in range(args.runs):
        rows.append(one_run(cfg, args.years, seed=1000 + i))
        print(f"  run {i+1}/{args.runs}  seed={1000+i}  "
              f"return={rows[-1]['return_pct']:+.1f}%  "
              f"DD={rows[-1]['max_dd_pct']:.1f}%  PF={rows[-1]['profit_factor']:.2f}")

    df = pd.DataFrame(rows)
    pf = df["profit_factor"].replace([np.inf], np.nan)

    print("\n" + "=" * 64)
    print(f"  ROBUSTNESS SUMMARY over {args.runs} random {args.years:g}-year histories")
    print("=" * 64)

    def stat(name, col, fmt="{:.2f}"):
        s = df[col]
        print(f"  {name:<18} median {fmt.format(s.median()):>8}   "
              f"[p10 {fmt.format(s.quantile(.1))}, p90 {fmt.format(s.quantile(.9))}]")

    stat("Total return %", "return_pct", "{:+.1f}")
    stat("CAGR %", "cagr_pct", "{:+.1f}")
    stat("Max drawdown %", "max_dd_pct", "{:.1f}")
    stat("Profit factor", "profit_factor")
    stat("Expectancy (R)", "expectancy_r", "{:+.3f}")
    stat("Win rate %", "win_rate", "{:.1f}")
    stat("Sharpe", "sharpe")
    stat("Trades", "trades", "{:.0f}")

    pct_profitable = (df["return_pct"] > 0).mean() * 100
    pct_pf125 = (pf > 1.25).mean() * 100
    worst_dd = df["max_dd_pct"].max()
    print("-" * 64)
    print(f"  Profitable histories     : {pct_profitable:.0f}%")
    print(f"  Profit factor > 1.25     : {pct_pf125:.0f}%")
    print(f"  Worst drawdown observed  : {worst_dd:.1f}%")
    print("=" * 64)

    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "robustness.csv", index=False)
    print(f"\nWrote {outdir/'robustness.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
