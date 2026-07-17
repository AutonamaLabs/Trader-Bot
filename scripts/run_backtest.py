#!/usr/bin/env python3
"""Run the H4-Donchian-Trend backtest.

Usage:
    python scripts/run_backtest.py                 # 8y synthetic data, default config
    python scripts/run_backtest.py --years 10 --seed 42
    python scripts/run_backtest.py --data-dir data # load real <PAIR>_H4.csv files

Outputs a console report and writes results/ (equity curve + trades).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot.backtest import Backtester
from trader_bot.config import load_config
from trader_bot.data import generate_synthetic, load_csv
from trader_bot.features import build_features
from trader_bot.metrics import compute_metrics, format_report, per_pair_summary


def build_data(cfg, args):
    """Build enriched features per pair. Indicators are computed on the FULL
    history (so warmup is real) and only THEN sliced to [start, end], keeping
    the first bars of the test window fully warmed up."""
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    data = {}
    for pair in cfg.pairs:
        if args.data_dir:
            path = Path(args.data_dir) / f"{pair}_H4.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing data file: {path}")
            raw = load_csv(path)
        else:
            raw = generate_synthetic(pair, years=args.years, seed=args.seed)
        feats = build_features(raw, cfg)
        if start is not None:
            feats = feats[feats.index >= start]
        if end is not None:
            feats = feats[feats.index <= end]
        data[pair] = feats
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="H4-Donchian-Trend backtester")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--data-dir", default=None, help="dir with <PAIR>_H4.csv real data")
    ap.add_argument("--years", type=float, default=8.0, help="synthetic history length")
    ap.add_argument("--seed", type=int, default=7, help="synthetic RNG seed")
    ap.add_argument("--start", default=None, help="backtest start date (YYYY-MM-DD, UTC)")
    ap.add_argument("--end", default=None, help="backtest end date (YYYY-MM-DD, UTC)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Loading data for {cfg.pairs} ...")
    data = build_data(cfg, args)
    total_bars = sum(len(df) for df in data.values())
    print(f"Loaded {total_bars:,} H4 bars across {len(data)} pairs. Running backtest...\n")

    bt = Backtester(cfg, data)
    result = bt.run()
    m = compute_metrics(result)

    print(format_report(result, m))
    pp = per_pair_summary(result)
    if not pp.empty:
        print("\nPer-pair breakdown:")
        print(pp.to_string())

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    result.equity_curve.to_csv(outdir / "equity_curve.csv", header=["equity"])
    pd.DataFrame([t.__dict__ for t in result.trades]).to_csv(
        outdir / "trades.csv", index=False
    )
    print(f"\nWrote {outdir/'equity_curve.csv'} and {outdir/'trades.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
