#!/usr/bin/env python3
"""Run the two-sleeve system on a given universe/window and report MR / trend /
combo stats + correlation. Used to compare survivor-only vs survivorship-free.

    python scripts/validate_universe.py --dir data/universe_broad --start 2006-01-01
    python scripts/validate_universe.py --dir data/universe_sf/data --start 2013-06-01 --end 2018-02-27
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import trend_sleeve as T
import swing_sweep as SW


def stats(ret):
    eqc = (1 + ret).cumprod()
    dd = (1 - eqc / eqc.cummax()).max()
    yrs = (eqc.index[-1] - eqc.index[0]).days / 365.25
    cagr = eqc.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
    shp = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else 0
    return cagr, shp, dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/universe_broad")
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--risk", type=float, default=0.01)
    args = ap.parse_args()

    # Trend sleeve (Donchian-200 / 3xATR / 6xATR trail)
    At, idxt = T.precompute_trend(args.dir, args.start, args.end, don=200)
    trend_eq = T.simulate(At, idxt, sl_atr=3.0, trail_atr=6.0, risk=args.risk)["equity"]

    # Mean-reversion sleeve (RSI<10 entry / exit>50)
    Am, idxm = SW.precompute(args.dir, args.start, args.end)
    mr_eq = T._mr_equity(Am, idxm, args.risk)

    tr_r = trend_eq.pct_change().dropna()
    mr_r = mr_eq.pct_change().dropna()
    common = tr_r.index.intersection(mr_r.index)
    tr_r, mr_r = tr_r.reindex(common).fillna(0), mr_r.reindex(common).fillna(0)
    combo = 0.5 * tr_r + 0.5 * mr_r
    corr = mr_r.corr(tr_r)

    n_syms = len(At)
    print("=" * 66)
    print(f"  Universe: {args.dir}  ({n_syms} symbols)")
    print(f"  Window  : {common[0].date()} -> {common[-1].date()}")
    print("=" * 66)
    print(f"  {'sleeve':<14}{'CAGR%':>8}{'Sharpe':>8}{'maxDD%':>8}")
    for name, r in [("MEAN-REV", mr_r), ("TREND", tr_r), ("50/50 COMBO", combo)]:
        c, s, d = stats(r)
        print(f"  {name:<14}{c*100:>+8.1f}{s:>8.2f}{d*100:>8.0f}")
    print(f"  Trend/MeanRev correlation: {corr:+.2f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
