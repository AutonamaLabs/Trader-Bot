#!/usr/bin/env python3
"""Integrity audit of the v2 factor (per the head-quant review).

Two biases inflated the originally-reported Sharpe 0.45:
  1. Selection bias — sleeves were picked from a full-sample grid sweep.
  2. Rebalance-phase luck — a 20-day-hold block has 20 possible start offsets;
     the reported result landed on a lucky one.

This script removes both by rebuilding every sleeve with OVERLAPPING TRANCHES
(daily formation, averaged over the hold period = phase-free), then stress-tests
the survivor with parameter-neighbourhood and drop-one-instrument checks. It
prints the honest, deflated numbers. Run: python scripts/audit_phasefree.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot import research as R

BASKET = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
          "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "XAUUSD"]
COST = 0.00012


def panel():
    c = {i: R.load(i, "d1")["close"] for i in BASKET}
    return pd.DataFrame(c).sort_index().ffill().dropna()


def tranche(pnl, lookback, hold, k, mode):
    """Phase-free sleeve return series via overlapping daily tranches."""
    rets = pnl.pct_change()
    vol = rets.rolling(20).std().replace(0, np.nan)
    sig = pnl.pct_change(lookback)
    Wf = pd.DataFrame(0.0, index=pnl.index, columns=pnl.columns)
    for i in range(lookback + 1, len(pnl)):
        s = sig.iloc[i - 1].dropna()
        if len(s) >= 2 * k:
            r = s.sort_values()
            lo, wi = r.index[:k], r.index[-k:]
            longs, shorts = (wi, lo) if mode == "momentum" else (lo, wi)
            w = pd.Series(0.0, index=pnl.columns)
            iv = 1.0 / vol.iloc[i - 1]
            for g, sg in [(longs, 1), (shorts, -1)]:
                gv = iv[g].fillna(0)
                w[g] = (sg * gv / gv.sum()).values if gv.sum() > 0 else sg / k
            Wf.iloc[i] = w
    Wh = Wf.rolling(hold).mean().shift(1).fillna(0.0)
    turn = (Wh - Wh.shift(1)).abs().sum(axis=1)
    return ((Wh * rets).sum(axis=1) - turn * COST).fillna(0.0)


def sharpe(p):
    p = p[p != 0]
    return np.sqrt(252) * p.mean() / p.std() if p.std() > 0 else 0.0


def posyears(p, lo=2013, hi=2021):
    yr = p.groupby(p.index.year).apply(lambda x: (1 + x).prod() - 1)
    yr = yr[(yr.index >= lo) & (yr.index <= hi)]
    return int((yr > 0).sum()), len(yr)


def main():
    pnl = panel()
    print("=" * 66)
    print("  PHASE-FREE (overlapping-tranche) sleeve results — the honest view")
    print("=" * 66)
    for name, (lb, h, k, m) in {
        "MOM_20_20": (20, 20, 3, "momentum"),
        "REV_20_5": (20, 5, 3, "reversal"),
        "REV_10_5": (10, 5, 3, "reversal"),
    }.items():
        p = tranche(pnl, lb, h, k, m)
        py, n = posyears(p)
        print(f"  {name:<10} Sharpe {sharpe(p):+.2f}   positive years {py}/{n}")

    print("\n  Reversal (20/5) parameter-neighbourhood stability:")
    for lb in (15, 20, 25):
        for k in (2, 3):
            p = tranche(pnl, lb, 5, k, "reversal")
            py, n = posyears(p)
            print(f"    lookback={lb} k={k}: Sharpe {sharpe(p):+.2f}  posY {py}/{n}")

    print("\n  Reversal (20/5) drop-one-instrument robustness:")
    base = sharpe(tranche(pnl, 20, 5, 3, "reversal"))
    rows = []
    for d in pnl.columns:
        rows.append((d, sharpe(tranche(pnl.drop(columns=[d]), 20, 5, 3, "reversal"))))
    rows.sort(key=lambda x: x[1])
    print(f"    full basket Sharpe {base:+.2f}")
    print(f"    worst 3 drops: " + ", ".join(f"-{d}->{s:+.2f}" for d, s in rows[:3]))
    print("=" * 66)
    print("  VERDICT: momentum sleeve is phase-free NEGATIVE (was luck). Only a")
    print("  weak, USDCHF-dependent short-term reversal tendency survives. The")
    print("  originally-reported Sharpe 0.45 was inflated by selection + phase")
    print("  bias. Honest edge here is marginal; see docs/RESEARCH.md.")
    print("=" * 66)


if __name__ == "__main__":
    main()
