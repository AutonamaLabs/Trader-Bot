#!/usr/bin/env python3
"""Option 3 — "go faster": phase-free cross-sectional reversal/momentum on H1.

Uses overlapping tranches (no rebalance-phase luck) at hourly horizons, net of a
realistic intraday cost. Also tests a session/time-of-day overlay (only rebalance
around the London-NY window). Honest question: does a faster horizon beat the
(marginal) daily reversal AFTER costs?
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot import research as R

BASKET = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
          "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "XAUUSD"]


def panel(tf="h1"):
    c = {i: R.load(i, tf)["close"] for i in BASKET}
    return pd.DataFrame(c).sort_index().ffill().dropna()


def tranche_returns(pnl, lookback, hold, k, mode, cost, hours=None):
    """Phase-free cross-sectional sleeve on a panel. `hours` optionally restricts
    NEW formations to a UTC-hour window (session overlay)."""
    rets = pnl.pct_change()
    vol = rets.rolling(24).std().replace(0, np.nan)
    sig = pnl.pct_change(lookback)
    hour = pnl.index.hour.to_numpy()
    Wf = pd.DataFrame(0.0, index=pnl.index, columns=pnl.columns)
    for i in range(lookback + 1, len(pnl)):
        if hours is not None and not (hours[0] <= hour[i] < hours[1]):
            continue
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
    return ((Wh * rets).sum(axis=1) - turn * cost).fillna(0.0)


def sharpe(p, ann):
    p = p[p != 0]
    return np.sqrt(ann) * p.mean() / p.std() if p.std() > 0 else 0.0


def posyears(p):
    yr = p.groupby(p.index.year).apply(lambda x: (1 + x).prod() - 1)
    yr = yr[(yr.index >= 2013) & (yr.index <= 2021)]
    return int((yr > 0).sum()), len(yr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.00012, help="one-way cost fraction")
    args = ap.parse_args()
    ANN = 252 * 6  # ~H1 bars per year for annualisation of hourly returns

    pnl = panel("h1")
    print(f"H1 panel: {len(pnl)} bars, {pnl.index[0].date()}->{pnl.index[-1].date()}, "
          f"cost={args.cost*1e4:.1f}bp/side\n")
    print(f"{'mode':<9}{'L(h)':>5}{'H(h)':>5}{'k':>3} | {'Sharpe':>7} {'posY':>5}")
    best = None
    for mode in ("reversal", "momentum"):
        for L, H, k in itertools.product([1, 2, 4, 8, 24], [1, 4, 12], [3]):
            p = tranche_returns(pnl, L, H, k, mode, args.cost)
            s = sharpe(p, ANN); py, n = posyears(p)
            flag = ""
            if s > 0.3 and py >= 6:
                flag = "  <=="
                if best is None or s > best[1]:
                    best = ((mode, L, H, k), s)
            print(f"{mode:<9}{L:>5}{H:>5}{k:>3} | {s:>+7.2f} {py:>3}/{n}{flag}")

    # Session overlay on the best reversal config (London-NY 7-16 UTC).
    print("\nSession overlay (rebalance only 07-16 UTC), reversal L2 H4 k3:")
    for hours in [(7, 16), (13, 21), (0, 24)]:
        p = tranche_returns(pnl, 2, 4, 3, "reversal", args.cost, hours=hours)
        s = sharpe(p, ANN); py, n = posyears(p)
        print(f"  hours {hours}: Sharpe {s:+.2f}  posY {py}/{n}")


if __name__ == "__main__":
    main()
