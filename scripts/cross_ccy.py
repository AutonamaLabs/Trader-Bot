#!/usr/bin/env python3
"""Cross-sectional G10 currency factor on ECB daily reference rates (1999-2026).

An independent, much longer, and CURRENT dataset (through 2026) — the real test
of whether the short-term reversal edge is genuine and persists into unseen
2020-2026 data, or was noise in the 2013-2022 CFD sample.

Numeraire is EUR (the dataset's base). Each currency's EUR-value = 1/rate. We run
a phase-free (overlapping-tranche) dollar-neutral long/short book across the
basket. Reports Sharpe by era, including the truly out-of-sample 2023-2026.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ECB = ROOT / "data" / "research" / "ecb_daily.csv"
COST = 0.00012

# Country label -> currency code. G10 core (liquid, freely floating).
G10 = {"Australia": "AUD", "Canada": "CAD", "Japan": "JPY", "New Zealand": "NZD",
       "Norway": "NOK", "Sweden": "SEK", "Switzerland": "CHF", "United Kingdom": "GBP"}
EM = {"Brazil": "BRL", "Mexico": "MXN", "South Africa": "ZAR", "South Korea": "KRW",
      "India": "INR", "Singapore": "SGD"}


def load_panel(include_em=False):
    df = pd.read_csv(ECB)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["rate"] = pd.to_numeric(df["Exchange rate"], errors="coerce")
    names = dict(G10)
    if include_em:
        names.update(EM)
    df = df[df["Country"].isin(names)].copy()
    df["ccy"] = df["Country"].map(names)
    wide = df.pivot_table(index="Date", columns="ccy", values="rate").sort_index()
    # EUR-value of one unit of each currency = 1 / (units per EUR).
    value = 1.0 / wide
    # Business-day grid, forward-fill small gaps, drop leading NaNs.
    value = value.resample("1D").last().ffill().dropna(how="any")
    return value


def tranche(pnl, lookback, hold, k, mode, cost=COST):
    rets = pnl.pct_change()
    vol = rets.rolling(20).std().replace(0, np.nan)
    sig = pnl.pct_change(lookback)
    Wf = pd.DataFrame(0.0, index=pnl.index, columns=pnl.columns)
    arr_sig = sig.to_numpy()
    for i in range(lookback + 1, len(pnl)):
        s = pd.Series(arr_sig[i - 1], index=pnl.columns).dropna()
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


def sharpe(p):
    p = p[p != 0]
    return np.sqrt(252) * p.mean() / p.std() if p.std() > 0 else 0.0


def era(p, lo, hi):
    q = p[(p.index.year >= lo) & (p.index.year <= hi)]
    yr = q.groupby(q.index.year).apply(lambda x: (1 + x).prod() - 1)
    posy = int((yr > 0).sum())
    return sharpe(q), posy, len(yr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--em", action="store_true", help="include EM currencies")
    args = ap.parse_args()
    pnl = load_panel(include_em=args.em)
    print(f"Basket: {list(pnl.columns)}  ({len(pnl)} days, "
          f"{pnl.index[0].date()} -> {pnl.index[-1].date()})\n")

    configs = [("reversal", 20, 5, 2), ("reversal", 20, 5, 3), ("reversal", 10, 5, 3),
               ("reversal", 5, 3, 3), ("momentum", 60, 20, 3), ("momentum", 250, 20, 3)]
    print(f"{'strat':<9}{'L':>4}{'H':>4}{'k':>3} | {'FULL':>13} | {'2013-19':>12} | "
          f"{'2020-26 OOS':>14} | {'2023-26 OOS':>13}")
    for mode, L, H, k in configs:
        p = tranche(pnl, L, H, k, mode)
        sf, _, _ = era(p, 1999, 2026)
        s1, y1, n1 = era(p, 2013, 2019)
        s2, y2, n2 = era(p, 2020, 2026)
        s3, y3, n3 = era(p, 2023, 2026)
        print(f"{mode:<9}{L:>4}{H:>4}{k:>3} | Sharpe {sf:+5.2f} | "
              f"{s1:+.2f} {y1}/{n1} | {s2:+.2f} {y2}/{n2} | {s3:+.2f} {y3}/{n3}")

    # Drop-one robustness on the headline reversal config.
    print("\nDrop-one robustness (reversal 20/5 k3, full sample):")
    base = sharpe(tranche(pnl, 20, 5, 3, "reversal"))
    rows = [(d, sharpe(tranche(pnl.drop(columns=[d]), 20, 5, 3, "reversal"))) for d in pnl.columns]
    rows.sort(key=lambda x: x[1])
    print(f"  full {base:+.2f} | worst drops: " + ", ".join(f"-{d}:{s:+.2f}" for d, s in rows[:3]))


if __name__ == "__main__":
    main()
