#!/usr/bin/env python3
"""Cross-sectional (market-neutral) momentum & reversal on an FX basket.

Time-series trend on single pairs died out-of-sample. Cross-sectional factors
are historically more robust because they are *relative*: each day rank the
basket by trailing return, go long the strongest and short the weakest (momentum)
or the reverse (reversal), equal-weight, dollar-neutral. Being long-and-short
nets out the common USD factor and much of the beta, isolating relative value.

We evaluate net of costs, per calendar year, with NO per-year fitting.
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

# One-way transaction cost as a fraction of notional (spread+slippage / price).
COST_FRAC = 0.00012


def build_panel(tf="d1"):
    closes = {}
    for inst in BASKET:
        try:
            closes[inst] = R.load(inst, tf)["close"]
        except FileNotFoundError:
            pass
    panel = pd.DataFrame(closes).sort_index().ffill().dropna()
    return panel


def backtest(panel, lookback=60, hold=5, k=3, mode="momentum", vol_target=True):
    """Return a daily net-return series for the L/S factor portfolio."""
    rets = panel.pct_change()
    signal = panel.pct_change(lookback)               # trailing return
    # Optional inverse-vol scaling so no single pair dominates risk.
    vol = rets.rolling(20).std().replace(0, np.nan)

    dates = panel.index
    weights = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    last_w = pd.Series(0.0, index=panel.columns)
    port = pd.Series(0.0, index=dates)
    cost_series = pd.Series(0.0, index=dates)

    for i in range(lookback + 1, len(dates)):
        d = dates[i]
        if (i - (lookback + 1)) % hold == 0:            # rebalance day
            s = signal.iloc[i - 1].dropna()             # info up to prior close
            if len(s) >= 2 * k:
                ranked = s.sort_values()
                losers, winners = ranked.index[:k], ranked.index[-k:]
                w = pd.Series(0.0, index=panel.columns)
                longs = winners if mode == "momentum" else losers
                shorts = losers if mode == "momentum" else winners
                w[longs] = 1.0 / k
                w[shorts] = -1.0 / k
                if vol_target:
                    iv = (1.0 / vol.iloc[i - 1]).reindex(w.index)
                    for grp, sign in [(longs, 1), (shorts, -1)]:
                        gv = iv[grp].fillna(0)
                        if gv.sum() > 0:
                            w[grp] = sign * (gv / gv.sum()).values
                turnover = (w - last_w).abs().sum()
                cost_series.iloc[i] = turnover * COST_FRAC
                last_w = w
        weights.iloc[i] = last_w
        port.iloc[i] = (last_w * rets.iloc[i]).sum() - cost_series.iloc[i]
    return port


def stats(port, ann=252):
    p = port.dropna()
    if p.std() == 0 or len(p) == 0:
        return dict(sharpe=0, ret=0, dd=0)
    eq = (1 + p).cumprod()
    sharpe = np.sqrt(ann) * p.mean() / p.std()
    cagr = eq.iloc[-1] ** (ann / len(p)) - 1
    dd = (1 - eq / eq.cummax()).max()
    return dict(sharpe=sharpe, ret=cagr, dd=dd)


def per_year(port):
    out = {}
    for y, grp in port.groupby(port.index.year):
        s = stats(grp)
        out[y] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="d1")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--cost_mult", type=float, default=1.0)
    args = ap.parse_args()

    global COST_FRAC
    COST_FRAC = COST_FRAC * args.cost_mult

    panel = build_panel(args.tf)
    print(f"Basket: {list(panel.columns)}  ({len(panel)} bars, "
          f"{panel.index[0].date()}->{panel.index[-1].date()})\n")

    if args.sweep:
        print(f"{'mode':<10}{'L':>5}{'H':>4}{'k':>3} | {'Sharpe':>7}{'CAGR%':>8}{'MaxDD%':>8}  posY")
        for mode in ("momentum", "reversal"):
            for L, H, k in itertools.product([5, 10, 20, 60, 120], [1, 5, 20], [2, 3]):
                port = backtest(panel, lookback=L, hold=H, k=k, mode=mode)
                s = stats(port)
                py = per_year(port)
                posy = sum(1 for v in py.values() if v["ret"] > 0)
                mark = "  <==" if s["sharpe"] > 0.5 and posy >= 6 else ""
                print(f"{mode:<10}{L:>5}{H:>4}{k:>3} | {s['sharpe']:>7.2f}{s['ret']*100:>7.1f}%"
                      f"{s['dd']*100:>7.1f}%   {posy}/9{mark}")
        return

    if args.ensemble:
        # Blend low-correlation robust sleeves at equal risk.
        sleeves = {
            "MOM_20_20_3": dict(lookback=20, hold=20, k=3, mode="momentum"),
            "MOM_20_20_2": dict(lookback=20, hold=20, k=2, mode="momentum"),
            "REV_20_5_3":  dict(lookback=20, hold=5,  k=3, mode="reversal"),
            "REV_10_5_3":  dict(lookback=10, hold=5,  k=3, mode="reversal"),
        }
        ports = {name: backtest(panel, **kw) for name, kw in sleeves.items()}
        # Correlation of sleeve daily returns.
        pr = pd.DataFrame(ports).dropna()
        print("Sleeve daily-return correlation:")
        print(pr.corr().round(2).to_string(), "\n")
        # Equal-risk blend (inverse-vol weight the sleeves).
        iv = 1.0 / pr.std()
        w = iv / iv.sum()
        combo = (pr * w).sum(axis=1)
        for name, p in ports.items():
            s = stats(p); py = per_year(p)
            posy = sum(1 for v in py.values() if v["ret"] > 0)
            print(f"  {name:<13} Sharpe {s['sharpe']:+.2f}  CAGR {s['ret']*100:+.1f}%  "
                  f"DD {s['dd']*100:.1f}%  posY {posy}/9")
        s = stats(combo); py = per_year(combo)
        posy = sum(1 for v in py.values() if v["ret"] > 0)
        print(f"\n  {'ENSEMBLE':<13} Sharpe {s['sharpe']:+.2f}  CAGR {s['ret']*100:+.1f}%  "
              f"DD {s['dd']*100:.1f}%  posY {posy}/9")
        print("\n  Ensemble per year:")
        for y, v in py.items():
            print(f"    {y}: Sharpe {v['sharpe']:+.2f}  ret {v['ret']*100:+.1f}%  DD {v['dd']*100:.1f}%")
        return

    # Single default run detail.
    port = backtest(panel, lookback=20, hold=20, k=3, mode="momentum")
    s = stats(port)
    print(f"Momentum L60 H5 k3: Sharpe {s['sharpe']:.2f}  CAGR {s['ret']*100:.1f}%  DD {s['dd']*100:.1f}%")
    for y, v in per_year(port).items():
        print(f"  {y}: Sharpe {v['sharpe']:+.2f}  ret {v['ret']*100:+.1f}%")


if __name__ == "__main__":
    main()
