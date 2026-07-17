#!/usr/bin/env python3
"""Walk-forward / per-year robustness for a FIXED strategy config.

A weak edge that happens to fit one out-of-sample window is not an edge. This
evaluates a single fixed parameter set across every calendar year (pooled across
instruments) and per-instrument, so we can see whether the edge is *persistent*
or just a lucky regime. A real edge is positive in a clear majority of years and
across most instruments — with NO parameter fitting per year.
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

INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
               "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "XAUUSD"]


def mean_reversion(df, n=3, rsi_lo=10, rsi_hi=90, ma_trend=200, ma_exit=5,
                   sl_atr=3.0, max_hold=10, side="both", adx_max=0):
    """Connors-style short-term mean reversion.

    Long when RSI(n) is oversold and price is above a long-term trend MA
    (buy dips in uptrends); exit when price reverts above a short MA. Mirror for
    shorts. Optional range-regime filter (ADX below adx_max).
    """
    a = R.atr(df, 14)
    r = R.rsi(df["close"], n)
    et = R.ema(df["close"], ma_trend)
    ex = R.ema(df["close"], ma_exit)
    long = (r < rsi_lo) & (df["close"] > et)
    short = (r > rsi_hi) & (df["close"] < et)
    if adx_max:
        ad = R.adx(df, 14)
        long &= ad < adx_max
        short &= ad < adx_max
    long = long.to_numpy()
    short = short.to_numpy() if side in ("both", "short") else np.zeros(len(df), bool)
    if side == "short":
        long = np.zeros(len(df), bool)
    exit_long = (df["close"] > ex).to_numpy()
    exit_short = (df["close"] < ex).to_numpy()
    kw = dict(sl_atr=sl_atr, exit_long=exit_long, exit_short=exit_short,
              time_stop=max_hold, allow_short=(side != "long"))
    return long, short, a.to_numpy(), kw


def pooled(results):
    rs = np.concatenate([r for r in results]) if results else np.array([])
    if len(rs) == 0:
        return 0, 0.0, 0.0, 0.0
    wins, losses = rs[rs > 0], rs[rs < 0]
    pf = wins.sum() / (-losses.sum()) if losses.sum() < 0 else float("inf")
    return len(rs), rs.mean(), pf, len(wins) / len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="d1")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--rsi_lo", type=int, default=10)
    ap.add_argument("--rsi_hi", type=int, default=90)
    ap.add_argument("--ma_trend", type=int, default=200)
    ap.add_argument("--ma_exit", type=int, default=5)
    ap.add_argument("--sl_atr", type=float, default=3.0)
    ap.add_argument("--max_hold", type=int, default=10)
    ap.add_argument("--side", default="both", choices=["both", "long", "short"])
    ap.add_argument("--adx_max", type=int, default=0)
    args = ap.parse_args()

    params = dict(n=args.n, rsi_lo=args.rsi_lo, rsi_hi=args.rsi_hi,
                  ma_trend=args.ma_trend, ma_exit=args.ma_exit, sl_atr=args.sl_atr,
                  max_hold=args.max_hold, side=args.side, adx_max=args.adx_max)
    print(f"Config: tf={args.tf} {params}\n")

    frames = {}
    for inst in INSTRUMENTS:
        try:
            frames[inst] = R.load(inst, args.tf)
        except FileNotFoundError:
            pass

    # Precompute signals+trades per instrument, tagged with exit-year.
    per_inst_r = {}     # inst -> list of (year, r)
    for inst, df in frames.items():
        long, short, a, kw = mean_reversion(df, **params)
        res = R.simulate(df, long, short, a, inst, **kw)
        # Re-run to capture exit times: reconstruct via a light wrapper.
        per_inst_r[inst] = res

    # --- Per-year pooled (uses a trade-level simulate that records years) ---
    print("=" * 74)
    print("  PER-YEAR (pooled across instruments)")
    print("=" * 74)
    print(f"  {'year':<6} {'trades':>7} {'exp_R':>8} {'pf':>6} {'win%':>6}")
    years = range(2013, 2022)
    year_flags = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        end = pd.Timestamp(f"{y}-12-31 23:59", tz="UTC")
        rs = []
        for inst, df in frames.items():
            mask = np.asarray((df.index >= start) & (df.index <= end))
            if mask.sum() < 30:
                continue
            long, short, a, kw = mean_reversion(df, **params)
            res = R.simulate(df[mask], long[mask], short[mask], a[mask], inst, **kw)
            rs.append(res.r_multiples)
        n, exp, pf, win = pooled(rs)
        flag = "+" if exp > 0 else "-"
        year_flags.append(exp > 0)
        print(f"  {y:<6} {n:>7} {exp:>+8.3f} {pf:>6.2f} {win*100:>5.1f}  {flag}")

    # --- Per-instrument full-sample ---
    print("=" * 74)
    print("  PER-INSTRUMENT (full sample)")
    print("=" * 74)
    print(f"  {'inst':<8} {'trades':>7} {'exp_R':>8} {'pf':>6} {'win%':>6} {'ret%':>7}")
    pos_inst = 0
    for inst, res in per_inst_r.items():
        if res.n < 20:
            continue
        flag = "+" if res.expectancy > 0 else "-"
        pos_inst += res.expectancy > 0
        print(f"  {inst:<8} {res.n:>7} {res.expectancy:>+8.3f} {res.profit_factor:>6.2f} "
              f"{res.win_rate*100:>5.1f} {res.total_return*100:>+6.1f}  {flag}")

    # --- Verdict ---
    pos_years = sum(year_flags)
    print("=" * 74)
    print(f"  Positive years:       {pos_years}/{len(year_flags)}")
    print(f"  Positive instruments: {pos_inst}/{sum(1 for r in per_inst_r.values() if r.n>=20)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
