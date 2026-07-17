#!/usr/bin/env python3
"""Edge-hunting driver: sweep strategy families across instruments/timeframes,
optimise IN-SAMPLE, then validate OUT-OF-SAMPLE with cross-instrument pooling.

An "edge" must (a) be positive in-sample, (b) stay positive out-of-sample, and
(c) work across a MAJORITY of instruments (breadth) — not rely on one lucky
symbol. That triple test is the guard against curve-fitting.

Usage:
    python scripts/research.py --tf h4
    python scripts/research.py --tf d1 --strategies ts_momentum,donchian_turtle
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot import research as R

IS_END = pd.Timestamp("2018-12-31", tz="UTC")

INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
               "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "XAUUSD"]


# ---------------------------------------------------------------------------
# Strategy builders: (df) -> (long_sig, short_sig, atr_arr, sim_kwargs)
# ---------------------------------------------------------------------------
def s_donchian_turtle(df, entry_n=40, exit_n=20, sl_atr=2.0, ema_trend=0):
    a = R.atr(df, 14)
    hi = df["high"].shift(1).rolling(entry_n).max()
    lo = df["low"].shift(1).rolling(entry_n).min()
    xhi = df["high"].shift(1).rolling(exit_n).max()
    xlo = df["low"].shift(1).rolling(exit_n).min()
    long = (df["close"] > hi).to_numpy()
    short = (df["close"] < lo).to_numpy()
    if ema_trend:
        e = R.ema(df["close"], ema_trend)
        long = long & (df["close"] > e).to_numpy()
        short = short & (df["close"] < e).to_numpy()
    exit_long = (df["close"] < xlo).to_numpy()
    exit_short = (df["close"] > xhi).to_numpy()
    return long, short, a.to_numpy(), dict(
        sl_atr=sl_atr, exit_long=exit_long, exit_short=exit_short, exit_on_opposite=True)


def s_ema_cross(df, fast=20, slow=100, sl_atr=2.0, trail_atr=0):
    a = R.atr(df, 14)
    ef, es = R.ema(df["close"], fast), R.ema(df["close"], slow)
    up = (ef > es) & (ef.shift(1) <= es.shift(1))
    dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    return up.to_numpy(), dn.to_numpy(), a.to_numpy(), dict(
        sl_atr=sl_atr, exit_on_opposite=True,
        trail_atr=(trail_atr or None))


def s_ts_momentum(df, lookback=100, sl_atr=3.0, ema_trend=0, trail_atr=0):
    a = R.atr(df, 14)
    mom_up = df["close"] > df["close"].shift(lookback)
    long = (mom_up & ~mom_up.shift(1).fillna(False)).to_numpy()
    short = (~mom_up & mom_up.shift(1).fillna(False)).to_numpy()
    if ema_trend:
        e = R.ema(df["close"], ema_trend)
        long = long & (df["close"] > e).to_numpy()
        short = short & (df["close"] < e).to_numpy()
    return long, short, a.to_numpy(), dict(
        sl_atr=sl_atr, exit_on_opposite=True, trail_atr=(trail_atr or None))


def s_bollinger_mr(df, period=20, k=2.0, adx_max=25, sl_atr=3.0):
    a = R.atr(df, 14)
    ma = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std()
    ad = R.adx(df, 14)
    rangey = (ad < adx_max)
    long = ((df["close"] < ma - k * sd) & rangey).to_numpy()
    short = ((df["close"] > ma + k * sd) & rangey).to_numpy()
    exit_long = (df["close"] >= ma).to_numpy()
    exit_short = (df["close"] <= ma).to_numpy()
    return long, short, a.to_numpy(), dict(
        sl_atr=sl_atr, exit_long=exit_long, exit_short=exit_short)


def s_rsi2_mr(df, lo=10, hi=90, sl_atr=3.0, ma_trend=200, ma_exit=5):
    a = R.atr(df, 14)
    r2 = R.rsi(df["close"], 2)
    et = R.ema(df["close"], ma_trend)
    ex = R.ema(df["close"], ma_exit)
    long = ((r2 < lo) & (df["close"] > et)).to_numpy()
    short = ((r2 > hi) & (df["close"] < et)).to_numpy()
    exit_long = (df["close"] > ex).to_numpy()
    exit_short = (df["close"] < ex).to_numpy()
    return long, short, a.to_numpy(), dict(
        sl_atr=sl_atr, exit_long=exit_long, exit_short=exit_short)


def s_donchian_chandelier(df, entry_n=20, sl_atr=2.0, trail_atr=3.0, time_stop=30, ema_trend=200):
    a = R.atr(df, 14)
    hi = df["high"].shift(1).rolling(entry_n).max()
    lo = df["low"].shift(1).rolling(entry_n).min()
    long = (df["close"] > hi).to_numpy()
    short = (df["close"] < lo).to_numpy()
    if ema_trend:
        e = R.ema(df["close"], ema_trend)
        long = long & (df["close"] > e).to_numpy()
        short = short & (df["close"] < e).to_numpy()
    return long, short, a.to_numpy(), dict(
        sl_atr=sl_atr, trail_atr=trail_atr, time_stop=time_stop, exit_on_opposite=True)


REGISTRY: Dict[str, Tuple[Callable, dict]] = {
    "donchian_turtle": (s_donchian_turtle, dict(
        entry_n=[20, 40, 55], exit_n=[10, 20], sl_atr=[2.0, 3.0], ema_trend=[0, 200])),
    "ema_cross": (s_ema_cross, dict(
        fast=[10, 20, 50], slow=[50, 100, 200], sl_atr=[2.0, 3.0], trail_atr=[0, 3])),
    "ts_momentum": (s_ts_momentum, dict(
        lookback=[50, 100, 200], sl_atr=[2.0, 3.0], ema_trend=[0, 200], trail_atr=[0, 4])),
    "bollinger_mr": (s_bollinger_mr, dict(
        period=[20, 30], k=[2.0, 2.5], adx_max=[20, 25], sl_atr=[2.0, 3.0])),
    "rsi2_mr": (s_rsi2_mr, dict(
        lo=[5, 10], hi=[90, 95], sl_atr=[2.0, 3.0], ma_trend=[100, 200])),
    "donchian_chandelier": (s_donchian_chandelier, dict(
        entry_n=[20, 40], sl_atr=[2.0], trail_atr=[3.0, 5.0], time_stop=[30, 0], ema_trend=[0, 200])),
}


def grid(param_dict):
    keys = list(param_dict)
    for vals in itertools.product(*[param_dict[k] for k in keys]):
        yield dict(zip(keys, vals))


def pooled(results: List[R.SimResult]):
    rs = np.concatenate([r.r_multiples for r in results]) if results else np.array([])
    if len(rs) == 0:
        return dict(n=0, exp_R=0, pf=0, win=0, breadth=0)
    wins, losses = rs[rs > 0], rs[rs < 0]
    pf = wins.sum() / (-losses.sum()) if losses.sum() < 0 else float("inf")
    pos = sum(1 for r in results if r.n >= 15 and r.expectancy > 0 and r.profit_factor > 1.0)
    tested = sum(1 for r in results if r.n >= 15)
    return dict(n=len(rs), exp_R=rs.mean(), pf=pf, win=len(wins) / len(rs),
                breadth=(pos / tested if tested else 0))


def run(tf: str, strategies: List[str]):
    # Cache loaded frames.
    frames = {}
    for inst in INSTRUMENTS:
        try:
            frames[inst] = R.load(inst, tf)
        except FileNotFoundError:
            pass
    print(f"Loaded {len(frames)} instruments @ {tf}. IS<= {IS_END.date()} < OOS\n")

    for name in strategies:
        builder, pgrid = REGISTRY[name]
        combos = list(grid(pgrid))
        rows = []
        for params in combos:
            is_res, oos_res = [], []
            for inst, df in frames.items():
                long, short, a, kw = builder(df, **params)
                is_mask = np.asarray(df.index <= IS_END)
                # in-sample slice
                res_i = R.simulate(df[is_mask], long[is_mask], short[is_mask],
                                   a[is_mask], inst, **kw)
                is_res.append(res_i)
                # out-of-sample slice
                res_o = R.simulate(df[~is_mask], long[~is_mask], short[~is_mask],
                                   a[~is_mask], inst, **kw)
                oos_res.append(res_o)
            pi, po = pooled(is_res), pooled(oos_res)
            rows.append((params, pi, po))
        # Rank by in-sample expectancy (with a trade-count floor).
        rows = [r for r in rows if r[1]["n"] >= 150]
        rows.sort(key=lambda r: r[1]["exp_R"], reverse=True)
        print("=" * 100)
        print(f"STRATEGY: {name}  ({len(combos)} combos, {tf})   [ranked by IS expectancy]")
        print("-" * 100)
        print(f"{'params':<58} | {'IS  expR   pf   n   brdth':<26} | OOS  expR   pf    n   brdth")
        for params, pi, po in rows[:6]:
            ps = ",".join(f"{k}={v}" for k, v in params.items())
            print(f"{ps:<58} | {pi['exp_R']:+.3f} {pi['pf']:.2f} {pi['n']:>5} {pi['breadth']:.2f}"
                  f"   | {po['exp_R']:+.3f} {po['pf']:.2f} {po['n']:>5} {po['breadth']:.2f}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="h4", choices=["h1", "h4", "d1"])
    ap.add_argument("--strategies", default=",".join(REGISTRY.keys()))
    args = ap.parse_args()
    run(args.tf, args.strategies.split(","))


if __name__ == "__main__":
    main()
