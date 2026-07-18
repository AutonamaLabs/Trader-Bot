#!/usr/bin/env python3
"""Fast quality sweep for TPS-5 — find configs with higher PROFIT FACTOR.

Loads the universe once, precomputes indicators, then runs the portfolio sim for
many (entry-depth, stop, exit, down-streak) combos. Reports PF/CAGR/DD/Sharpe so
we can raise trade quality without curve-fitting (judge by PF stability across
the grid, not a single lucky cell).

    python scripts/swing_sweep.py --dir data/swing_stocks --start 2004-01-01
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
from swing_tps5 import load_daily


def precompute(dirpath, start, end, sma=200, rv_spike=3.0):
    idx_set = set()
    stocks = {}
    for f in sorted(Path(dirpath).glob("*.csv")):
        try:
            df = load_daily(f)
        except Exception:
            continue
        if len(df) < sma + 60:
            continue
        close = df["close"]
        down = (close < close.shift(1)).astype(int)
        # consecutive down-day streak
        streak = down * (down.groupby((down == 0).cumsum()).cumcount() + 1)
        lr = np.log(close / close.shift(1))
        rv5 = lr.rolling(5).std()
        stocks[f.stem.upper()] = dict(
            open=df["open"], high=df["high"], low=df["low"], close=close,
            atr=R.atr(df, 14), rsi=R.rsi(close, 3), sma=close.rolling(sma).mean(),
            calm=(rv5 < rv_spike * rv5.rolling(100).median()),
            streak=streak,
        )
        idx_set |= set(df.index)
    idx = pd.DatetimeIndex(sorted(idx_set))
    if start:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
    # Reindex to arrays.
    A = {}
    for s, d in stocks.items():
        A[s] = {}
        for k, v in d.items():
            if k == "calm":
                A[s][k] = v.reindex(idx).fillna(False).to_numpy(dtype=bool)
            elif k == "streak":
                A[s][k] = v.reindex(idx).fillna(0).to_numpy(dtype=float)
            else:
                A[s][k] = v.reindex(idx).to_numpy(dtype=float)
    return A, idx


def simulate(A, idx, rsi_lo, sl_atr, rsi_exit, streak_min,
             equity=5000, risk=0.01, max_pos=10, max_total=0.10, max_notional=0.20,
             time_stop=10):
    # Precompute per-combo signal booleans.
    L, X = {}, {}
    for s, d in A.items():
        L[s] = (d["close"] > d["sma"]) & d["calm"] & (d["rsi"] < rsi_lo) & (d["streak"] >= streak_min)
        X[s] = (d["rsi"] > rsi_exit) | (d["close"] < d["sma"])
    cash = equity; positions = {}; pe, px = {}, set(); pnls = []
    eq = np.empty(len(idx)); eq[:] = np.nan
    for i in range(len(idx)):
        for s in list(positions):
            p = positions[s]; lo = A[s]["low"][i]; op = A[s]["open"][i]
            if np.isnan(lo):
                continue
            p["bars"] += 1
            hit = op if op <= p["stop"] else (p["stop"] if lo <= p["stop"] else None)
            if hit is not None:
                v = (hit - p["entry"]) * p["sh"]; cash += v; pnls.append(v); del positions[s]
        for s in list(px):
            px.discard(s)
            if s in positions and not np.isnan(A[s]["open"][i]):
                p = positions[s]; v = (A[s]["open"][i] - p["entry"]) * p["sh"]
                cash += v; pnls.append(v); del positions[s]
        def marked():
            m = cash
            for s, p in positions.items():
                c = A[s]["close"][i]
                if not np.isnan(c):
                    m += (c - p["entry"]) * p["sh"]
            return m
        equity_now = marked()
        open_risk = sum(p["risk"] for p in positions.values())
        for s in sorted(pe, key=lambda x: pe[x]):
            if s in positions or np.isnan(A[s]["open"][i]) or np.isnan(A[s]["atr"][i]):
                continue
            if len(positions) >= max_pos or open_risk + risk*equity_now > max_total*equity_now + 1e-9:
                continue
            entry = A[s]["open"][i]; stop = entry - sl_atr*A[s]["atr"][i]
            if entry <= stop:
                continue
            rd = risk*equity_now; sh = rd/(entry-stop)
            cap = max_notional*equity_now/entry
            if sh > cap:
                sh = cap; rd = sh*(entry-stop)
            positions[s] = dict(entry=entry, stop=stop, sh=sh, bars=0, risk=rd); open_risk += rd
        pe = {}
        for s, p in positions.items():
            if X[s][i] or p["bars"] >= time_stop:
                px.add(s)
        for s in A:
            if L[s][i] and s not in positions:
                pe[s] = A[s]["rsi"][i]
        eq[i] = marked()
    eqs = pd.Series(eq, index=idx).ffill().dropna()
    t = np.array(pnls)
    wins, losses = t[t > 0], t[t < 0]
    pf = wins.sum()/-losses.sum() if losses.sum() < 0 else float("inf")
    yrs = (eqs.index[-1]-eqs.index[0]).days/365.25
    cagr = (eqs.iloc[-1]/eqs.iloc[0])**(1/yrs)-1
    dd = (1-eqs/eqs.cummax()).max()
    r = eqs.pct_change().dropna()
    sharpe = np.sqrt(252)*r.mean()/r.std() if r.std() > 0 else 0
    return dict(pf=pf, cagr=cagr, dd=dd, sharpe=sharpe, n=len(t),
                win=len(wins)/len(t) if len(t) else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/swing_stocks")
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    print("Loading + precomputing indicators (once)...")
    A, idx = precompute(args.dir, args.start, args.end)
    print(f"{len(A)} symbols, {len(idx)} days {idx[0].date()}->{idx[-1].date()}\n")
    print(f"{'rsi_lo':>6}{'sl_atr':>7}{'exit':>6}{'streak':>7} | {'PF':>5}{'win%':>6}{'CAGR%':>7}{'DD%':>6}{'Shrp':>6}{'trades':>7}")
    grid = itertools.product([5, 10, 15], [2.0, 2.5, 3.0], [50, 65, 75], [0, 1, 2])
    rows = []
    for rsi_lo, sl_atr, rsi_exit, streak in grid:
        r = simulate(A, idx, rsi_lo, sl_atr, rsi_exit, streak)
        rows.append(((rsi_lo, sl_atr, rsi_exit, streak), r))
    # Rank by profit factor (require a sane trade count).
    rows = [r for r in rows if r[1]["n"] >= 200]
    rows.sort(key=lambda x: x[1]["pf"], reverse=True)
    for (rl, sa, ex, st), r in rows[:15]:
        print(f"{rl:>6}{sa:>7}{ex:>6}{st:>7} | {r['pf']:>5.2f}{r['win']*100:>6.0f}"
              f"{r['cagr']*100:>+7.1f}{r['dd']*100:>6.0f}{r['sharpe']:>6.2f}{r['n']:>7}")


if __name__ == "__main__":
    main()
