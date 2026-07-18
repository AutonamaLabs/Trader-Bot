#!/usr/bin/env python3
"""Rolling walk-forward for the two-sleeve equity system.

For each out-of-sample year, re-select each sleeve's parameters on the PRIOR
in-sample window (by in-sample Sharpe), then trade that OOS year with those
frozen params starting flat. Concatenate the OOS years into one walk-forward
equity curve. This tests whether the parameter tuning GENERALISES — the gate
before any live/demo deployment.

    python scripts/walkforward_equity.py --dir data/swing_stocks
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

MR_GRID = list(itertools.product([5, 10, 15], [50, 65]))          # (rsi_lo, rsi_exit)
TR_GRID = list(itertools.product([2.0, 3.0], [4.0, 6.0]))         # (sl_atr, trail_atr)


def precompute(dirpath, don=200, sma=200):
    idx_set, stocks = set(), {}
    for f in sorted(Path(dirpath).glob("*.csv")):
        try:
            df = load_daily(f)
        except Exception:
            continue
        if len(df) < sma + don + 40:
            continue
        close, high, low = df["close"], df["high"], df["low"]
        s200 = close.rolling(sma).mean()
        lr = np.log(close / close.shift(1))
        rv5 = lr.rolling(5).std()
        stocks[f.stem.upper()] = dict(
            open=df["open"], high=high, low=low, close=close,
            atr=R.atr(df, 14), rsi=R.rsi(close, 3), sma=s200,
            calm=(rv5 < 3.0 * rv5.rolling(100).median()),
            sma_rising=(s200 > s200.shift(20)),
            don_hi=high.shift(1).rolling(don).max(),
        )
        idx_set |= set(df.index)
    idx = pd.DatetimeIndex(sorted(idx_set))
    A = {}
    for s, d in stocks.items():
        A[s] = {}
        for k, v in d.items():
            r = v.reindex(idx)
            A[s][k] = r.fillna(False).to_numpy(bool) if v.dtype == bool else r.to_numpy(float)
    return A, idx


def _run(A, lo, hi, mode, sl_atr, trail_atr, rsi_lo, rsi_exit,
         risk=0.01, max_pos=10, max_total=0.10, max_notional=0.20, time_stop=10):
    """Run one sleeve over integer slice [lo:hi]; return daily equity np-array."""
    rng = range(lo, hi)
    cash = 1.0; pos = {}; px = set(); pe = {}
    eq = []
    for i in rng:
        for s in list(pos):
            p = pos[s]; L = A[s]["low"][i]; op = A[s]["open"][i]; hiP = A[s]["high"][i]; atr = A[s]["atr"][i]
            if np.isnan(L):
                continue
            p["bars"] += 1
            hit = op if op <= p["stop"] else (p["stop"] if L <= p["stop"] else None)
            if hit is not None:
                cash += (hit - p["entry"]) * p["sh"]; del pos[s]; continue
            if mode == "trend":
                if not np.isnan(hiP):
                    p["ext"] = max(p["ext"], hiP)
                if not np.isnan(atr):
                    p["stop"] = max(p["stop"], p["ext"] - trail_atr * atr)
        for s in list(px):
            px.discard(s)
            if s in pos and not np.isnan(A[s]["open"][i]):
                p = pos[s]; cash += (A[s]["open"][i] - p["entry"]) * p["sh"]; del pos[s]
        def mk():
            m = cash
            for s, p in pos.items():
                c = A[s]["close"][i]
                if not np.isnan(c):
                    m += (c - p["entry"]) * p["sh"]
            return m
        eqn = mk(); orisk = sum(p["risk"] for p in pos.values())
        for s in sorted(pe, key=lambda x: pe[x]):
            if s in pos or np.isnan(A[s]["open"][i]) or np.isnan(A[s]["atr"][i]) or len(pos) >= max_pos:
                continue
            if orisk + risk * eqn > max_total * eqn + 1e-9:
                continue
            entry = A[s]["open"][i]; stop = entry - sl_atr * A[s]["atr"][i]
            if entry <= stop:
                continue
            rd = risk * eqn; sh = rd / (entry - stop); cap = max_notional * eqn / entry
            if sh > cap:
                sh = cap; rd = sh * (entry - stop)
            pos[s] = dict(entry=entry, stop=stop, sh=sh, bars=0, risk=rd, ext=entry); orisk += rd
        pe = {}
        for s, p in pos.items():
            regime_exit = A[s]["close"][i] < A[s]["sma"][i]
            if mode == "mr":
                if regime_exit or A[s]["rsi"][i] > rsi_exit or p["bars"] >= time_stop:
                    px.add(s)
            else:
                if regime_exit:
                    px.add(s)
        if mode == "mr":
            for s in A:
                if (A[s]["close"][i] > A[s]["sma"][i] and A[s]["calm"][i]
                        and A[s]["rsi"][i] < rsi_lo and s not in pos):
                    pe[s] = A[s]["rsi"][i]
        else:
            for s in A:
                if (A[s]["close"][i] > A[s]["don_hi"][i] and A[s]["close"][i] > A[s]["sma"][i]
                        and A[s]["sma_rising"][i] and s not in pos):
                    pe[s] = -A[s]["close"][i]   # rank arbitrary
        eq.append(mk())
    return np.array(eq)


def sharpe(eq):
    r = np.diff(eq) / eq[:-1]
    return np.sqrt(252) * r.mean() / r.std() if len(r) and r.std() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/swing_stocks")
    ap.add_argument("--is_years", type=int, default=5)
    ap.add_argument("--oos_from", type=int, default=2006, help="skip sparse early years")
    args = ap.parse_args()
    print("Precomputing (once)...")
    A, idx = precompute(args.dir)
    yrs = idx.year
    print(f"{len(A)} symbols, {idx[0].date()}->{idx[-1].date()}\n")

    oos_years = [y for y in range(max(idx.year.min() + args.is_years, args.oos_from),
                                   idx.year.max() + 1)]
    mr_ret, tr_ret, dates = [], [], []
    print(f"{'OOS yr':<7}{'MR params':<14}{'TR params':<14}{'MR Shrp':>8}{'TR Shrp':>8}")
    for y in oos_years:
        is_mask = np.where((yrs >= y - args.is_years) & (yrs < y))[0]
        oos_mask = np.where(yrs == y)[0]
        if len(is_mask) < 200 or len(oos_mask) < 50:
            continue
        a, b = is_mask[0], is_mask[-1] + 1
        oa, ob = oos_mask[0], oos_mask[-1] + 1
        # select MR params by IS sharpe
        best_mr = max(MR_GRID, key=lambda g: sharpe(_run(A, a, b, "mr", 3.0, 0, g[0], g[1])))
        best_tr = max(TR_GRID, key=lambda g: sharpe(_run(A, a, b, "trend", g[0], g[1], 0, 0)))
        eq_mr = _run(A, oa, ob, "mr", 3.0, 0, best_mr[0], best_mr[1])
        eq_tr = _run(A, oa, ob, "trend", best_tr[0], best_tr[1], 0, 0)
        mr_ret.append(np.diff(eq_mr) / eq_mr[:-1])
        tr_ret.append(np.diff(eq_tr) / eq_tr[:-1])
        dates.append(idx[oa + 1:ob])
        print(f"{y:<7}{str(best_mr):<14}{str(best_tr):<14}{sharpe(eq_mr):>8.2f}{sharpe(eq_tr):>8.2f}")

    mr = pd.Series(np.concatenate(mr_ret), index=pd.DatetimeIndex(np.concatenate(dates)))
    tr = pd.Series(np.concatenate(tr_ret), index=mr.index)
    combo = 0.5 * mr + 0.5 * tr
    print("\n== WALK-FORWARD (out-of-sample only) ==")
    for name, s in [("MEAN-REV", mr), ("TREND", tr), ("50/50 COMBO", combo)]:
        eqc = (1 + s).cumprod(); dd = (1 - eqc / eqc.cummax()).max()
        yrs_span = (eqc.index[-1] - eqc.index[0]).days / 365.25
        cagr = eqc.iloc[-1] ** (1 / yrs_span) - 1
        shp = np.sqrt(252) * s.mean() / s.std() if s.std() > 0 else 0
        print(f"  {name:<12} CAGR {cagr*100:+5.1f}%  Sharpe {shp:.2f}  maxDD {dd*100:.0f}%")
    print(f"  Trend/MeanRev OOS correlation: {mr.corr(tr):+.2f}")


if __name__ == "__main__":
    main()
