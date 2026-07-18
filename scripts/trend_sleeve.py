#!/usr/bin/env python3
"""Trend sleeve for equities — the complement to mean-reversion TPS-5.

Long-only breakout momentum: enter when price breaks a Donchian N-day high while
in a confirmed uptrend (above rising SMA200); ride a chandelier ATR trailing stop
(THIS is where "let winners run" pays); exit on the trail or a regime break. This
is positive-skew (low win rate, big winners) — the opposite shape to TPS-5, so
the two sleeves should be lowly correlated and combine well.

    python scripts/trend_sleeve.py --dir data/swing_stocks --start 2004-01-01
    python scripts/trend_sleeve.py --combine   # blend with the MR sleeve
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


def precompute_trend(dirpath, start, end, don=100, sma=200):
    idx_set, stocks = set(), {}
    for f in sorted(Path(dirpath).glob("*.csv")):
        try:
            df = load_daily(f)
        except Exception:
            continue
        if len(df) < sma + don + 20:
            continue
        close, high, low = df["close"], df["high"], df["low"]
        s200 = close.rolling(sma).mean()
        stocks[f.stem.upper()] = dict(
            open=df["open"], high=high, low=low, close=close,
            atr=R.atr(df, 14), sma=s200,
            sma_rising=(s200 > s200.shift(20)),
            don_hi=high.shift(1).rolling(don).max(),
        )
        idx_set |= set(df.index)
    idx = pd.DatetimeIndex(sorted(idx_set))
    if start:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
    A = {}
    for s, d in stocks.items():
        A[s] = {k: (v.reindex(idx).fillna(False).to_numpy(dtype=bool) if v.dtype == bool
                    else v.reindex(idx).to_numpy(dtype=float)) for k, v in d.items()}
    return A, idx


def simulate(A, idx, sl_atr=3.0, trail_atr=4.0,
             equity=5000, risk=0.01, max_pos=10, max_total=0.10, max_notional=0.20):
    ENTRY = {s: (A[s]["close"] > A[s]["don_hi"]) & (A[s]["close"] > A[s]["sma"]) & A[s]["sma_rising"]
             for s in A}
    cash = equity; pos = {}; pnls = []; holds = []
    pe, px = [], set()
    eq = np.empty(len(idx)); eq[:] = np.nan
    for i in range(len(idx)):
        # manage opens: trailing stop (gap-aware)
        for s in list(pos):
            p = pos[s]; lo = A[s]["low"][i]; op = A[s]["open"][i]; hi = A[s]["high"][i]; atr = A[s]["atr"][i]
            if np.isnan(lo):
                continue
            p["bars"] += 1
            hit = op if op <= p["stop"] else (p["stop"] if lo <= p["stop"] else None)
            if hit is not None:
                v = (hit - p["entry"]) * p["sh"]; cash += v; pnls.append(v); holds.append(p["bars"]); del pos[s]; continue
            if not np.isnan(hi):
                p["ext"] = max(p["ext"], hi)
            if not np.isnan(atr):
                p["stop"] = max(p["stop"], p["ext"] - trail_atr * atr)
        # pending regime-exit at open
        for s in list(px):
            px.discard(s)
            if s in pos and not np.isnan(A[s]["open"][i]):
                p = pos[s]; v = (A[s]["open"][i] - p["entry"]) * p["sh"]
                cash += v; pnls.append(v); holds.append(p["bars"]); del pos[s]
        def marked():
            m = cash
            for s, p in pos.items():
                c = A[s]["close"][i]
                if not np.isnan(c):
                    m += (c - p["entry"]) * p["sh"]
            return m
        equity_now = marked()
        # entries at open
        open_risk = sum(p["risk"] for p in pos.values())
        for s in pe:
            if s in pos or np.isnan(A[s]["open"][i]) or np.isnan(A[s]["atr"][i]):
                continue
            if len(pos) >= max_pos or open_risk + risk*equity_now > max_total*equity_now + 1e-9:
                continue
            entry = A[s]["open"][i]; stop = entry - sl_atr*A[s]["atr"][i]
            if entry <= stop:
                continue
            rd = risk*equity_now; sh = rd/(entry-stop)
            cap = max_notional*equity_now/entry
            if sh > cap:
                sh = cap; rd = sh*(entry-stop)
            pos[s] = dict(entry=entry, stop=stop, sh=sh, bars=0, risk=rd, ext=entry); open_risk += rd
        pe = []
        # signals on close
        for s, p in pos.items():
            if A[s]["close"][i] < A[s]["sma"][i]:
                px.add(s)
        pe = [s for s in A if ENTRY[s][i] and s not in pos]
        eq[i] = marked()
    eqs = pd.Series(eq, index=idx).ffill().dropna()
    t = np.array(pnls); w = t[t > 0]; l = t[t < 0]
    pf = w.sum()/-l.sum() if l.sum() < 0 else float("inf")
    yrs = (eqs.index[-1]-eqs.index[0]).days/365.25
    cagr = (eqs.iloc[-1]/eqs.iloc[0])**(1/yrs)-1
    dd = (1-eqs/eqs.cummax()).max()
    r = eqs.pct_change().dropna(); sh = np.sqrt(252)*r.mean()/r.std() if r.std() > 0 else 0
    return dict(pf=pf, cagr=cagr, dd=dd, sharpe=sh, n=len(t),
                win=len(w)/len(t) if len(t) else 0, hold=np.mean(holds) if holds else 0,
                avgwin=w.mean() if len(w) else 0, avgloss=l.mean() if len(l) else 0,
                equity=eqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/swing_stocks")
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    print("Precomputing trend features (Donchian=100)...")
    A, idx = precompute_trend(args.dir, args.start, args.end, don=100)
    print(f"{len(A)} symbols, {idx[0].date()}->{idx[-1].date()}\n")
    hdr = f"{'don/sl/trail':<16}{'PF':>6}{'win%':>6}{'avgW':>7}{'avgL':>7}{'hold':>6}{'CAGR%':>7}{'DD%':>6}{'Shrp':>6}{'trades':>7}"
    print("== Trend sleeve sweep ==\n" + hdr)
    best = None
    for don in (50, 100, 200):
        Ad, _ = precompute_trend(args.dir, args.start, args.end, don=don)
        for sl, tr in [(2.0, 3.0), (3.0, 4.0), (3.0, 6.0)]:
            r = simulate(Ad, idx, sl_atr=sl, trail_atr=tr, risk=args.risk)
            print(f"{f'{don}/{sl}/{tr}':<16}{r['pf']:>6.2f}{r['win']*100:>6.0f}{r['avgwin']:>7.0f}"
                  f"{r['avgloss']:>7.0f}{r['hold']:>6.0f}{r['cagr']*100:>+7.1f}{r['dd']*100:>6.0f}{r['sharpe']:>6.2f}{r['n']:>7}")
            if best is None or r["sharpe"] > best[1]["sharpe"]:
                best = ((don, sl, tr), r)

    if args.combine:
        print("\n== Combine trend sleeve with mean-reversion (TPS-5) ==")
        (don, sl, tr), rtr = best
        Ad, _ = precompute_trend(args.dir, args.start, args.end, don=don)
        trend_eq = simulate(Ad, idx, sl_atr=sl, trail_atr=tr, risk=args.risk)["equity"]
        # mean-reversion sleeve equity via the production scanner internals
        import swing_sweep as SW
        Amr, idxmr = SW.precompute(args.dir, args.start, args.end)
        mr = SW.simulate(Amr, idxmr, rsi_lo=10, sl_atr=3.0, rsi_exit=50, streak_min=0, risk=args.risk)
        # rebuild MR equity curve (SW.simulate returns stats; recompute eq here)
        mr_eq = _mr_equity(Amr, idxmr, args.risk)
        tr_r = trend_eq.pct_change().reindex(idx).fillna(0)
        mr_r = mr_eq.pct_change().reindex(idx).fillna(0)
        corr = pd.concat([tr_r.rename("trend"), mr_r.rename("mr")], axis=1).corr().iloc[0, 1]
        combo = 0.5 * tr_r + 0.5 * mr_r
        for name, s in [("TREND", tr_r), ("MEAN-REV", mr_r), ("50/50 COMBO", combo)]:
            eqc = (1 + s).cumprod()
            dd = (1 - eqc / eqc.cummax()).max()
            shp = np.sqrt(252) * s.mean() / s.std() if s.std() > 0 else 0
            yrs = (eqc.index[-1] - eqc.index[0]).days / 365.25
            cg = eqc.iloc[-1] ** (1 / yrs) - 1
            print(f"  {name:<12} CAGR {cg*100:+5.1f}%  Sharpe {shp:.2f}  maxDD {dd*100:.0f}%")
        print(f"  Trend/MeanRev daily-return correlation: {corr:+.2f}")


def _mr_equity(A, idx, risk):
    """Recompute the mean-reversion sleeve's equity curve (mirrors swing_sweep)."""
    import swing_sweep as SW
    # Monkey-path: SW.simulate returns stats only; replicate its loop for equity.
    rsi_lo, sl_atr, rsi_exit = 10, 3.0, 50
    L = {s: (A[s]["close"] > A[s]["sma"]) & A[s]["calm"] & (A[s]["rsi"] < rsi_lo) for s in A}
    X = {s: (A[s]["rsi"] > rsi_exit) | (A[s]["close"] < A[s]["sma"]) for s in A}
    cash = 5000; pos = {}; pe, px = {}, set(); eq = np.empty(len(idx)); eq[:] = np.nan
    for i in range(len(idx)):
        for s in list(pos):
            p = pos[s]; lo = A[s]["low"][i]; op = A[s]["open"][i]
            if np.isnan(lo):
                continue
            p["bars"] += 1
            hit = op if op <= p["stop"] else (p["stop"] if lo <= p["stop"] else None)
            if hit is not None:
                cash += (hit - p["entry"]) * p["sh"]; del pos[s]
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
            if s in pos or np.isnan(A[s]["open"][i]) or np.isnan(A[s]["atr"][i]) or len(pos) >= 10:
                continue
            if orisk + risk*eqn > 0.10*eqn + 1e-9:
                continue
            entry = A[s]["open"][i]; stop = entry - sl_atr*A[s]["atr"][i]
            if entry <= stop:
                continue
            rd = risk*eqn; sh = rd/(entry-stop); cap = 0.20*eqn/entry
            if sh > cap:
                sh = cap; rd = sh*(entry-stop)
            pos[s] = dict(entry=entry, stop=stop, sh=sh, bars=0, risk=rd); orisk += rd
        pe = {}
        for s, p in pos.items():
            if X[s][i] or p["bars"] >= 10:
                px.add(s)
        for s in A:
            if L[s][i] and s not in pos:
                pe[s] = A[s]["rsi"][i]
        eq[i] = mk()
    return pd.Series(eq, index=idx).ffill().dropna()


if __name__ == "__main__":
    main()
