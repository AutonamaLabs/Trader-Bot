#!/usr/bin/env python3
"""Exit-management experiments for TPS-5 — trailing stops, scale-in, scale-out.

TPS-5 is a MEAN-REVERSION system, so the received trend-following wisdom ("let
winners run", "pyramid winners") may not apply — the edge is the snap-back to the
mean. This tests each idea empirically on the stock universe, capital-aware, with
next-open fills, and reports PF / avg win / hold / CAGR / DD / Sharpe so we can
see what actually helps.

Variants:
  base           RSI>exit target exit (current tuned default)
  trail_replace  no target exit; ride a chandelier ATR trailing stop from entry
  trail_after    take no exit at target; switch to a chandelier trail (let it run)
  partial_trail  sell FRACTION at target, trail the remainder (scale-out)
  scalein_weak   add on further weakness (RSI<add_thresh), then normal exit
  combo          scalein_weak + partial_trail

    python scripts/swing_exits.py --dir data/swing_stocks --start 2004-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swing_sweep import precompute   # reuse one-time indicator precompute


def simulate(A, idx, mode, *, rsi_lo=10, rsi_exit=50, sl_atr=3.0, time_stop=10,
             trail_atr=3.0, partial_frac=0.5, add_thresh=5, max_adds=1, add_frac=1.0,
             equity=5000, risk=0.01, max_pos=10, max_total=0.10, max_notional=0.20):
    L, X = {}, {}
    for s, d in A.items():
        L[s] = (d["close"] > d["sma"]) & d["calm"] & (d["rsi"] < rsi_lo)
        X[s] = (d["rsi"] > rsi_exit)                      # target (reached the mean)
        # regime abort handled separately via close<sma
    use_trail = mode in ("trail_replace", "trail_after", "partial_trail", "combo")
    do_partial = mode in ("partial_trail", "combo")
    do_scalein = mode in ("scalein_weak", "combo")
    target_exits = mode in ("base", "scalein_weak")       # exit fully at target

    cash = equity; pos = {}; pnls = []; holds = []
    pend_entry = {}      # sym -> rsi at signal (rank key)
    pend_act = {}        # sym -> 'exit'|'partial'|'add'
    eq = np.empty(len(idx)); eq[:] = np.nan

    for i in range(len(idx)):
        # ---- 1) manage opens: stop (gap-aware) + trailing-stop update ----
        for s in list(pos):
            p = pos[s]; lo = A[s]["low"][i]; op = A[s]["open"][i]; hi = A[s]["high"][i]; atr = A[s]["atr"][i]
            if np.isnan(lo):
                continue
            p["bars"] += 1
            hit = op if op <= p["stop"] else (p["stop"] if lo <= p["stop"] else None)
            if hit is not None:
                v = (hit - p["entry"]) * p["sh"]; cash += v; pnls.append(v); holds.append(p["bars"]); del pos[s]; continue
            if mode == "trail_replace":
                p["trailing"] = True
            if not np.isnan(hi):
                p["ext"] = max(p["ext"], hi)
            if use_trail and p.get("trailing") and not np.isnan(atr):
                p["stop"] = max(p["stop"], p["ext"] - trail_atr * atr)

        # ---- 2) execute pending exits/partials/adds at today's open ----
        for s, act in list(pend_act.items()):
            del pend_act[s]; op = A[s]["open"][i]; atr = A[s]["atr"][i]
            if s not in pos or np.isnan(op):
                continue
            p = pos[s]
            if act == "exit":
                v = (op - p["entry"]) * p["sh"]; cash += v; pnls.append(v); holds.append(p["bars"]); del pos[s]
            elif act == "partial":
                sh = p["sh"] * partial_frac
                v = (op - p["entry"]) * sh; cash += v; pnls.append(v)
                p["sh"] -= sh; p["trailing"] = True
            elif act == "add" and not np.isnan(atr):
                eqn = _marked(cash, pos, A, i)
                add_sh = (risk * add_frac * eqn) / (sl_atr * atr)
                cap = max_notional * eqn / op
                add_sh = min(add_sh, max(0.0, cap - p["sh"]))
                if add_sh > 0:
                    p["entry"] = (p["entry"] * p["sh"] + op * add_sh) / (p["sh"] + add_sh)
                    p["sh"] += add_sh; p["adds"] += 1; p["risk"] += add_sh * (op - p["stop"])

        # ---- 3) execute pending entries at today's open (ranked, capped) ----
        for s in sorted(pend_entry, key=lambda x: pend_entry[x]):
            if s in pos or np.isnan(A[s]["open"][i]) or np.isnan(A[s]["atr"][i]):
                continue
            eqn = _marked(cash, pos, A, i)
            open_risk = sum(pp["risk"] for pp in pos.values())
            if len(pos) >= max_pos or open_risk + risk * eqn > max_total * eqn + 1e-9:
                continue
            entry = A[s]["open"][i]; stop = entry - sl_atr * A[s]["atr"][i]
            if entry <= stop:
                continue
            rd = risk * eqn; sh = rd / (entry - stop)
            capsh = max_notional * eqn / entry
            if sh > capsh:
                sh = capsh; rd = sh * (entry - stop)
            pos[s] = dict(entry=entry, stop=stop, sh=sh, bars=0, risk=rd,
                          ext=entry, adds=0, trailing=False, partialed=False)
        pend_entry = {}

        # ---- 4) signals on today's close -> pending for next open ----
        for s, p in list(pos.items()):
            cl = A[s]["close"][i]; sma = A[s]["sma"][i]
            if np.isnan(cl):
                continue
            if cl < sma or p["bars"] >= time_stop:
                pend_act[s] = "exit"; continue
            if X[s][i]:                                   # reached the mean
                if do_partial and not p["partialed"]:
                    pend_act[s] = "partial"; p["partialed"] = True
                elif target_exits:
                    pend_act[s] = "exit"
                elif mode == "trail_after":
                    p["trailing"] = True
                continue
            if do_scalein and p["adds"] < max_adds and A[s]["rsi"][i] < add_thresh:
                pend_act[s] = "add"
        for s in A:
            if L[s][i] and s not in pos:
                pend_entry[s] = A[s]["rsi"][i]
        eq[i] = _marked(cash, pos, A, i)

    return _finalize(eq, idx, pnls, holds)


def _marked(cash, pos, A, i):
    m = cash
    for s, p in pos.items():
        c = A[s]["close"][i]
        if not np.isnan(c):
            m += (c - p["entry"]) * p["sh"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/swing_stocks")
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--trail_atr", type=float, default=3.0)
    args = ap.parse_args()
    print("Loading + precomputing (once)...")
    A, idx = precompute(args.dir, args.start, args.end)
    print(f"{len(A)} symbols, {len(idx)} days {idx[0].date()}->{idx[-1].date()}\n")
    modes = ["base", "trail_replace", "trail_after", "partial_trail", "scalein_weak", "combo"]
    print(f"{'variant':<15}{'PF':>6}{'win%':>6}{'avgW':>7}{'avgL':>7}{'hold':>6}{'CAGR%':>7}{'DD%':>6}{'Shrp':>6}{'trades':>7}")
    for m in modes:
        r = simulate(A, idx, m, risk=args.risk, trail_atr=args.trail_atr)
        print(f"{m:<15}{r['pf']:>6.2f}{r['win']*100:>6.0f}{r['avgwin']:>7.0f}{r['avgloss']:>7.0f}"
              f"{r['hold']:>6.1f}{r['cagr']*100:>+7.1f}{r['dd']*100:>6.0f}{r['sharpe']:>6.2f}{r['n']:>7}")


def _finalize(eq, idx, pnls, holds):
    eqs = pd.Series(eq, index=idx).ffill().dropna()
    t = np.array(pnls); w = t[t > 0]; l = t[t < 0]
    pf = w.sum() / -l.sum() if l.sum() < 0 else float("inf")
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    cagr = (eqs.iloc[-1] / eqs.iloc[0]) ** (1 / yrs) - 1
    dd = (1 - eqs / eqs.cummax()).max()
    r = eqs.pct_change().dropna(); sh = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0
    return dict(pf=pf, cagr=cagr, dd=dd, sharpe=sh, n=len(t),
                win=len(w) / len(t) if len(t) else 0,
                avgwin=w.mean() if len(w) else 0, avgloss=l.mean() if len(l) else 0,
                hold=np.mean(holds) if holds else 0)


if __name__ == "__main__":
    main()
