#!/usr/bin/env python3
"""Point-in-time S&P 500 validation, 2000-2024 (docs/RESEARCH.md, "Point-in-time
S&P 500 validation"). Runs the MR sleeve (RSI(3)<10 dip / RSI(3)>50 or
close<SMA200 exit / 3x ATR stop / 10-day time stop -- see
src/trader_bot/equity_strategy.py) on data/universe_pit/prices, but unlike
validate_universe.py the tradable universe on any given day is gated by
data/universe_pit/membership.csv: a symbol can only be a NEW entry on dates it
was an actual S&P 500 constituent (no lookahead on index membership). Already
open positions run to their normal strategy exit even if the name later
leaves the index (see simulate_pit docstring).

Reuses scripts/swing_sweep.py's precompute() for indicators (same Wilder
RSI/ATR as the rest of the project) and mirrors its simulate() portfolio
engine, adding the membership gate and per-trade close dates so results can be
broken out by sub-period (2000-07, 2008-09 GFC, 2010-19, 2020 COVID, 2021-22
bear, 2023-24).

    python scripts/prepare_pit_universe.py       # once, after fetch_pit_universe.sh
    python scripts/validate_pit_universe.py --risk 0.01
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import swing_sweep as SW  # noqa: E402

PIT = ROOT / "data" / "universe_pit"

PERIODS = [
    ("2000-2007", "2000-01-01", "2007-12-31"),
    ("2008-2009 (GFC)", "2008-01-01", "2009-12-31"),
    ("2010-2019", "2010-01-01", "2019-12-31"),
    ("2020 (COVID)", "2020-01-01", "2020-12-31"),
    ("2021-2022 (bear)", "2021-01-01", "2022-12-31"),
    ("2023-2024", "2023-01-01", "2024-12-31"),
]


def load_membership(idx: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    with open(PIT / "membership.csv") as f:
        r = csv.reader(f)
        next(r)
        rows = [(pd.Timestamp(d, tz="UTC"), set(t.split(","))) for d, t in r]
    rows.sort(key=lambda x: x[0])
    change_dates = np.array([d.value for d, _ in rows], dtype="int64")
    sets = [s for _, s in rows]
    idx_vals = idx.values.astype("int64")
    pos = np.searchsorted(change_dates, idx_vals, side="right") - 1
    pos = np.clip(pos, 0, len(sets) - 1)
    all_syms = set()
    for s in sets:
        all_syms |= s
    membership = {}
    for sym in all_syms:
        in_set = np.fromiter((sym in s for s in sets), dtype=bool, count=len(sets))
        membership[sym] = in_set[pos]
    return membership


def simulate_pit(A, idx, membership, rsi_lo=10, sl_atr=3.0, rsi_exit=50,
                  equity=5000, risk=0.01, max_pos=10, max_total=0.10,
                  max_notional=0.20, time_stop=10):
    """swing_sweep.simulate's engine, plus a point-in-time membership gate on
    NEW entries. Positions already open are allowed to run to their normal
    strategy exit even if the name leaves the index mid-trade (MR holds are
    ~3-4 days on average, so this grace period is realistic and doesn't model
    forced-liquidation-on-deletion, which the underlying strategy doesn't
    otherwise represent). Returns the daily equity curve plus a (date, pnl)
    list per closed trade so results can be sliced by sub-period."""
    L, X = {}, {}
    for s, d in A.items():
        base_long = (d["close"] > d["sma"]) & d["calm"] & (d["rsi"] < rsi_lo)
        mem = membership.get(s)
        L[s] = base_long & mem if mem is not None else np.zeros(len(idx), dtype=bool)
        X[s] = (d["rsi"] > rsi_exit) | (d["close"] < d["sma"])
    cash = equity
    positions: dict = {}
    pe, px = {}, set()
    trades: list[tuple] = []  # (close_date, pnl)
    eq = np.empty(len(idx)); eq[:] = np.nan
    for i in range(len(idx)):
        for s in list(positions):
            p = positions[s]; lo = A[s]["low"][i]; op = A[s]["open"][i]
            if np.isnan(lo):
                continue
            p["bars"] += 1
            hit = op if op <= p["stop"] else (p["stop"] if lo <= p["stop"] else None)
            if hit is not None:
                v = (hit - p["entry"]) * p["sh"]; cash += v
                trades.append((idx[i], v)); del positions[s]
        for s in list(px):
            px.discard(s)
            if s in positions and not np.isnan(A[s]["open"][i]):
                p = positions[s]; v = (A[s]["open"][i] - p["entry"]) * p["sh"]
                cash += v; trades.append((idx[i], v)); del positions[s]

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
            if len(positions) >= max_pos or open_risk + risk * equity_now > max_total * equity_now + 1e-9:
                continue
            entry = A[s]["open"][i]; stop = entry - sl_atr * A[s]["atr"][i]
            if entry <= stop:
                continue
            rd = risk * equity_now; sh = rd / (entry - stop)
            cap = max_notional * equity_now / entry
            if sh > cap:
                sh = cap; rd = sh * (entry - stop)
            positions[s] = dict(entry=entry, stop=stop, sh=sh, bars=0, risk=rd)
            open_risk += rd
        pe = {}
        for s, p in positions.items():
            if X[s][i] or p["bars"] >= time_stop:
                px.add(s)
        for s in A:
            if L[s][i] and s not in positions:
                pe[s] = A[s]["rsi"][i]
        eq[i] = marked()
    eqs = pd.Series(eq, index=idx).ffill().dropna()
    return eqs, trades


def period_stats(eqs: pd.Series, trades: list, start=None, end=None):
    if start:
        eqs = eqs[eqs.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        eqs = eqs[eqs.index <= pd.Timestamp(end, tz="UTC")]
    if len(eqs) < 2:
        return None
    r = eqs.pct_change().dropna()
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    cagr = (eqs.iloc[-1] / eqs.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd = (1 - eqs / eqs.cummax()).max()
    sharpe = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0.0
    t = np.array([v for d, v in trades
                  if (start is None or d >= pd.Timestamp(start, tz="UTC"))
                  and (end is None or d <= pd.Timestamp(end, tz="UTC"))])
    wins, losses = t[t > 0], t[t < 0]
    pf = wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else float("inf")
    win_rate = len(wins) / len(t) if len(t) else 0.0
    return dict(cagr=cagr, sharpe=sharpe, dd=dd, pf=pf, win=win_rate, n=len(t),
                start=eqs.index[0].date(), end=eqs.index[-1].date())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PIT / "prices"))
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--rsi_lo", type=int, default=10)
    ap.add_argument("--sl_atr", type=float, default=3.0)
    ap.add_argument("--rsi_exit", type=int, default=50)
    args = ap.parse_args()

    print("Loading + precomputing indicators (once)...")
    A, idx = SW.precompute(args.dir, args.start, args.end)
    print(f"{len(A)} symbols, {len(idx)} days  {idx[0].date()} -> {idx[-1].date()}")

    print("Loading point-in-time S&P 500 membership...")
    membership = load_membership(idx)
    matched = len(set(A) & set(membership))
    print(f"{matched}/{len(A)} priced symbols matched in the membership file\n")

    eqs, trades = simulate_pit(A, idx, membership, rsi_lo=args.rsi_lo,
                                sl_atr=args.sl_atr, rsi_exit=args.rsi_exit,
                                risk=args.risk)

    print("=" * 78)
    print(f"  POINT-IN-TIME S&P 500 MR SLEEVE  (risk={args.risk}, "
          f"rsi_lo={args.rsi_lo}, sl_atr={args.sl_atr}, rsi_exit={args.rsi_exit})")
    print("=" * 78)
    print(f"  {'period':<20}{'CAGR%':>8}{'Sharpe':>8}{'maxDD%':>8}{'PF':>7}{'win%':>7}{'trades':>8}")
    full = period_stats(eqs, trades)
    print(f"  {'FULL ' + str(full['start']) + '->' + str(full['end']):<20}"
          f"{full['cagr']*100:>+8.1f}{full['sharpe']:>8.2f}{full['dd']*100:>8.0f}"
          f"{full['pf']:>7.2f}{full['win']*100:>7.0f}{full['n']:>8}")
    print("  " + "-" * 74)
    for name, s, e in PERIODS:
        st = period_stats(eqs, trades, s, e)
        if st is None:
            print(f"  {name:<20}  (no data)")
            continue
        print(f"  {name:<20}{st['cagr']*100:>+8.1f}{st['sharpe']:>8.2f}{st['dd']*100:>8.0f}"
              f"{st['pf']:>7.2f}{st['win']*100:>7.0f}{st['n']:>8}")
    print("=" * 78)


if __name__ == "__main__":
    main()
