#!/usr/bin/env python3
"""Capital-aware portfolio scanner for TPS-5.

The single-instrument edge is real but low-frequency, so capital sits idle. This
runs ONE compounding account across a whole universe: every day it scans all
symbols, opens the most-oversold valid signals up to concurrency/total-risk
caps, and manages exits — so capital is actually deployed and returns scale with
breadth. Reports the honest monthly-return distribution.

Signals fire on the daily close; fills at the next day's open. Stops are
intrabar with gap-through-open logic. Exits (RSI>exit or Close<SMA200) fill at
next open; a hard time stop and the ATR stop also apply.

    python scripts/swing_portfolio.py --dir data/swing --risk 0.01 --max_pos 10
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
from swing_tps5 import load_daily  # reuse the flexible loader


def prep(df, sma, rsi_n, rsi_lo, rsi_exit, rv_spike):
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"] = df["open"], df["high"], df["low"], close
    out["sma"] = close.rolling(sma).mean()
    out["rsi"] = R.rsi(close, rsi_n)
    out["atr"] = R.atr(df, 14)
    lr = np.log(close / close.shift(1))
    rv5 = lr.rolling(5).std()
    out["calm"] = rv5 < rv_spike * rv5.rolling(100).median()
    out["long"] = (close > out["sma"]) & out["calm"] & (out["rsi"] < rsi_lo)
    out["exit"] = (out["rsi"] > rsi_exit) | (close < out["sma"])
    return out


def run(dirpath, args):
    files = sorted(Path(dirpath).glob("*.csv"))
    data = {}
    for f in files:
        name = f.stem.upper()
        try:
            df = load_daily(f)
        except Exception:
            continue
        if len(df) >= args.sma + 60:
            data[name] = prep(df, args.sma, args.rsi_n, args.rsi_lo, args.rsi_exit, args.rv_spike)
    if not data:
        print(f"No usable data in {dirpath}"); return
    # Optional exclude list (e.g., crypto/gold that lose on this edge).
    for ex in args.exclude.split(",") if args.exclude else []:
        data.pop(ex.strip().upper(), None)

    all_dates = sorted(set().union(*[set(d.index) for d in data.values()]))
    idx = pd.DatetimeIndex(all_dates)
    if args.start:
        idx = idx[idx >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        idx = idx[idx <= pd.Timestamp(args.end, tz="UTC")]

    # Market regime gate: only allow entries when a market proxy is healthy.
    market_ok = np.ones(len(idx), bool)
    if args.market_filter:
        mkt = load_daily(Path(args.market_file))
        mclose = mkt["close"].reindex(idx, method="ffill")
        msma = mkt["close"].rolling(args.market_sma).mean().reindex(idx, method="ffill")
        market_ok = (mclose > msma).to_numpy()
    # Cache per-symbol row dicts for O(1) access.
    cols = ["open", "high", "low", "close", "atr", "rsi", "long", "exit"]
    A = {s: {c: d[c].reindex(idx).to_numpy() for c in cols} for s, d in data.items()}

    cash = args.equity
    positions = {}                 # sym -> dict(entry, stop, shares, bars, risk$)
    pend_entry, pend_exit = {}, set()   # pend_entry: sym -> rsi at signal (rank key)
    eq_curve = np.empty(len(idx)); eq_curve[:] = np.nan
    trade_pnls = []                # realized P&L per closed trade (for PF/win-rate)

    for i, dt in enumerate(idx):
        # ---- 1) manage open positions: stops intrabar (gap-aware) ----
        for s in list(positions.keys()):
            p = positions[s]; lo = A[s]["low"][i]; op = A[s]["open"][i]
            if np.isnan(lo):
                continue
            p["bars"] += 1
            hit = None
            if op <= p["stop"]:
                hit = op                       # gapped through the stop
            elif lo <= p["stop"]:
                hit = p["stop"]
            if hit is not None:
                pnl = (hit - p["entry"]) * p["shares"]
                cash += pnl; trade_pnls.append(pnl)
                del positions[s]

        # ---- 2) pending exits (signal on prior close) at today's open ----
        for s in list(pend_exit):
            pend_exit.discard(s)
            if s in positions and not np.isnan(A[s]["open"][i]):
                p = positions[s]
                pnl = (A[s]["open"][i] - p["entry"]) * p["shares"]
                cash += pnl; trade_pnls.append(pnl)
                del positions[s]

        # ---- mark equity (cash + open unrealized at today's close) ----
        def marked():
            m = cash
            for s, p in positions.items():
                c = A[s]["close"][i]
                if not np.isnan(c):
                    m += (c - p["entry"]) * p["shares"]
            return m
        equity = marked()

        # ---- 3) pending entries at today's open, subject to caps ----
        open_risk = sum(p["risk"] for p in positions.values())
        for s in sorted(pend_entry, key=lambda x: pend_entry[x]):   # most oversold first
            if s in positions or np.isnan(A[s]["open"][i]) or np.isnan(A[s]["atr"][i]):
                continue
            if len(positions) >= args.max_pos:
                continue
            if open_risk + args.risk * equity > args.max_total_risk * equity + 1e-9:
                continue
            entry = A[s]["open"][i]
            stop = entry - args.sl_atr * A[s]["atr"][i]
            if entry <= stop:
                continue
            risk_dollars = args.risk * equity
            shares = risk_dollars / (entry - stop)
            # Cap notional per position (tight stops must not create huge size).
            max_shares = args.max_notional * equity / entry
            if shares > max_shares:
                shares = max_shares
                risk_dollars = shares * (entry - stop)
            positions[s] = dict(entry=entry, stop=stop, shares=shares, bars=0, risk=risk_dollars)
            open_risk += risk_dollars
        pend_entry = {}   # unfilled signals are discarded, not queued (per spec)

        # ---- 4) generate signals on today's close for next-day action ----
        for s, p in positions.items():
            if A[s]["exit"][i] or p["bars"] >= args.time_stop:
                pend_exit.add(s)
        # entries: store RSI so tomorrow we fill the most-oversold first
        # (gated by the market regime filter, evaluated on the signal day)
        if market_ok[i]:
            for s in A:
                if A[s]["long"][i] and s not in positions:
                    pend_entry[s] = A[s]["rsi"][i]
        eq_curve[i] = marked()

    eq = pd.Series(eq_curve, index=idx).ffill().dropna()
    report(eq, args, trade_pnls)


def report(eq, args, trade_pnls=None):
    r = eq.pct_change().dropna()
    dd = (1 - eq / eq.cummax()).max()
    sharpe = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0
    monthly = eq.resample("ME").last().pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    print("=" * 64)
    print(f"  TPS-5 PORTFOLIO SCAN  (risk {args.risk:.1%}/trade, max {args.max_pos} pos)")
    print("=" * 64)
    print(f"  Period          : {eq.index[0].date()} -> {eq.index[-1].date()} ({yrs:.1f}y)")
    print(f"  Start -> End     : ${eq.iloc[0]:,.0f} -> ${eq.iloc[-1]:,.0f}")
    print(f"  CAGR            : {cagr*100:+.1f}%")
    print(f"  Max drawdown    : {dd*100:.1f}%")
    print(f"  Sharpe (daily)  : {sharpe:.2f}")
    if trade_pnls:
        t = np.array(trade_pnls)
        wins, losses = t[t > 0], t[t < 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        print(f"  Trades          : {len(t)}   win {len(wins)/len(t)*100:.0f}%   "
              f"profit factor {pf:.2f}   avg win ${wins.mean():.0f} / avg loss ${losses.mean():.0f}")
    print(f"  Monthly mean    : {monthly.mean()*100:+.2f}%   median {monthly.median()*100:+.2f}%   std {monthly.std()*100:.2f}%")
    print(f"  Positive months : {(monthly>0).mean()*100:.0f}%   best {monthly.max()*100:+.1f}%   worst {monthly.min()*100:+.1f}%")
    print(f"  Months >= +10%  : {(monthly>=0.10).mean()*100:.0f}%    >= +5%: {(monthly>=0.05).mean()*100:.0f}%")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/swing")
    ap.add_argument("--exclude", default="BTCUSD,ETHUSD,XAUUSD")
    ap.add_argument("--equity", type=float, default=5000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--max_pos", type=int, default=10)
    ap.add_argument("--max_total_risk", type=float, default=0.10)
    ap.add_argument("--max_notional", type=float, default=0.20, help="max notional/pos as frac of equity")
    ap.add_argument("--start", default=None, help="backtest start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="backtest end date (YYYY-MM-DD)")
    ap.add_argument("--market_filter", action="store_true", help="gate entries on market regime")
    ap.add_argument("--market_file", default="data/swing/SPY.csv")
    ap.add_argument("--market_sma", type=int, default=200)
    ap.add_argument("--sma", type=int, default=200)
    ap.add_argument("--rsi_n", type=int, default=3)
    ap.add_argument("--rsi_lo", type=int, default=10)
    ap.add_argument("--rsi_exit", type=int, default=50)
    ap.add_argument("--rv_spike", type=float, default=3.0)
    ap.add_argument("--sl_atr", type=float, default=3.0)
    ap.add_argument("--time_stop", type=int, default=10)
    args = ap.parse_args()
    run(args.dir, args)


if __name__ == "__main__":
    main()
