#!/usr/bin/env python3
"""TPS-5 — Trend-Pullback Swing (Fable design), multi-asset daily.

Long-only: buy short-term weakness (RSI-3 oversold) inside a confirmed uptrend
(Close > SMA200), skip vol spikes, exit into strength (RSI-3 > 65), regime abort
(Close < SMA200), 3xATR stop, 10-day time stop. Tested on the RIGHT asset class
for this anomaly: equity indices, gold, crypto — NOT FX majors.

Reads daily OHLC CSVs from data/swing/<NAME>.csv (flexible header). Reports
per-instrument and pooled stats, per-year, and the monthly return distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot import research as R  # reuse simulate + indicators

SWING_DIR = ROOT / "data" / "swing"

# Approx one-way cost in PRICE units (bp x typical price); negligible vs ATR here.
COST = {"SPY": 0.03, "QQQ": 0.03, "US500": 0.5, "NAS100": 2.0,
        "XAUUSD": 0.25, "BTCUSD": 20.0, "ETHUSD": 1.5}


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in df.columns if c in ("date", "time", "timestamp", "datetime"))
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).rename(columns={tcol: "time"})
    # Map OHLC columns, tolerating 'adj close'/'price'/close-only sources.
    cols = {}
    for name in ("open", "high", "low", "close"):
        if name in df.columns:
            cols[name] = name
    if "close" not in cols:
        for alt in ("adj close", "adjclose", "price", "last"):
            if alt in df.columns:
                cols["close"] = alt; break
    out = df[["time"] + [cols[c] for c in ("open", "high", "low", "close") if c in cols]].copy()
    out.columns = ["time"] + [c for c in ("open", "high", "low", "close") if c in cols]
    for c in ("open", "high", "low", "close"):
        if c not in out.columns:
            out[c] = out["close"]  # some sources are close-only
    out = out.dropna(subset=["close"]).sort_values("time").drop_duplicates("time")
    return out.set_index("time")[["open", "high", "low", "close"]].astype(float)


def signals(df, sma=200, rsi_n=3, rsi_lo=15, rsi_exit=65, rv_spike=3.0):
    close = df["close"]
    sma200 = close.rolling(sma).mean()
    rsi = R.rsi(close, rsi_n)
    atr = R.atr(df, 14)
    logret = np.log(close / close.shift(1))
    rv5 = logret.rolling(5).std()
    rv_med = rv5.rolling(100).median()

    uptrend = close > sma200
    calm = rv5 < rv_spike * rv_med
    long_sig = (uptrend & calm & (rsi < rsi_lo)).to_numpy()
    exit_long = ((rsi > rsi_exit) | (close < sma200)).to_numpy()
    short_sig = np.zeros(len(df), bool)
    return long_sig, short_sig, atr.to_numpy(), exit_long


def run_instrument(name, df, args):
    long_sig, short_sig, atr, exit_long = signals(
        df, args.sma, args.rsi_n, args.rsi_lo, args.rsi_exit, args.rv_spike)
    # patch cost for this instrument into research module
    R._COST_PRICE[name] = COST.get(name, 0.0005 * float(df["close"].median()))
    res = R.simulate(
        df, long_sig, short_sig, atr, name,
        sl_atr=args.sl_atr, exit_long=exit_long, time_stop=args.time_stop,
        allow_short=False, risk_frac=args.risk,
    )
    return res


def monthly_stats(equity_index, r_multiples, per_inst_returns):
    pass  # (portfolio monthly handled in main via combined equity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sma", type=int, default=200)
    ap.add_argument("--rsi_n", type=int, default=3)
    ap.add_argument("--rsi_lo", type=int, default=15)
    ap.add_argument("--rsi_exit", type=int, default=65)
    ap.add_argument("--rv_spike", type=float, default=3.0)
    ap.add_argument("--sl_atr", type=float, default=3.0)
    ap.add_argument("--time_stop", type=int, default=10)
    ap.add_argument("--risk", type=float, default=0.0075)
    args = ap.parse_args()

    files = sorted(SWING_DIR.glob("*.csv"))
    if not files:
        print(f"No data in {SWING_DIR}. Populate it first (see scripts/fetch_swing_data.sh).")
        return
    print(f"{'instrument':<10}{'trades':>7}{'win%':>7}{'pf':>6}{'expR':>7}{'ret%':>9}{'DD%':>7}{'range'}")
    all_r = []
    port_daily = {}
    for f in files:
        name = f.stem.upper()
        df = load_daily(f)
        if len(df) < 260:
            continue
        res = run_instrument(name, df, args)
        all_r.append((name, res))
        eq = pd.Series(res.equity, index=df.index)
        port_daily[name] = eq.pct_change().fillna(0.0)
        print(f"{name:<10}{res.n:>7}{res.win_rate*100:>6.1f}{res.profit_factor:>6.2f}"
              f"{res.expectancy:>+7.3f}{res.total_return*100:>+8.1f}{res.max_dd*100:>6.1f}  "
              f"{df.index[0].date()}->{df.index[-1].date()}")

    if not port_daily:
        return
    # Equal-weight portfolio of instrument equity curves.
    P = pd.DataFrame(port_daily).fillna(0.0)
    port = P.mean(axis=1)
    eq = (1 + port).cumprod()
    dd = (1 - eq / eq.cummax()).max()
    sharpe = np.sqrt(252) * port.mean() / port.std() if port.std() > 0 else 0
    monthly = eq.resample("ME").last().pct_change().dropna()
    print("\n" + "=" * 62)
    print("  TPS-5 EQUAL-WEIGHT PORTFOLIO")
    print("=" * 62)
    print(f"  Period         : {eq.index[0].date()} -> {eq.index[-1].date()}")
    print(f"  Total return   : {(eq.iloc[-1]-1)*100:+.1f}%")
    yrs = (eq.index[-1]-eq.index[0]).days/365.25
    print(f"  CAGR           : {((eq.iloc[-1])**(1/yrs)-1)*100:+.1f}%")
    print(f"  Max drawdown   : {dd*100:.1f}%")
    print(f"  Sharpe (daily) : {sharpe:.2f}")
    print(f"  Monthly return : mean {monthly.mean()*100:+.2f}%  median {monthly.median()*100:+.2f}%  "
          f"std {monthly.std()*100:.2f}%")
    print(f"  Positive months: {(monthly>0).mean()*100:.0f}%   best {monthly.max()*100:+.1f}%   "
          f"worst {monthly.min()*100:+.1f}%")
    print(f"  Months >= +10% : {(monthly>=0.10).mean()*100:.0f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()
