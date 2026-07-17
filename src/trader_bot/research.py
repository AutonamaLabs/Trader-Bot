"""Research harness: fast single-instrument event simulator + data loader +
strategy library, for hunting a real, out-of-sample-robust edge.

This is intentionally separate from the production ``backtest.py`` (which is a
faithful portfolio engine). Here we prioritise speed and breadth so we can sweep
many strategies/params/instruments/timeframes and validate honestly with
in-sample → out-of-sample splits.

Costs are charged on every fill. Sizing is fixed-fractional risk with
compounding, so results are comparable across instruments via R-multiples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "research" / "raw"

# Approx price levels used to auto-detect the point->price divisor.
_EXPECTED = {
    "EURUSD": 1.15, "GBPUSD": 1.4, "USDJPY": 105, "AUDUSD": 0.78, "USDCAD": 1.28,
    "USDCHF": 0.98, "EURJPY": 125, "GBPJPY": 155, "EURGBP": 0.82, "AUDJPY": 82,
    "XAUUSD": 1400,
}
# One-way cost (half-spread + slippage) expressed in PRICE units, realistic
# retail values. JPY pairs and gold have larger absolute pip sizes.
_COST_PRICE = {
    "EURUSD": 0.00010, "GBPUSD": 0.00013, "USDJPY": 0.010, "AUDUSD": 0.00013,
    "USDCAD": 0.00015, "USDCHF": 0.00015, "EURJPY": 0.014, "GBPJPY": 0.020,
    "EURGBP": 0.00015, "AUDJPY": 0.015, "XAUUSD": 0.20,
}


def cost_price(inst: str) -> float:
    return _COST_PRICE.get(inst, 0.00015)


def _auto_divisor(inst: str, median_close: float) -> float:
    expected = _EXPECTED.get(inst, 1.0)
    best, best_err = 1.0, float("inf")
    for d in (1, 10, 100, 1e3, 1e4, 1e5, 1e6):
        err = abs(np.log((median_close / d) / expected))
        if err < best_err:
            best, best_err = d, err
    return best


def load(inst: str, tf: str) -> pd.DataFrame:
    """Load one instrument/timeframe as UTC OHLC with prices auto-scaled."""
    path = RAW_DIR / f"{inst}_{tf}.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["date"], utc=True)
    d = _auto_divisor(inst, float(df["close"].median()))
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float) / d
    return df[["time", "open", "high", "low", "close"]].set_index("time").sort_index()


# ---------------------------------------------------------------------------
# Indicators (vectorised)
# ---------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return wilder(tr, n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = wilder(tr, n).replace(0, np.nan)
    pdi = 100 * wilder(pd.Series(plus, index=df.index), n) / atr_
    mdi = 100 * wilder(pd.Series(minus, index=df.index), n) / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx.fillna(0), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# Event simulator
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    r_multiples: np.ndarray
    equity: np.ndarray
    n: int
    win_rate: float
    profit_factor: float
    expectancy: float          # mean R
    total_return: float
    max_dd: float
    sharpe: float
    exposure: float            # fraction of bars in a position

    def as_row(self) -> dict:
        return {
            "n": self.n, "win_rate": round(self.win_rate * 100, 1),
            "pf": round(self.profit_factor, 3), "exp_R": round(self.expectancy, 4),
            "ret_%": round(self.total_return * 100, 1), "maxDD_%": round(self.max_dd * 100, 1),
            "sharpe": round(self.sharpe, 2),
        }


def simulate(
    df: pd.DataFrame,
    long_sig: np.ndarray,
    short_sig: np.ndarray,
    atr_arr: np.ndarray,
    inst: str,
    *,
    sl_atr: float = 2.0,
    tp_atr: Optional[float] = None,
    trail_atr: Optional[float] = None,
    time_stop: Optional[int] = None,
    exit_long: Optional[np.ndarray] = None,
    exit_short: Optional[np.ndarray] = None,
    exit_on_opposite: bool = False,
    risk_frac: float = 0.005,
    allow_short: bool = True,
    session_mask: Optional[np.ndarray] = None,
) -> SimResult:
    """Path-dependent one-position-at-a-time simulator.

    Signals are evaluated on bar close and filled at the NEXT bar's open. Stops
    and targets are checked intrabar (stop wins ambiguous bars). Costs charged
    both sides. Returns R-multiples and a compounding equity curve.
    """
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    n_bars = len(df)
    cost = cost_price(inst)

    equity = 1.0
    eq_curve = np.empty(n_bars); eq_curve[:] = np.nan
    r_list: List[float] = []
    in_pos = 0            # 0 flat, +1 long, -1 short
    entry = stop = 0.0
    r_price = 0.0
    extreme = 0.0
    bars_held = 0
    risk_at_entry = 0.0
    exposure_bars = 0

    for i in range(1, n_bars):
        if in_pos != 0:
            exposure_bars += 1
            bars_held += 1
            hi, lo = h[i], l[i]
            exit_price = None
            # 1) stop (intrabar)
            if in_pos > 0 and lo <= stop:
                exit_price = stop
            elif in_pos < 0 and hi >= stop:
                exit_price = stop
            # 2) take-profit (intrabar), only if stop not hit
            if exit_price is None and tp_atr is not None:
                tp = entry + in_pos * tp_atr * (r_price / sl_atr)  # tp_atr in ATR units
                if in_pos > 0 and hi >= tp:
                    exit_price = tp
                elif in_pos < 0 and lo <= tp:
                    exit_price = tp
            # update extreme + trailing
            if exit_price is None:
                if in_pos > 0:
                    extreme = max(extreme, hi)
                    if trail_atr is not None:
                        stop = max(stop, extreme - trail_atr * atr_arr[i])
                else:
                    extreme = min(extreme, lo)
                    if trail_atr is not None:
                        stop = min(stop, extreme + trail_atr * atr_arr[i])
                # re-check trailing stop against this bar
                if in_pos > 0 and lo <= stop:
                    exit_price = stop
                elif in_pos < 0 and hi >= stop:
                    exit_price = stop
            # 3) signal-based exits (on close)
            if exit_price is None:
                opp = (in_pos > 0 and short_sig[i]) or (in_pos < 0 and long_sig[i])
                sig_exit = ((in_pos > 0 and exit_long is not None and exit_long[i]) or
                            (in_pos < 0 and exit_short is not None and exit_short[i]))
                time_exit = time_stop is not None and bars_held >= time_stop
                if (exit_on_opposite and opp) or sig_exit or time_exit:
                    exit_price = c[i]
            if exit_price is not None:
                fill = exit_price - in_pos * cost
                pnl_price = (fill - entry) * in_pos
                r = pnl_price / r_price
                equity *= (1 + risk_frac * r)
                r_list.append(r)
                in_pos = 0
        # entries at this bar's open using previous bar's signal
        if in_pos == 0:
            sig = 0
            if long_sig[i - 1]:
                sig = 1
            elif short_sig[i - 1] and allow_short:
                sig = -1
            if sig != 0 and (session_mask is None or session_mask[i]):
                a = atr_arr[i - 1]
                if np.isfinite(a) and a > 0:
                    entry = o[i] + sig * cost
                    r_price = sl_atr * a
                    stop = entry - sig * r_price
                    extreme = entry
                    in_pos = sig
                    bars_held = 0
        eq_curve[i] = equity

    eq = pd.Series(eq_curve).ffill().fillna(1.0).to_numpy()
    r = np.array(r_list)
    if len(r) == 0:
        return SimResult(r, eq, 0, 0, 0, 0, 0, 0, 0, 0)
    wins = r[r > 0]; losses = r[r < 0]
    gross_w = wins.sum(); gross_l = -losses.sum()
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    dd = 1 - eq / np.maximum.accumulate(eq)
    # crude annualised sharpe from per-bar equity returns
    rets = np.diff(eq) / eq[:-1]
    bars_per_year = {"h1": 6000, "h4": 1500, "d1": 260}
    sh = 0.0
    if rets.std() > 0:
        sh = np.sqrt(1500) * rets.mean() / rets.std()
    return SimResult(
        r_multiples=r, equity=eq, n=len(r),
        win_rate=len(wins) / len(r), profit_factor=pf, expectancy=r.mean(),
        total_return=eq[-1] - 1, max_dd=dd.max(),
        sharpe=sh, exposure=exposure_bars / n_bars,
    )


def utc_hour_mask(df: pd.DataFrame, start_h: int, end_h: int) -> np.ndarray:
    """Boolean mask: True where the bar's UTC hour is in [start_h, end_h)."""
    hrs = df.index.hour.to_numpy()
    if start_h <= end_h:
        return (hrs >= start_h) & (hrs < end_h)
    return (hrs >= start_h) | (hrs < end_h)
