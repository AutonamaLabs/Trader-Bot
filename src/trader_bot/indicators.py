"""Technical indicators used by the H4-Donchian-Trend strategy.

All functions are pure and operate on pandas Series / DataFrames. Everything is
computed on *closed* bars — the caller is responsible for never feeding a
still-forming bar into a signal decision.

Wilder's smoothing (RMA) is used for ATR and ADX to match how MetaTrader / most
brokers compute these, so backtest and live agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's running moving average (a.k.a. RMA / SMMA).

    Equivalent to an EMA with alpha = 1/period. Seeded with the simple mean of
    the first ``period`` values so results match reference platforms.
    """
    arr = series.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    if len(arr) < period:
        return pd.Series(out, index=series.index)
    # Seed with SMA of first `period` values.
    seed = np.nanmean(arr[:period])
    out[period - 1] = seed
    alpha = 1.0 / period
    for i in range(period, len(arr)):
        out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard exponential moving average (adjust=False, like trading platforms)."""
    return series.ewm(span=period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(H-L, |H-prevClose|, |L-prevClose|)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    tr = true_range(high, low, close)
    return wilder_rma(tr, period)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index (Wilder). Returns the ADX line only.

    ADX measures trend *strength* (not direction); values > ~20-25 indicate a
    market that is actually trending rather than chopping.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_ = wilder_rma(tr, period)

    # Avoid division by zero.
    atr_safe = atr_.replace(0.0, np.nan)
    plus_di = 100.0 * wilder_rma(plus_dm, period) / atr_safe
    minus_di = 100.0 * wilder_rma(minus_dm, period) / atr_safe

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return wilder_rma(dx.fillna(0.0), period)


def donchian(high: pd.Series, low: pd.Series, period: int = 20):
    """Donchian channel over the last ``period`` bars, EXCLUDING the current bar.

    Excluding the current bar is critical: the breakout trigger compares the
    current close to the extreme of the *prior* N bars, so a new high must
    exceed the channel that existed before this bar formed.

    Returns (upper, lower).
    """
    upper = high.shift(1).rolling(period).max()
    lower = low.shift(1).rolling(period).min()
    return upper, lower


def rolling_median(series: pd.Series, period: int) -> pd.Series:
    """Rolling median — used for the ATR volatility-regime baseline."""
    return series.rolling(period).median()
