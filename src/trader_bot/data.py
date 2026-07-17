"""Market data: CSV loading for real broker history + a reproducible synthetic
regime-switching generator so the backtester runs with no broker connection.

Real data path: drop OANDA/Dukascopy H4 CSVs into ``data/<PAIR>_H4.csv`` with
columns [time, open, high, low, close] (UTC). ``load_csv`` reads them.

Synthetic path: ``generate_synthetic`` produces bars with alternating TREND and
RANGE regimes and volatility clustering, so the strategy's behaviour in both
trending and choppy markets can be observed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Rough spot levels for the v1 universe, used to anchor synthetic series.
_ANCHOR_PRICE = {
    "EUR_USD": 1.1000,
    "GBP_USD": 1.2700,
    "USD_JPY": 145.00,
    "AUD_USD": 0.6500,
}
# Approx annualised vol per pair (used to scale synthetic H4 returns).
_ANNUAL_VOL = {
    "EUR_USD": 0.075,
    "GBP_USD": 0.090,
    "USD_JPY": 0.095,
    "AUD_USD": 0.100,
}

# H4 bars per year on a Mon-Fri, 6-bars-per-day calendar (~260 weekdays * 6).
_BARS_PER_YEAR = 260 * 6


def load_csv(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    """Load an H4 OHLC CSV indexed by UTC timestamp.

    Expected columns (case-insensitive): time, open, high, low, close.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next(c for c in df.columns if c in ("time", "date", "datetime", "timestamp"))
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.rename(columns={time_col: "time"}).set_index("time").sort_index()
    return df[["open", "high", "low", "close"]].astype(float)


def _weekday_h4_index(start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    """Build a Mon-Fri, 6-bar/day H4 index (hours 0,4,8,12,16,20 UTC)."""
    stamps: list[pd.Timestamp] = []
    t = start
    while len(stamps) < periods:
        # Only weekdays (Mon=0 .. Fri=4); FX weekend is dropped for synthetic data.
        if t.weekday() < 5:
            stamps.append(t)
        t = t + pd.Timedelta(hours=4)
    return pd.DatetimeIndex(stamps)


def generate_synthetic(
    pair: str,
    years: float = 8.0,
    seed: int = 7,
    start: str = "2016-01-04",
) -> pd.DataFrame:
    """Generate reproducible H4 OHLC with regime switching.

    The hidden regime alternates between TREND (persistent drift) and RANGE
    (mean-reverting, ~zero drift). Volatility clusters via a slow-moving
    multiplier. Output is deterministic for a given (pair, seed).
    """
    rng = np.random.default_rng(seed + abs(hash(pair)) % 10_000)
    n = int(years * _BARS_PER_YEAR)

    anchor = _ANCHOR_PRICE[pair]
    ann_vol = _ANNUAL_VOL[pair]
    bar_vol = ann_vol / np.sqrt(_BARS_PER_YEAR)  # per-bar stdev of log returns

    # --- Build the hidden regime path -------------------------------------
    # Regimes last a random number of bars; trend regimes get a signed drift.
    drift = np.zeros(n)
    vol_mult = np.ones(n)
    i = 0
    while i < n:
        is_trend = rng.random() < 0.45
        length = int(rng.integers(60, 300))  # ~2-8 weeks per regime
        end = min(i + length, n)
        if is_trend:
            # Drift as a fraction of per-bar vol; sign random.
            sign = 1.0 if rng.random() < 0.5 else -1.0
            strength = rng.uniform(0.10, 0.35)
            drift[i:end] = sign * strength * bar_vol
        # Volatility regime: quiet vs active.
        vol_mult[i:end] = rng.uniform(0.7, 1.6)
        i = end

    # --- Generate log returns and price path ------------------------------
    shocks = rng.standard_normal(n) * bar_vol * vol_mult
    log_ret = drift + shocks
    log_price = np.log(anchor) + np.cumsum(log_ret)
    close = np.exp(log_price)
    open_ = np.empty(n)
    open_[0] = anchor
    open_[1:] = close[:-1]

    # Intrabar high/low: extend beyond the open/close range by a noise term.
    span = np.abs(close - open_)
    wick = np.abs(rng.standard_normal(n)) * bar_vol * vol_mult * close * 0.5
    high = np.maximum(open_, close) + wick + span * 0.1
    low = np.minimum(open_, close) - wick - span * 0.1

    idx = _weekday_h4_index(pd.Timestamp(start, tz="UTC"), n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )
    df.index.name = "time"
    return df


def resample_d1(h4: pd.DataFrame) -> pd.DataFrame:
    """Aggregate H4 bars to daily OHLC (UTC calendar day).

    The returned frame is indexed by the day's *availability* timestamp — the
    midnight AFTER the day closes — so that merging onto H4 bars can only ever
    surface fully completed daily bars (no lookahead).
    """
    daily = h4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    # Shift index to next-midnight availability.
    daily.index = daily.index + pd.Timedelta(days=1)
    daily.index.name = "available_at"
    return daily
