"""Feature engineering: turn raw H4 OHLC into a frame carrying every column the
strategy state machine needs, including the no-lookahead D1 regime filter.

Output columns added to the H4 frame:
    donchian_up, donchian_dn      breakout channel (prior N bars)
    atr                           ATR(14) Wilder, H4
    adx                           ADX(14) Wilder, H4
    atr_median                    rolling median of ATR over N bars
    d1_close                      last COMPLETED daily close
    d1_ema                        D1 EMA(50) as of last completed day
    d1_ema_prev                   D1 EMA(50) `slope_lookback` days earlier
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .config import Config
from .data import resample_d1


def build_features(h4: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ic = cfg.indicators
    df = h4.copy()

    # --- H4 indicators -----------------------------------------------------
    up, dn = ind.donchian(df["high"], df["low"], ic.donchian_period)
    df["donchian_up"] = up
    df["donchian_dn"] = dn
    df["atr"] = ind.atr(df["high"], df["low"], df["close"], ic.atr_period)
    df["adx"] = ind.adx(df["high"], df["low"], df["close"], ic.adx_period)
    df["atr_median"] = ind.rolling_median(df["atr"], ic.atr_median_period)

    # --- D1 regime filter (mapped without lookahead) -----------------------
    daily = resample_d1(h4)
    daily_ema = ind.ema(daily["close"], ic.ema_filter_period)
    daily_ema_prev = daily_ema.shift(ic.ema_slope_lookback_days)

    d1 = pd.DataFrame(
        {
            "d1_close": daily["close"],
            "d1_ema": daily_ema,
            "d1_ema_prev": daily_ema_prev,
        }
    )
    # Carry the availability timestamp so the strategy can detect when a *new*
    # completed daily bar has arrived (regime-flip streak counting).
    d1["d1_available_at"] = d1.index
    # merge_asof (backward) maps each H4 bar to the most recent daily bar whose
    # availability timestamp is <= the H4 bar's timestamp => only completed days.
    df = pd.merge_asof(
        df.sort_index(),
        d1.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return df
