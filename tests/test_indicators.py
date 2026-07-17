import numpy as np
import pandas as pd

from trader_bot import indicators as ind


def test_wilder_rma_constant_series():
    s = pd.Series([5.0] * 50)
    rma = ind.wilder_rma(s, 14)
    # A constant series smooths to the constant.
    assert np.isclose(rma.iloc[-1], 5.0)


def test_atr_positive_and_reasonable():
    n = 200
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.standard_normal(n)) * 0.001
    high = close + 0.0005
    low = close - 0.0005
    atr = ind.atr(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    assert atr.dropna().gt(0).all()
    # ATR should be roughly the bar range (~0.001) in magnitude.
    assert 0.0003 < atr.iloc[-1] < 0.005


def test_adx_bounded_0_100():
    n = 300
    rng = np.random.default_rng(1)
    close = 1.20 + np.cumsum(rng.standard_normal(n)) * 0.001
    high = close + abs(rng.standard_normal(n)) * 0.0005
    low = close - abs(rng.standard_normal(n)) * 0.0005
    adx = ind.adx(pd.Series(high), pd.Series(low), pd.Series(close), 14).dropna()
    assert adx.between(0, 100).all()


def test_adx_high_in_strong_trend():
    n = 200
    close = pd.Series(np.linspace(1.0, 1.2, n))  # pure uptrend
    high = close + 0.0002
    low = close - 0.0002
    adx = ind.adx(high, low, close, 14).dropna()
    assert adx.iloc[-1] > 40  # strong, persistent trend -> high ADX


def test_donchian_excludes_current_bar():
    high = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    low = pd.Series([0, 1, 2, 3, 4, 5], dtype=float)
    up, dn = ind.donchian(high, low, 3)
    # At index 5, channel high is max of highs[2:5] = max(3,4,5) = 5 (not 6).
    assert up.iloc[5] == 5.0
    assert dn.iloc[5] == 2.0
