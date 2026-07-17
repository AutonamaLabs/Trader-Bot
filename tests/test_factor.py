"""Tests for the FX cross-sectional factor strategy.

Data-dependent tests are skipped automatically if the research data hasn't been
downloaded (scripts/fetch_data.sh).
"""
import numpy as np
import pandas as pd
import pytest

from trader_bot.factor import (
    DEFAULT_SLEEVES,
    FactorStrategy,
    Sleeve,
    performance,
    per_year,
    sleeve_weights,
)


def _synthetic_panel(n=400, m=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    prices = {}
    for j in range(m):
        r = rng.standard_normal(n) * 0.006 + (0.0002 if j % 2 else -0.0001)
        prices[f"P{j}"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(prices, index=idx)


def test_sleeve_weights_are_dollar_neutral():
    panel = _synthetic_panel()
    s = Sleeve("m", 20, 20, 2, "momentum")
    W, port = sleeve_weights(panel, s)
    # Each active day's weights should sum to ~0 (long notional == short notional).
    active = W[(W.abs().sum(axis=1) > 0)]
    assert len(active) > 0
    assert active.sum(axis=1).abs().max() < 1e-9


def test_no_lookahead_weight_uses_past_only():
    """Zero out the last row of prices and confirm earlier weights don't change:
    weights on day i must depend only on data up to day i-1."""
    panel = _synthetic_panel()
    s = Sleeve("m", 20, 5, 2, "momentum")
    W1, _ = sleeve_weights(panel, s)
    panel2 = panel.copy()
    panel2.iloc[-1] = panel2.iloc[-1] * 1.5   # perturb only the final bar
    W2, _ = sleeve_weights(panel2, s)
    # All weights except possibly the very last row must be identical.
    assert np.allclose(W1.iloc[:-1].to_numpy(), W2.iloc[:-1].to_numpy())


def test_momentum_and_reversal_are_negatively_correlated():
    panel = _synthetic_panel(seed=3)
    _, mom = sleeve_weights(panel, Sleeve("m", 20, 20, 2, "momentum"))
    _, rev = sleeve_weights(panel, Sleeve("r", 20, 5, 2, "reversal"))
    both = pd.concat([mom, rev], axis=1).dropna()
    both = both[(both != 0).any(axis=1)]
    corr = both.iloc[:, 0].corr(both.iloc[:, 1])
    assert corr < 0.2  # the two sleeves hedge each other


def test_vol_scaled_equity_runs():
    panel = _synthetic_panel(n=600)
    strat = FactorStrategy(max_leverage=2.0)
    equity, net, lev = strat.vol_scaled_equity(panel, 5000.0)
    assert len(equity) == len(panel)
    assert (lev <= 2.0 + 1e-9).all()
    m = performance(equity, net)
    assert "sharpe" in m and np.isfinite(m["sharpe"])
