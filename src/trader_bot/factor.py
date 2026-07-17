"""FX cross-sectional factor strategy (Trader-Bot v2 — the researched edge).

Outright trend/mean-reversion on single FX pairs did not survive out-of-sample
(see docs/RESEARCH.md). What DID survive is a market-neutral, cross-sectional
factor: every day rank a basket by trailing return and hold a dollar-neutral
long/short book. We blend two negatively-correlated sleeves — medium-term
MOMENTUM (20d formation, 20d hold) and short-term REVERSAL (5-10d) — which
hedge each other and cut drawdown sharply.

Properties (2013-2022, 11-instrument basket, net of costs):
    Sharpe ~0.44, ~8/9 calendar years positive, max DD ~8%, robust to 2x costs
    and to dropping any instrument. Market-neutral => regime-independent.

This module is self-contained: build a price panel, compute daily target
weights, and either backtest to a $ equity curve (vol-targeted) or emit today's
target book for live execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# Default basket: 10 liquid FX pairs + gold. The edge is stable to changing it.
DEFAULT_BASKET = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
                  "USD_CHF", "EUR_JPY", "GBP_JPY", "EUR_GBP", "AUD_JPY", "XAU_USD"]

# One-way transaction cost as a fraction of notional (spread+slippage / price).
COST_FRAC = 0.00012


@dataclass
class Sleeve:
    name: str
    lookback: int
    hold: int
    k: int
    mode: str            # "momentum" | "reversal"


DEFAULT_SLEEVES = [
    Sleeve("MOM_20_20_3", 20, 20, 3, "momentum"),
    Sleeve("MOM_20_20_2", 20, 20, 2, "momentum"),
    Sleeve("REV_20_5_3", 20, 5, 3, "reversal"),
    Sleeve("REV_10_5_3", 10, 5, 3, "reversal"),
]


def sleeve_weights(panel: pd.DataFrame, s: Sleeve, cost_frac: float = COST_FRAC):
    """Daily target-weight matrix and net daily returns for one sleeve.

    No lookahead: weights on day i are formed from information up to day i-1 and
    earn day i's return. Positions rebalance every ``hold`` days; between
    rebalances weights are held constant.
    """
    rets = panel.pct_change()
    signal = panel.pct_change(s.lookback)
    vol = rets.rolling(20).std().replace(0, np.nan)

    dates = panel.index
    W = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    port = pd.Series(0.0, index=dates)
    last_w = pd.Series(0.0, index=panel.columns)

    for i in range(s.lookback + 1, len(dates)):
        cost = 0.0
        if (i - (s.lookback + 1)) % s.hold == 0:
            sig = signal.iloc[i - 1].dropna()
            if len(sig) >= 2 * s.k:
                ranked = sig.sort_values()
                losers, winners = ranked.index[:s.k], ranked.index[-s.k:]
                longs = winners if s.mode == "momentum" else losers
                shorts = losers if s.mode == "momentum" else winners
                w = pd.Series(0.0, index=panel.columns)
                iv = (1.0 / vol.iloc[i - 1])
                for grp, sign in [(longs, 1), (shorts, -1)]:
                    gv = iv[grp].fillna(0.0)
                    w[grp] = (sign * gv / gv.sum()).values if gv.sum() > 0 else sign / s.k
                cost = (w - last_w).abs().sum() * cost_frac
                last_w = w
        W.iloc[i] = last_w
        port.iloc[i] = (last_w * rets.iloc[i]).sum() - cost
    return W, port


@dataclass
class FactorStrategy:
    sleeves: List[Sleeve] = field(default_factory=lambda: list(DEFAULT_SLEEVES))
    cost_frac: float = COST_FRAC
    target_vol: float = 0.10        # annualised vol target
    max_leverage: float = 2.0       # conservative cap for a small account
    vol_lookback: int = 20

    def ensemble_returns(self, panel: pd.DataFrame):
        """Inverse-vol blend of sleeve daily returns + blended target weights."""
        pw, pr = {}, {}
        for s in self.sleeves:
            W, port = sleeve_weights(panel, s, self.cost_frac)
            pw[s.name], pr[s.name] = W, port
        R = pd.DataFrame(pr).dropna(how="all")
        iv = 1.0 / R.std()
        sw = iv / iv.sum()
        combo_ret = sum(R[name] * sw[name] for name in R.columns)
        # Blended target weights (same inverse-vol sleeve weights).
        combo_w = sum(pw[name] * sw[name] for name in R.columns)
        return combo_ret, combo_w, sw

    def vol_scaled_equity(self, panel: pd.DataFrame, starting_equity: float = 5000.0):
        """Backtest to a $ equity curve, scaling exposure to the vol target."""
        combo_ret, _, _ = self.ensemble_returns(panel)
        r = combo_ret.fillna(0.0)
        realized = r.rolling(self.vol_lookback).std() * np.sqrt(252)
        lev = (self.target_vol / realized).clip(upper=self.max_leverage).shift(1).fillna(0.0)
        net = lev * r
        equity = starting_equity * (1 + net).cumprod()
        return equity, net, lev

    def target_book_today(self, panel: pd.DataFrame, equity: float):
        """Return today's target $ notional per instrument for live execution."""
        combo_ret, combo_w, _ = self.ensemble_returns(panel)
        r = combo_ret.fillna(0.0)
        realized = r.rolling(self.vol_lookback).std().iloc[-1] * np.sqrt(252)
        lev = min(self.max_leverage, self.target_vol / realized) if realized > 0 else 0.0
        w_today = combo_w.iloc[-1]
        return (w_today * lev * equity).round(2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def performance(equity: pd.Series, net: pd.Series) -> dict:
    eq = equity.dropna()
    r = net.dropna()
    if len(eq) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    dd = (1 - eq / eq.cummax()).max()
    sharpe = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0.0
    return dict(
        final=float(eq.iloc[-1]), cagr=float(cagr), max_dd=float(dd),
        sharpe=float(sharpe), vol=float(r.std() * np.sqrt(252)),
    )


def per_year(net: pd.Series) -> pd.Series:
    return net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)
