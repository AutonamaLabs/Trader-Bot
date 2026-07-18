"""Two-sleeve equity swing strategy — the single source of truth for both the
backtest and live execution.

Sleeve A (mean-reversion, TPS-5): buy RSI(3)<10 dips while Close>SMA200 and vol
is calm; exit RSI(3)>50 or Close<SMA200; 3xATR stop; 10-day time stop; one
scale-in add at RSI<5. Short holds (~3.6d) — CFD-financing-safe.

Sleeve B (trend): Donchian-200 breakout while above a rising SMA200; ride a
6xATR chandelier trailing stop. Long holds (months) — NOT recommended on a CFD
broker (financing eats the edge); include only on a cash-equity venue.

Live usage: feed the last ~300 daily (dividend/split-ADJUSTED) bars per symbol,
the current broker positions, and account equity; get back the day's actions.
Compute signals on YOUR adjusted data — never the broker's unadjusted CFD feed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class EquityParams:
    sma: int = 200
    rsi_n: int = 3
    rsi_lo: int = 10          # MR entry (oversold)
    rsi_exit: int = 50        # MR exit (reversion to mean)
    add_thresh: int = 5       # MR scale-in on further weakness
    max_adds: int = 1
    rv_spike: float = 3.0     # skip vol spikes
    don: int = 200            # trend breakout lookback
    sl_atr: float = 3.0
    trail_atr: float = 6.0    # trend chandelier trail
    time_stop: int = 10       # MR time stop (bars)
    risk_per_trade: float = 0.01
    max_positions: int = 10
    max_total_risk: float = 0.10
    max_notional: float = 0.20
    use_trend_sleeve: bool = False   # OFF by default on CFD brokers (financing)


@dataclass
class Action:
    kind: str          # OPEN | ADD | CLOSE
    symbol: str
    sleeve: str        # mr | trend
    units: float = 0.0
    stop: float = 0.0
    reason: str = ""


@dataclass
class LivePosition:
    symbol: str
    sleeve: str
    units: float
    entry: float
    stop: float
    bars_held: int = 0
    adds: int = 0
    extreme: float = 0.0     # highest high since entry (trend trail)


def _features(df: pd.DataFrame, p: EquityParams) -> dict:
    close = df["close"]
    sma = close.rolling(p.sma).mean()
    rsi = ind.rsi(close, p.rsi_n)
    atr = ind.atr(df["high"], df["low"], close, 14)
    lr = np.log(close / close.shift(1))
    rv5 = lr.rolling(5).std()
    calm = rv5 < p.rv_spike * rv5.rolling(100).median()
    don_hi = df["high"].shift(1).rolling(p.don).max()
    sma_rising = sma > sma.shift(20)
    last = -1
    return dict(
        close=float(close.iloc[last]), sma=float(sma.iloc[last]),
        rsi=float(rsi.iloc[last]), atr=float(atr.iloc[last]),
        calm=bool(calm.iloc[last]) if not pd.isna(calm.iloc[last]) else False,
        don_hi=float(don_hi.iloc[last]) if not pd.isna(don_hi.iloc[last]) else np.inf,
        sma_rising=bool(sma_rising.iloc[last]) if not pd.isna(sma_rising.iloc[last]) else False,
    )


def decide(bars: Dict[str, pd.DataFrame], positions: Dict[str, LivePosition],
           equity: float, p: EquityParams) -> List[Action]:
    """Return today's actions given adjusted daily bars, current positions and
    equity. Signals are read off the latest COMPLETED bar; the caller executes
    at the next session's open."""
    feats = {s: _features(df, p) for s, df in bars.items() if len(df) >= p.sma + p.don + 5}
    actions: List[Action] = []

    # --- exits / adds on existing positions ---
    for s, pos in positions.items():
        f = feats.get(s)
        if f is None:
            continue
        if pos.sleeve == "mr":
            if f["close"] < f["sma"] or f["rsi"] > p.rsi_exit or pos.bars_held >= p.time_stop:
                actions.append(Action("CLOSE", s, "mr", reason="target/regime/time"))
            elif pos.adds < p.max_adds and f["rsi"] < p.add_thresh:
                actions.append(Action("ADD", s, "mr", reason="further weakness"))
        else:  # trend: exit only on regime break (trail handled broker-side)
            if f["close"] < f["sma"]:
                actions.append(Action("CLOSE", s, "trend", reason="regime break"))

    # --- new entries, most-oversold first, subject to caps ---
    open_risk = sum(abs(q.units) * abs(q.entry - q.stop) for q in positions.values())
    slots = p.max_positions - len(positions)

    mr_cands = sorted(
        [(s, f) for s, f in feats.items()
         if s not in positions and f["close"] > f["sma"] and f["calm"] and f["rsi"] < p.rsi_lo],
        key=lambda x: x[1]["rsi"])
    trend_cands = []
    if p.use_trend_sleeve:
        trend_cands = [(s, f) for s, f in feats.items()
                       if s not in positions and f["close"] > f["don_hi"]
                       and f["close"] > f["sma"] and f["sma_rising"]]

    for sleeve, cands in (("mr", mr_cands), ("trend", trend_cands)):
        for s, f in cands:
            if slots <= 0:
                break
            risk_dollars = p.risk_per_trade * equity
            if open_risk + risk_dollars > p.max_total_risk * equity + 1e-9:
                break
            entry = f["close"]                    # proxy; caller fills at next open
            stop = entry - p.sl_atr * f["atr"]
            if entry <= stop:
                continue
            units = risk_dollars / (entry - stop)
            units = min(units, p.max_notional * equity / entry)
            if units <= 0:
                continue
            actions.append(Action("OPEN", s, sleeve, units=units, stop=stop, reason="signal"))
            open_risk += risk_dollars
            slots -= 1

    return actions
