"""H4-Donchian-Trend strategy logic: entry signal generation and per-bar exit
management. Pure functions over enriched feature rows and Position objects — the
backtester and live runner share this exact code so their behaviour matches.

See STRATEGY.md Sections 6 (entries) and 7 (exits).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import Config
from .models import PosState, Position, Side, Signal


def _finite(*vals: float) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)


# ---------------------------------------------------------------------------
# Entry signal generation
# ---------------------------------------------------------------------------
def indicator_filters_pass(row: pd.Series, cfg: Config) -> bool:
    """F6 (ADX) and F7 (ATR volatility band) — the indicator-based entry gates."""
    ic = cfg.indicators
    atr, adx, atr_med = row["atr"], row["adx"], row["atr_median"]
    if not _finite(atr, adx, atr_med) or atr_med <= 0:
        return False
    if adx <= ic.adx_min:
        return False
    ratio = atr / atr_med
    return ic.atr_band_low <= ratio <= ic.atr_band_high


def entry_signal(pair: str, row: pd.Series, cfg: Config) -> Optional[Signal]:
    """Return a directional Signal if the breakout + regime rules fire on this
    closed bar, else None. Portfolio/session/spread gates are applied elsewhere.
    """
    required = ["close", "donchian_up", "donchian_dn", "d1_close", "d1_ema", "d1_ema_prev"]
    if not _finite(*[row[c] for c in required]):
        return None
    if not indicator_filters_pass(row, cfg):
        return None

    close = row["close"]
    d1_up = row["d1_close"] > row["d1_ema"] and row["d1_ema"] > row["d1_ema_prev"]
    d1_dn = row["d1_close"] < row["d1_ema"] and row["d1_ema"] < row["d1_ema_prev"]

    # Long: breakout close above the prior-N-bar high, daily regime up + rising.
    if close > row["donchian_up"] and d1_up:
        return Signal(pair=pair, side=Side.LONG, time=row.name)
    # Short: mirror.
    if close < row["donchian_dn"] and d1_dn:
        return Signal(pair=pair, side=Side.SHORT, time=row.name)
    return None


def initial_stop(entry_price: float, side: Side, atr: float, cfg: Config) -> float:
    dist = cfg.exits.sl_atr_mult * atr
    return entry_price - dist * side.sign


# ---------------------------------------------------------------------------
# Exit management
# ---------------------------------------------------------------------------
@dataclass
class ExitEvent:
    kind: str          # "partial" | "close"
    price: float       # fill price (pre-slippage; engine applies costs)
    fraction: float    # fraction of the *remaining* position closed
    reason: str        # "stop" | "trail" | "partial_tp" | "time_stop" | "regime_flip"


def process_bar_exits(
    pos: Position,
    row: pd.Series,
    cfg: Config,
) -> List[ExitEvent]:
    """Advance a position through one H4 bar and return the fill events.

    Ordering within a bar (STRATEGY.md Section 7):
      1. stop / trailing stop  (intrabar)
      2. partial take-profit   (intrabar, only while state == OPEN)
      3. time stop             (on close)
      4. regime-flip exit      (on close)
    On an ambiguous bar (touches both stop and target) the stop fills first.
    Trailing stop for the remainder is ratcheted at the bar close for FUTURE bars.
    """
    ex = cfg.exits
    events: List[ExitEvent] = []
    sign = pos.side.sign
    high, low, close = row["high"], row["low"], row["close"]
    R = pos.r_price

    # Track whether +1R was ever reached (for the time stop).
    if sign > 0 and high >= pos.entry_price + R:
        pos.reached_1r = True
    elif sign < 0 and low <= pos.entry_price - R:
        pos.reached_1r = True

    # --- 1. Stop / trailing stop (intrabar) --------------------------------
    stop_hit = (low <= pos.stop) if sign > 0 else (high >= pos.stop)
    if stop_hit:
        events.append(ExitEvent("close", pos.stop, 1.0, "stop"))
        return events  # position fully closed

    # --- 2. Partial take-profit (intrabar, only pre-partial) ---------------
    if pos.state == PosState.OPEN:
        tp_price = pos.entry_price + ex.partial_tp_r * R * sign
        tp_hit = (high >= tp_price) if sign > 0 else (low <= tp_price)
        if tp_hit:
            events.append(ExitEvent("partial", tp_price, ex.partial_tp_fraction, "partial_tp"))
            pos.state = PosState.PARTIAL_TAKEN
            # Move stop to breakeven + token profit.
            pos.stop = pos.entry_price + ex.breakeven_offset_r * R * sign

    # --- update extreme excursion (for chandelier trail) -------------------
    if sign > 0:
        pos.extreme_price = max(pos.extreme_price, high)
    else:
        pos.extreme_price = min(pos.extreme_price, low)

    # --- 3. Chandelier trailing stop on the remainder (ratchet on close) ---
    if pos.state == PosState.PARTIAL_TAKEN and _finite(row["atr"]):
        trail = pos.extreme_price - ex.trail_atr_mult * row["atr"] * sign
        # Only ever tighten in the trade's favour.
        pos.stop = max(pos.stop, trail) if sign > 0 else min(pos.stop, trail)

    # --- 4. Time stop (on close) ------------------------------------------
    if pos.bars_held >= ex.time_stop_bars and not pos.reached_1r:
        events.append(ExitEvent("close", close, 1.0, "time_stop"))
        return events

    # --- 5. Regime-flip exit (on close) -----------------------------------
    if _finite(row.get("d1_close"), row.get("d1_ema")):
        against = (row["d1_close"] < row["d1_ema"]) if sign > 0 else (row["d1_close"] > row["d1_ema"])
        last_d1 = pos.tags.get("last_d1_available")
        cur_d1 = row.get("d1_available_at")
        if cur_d1 is not None and cur_d1 != last_d1:
            # New completed daily bar since last check -> update the streak.
            pos.tags["last_d1_available"] = cur_d1
            pos.regime_against_days = pos.regime_against_days + 1 if against else 0
        if pos.regime_against_days >= ex.regime_flip_days:
            events.append(ExitEvent("close", close, 1.0, "regime_flip"))
            return events

    return events
