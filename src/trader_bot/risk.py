"""Risk management: position sizing, exposure/correlation limits and the
portfolio-level circuit breakers from STRATEGY.md Section 8.

The RiskManager owns account equity/balance and the set of open positions; the
backtester (and live runner) delegate every "may I open this?" and "how big?"
decision here so the rules live in exactly one place.
"""
from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd

from . import instruments as instr
from .config import Config
from .models import Position, Side


class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.starting_equity = cfg.starting_equity
        self.balance = cfg.starting_equity        # realized cash
        self.equity = cfg.starting_equity         # balance + open P&L (marked)
        self.high_water_mark = cfg.starting_equity
        self.halted = False                       # hard drawdown halt
        # Circuit-breaker anchors.
        self._day: pd.Timestamp | None = None
        self._week: tuple[int, int] | None = None
        self._day_start_equity = cfg.starting_equity
        self._week_start_equity = cfg.starting_equity
        self._entries_blocked_today = False
        self._entries_blocked_week = False

    # ------------------------------------------------------------------ time
    def roll_calendar(self, ts: pd.Timestamp) -> None:
        """Reset daily/weekly circuit breakers on day/week boundaries."""
        day = ts.normalize()
        if self._day is None or day != self._day:
            self._day = day
            self._day_start_equity = self.equity
            self._entries_blocked_today = False
        iso = ts.isocalendar()
        week = (iso[0], iso[1])
        if self._week is None or week != self._week:
            self._week = week
            self._week_start_equity = self.equity
            self._entries_blocked_week = False

    # --------------------------------------------------------------- equity
    def mark(self, open_pnl: float) -> None:
        """Update marked equity from realized balance + unrealized open P&L."""
        self.equity = self.balance + open_pnl
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity
        # Hard drawdown halt.
        dd = 1.0 - self.equity / self.high_water_mark
        if dd >= self.cfg.risk.max_drawdown_halt:
            self.halted = True

    def realize(self, pnl: float) -> None:
        """Book realized P&L to balance."""
        self.balance += pnl

    def sizing_equity(self) -> float:
        """Equity used for position sizing — floored at HWM * sizing_floor_dd so
        size doesn't collapse during a drawdown (anti-death-spiral), and never
        exceeds current equity."""
        floor = self.high_water_mark * self.cfg.risk.sizing_floor_dd
        return max(self.equity, floor)

    # ----------------------------------------------------------- breakers
    def update_breakers(self) -> None:
        r = self.cfg.risk
        day_loss = 1.0 - self.equity / self._day_start_equity
        week_loss = 1.0 - self.equity / self._week_start_equity
        if day_loss >= r.daily_loss_halt:
            self._entries_blocked_today = True
        if week_loss >= r.weekly_loss_halt:
            self._entries_blocked_week = True

    def entries_allowed(self) -> bool:
        return not (self.halted or self._entries_blocked_today or self._entries_blocked_week)

    # --------------------------------------------------------------- sizing
    def position_size(self, pair: str, entry_price: float, stop_distance: float) -> float:
        """Units to trade so that hitting the stop loses exactly risk_per_trade.

        Returns 0 if the trade can't be sized within the notional cap / min size.
        """
        r = self.cfg.risk
        risk_usd = self.sizing_equity() * r.risk_per_trade
        pip_val = instr.pip_value_per_unit(pair, entry_price, self.cfg.account_ccy)
        stop_pips = instr.price_to_pips(pair, stop_distance)
        if stop_pips <= 0 or pip_val <= 0:
            return 0.0
        units = risk_usd / (stop_pips * pip_val)

        # Notional cap: notional (in account ccy) <= notional_cap_mult * equity.
        notional_per_unit = self._notional_per_unit(pair, entry_price)
        max_units_cap = (r.notional_cap_mult * self.equity) / notional_per_unit
        units = min(units, max_units_cap)
        return math.floor(units)

    def _notional_per_unit(self, pair: str, price: float) -> float:
        """Account-ccy notional of one unit of base currency."""
        base, quote = pair.split("_")
        if base == self.cfg.account_ccy:
            return 1.0                      # USD/JPY: 1 unit = 1 USD notional
        if quote == self.cfg.account_ccy:
            return price                    # EUR/USD: 1 unit = price USD notional
        raise ValueError(f"cannot value notional for {pair}")

    # ---------------------------------------------------- exposure limits
    def can_open(self, pair: str, side: Side, open_positions: Dict[str, Position]) -> bool:
        r = self.cfg.risk
        if not self.entries_allowed():
            return False
        if len(open_positions) >= r.max_concurrent:
            return False
        # Max 1 position per pair.
        if sum(1 for p in open_positions.values() if p.pair == pair) >= r.max_per_pair:
            return False
        # Max positions sharing any single currency.
        base, quote = pair.split("_")
        for ccy in (base, quote):
            shared = sum(
                1 for p in open_positions.values() if ccy in (p.base, p.quote)
            )
            if shared >= r.max_per_currency:
                return False
        # Correlation buckets: never same direction on correlated pairs.
        if self._violates_correlation(pair, side, open_positions):
            return False
        return True

    def _violates_correlation(
        self, pair: str, side: Side, open_positions: Dict[str, Position]
    ) -> bool:
        for bucket in self.cfg.risk.correlation_buckets:
            if pair in bucket:
                for p in open_positions.values():
                    if p.pair in bucket and p.pair != pair and p.side == side:
                        return True
        return False
