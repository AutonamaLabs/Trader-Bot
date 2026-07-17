"""Shared data models for positions and signals."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Side(Enum):
    LONG = 1
    SHORT = -1

    @property
    def sign(self) -> int:
        return self.value


class PosState(Enum):
    FLAT = "FLAT"
    OPEN = "OPEN"            # full size, pre-partial
    PARTIAL_TAKEN = "PARTIAL_TAKEN"


@dataclass
class Signal:
    pair: str
    side: Side
    time: pd.Timestamp      # bar-close time the signal was generated on


@dataclass
class Position:
    pair: str
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    units: float                 # base-currency units (always positive)
    initial_units: float
    atr_at_entry: float
    r_price: float               # 1R in price terms (= sl_atr_mult * atr_at_entry)
    stop: float                  # current stop price (broker-side)
    state: PosState = PosState.OPEN
    bars_held: int = 0
    reached_1r: bool = False
    extreme_price: float = 0.0   # highest high (long) / lowest low (short) since entry
    regime_against_days: int = 0
    realized_pnl: float = 0.0    # from the partial fill, in account ccy
    tags: dict = field(default_factory=dict)

    @property
    def base(self) -> str:
        return self.pair.split("_")[0]

    @property
    def quote(self) -> str:
        return self.pair.split("_")[1]

    def signed_currencies(self) -> dict[str, int]:
        """Directional exposure per currency (base +side, quote -side)."""
        return {self.base: self.side.sign, self.quote: -self.side.sign}
