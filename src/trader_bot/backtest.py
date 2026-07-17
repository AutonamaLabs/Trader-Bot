"""Event-driven portfolio backtester for H4-Donchian-Trend.

Steps bar-by-bar across ALL pairs on a unified timeline so portfolio-level rules
(max concurrent, currency exposure, circuit breakers, compounding equity) are
enforced exactly as they would be live. Signals fire on a closed bar and fill at
the next bar's open; spread + slippage are charged on every fill; on an
ambiguous bar the stop is assumed to fill before the target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import instruments as instr
from . import sessions as sess
from .config import Config
from .models import PosState, Position, Side, Signal
from .risk import RiskManager
from .strategy import ExitEvent, entry_signal, initial_stop, process_bar_exits


@dataclass
class Trade:
    pair: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    r_multiple: float
    reason: str
    bars_held: int


@dataclass
class PendingEntry:
    signal: Signal
    atr: float


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    config: Config
    final_equity: float
    starting_equity: float


class Backtester:
    def __init__(self, cfg: Config, data: Dict[str, pd.DataFrame]):
        self.cfg = cfg
        self.data = data  # pair -> enriched H4 feature frame
        self.risk = RiskManager(cfg)
        self.positions: Dict[str, Position] = {}
        self.pending: Dict[str, PendingEntry] = {}
        self.trades: List[Trade] = []
        self.last_close: Dict[str, float] = {}
        self._equity_times: List[pd.Timestamp] = []
        self._equity_vals: List[float] = []

    # ---------------------------------------------------------------- costs
    def _half_spread_price(self, pair: str) -> float:
        pips = self.cfg.costs.typical_spread_pips[pair] / 2.0
        return instr.pips_to_price(pair, pips)

    def _slippage_price(self, pair: str) -> float:
        return instr.pips_to_price(pair, self.cfg.costs.slippage_pips)

    def _entry_fill(self, pair: str, side: Side, raw: float) -> float:
        """Adverse fill: buy at ask (+), sell at bid (-), plus slippage."""
        cost = self._half_spread_price(pair) + self._slippage_price(pair)
        return raw + cost * side.sign

    def _exit_fill(self, pair: str, side: Side, raw: float, market: bool) -> float:
        """Closing fill: long closes at bid (-), short at ask (+).
        Market/stop exits also pay slippage; limit (partial TP) exits do not."""
        cost = self._half_spread_price(pair) + (self._slippage_price(pair) if market else 0.0)
        # Closing a long is a sell (adverse = lower); closing a short is a buy (higher).
        return raw - cost * side.sign

    # ------------------------------------------------------------------ pnl
    def _pnl(self, pair: str, side: Side, entry: float, exit_: float, units: float) -> float:
        gross = (exit_ - entry) * side.sign * units
        base, quote = pair.split("_")
        if quote == self.cfg.account_ccy:
            return gross                       # e.g. EUR/USD -> already USD
        if base == self.cfg.account_ccy:
            return gross / exit_               # e.g. USD/JPY -> JPY converted to USD
        raise ValueError(f"pnl for {pair} unsupported")

    # --------------------------------------------------------------- filters
    def _spread_ok(self, pair: str, atr: float) -> bool:
        c = self.cfg.costs
        spread = c.typical_spread_pips[pair]
        if spread > c.max_spread_pips[pair]:
            return False
        stop_pips = instr.price_to_pips(pair, self.cfg.exits.sl_atr_mult * atr)
        return stop_pips > 0 and spread < c.spread_stop_ratio_max * stop_pips

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        # Unified, sorted timeline of (timestamp, pair) bar events.
        events: List[tuple[pd.Timestamp, str]] = []
        for pair, df in self.data.items():
            for ts in df.index:
                events.append((ts, pair))
        events.sort(key=lambda e: (e[0], e[1]))

        # Fast row lookup.
        row_cache = {pair: df for pair, df in self.data.items()}

        current_ts: Optional[pd.Timestamp] = None
        for ts, pair in events:
            if ts != current_ts:
                # New timestamp: roll circuit-breaker calendar once.
                self.risk.roll_calendar(ts)
                current_ts = ts

            row = row_cache[pair].loc[ts]
            self.last_close[pair] = row["close"]

            # 1) Manage an existing open position through this bar.
            if pair in self.positions:
                self._manage_position(pair, ts, row)

            # 2) Fill a pending entry at this bar's open.
            if pair in self.pending and pair not in self.positions:
                self._fill_pending(pair, ts, row)

            # 3) Generate a fresh signal on this closed bar for next-bar entry.
            if pair not in self.positions and pair not in self.pending:
                self._maybe_signal(pair, ts, row)

            # Mark portfolio equity & breakers after each event.
            self._mark(ts)

        # Force-close anything still open at the end (mark-to-last-close).
        self._final_liquidation()
        equity = pd.Series(self._equity_vals, index=pd.DatetimeIndex(self._equity_times))
        equity = equity[~equity.index.duplicated(keep="last")]
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity,
            config=self.cfg,
            final_equity=self.risk.equity,
            starting_equity=self.cfg.starting_equity,
        )

    # ---------------------------------------------------------- internals
    def _manage_position(self, pair: str, ts: pd.Timestamp, row: pd.Series) -> None:
        pos = self.positions[pair]
        pos.bars_held += 1
        events = process_bar_exits(pos, row, self.cfg)
        for ev in events:
            self._apply_exit(pos, ev, ts)
            if pair not in self.positions:
                break

    def _apply_exit(self, pos: Position, ev: ExitEvent, ts: pd.Timestamp) -> None:
        market = ev.reason != "partial_tp"
        fill = self._exit_fill(pos.pair, pos.side, ev.price, market)
        if ev.kind == "partial":
            close_units = pos.units * ev.fraction
            pnl = self._pnl(pos.pair, pos.side, pos.entry_price, fill, close_units)
            self.risk.realize(pnl)
            pos.realized_pnl += pnl
            pos.units -= close_units
        else:  # full close of remaining units
            pnl = self._pnl(pos.pair, pos.side, pos.entry_price, fill, pos.units)
            self.risk.realize(pnl)
            total_pnl = pos.realized_pnl + pnl
            r_mult = total_pnl / max(self._trade_risk_usd(pos), 1e-9)
            self.trades.append(
                Trade(
                    pair=pos.pair,
                    side=pos.side.name,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=fill,
                    units=pos.initial_units,
                    pnl=total_pnl,
                    r_multiple=r_mult,
                    reason=ev.reason,
                    bars_held=pos.bars_held,
                )
            )
            del self.positions[pos.pair]

    def _trade_risk_usd(self, pos: Position) -> float:
        """The $ that 1R represented at entry (for R-multiple bookkeeping)."""
        stop_dist = abs(pos.entry_price - initial_stop(pos.entry_price, pos.side, pos.atr_at_entry, self.cfg))
        pip_val = instr.pip_value_per_unit(pos.pair, pos.entry_price, self.cfg.account_ccy)
        stop_pips = instr.price_to_pips(pos.pair, stop_dist)
        return stop_pips * pip_val * pos.initial_units

    def _fill_pending(self, pair: str, ts: pd.Timestamp, row: pd.Series) -> None:
        pend = self.pending.pop(pair)
        sig = pend.signal
        # Re-check portfolio gates at fill time (state may have changed).
        if not self.risk.can_open(pair, sig.side, self.positions):
            return
        raw_open = row["open"]
        entry = self._entry_fill(pair, sig.side, raw_open)
        atr = pend.atr
        stop = initial_stop(entry, sig.side, atr, self.cfg)
        stop_dist = abs(entry - stop)
        units = self.risk.position_size(pair, entry, stop_dist)
        if units <= 0:
            return
        pos = Position(
            pair=pair,
            side=sig.side,
            entry_time=ts,
            entry_price=entry,
            units=units,
            initial_units=units,
            atr_at_entry=atr,
            r_price=self.cfg.exits.sl_atr_mult * atr,
            stop=stop,
            state=PosState.OPEN,
            extreme_price=entry,
        )
        pos.tags["last_d1_available"] = row.get("d1_available_at")
        self.positions[pair] = pos

    def _maybe_signal(self, pair: str, ts: pd.Timestamp, row: pd.Series) -> None:
        if not self.risk.entries_allowed():
            return
        # Entry executes on the NEXT bar; session gate uses the next bar's hour.
        next_open_ts = ts + pd.Timedelta(hours=4)
        if not sess.can_enter_now(next_open_ts, self.cfg):
            return
        sig = entry_signal(pair, row, self.cfg)
        if sig is None:
            return
        if not self._spread_ok(pair, row["atr"]):
            return
        if not self.risk.can_open(pair, sig.side, self.positions):
            return
        self.pending[pair] = PendingEntry(signal=sig, atr=row["atr"])

    def _open_pnl(self) -> float:
        total = 0.0
        for pair, pos in self.positions.items():
            price = self.last_close.get(pair, pos.entry_price)
            total += self._pnl(pair, pos.side, pos.entry_price, price, pos.units)
        return total

    def _mark(self, ts: pd.Timestamp) -> None:
        self.risk.mark(self._open_pnl())
        self.risk.update_breakers()
        self._equity_times.append(ts)
        self._equity_vals.append(self.risk.equity)

    def _final_liquidation(self) -> None:
        for pair in list(self.positions.keys()):
            pos = self.positions[pair]
            price = self.last_close.get(pair, pos.entry_price)
            self._apply_exit(pos, ExitEvent("close", price, 1.0, "eod_liquidation"),
                             self._equity_times[-1] if self._equity_times else pos.entry_time)
        self.risk.mark(0.0)
