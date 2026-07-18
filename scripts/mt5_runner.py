#!/usr/bin/env python3
"""Pepperstone MT5 demo forward-test runner (RUN THIS ON YOUR OWN MACHINE).

Per Fable's guidance:
  * Signals are computed from YOUR dividend/split-ADJUSTED daily CSVs (data/swing_stocks
    or data/universe_broad), NOT the broker's unadjusted CFD feed.
  * MT5 is used only for EXECUTION (positions, orders, live prices).
  * The mean-reversion sleeve is on by default (short holds, financing-safe). The
    trend sleeve is OFF by default (CFD overnight financing eats its months-long
    holds); enable only on a cash-equity venue.

Schedule this once per trading day AFTER the US cash close (signals use completed
daily bars; orders fill at the next session open). DRY-RUN by default — it logs
the actions it *would* take and places nothing. Pass --live to actually trade the
DEMO account.

    pip install MetaTrader5 pandas numpy pyyaml
    python scripts/mt5_runner.py                 # dry-run
    python scripts/mt5_runner.py --live          # place orders on the DEMO
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trader_bot.equity_strategy import EquityParams, LivePosition, decide
from swing_tps5 import load_daily

log = logging.getLogger("mt5_runner")

# Map your data symbol -> the broker's CFD instrument name (check in MT5).
SYMBOL_MAP_EXAMPLE = {"AAPL": "AAPL.US", "MSFT": "MSFT.US"}   # edit for Pepperstone


def load_signal_bars(data_dir: Path, tail: int = 320) -> dict:
    bars = {}
    for f in sorted(data_dir.glob("*.csv")):
        try:
            df = load_daily(f)
        except Exception:
            continue
        if len(df) >= 260:
            bars[f.stem.upper()] = df.tail(tail)
    return bars


def connect_mt5(login=None, password=None, server=None):
    import MetaTrader5 as mt5
    if not mt5.initialize(login=login, password=password, server=server):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("MT5 account_info() is None — not logged in.")
    if not getattr(info, "trade_mode", 0) == 0 and "demo" not in (server or "").lower():
        log.warning("Account may not be a DEMO — refusing to place live orders is your job.")
    return mt5, info


def read_positions(mt5, symbol_map) -> dict:
    inv = {v: k for k, v in symbol_map.items()}
    out = {}
    for pos in (mt5.positions_get() or []):
        sym = inv.get(pos.symbol, pos.symbol)
        out[sym] = LivePosition(symbol=sym, sleeve=pos.comment or "mr",
                                units=pos.volume, entry=pos.price_open, stop=pos.sl or 0.0)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "swing_stocks"))
    ap.add_argument("--live", action="store_true", help="place orders on the DEMO account")
    ap.add_argument("--use-trend", action="store_true", help="enable trend sleeve (NOT for CFDs)")
    ap.add_argument("--login", type=int, default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--server", default="Pepperstone-Demo")
    args = ap.parse_args()

    p = EquityParams(use_trend_sleeve=args.use_trend)
    bars = load_signal_bars(Path(args.data_dir))
    log.info("Loaded adjusted daily bars for %d symbols.", len(bars))

    if not args.live and "--force-connect" not in sys.argv:
        # Dry-run without a broker: assume flat, use a nominal equity.
        actions = decide(bars, positions={}, equity=5000.0, p=p)
        log.info("DRY-RUN (no broker). %d actions on the latest close:", len(actions))
        for a in actions:
            log.info("  %-5s %-6s %-5s units=%.2f stop=%.2f (%s)",
                     a.kind, a.symbol, a.sleeve, a.units, a.stop, a.reason)
        log.info("Connect to MT5 and pass --live to execute on the demo.")
        return

    symbol_map = SYMBOL_MAP_EXAMPLE  # EDIT this mapping for your broker
    mt5, info = connect_mt5(args.login, args.password, args.server)
    equity = float(info.equity)
    positions = read_positions(mt5, symbol_map)
    log.info("MT5 connected. equity=%.2f open_positions=%d", equity, len(positions))

    actions = decide(bars, positions, equity, p)
    for a in actions:
        broker_sym = symbol_map.get(a.symbol)
        if broker_sym is None:
            log.warning("  skip %s — no broker symbol mapping", a.symbol)
            continue
        log.info("  %-5s %-6s -> %-8s units=%.2f stop=%.2f", a.kind, a.symbol, broker_sym, a.units, a.stop)
        if not args.live:
            continue
        _execute(mt5, a, broker_sym)
    mt5.shutdown()


def _execute(mt5, a, broker_sym):
    """Place a market order / close on the DEMO. Uses attached SL for OPEN/ADD."""
    tick = mt5.symbol_info_tick(broker_sym)
    if a.kind in ("OPEN", "ADD"):
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": broker_sym,
            "volume": float(round(a.units, 2)), "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask, "sl": float(a.stop), "deviation": 20,
            "comment": a.sleeve, "type_filling": mt5.ORDER_FILLING_FOK,
        }
    else:  # CLOSE
        pos = next((x for x in (mt5.positions_get(symbol=broker_sym) or [])), None)
        if pos is None:
            return
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": broker_sym, "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL, "position": pos.ticket, "price": tick.bid,
            "deviation": 20, "comment": "exit", "type_filling": mt5.ORDER_FILLING_FOK,
        }
    res = mt5.order_send(req)
    log.info("    order_send retcode=%s", getattr(res, "retcode", "?"))


if __name__ == "__main__":
    main()
