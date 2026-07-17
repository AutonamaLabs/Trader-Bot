"""Live / dry-run trading engine.

Designed to be invoked ONCE per H4 close (e.g. from cron at 00/04/08/12/16/20
UTC + a minute). It pulls the latest completed candles, rebuilds features, and
reconciles desired vs. actual positions through the Broker interface. It reuses
the exact signal/exit logic from ``strategy.py`` so live behaviour matches the
backtest.

Defaults to DRY-RUN: it logs the actions it *would* take and places nothing.
Pass ``--live`` (and provide broker credentials) to actually trade — start on a
practice account.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trader_bot import instruments as instr
from trader_bot import sessions as sess
from trader_bot.broker import Broker, PaperBroker
from trader_bot.config import Config, load_config
from trader_bot.features import build_features
from trader_bot.risk import RiskManager
from trader_bot.strategy import entry_signal, indicator_filters_pass, initial_stop

log = logging.getLogger("trader_bot.live")


@dataclass
class Action:
    kind: str            # "OPEN" | "CLOSE" | "MODIFY_STOP" | "SKIP"
    pair: str
    detail: str


class LiveEngine:
    # How many H4 bars of history to pull (enough for D1 EMA50 + ATR median).
    HISTORY_BARS = 900

    def __init__(self, broker: Broker, cfg: Config, dry_run: bool = True):
        self.broker = broker
        self.cfg = cfg
        self.dry_run = dry_run
        self.risk = RiskManager(cfg)

    def _features_for(self, pair: str) -> pd.DataFrame:
        candles = self.broker.get_candles(pair, self.cfg.signal_tf, self.HISTORY_BARS)
        return build_features(candles, self.cfg)

    def run_once(self, now: Optional[pd.Timestamp] = None) -> List[Action]:
        """Evaluate every pair against the latest CLOSED bar; return actions."""
        now = now or pd.Timestamp.now(tz="UTC")
        self.risk.roll_calendar(now)
        equity = self.broker.account_equity()
        self.risk.equity = equity
        self.risk.high_water_mark = max(self.risk.high_water_mark, equity)

        broker_pos = self.broker.open_positions()
        actions: List[Action] = []

        for pair in self.cfg.pairs:
            feats = self._features_for(pair)
            if len(feats) < 2:
                continue
            row = feats.iloc[-1]  # latest COMPLETE bar

            if pair in broker_pos:
                actions.append(self._manage_open(pair, row, broker_pos[pair]))
            else:
                act = self._maybe_open(pair, row, now, broker_pos)
                if act:
                    actions.append(act)
                    # Reflect the intended open so exposure limits see it.
                    if act.kind == "OPEN":
                        broker_pos[pair] = None  # placeholder for count

        for a in actions:
            log.info("%s %s — %s", a.kind, a.pair, a.detail)
        return actions

    def _maybe_open(self, pair, row, now, broker_pos) -> Optional[Action]:
        if not self.risk.entries_allowed():
            return None
        next_open = now.floor("4h") + pd.Timedelta(hours=4)
        if not sess.can_enter_now(next_open, self.cfg):
            return None
        sig = entry_signal(pair, row, self.cfg)
        if sig is None:
            return None
        # Spread gate using the live quote.
        bid, ask = self.broker.get_price(pair)
        spread_pips = instr.price_to_pips(pair, ask - bid)
        if spread_pips > self.cfg.costs.max_spread_pips[pair]:
            return Action("SKIP", pair, f"spread {spread_pips:.1f}p too wide")
        # Exposure limits (count placeholders too).
        current = {p: v for p, v in broker_pos.items() if v is not None}
        if not self.risk.can_open(pair, sig.side, current):
            return Action("SKIP", pair, "blocked by exposure/risk limits")

        atr = row["atr"]
        entry = ask if sig.side.sign > 0 else bid
        stop = initial_stop(entry, sig.side, atr, self.cfg)
        units = self.risk.position_size(pair, entry, abs(entry - stop)) * sig.side.sign
        if abs(units) < 1:
            return Action("SKIP", pair, "size rounds to zero")

        detail = f"{sig.side.name} {abs(units):.0f}u @~{entry:.5f} stop {stop:.5f}"
        if not self.dry_run:
            self.broker.market_order(pair, units, stop)
        return Action("OPEN", pair, ("[DRY] " if self.dry_run else "") + detail)

    def _manage_open(self, pair, row, bpos) -> Action:
        """Recompute the stop for an open position (regime/trailing handled by
        the resting stop + this periodic recompute). Full trailing requires the
        broker OMS to expose the stop order id (see OandaBroker.modify_stop)."""
        # Regime-flip check: if the D1 filter has flipped against us, close.
        side_sign = 1 if bpos.units > 0 else -1
        against = (row["d1_close"] < row["d1_ema"]) if side_sign > 0 else (row["d1_close"] > row["d1_ema"])
        if against:
            detail = "D1 regime flipped against position — flatten"
            if not self.dry_run:
                self.broker.close_position(pair, 1.0)
            return Action("CLOSE", pair, ("[DRY] " if self.dry_run else "") + detail)
        return Action("SKIP", pair, "hold; resting stop manages exit")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="H4-Donchian-Trend live/dry-run runner")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--live", action="store_true", help="place real orders (needs broker creds)")
    ap.add_argument("--broker", choices=["paper", "oanda"], default="paper")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.broker == "oanda":
        from trader_bot.broker import OandaBroker
        broker: Broker = OandaBroker(practice=True)
    else:
        # Paper broker needs a price feed; in a real deployment you'd wire a
        # cached candle store. Here we feed synthetic candles so the dry-run
        # exercises the full decision path with no network.
        from trader_bot.data import generate_synthetic
        feed = {p: generate_synthetic(p, years=4.0, seed=7) for p in cfg.pairs}
        broker = PaperBroker(cfg.starting_equity, price_feed=feed)
        log.warning("PaperBroker running on SYNTHETIC feed — wiring demo only.")

    engine = LiveEngine(broker, cfg, dry_run=not args.live)
    engine.run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
