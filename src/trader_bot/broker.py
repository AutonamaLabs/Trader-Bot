"""Broker abstraction + a paper broker and an OANDA v20 REST adapter.

The live runner talks only to the ``Broker`` interface, so the same strategy
code drives a paper account, a practice account, or a live account by swapping
the implementation. No credentials are required for paper trading or backtests.

SECURITY: never hard-code API tokens. The OANDA adapter reads them from the
environment (OANDA_TOKEN / OANDA_ACCOUNT_ID). See .env.example.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class BrokerPosition:
    pair: str
    units: float          # signed: + long, - short
    avg_price: float


@dataclass
class Candle:
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    complete: bool = True


class Broker(ABC):
    @abstractmethod
    def get_candles(self, pair: str, granularity: str, count: int) -> pd.DataFrame:
        """Return the last ``count`` COMPLETE candles as an OHLC DataFrame."""

    @abstractmethod
    def get_price(self, pair: str) -> tuple[float, float]:
        """Return (bid, ask)."""

    @abstractmethod
    def account_equity(self) -> float:
        ...

    @abstractmethod
    def open_positions(self) -> Dict[str, BrokerPosition]:
        ...

    @abstractmethod
    def market_order(self, pair: str, units: float, stop: float,
                     take_profit: Optional[float] = None) -> str:
        """Submit a market order with an attached stop (units signed). Returns id."""

    @abstractmethod
    def close_position(self, pair: str, fraction: float = 1.0) -> None:
        ...

    @abstractmethod
    def modify_stop(self, pair: str, stop: float) -> None:
        ...


# ---------------------------------------------------------------------------
# Paper broker — fills at the provided price, tracks positions & equity locally.
# ---------------------------------------------------------------------------
class PaperBroker(Broker):
    def __init__(self, starting_equity: float, price_feed: Optional[Dict[str, pd.DataFrame]] = None):
        self._equity = starting_equity
        self._positions: Dict[str, BrokerPosition] = {}
        self._stops: Dict[str, float] = {}
        self._feed = price_feed or {}
        self._log: List[str] = []

    def get_candles(self, pair: str, granularity: str, count: int) -> pd.DataFrame:
        df = self._feed.get(pair)
        if df is None:
            raise RuntimeError(f"PaperBroker has no feed for {pair}")
        return df.tail(count)

    def get_price(self, pair: str) -> tuple[float, float]:
        df = self._feed.get(pair)
        last = float(df["close"].iloc[-1]) if df is not None else 1.0
        return last, last

    def account_equity(self) -> float:
        return self._equity

    def open_positions(self) -> Dict[str, BrokerPosition]:
        return dict(self._positions)

    def market_order(self, pair: str, units: float, stop: float,
                     take_profit: Optional[float] = None) -> str:
        price = self.get_price(pair)[1 if units > 0 else 0]
        self._positions[pair] = BrokerPosition(pair, units, price)
        self._stops[pair] = stop
        self._log.append(f"OPEN {pair} units={units:.0f} @ {price:.5f} stop={stop:.5f}")
        return f"paper-{pair}-{len(self._log)}"

    def close_position(self, pair: str, fraction: float = 1.0) -> None:
        pos = self._positions.get(pair)
        if not pos:
            return
        if fraction >= 1.0:
            del self._positions[pair]
            self._stops.pop(pair, None)
            self._log.append(f"CLOSE {pair} (full)")
        else:
            pos.units *= (1 - fraction)
            self._log.append(f"CLOSE {pair} ({fraction:.0%})")

    def modify_stop(self, pair: str, stop: float) -> None:
        if pair in self._positions:
            self._stops[pair] = stop

    @property
    def log(self) -> List[str]:
        return self._log


# ---------------------------------------------------------------------------
# OANDA v20 REST adapter (used only when credentials are present).
# ---------------------------------------------------------------------------
class OandaBroker(Broker):
    PRACTICE = "https://api-fxpractice.oanda.com"
    LIVE = "https://api-fxtrade.oanda.com"

    def __init__(self, practice: bool = True):
        self.token = os.environ.get("OANDA_TOKEN")
        self.account_id = os.environ.get("OANDA_ACCOUNT_ID")
        if not self.token or not self.account_id:
            raise RuntimeError(
                "OANDA_TOKEN and OANDA_ACCOUNT_ID must be set in the environment."
            )
        self.base = self.PRACTICE if practice else self.LIVE
        import requests  # local import so backtests don't need the dependency
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })

    def _get(self, path: str, **params):
        r = self._session.get(f"{self.base}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict):
        r = self._session.post(f"{self.base}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def _put(self, path: str, payload: dict):
        r = self._session.put(f"{self.base}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_candles(self, pair: str, granularity: str, count: int) -> pd.DataFrame:
        data = self._get(
            f"/v3/instruments/{pair}/candles",
            granularity=granularity, count=count, price="M",
        )
        rows = []
        for c in data["candles"]:
            if not c["complete"]:
                continue
            m = c["mid"]
            rows.append({
                "time": pd.to_datetime(c["time"], utc=True),
                "open": float(m["o"]), "high": float(m["h"]),
                "low": float(m["l"]), "close": float(m["c"]),
            })
        return pd.DataFrame(rows).set_index("time")

    def get_price(self, pair: str) -> tuple[float, float]:
        data = self._get(f"/v3/accounts/{self.account_id}/pricing", instruments=pair)
        p = data["prices"][0]
        return float(p["bids"][0]["price"]), float(p["asks"][0]["price"])

    def account_equity(self) -> float:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        return float(data["account"]["NAV"])

    def open_positions(self) -> Dict[str, BrokerPosition]:
        data = self._get(f"/v3/accounts/{self.account_id}/openPositions")
        out: Dict[str, BrokerPosition] = {}
        for p in data.get("positions", []):
            long_u = float(p["long"]["units"])
            short_u = float(p["short"]["units"])
            units = long_u + short_u
            if units == 0:
                continue
            avg = float(p["long"]["averagePrice"]) if long_u else float(p["short"]["averagePrice"])
            out[p["instrument"]] = BrokerPosition(p["instrument"], units, avg)
        return out

    def market_order(self, pair: str, units: float, stop: float,
                     take_profit: Optional[float] = None) -> str:
        order = {
            "order": {
                "type": "MARKET",
                "instrument": pair,
                "units": str(int(units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{stop:.5f}", "timeInForce": "GTC"},
            }
        }
        if take_profit is not None:
            order["order"]["takeProfitOnFill"] = {"price": f"{take_profit:.5f}"}
        resp = self._post(f"/v3/accounts/{self.account_id}/orders", order)
        return resp.get("orderFillTransaction", {}).get("id", "unknown")

    def close_position(self, pair: str, fraction: float = 1.0) -> None:
        pos = self.open_positions().get(pair)
        if not pos:
            return
        side = "longUnits" if pos.units > 0 else "shortUnits"
        value = "ALL" if fraction >= 1.0 else str(int(abs(pos.units) * fraction))
        self._put(f"/v3/accounts/{self.account_id}/positions/{pair}/close", {side: value})

    def modify_stop(self, pair: str, stop: float) -> None:
        # A production implementation tracks the SL order id and PUTs a
        # replacement; left as a focused extension point.
        raise NotImplementedError(
            "modify_stop requires tracking the stop-loss order id; wire this to "
            "your order-management layer before live trailing."
        )
