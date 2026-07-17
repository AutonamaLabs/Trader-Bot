#!/usr/bin/env python3
"""Convert raw ejtraderLabs MT4-point H4 CSVs into the loader's format.

Source columns: Date, open, high, low, close, tick_volume
  - prices are stored as integer "points": divide by 1e5 (non-JPY) or 1e3 (JPY).
Output: data/<PAIR>_H4.csv with columns time,open,high,low,close (UTC).

Note: the source is MetaTrader broker-time data (~UTC+2/UTC+3), not strictly
UTC. We treat the timestamps as UTC; a couple-hour offset can nudge the
session/entry-window filters but does not change the strategy's substance. For a
production run, align the broker's timezone exactly (see STRATEGY.md §13).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PAIRS = {  # config name -> raw file stem
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "AUD_USD": "AUDUSD",
}


def divisor(pair: str) -> float:
    return 1e3 if pair.endswith("JPY") else 1e5


def main() -> int:
    for cfg_name, stem in PAIRS.items():
        raw_path = DATA / f"{stem}_raw.csv"
        if not raw_path.exists():
            print(f"!! missing {raw_path}; download it first")
            return 1
        df = pd.read_csv(raw_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df["time"] = pd.to_datetime(df["date"], utc=True)
        d = divisor(cfg_name)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float) / d
        out = df[["time", "open", "high", "low", "close"]].sort_values("time")
        out = out.drop_duplicates(subset="time")
        out_path = DATA / f"{cfg_name}_H4.csv"
        out.to_csv(out_path, index=False)
        print(f"  {cfg_name}: {len(out):,} bars  "
              f"{out['time'].iloc[0].date()} -> {out['time'].iloc[-1].date()}  "
              f"(px {out['close'].iloc[-1]:.5f})  -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
