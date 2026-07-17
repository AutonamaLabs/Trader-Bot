"""Typed configuration loader for Trader-Bot.

Reads ``config.yaml`` into nested dataclasses so the rest of the code gets
attribute access and IDE completion instead of raw dict indexing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class Indicators:
    donchian_period: int
    ema_filter_period: int
    ema_slope_lookback_days: int
    atr_period: int
    adx_period: int
    adx_min: float
    atr_median_period: int
    atr_band_low: float
    atr_band_high: float


@dataclass
class Exits:
    sl_atr_mult: float
    partial_tp_r: float
    partial_tp_fraction: float
    breakeven_offset_r: float
    trail_atr_mult: float
    time_stop_bars: int
    regime_flip_days: int


@dataclass
class Risk:
    risk_per_trade: float
    max_concurrent: int
    max_per_pair: int
    max_per_currency: int
    notional_cap_mult: float
    daily_loss_halt: float
    weekly_loss_halt: float
    max_drawdown_halt: float
    sizing_floor_dd: float
    correlation_buckets: List[List[str]]


@dataclass
class Sessions:
    entry_window_start: int
    entry_window_end: int
    friday_cutoff_hour: int
    rollover_block_start: int
    rollover_block_end: int


@dataclass
class News:
    block_before_min: int
    block_after_min: int
    always_block_events: List[str]


@dataclass
class Costs:
    slippage_pips: float
    max_spread_pips: Dict[str, float]
    typical_spread_pips: Dict[str, float]
    spread_stop_ratio_max: float


@dataclass
class Config:
    starting_equity: float
    account_ccy: str
    pairs: List[str]
    signal_tf: str
    filter_tf: str
    indicators: Indicators
    exits: Exits
    risk: Risk
    sessions: Sessions
    news: News
    costs: Costs
    raw: dict = field(default_factory=dict, repr=False)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    return Config(
        starting_equity=raw["account"]["starting_equity"],
        account_ccy=raw["account"]["account_ccy"],
        pairs=raw["pairs"],
        signal_tf=raw["signal_tf"],
        filter_tf=raw["filter_tf"],
        indicators=Indicators(**raw["indicators"]),
        exits=Exits(**raw["exits"]),
        risk=Risk(**raw["risk"]),
        sessions=Sessions(**raw["sessions"]),
        news=News(**raw["news"]),
        costs=Costs(**raw["costs"]),
        raw=raw,
    )
