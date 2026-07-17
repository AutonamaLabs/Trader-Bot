import pandas as pd
import pytest

from trader_bot.config import load_config
from trader_bot.models import PosState, Position, Side
from trader_bot.strategy import (
    entry_signal,
    initial_stop,
    process_bar_exits,
)


@pytest.fixture
def cfg():
    return load_config()


def _row(**kw):
    base = dict(
        close=1.1050, high=1.1060, low=1.1040, open=1.1045,
        donchian_up=1.1040, donchian_dn=1.0950,
        atr=0.0025, adx=30.0, atr_median=0.0025,
        d1_close=1.1000, d1_ema=1.0950, d1_ema_prev=1.0900,
        d1_available_at=pd.Timestamp("2020-01-02", tz="UTC"),
    )
    base.update(kw)
    s = pd.Series(base)
    s.name = pd.Timestamp("2020-01-02 08:00", tz="UTC")
    return s


def test_long_signal_fires_on_breakout_with_uptrend(cfg):
    sig = entry_signal("EUR_USD", _row(), cfg)
    assert sig is not None and sig.side == Side.LONG


def test_no_signal_when_regime_down(cfg):
    # Breakout up but daily regime is down -> no long.
    row = _row(d1_close=1.0900, d1_ema=1.0950, d1_ema_prev=1.0960)
    assert entry_signal("EUR_USD", row, cfg) is None


def test_no_signal_when_adx_too_low(cfg):
    row = _row(adx=10.0)  # below adx_min
    assert entry_signal("EUR_USD", row, cfg) is None


def test_no_signal_when_volatility_out_of_band(cfg):
    row = _row(atr=0.0300, atr_median=0.0025)  # ratio 12 >> band high
    assert entry_signal("EUR_USD", row, cfg) is None


def test_short_signal(cfg):
    row = _row(
        close=1.0940, donchian_dn=1.0950, donchian_up=1.1100,
        d1_close=1.0800, d1_ema=1.0850, d1_ema_prev=1.0900,
    )
    sig = entry_signal("EUR_USD", row, cfg)
    assert sig is not None and sig.side == Side.SHORT


def _long_pos(cfg, entry=1.1000, atr=0.0025):
    r = cfg.exits.sl_atr_mult * atr
    return Position(
        pair="EUR_USD", side=Side.LONG, entry_time=pd.Timestamp("2020-01-02", tz="UTC"),
        entry_price=entry, units=10000, initial_units=10000, atr_at_entry=atr,
        r_price=r, stop=entry - r, state=PosState.OPEN, extreme_price=entry,
    )


def test_stop_loss_triggers(cfg):
    pos = _long_pos(cfg)
    # Bar trades down through the stop.
    row = _row(high=1.1005, low=pos.stop - 0.0005, close=pos.stop - 0.0003)
    events = process_bar_exits(pos, row, cfg)
    assert any(e.reason == "stop" for e in events)


def test_partial_tp_then_breakeven(cfg):
    pos = _long_pos(cfg)
    tp = pos.entry_price + cfg.exits.partial_tp_r * pos.r_price
    row = _row(high=tp + 0.0002, low=1.0998, close=tp, atr=0.0025,
               d1_close=1.10, d1_ema=1.05)
    events = process_bar_exits(pos, row, cfg)
    assert any(e.reason == "partial_tp" for e in events)
    assert pos.state == PosState.PARTIAL_TAKEN
    # Stop moved to (at least) breakeven region, above entry.
    assert pos.stop >= pos.entry_price


def test_stop_fills_before_target_on_ambiguous_bar(cfg):
    pos = _long_pos(cfg)
    tp = pos.entry_price + cfg.exits.partial_tp_r * pos.r_price
    # Bar spans BOTH the stop and the take-profit.
    row = _row(high=tp + 0.001, low=pos.stop - 0.001, close=pos.entry_price)
    events = process_bar_exits(pos, row, cfg)
    assert events and events[0].reason == "stop"
    assert not any(e.reason == "partial_tp" for e in events)


def test_time_stop_when_no_progress(cfg):
    pos = _long_pos(cfg)
    pos.bars_held = cfg.exits.time_stop_bars
    pos.reached_1r = False
    # Quiet bar, no stop/tp hit, price near entry.
    row = _row(high=1.1005, low=1.0996, close=1.1000)
    events = process_bar_exits(pos, row, cfg)
    assert any(e.reason == "time_stop" for e in events)
