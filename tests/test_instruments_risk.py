import pytest

from trader_bot import instruments as instr
from trader_bot.config import load_config
from trader_bot.models import Position, Side
from trader_bot.risk import RiskManager


def test_pip_size():
    assert instr.pip_size("EUR_USD") == 0.0001
    assert instr.pip_size("USD_JPY") == 0.01


def test_pip_value_xxx_usd():
    # For EUR/USD with a USD account, 1 pip on 1 unit = 0.0001 USD.
    assert instr.pip_value_per_unit("EUR_USD", 1.10) == pytest.approx(0.0001)


def test_pip_value_usd_jpy():
    # USD/JPY: pip value per unit = 0.01 / price.
    assert instr.pip_value_per_unit("USD_JPY", 145.0) == pytest.approx(0.01 / 145.0)


def test_position_size_risks_target_fraction():
    cfg = load_config()
    rm = RiskManager(cfg)
    entry = 1.10
    atr = 0.0025                        # 25 pips
    stop_dist = cfg.exits.sl_atr_mult * atr   # 50 pips
    units = rm.position_size("EUR_USD", entry, stop_dist)
    # Loss if stopped = stop_dist * units (in USD for XXX/USD).
    loss_at_stop = stop_dist * units
    target = cfg.starting_equity * cfg.risk.risk_per_trade
    # Should risk approximately the target (rounding down units only).
    assert loss_at_stop <= target
    assert loss_at_stop > target * 0.95


def _pos(pair, side):
    return Position(pair=pair, side=side, entry_time=None, entry_price=1.0,
                    units=1000, initial_units=1000, atr_at_entry=0.001,
                    r_price=0.002, stop=0.998)


def test_max_concurrent_enforced():
    cfg = load_config()
    rm = RiskManager(cfg)
    open_pos = {
        "USD_JPY": _pos("USD_JPY", Side.LONG),
        "AUD_USD": _pos("AUD_USD", Side.LONG),
        "NZD_USD": _pos("NZD_USD", Side.LONG),
    }
    assert not rm.can_open("EUR_USD", Side.LONG, open_pos)  # already at max 3


def test_correlation_bucket_blocks_same_direction():
    cfg = load_config()
    rm = RiskManager(cfg)
    open_pos = {"EUR_USD": _pos("EUR_USD", Side.LONG)}
    # GBP_USD long is blocked (same correlation bucket, same direction)...
    assert not rm.can_open("GBP_USD", Side.LONG, open_pos)
    # ...but a short would be allowed by the correlation rule.
    # (Currency-exposure rule may still allow one USD-sharing position.)
    assert rm.can_open("GBP_USD", Side.SHORT, open_pos)


def test_currency_exposure_limit():
    cfg = load_config()
    rm = RiskManager(cfg)
    open_pos = {
        "EUR_USD": _pos("EUR_USD", Side.LONG),
        "USD_JPY": _pos("USD_JPY", Side.LONG),
    }
    # Both positions already involve USD -> a third USD pair hits max_per_currency=2.
    assert not rm.can_open("AUD_USD", Side.SHORT, open_pos)


def test_drawdown_halt():
    cfg = load_config()
    rm = RiskManager(cfg)
    rm.high_water_mark = 5000
    rm.mark(-800)  # equity 4200 -> 16% DD > 15% halt
    assert rm.halted
    assert not rm.entries_allowed()
