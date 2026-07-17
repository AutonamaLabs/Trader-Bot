import pandas as pd

from trader_bot.backtest import Backtester
from trader_bot.config import load_config
from trader_bot.data import generate_synthetic, resample_d1
from trader_bot.features import build_features
from trader_bot.metrics import compute_metrics


def test_no_lookahead_in_daily_mapping():
    """Each H4 bar must only ever see a daily bar that closed strictly before it."""
    cfg = load_config()
    raw = generate_synthetic("EUR_USD", years=1.0, seed=3)
    feats = build_features(raw, cfg)
    daily = resample_d1(raw)
    sample = feats.dropna(subset=["d1_available_at"]).iloc[-50:]
    for ts, row in sample.iterrows():
        # The mapped daily bar's availability timestamp must be <= H4 bar time.
        assert row["d1_available_at"] <= ts


def test_backtest_runs_and_is_self_consistent():
    cfg = load_config()
    data = {p: build_features(generate_synthetic(p, years=3.0, seed=5), cfg)
            for p in cfg.pairs}
    result = Backtester(cfg, data).run()
    m = compute_metrics(result)

    # Basic sanity: it produced trades and an equity curve.
    assert m.n_trades > 0
    assert len(result.equity_curve) > 0
    # Equity accounting is coherent: final equity equals starting + sum of pnl
    # (no open positions after final liquidation).
    total_pnl = sum(t.pnl for t in result.trades)
    assert abs(result.final_equity - (result.starting_equity + total_pnl)) < 1e-6


def test_risk_per_trade_bounded():
    """No single losing trade should lose materially more than the configured
    risk budget at entry (allowing a small slippage/gap margin)."""
    cfg = load_config()
    data = {p: build_features(generate_synthetic(p, years=4.0, seed=9), cfg)
            for p in cfg.pairs}
    result = Backtester(cfg, data).run()
    for t in result.trades:
        if t.pnl < 0:
            # Loss should not exceed ~1.5R (stop + slippage/gap tolerance).
            assert t.r_multiple > -1.6, f"{t.pair} lost {t.r_multiple:.2f}R"
