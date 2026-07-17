"""Performance metrics and reporting for a BacktestResult."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd

from .backtest import BacktestResult


@dataclass
class Metrics:
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    final_equity: float
    avg_bars_held: float

    def as_dict(self) -> Dict:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = 1.0 - equity / running_max
    return float(dd.max()) if len(dd) else 0.0


def compute_metrics(result: BacktestResult) -> Metrics:
    trades = result.trades
    eq = result.equity_curve
    n = len(trades)

    if n == 0:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, _max_drawdown(eq) * 100,
                       0, result.final_equity, 0)

    pnls = np.array([t.pnl for t in trades])
    rs = np.array([t.r_multiple for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / n
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    expectancy_r = float(rs.mean())
    avg_win_r = float(rs[rs > 0].mean()) if (rs > 0).any() else 0.0
    avg_loss_r = float(rs[rs < 0].mean()) if (rs < 0).any() else 0.0

    total_return = result.final_equity / result.starting_equity - 1.0

    # CAGR from the equity curve time span.
    if len(eq) > 1:
        span_days = (eq.index[-1] - eq.index[0]).days or 1
        years = span_days / 365.25
        cagr = (result.final_equity / result.starting_equity) ** (1 / years) - 1 if years > 0 else 0.0
    else:
        cagr = 0.0

    # Sharpe from daily equity returns (annualised, 252d).
    daily = eq.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0.0

    avg_bars = float(np.mean([t.bars_held for t in trades]))

    return Metrics(
        n_trades=n,
        win_rate=win_rate,
        profit_factor=float(profit_factor),
        expectancy_r=expectancy_r,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        max_drawdown_pct=_max_drawdown(eq) * 100,
        sharpe=sharpe,
        final_equity=result.final_equity,
        avg_bars_held=avg_bars,
    )


def format_report(result: BacktestResult, m: Metrics) -> str:
    lines = [
        "=" * 60,
        "  H4-Donchian-Trend — Backtest Report",
        "=" * 60,
        f"  Period            : {result.equity_curve.index[0].date()} "
        f"-> {result.equity_curve.index[-1].date()}",
        f"  Starting equity   : ${result.starting_equity:,.2f}",
        f"  Final equity      : ${m.final_equity:,.2f}",
        f"  Total return      : {m.total_return_pct:+.2f}%",
        f"  CAGR              : {m.cagr_pct:+.2f}%",
        f"  Max drawdown      : {m.max_drawdown_pct:.2f}%",
        f"  Sharpe (daily)    : {m.sharpe:.2f}",
        "-" * 60,
        f"  Trades            : {m.n_trades}",
        f"  Win rate          : {m.win_rate*100:.1f}%",
        f"  Profit factor     : {m.profit_factor:.2f}",
        f"  Expectancy        : {m.expectancy_r:+.3f} R / trade",
        f"  Avg win / loss    : {m.avg_win_r:+.2f}R / {m.avg_loss_r:+.2f}R",
        f"  Avg bars held     : {m.avg_bars_held:.1f} (H4 bars)",
        "=" * 60,
    ]
    return "\n".join(lines)


def per_pair_summary(result: BacktestResult) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "pair": t.pair, "pnl": t.pnl, "r": t.r_multiple, "win": t.pnl > 0,
    } for t in result.trades])
    g = df.groupby("pair").agg(
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        win_rate=("win", "mean"),
        expectancy_r=("r", "mean"),
    )
    g["win_rate"] = (g["win_rate"] * 100).round(1)
    g["net_pnl"] = g["net_pnl"].round(2)
    g["expectancy_r"] = g["expectancy_r"].round(3)
    return g
