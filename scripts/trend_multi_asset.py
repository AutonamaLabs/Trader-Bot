#!/usr/bin/env python3
"""Diversified multi-asset time-series momentum (CTA-style trend-following).

Implements the Hurst/Ooi/Pedersen (AQR, 2017) "A Century of Evidence on
Trend-Following Investing" methodology: for each market, go long if trailing
return is positive and short if negative, size every market to a common
ex-ante volatility target (so no single market dominates portfolio risk), and
combine equal-risk-weighted across markets AND asset classes. The academic
claim is that the edge comes from BREADTH across many weakly-correlated
trending markets, not from any one market trending reliably — this is the
real mechanism behind CTAs like AHL/Winton, and the reason the earlier
FX-only attempt in this repo (docs/RESEARCH.md) was too narrow to test it.

Signal (per market, using only information available as of the rebalance
date — no lookahead):
    signal = sign(trailing N-month total return)      [--lookback 12m]
    or      = average sign of 1M/3M/12M returns        [--lookback blend]

Sizing: inverse-volatility so each market is scaled to the SAME ex-ante
annualised vol (--inst-vol, default 10%). Because every market's own return
series is pre-scaled to equal vol, a plain average across markets in a class
gives each MARKET equal risk; averaging across class-level indices gives each
ASSET CLASS equal risk too (two-level equal-risk book).

Rebalance: monthly by default (--rebalance M), weekly available for a
robustness check (--rebalance W). Weights computed at the close of the
rebalance date are applied starting the NEXT trading day (shift(1)) — the
rebalance day's own return is never touched by its own just-computed signal.

Costs: charged in bps of notional on every rebalance-day weight CHANGE, per
asset class (see COST_BPS below) — FX/futures-like markets are cheap, ETF
proxies for equities/bonds/commodities carry a wider assumed spread.

    python scripts/trend_multi_asset.py
    python scripts/trend_multi_asset.py --lookback blend --rebalance W
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from swing_tps5 import load_daily

SWING_DIR = ROOT / "data" / "swing"
MULTI_DIR = ROOT / "data" / "multi_asset"
ECB_CSV = ROOT / "data" / "research" / "ecb_daily.csv"

# ---------------------------------------------------------------------------
# Universe. asset_class buckets are what get equal-risk-weighted at the top
# level; cost_bps is a ROUND-TRIP assumption (charged pro-rata on turnover).
# ---------------------------------------------------------------------------
# G10 currencies, EUR-denominated, from the ECB daily reference-rate mirror
# (1999-present) — much longer history than the 2012-2022 MT4 mirror, so this
# is what removes FX as the binding constraint on the common multi-asset-class
# test window and lets 2008/2020/2022 all be checked.
ECB_G10 = {"Australia": "AUD", "Canada": "CAD", "Japan": "JPY", "New Zealand": "NZD",
           "Norway": "NOK", "Sweden": "SEK", "Switzerland": "CHF", "United Kingdom": "GBP"}

COST_BPS = {
    "fx": 1.0,          # institutional/CTA-style FX forward, half-spread~0.5bp/side
    "equity": 2.0,       # liquid index ETF, ~1bp/side
    "bond": 2.0,          # liquid Treasury ETF, ~1bp/side
    "commodity": 6.0,     # commodity ETP, wider spread, ~3bp/side
}


def _load_ecb_fx() -> dict:
    """Load G10 currency series (EUR-denominated value) from the ECB mirror."""
    if not ECB_CSV.exists():
        return {}
    df = pd.read_csv(ECB_CSV)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"])
    df["rate"] = pd.to_numeric(df["Exchange rate"], errors="coerce")
    df = df[df["Country"].isin(ECB_G10)].copy()
    df["ccy"] = df["Country"].map(ECB_G10)
    wide = df.pivot_table(index="Date", columns="ccy", values="rate").sort_index()
    value = (1.0 / wide).resample("1D").last().ffill()
    out = {}
    for ccy in value.columns:
        s = value[ccy].dropna()
        if len(s) > 320:
            out[f"{ccy}EUR"] = s
    return out


def _load_swing_csv(name: str, path: Path) -> pd.Series:
    df = load_daily(path)
    return df["close"]


def build_universe(extra_dir: Path = MULTI_DIR) -> dict:
    """Returns {symbol: {"close": pd.Series, "class": str, "cost_bps": float}}."""
    universe = {}
    for sym, s in _load_ecb_fx().items():
        universe[sym] = dict(close=s, asset_class="fx", cost_bps=COST_BPS["fx"])

    swing_map = {
        "XAUUSD": "commodity", "SPY": "equity", "QQQ": "equity",
    }
    for name, cls in swing_map.items():
        p = SWING_DIR / f"{name}.csv"
        if p.exists():
            universe[name] = dict(close=_load_swing_csv(name, p), asset_class=cls,
                                   cost_bps=COST_BPS[cls])

    # Extra multi-asset-class markets (bonds, commodities, global indices) —
    # fetched by scripts/fetch_multi_asset.sh into data/multi_asset/<SYM>.csv
    # with a manifest mapping symbol -> asset class.
    manifest_path = extra_dir / "manifest.csv"
    if manifest_path.exists():
        man = pd.read_csv(manifest_path)
        for _, row in man.iterrows():
            sym, cls = row["symbol"], row["asset_class"]
            p = extra_dir / f"{sym}.csv"
            if not p.exists():
                continue
            try:
                s = _load_swing_csv(sym, p)
            except Exception as e:
                print(f"  skip {sym}: {e}")
                continue
            if len(s) < 300:
                continue
            universe[sym] = dict(close=s, asset_class=cls, cost_bps=COST_BPS.get(cls, 4.0))
    return universe


# ---------------------------------------------------------------------------
# Signal + sizing
# ---------------------------------------------------------------------------
def trailing_signal(close: pd.Series, lookback: str) -> pd.Series:
    """Sign of trailing return, using only data up to and including `t`."""
    if lookback == "12m":
        ret = close / close.shift(252) - 1
        return np.sign(ret)
    if lookback == "blend":
        sigs = []
        for days in (21, 63, 252):
            ret = close / close.shift(days) - 1
            sigs.append(np.sign(ret))
        return sum(sigs) / len(sigs)
    raise ValueError(lookback)


def ex_ante_vol(close: pd.Series, halflife: int = 60) -> pd.Series:
    """Annualised EWM vol of daily log returns (AQR-style ex-ante vol)."""
    logret = np.log(close / close.shift(1))
    return logret.ewm(halflife=halflife, min_periods=40).std() * np.sqrt(252)


def market_book(close: pd.Series, lookback: str, inst_vol: float,
                 rebalance: str) -> tuple[pd.Series, pd.Series]:
    """Returns (net_daily_return, gross_weight) for ONE market, vol-targeted to
    `inst_vol`, rebalanced on `rebalance` frequency, no lookahead: the weight
    decided using data through rebalance date t is applied starting t+1."""
    logret = np.log(close / close.shift(1))
    simple_ret = close.pct_change()
    sig = trailing_signal(close, lookback)
    vol = ex_ante_vol(close)
    raw_w = (sig * inst_vol / vol).clip(-3, 3)  # cap gross leverage per market

    # Sample the raw weight only on rebalance dates, forward-fill between them.
    if rebalance == "M":
        rebal_dates = close.resample("ME").last().index
    elif rebalance == "W":
        rebal_dates = close.resample("W-FRI").last().index
    else:
        raise ValueError(rebalance)
    rebal_dates = rebal_dates.intersection(raw_w.index.floor("D").unique())
    # map each rebalance calendar date to the actual last available trading date <= it
    avail = raw_w.dropna().index
    if len(avail) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty
    step = pd.Series(np.nan, index=close.index)
    reb_set = set()
    for d in rebal_dates:
        idx_pos = avail.searchsorted(d, side="right") - 1
        if idx_pos >= 0:
            reb_set.add(avail[idx_pos])
    for d in sorted(reb_set):
        step.loc[d] = raw_w.loc[d]
    weight = step.ffill().shift(1)  # apply from the NEXT trading day only
    weight = weight.fillna(0.0)

    return simple_ret, weight


def apply_costs(simple_ret: pd.Series, weight: pd.Series, cost_bps: float) -> pd.Series:
    turnover = weight.diff().abs().fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    net = weight * simple_ret - cost
    # Only defined once the market has actually started trading (post warm-up
    # this is exactly 0 on flat days, which is real — but before the series
    # starts / after it ends there must be NaN, not 0, so it doesn't dilute
    # the cross-market average during eras this market wasn't in the universe.
    net = net.where(simple_ret.notna() | weight.ne(0))
    return net


# ---------------------------------------------------------------------------
# Portfolio construction: equal-risk within class, equal-risk across classes
# ---------------------------------------------------------------------------
def build_portfolio(universe: dict, lookback: str, rebalance: str,
                     inst_vol: float = 0.10) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict, tuple]:
    per_market = {}
    classes = {}
    for sym, info in universe.items():
        close = info["close"].dropna()
        close = close[~close.index.duplicated()]
        # Guard against one-off data anomalies (e.g. WTI front-month futures
        # traded at -$36.98 on 2020-04-20, a real but mechanically-driven
        # expiry/settlement quirk unique to that contract, not a market move
        # a trend-following system is meant to capture). Clipping to a small
        # positive floor would create a fake near-infinite return the next
        # time price recovers, so instead treat the single bad print as
        # missing data and carry the last good price forward one day.
        close = close.mask(close <= 0).ffill()
        if len(close) < 320:
            continue
        simple_ret, weight = market_book(close, lookback, inst_vol, rebalance)
        if weight.abs().sum() == 0:
            continue
        net = apply_costs(simple_ret, weight, info["cost_bps"])
        per_market[sym] = net
        classes.setdefault(info["asset_class"], []).append(sym)

    if not per_market:
        raise RuntimeError("no markets produced a usable return series")

    all_idx = sorted(set().union(*[s.index for s in per_market.values()]))
    all_idx = pd.DatetimeIndex(all_idx)
    # NaN (not 0) outside each market's own trading history — a plain mean()
    # with skipna then correctly excludes markets/classes that simply weren't
    # trading yet, instead of silently diluting the book with fake zeros.
    R_df = pd.DataFrame({s: per_market[s].reindex(all_idx) for s in per_market})

    class_series = {}
    class_span = {}
    for cls, syms in classes.items():
        class_series[cls] = R_df[syms].mean(axis=1, skipna=True)
        active = R_df[syms].notna().any(axis=1)
        active_idx = active[active].index
        class_span[cls] = (active_idx.min(), active_idx.max())
    class_df = pd.DataFrame(class_series)
    portfolio = class_df.mean(axis=1, skipna=True).dropna()

    common_start = max(s for s, _ in class_span.values())
    common_end = min(e for _, e in class_span.values())
    return portfolio, class_df, R_df, classes, (common_start, common_end)


def stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    dd = (1 - eq / eq.cummax()).max()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0.0
    sh = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else 0.0
    return dict(cagr=cagr, sharpe=sh, maxdd=dd, yrs=yrs, n=len(ret))


def vol_target_scale(ret: pd.Series, target: float = 0.12, lev_cap: float = 2.0) -> pd.Series:
    real_vol = ret.std() * np.sqrt(252)
    if real_vol <= 0:
        return ret
    lev = min(target / real_vol, lev_cap)
    return ret * lev


def build_60_40(universe: dict) -> pd.Series | None:
    if "SPY" not in universe:
        return None
    spy = universe["SPY"]["close"].pct_change()
    bond_sym = None
    for cand in ("TLT", "IEF"):
        if cand in universe:
            bond_sym = cand
            break
    if bond_sym is None:
        return None
    bond = universe[bond_sym]["close"].pct_change()
    idx = spy.index.intersection(bond.index)
    return 0.6 * spy.reindex(idx) + 0.4 * bond.reindex(idx)


def sub_period(ret: pd.Series, start, end, label):
    s = ret.loc[start:end]
    if len(s) < 10:
        print(f"  {label:<28}  NO DATA in this window (universe not live then)")
        return None
    st = stats(s)
    print(f"  {label:<28}{st['cagr']*100:>+8.1f}%  Sharpe {st['sharpe']:>6.2f}  "
          f"maxDD {st['maxdd']*100:>5.1f}%  ({s.index[0].date()}..{s.index[-1].date()}, n={st['n']})")
    return st


def report_book(universe: dict, label: str, lookback: str, rebalance: str, inst_vol: float,
                 crisis_windows: list, show_class_breakdown: bool = True):
    portfolio_full, class_df, _R_df, classes, (common_start, common_end) = build_portfolio(
        universe, lookback, rebalance, inst_vol)
    print(f"\n{'='*74}\n  {label}\n{'='*74}")
    print(f"  Common overlap window (every included asset class simultaneously live): "
          f"{common_start.date()} .. {common_end.date()}")
    portfolio = portfolio_full.loc[common_start:common_end]

    st = stats(portfolio)
    print(f"\n  Raw (unscaled) book:        CAGR {st['cagr']*100:+6.2f}%  Sharpe {st['sharpe']:.2f}  "
          f"maxDD {st['maxdd']*100:5.1f}%  span {st['yrs']:.1f}y  n={st['n']}")
    scaled = vol_target_scale(portfolio, target=0.12, lev_cap=2.0)
    sst = stats(scaled)
    print(f"  Vol-targeted 12% (2x cap):  CAGR {sst['cagr']*100:+6.2f}%  Sharpe {sst['sharpe']:.2f}  "
          f"maxDD {sst['maxdd']*100:5.1f}%")

    if show_class_breakdown:
        print("\n  Per-asset-class sub-index (equal-risk within class):")
        for cls in sorted(class_df.columns):
            cst = stats(class_df[cls].loc[common_start:common_end])
            print(f"    {cls:<10} CAGR {cst['cagr']*100:+6.2f}%  Sharpe {cst['sharpe']:.2f}  "
                  f"maxDD {cst['maxdd']*100:5.1f}%  markets={len(classes[cls])}")
        corr = class_df.loc[common_start:common_end].corr()
        print("\n  Asset-class correlation matrix:")
        print("  " + corr.round(2).to_string().replace("\n", "\n  "))

    print("\n  Sub-period / crisis behaviour:")
    for start, end, wlabel in crisis_windows:
        sub_period(portfolio_full, start, end, wlabel)

    return portfolio, sst, common_start, common_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", choices=["12m", "blend"], default="12m")
    ap.add_argument("--rebalance", choices=["M", "W"], default="M")
    ap.add_argument("--inst-vol", type=float, default=0.10)
    args = ap.parse_args()

    universe = build_universe()
    n_classes = len(set(v["asset_class"] for v in universe.values()))
    print("=" * 74)
    print(f"  Diversified multi-asset trend-following (CTA-style)")
    print(f"  Full universe: {len(universe)} markets across {n_classes} asset classes")
    by_class = {}
    for sym, info in universe.items():
        by_class.setdefault(info["asset_class"], []).append(sym)
    for cls, syms in sorted(by_class.items()):
        print(f"    {cls:<10} ({len(syms)}): {', '.join(sorted(syms))}")

    # CORE book: all 4 asset classes including bonds. Bonds only have reachable
    # data through 2017-11-10 (see fetch_multi_asset.sh), so this book's common
    # window is bond-limited — but it DOES cover the 2008 GFC. (2020/2022 are
    # NOT shown here: bonds aren't live then, so those numbers would just be
    # a relabeled copy of the EXTENDED book's — shown once, below, instead.)
    core_port, core_sst, cs, ce = report_book(
        universe, "CORE BOOK (FX + equity + commodity + bond)",
        args.lookback, args.rebalance, args.inst_vol,
        [("2007-06-01", "2009-06-30", "2008 GFC crash (bonds live)")])

    bench = build_60_40(universe)
    if bench is not None:
        common = core_port.index.intersection(bench.index)
        if len(common) > 50:
            bst = stats(bench.reindex(common))
            pc = core_port.reindex(common).corr(bench.reindex(common))
            print(f"\n  vs 60/40 SPY/TLT benchmark ({common[0].date()}..{common[-1].date()}):")
            print(f"    60/40 CAGR {bst['cagr']*100:+6.2f}%  Sharpe {bst['sharpe']:.2f}  maxDD {bst['maxdd']*100:5.1f}%")
            print(f"    Correlation (trend vs 60/40): {pc:+.2f}")

    # EXTENDED book: drop bonds (no source past 2017 was reachable from this
    # sandbox — see fetch_multi_asset.sh) to get a MUCH longer common window
    # via ECB FX (1999-2026) + SPY (2000-2025) + WTI/Brent/NatGas (1986/97-2026)
    # + XAUUSD (2012-2022), so 2020 and 2022 can actually be tested.
    ex_bonds = {k: v for k, v in universe.items() if v["asset_class"] != "bond"}
    ext_port, ext_sst, ecs, ece = report_book(
        ex_bonds, "EXTENDED BOOK (FX + equity + commodity, NO bonds — longer span)",
        args.lookback, args.rebalance, args.inst_vol,
        [("2007-06-01", "2009-06-30", "2008 GFC crash"),
         ("2020-01-01", "2020-06-30", "2020 COVID crash"),
         ("2022-01-01", "2022-12-31", "2022 stock+bond selloff")])

    print(f"\n{'='*74}\n  Robustness: lookback/rebalance variants on the EXTENDED book\n"
          f"  (no cherry-picking — every combination shown)\n{'='*74}")
    for lb in ("12m", "blend"):
        for rb in ("M", "W"):
            try:
                p2, _, _, _, (s2, e2) = build_portfolio(ex_bonds, lb, rb, args.inst_vol)
                p2 = p2.loc[s2:e2]
                st2 = stats(p2)
                tag = "  <- selected headline" if (lb == args.lookback and rb == args.rebalance) else ""
                print(f"  lookback={lb:<6} rebalance={rb}  CAGR {st2['cagr']*100:+6.2f}%  "
                      f"Sharpe {st2['sharpe']:.2f}  maxDD {st2['maxdd']*100:5.1f}%{tag}")
            except Exception as e:
                print(f"  lookback={lb} rebalance={rb}: FAILED ({e})")

    core_scaled_stats = core_sst
    ext_scaled_stats = ext_sst
    print(f"\n{'='*74}\n  Bank-rate comparison\n{'='*74}")
    print(f"  ~4-5% risk-free bank savings rate")
    print(f"  CORE (4-class, vol-targeted 12%):     CAGR {core_scaled_stats['cagr']*100:+.2f}%  "
          f"Sharpe {core_scaled_stats['sharpe']:.2f}  maxDD {core_scaled_stats['maxdd']*100:.1f}%")
    print(f"  EXTENDED (3-class, vol-targeted 12%): CAGR {ext_scaled_stats['cagr']*100:+.2f}%  "
          f"Sharpe {ext_scaled_stats['sharpe']:.2f}  maxDD {ext_scaled_stats['maxdd']*100:.1f}%")


if __name__ == "__main__":
    main()
