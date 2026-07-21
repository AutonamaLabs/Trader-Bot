#!/usr/bin/env python3
"""Blended-book leverage sweep: funding-arb (crypto) + equity swing (MR),
combined at PORTFOLIO level -- can leveraging the COMBINED book reach the
20-30%/yr the user asked about? See docs/RESEARCH.md, "Can leverage reach
20-30%/yr? (the honest ceiling)" for the full write-up.

This has NOT been tested before in this repo. Every prior leverage test
(funding_arb.py's leverage_survival_report / the 1.3x recommendation) applied
leverage to the funding-arb sleeve ALONE. This script (1) blends it with the
equity swing sleeve, (2) sweeps ADDITIONAL leverage on the whole blended book,
and (3) checks whether that additional leverage would have blown up the
funding-arb leg specifically -- the perp margin is the only leg in this
project with a real, quantified liquidation mechanic (the equity sleeve is a
long-only cash swing system with no margin/liquidation model at all).

Capital-accounting discipline (matches the correction already applied to the
funding_arb.py headline numbers, previously asserted in docs/RESEARCH.md
without a backing computation -- flagged as a real gap, closed here):
funding_arb.py's own functions (e.g. diversification_report) report the
50/50 BTC+ETH blend on a NOTIONAL basis. Converting to return-on-CAPITAL-
DEPLOYED (spot notional bought outright + perp margin, not notional alone)
means multiplying by L/(L+1) where L is leverage on the perp leg -- see
capital_deployed_multiplier() below. Every return series in this script is on
a capital-deployed basis, computed explicitly, not asserted.

Data overlap: funding-rate data covers 2020-01-01 -> 2023-12-31 only (see
funding_arb.py's docstring). The equity sleeve's honest data
(data/universe_pit, 2000-2024 point-in-time S&P 500 membership-gated MR
sleeve) is sliced to that SAME window for every correlation/blend/leverage
number below -- a real but short (4-year) overlap, NOT the sleeve's full
2000-2024 history. The full 2000-2023 history is still used to warm up
SMA200/precompute (avoiding the truncated-start edge artifact documented in
validate_pit_universe.py), but only 2020-2023 returns feed the stats.

Usage:
    python scripts/blended_leverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import funding_arb as FA          # noqa: E402
import validate_pit_universe as PIT  # noqa: E402
import swing_sweep as SW          # noqa: E402

OVERLAP_START = "2020-01-01"
OVERLAP_END = "2023-12-31"
L_EMBED_BASE = 1.3  # already-recommended embedded leverage on the funding-arb
                     # perp leg (docs/RESEARCH.md "Final recommendation")
LEV_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)   # ADDITIONAL portfolio-level leverage


def capital_deployed_multiplier(L: float) -> float:
    """Capital deployed for the funding-arb sleeve = spot notional (bought
    outright, no leverage on that leg -- only the perp leg is levered) + perp
    margin (notional / L). Return on capital = return_on_notional * notional
    / (notional * (1 + 1/L)) = return_on_notional * L / (L + 1). This is the
    SAME conversion that produces the capital-deployed numbers already quoted
    in docs/RESEARCH.md ("Final recommendation" table), now shown as code
    instead of asserted."""
    return L / (L + 1.0)


def funding_arb_notional_return(coins=("BTC", "ETH"), exchange="binance") -> pd.Series:
    """50/50 BTC+ETH equal-weighted daily return on a NOTIONAL basis --
    reuses funding_arb.py's own idea-3 diversification logic directly (same
    ROUND_TRIP_COST charged once per coin on entry), not re-derived."""
    legs = []
    for coin in coins:
        r = FA.load_funding(coin, exchange).resample("1D").sum().copy()
        r.iloc[0] -= FA.ROUND_TRIP_COST
        legs.append(r.rename(coin))
    df = pd.concat(legs, axis=1).fillna(0.0)
    return df.mean(axis=1)


def funding_arb_capital_return(L_embed: float = L_EMBED_BASE, coins=("BTC", "ETH"),
                                exchange="binance") -> pd.Series:
    """Notional return, converted to capital-deployed basis at embedded
    leverage L_embed via the L/(L+1) multiplier above."""
    notional = funding_arb_notional_return(coins, exchange)
    return notional * capital_deployed_multiplier(L_embed)


def equity_sleeve_return(overlap_start=OVERLAP_START, overlap_end=OVERLAP_END,
                          risk: float = 0.01) -> pd.Series:
    """Standard per-trade-risk (1%, the project default) MR sleeve, point-in-
    time S&P 500 membership gated. Precomputes/simulates the FULL 2000-2023
    history (avoiding the truncated-start SMA200 warmup artifact -- see
    validate_pit_universe.py), then slices the resulting daily returns down
    to the funding-data overlap window only."""
    print("  (precomputing equity sleeve indicators + simulating 2000-2023, once...)")
    A, idx = SW.precompute(str(PIT.PIT / "prices"), "2000-01-01", overlap_end)
    membership = PIT.load_membership(idx)
    eqs, trades, coverage_end_closes = PIT.simulate_pit(A, idx, membership, risk=risk)
    eqs = eqs[(eqs.index >= pd.Timestamp(overlap_start, tz="UTC")) &
              (eqs.index <= pd.Timestamp(overlap_end, tz="UTC"))]
    r = eqs.pct_change().dropna()
    r.index = r.index.tz_localize(None)
    return r


def to_weekly(daily_ret: pd.Series) -> pd.Series:
    """Compound daily returns to weekly (W-FRI). Equity trades ~5 days/week
    (NYSE calendar), crypto/funding trades 7 -- weekly compounding is the
    common frequency at which both series can be fairly correlated and
    blended, rather than forcing a 24/7 series onto a 5-day one (or vice
    versa) at daily granularity."""
    return daily_ret.add(1.0).resample("W-FRI").prod() - 1.0


def stats_weekly(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "maxdd": 0.0, "years": 0.0, "weeks": len(r)}
    eq = (1 + r).cumprod()
    yrs = len(r) / 52.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0.0
    sharpe = np.sqrt(52) * r.mean() / r.std() if r.std() > 0 else 0.0
    dd = (1 - eq / eq.cummax()).max()
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": dd, "years": yrs, "weeks": len(r)}


def build_weights(vol_f: float, vol_e: float) -> dict[str, tuple[float, float]]:
    """Two reasonable weighting schemes -- checked side by side, not
    cherry-picked: naive 50/50 notional, and inverse-vol (equal risk
    contribution, using each sleeve's own weekly-annualized vol over the
    common 2020-2023 window)."""
    iv_f = (1.0 / vol_f) / (1.0 / vol_f + 1.0 / vol_e)
    iv_e = 1.0 - iv_f
    return {
        "50/50 naive": (0.5, 0.5),
        f"inverse-vol ({iv_f*100:.0f}/{iv_e*100:.0f} fund/eq)": (iv_f, iv_e),
    }


def _worst_runup_by_cadence(cadence: int) -> dict:
    worst = {}
    for coin in ("BTC", "ETH"):
        ohlc = FA.load_spot_ohlc(coin)
        ohlc = ohlc[(ohlc.index >= OVERLAP_START) & (ohlc.index <= OVERLAP_END)]
        w, wd = FA.worst_intraday_runup(ohlc, cadence)
        worst[coin] = (w, wd)
    return worst


def funding_leg_survival(L_port: float, cadence: int = 2, worst: dict | None = None) -> dict:
    """Does the funding-arb leg's perp margin survive this L_port, GIVEN it
    already carries L_EMBED_BASE of its own leverage? The two compound
    (L_eff = L_EMBED_BASE * L_port), they do not add -- portfolio leverage
    scales up the notional this sleeve holds per dollar of its own allocated
    capital by the same factor it scales the rest of the book, so the
    perp leg's effective leverage relative to ITS OWN capital allocation
    grows multiplicatively. Reuses funding_arb.py's own
    worst_intraday_runup() (cadence=2 by default, matching the every-2-day
    rebalance cadence the 1.3x embedded leverage was itself validated at --
    cadence=1, daily rebalance, is also checked separately as a sensitivity)
    and MAINTENANCE_MARGIN -- the exact same liquidation mechanic already
    used for the sleeve-alone leverage_survival_report(), not a new one."""
    L_eff = L_EMBED_BASE * L_port
    maint = max(FA.MAINTENANCE_MARGIN.values())
    init_margin = 1.0 / L_eff
    liq_move = init_margin - maint
    if worst is None:
        worst = _worst_runup_by_cadence(cadence)
    binding_coin = max(worst, key=lambda c: worst[c][0])
    binding_worst = worst[binding_coin][0]
    survives = binding_worst < liq_move
    return dict(L_eff=L_eff, init_margin=init_margin, liq_move=liq_move,
                binding_coin=binding_coin, binding_worst=binding_worst,
                survives=survives, worst=worst)


def equity_leg_naive_check(r_eq_weekly: pd.Series, L_port: float) -> dict:
    """The equity sleeve has NO margin/liquidation model anywhere in this
    project -- it is a long-only cash swing system, not a levered perp. This
    is only a crude sanity check, not a real stress test: under a naive
    "borrow to scale the whole book, no dynamic margin top-up" assumption, a
    single week's loss of more than 1/L_port would wipe out the capital
    ALLOCATED to this leg. Reports the worst single week observed in the real
    2020-2023 sample against that threshold -- informative, but explicitly
    flagged as much less rigorous than the funding-arb leg's check above."""
    worst_week = r_eq_weekly.min()
    threshold = -1.0 / L_port
    wiped = worst_week < threshold
    return dict(worst_week=worst_week, threshold=threshold, wiped=wiped)


def main():
    print("=" * 88)
    print("  BLENDED BOOK: funding-arb (BTC+ETH 50/50, capital-deployed) + equity MR swing")
    print(f"  Overlap window used throughout: {OVERLAP_START} -> {OVERLAP_END} "
          "(funding data's full span -- the only real overlap with the honest, "
          "point-in-time equity data)")
    print("=" * 88)

    r_fund_notional = funding_arb_notional_return()
    r_fund_notional = r_fund_notional[(r_fund_notional.index >= OVERLAP_START) &
                                       (r_fund_notional.index <= OVERLAP_END)]
    r_fund_capital = r_fund_notional * capital_deployed_multiplier(L_EMBED_BASE)
    r_eq = equity_sleeve_return()

    s_fund_d = FA.stats(r_fund_capital)
    print(f"\n  Funding-arb sleeve (BTC+ETH 50/50, binance, capital-deployed basis, "
          f"L_embed={L_EMBED_BASE:.1f}x):")
    print(f"    daily: CAGR {s_fund_d['cagr']*100:+.2f}%  Sharpe {s_fund_d['sharpe']:.2f}  "
          f"MaxDD {s_fund_d['maxdd']*100:.2f}%  ({s_fund_d['years']:.2f}y, "
          f"{len(r_fund_capital)} obs)")
    s_eq_d = FA.stats(r_eq)
    print(f"  Equity MR swing sleeve (point-in-time S&P 500, risk=1%/trade):")
    print(f"    daily: CAGR {s_eq_d['cagr']*100:+.2f}%  Sharpe {s_eq_d['sharpe']:.2f}  "
          f"MaxDD {s_eq_d['maxdd']*100:.2f}%  ({s_eq_d['years']:.2f}y, {len(r_eq)} obs)")

    r_fund_w = to_weekly(r_fund_capital)
    r_eq_w = to_weekly(r_eq)
    common = r_fund_w.index.intersection(r_eq_w.index)
    f_w, e_w = r_fund_w.reindex(common).dropna(), r_eq_w.reindex(common).dropna()
    common = f_w.index.intersection(e_w.index)
    f_w, e_w = f_w.reindex(common), e_w.reindex(common)
    corr = f_w.corr(e_w)
    vol_f = f_w.std() * np.sqrt(52)
    vol_e = e_w.std() * np.sqrt(52)

    print(f"\n  Weekly-resampled overlap for correlation/blend: {len(common)} weeks "
          f"({common[0].date()} -> {common[-1].date()})")
    print(f"  Correlation(funding-arb capital-return, equity MR return), weekly: "
          f"{corr:+.3f}")
    print(f"  Annualized weekly-return vol: funding-arb {vol_f*100:.2f}%  "
          f"equity {vol_e*100:.2f}%")
    if len(common) < 100:
        print(f"  CAVEAT: only {len(common)} weekly observations -- a short sample "
              "(the honest overlap is inherently only ~4 years). Treat the correlation "
              "and every downstream number as indicative, not precise.")

    weights = build_weights(vol_f, vol_e)
    print(f"\n  Weighting schemes tested (not cherry-picked -- both reported):")
    for name, (wf, we) in weights.items():
        print(f"    {name}: funding-arb {wf*100:.0f}% / equity {we*100:.0f}%")

    print("\n" + "-" * 88)
    print("  PORTFOLIO-LEVERAGE SWEEP (additional leverage on the WHOLE blended book,")
    print("  stacking multiplicatively on top of the funding-arb sleeve's own 1.3x)")
    print("-" * 88)

    worst_c2 = _worst_runup_by_cadence(2)  # primary cadence -- matches the 1.3x embedded rec.
    worst_c1 = _worst_runup_by_cadence(1)  # sensitivity -- most-active, daily rebalance
    print(f"\n  Worst adverse rally (2020-2023, spot-as-perp-proxy), reused from funding_arb.py:")
    for cad, w in (("2-day (primary)", worst_c2), ("1-day (sensitivity)", worst_c1)):
        print(f"    cadence={cad}: BTC {w['BTC'][0]*100:.1f}%  ETH {w['ETH'][0]*100:.1f}%")

    results = {}
    for wname, (wf, we) in weights.items():
        print(f"\n  === Weighting: {wname} ===")
        blend_w = wf * f_w + we * e_w
        print(f"  {'L_port':>7} | {'L_eff (fund leg)':>16} | {'CAGR':>8} {'Sharpe':>7} "
              f"{'MaxDD':>7} | {'Fund leg, 2d cadence':>24} | {'Fund leg, 1d cadence':>24} | "
              f"{'Equity leg (naive)':>19}")
        rows = []
        for L in LEV_GRID:
            r_lev = blend_w * L
            s = stats_weekly(r_lev)
            surv = funding_leg_survival(L, worst=worst_c2)
            surv1 = funding_leg_survival(L, worst=worst_c1)
            eqchk = equity_leg_naive_check(e_w, L)

            def _fmt(sv):
                if sv["survives"]:
                    return f"yes ({sv['liq_move']*100:.1f}% cushion > {sv['binding_worst']*100:.1f}% worst)"
                return f"NO -LIQ- ({sv['binding_worst']*100:.1f}% worst > {sv['liq_move']*100:.1f}% cushion)"

            eq_str = "wiped (naive)" if eqchk["wiped"] else "ok"
            print(f"  {L:>6.1f}x | {surv['L_eff']:>15.2f}x | {s['cagr']*100:7.2f}% "
                  f"{s['sharpe']:7.2f} {s['maxdd']*100:6.2f}% | {_fmt(surv):>24} | "
                  f"{_fmt(surv1):>24} | {eq_str:>19}")
            rows.append(dict(L_port=L, L_eff=surv["L_eff"], cagr=s["cagr"],
                              sharpe=s["sharpe"], maxdd=s["maxdd"],
                              fund_survives=surv["survives"],
                              fund_survives_1d=surv1["survives"],
                              eq_wiped=eqchk["wiped"]))
        results[wname] = rows

    print("\n" + "-" * 88)
    print("  20-30%/yr TARGET CHECK")
    print("-" * 88)
    for wname, rows in results.items():
        hits = [r for r in rows if 0.20 <= r["cagr"] <= 0.30]
        safe_hits = [r for r in hits if r["fund_survives"] and not r["eq_wiped"]]
        print(f"\n  {wname}:")
        if hits:
            for r in hits:
                tag = "SAFE (survived 2020-2023 incl. Jan'21 rally + 2022 crash)" \
                    if (r["fund_survives"] and not r["eq_wiped"]) else \
                    "UNSAFE -- would have been liquidated/wiped in-sample"
                print(f"    L_port={r['L_port']:.1f}x -> CAGR {r['cagr']*100:.1f}% -- {tag}")
        else:
            below = [r for r in rows if r["cagr"] < 0.20]
            above = [r for r in rows if r["cagr"] > 0.30]
            if below and not above:
                print("    Never reaches 20% in the tested grid (1x-5x) -- would need even "
                      "more leverage.")
            elif above:
                print("    Jumps from <20% straight past 30% between grid points -- no "
                      "leverage level in the tested grid lands inside the 20-30% band exactly; "
                      "see full table above.")
        if not safe_hits and hits:
            print("    -> None of the 20-30% hits are SAFE per the historical liquidation "
                  "check.")
        max_safe = max((r for r in rows if r["fund_survives"] and not r["eq_wiped"]),
                       key=lambda r: r["cagr"], default=None)
        if max_safe:
            print(f"    Highest SAFE CAGR (2-day cadence, primary): {max_safe['cagr']*100:.1f}% "
                  f"at L_port={max_safe['L_port']:.1f}x (MaxDD {max_safe['maxdd']*100:.1f}%)")
        max_safe_1d = max((r for r in rows if r["fund_survives_1d"] and not r["eq_wiped"]),
                           key=lambda r: r["cagr"], default=None)
        if max_safe_1d:
            print(f"    Highest SAFE CAGR (1-day cadence, most-active sensitivity): "
                  f"{max_safe_1d['cagr']*100:.1f}% at L_port={max_safe_1d['L_port']:.1f}x "
                  f"(MaxDD {max_safe_1d['maxdd']*100:.1f}%)")

    print("\n" + "=" * 88)
    print("  See docs/RESEARCH.md, 'Can leverage reach 20-30%/yr? (the honest ceiling)'")
    print("  for the full interpretation and honest verdict.")
    print("=" * 88)


if __name__ == "__main__":
    main()
