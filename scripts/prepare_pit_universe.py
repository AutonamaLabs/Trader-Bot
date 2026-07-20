#!/usr/bin/env python3
"""Build the point-in-time S&P 500 universe: splice the free 2000-2017 Stooq-
sourced history (data/universe_pit/hsmd_raw/, incl. many delisted/acquired
names) with the adjusted 2017-2024 survivor history already in the repo
(data/universe_broad/) into one continuous per-symbol CSV per ticker, plus a
compact point-in-time membership mask usable by scripts/validate_pit_universe.py.

Run scripts/fetch_pit_universe.sh first. Output (all git-ignored, rebuild
locally):
    data/universe_pit/prices/<TICKER>.csv   -- date,open,high,low,close,volume
    data/universe_pit/membership.csv        -- unchanged, copied through

Splice method: where a ticker has BOTH sources, the pre-2017-11-10 (HSMD) leg
is rescaled by a single constant factor so its last close matches the
universe_broad close on the first trading day after the cutover -- i.e. the
switchover day itself carries ~0% fabricated return, and each source's own
day-to-day % moves (what the strategy actually trades on) are untouched. This
is the same idea as back-adjusting a continuous futures contract at rollover.

    python scripts/prepare_pit_universe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIT = ROOT / "data" / "universe_pit"
HSMD_DIR = PIT / "hsmd_raw"
BROAD_DIR = ROOT / "data" / "universe_broad"
OUT_DIR = PIT / "prices"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # utc=True + normalize(): collapse both tz-naive (HSMD) and tz-aware
    # (universe_broad, "-05:00" etc.) sources onto the SAME calendar-date grid
    # so a date shared by both sources dedupes cleanly instead of surviving as
    # two near-identical rows a few hours apart in UTC.
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    return df


load_hsmd = _load
load_broad = _load


def splice(hsmd: pd.DataFrame | None, broad: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    """Returns (combined_df, source_tag)."""
    if hsmd is None and broad is None:
        return None, "none"
    if hsmd is None:
        return broad, "broad_only"
    if broad is None:
        return hsmd, "hsmd_only"

    cutover = hsmd.index.max()
    broad_after = broad[broad.index > cutover]
    if broad_after.empty:
        return hsmd, "hsmd_only"  # broad data doesn't extend past the HSMD cutoff
    anchor_close = broad_after["close"].iloc[0]
    hsmd_close = hsmd["close"].iloc[-1]
    if hsmd_close <= 0:
        return broad_after, "broad_only"
    scale = anchor_close / hsmd_close
    hsmd_scaled = hsmd.copy()
    for c in ("open", "high", "low", "close"):
        hsmd_scaled[c] *= scale
    combined = pd.concat([hsmd_scaled, broad_after])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined, "spliced"


def main():
    if not PIT.exists():
        print(f"!! {PIT} missing; run scripts/fetch_pit_universe.sh first")
        return 1
    membership_src = PIT / "membership.csv"
    if not membership_src.exists():
        print(f"!! {membership_src} missing; run scripts/fetch_pit_universe.sh first")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hsmd_files = {f.stem.upper(): f for f in HSMD_DIR.glob("*.csv")} if HSMD_DIR.exists() else {}
    broad_files = {f.stem.upper(): f for f in BROAD_DIR.glob("*.csv")} if BROAD_DIR.exists() else {}
    all_tickers = sorted(set(hsmd_files) | set(broad_files))

    counts = {"spliced": 0, "hsmd_only": 0, "broad_only": 0, "skipped": 0}
    for t in all_tickers:
        try:
            hsmd = load_hsmd(hsmd_files[t]) if t in hsmd_files else None
        except Exception:
            hsmd = None
        try:
            broad = load_broad(broad_files[t]) if t in broad_files else None
        except Exception:
            broad = None
        combined, tag = splice(hsmd, broad)
        if combined is None or len(combined) < 260:
            counts["skipped"] += 1
            continue
        combined = combined.reset_index().rename(columns={"date": "date"})
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        combined.to_csv(OUT_DIR / f"{t}.csv", index=False)
        counts[tag] += 1

    print(f"Wrote {sum(v for k, v in counts.items() if k != 'skipped')} symbol CSVs to {OUT_DIR}")
    print(f"  spliced (hsmd->broad) : {counts['spliced']}")
    print(f"  hsmd only (ends ~2017): {counts['hsmd_only']}")
    print(f"  broad only (recent IPO/current-only): {counts['broad_only']}")
    print(f"  skipped (< 260 rows)  : {counts['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
