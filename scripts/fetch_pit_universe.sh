#!/usr/bin/env bash
# Download the raw inputs for the point-in-time S&P 500 validation
# (docs/RESEARCH.md, "Point-in-time S&P 500 validation (2000-2024)"):
#
#   1. Point-in-time constituent history (hanshof/sp500_constituents) — one row
#      per membership-change date, 1996-01-02 to present, so we know exactly
#      which tickers were IN the S&P 500 on any given trading day (no
#      lookahead on index membership).
#   2. Bulk historical daily OHLC for as much of the constituent universe as a
#      free source will give us (neo-zhao's GitHub mirror of the Kaggle "Huge
#      Stock Market Dataset", itself sourced from Stooq's bulk EOD archive).
#      That mirror stops 2017-11-10 but — critically — it still carries many
#      tickers that were delisted/acquired/bankrupt *before* that cutoff
#      (e.g. Wachovia-era WB, Circuit City-era CC), which is exactly the
#      survivorship-bias-prone data current-ticker-only sources lack.
#
# Everything lands in data/universe_pit/ (git-ignored, like data/swing*).
# Re-run scripts/prepare_pit_universe.py afterwards to splice this with
# data/universe_broad/ (adjusted, current survivors, already in the repo) and
# build the final per-symbol CSVs used by scripts/validate_pit_universe.py.
#
# Known limitation (see docs/RESEARCH.md): ticker symbols get recycled by
# unrelated companies over 20+ years (WB, CC above are two examples) — a
# symbol-only free dataset cannot fully rule this out. Also, roughly 300 of
# the ~1000 tickers that were ever S&P 500 members 2000-2024 are not present
# in this mirror at all (mostly bankruptcy-suffix tickers like AAMRQ/ABKFQ,
# or names that took their current ticker only after 2017-11-10, e.g. BKNG,
# CARR, AMCR) — those simply drop out of the backtest universe rather than
# being faked.
set -uo pipefail

OUT="$(dirname "$0")/../data/universe_pit"
RAW="$OUT/hsmd_raw"
mkdir -p "$RAW"

echo "== Point-in-time S&P 500 constituent history (hanshof/sp500_constituents) =="
curl -sS --max-time 60 \
  "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp_500_historical_components.csv" \
  -o "$OUT/membership.csv" \
  && echo "  ok membership.csv ($(wc -l < "$OUT/membership.csv") rows)" \
  || echo "  FAIL membership.csv"

echo "== Bulk historical OHLC (neo-zhao HSMD mirror, ~7000 tickers to 2017-11-10) =="
CLONE_DIR="$(mktemp -d)"
if git clone --depth 1 -q \
    https://github.com/neo-zhao/CMSC320_Final_Tutorial_Huge_Stock_Market_Dataset \
    "$CLONE_DIR"; then
  # Copy only the tickers that were ever an S&P 500 constituent 2000-2024
  # (from membership.csv) to keep the working tree small.
  python3 - "$OUT/membership.csv" "$CLONE_DIR/Stocks" "$RAW" <<'PY'
import csv, shutil, sys
from pathlib import Path

membership_csv, src_dir, dst_dir = sys.argv[1:4]
src_dir, dst_dir = Path(src_dir), Path(dst_dir)

tickers = set()
with open(membership_csv) as f:
    r = csv.reader(f)
    next(r)
    for date, tick_str in r:
        if "2000-01-01" <= date <= "2024-12-31":
            tickers |= set(tick_str.split(","))

n = 0
for t in sorted(tickers):
    fname = t.replace(".", "-").lower() + ".us.txt"
    src = src_dir / fname
    if src.exists() and src.stat().st_size > 0:
        shutil.copy(src, dst_dir / f"{t}.csv")
        n += 1
print(f"  copied {n}/{len(tickers)} matched tickers into {dst_dir}")
PY
  rm -rf "$CLONE_DIR"
else
  echo "  FAIL cloning HSMD mirror"
fi

echo "Done. Raw inputs in $OUT — run scripts/prepare_pit_universe.py next."
