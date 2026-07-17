#!/usr/bin/env bash
# Download real daily/H4/H1 FX + gold history (ejtraderLabs mirror of MT4 data,
# 2012-2022) into data/research/raw/. These files are git-ignored and
# re-downloadable, so the repo stays lean.
set -euo pipefail

BASE="https://raw.githubusercontent.com/ejtraderLabs/historical-data/main"
OUT="$(dirname "$0")/../data/research/raw"
mkdir -p "$OUT"

INSTRUMENTS="EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF EURJPY GBPJPY EURGBP AUDJPY XAUUSD"
TFS="h1 h4 d1"

n=0
for p in $INSTRUMENTS; do
  for tf in $TFS; do
    curl -sS --max-time 120 "$BASE/$p/${p}${tf}.csv" -o "$OUT/${p}_${tf}.csv" &
    n=$((n+1)); [ $((n % 6)) -eq 0 ] && wait
  done
done
wait
echo "Downloaded $(ls -1 "$OUT" | wc -l) files into $OUT"
