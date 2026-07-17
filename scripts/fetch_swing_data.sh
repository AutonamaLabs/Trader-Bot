#!/usr/bin/env bash
# Download daily OHLC data for the TPS-5 swing universe (index/crypto) into
# data/swing/. All git-ignored + re-downloadable. Gold comes from the FX fetch.
set -uo pipefail
OUT="$(dirname "$0")/../data/swing"
mkdir -p "$OUT"

dl(){ curl -sS --max-time 120 "$1" -o "$2" && echo "  ok $(basename "$2")" || echo "  FAIL $2"; }

# Equity indices (ETFs)
dl "https://raw.githubusercontent.com/willhjw/big_movers/main/SPY%20Historical%20Data.csv" "$OUT/SPY.csv"
dl "https://raw.githubusercontent.com/nateGeorge/simulate_leveraged_ETFs/master/eod_data/QQQ.csv" "$OUT/QQQ.csv"
# Crypto (lose on this edge by default; kept for completeness / ablation)
dl "https://raw.githubusercontent.com/Dat-TG/Cryptocurrency-Price-Prediction/main/csvdata/BTC-USD.csv" "$OUT/BTCUSD.csv"
dl "https://raw.githubusercontent.com/Dat-TG/Cryptocurrency-Price-Prediction/main/csvdata/ETH-USD.csv" "$OUT/ETHUSD.csv"

# Gold: derive from the FX research data (auto-scaled) if present.
python3 - <<'PY' 2>/dev/null || echo "  (skip gold: run scripts/fetch_data.sh first)"
import sys; sys.path.insert(0,"src")
from pathlib import Path
from trader_bot import research as R
R.load("XAUUSD","d1").to_csv("data/swing/XAUUSD.csv"); print("  ok XAUUSD.csv")
PY
echo "Swing data in $OUT"
