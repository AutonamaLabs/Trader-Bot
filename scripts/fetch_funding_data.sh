#!/usr/bin/env bash
# Download real historical perpetual-futures funding-rate history for BTC and
# ETH (USDT-margined) across Binance / Bybit / Gate.io into data/funding/.
# Source: supervik/historical-funding-rates-fetcher, a public GitHub mirror
# that scraped each exchange's funding-history API (2020-2023ish). All
# git-ignored + re-downloadable, same pattern as fetch_data.sh /
# fetch_swing_data.sh (raw.githubusercontent.com is the only reachable host
# for exchange-adjacent data in this sandbox; direct Binance/Bybit/OKX/Deribit
# REST APIs all return 403 here).
set -uo pipefail
OUT="$(dirname "$0")/../data/funding"
mkdir -p "$OUT"
BASE="https://raw.githubusercontent.com/supervik/historical-funding-rates-fetcher/main/data"

dl(){
  code=$(curl -sS --max-time 60 -o "$2" -w "%{http_code}" "$1")
  if [ "$code" = "200" ]; then echo "  ok $(basename "$2") ($(wc -l < "$2") rows)"; else echo "  FAIL ($code) $2"; rm -f "$2"; fi
}

dl "$BASE/BTC-USDT/BTC-USDT_binance_2020-01-01_2024-01-01_funding_history.csv" "$OUT/BTCUSDT_binance.csv"
dl "$BASE/BTC-USDT/BTC-USDT_bybit_2021-01-01_2024-01-01_funding_history.csv"   "$OUT/BTCUSDT_bybit.csv"
dl "$BASE/BTC-USDT/BTC-USDT_gate_2020-01-01_2024-01-01_funding_history.csv"    "$OUT/BTCUSDT_gate.csv"
dl "$BASE/ETH-USDT/ETH-USDT_binance_2020-01-01_2024-01-01_funding_history.csv" "$OUT/ETHUSDT_binance.csv"
dl "$BASE/ETH-USDT/ETH-USDT_bybit_2021-01-01_2024-01-01_funding_history.csv"   "$OUT/ETHUSDT_bybit.csv"
dl "$BASE/ETH-USDT/ETH-USDT_gate_2020-01-01_2024-01-01_funding_history.csv"    "$OUT/ETHUSDT_gate.csv"

echo "Funding-rate data in $OUT"
echo "(Spot price history reused from data/swing/BTCUSD.csv, data/swing/ETHUSD.csv"
echo " -- run scripts/fetch_swing_data.sh first if those are missing.)"
