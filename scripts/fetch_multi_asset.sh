#!/usr/bin/env bash
# Download daily OHLC data for the diversified multi-asset-class trend-following
# test (docs/RESEARCH.md, "Diversified multi-asset trend-following (CTA-style)")
# into data/multi_asset/. All git-ignored + re-downloadable.
#
# Sources (both reachable at raw.githubusercontent.com, verified with curl -w
# "%{http_code}" before use — this sandbox cannot reach Yahoo/Stooq/FRED):
#
#   1. scienclick/stocks (mirror of the Kaggle "Huge Stock Market Dataset",
#      same underlying data as scripts/fetch_pit_universe.sh's HSMD source) —
#      daily OHLC+Volume for ~30 ETFs, ALL FROZEN 2005-02-25 -> 2017-11-10.
#      Gives bond ETF proxies (TLT/IEF/SHY/BWX/IGOV), commodity ETFs
#      (SLV/JJC/CORN/WEAT/SOYB/CANE/JO/BAL), a USD-index proxy (UUP), and
#      global equity index ETFs (IWM/DIA/EFA/EEM/EWJ/EWG/EWU/EWQ/EWY).
#   2. datasets/oil-prices + datasets/natural-gas (Frictionless Data / GitHub
#      org "datasets") — WTI, Brent, Henry Hub natural gas, close-only but
#      LIVE-UPDATED through the present (1986/1987/1997 -> today). These are
#      what let the commodity asset class stay live past 2017 so 2020/2022
#      can be tested without bonds.
#
# Known limitation (documented in RESEARCH.md, not hidden): the bond class has
# NO source past 2017-11-10 reachable from this sandbox (FRED/Treasury/Stooq
# are all 403'd here) — no single-country Bund/Gilt/JGB series was found
# either. Bond exposure in this backtest is therefore real but WINDOW-LIMITED
# to 2005-2017.
set -uo pipefail
OUT="$(dirname "$0")/../data/multi_asset"
mkdir -p "$OUT"

dl(){ curl -sS --max-time 60 "$1" -o "$2" && echo "  ok $(basename "$2")" || echo "  FAIL $2"; }

SC="https://raw.githubusercontent.com/scienclick/stocks/master/data/ETFs"

echo "== Bonds (2005-2017 window; best available from this sandbox) =="
dl "$SC/tlt.us.txt"  "$OUT/TLT.csv"    # 20+Y UST
dl "$SC/ief.us.txt"  "$OUT/IEF.csv"    # 7-10Y UST
dl "$SC/shy.us.txt"  "$OUT/SHY.csv"    # 1-3Y UST
dl "$SC/bwx.us.txt"  "$OUT/BWX.csv"    # int'l govt bonds ex-US (unhedged)
dl "$SC/igov.us.txt" "$OUT/IGOV.csv"   # int'l treasury basket (incl. JP/UK/DE)

echo "== Commodities =="
dl "$SC/slv.us.txt"  "$OUT/SLV.csv"    # silver
dl "$SC/jjc.us.txt"  "$OUT/JJC.csv"    # copper
dl "$SC/corn.us.txt" "$OUT/CORN.csv"
dl "$SC/weat.us.txt" "$OUT/WEAT.csv"   # wheat
dl "$SC/soyb.us.txt" "$OUT/SOYB.csv"   # soybeans
dl "$SC/cane.us.txt" "$OUT/CANE.csv"   # sugar
dl "$SC/jo.us.txt"   "$OUT/JO.csv"     # coffee
dl "$SC/bal.us.txt"  "$OUT/BAL.csv"    # cotton
dl "https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv" "$OUT/WTI.csv"
dl "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv" "$OUT/BRENT.csv"
dl "https://raw.githubusercontent.com/datasets/natural-gas/main/data/daily.csv" "$OUT/NATGAS.csv"

echo "== Global equity index ETFs =="
dl "$SC/iwm.us.txt" "$OUT/IWM.csv"    # Russell 2000
dl "$SC/dia.us.txt" "$OUT/DIA.csv"    # Dow Jones
dl "$SC/efa.us.txt" "$OUT/EFA.csv"    # MSCI EAFE (developed intl)
dl "$SC/eem.us.txt" "$OUT/EEM.csv"    # MSCI Emerging Markets
dl "$SC/ewj.us.txt" "$OUT/EWJ.csv"    # Japan
dl "$SC/ewg.us.txt" "$OUT/EWG.csv"    # Germany (DAX proxy)
dl "$SC/ewu.us.txt" "$OUT/EWU.csv"    # UK (FTSE proxy)
dl "$SC/ewq.us.txt" "$OUT/EWQ.csv"    # France
dl "$SC/ewy.us.txt" "$OUT/EWY.csv"    # South Korea

echo "== FX (extra: USD index) =="
dl "$SC/uup.us.txt" "$OUT/UUP.csv"    # USD index

# Manifest: symbol -> asset_class, consumed by scripts/trend_multi_asset.py
cat > "$OUT/manifest.csv" <<'CSV'
symbol,asset_class
TLT,bond
IEF,bond
SHY,bond
BWX,bond
IGOV,bond
SLV,commodity
JJC,commodity
CORN,commodity
WEAT,commodity
SOYB,commodity
CANE,commodity
JO,commodity
BAL,commodity
WTI,commodity
BRENT,commodity
NATGAS,commodity
IWM,equity
DIA,equity
EFA,equity
EEM,equity
EWJ,equity
EWG,equity
EWU,equity
EWQ,equity
EWY,equity
UUP,fx
CSV
echo "Manifest written: $OUT/manifest.csv"
echo "Multi-asset data in $OUT"
