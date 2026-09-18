#!/usr/bin/env bash
#
# Production-path sweep: the agents decide the structure (no FIXED_CONF / TARGET_LEAF) and the
# index is built without PILOT_DIAGNOSTICS. Measures memory, point-query and range-query latency.
#
# Two conditions per dataset:
#   off -> every leaf stays a hash leaf (Chameleon as-is)
#   on  -> LEAF_WINDOW_THRESHOLD=1e9, i.e. every eligible leaf becomes an ordered leaf
#
# Overridable: DATASETS, CONDS, SELS, N, Q, REPEATS
# Usage: scripts/sweep_prod.sh [output.csv]
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/Release/prod_bench"
OUT="${1:-$REPO/Release/prod_results.csv}"
LOG="$REPO/Release/logs_prod"

DATASETS="${DATASETS:-uden.data face.data local_skew.data mixed.data}"
CONDS="${CONDS:-off on}"
SELS="${SELS:-0.000001 0.00001 0.0001 0.001 0.01}"
N="${N:-20000000}"
Q="${Q:-40000000}"
REPEATS="${REPEATS:-5}"

mkdir -p "$LOG"
: > "$OUT"

run() {
    local ds="$1" cond="$2" tag="$3" ; shift 3
    local w=""
    [ "$cond" = "on" ] && w="1000000000"
    env PROD_DATASET="$ds" PROD_LENGTH="$N" PROD_QUERIES="$Q" PROD_REPEATS="$REPEATS" \
        ${w:+LEAF_WINDOW_THRESHOLD=$w} "$@" "$BIN" > "$LOG/${ds}_${tag}.log" 2>&1
    grep -E "^prod:|^  (point|range):" "$LOG/${ds}_${tag}.log" \
        | sed -E "s/^prod: //; s/^  /  /" | sed "s/^/$ds,$cond,$tag,/" >> "$OUT"
    printf '[%s] %-17s %-4s %s\n' "$(date +%H:%M:%S)" "$ds" "$cond" "$tag"
}

for ds in $DATASETS; do
    for cond in $CONDS; do
        run "$ds" "$cond" point
        for s in $SELS; do
            run "$ds" "$cond" "r$s" PROD_RANGE="$s"
        done
    done
done

echo "wrote $OUT"
