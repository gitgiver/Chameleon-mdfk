#!/usr/bin/env bash
#
# Production-path sweep: the agents decide the structure (no FIXED_CONF / TARGET_LEAF) and the
# index is built without PILOT_DIAGNOSTICS. One process per (dataset, leaf-layout condition) so a
# single build serves the point measurement and every range selectivity -- the structure and the
# leaf layout do not depend on s.
#
# Conditions are window thresholds:
#   off  -> every leaf stays a hash leaf (Chameleon as-is)
#   4    -> the conservative knee from phase 1: small point-query cost, small memory gain
#   16   -> the knee phase 1 reported for skewed data
#   1e9  -> every eligible leaf ordered: the ceiling, for bracketing
#
# Overridable: DATASETS, CONDS, SELS, N, Q, REPEATS
# Usage: scripts/sweep_prod.sh [log_dir]
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/Release/prod_bench"
LOGDIR="${1:-$REPO/Release/logs_prod}"

DATASETS="${DATASETS:-uden.data face.data local_skew.data mixed.data}"
CONDS="${CONDS:-off 4 16 1e9}"
SELS="${SELS:-0.000001,0.00001,0.0001,0.001,0.01}"
N="${N:-20000000}"
Q="${Q:-40000000}"
REPEATS="${REPEATS:-5}"

mkdir -p "$LOGDIR"

for ds in $DATASETS; do
    for cond in $CONDS; do
        w=""
        [ "$cond" != "off" ] && w="$cond"
        log="$LOGDIR/${ds}_w${cond}.log"
        env PROD_DATASET="$ds" PROD_LENGTH="$N" PROD_QUERIES="$Q" PROD_REPEATS="$REPEATS" \
            PROD_RANGE="$SELS" ${w:+LEAF_WINDOW_THRESHOLD=$w} "$BIN" > "$log" 2>&1
        printf '[%s] %-17s tau=%-5s exit=%d\n' "$(date +%H:%M:%S)" "$ds" "$cond" "$?"
    done
done

echo "wrote logs to $LOGDIR"
