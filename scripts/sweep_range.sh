#!/usr/bin/env bash
#
# Range-query selectivity sweep.
#
# Runs the harness once per (selectivity, leaf layout) cell and writes the parsed summary to $1.
# The query set is deterministic (fixed seed inside the harness), so repeating this script measures
# measurement noise only -- that is the intended way to check whether a difference is real.
#
# Overridable: DS, N, CONF, TARGET_LEAF, TARGET_FANOUT, REPEATS, EMIT, SELS, WINDOW
# Usage: scripts/sweep_range.sh [output.csv]
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/Release/pilot_range"
OUT="${1:-$REPO/Release/pilot_range_runs.csv}"
LOGDIR="${LOGDIR:-$REPO/Release/logs_range}"

DS="${DS:-uden.data}"
N="${N:-20000000}"
CONF="${CONF:-256,16}"
TARGET_LEAF="${TARGET_LEAF:-1024}"
TARGET_FANOUT="${TARGET_FANOUT:-16}"
REPEATS="${REPEATS:-5}"
EMIT="${EMIT:-20000000}"
# Large enough to convert every eligible leaf, i.e. the ALL-ORDERED condition. A modest value
# such as 16 converts only a fraction of the leaves on skewed data (26% on face), which measures a
# hash/ordered mixture rather than the ordered layout.
WINDOW="${WINDOW:-1000000000}"
SELS="${SELS:-0.000001 0.00001 0.0001 0.001 0.01}"

mkdir -p "$LOGDIR"
echo "s,mode,ns_min,ns_median,ns_max,result_avg,errors" > "$OUT"

for s in $SELS; do
    for mode in hash ordered; do
        common=(PILOT_DATASET="$DS" PILOT_LENGTH="$N" PILOT_RANGE="$s" PILOT_REPEATS="$REPEATS"
                PILOT_EMIT_VOLUME="$EMIT" FIXED_CONF="$CONF" TARGET_LEAF="$TARGET_LEAF"
                TARGET_FANOUT="$TARGET_FANOUT")
        LOG="$LOGDIR/${DS}_s${s}_${mode}.log"
        cd "$REPO/Release" || exit 1
        if [ "$mode" = "ordered" ]; then
            env "${common[@]}" LEAF_WINDOW_THRESHOLD="$WINDOW" "$BIN" > "$LOG" 2>&1
        else
            env "${common[@]}" "$BIN" > "$LOG" 2>&1
        fi
        grep -oE "result_avg=[0-9.]+ +ns=[0-9.]+ +\(med [0-9.]+ max [0-9.]+.*errors=[0-9]+" "$LOG" \
            | sed -E "s/result_avg=([0-9.]+) +ns=([0-9.]+) +\(med ([0-9.]+) max ([0-9.]+).*errors=([0-9]+)/$s,$mode,\2,\3,\4,\1,\5/" \
            >> "$OUT"
        printf '[%s] %s s=%-9s %s\n' "$(date +%H:%M:%S)" "$DS" "$s" "$mode"
    done
done

echo "wrote $OUT"
