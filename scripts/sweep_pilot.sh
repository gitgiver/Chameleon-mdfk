#!/usr/bin/env bash
#
# Phase-1 sweep: the leaf-layout criterion at a fixed index structure.
#
# The structure is pinned so that the leaf layout is the only independent variable:
#   FIXED_CONF     bypass DARE, so every dataset gets the same Configuration
#   TARGET_LEAF    bypass TSMDP, so splitting stops at a bounded leaf size
# Without this, the policy gives uden ~142-key leaves and face ~14.7k-key leaves, and any
# difference between them is confounded with leaf size. TARGET_LEAF bounds the size but does not
# equalise it (children are cut by equal-width intervals, so skewed data still yields small
# leaves), so the per-leaf analysis must additionally condition on leaf size.
#
# Each run appends one row to Release/pilot_results.csv and writes its per-leaf diagnostics to
# Release/leaf_stats_<tag>.csv. Seeds are fixed inside the harness, so runs are deterministic.
#
# Overridable: CONF, TARGET_LEAF, TARGET_FANOUT, N, Q, DATASETS, THRESHOLDS
# Usage: scripts/sweep_pilot.sh [log_dir]
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/Release/pilot_leaf"
LOGDIR="${1:-$REPO/Release/logs}"

CONF="${CONF:-256,16}"
TARGET_LEAF="${TARGET_LEAF:-1024}"
TARGET_FANOUT="${TARGET_FANOUT:-16}"
N="${N:-20000000}"
# 2 queries per key, split into REPEATS passes; the harness reports the minimum pass. With one
# pass per 2M queries the reported latency moved ~25% between identical runs, with this it is
# reproducible to ~1% (333/331/337 ns over three runs of one configuration).
Q="${Q:-40000000}"
REPEATS="${REPEATS:-5}"
DATASETS="${DATASETS:-uden.data face.data local_skew.data mixed.data}"
THRESHOLDS="${THRESHOLDS:-0 4 16 64 256}"

mkdir -p "$LOGDIR"
cd "$REPO/Release" || exit 1

run() {
    local ds="$1" ; shift
    env PILOT_DATASET="$ds" PILOT_LENGTH="$N" PILOT_QUERIES="$Q" PILOT_REPEATS="$REPEATS" \
        FIXED_CONF="$CONF" TARGET_LEAF="$TARGET_LEAF" TARGET_FANOUT="$TARGET_FANOUT" \
        "$@" "$BIN" > "$LOGDIR/${ds}_${LOG_SUFFIX}.log" 2>&1
    printf '[%s] %-16s %-10s exit=%d\n' "$(date +%H:%M:%S)" "$ds" "$LOG_SUFFIX" "$?"
}

echo "structure: FIXED_CONF=$CONF TARGET_LEAF=$TARGET_LEAF TARGET_FANOUT=$TARGET_FANOUT"
echo "n=$N queries=$Q"
for ds in $DATASETS; do
    for w in $THRESHOLDS; do
        if [ "$w" = "0" ]; then
            # the criterion must be > 0 to be enabled, so threshold 0 means the all-hash
            # baseline, i.e. the reference the ordered runs are compared against
            LOG_SUFFIX="base"
            run "$ds"
        else
            LOG_SUFFIX="w${w}"
            run "$ds" LEAF_WINDOW_THRESHOLD="$w"
        fi
    done
done
echo "done; results in $REPO/Release/pilot_results.csv"
