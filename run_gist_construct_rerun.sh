#!/usr/bin/env bash
# Rerun GIST 1M construction configs whose *_build.txt currently has
# final_recall=0 (GT wasn't loaded at the time), plus the rptree+nofilter
# config which has no build report at all.
#
# Random init — 7 configs with recall=0 to rerun:
#   pt0.60, pt0.65, pt0.70, pt0.75, pt0.80, pt0.85, pt0.90
# rptree init — 9 pt configs with recall=0 + nofilter missing entirely (10):
#   nofilter, pt0.60, pt0.65, pt0.70, pt0.75, pt0.80, pt0.85, pt0.90, pt0.95, pt0.99
# Skipped (already have valid recall):
#   random nofilter, random pt0.95, random pt0.99

set -euo pipefail
trap 'echo "[error] Command failed: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/build/nn_descent}"

DATA="$SCRIPT_DIR/data/gist/gist_base.fvecs"
# Override with GT=<path> ./run_gist_construct_rerun.sh if GT lives elsewhere.
GT="${GT:-$SCRIPT_DIR/data/gist1m_l2_gt.bin}"
TAG="gist1m"

K=10
MC=40
PROJ=32

# Configs to rerun: "<init>:<pt>" pairs. pt="nofilter" means no --proj-filter flags.
CONFIGS=(
  "random:0.90"
  "random:0.85"
  "random:0.80"
  "random:0.75"
  "random:0.70"
  "random:0.65"
  "random:0.60"
  "rptree:nofilter"
  "rptree:0.99"
  "rptree:0.95"
  "rptree:0.90"
  "rptree:0.85"
  "rptree:0.80"
  "rptree:0.75"
  "rptree:0.70"
  "rptree:0.65"
  "rptree:0.60"
)

MAX_JOBS="${MAX_JOBS:-0}"
job_count=0

RESULTS_DIR="$SCRIPT_DIR/results"
GRAPHS_DIR="$SCRIPT_DIR/graphs"

if [[ ! -x "$BIN" ]]; then
  echo "[error] Binary not found or not executable: $BIN"
  echo "Build first with:"
  echo "  cmake -S \"$SCRIPT_DIR\" -B \"$SCRIPT_DIR/build\" -DCMAKE_BUILD_TYPE=Release"
  echo "  cmake --build \"$SCRIPT_DIR/build\" -j"
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "[error] Data file not found: $DATA"
  exit 1
fi

if [[ ! -f "$GT" ]]; then
  echo "[error] Ground-truth file not found: $GT"
  echo "        Set GT=<path> env var to point at the construction GT file."
  exit 1
fi

mkdir -p "$RESULTS_DIR" "$GRAPHS_DIR"

run_one() {
  local init="$1"
  local pt="$2"
  local out_base="$RESULTS_DIR/${TAG}_${init}"
  local graph_base="$GRAPHS_DIR/${TAG}_${init}"
  local cmd=(
    "$BIN"
    --data "$DATA"
    --k "$K"
    --mc "$MC"
    --init "$init"
    --load-gt "$GT"
  )

  local suffix
  if [[ "$pt" == "nofilter" ]]; then
    suffix="nofilter"
  else
    suffix="pt${pt/./}"
    cmd+=(--proj-filter --num-projections "$PROJ" --filter-confidence "$pt")
  fi

  local report="${out_base}_${suffix}_build.txt"
  local graph="${graph_base}_${suffix}.knng"

  cmd+=(--save-graph "$graph" --output "$report")

  echo
  echo "============================================================"
  echo "[$TAG] init=$init config=$pt  (rerun with GT for recall)"
  echo "report=$report"
  echo "graph=$graph"
  echo "============================================================"
  "${cmd[@]}"
}

echo "=== GIST 1M construction rerun (recall=0 configs) ==="
echo "repo=$SCRIPT_DIR"
echo "data=$DATA"
echo "gt=$GT"
echo "k=$K mc=$MC projections=$PROJ"
echo "configs=${#CONFIGS[@]}"
if [[ "$MAX_JOBS" != "0" ]]; then
  echo "max_jobs=$MAX_JOBS"
fi

for entry in "${CONFIGS[@]}"; do
  init="${entry%%:*}"
  pt="${entry##*:}"
  run_one "$init" "$pt"
  job_count=$((job_count + 1))
  if [[ "$MAX_JOBS" != "0" && "$job_count" -ge "$MAX_JOBS" ]]; then
    echo
    echo "=== Stopped after $job_count job(s) because MAX_JOBS=$MAX_JOBS ==="
    exit 0
  fi
done

echo
echo "=== GIST 1M construction rerun complete (${job_count} jobs) ==="
