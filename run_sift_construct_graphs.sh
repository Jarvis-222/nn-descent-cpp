#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[error] Command failed: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/build/nn_descent}"

DATA="$SCRIPT_DIR/data/sift/sift_base.fvecs"
GT="$SCRIPT_DIR/data/sift1M_l2_gt.bin"
TAG="sift1m"

K=10
MC=40
PROJ=32
PTS=("nofilter" "0.99" "0.95" "0.90" "0.85" "0.80" "0.75" "0.70" "0.65" "0.60")
INITS=("random" "rptree")
ONLY_INIT="${ONLY_INIT:-}"
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
  echo "[$TAG] init=$init config=$pt"
  echo "report=$report"
  echo "graph=$graph"
  echo "============================================================"
  "${cmd[@]}"
}

echo "=== SIFT 1M graph construction sweep ==="
echo "repo=$SCRIPT_DIR"
echo "data=$DATA"
echo "gt=$GT"
echo "k=$K mc=$MC projections=$PROJ"
if [[ -n "$ONLY_INIT" ]]; then
  echo "only_init=$ONLY_INIT"
fi
if [[ "$MAX_JOBS" != "0" ]]; then
  echo "max_jobs=$MAX_JOBS"
fi

for init in "${INITS[@]}"; do
  if [[ -n "$ONLY_INIT" && "$init" != "$ONLY_INIT" ]]; then
    continue
  fi
  for pt in "${PTS[@]}"; do
    run_one "$init" "$pt"
    job_count=$((job_count + 1))
    if [[ "$MAX_JOBS" != "0" && "$job_count" -ge "$MAX_JOBS" ]]; then
      echo
      echo "=== Stopped after $job_count job(s) because MAX_JOBS=$MAX_JOBS ==="
      exit 0
    fi
  done
done

echo
echo "=== All SIFT 1M graph construction jobs completed ==="

