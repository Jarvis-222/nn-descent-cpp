#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/nn_descent}"
DATASET="${1:-100k}"

case "$DATASET" in
  100k)
    DATA="data/gist/gist_base_100k.fvecs"
    GT="data/gist100k_l2_gt.bin"
    TAG="gist100k"
    ;;
  1m)
    DATA="data/gist/gist_base.fvecs"
    GT="data/gist1m_l2_gt.bin"
    TAG="gist1m"
    ;;
  *)
    echo "Usage: $0 [100k|1m]"
    exit 1
    ;;
esac

K=10
MC=40
PROJ=32
PTS=("nofilter" "0.99" "0.95" "0.90" "0.85" "0.80" "0.75" "0.70" "0.65" "0.60")
INITS=("random" "rptree")
ONLY_INIT="${ONLY_INIT:-}"
MAX_JOBS="${MAX_JOBS:-0}"
job_count=0

mkdir -p results graphs

run_one() {
  local init="$1"
  local pt="$2"
  local out_base="results/${TAG}_${init}"
  local graph_base="graphs/${TAG}_${init}"
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

echo "=== GIST graph construction sweep ==="
echo "dataset=$DATASET"
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
echo "=== All graph construction jobs completed ==="
