#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[error] Command failed: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/build/nn_descent}"

DATA="$SCRIPT_DIR/data/sift/sift_base.fvecs"
GT="$SCRIPT_DIR/data/sift1m_l2_gt.bin"
TAG="sift1m"

K=10
INIT="random"
# mc=40 already exists in sift1m_random_nofilter_build.txt
MCS=(10 15 20 25 30 35)

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

echo "=== SIFT 1M mc sweep (nofilter, init=$INIT) ==="
echo "repo=$SCRIPT_DIR"
echo "data=$DATA"
echo "gt=$GT"
echo "k=$K mcs=${MCS[*]}"

for MC in "${MCS[@]}"; do
  report="$RESULTS_DIR/${TAG}_mc${MC}_nofilter_r1.txt"
  graph="$GRAPHS_DIR/${TAG}_mc${MC}_nofilter_r1.knng"

  echo
  echo "============================================================"
  echo "[$TAG] init=$INIT mc=$MC (nofilter)"
  echo "report=$report"
  echo "graph=$graph"
  echo "============================================================"

  "$BIN" \
    --data "$DATA" \
    --k "$K" \
    --mc "$MC" \
    --init "$INIT" \
    --load-gt "$GT" \
    --save-graph "$graph" \
    --output "$report"
done

echo
echo "=== SIFT 1M mc sweep complete ==="
