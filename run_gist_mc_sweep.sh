#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[error] Command failed: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/build/nn_descent}"

DATA="$SCRIPT_DIR/data/gist/gist_base.fvecs"
TAG="gist1m"

K=10
INIT="random"
# Already have: mc=25 (mc25_nofilter_r1), mc=30 (gist1m_nofilter_r1.txt),
#               mc=35 (mc35_nofilter_r1), mc=40 (gist1m_random_nofilter_build.txt)
MCS=(10 15 20)

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

mkdir -p "$RESULTS_DIR" "$GRAPHS_DIR"

# GT path on the uni machine — override with `GT=/some/path ./run_gist_mc_sweep.sh`
# if you've put the construction GT somewhere else.
GT="${GT:-$SCRIPT_DIR/data/gist1m_l2_gt.bin}"

echo "=== GIST 1M mc sweep (nofilter, init=$INIT) ==="
echo "repo=$SCRIPT_DIR"
echo "data=$DATA"
echo "gt=$GT"
echo "k=$K mcs=${MCS[*]}"

if [[ ! -f "$GT" ]]; then
  echo "[error] Ground-truth file not found: $GT"
  echo "        Set GT=<path> env var to point at the construction GT file."
  exit 1
fi

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
echo "=== GIST 1M mc sweep complete ==="
