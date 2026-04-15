#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[error] Command failed: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/build/nn_descent}"
TAG="gist1m"

DATA="$SCRIPT_DIR/data/gist/gist_base.fvecs"
QUERIES="$SCRIPT_DIR/data/gist/gist_query.fvecs"
SEARCH_GT="$SCRIPT_DIR/data/gist/gist_groundtruth.ivecs"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
RESULTS_DIR="$SCRIPT_DIR/results/search"

K=10
EFS="10,20,50,100,200,400"

if [[ ! -x "$BIN" ]]; then
  echo "[error] Binary not found: $BIN"
  exit 1
fi

for f in "$DATA" "$QUERIES" "$SEARCH_GT"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] File not found: $f"
    exit 1
  fi
done

mkdir -p "$RESULTS_DIR"

echo "=== GIST 1M Search Sweep ==="
echo "data=$DATA"
echo "queries=$QUERIES"
echo "search_gt=$SEARCH_GT"
echo "k=$K  ef=$EFS"
echo

for graph_file in "$GRAPHS_DIR"/${TAG}_*.knng; do
  if [[ ! -f "$graph_file" ]]; then
    echo "[warn] No graphs found in $GRAPHS_DIR"
    exit 1
  fi

  name="$(basename "$graph_file" .knng)"
  report="$RESULTS_DIR/${name}_search.txt"

  echo "============================================================"
  echo "graph=$name"
  echo "report=$report"
  echo "============================================================"

  "$BIN" \
    --data "$DATA" \
    --k "$K" \
    --no-gt \
    --load-graph "$graph_file" \
    --query "$QUERIES" \
    --search-gt "$SEARCH_GT" \
    --search-k "$K" \
    --ef "$EFS" \
	--entry-points 10 \
    2>&1 | tee "$report"
 
  echo
done

echo "=== All searches completed ==="
echo "Results in: $RESULTS_DIR/"
