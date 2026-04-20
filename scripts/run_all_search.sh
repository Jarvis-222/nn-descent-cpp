#!/bin/bash
# ============================================================
# Run graph search on ALL built graphs for SIFT 1M and GIST 1M.
# Assumes graphs already exist in $BASE/graphs/.
# Run this on the university cluster where the .knng files live.
# ============================================================
set -euo pipefail

BASE="/home/j2pandya/nn-descent/nn-descent-cpp"
BIN="$BASE/build/nn_descent"
GRAPHS="$BASE/graphs"
SEARCH_OUT="$BASE/results/search"

mkdir -p "$SEARCH_OUT"

EF="10,20,50,100,200,400"

# --- SIFT 1M ---
SIFT_DATA="$BASE/data/sift/sift_base.fvecs"
SIFT_QUERY="$BASE/data/sift/sift_query.fvecs"
SIFT_SGT="$BASE/data/sift/sift_groundtruth.ivecs"
SIFT_BGT="$BASE/data/sift1m_l2_gt.bin"

# --- GIST 1M ---
GIST_DATA="$BASE/data/gist/gist_base.fvecs"
GIST_QUERY="$BASE/data/gist/gist_query.fvecs"
GIST_SGT="$BASE/data/gist/gist_groundtruth.ivecs"
GIST_BGT="$BASE/data/gist1m_l2_gt.bin"

run_search() {
    local data="$1"
    local query="$2"
    local search_gt="$3"
    local build_gt="$4"
    local graph="$5"
    local output="$6"

    if [ -f "$output" ]; then
        echo "[SKIP] $output already exists"
        return
    fi

    if [ ! -f "$graph" ]; then
        echo "[MISS] Graph not found: $graph"
        return
    fi

    echo "[RUN]  $output"
    "$BIN" \
        --data "$data" \
        --load-graph "$graph" \
        --load-gt "$build_gt" \
        --query "$query" \
        --search-gt "$search_gt" \
        --ef "$EF" \
        --entry-points 10 \
        --output "$output" \
        --no-gt
}

echo "========================================"
echo "  SIFT 1M — filter sweep (random init)"
echo "========================================"
for PT in nofilter pt060 pt065 pt070 pt075 pt080 pt085 pt090 pt095 pt099; do
    TAG="sift1m_random_${PT}"
    run_search "$SIFT_DATA" "$SIFT_QUERY" "$SIFT_SGT" "$SIFT_BGT" \
        "$GRAPHS/${TAG}.knng" "$SEARCH_OUT/${TAG}_search.txt"
done

echo "========================================"
echo "  SIFT 1M — filter sweep (rptree init)"
echo "========================================"
for PT in nofilter pt060 pt065 pt070 pt075 pt080 pt085 pt090 pt095 pt099; do
    TAG="sift1m_rptree_${PT}"
    run_search "$SIFT_DATA" "$SIFT_QUERY" "$SIFT_SGT" "$SIFT_BGT" \
        "$GRAPHS/${TAG}.knng" "$SEARCH_OUT/${TAG}_search.txt"
done

echo "========================================"
echo "  SIFT 1M — mc sweep"
echo "========================================"
for MC in 10 15 20 25 30 35; do
    TAG="sift1m_mc${MC}_nofilter_r1"
    run_search "$SIFT_DATA" "$SIFT_QUERY" "$SIFT_SGT" "$SIFT_BGT" \
        "$GRAPHS/${TAG}.knng" "$SEARCH_OUT/${TAG}_search.txt"
done

echo "========================================"
echo "  GIST 1M — filter sweep (random init)"
echo "========================================"
for PT in nofilter pt060 pt065 pt070 pt075 pt080 pt085 pt090 pt095 pt099; do
    TAG="gist1m_random_${PT}"
    run_search "$GIST_DATA" "$GIST_QUERY" "$GIST_SGT" "$GIST_BGT" \
        "$GRAPHS/${TAG}.knng" "$SEARCH_OUT/${TAG}_search.txt"
done

echo "========================================"
echo "  GIST 1M — filter sweep (rptree init)"
echo "========================================"
for PT in nofilter pt060 pt065 pt070 pt075 pt080 pt085 pt090 pt095 pt099; do
    TAG="gist1m_rptree_${PT}"
    run_search "$GIST_DATA" "$GIST_QUERY" "$GIST_SGT" "$GIST_BGT" \
        "$GRAPHS/${TAG}.knng" "$SEARCH_OUT/${TAG}_search.txt"
done

echo "========================================"
echo "  GIST 1M — mc sweep"
echo "========================================"
for MC in 10 15 20 25 30 35; do
    TAG="gist1m_mc${MC}_nofilter_r1"
    run_search "$GIST_DATA" "$GIST_QUERY" "$GIST_SGT" "$GIST_BGT" \
        "$GRAPHS/${TAG}.knng" "$SEARCH_OUT/${TAG}_search.txt"
done

echo ""
echo "========== ALL DONE =========="
