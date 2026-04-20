#!/bin/bash
# ============================================================
# Build ALL graph configs and save .knng files.
# Run this first if graphs are missing, then run_all_search.sh.
# ============================================================
set -euo pipefail

BASE="/home/j2pandya/nn-descent/nn-descent-cpp"
BIN="$BASE/build/nn_descent"
GRAPHS="$BASE/graphs"
RESULTS="$BASE/results"

mkdir -p "$GRAPHS" "$RESULTS"

SIFT_DATA="$BASE/data/sift/sift_base.fvecs"
SIFT_GT="$BASE/data/sift1m_l2_gt.bin"
GIST_DATA="$BASE/data/gist/gist_base.fvecs"
GIST_GT="$BASE/data/gist1m_l2_gt.bin"

build_graph() {
    local data="$1"
    local gt="$2"
    local graph="$3"
    local report="$4"
    shift 4
    # remaining args are extra flags

    if [ -f "$graph" ]; then
        echo "[SKIP] $graph already exists"
        return
    fi

    echo "[BUILD] $graph"
    "$BIN" \
        --data "$data" \
        --load-gt "$gt" \
        --k 10 \
        --save-graph "$graph" \
        --output "$report" \
        "$@"
}

# ===================== SIFT 1M =====================

echo "========================================"
echo "  SIFT 1M — random init, filter sweep"
echo "========================================"
build_graph "$SIFT_DATA" "$SIFT_GT" \
    "$GRAPHS/sift1m_random_nofilter.knng" \
    "$RESULTS/sift1m_random_nofilter_build.txt" \
    --init random

for PT in 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99; do
    PTAG=$(echo $PT | tr -d '.')
    build_graph "$SIFT_DATA" "$SIFT_GT" \
        "$GRAPHS/sift1m_random_pt${PTAG}.knng" \
        "$RESULTS/sift1m_random_pt${PTAG}_build.txt" \
        --init random --proj-filter --filter-confidence "$PT"
done

echo "========================================"
echo "  SIFT 1M — rptree init, filter sweep"
echo "========================================"
build_graph "$SIFT_DATA" "$SIFT_GT" \
    "$GRAPHS/sift1m_rptree_nofilter.knng" \
    "$RESULTS/sift1m_rptree_nofilter_build.txt" \
    --init rptree

for PT in 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99; do
    PTAG=$(echo $PT | tr -d '.')
    build_graph "$SIFT_DATA" "$SIFT_GT" \
        "$GRAPHS/sift1m_rptree_pt${PTAG}.knng" \
        "$RESULTS/sift1m_rptree_pt${PTAG}_build.txt" \
        --init rptree --proj-filter --filter-confidence "$PT"
done

echo "========================================"
echo "  SIFT 1M — mc sweep (no filter)"
echo "========================================"
for MC in 10 15 20 25 30 35; do
    build_graph "$SIFT_DATA" "$SIFT_GT" \
        "$GRAPHS/sift1m_mc${MC}_nofilter_r1.knng" \
        "$RESULTS/sift1m_mc${MC}_nofilter_r1.txt" \
        --init random --max-candidates "$MC"
done

# ===================== GIST 1M =====================

echo "========================================"
echo "  GIST 1M — random init, filter sweep"
echo "========================================"
build_graph "$GIST_DATA" "$GIST_GT" \
    "$GRAPHS/gist1m_random_nofilter.knng" \
    "$RESULTS/gist1m_random_nofilter_build.txt" \
    --init random

for PT in 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99; do
    PTAG=$(echo $PT | tr -d '.')
    build_graph "$GIST_DATA" "$GIST_GT" \
        "$GRAPHS/gist1m_random_pt${PTAG}.knng" \
        "$RESULTS/gist1m_random_pt${PTAG}_build.txt" \
        --init random --proj-filter --filter-confidence "$PT"
done

echo "========================================"
echo "  GIST 1M — rptree init, filter sweep"
echo "========================================"
build_graph "$GIST_DATA" "$GIST_GT" \
    "$GRAPHS/gist1m_rptree_nofilter.knng" \
    "$RESULTS/gist1m_rptree_nofilter_build.txt" \
    --init rptree

for PT in 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99; do
    PTAG=$(echo $PT | tr -d '.')
    build_graph "$GIST_DATA" "$GIST_GT" \
        "$GRAPHS/gist1m_rptree_pt${PTAG}.knng" \
        "$RESULTS/gist1m_rptree_pt${PTAG}_build.txt" \
        --init rptree --proj-filter --filter-confidence "$PT"
done

echo "========================================"
echo "  GIST 1M — mc sweep (no filter)"
echo "========================================"
for MC in 10 15 20 25 30 35; do
    build_graph "$GIST_DATA" "$GIST_GT" \
        "$GRAPHS/gist1m_mc${MC}_nofilter_r1.knng" \
        "$RESULTS/gist1m_mc${MC}_nofilter_r1.txt" \
        --init random --max-candidates "$MC"
done

echo ""
echo "========== ALL BUILDS DONE =========="
