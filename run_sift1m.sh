#!/bin/bash
BIN=./build/nn_descent
DATA=data/sift/sift_base.fvecs
GT=data/sift1M_l2_gt.bin
K=10
MC=40

mkdir -p results

echo "=== SIFT 1M Benchmark ==="

# No filter — 5 runs
for i in 1 2 3 4 5; do
    echo "--- No filter, run $i ---"
    $BIN --data $DATA --k $K --mc $MC --init random --load-gt $GT --output results/sift1m_nofilter_r${i}.txt
done

# Projection filter — pτ = 0.99, 0.95, 0.90, 0.80
for PT in 0.99 0.95 0.90 0.80; do
    for i in 1 2 3 4 5; do
        echo "--- proj-filter pτ=$PT, run $i ---"
        $BIN --data $DATA --k $K --mc $MC --init random --proj-filter --num-projections 32 --filter-confidence $PT --load-gt $GT --output results/sift1m_pt${PT}_r${i}.txt
    done
done

echo "=== All done ==="

