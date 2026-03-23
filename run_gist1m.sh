#!/bin/bash
BIN=./build/nn_descent
DATA=data/gist/gist_base.fvecs
GT=data/gist1m_l2_gt.bin

echo "=== GIST 1M Sequential Benchmark ==="

echo "--- mc=40 pτ=0.99 ---"
$BIN --data $DATA --k 10 --mc 40 --init random --proj-filter --num-projections 32 --filter-confidence 0.99 --load-gt $GT --output results/gist1m_pt0.99_r1.txt

echo "--- mc=40 pτ=0.95 ---"
$BIN --data $DATA --k 10 --mc 40 --init random --proj-filter --num-projections 32 --filter-confidence 0.95 --load-gt $GT --output results/gist1m_pt0.95_r1.txt

echo "--- mc=35 no filter ---"
$BIN --data $DATA --k 10 --mc 35 --init random --load-gt $GT --output results/gist1m_mc35_nofilter_r1.txt

echo "--- mc=25 no filter ---"
$BIN --data $DATA --k 10 --mc 25 --init random --load-gt $GT --output results/gist1m_mc25_nofilter_r1.txt

echo "=== All done! ==="
