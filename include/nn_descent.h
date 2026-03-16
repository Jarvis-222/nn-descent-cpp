#pragma once

#include <vector>
#include "types.h"
#include "distance.h"
#include "initializer.h"

struct NNDescentResult {
    KNNGraph graph;
    std::vector<IterationStats> iter_log;
    long long total_dist_comps;
    long long init_dist_comps;
    double init_time_sec;
    double total_time_sec;
};

NNDescentResult run_nn_descent(
    const std::vector<std::vector<float>>& data,
    const NNDescentConfig& config,
    const std::vector<std::vector<int>>& ground_truth = {}
);
