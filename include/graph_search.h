#pragma once

#include <vector>
#include "types.h"
#include "distance.h"

// Result of searching queries against a built k-NN graph.
struct SearchResult {
    std::vector<std::vector<int>> indices;      // [nq][k] neighbor indices per query
    std::vector<std::vector<float>> distances;  // [nq][k] neighbor distances per query
    double total_time_sec;
    double qps;                                  // queries per second
    long long total_dist_comps;
};

// Greedy beam search on a k-NN graph (PyNNDescent-style).
//
// For each query q:
//   1. Use `num_entry_points` random projection trees to seed initial candidates.
//   2. Maintain a candidate pool of size `ef` (beam width).
//   3. Greedily expand the closest unvisited candidate's neighbors.
//   4. Stop when the closest candidate is farther than the ef-th best found.
//   5. Return top `search_k` results.
//
// The `epsilon` parameter controls backtracking tolerance:
//   epsilon = 0.0: pure greedy (fast, may get stuck in local minima)
//   epsilon = 0.1: default PyNNDescent (allows some exploration)
//   epsilon = 0.3+: more thorough search (higher recall, slower)
//
// Parameters:
//   graph: the constructed k-NN graph
//   data: the indexed dataset (n points)
//   queries: query points (nq points)
//   search_k: how many neighbors to return per query
//   ef: beam width / search expansion factor (ef >= search_k)
//   epsilon: backtracking tolerance (0.0 = pure greedy)
//   num_entry_points: number of RP-tree starts / search trees to probe
//   dist_fn: distance function
SearchResult graph_search(
    const KNNGraph& graph,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    int search_k, int ef, float epsilon,
    int num_entry_points,
    DistFunc dist_fn);
