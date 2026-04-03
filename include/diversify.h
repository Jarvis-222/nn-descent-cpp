#pragma once

#include "types.h"
#include "distance.h"

// RNG (Relative Neighborhood Graph) diversification — PyNNDescent post-processing.
//
// For each node u with sorted neighbors [v0, v1, ...]:
//   Keep vi only if no previously kept neighbor vj (j < i) satisfies
//   dist(vj, vi) < alpha * dist(u, vi).
//
// This removes "redundant" edges where a closer neighbor already covers
// that region of the space, producing a sparser but more navigable graph.
//
// alpha = 1.0: strict RNG pruning (aggressive)
// alpha > 1.0: relaxed pruning (keeps more edges, typical: 1.0–1.2)
//
// Applied after NN-Descent convergence. Modifies the graph in-place.

struct DiversifyStats {
    int total_before;     // total edges before pruning
    int total_after;      // total edges after pruning
    long long dist_comps; // distance computations during pruning
    double time_sec;
};

DiversifyStats diversify_graph(
    KNNGraph& graph,
    const std::vector<std::vector<float>>& data,
    float alpha,
    DistFunc dist_fn);
