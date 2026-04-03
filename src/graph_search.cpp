#include "graph_search.h"
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>

// Greedy beam search on the k-NN graph.
//
// Algorithm per query (matches PyNNDescent's search approach):
//   1. Initialize with random entry points.
//   2. Maintain two structures:
//      - `candidates`: min-heap of nodes to explore (closest first)
//      - `result`: max-heap of best ef nodes found (farthest on top for easy eviction)
//   3. Pop closest candidate. If it's farther than (1+epsilon) * ef-th best, stop.
//   4. Expand its graph neighbors: compute distance, insert into both heaps if promising.
//   5. Return top search_k from result.

SearchResult graph_search(
    const KNNGraph& graph,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    int search_k, int ef, float epsilon,
    int num_entry_points,
    DistFunc dist_fn)
{
    int n = graph.n;
    int dim = (int)data[0].size();
    int nq = (int)queries.size();

    if (ef < search_k) ef = search_k;

    SearchResult result;
    result.indices.resize(nq);
    result.distances.resize(nq);
    result.total_dist_comps = 0;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> uid(0, n - 1);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < nq; q++) {
        const float* query = queries[q].data();
        long long q_dist_comps = 0;

        // Visited set (bit-packed for cache efficiency)
        std::vector<bool> visited(n, false);

        // Min-heap for candidates (closest first)
        // pair<distance, node_index>
        using MinPair = std::pair<float, int>;
        std::priority_queue<MinPair, std::vector<MinPair>, std::greater<MinPair>> candidates;

        // Max-heap for result (farthest on top for eviction)
        using MaxPair = std::pair<float, int>;
        std::priority_queue<MaxPair> result_heap;

        // Initialize with random entry points
        for (int e = 0; e < num_entry_points; e++) {
            int ep = uid(rng);
            if (visited[ep]) continue;
            visited[ep] = true;
            float d = dist_fn(query, data[ep].data(), dim);
            q_dist_comps++;
            candidates.push({d, ep});
            result_heap.push({d, ep});
        }

        // Greedy expansion
        while (!candidates.empty()) {
            auto [c_dist, c_idx] = candidates.top();
            candidates.pop();

            // Stopping condition: if closest candidate is farther than
            // (1 + epsilon) * current ef-th best, no more improvements possible
            float bound = result_heap.top().first;
            if (c_dist > bound * (1.0f + epsilon) && (int)result_heap.size() >= ef) {
                break;
            }

            // Expand neighbors of this candidate
            for (auto& nb : graph.neighbors[c_idx]) {
                int nb_idx = nb.index;
                if (nb_idx < 0 || visited[nb_idx]) continue;
                visited[nb_idx] = true;

                float d = dist_fn(query, data[nb_idx].data(), dim);
                q_dist_comps++;

                // Insert if result heap not full, or if better than worst in result
                if ((int)result_heap.size() < ef || d < result_heap.top().first) {
                    candidates.push({d, nb_idx});
                    result_heap.push({d, nb_idx});
                    if ((int)result_heap.size() > ef) {
                        result_heap.pop();  // evict farthest
                    }
                }
            }
        }

        // Extract top search_k from result heap
        std::vector<std::pair<float, int>> res_vec;
        res_vec.reserve(result_heap.size());
        while (!result_heap.empty()) {
            res_vec.push_back({result_heap.top().first, result_heap.top().second});
            result_heap.pop();
        }
        // Sort ascending by distance
        std::sort(res_vec.begin(), res_vec.end());

        int out_k = std::min(search_k, (int)res_vec.size());
        result.indices[q].resize(out_k);
        result.distances[q].resize(out_k);
        for (int i = 0; i < out_k; i++) {
            result.distances[q][i] = res_vec[i].first;
            result.indices[q][i] = res_vec[i].second;
        }

        result.total_dist_comps += q_dist_comps;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_time_sec = std::chrono::duration<double>(t_end - t_start).count();
    result.qps = (result.total_time_sec > 0) ? nq / result.total_time_sec : 0.0;

    return result;
}
