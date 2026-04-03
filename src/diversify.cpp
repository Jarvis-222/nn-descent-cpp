#include "diversify.h"
#include <chrono>
#include <iostream>
#include <unordered_map>

DiversifyStats diversify_graph(
    KNNGraph& graph,
    const std::vector<std::vector<float>>& data,
    float alpha,
    DistFunc dist_fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n = graph.n;
    int dim = (int)data[0].size();
    long long dist_comps = 0;
    int total_before = 0;
    int total_after = 0;

    // Cache neighbor-to-neighbor distances to avoid redundant computation.
    // Key: (min_id, max_id) packed into uint64_t.
    auto pack_key = [](int a, int b) -> uint64_t {
        if (a > b) std::swap(a, b);
        return ((uint64_t)a << 32) | (uint64_t)b;
    };

    for (int u = 0; u < n; u++) {
        auto& nbs = graph.neighbors[u];
        total_before += (int)nbs.size();

        // Neighbors are already sorted by distance (closest first).
        // Greedily keep a neighbor if no already-kept neighbor is closer to it.
        std::vector<Neighbor> kept;
        kept.reserve(nbs.size());

        // Local cache for inter-neighbor distances within this node's list
        std::unordered_map<uint64_t, float> dist_cache;

        for (auto& nb : nbs) {
            bool redundant = false;
            for (auto& w : kept) {
                // Check: is w closer to nb than u is to nb?
                // i.e., dist(w, nb) < alpha * dist(u, nb)
                float threshold = alpha * nb.distance;

                uint64_t key = pack_key(w.index, nb.index);
                float d_wv;
                auto it = dist_cache.find(key);
                if (it != dist_cache.end()) {
                    d_wv = it->second;
                } else {
                    d_wv = dist_fn(data[w.index].data(), data[nb.index].data(), dim);
                    dist_cache[key] = d_wv;
                    dist_comps++;
                }

                if (d_wv < threshold) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) {
                kept.push_back(nb);
            }
        }

        nbs = std::move(kept);
        total_after += (int)nbs.size();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[diversify] alpha=" << alpha
              << " edges: " << total_before << " -> " << total_after
              << " (removed " << (total_before - total_after)
              << ", " << 100.0 * (1.0 - (double)total_after / total_before) << "%)"
              << " dist_comps=" << dist_comps
              << " time=" << elapsed << "s\n";

    return {total_before, total_after, dist_comps, elapsed};
}
