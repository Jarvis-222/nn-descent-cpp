#include "nn_descent.h"
#include "recall.h"
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

NNDescentResult run_nn_descent(
    const std::vector<std::vector<float>>& data,
    const NNDescentConfig& config,
    const std::vector<std::vector<int>>& ground_truth)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    int n = (int)data.size();
    int dim = (int)data[0].size();
    int k = config.k;
    DistFunc dist_fn = get_distance_function(config.metric);

    NNDescentResult result;
    result.init_dist_comps = 0;

    // Phase 1: Initialization

    auto t_init_start = std::chrono::high_resolution_clock::now();
    KNNGraph graph;
    CollisionTable collision_table;

    switch (config.init_method) {
        case InitMethod::LSH:
            std::cout << "[init] LSH (L=" << config.num_tables
                      << " K=" << config.num_hash_functions
                      << " probes=" << config.num_probes << ")\n";
            graph = init_lsh(data, k, config.num_tables, config.num_hash_functions,
                             config.num_probes, dist_fn, collision_table,
                             result.init_dist_comps);
            break;
        case InitMethod::RP_TREE:
            std::cout << "[init] RP-Tree (L=" << config.num_tables << ")\n";
            graph = init_rp_tree(data, k, config.num_tables, dist_fn,
                                 collision_table, result.init_dist_comps);
            break;
        default:
            std::cout << "[init] Random\n";
            graph = init_random(data, k, dist_fn, result.init_dist_comps);
            if (config.use_collision_filter) {
                std::cout << "[init] Building collision table (L=" << config.num_tables
                          << " K=" << config.num_hash_functions << ")\n";
                collision_table = build_lsh_table(data, config.num_tables,
                                                 config.num_hash_functions);
            }
            break;
    }

    auto t_init_end = std::chrono::high_resolution_clock::now();
    result.init_time_sec = std::chrono::duration<double>(t_init_end - t_init_start).count();
    std::cout << "[init] Done. dist_comps=" << result.init_dist_comps
              << " time=" << result.init_time_sec << "s\n";

    // Phase 2: NN-Descent iterations

    bool filtering = config.use_collision_filter && collision_table.L > 0;
    long long cumul_dist = result.init_dist_comps;
    std::mt19937 rng(42);
    float convergence_threshold = config.delta * n * k;

    // Precompute reference collision counts once, update incrementally
    // Direct pointer access to fingerprints (avoid CollisionTable indirection)
    std::vector<int> ref_collisions;
    const uint64_t* fp_ptr = nullptr;
    int margin = config.margin;
    if (filtering) {
        ref_collisions.resize(n, 0);
        fp_ptr = collision_table.fingerprints.data();
        for (int v = 0; v < n; v++) {
            int far = graph.farthest_index(v);
            if (far >= 0) {
                uint64_t s0 = fp_ptr[v * 2] & fp_ptr[far * 2];
                uint64_t s1 = fp_ptr[v * 2 + 1] & fp_ptr[far * 2 + 1];
                ref_collisions[v] = __builtin_popcountll(s0) + __builtin_popcountll(s1);
            }
        }
    }

    for (int iter = 0; iter < config.max_iterations; iter++) {
        auto t_iter_start = std::chrono::high_resolution_clock::now();

        // Build reverse neighbor lists
        std::vector<std::vector<int>> reverse(n);
        for (int v = 0; v < n; v++)
            for (auto& nb : graph.neighbors[v])
                reverse[nb.index].push_back(v);

        // Partition neighbors into new/old candidate lists (forward + reverse)
        std::vector<std::vector<int>> new_lists(n), old_lists(n);
        for (int v = 0; v < n; v++) {
            for (auto& nb : graph.neighbors[v]) {
                if (nb.is_new) new_lists[v].push_back(nb.index);
                else           old_lists[v].push_back(nb.index);
            }
            for (int u : reverse[v]) {
                bool is_new = false;
                for (auto& nb : graph.neighbors[u])
                    if (nb.index == v && nb.is_new) { is_new = true; break; }
                if (is_new) new_lists[v].push_back(u);
                else        old_lists[v].push_back(u);
            }
        }

        // Rho-sampling: limit candidate list sizes
        int max_sample = std::max(1, (int)(config.rho * k));
        for (int v = 0; v < n; v++) {
            if ((int)new_lists[v].size() > max_sample) {
                std::shuffle(new_lists[v].begin(), new_lists[v].end(), rng);
                new_lists[v].resize(max_sample);
            }
            if ((int)old_lists[v].size() > max_sample) {
                std::shuffle(old_lists[v].begin(), old_lists[v].end(), rng);
                old_lists[v].resize(max_sample);
            }
        }

        graph.mark_all_old();

        // Local join: compare candidate pairs (no dedup — cheaper than hash table)
        int updates = 0;
        long long iter_dist = 0, iter_filtered = 0;

        auto t_join_start = std::chrono::high_resolution_clock::now();

        // Collect all candidate pairs per vertex, then process with prefetching
        std::vector<std::pair<int,int>> pairs;
        pairs.reserve(n * k * 2);

        for (int v = 0; v < n; v++) {
            auto& nlist = new_lists[v];
            auto& olist = old_lists[v];

            // new x new
            for (int i = 0; i < (int)nlist.size(); i++)
                for (int j = i + 1; j < (int)nlist.size(); j++)
                    pairs.push_back({nlist[i], nlist[j]});

            // new x old
            for (int i = 0; i < (int)nlist.size(); i++)
                for (int j = 0; j < (int)olist.size(); j++)
                    if (nlist[i] != olist[j])
                        pairs.push_back({nlist[i], olist[j]});
        }

        int np = (int)pairs.size();
        int prefetch_ahead = 8;

        for (int p = 0; p < np; p++) {
            // Prefetch fingerprints for a future pair
            if (filtering && p + prefetch_ahead < np) {
                int pu1 = pairs[p + prefetch_ahead].first;
                int pu2 = pairs[p + prefetch_ahead].second;
                __builtin_prefetch(&fp_ptr[pu1 * 2], 0, 0);
                __builtin_prefetch(&fp_ptr[pu2 * 2], 0, 0);
            }

            int u1 = pairs[p].first, u2 = pairs[p].second;

            // Filter check — cheap O(1) popcount, skip expensive distance
            if (filtering) {
                int ref1 = ref_collisions[u1];
                int ref2 = ref_collisions[u2];
                uint64_t s0 = fp_ptr[u1 * 2] & fp_ptr[u2 * 2];
                uint64_t s1 = fp_ptr[u1 * 2 + 1] & fp_ptr[u2 * 2 + 1];
                int col = __builtin_popcountll(s0) + __builtin_popcountll(s1);
                if ((col < ref1 - margin) && (col < ref2 - margin)) {
                    iter_filtered++;
                    continue;
                }
            }

            float d = dist_fn(data[u1].data(), data[u2].data(), dim);
            iter_dist++;
            if (graph.try_update(u1, u2, d)) {
                updates++;
                if (filtering) {
                    int far = graph.farthest_index(u1);
                    if (far >= 0) {
                        uint64_t s0 = fp_ptr[u1 * 2] & fp_ptr[far * 2];
                        uint64_t s1 = fp_ptr[u1 * 2 + 1] & fp_ptr[far * 2 + 1];
                        ref_collisions[u1] = __builtin_popcountll(s0) + __builtin_popcountll(s1);
                    } else { ref_collisions[u1] = 0; }
                }
            }
            if (graph.try_update(u2, u1, d)) {
                updates++;
                if (filtering) {
                    int far = graph.farthest_index(u2);
                    if (far >= 0) {
                        uint64_t s0 = fp_ptr[u2 * 2] & fp_ptr[far * 2];
                        uint64_t s1 = fp_ptr[u2 * 2 + 1] & fp_ptr[far * 2 + 1];
                        ref_collisions[u2] = __builtin_popcountll(s0) + __builtin_popcountll(s1);
                    } else { ref_collisions[u2] = 0; }
                }
            }
        }

        auto t_join_end = std::chrono::high_resolution_clock::now();
        double join_sec = std::chrono::duration<double>(t_join_end - t_join_start).count();

        cumul_dist += iter_dist;

        // Compute recall if ground truth is available
        double recall_val = 0.0;
        if (!ground_truth.empty()) {
            auto predicted = graph.get_index_matrix();
            recall_val = compute_recall(predicted, ground_truth);
        }

        double filter_rate = (iter_filtered + iter_dist > 0)
            ? (double)iter_filtered / (iter_filtered + iter_dist) : 0.0;

        auto t_iter_end = std::chrono::high_resolution_clock::now();
        double iter_sec = std::chrono::duration<double>(t_iter_end - t_iter_start).count();

        result.iter_log.push_back({iter + 1, updates, iter_dist, cumul_dist,
                                   iter_filtered, filter_rate, recall_val});

        std::cout << "[iter " << (iter + 1) << "] updates=" << updates
                  << " dist=" << iter_dist << " filtered=" << iter_filtered
                  << " rate=" << (filter_rate * 100.0) << "%"
                  << " cumul=" << cumul_dist;
        if (!ground_truth.empty()) std::cout << " recall=" << recall_val;
        std::cout << " join=" << join_sec << "s"
                  << " time=" << iter_sec << "s\n";

        if (updates < convergence_threshold) {
            std::cout << "[converged] updates=" << updates
                      << " < threshold=" << convergence_threshold << "\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_time_sec = std::chrono::duration<double>(t_end - t_start).count();
    result.total_dist_comps = cumul_dist;
    result.graph = std::move(graph);
    return result;
}
