#include "initializer.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <iostream>

// ============================================================================
//  Helper: fill empty neighbor slots with random points
// ============================================================================

static void random_fill(KNNGraph& graph, int i, int k,
                        const std::vector<std::vector<float>>& data,
                        DistFunc dist_fn, std::mt19937& rng, long long& dist_comps) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    if ((int)graph.neighbors[i].size() >= k) return;

    std::unordered_set<int> have;
    for (auto& nb : graph.neighbors[i]) have.insert(nb.index);

    std::uniform_int_distribution<int> unif(0, n - 1);
    int attempts = 0;
    while ((int)graph.neighbors[i].size() < k && attempts < k * 10) {
        int j = unif(rng);
        if (j != i && !have.count(j)) {
            have.insert(j);
            float d = dist_fn(data[i].data(), data[j].data(), dim);
            dist_comps++;
            graph.neighbors[i].emplace_back(j, d, true);
        }
        attempts++;
    }
    std::sort(graph.neighbors[i].begin(), graph.neighbors[i].end());
}

// ============================================================================
//  1. Random Initialization
// ============================================================================

KNNGraph init_random(const std::vector<std::vector<float>>& data,
                     int k, DistFunc dist_fn, long long& dist_comps) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    KNNGraph graph(n, k);
    std::mt19937 rng(42);

    std::uniform_int_distribution<int> uid(0, n - 1);
    for (int i = 0; i < n; i++) {
        std::unordered_set<int> chosen;
        while ((int)chosen.size() < k) {
            int j = uid(rng);
            if (j != i) chosen.insert(j);
        }
        for (int j : chosen) {
            float d = dist_fn(data[i].data(), data[j].data(), dim);
            dist_comps++;
            graph.neighbors[i].emplace_back(j, d, true);
        }
        std::sort(graph.neighbors[i].begin(), graph.neighbors[i].end());
    }
    return graph;
}

// ============================================================================
//  2. E2LSH Initialization (multi-probe, bucket-based)
// ============================================================================

// --- E2LSH internals ---

struct E2LSHState {
    int L, K, dim, n;
    float w;                 // bucket width
    std::vector<float> a;    // projection vectors: [l*K*dim + k*dim + d]
    std::vector<float> b;    // random offsets:     [l*K + k]
    std::vector<int> raw;    // per-component hash: [i*L*K + l*K + k]
};

static E2LSHState g_e2lsh;

static inline int& raw_hash(int i, int l, int k) {
    return g_e2lsh.raw[i * g_e2lsh.L * g_e2lsh.K + l * g_e2lsh.K + k];
}
static inline const float* proj_vec(int l, int k) {
    return &g_e2lsh.a[(l * g_e2lsh.K + k) * g_e2lsh.dim];
}
static inline float proj_offset(int l, int k) {
    return g_e2lsh.b[l * g_e2lsh.K + k];
}

// FNV-1a: combine K integer hash keys into a single compound bucket hash.
static inline int combine_hash_keys(const int* keys, int K) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < K; i++) {
        h ^= (unsigned int)keys[i];
        h *= 16777619u;
    }
    return (int)h;
}

// Auto-calibrate bucket width from sampled pairwise distances (25th percentile).
static float auto_calibrate_w(const std::vector<std::vector<float>>& data,
                               DistFunc dist_fn) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    int sample = std::min(200, n);
    std::mt19937 rng(123);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    int half = sample / 2;
    std::vector<float> dists;
    for (int i = 0; i < half; i++)
        for (int j = half; j < sample; j++)
            dists.push_back(dist_fn(data[idx[i]].data(), data[idx[j]].data(), dim));

    std::sort(dists.begin(), dists.end());
    return std::max(dists[dists.size() / 4], 1e-6f);
}

// Build L tables with K hash functions each. Returns collision table with
// compound codes (for bucket lookup) and per-function codes (for fingerprints).
CollisionTable build_lsh_table(const std::vector<std::vector<float>>& data, int L, int K) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    int total_hashes = L * K;

    CollisionTable table;
    table.n = n;
    table.L = L;
    table.total_hashes = total_hashes;
    table.codes.assign(n, std::vector<int>(L, 0));
    table.filter_codes.assign(n, std::vector<int>(total_hashes, 0));

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    float w = auto_calibrate_w(data, euclidean_distance);

    g_e2lsh = {L, K, dim, n, w, {}, {}, {}};
    g_e2lsh.a.resize(L * K * dim);
    g_e2lsh.b.resize(L * K);
    g_e2lsh.raw.resize(n * L * K);

    std::uniform_real_distribution<float> unif(0.0f, w);
    for (int l = 0; l < L; l++) {
        for (int kk = 0; kk < K; kk++) {
            float* avec = &g_e2lsh.a[(l * K + kk) * dim];
            for (int d = 0; d < dim; d++)
                avec[d] = normal(rng);
            g_e2lsh.b[l * K + kk] = unif(rng);
        }
    }

    std::vector<int> keys(K);
    for (int i = 0; i < n; i++) {
        const float* point = data[i].data();
        for (int l = 0; l < L; l++) {
            for (int kk = 0; kk < K; kk++) {
                const float* avec = proj_vec(l, kk);
                float proj = 0.0f;
                for (int d = 0; d < dim; d++)
                    proj += avec[d] * point[d];
                int h = (int)std::floor((proj + proj_offset(l, kk)) / w);
                raw_hash(i, l, kk) = h;
                table.filter_codes[i][l * K + kk] = h;
                keys[kk] = h;
            }
            table.codes[i][l] = combine_hash_keys(keys.data(), K);
        }
    }

    std::cout << "[lsh] E2LSH built: L=" << L << " K=" << K << " w=" << w << "\n";
    table.build_fingerprints();
    return table;
}

// --- Bucket helpers ---

static std::vector<std::unordered_map<int, std::vector<int>>>
build_buckets(const CollisionTable& table) {
    std::vector<std::unordered_map<int, std::vector<int>>> buckets(table.L);
    for (int l = 0; l < table.L; l++)
        for (int i = 0; i < table.n; i++)
            buckets[l][table.codes[i][l]].push_back(i);
    return buckets;
}

// Rank candidates by collision count, evaluate top ones.
// Used by both multi-probe and simple bucket init.
static void evaluate_candidates(
    KNNGraph& graph, int i,
    const std::vector<std::vector<float>>& data, int dim,
    int k, DistFunc dist_fn, long long& dist_comps,
    std::vector<int>& dirty, std::vector<int>& col_count)
{
    int eval_count = std::min((int)dirty.size(), std::max(k * 3, 30));
    if (eval_count < (int)dirty.size()) {
        std::nth_element(dirty.begin(), dirty.begin() + eval_count, dirty.end(),
            [&](int a, int b) { return col_count[a] > col_count[b]; });
    }
    std::sort(dirty.begin(), dirty.begin() + eval_count,
              [&](int a, int b) { return col_count[a] > col_count[b]; });

    for (int t = 0; t < eval_count; t++) {
        int j = dirty[t];
        float d = dist_fn(data[i].data(), data[j].data(), dim);
        dist_comps++;
        graph.try_update(i, j, d);
    }
}

// Multi-probe LSH init: for each point, probe exact + neighboring buckets.
static KNNGraph init_from_buckets_multiprobe(
    const std::vector<std::vector<float>>& data,
    int k, const CollisionTable& table,
    int num_probes, DistFunc dist_fn, long long& dist_comps)
{
    int n = (int)data.size();
    int dim = (int)data[0].size();
    int K = g_e2lsh.K;
    int L = table.L;

    auto buckets = build_buckets(table);
    KNNGraph graph(n, k);
    std::mt19937 rng(42);

    std::vector<int> col_count(n, 0);
    std::vector<int> dirty;
    dirty.reserve(n / 2);
    std::vector<int> perturbed(K);

    for (int i = 0; i < n; i++) {
        for (int idx : dirty) col_count[idx] = 0;
        dirty.clear();

        for (int l = 0; l < L; l++) {
            // Exact bucket
            auto it = buckets[l].find(table.codes[i][l]);
            if (it != buckets[l].end()) {
                for (int j : it->second) {
                    if (j != i) {
                        if (col_count[j] == 0) dirty.push_back(j);
                        col_count[j]++;
                    }
                }
            }

            // Multi-probe: perturb each hash component by +/-1
            int probes_done = 0;
            const int* base = &g_e2lsh.raw[i * L * K + l * K];
            for (int kk = 0; kk < K && probes_done < num_probes; kk++) {
                for (int delta = -1; delta <= 1; delta += 2) {
                    if (probes_done >= num_probes) break;
                    for (int q = 0; q < K; q++) perturbed[q] = base[q];
                    perturbed[kk] += delta;
                    int probe_code = combine_hash_keys(perturbed.data(), K);
                    auto pit = buckets[l].find(probe_code);
                    if (pit != buckets[l].end()) {
                        for (int j : pit->second) {
                            if (j != i) {
                                if (col_count[j] == 0) dirty.push_back(j);
                                col_count[j]++;
                            }
                        }
                    }
                    probes_done++;
                }
            }
        }

        evaluate_candidates(graph, i, data, dim, k, dist_fn, dist_comps, dirty, col_count);
        random_fill(graph, i, k, data, dist_fn, rng, dist_comps);
    }
    return graph;
}

// Simple bucket-based init (no multi-probe). Used by RP-tree collision path.
static KNNGraph init_from_buckets(const std::vector<std::vector<float>>& data,
                                   int k, const CollisionTable& table,
                                   DistFunc dist_fn, long long& dist_comps) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    auto buckets = build_buckets(table);
    KNNGraph graph(n, k);
    std::mt19937 rng(42);

    std::vector<int> col_count(n, 0);
    std::vector<int> dirty;
    dirty.reserve(n / 2);

    for (int i = 0; i < n; i++) {
        for (int idx : dirty) col_count[idx] = 0;
        dirty.clear();

        for (int l = 0; l < table.L; l++) {
            auto it = buckets[l].find(table.codes[i][l]);
            if (it != buckets[l].end()) {
                for (int j : it->second) {
                    if (j != i) {
                        if (col_count[j] == 0) dirty.push_back(j);
                        col_count[j]++;
                    }
                }
            }
        }

        evaluate_candidates(graph, i, data, dim, k, dist_fn, dist_comps, dirty, col_count);
        random_fill(graph, i, k, data, dist_fn, rng, dist_comps);
    }
    return graph;
}

KNNGraph init_lsh(const std::vector<std::vector<float>>& data,
                  int k, int L, int K, int num_probes, DistFunc dist_fn,
                  CollisionTable& table_out, long long& dist_comps) {
    table_out = build_lsh_table(data, L, K);
    return init_from_buckets_multiprobe(data, k, table_out, num_probes, dist_fn, dist_comps);
}

// ============================================================================
//  3. RP-Tree Initialization (PyNNDescent-style)
//
//  Split: pick two random DATA POINTS, use the perpendicular bisector hyperplane.
//  Leaf size: max(60, min(256, 5*k)) — large leaves for dense local neighborhoods.
//  Init: all pairwise distances within each leaf (no collision counting).
// ============================================================================

// Recursively split indices by two-point hyperplane until leaves are small enough.
static void rp_tree_split(
    const std::vector<std::vector<float>>& data, int dim,
    std::vector<int>& indices, int leaf_size, int max_depth, int depth,
    std::mt19937& rng,
    std::vector<std::vector<int>>& all_leaves)
{
    if ((int)indices.size() <= leaf_size || depth >= max_depth) {
        all_leaves.push_back(indices);
        return;
    }

    // Pick two distinct random pivot points
    std::uniform_int_distribution<int> uid(0, (int)indices.size() - 1);
    int ai = uid(rng), bi = uid(rng);
    while (bi == ai) bi = uid(rng);
    int pa = indices[ai], pb = indices[bi];

    // Hyperplane: normal = (b - a), split at midpoint projection
    float midpoint_proj = 0.0f;
    std::vector<float> normal(dim);
    for (int d = 0; d < dim; d++) {
        normal[d] = data[pb][d] - data[pa][d];
        midpoint_proj += normal[d] * (data[pa][d] + data[pb][d]) * 0.5f;
    }

    // Partition points by which side of the hyperplane they fall on
    std::vector<int> left, right;
    left.reserve(indices.size() / 2);
    right.reserve(indices.size() / 2);
    for (int idx : indices) {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++)
            proj += normal[d] * data[idx][d];
        if (proj <= midpoint_proj)
            left.push_back(idx);
        else
            right.push_back(idx);
    }

    // Degenerate split: all points on one side → make this a leaf
    if (left.empty() || right.empty()) {
        all_leaves.push_back(indices);
        return;
    }

    rp_tree_split(data, dim, left, leaf_size, max_depth, depth + 1, rng, all_leaves);
    rp_tree_split(data, dim, right, leaf_size, max_depth, depth + 1, rng, all_leaves);
}

// Build L RP-trees, compute all pairwise distances within each leaf.
KNNGraph init_rp_tree(const std::vector<std::vector<float>>& data,
                      int k, int L, DistFunc dist_fn,
                      CollisionTable& table_out, long long& dist_comps) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    KNNGraph graph(n, k);
    std::mt19937 rng(42);

    int leaf_size = std::max(60, std::min(256, 5 * k));
    int max_depth = 200;

    std::cout << "[rp-tree] PyNNDescent-style: L=" << L
              << " leaf_size=" << leaf_size << "\n";

    long long leaf_pairs = 0;
    for (int l = 0; l < L; l++) {
        std::vector<std::vector<int>> leaves;
        std::vector<int> all_indices(n);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        rp_tree_split(data, dim, all_indices, leaf_size, max_depth, 0, rng, leaves);

        for (auto& leaf : leaves) {
            int sz = (int)leaf.size();
            for (int i = 0; i < sz; i++) {
                for (int j = i + 1; j < sz; j++) {
                    float d = dist_fn(data[leaf[i]].data(), data[leaf[j]].data(), dim);
                    dist_comps++;
                    graph.try_update(leaf[i], leaf[j], d);
                    graph.try_update(leaf[j], leaf[i], d);
                }
            }
            leaf_pairs += (long long)sz * (sz - 1) / 2;
        }
    }

    for (int i = 0; i < n; i++)
        random_fill(graph, i, k, data, dist_fn, rng, dist_comps);

    std::cout << "[rp-tree] Done. trees=" << L << " leaf_pairs=" << leaf_pairs
              << " dist_comps=" << dist_comps << "\n";

    // Build collision table fingerprints (needed if collision filter is used later)
    table_out = build_rp_tree_table(data, L);
    return graph;
}

// ============================================================================
//  4. Legacy RP-tree table builder (for collision filter fingerprints only)
//
//  Uses random Gaussian direction + median split — different from the
//  PyNNDescent two-point split above. Only used to generate fingerprints
//  when collision filtering is enabled alongside RP-tree init.
// ============================================================================

static void rp_tree_split_legacy(const std::vector<std::vector<float>>& data,
                                  const std::vector<float>& proj_dir, int dim,
                                  std::vector<int>& indices, std::vector<int>& codes,
                                  int current_code, int depth, int max_depth,
                                  std::mt19937& rng) {
    if ((int)indices.size() <= 8 || depth >= max_depth) {
        for (int idx : indices) codes[idx] = current_code;
        return;
    }

    std::vector<std::pair<float, int>> projections;
    projections.reserve(indices.size());
    for (int idx : indices) {
        float p = 0.0f;
        for (int d = 0; d < dim; d++) p += proj_dir[d] * data[idx][d];
        projections.push_back({p, idx});
    }
    std::sort(projections.begin(), projections.end());

    int mid = (int)projections.size() / 2;
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::vector<float> child_dir(dim);
    for (int d = 0; d < dim; d++) child_dir[d] = normal_dist(rng);

    std::vector<int> left_idx, right_idx;
    for (int i = 0; i < mid; i++) left_idx.push_back(projections[i].second);
    for (int i = mid; i < (int)projections.size(); i++) right_idx.push_back(projections[i].second);

    rp_tree_split_legacy(data, child_dir, dim, left_idx, codes, current_code * 2, depth + 1, max_depth, rng);
    rp_tree_split_legacy(data, child_dir, dim, right_idx, codes, current_code * 2 + 1, depth + 1, max_depth, rng);
}

CollisionTable build_rp_tree_table(const std::vector<std::vector<float>>& data, int L) {
    int n = (int)data.size();
    int dim = (int)data[0].size();

    CollisionTable table;
    table.n = n;
    table.L = L;
    table.total_hashes = L;
    table.codes.assign(n, std::vector<int>(L, 0));
    table.filter_codes.assign(n, std::vector<int>(L, 0));

    std::mt19937 rng(42);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    int max_depth = std::max(4, (int)std::ceil(std::log2(n / 8.0)));

    for (int l = 0; l < L; l++) {
        std::vector<float> dir(dim);
        for (int d = 0; d < dim; d++) dir[d] = normal_dist(rng);
        std::vector<int> all_indices(n);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::vector<int> codes(n, 0);
        rp_tree_split_legacy(data, dir, dim, all_indices, codes, 1, 0, max_depth, rng);
        for (int i = 0; i < n; i++) {
            table.codes[i][l] = codes[i];
            table.filter_codes[i][l] = codes[i];
        }
    }

    table.build_fingerprints();
    return table;
}
