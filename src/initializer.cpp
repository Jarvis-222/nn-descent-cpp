#include "initializer.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <iostream>

// ===== Random Initialization =====

KNNGraph init_random(const std::vector<std::vector<float>>& data,
                     int k, DistFunc dist_fn, long long& dist_comps) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    KNNGraph graph(n, k);
    std::mt19937 rng(42);

    for (int i = 0; i < n; i++) {
        std::vector<int> candidates(n - 1);
        int idx = 0;
        for (int j = 0; j < n; j++)
            if (j != i) candidates[idx++] = j;
        std::shuffle(candidates.begin(), candidates.end(), rng);

        int take = std::min(k, (int)candidates.size());
        for (int t = 0; t < take; t++) {
            int j = candidates[t];
            float d = dist_fn(data[i].data(), data[j].data(), dim);
            dist_comps++;
            graph.neighbors[i].emplace_back(j, d, true);
        }
        std::sort(graph.neighbors[i].begin(), graph.neighbors[i].end());
    }
    return graph;
}

// ===== E2LSH Hash Table Construction =====

// Auto-calibrate the bucket width w from a sample of pairwise distances.
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

// Global E2LSH state: projection vectors and per-component hash values.
// Stored flat for cache-friendly access during multi-probe init.
struct E2LSHState {
    int L, K, dim, n;
    float w;
    std::vector<float> a;   // projection vectors: [l*K*dim + k*dim + d]
    std::vector<float> b;   // random offsets:     [l*K + k]
    std::vector<int> raw;   // per-component hash: [i*L*K + l*K + k]
};

static E2LSHState g_e2lsh;

// FNV-1a combine K integer hash keys into a single compound hash.
static inline int combine_hash_keys(const int* keys, int K) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < K; i++) {
        h ^= (unsigned int)keys[i];
        h *= 16777619u;
    }
    return (int)h;
}

static inline int& raw_hash(int i, int l, int k) {
    return g_e2lsh.raw[i * g_e2lsh.L * g_e2lsh.K + l * g_e2lsh.K + k];
}
static inline const float* proj_vec(int l, int k) {
    return &g_e2lsh.a[(l * g_e2lsh.K + k) * g_e2lsh.dim];
}
static inline float proj_offset(int l, int k) {
    return g_e2lsh.b[l * g_e2lsh.K + k];
}

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

    // Initialize global E2LSH state
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

    // Compute hash codes for all points
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

// ===== Bucket-Based Initialization =====

static std::vector<std::unordered_map<int, std::vector<int>>>
build_buckets(const CollisionTable& table) {
    std::vector<std::unordered_map<int, std::vector<int>>> buckets(table.L);
    for (int l = 0; l < table.L; l++)
        for (int i = 0; i < table.n; i++)
            buckets[l][table.codes[i][l]].push_back(i);
    return buckets;
}

// Fill remaining neighbor slots with random points if LSH didn't find enough.
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

// LSH multi-probe initialization: for each point, probe exact + neighboring buckets,
// rank candidates by collision count, compute distances to top candidates.
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

        // Probe buckets across all tables
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

        // Select top candidates by collision count, compute distances
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

        random_fill(graph, i, k, data, dist_fn, rng, dist_comps);
    }
    return graph;
}

// Simple bucket-based init (no multi-probe, used by RP-tree).
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

// ===== RP-Tree Initialization =====

static void rp_tree_split(const std::vector<std::vector<float>>& data,
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
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> child_dir(dim);
    for (int d = 0; d < dim; d++) child_dir[d] = normal(rng);

    std::vector<int> left_idx, right_idx;
    for (int i = 0; i < mid; i++) left_idx.push_back(projections[i].second);
    for (int i = mid; i < (int)projections.size(); i++) right_idx.push_back(projections[i].second);

    rp_tree_split(data, child_dir, dim, left_idx, codes, current_code * 2, depth + 1, max_depth, rng);
    rp_tree_split(data, child_dir, dim, right_idx, codes, current_code * 2 + 1, depth + 1, max_depth, rng);
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
    std::normal_distribution<float> normal(0.0f, 1.0f);
    int max_depth = std::max(4, (int)std::ceil(std::log2(n / 8.0)));

    for (int l = 0; l < L; l++) {
        std::vector<float> dir(dim);
        for (int d = 0; d < dim; d++) dir[d] = normal(rng);
        std::vector<int> all_indices(n);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::vector<int> codes(n, 0);
        rp_tree_split(data, dir, dim, all_indices, codes, 1, 0, max_depth, rng);
        for (int i = 0; i < n; i++) {
            table.codes[i][l] = codes[i];
            table.filter_codes[i][l] = codes[i];
        }
    }

    table.build_fingerprints();
    return table;
}

KNNGraph init_rp_tree(const std::vector<std::vector<float>>& data,
                      int k, int L, DistFunc dist_fn,
                      CollisionTable& table_out, long long& dist_comps) {
    table_out = build_rp_tree_table(data, L);
    return init_from_buckets(data, k, table_out, dist_fn, dist_comps);
}
