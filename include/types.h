#pragma once

#include <vector>
#include <algorithm>
#include <limits>
#include <string>

struct Neighbor {
    int index;
    float distance;
    bool is_new;

    Neighbor() : index(-1), distance(std::numeric_limits<float>::max()), is_new(true) {}
    Neighbor(int idx, float dist, bool new_flag = true)
        : index(idx), distance(dist), is_new(new_flag) {}

    bool operator<(const Neighbor& o) const { return distance < o.distance; }
};

struct KNNGraph {
    int n;
    int k;
    std::vector<std::vector<Neighbor>> neighbors;

    KNNGraph() : n(0), k(0) {}
    KNNGraph(int n_, int k_) : n(n_), k(k_), neighbors(n_) {}

    int try_update(int vertex, int candidate_idx, float dist) {
        if (vertex == candidate_idx) return 0;
        auto& nbs = neighbors[vertex];
        for (auto& nb : nbs)
            if (nb.index == candidate_idx) return 0;
        if ((int)nbs.size() < k) {
            nbs.emplace_back(candidate_idx, dist, true);
            std::sort(nbs.begin(), nbs.end());
            return 1;
        }
        if (dist < nbs.back().distance) {
            nbs.back() = Neighbor(candidate_idx, dist, true);
            std::sort(nbs.begin(), nbs.end());
            return 1;
        }
        return 0;
    }

    int farthest_index(int vertex) const {
        const auto& nbs = neighbors[vertex];
        if (nbs.empty()) return -1;
        return nbs.back().index;
    }

    void mark_all_old() {
        for (auto& nbs : neighbors)
            for (auto& nb : nbs)
                nb.is_new = false;
    }

    std::vector<std::vector<int>> get_index_matrix() const {
        std::vector<std::vector<int>> mat(n);
        for (int i = 0; i < n; i++) {
            mat[i].reserve(neighbors[i].size());
            for (auto& nb : neighbors[i])
                mat[i].push_back(nb.index);
        }
        return mat;
    }
};

enum class InitMethod { RANDOM, LSH, RP_TREE };
enum class DistanceMetric { EUCLIDEAN, COSINE, MANHATTAN };

struct NNDescentConfig {
    int n = 0;
    int dim = 0;
    int k = 10;

    InitMethod init_method = InitMethod::RANDOM;

    float rho = 0.5f;
    int max_candidates = 0;     // mc: if > 0, caps candidate lists at mc (overrides rho)
    float delta = 0.001f;
    int max_iterations = 20;

    DistanceMetric metric = DistanceMetric::EUCLIDEAN;

    // LSH parameters (used for both init and filtering)
    int num_tables = 20;        // L: number of hash tables
    int num_hash_functions = 4; // K: hash functions per table (E2LSH)
    int num_probes = 5;         // multi-probe neighboring buckets

    // Collision-based distance filtering
    bool use_collision_filter = false;
    int margin = 0;             // safety margin (higher = less aggressive filtering)

    // Projection-based distance filtering (LSH-APG style)
    bool use_projection_filter = false;
    int num_projections = 16;   // m: number of random projections
    float filter_confidence = 0.95f; // pτ: confidence level for chi-squared threshold

    std::string output_file = "report.txt";
};

struct IterationStats {
    int iteration;
    int updates;
    long long dist_comps;
    long long cumul_dist;
    long long filtered;
    double filter_rate;
    double recall;
};
