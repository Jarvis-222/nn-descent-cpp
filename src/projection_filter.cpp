#include "projection_filter.h"
#include "distance.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

// Precomputed chi-squared inverse CDF (lower tail) values.
// chi2_lower[m][p] = χ²_p(m)
// We need the LOWER quantile: if proj_dist² < χ²_p(m) * dk², then
// the true distance could be < dk, so we should NOT filter.
// Equivalently, filter if proj_dist² > χ²_p(m) * dk².
//
// But LSH-APG uses: keep if proj_dist < t * dk, where t = √(χ²_pτ(m)).
// pτ is the upper quantile — "with probability pτ, a true neighbor's
// projected distance is below this threshold."
//
// So we need the UPPER quantile of χ²(m): χ²_pτ(m).
// For pτ = 0.95 and m = 16: χ²_0.95(16) = 26.296
// For pτ = 0.90 and m = 16: χ²_0.90(16) = 23.542

// Approximate inverse chi-squared CDF using Wilson-Hilferty transformation:
//   χ²_p(m) ≈ m * (1 - 2/(9m) + z_p * √(2/(9m)))³
// where z_p is the standard normal quantile.
static float chi2_inv(float p, int m) {
    // Standard normal quantile approximation (Abramowitz & Stegun 26.2.23)
    // For p in (0.5, 1):
    float t;
    if (p >= 0.5f) {
        float y = -2.0f * std::log(1.0f - p);
        t = std::sqrt(y) - (2.515517f + 0.802853f * std::sqrt(y) + 0.010328f * y)
            / (1.0f + 1.432788f * std::sqrt(y) + 0.189269f * y + 0.001308f * y * std::sqrt(y));
    } else {
        float y = -2.0f * std::log(p);
        t = -(std::sqrt(y) - (2.515517f + 0.802853f * std::sqrt(y) + 0.010328f * y)
            / (1.0f + 1.432788f * std::sqrt(y) + 0.189269f * y + 0.001308f * y * std::sqrt(y)));
    }

    // Wilson-Hilferty transformation
    float mf = (float)m;
    float a = 1.0f - 2.0f / (9.0f * mf);
    float b = std::sqrt(2.0f / (9.0f * mf));
    float cube = a + t * b;
    return mf * cube * cube * cube;
}

void ProjectionFilter::build(const std::vector<std::vector<float>>& data,
                              int m, float p_tau) {
    n_ = (int)data.size();
    m_ = m;
    int dim = (int)data[0].size();

    // Compute chi-squared threshold
    float chi2_val = chi2_inv(p_tau, m);
    t_sq_ = chi2_val;
    t_ = std::sqrt(chi2_val);

    std::cout << "[proj-filter] Building: m=" << m << " pτ=" << p_tau
              << " χ²=" << chi2_val << " t=" << t_ << "\n";

    // Generate m random Gaussian projection vectors
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<std::vector<float>> proj_vectors(m, std::vector<float>(dim));
    for (int j = 0; j < m; j++)
        for (int d = 0; d < dim; d++)
            proj_vectors[j][d] = normal(rng);

    // Project all data points: projections[i * m + j] = proj_vectors[j] · data[i]
    projections_.resize((size_t)n_ * m_);

    for (int i = 0; i < n_; i++) {
        const float* xi = data[i].data();
        for (int j = 0; j < m; j++) {
            const float* aj = proj_vectors[j].data();
            float dot = 0.0f;
            for (int d = 0; d < dim; d++)
                dot += aj[d] * xi[d];
            projections_[i * m_ + j] = dot;
        }
    }

    std::cout << "[proj-filter] Done. Storage: "
              << (n_ * m_ * 4) / (1024 * 1024) << " MB\n";
}

float ProjectionFilter::projected_dist_sq(int u, int v) const {
    const float* pu = &projections_[u * m_];
    const float* pv = &projections_[v * m_];
    float sum = 0.0f;
    for (int j = 0; j < m_; j++) {
        float d = pu[j] - pv[j];
        sum += d * d;
    }
    return sum;
}

bool ProjectionFilter::should_filter(int u1, int u2, float dk1, float dk2) const {
    float proj_dsq = projected_dist_sq(u1, u2);

    // Filter if projected distance suggests pair is far from BOTH perspectives.
    // proj_dist² > t² * dk² means: with confidence pτ, true dist > dk.
    // AND condition: both must agree (conservative — same as collision filter).
    return (proj_dsq > t_sq_ * dk1 * dk1) && (proj_dsq > t_sq_ * dk2 * dk2);
}

KNNGraph ProjectionFilter::init_knn(const std::vector<std::vector<float>>& data,
                                     int k, DistFunc dist_fn,
                                     long long& dist_comps) const {
    int n = n_;
    int dim = (int)data[0].size();
    int m = m_;

    // Strategy: sort all points along each projection axis, then for each
    // point collect its nearest neighbors across all axes. This avoids O(n²).
    //
    // For each of the m projection axes, sort points by projected value.
    // For each point, its neighbors in sorted order along each axis are
    // likely close in projected space. Collect the W nearest along each
    // axis → m*W candidates per point, then verify with true distance.

    int W = k; // window per axis = K neighbors each side
    int total_cands_per_point = m * W;

    std::cout << "[proj-init] Axis-sweep init: m=" << m << " W=" << W
              << " → ~" << total_cands_per_point << " candidates/point\n";

    // Build sorted index per axis
    std::vector<std::vector<int>> sorted_idx(m);
    std::vector<std::vector<int>> rank(m); // rank[axis][point] = position in sorted order

    for (int j = 0; j < m; j++) {
        sorted_idx[j].resize(n);
        rank[j].resize(n);
        for (int i = 0; i < n; i++) sorted_idx[j][i] = i;

        // Sort by projected value on axis j
        std::sort(sorted_idx[j].begin(), sorted_idx[j].end(),
            [&](int a, int b) {
                return projections_[a * m + j] < projections_[b * m + j];
            });

        // Build rank lookup
        for (int r = 0; r < n; r++)
            rank[j][sorted_idx[j][r]] = r;
    }

    KNNGraph graph(n, k);
    dist_comps = 0;

    // Timestamp-based dedup: avoids allocating per-point seen arrays
    std::vector<int> seen_stamp(n, -1);
    std::vector<int> candidates;
    candidates.reserve(total_cands_per_point);

    for (int i = 0; i < n; i++) {
        candidates.clear();

        for (int j = 0; j < m; j++) {
            int r = rank[j][i];
            int half = W / 2;
            int lo = std::max(0, r - half);
            int hi = std::min(n - 1, r + half);
            for (int p = lo; p <= hi; p++) {
                int nb = sorted_idx[j][p];
                if (nb != i && seen_stamp[nb] != i) {
                    seen_stamp[nb] = i;
                    candidates.push_back(nb);
                }
            }
        }

        for (int c : candidates) {
            float d = dist_fn(data[i].data(), data[c].data(), dim);
            dist_comps++;
            graph.try_update(i, c, d);
        }

        if (i % 10000 == 0 && i > 0)
            std::cout << "[proj-init] " << i << "/" << n << " points done\n";
    }

    std::cout << "[proj-init] Done. dist_comps=" << dist_comps << "\n";
    return graph;
}