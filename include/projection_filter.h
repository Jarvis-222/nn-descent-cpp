#pragma once

#include <vector>
#include <cmath>
#include "types.h"
#include "distance.h"

// ProjectionFilter: LSH-APG style projected-distance filter for NN-Descent.
//
// Instead of binary fingerprints + popcount (our collision filter),
// this stores continuous random projections P(x) = (a_1·x, ..., a_m·x)
// and uses the chi-squared concentration property:
//
//   ‖P(o1) - P(o2)‖² / ‖o1 - o2‖²  ~  χ²(m)
//
// Filter condition: skip pair (u1, u2) if
//   proj_dist(u1, u2) > t * dk
// where t = √(χ²_{pτ}(m)) and dk = farthest neighbor distance.
//
// Also supports initialization: find K-NN in the cheap m-dim projected
// space, then refine with true distances. One set of projections for
// both init and filtering.
//
// Reference: LSH-APG (PVLDB 2023), Section 5, Equation 4.

class ProjectionFilter {
public:
    ProjectionFilter() = default;

    // Build random projections for all data points.
    //   data: n points, each of dimension dim
    //   m: number of random projections
    //   p_tau: confidence level (e.g., 0.95 means 95% chance of not
    //          filtering a true neighbor)
    void build(const std::vector<std::vector<float>>& data, int m, float p_tau);

    // Squared projected distance between points u and v in m-dim space.
    float projected_dist_sq(int u, int v) const;

    // Should we filter (skip) the pair (u1, u2)?
    bool should_filter(int u1, int u2, float dk1, float dk2) const;

    // Getters
    int num_projections() const { return m_; }
    float threshold() const { return t_; }
    bool is_built() const { return !projections_.empty(); }

    // Direct access for prefetching
    const float* projection_ptr(int point_id) const {
        return &projections_[point_id * m_];
    }

private:
    int n_ = 0;
    int m_ = 0;                         // number of projections
    float t_ = 0.0f;                    // √(χ²_pτ(m)) threshold
    float t_sq_ = 0.0f;                 // t² = χ²_pτ(m) (avoid sqrt in hot loop)
    std::vector<float> projections_;    // flat: n * m floats
};