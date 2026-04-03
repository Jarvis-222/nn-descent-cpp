#pragma once

#include <cmath>
#include "types.h"

// ---------------------------------------------------------------------------
// Distance functions  (operate on raw float pointers for speed)
// ---------------------------------------------------------------------------

inline float euclidean_distance(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

inline float cosine_distance(cruonst float* a, const float* b, int dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-12f) return 1.0f;
    return 1.0f - dot / denom;
}

inline float manhattan_distance(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += std::fabs(a[i] - b[i]);
    }
    return sum;
}

using DistFunc = float(*)(const float*, const float*, int);

inline DistFunc get_distance_function(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::COSINE:    return cosine_distance;
        case DistanceMetric::MANHATTAN: return manhattan_distance;
        default:                        return euclidean_distance;
    }
}
