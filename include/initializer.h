#pragma once

#include <vector>
#include <cstdint>
#include "types.h"
#include "distance.h"

// CollisionTable: stores LSH/RP-tree hash codes for bucket-based initialization
// and compact fingerprints for fast collision-based distance filtering.
//
// For E2LSH with L tables and K hash functions per table:
//   - codes[i][l]: compound hash for point i in table l (used for bucket lookups)
//   - fingerprints[i*2..i*2+1]: 128-bit fingerprint (used for collision counting)
struct CollisionTable {
    int n = 0;
    int L = 0;
    int total_hashes = 0;                           // L * K
    std::vector<std::vector<int>> codes;            // compound hash per table (for buckets)
    std::vector<std::vector<int>> filter_codes;     // individual hash per function (for fingerprints)
    std::vector<uint64_t> fingerprints;             // flat: 2 x uint64 per point (128 bits)

    // Build 128-bit fingerprints from filter_codes.
    // Each hash function sets one bit; shared bits between two points
    // approximate their collision count.
    void build_fingerprints() {
        fingerprints.assign(n * 2, 0ULL);
        for (int i = 0; i < n; i++) {
            uint64_t fp0 = 0, fp1 = 0;
            for (int h = 0; h < total_hashes; h++) {
                unsigned int mixed = (unsigned int)h * 2654435761u
                                   ^ (unsigned int)filter_codes[i][h] * 2246822519u;
                int bit = mixed % 128;
                if (bit < 64) fp0 |= (1ULL << bit);
                else          fp1 |= (1ULL << (bit - 64));
            }
            fingerprints[i * 2]     = fp0;
            fingerprints[i * 2 + 1] = fp1;
        }
    }

    // Approximate collision count via popcount on shared fingerprint bits.
    int collision_count(int u, int v) const {
        uint64_t shared0 = fingerprints[u * 2]     & fingerprints[v * 2];
        uint64_t shared1 = fingerprints[u * 2 + 1] & fingerprints[v * 2 + 1];
        return __builtin_popcountll(shared0) + __builtin_popcountll(shared1);
    }
};

// Initialization strategies
KNNGraph init_random(const std::vector<std::vector<float>>& data,
                     int k, DistFunc dist_fn, long long& dist_comps);

KNNGraph init_lsh(const std::vector<std::vector<float>>& data,
                  int k, int L, int K, int num_probes, DistFunc dist_fn,
                  CollisionTable& table_out, long long& dist_comps);

KNNGraph init_rp_tree(const std::vector<std::vector<float>>& data,
                      int k, int L, DistFunc dist_fn,
                      long long& dist_comps);

// Build hash tables (can be called separately for random init + filter)
CollisionTable build_lsh_table(const std::vector<std::vector<float>>& data, int L, int K = 4);
