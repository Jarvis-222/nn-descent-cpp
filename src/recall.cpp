#include "recall.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>

// ── Recall Computation ───────────────────────────────────────────────────

double compute_recall(const std::vector<std::vector<int>>& predicted,
                      const std::vector<std::vector<int>>& ground_truth) {
    int correct = 0, total = 0;
    for (int i = 0; i < (int)predicted.size(); i++) {
        std::unordered_set<int> gt_set(ground_truth[i].begin(), ground_truth[i].end());
        for (int idx : predicted[i])
            if (gt_set.count(idx)) correct++;
        total += (int)ground_truth[i].size();
    }
    return (total > 0) ? (double)correct / total : 0.0;
}

std::vector<std::vector<int>> compute_ground_truth(
    const std::vector<std::vector<float>>& data, int k, DistFunc dist_fn) {
    int n = (int)data.size();
    int dim = (int)data[0].size();
    std::vector<std::vector<int>> gt(n);
    std::cout << "[ground-truth] Brute-force k-NN: n=" << n << " k=" << k << "\n";

    for (int i = 0; i < n; i++) {
        std::vector<std::pair<float, int>> dists;
        dists.reserve(n - 1);
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            dists.push_back({dist_fn(data[i].data(), data[j].data(), dim), j});
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[i].resize(k);
        for (int t = 0; t < k; t++) gt[i][t] = dists[t].second;

        if ((i + 1) % 1000 == 0 || i == n - 1)
            std::cout << "  " << (i + 1) << "/" << n << "\r" << std::flush;
    }
    std::cout << "\n[ground-truth] Done.\n";
    return gt;
}

// ── Report Writer ────────────────────────────────────────────────────────

static const char* init_str(InitMethod m) {
    switch (m) {
        case InitMethod::LSH:     return "LSH";
        case InitMethod::RP_TREE: return "RP-Tree";
        default:                  return "Random";
    }
}

static const char* metric_str(DistanceMetric m) {
    switch (m) {
        case DistanceMetric::COSINE:    return "Cosine";
        case DistanceMetric::MANHATTAN: return "Manhattan";
        default:                        return "Euclidean";
    }
}

void write_report(const std::string& filename,
                  const NNDescentConfig& config,
                  const std::vector<IterationStats>& iter_log,
                  long long init_dist_comps, double init_time_sec,
                  double total_time_sec, double final_recall,
                  const std::vector<std::vector<int>>& predicted,
                  const std::vector<std::vector<int>>& ground_truth) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "ERROR: Cannot open " << filename << " for writing.\n";
        return;
    }
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    out << "================================================================\n"
        << "  NN-Descent Experiment Report\n"
        << "  Generated: " << std::ctime(&t)
        << "================================================================\n\n";

    // Configuration
    out << "--- Configuration ---\n"
        << "n=" << config.n << " dim=" << config.dim << " k=" << config.k << "\n"
        << "init=" << init_str(config.init_method)
        << " metric=" << metric_str(config.metric) << "\n"
        << "rho=" << config.rho << " delta=" << config.delta
        << " max_iter=" << config.max_iterations << "\n"
        << "filter=" << (config.use_collision_filter ? "ON" : "OFF");
    if (config.use_collision_filter)
        out << " tables=" << config.num_tables
            << " margin=" << config.margin;
    out << "\n\n";

    // Summary
    long long total_dist = iter_log.empty() ? init_dist_comps : iter_log.back().cumul_dist;
    long long total_filtered = 0;
    for (auto& s : iter_log) total_filtered += s.filtered;

    out << "--- Summary ---\n"
        << "init_dist_comps=" << init_dist_comps << " init_time=" << std::fixed
        << std::setprecision(4) << init_time_sec << "s\n"
        << "iterations=" << (int)iter_log.size()
        << " total_dist=" << total_dist << " total_filtered=" << total_filtered << "\n"
        << "total_time=" << std::fixed << std::setprecision(4) << total_time_sec << "s"
        << " final_recall=" << std::fixed << std::setprecision(6) << final_recall << "\n\n";

    // Per-iteration table
    out << "--- Per-Iteration ---\n"
        << std::left << std::setw(6) << "Iter" << std::setw(10) << "Updates"
        << std::setw(14) << "DistComps" << std::setw(14) << "Filtered"
        << std::setw(10) << "Rate%" << std::setw(14) << "CumulDist"
        << std::setw(10) << "Recall" << "\n"
        << std::string(78, '-') << "\n";
    for (auto& s : iter_log) {
        out << std::left << std::setw(6) << s.iteration
            << std::setw(10) << s.updates
            << std::setw(14) << s.dist_comps
            << std::setw(14) << s.filtered
            << std::setw(10) << std::fixed << std::setprecision(2) << (s.filter_rate * 100.0)
            << std::setw(14) << s.cumul_dist
            << std::setw(10) << std::fixed << std::setprecision(4) << s.recall << "\n";
    }
    out << "\n";

    // CSV block for easy parsing
    out << "--- CSV ---\n"
        << "iter,dist_comps,filtered,filter_rate,updates,cumul_dist,recall\n";
    for (auto& s : iter_log) {
        out << s.iteration << "," << s.dist_comps << "," << s.filtered << ","
            << std::fixed << std::setprecision(6) << s.filter_rate << ","
            << s.updates << "," << s.cumul_dist << "," << s.recall << "\n";
    }

    out << "\n================================================================\n";
    out.close();
    std::cout << "[report] Written to " << filename << "\n";
}

// ── Data Loaders ─────────────────────────────────────────────────────────

std::vector<std::vector<float>> load_data(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream in(filename);
    if (!in.is_open()) { std::cerr << "ERROR: Cannot open " << filename << "\n"; return data; }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<float> point;
        std::stringstream ss(line);
        float val; char delim;
        while (ss >> val) { point.push_back(val); ss >> delim; }
        if (!point.empty()) data.push_back(std::move(point));
    }
    std::cout << "[data] Loaded " << data.size() << " points, dim="
              << (data.empty() ? 0 : (int)data[0].size()) << " from " << filename << "\n";
    return data;
}

std::vector<std::vector<float>> load_mnist_idx(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) { std::cerr << "ERROR: Cannot open " << filename << "\n"; return data; }

    auto read_be32 = [&]() -> int {
        unsigned char buf[4];
        in.read(reinterpret_cast<char*>(buf), 4);
        return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
    };

    int magic = read_be32();
    if (magic != 2051) { std::cerr << "ERROR: Not an IDX file (magic=" << magic << ")\n"; return data; }
    int n = read_be32(), rows = read_be32(), cols = read_be32();
    int dim = rows * cols;

    data.resize(n, std::vector<float>(dim));
    for (int i = 0; i < n; i++) {
        std::vector<unsigned char> pixels(dim);
        in.read(reinterpret_cast<char*>(pixels.data()), dim);
        for (int d = 0; d < dim; d++) data[i][d] = (float)pixels[d];
    }
    std::cout << "[data] Loaded " << n << " points, dim=" << dim << " from " << filename << " (IDX)\n";
    return data;
}

std::vector<std::vector<float>> load_fvecs(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) { std::cerr << "ERROR: Cannot open " << filename << "\n"; return data; }
    while (in.good()) {
        int dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in.good()) break;
        std::vector<float> point(dim);
        in.read(reinterpret_cast<char*>(point.data()), dim * sizeof(float));
        if (!in.good()) break;
        data.push_back(std::move(point));
    }
    std::cout << "[data] Loaded " << data.size() << " points, dim="
              << (data.empty() ? 0 : (int)data[0].size()) << " from " << filename << " (fvecs)\n";
    return data;
}

// ── Ground Truth I/O ─────────────────────────────────────────────────────

void save_ground_truth(const std::string& filename,
                       const std::vector<std::vector<int>>& gt) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) { std::cerr << "ERROR: Cannot open " << filename << "\n"; return; }
    int n = (int)gt.size();
    int k = gt.empty() ? 0 : (int)gt[0].size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    out.write(reinterpret_cast<const char*>(&k), sizeof(int));
    for (int i = 0; i < n; i++)
        out.write(reinterpret_cast<const char*>(gt[i].data()), k * sizeof(int));
    out.close();
    std::cout << "[ground-truth] Saved to " << filename << " (n=" << n << " k=" << k << ")\n";
}

std::vector<std::vector<int>> load_ground_truth(const std::string& filename) {
    std::vector<std::vector<int>> gt;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) { std::cerr << "ERROR: Cannot open " << filename << "\n"; return gt; }
    int n, k;
    in.read(reinterpret_cast<char*>(&n), sizeof(int));
    in.read(reinterpret_cast<char*>(&k), sizeof(int));
    gt.resize(n, std::vector<int>(k));
    for (int i = 0; i < n; i++)
        in.read(reinterpret_cast<char*>(gt[i].data()), k * sizeof(int));
    std::cout << "[ground-truth] Loaded from " << filename << " (n=" << n << " k=" << k << ")\n";
    return gt;
}
