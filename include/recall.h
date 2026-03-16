#pragma once

#include <vector>
#include <string>
#include "types.h"
#include "distance.h"

double compute_recall(const std::vector<std::vector<int>>& predicted,
                      const std::vector<std::vector<int>>& ground_truth);

std::vector<std::vector<int>> compute_ground_truth(
    const std::vector<std::vector<float>>& data,
    int k, DistFunc dist_fn);

void write_report(const std::string& filename,
                  const NNDescentConfig& config,
                  const std::vector<IterationStats>& iter_log,
                  long long init_dist_comps,
                  double init_time_sec,
                  double total_time_sec,
                  double final_recall,
                  const std::vector<std::vector<int>>& predicted,
                  const std::vector<std::vector<int>>& ground_truth);

std::vector<std::vector<float>> load_data(const std::string& filename);
std::vector<std::vector<float>> load_fvecs(const std::string& filename);
std::vector<std::vector<float>> load_mnist_idx(const std::string& filename);

void save_ground_truth(const std::string& filename,
                       const std::vector<std::vector<int>>& gt);
std::vector<std::vector<int>> load_ground_truth(const std::string& filename);
