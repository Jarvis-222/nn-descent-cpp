#pragma once

#include <string>
#include "types.h"

bool save_knn_graph(const std::string& filename, const KNNGraph& graph);
bool load_knn_graph(const std::string& filename, KNNGraph& graph);
