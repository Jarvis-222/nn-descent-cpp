#include "graph_io.h"

#include <cstdint>
#include <fstream>

namespace {

constexpr std::uint32_t kGraphMagic = 0x4b4e4e47; // "KNNG"
constexpr std::uint32_t kGraphVersion = 1;

} // namespace

bool save_knn_graph(const std::string& filename, const KNNGraph& graph) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    std::uint32_t magic = kGraphMagic;
    std::uint32_t version = kGraphVersion;
    std::int32_t n = graph.n;
    std::int32_t k = graph.k;

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&k), sizeof(k));

    for (int i = 0; i < graph.n; i++) {
        std::int32_t degree = static_cast<std::int32_t>(graph.neighbors[i].size());
        out.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
        for (const auto& nb : graph.neighbors[i]) {
            std::int32_t idx = nb.index;
            float dist = nb.distance;
            out.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
            out.write(reinterpret_cast<const char*>(&dist), sizeof(dist));
        }
    }

    return static_cast<bool>(out);
}

bool load_knn_graph(const std::string& filename, KNNGraph& graph) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    std::uint32_t magic = 0, version = 0;
    std::int32_t n = 0, k = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&k), sizeof(k));

    if (!in || magic != kGraphMagic || version != kGraphVersion || n < 0 || k < 0)
        return false;

    graph = KNNGraph(n, k);
    for (int i = 0; i < n; i++) {
        std::int32_t degree = 0;
        in.read(reinterpret_cast<char*>(&degree), sizeof(degree));
        if (!in || degree < 0) return false;

        graph.neighbors[i].clear();
        graph.neighbors[i].reserve(degree);
        for (int j = 0; j < degree; j++) {
            std::int32_t idx = -1;
            float dist = 0.0f;
            in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            in.read(reinterpret_cast<char*>(&dist), sizeof(dist));
            if (!in) return false;
            graph.neighbors[i].emplace_back(idx, dist, false);
        }
    }

    return true;
}
