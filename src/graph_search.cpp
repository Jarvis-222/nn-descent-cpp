#include "graph_search.h"
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <unordered_set>

namespace {

struct SearchTreeNode {
    int left = -1;
    int right = -1;
    float hyperplane_offset = 0.0f;
    std::vector<float> normal;
    std::vector<int> points;

    bool is_leaf() const { return left < 0 && right < 0; }
};

struct SearchTree {
    std::vector<SearchTreeNode> nodes;
    int root = -1;
};

struct SearchForestCache {
    const std::vector<std::vector<float>>* data_ptr = nullptr;
    int num_trees = 0;
    int leaf_size = 0;
    std::vector<SearchTree> forest;
};

SearchForestCache g_search_forest_cache;

int build_search_tree(
    const std::vector<std::vector<float>>& data,
    int dim,
    std::vector<int>& indices,
    int leaf_size,
    int max_depth,
    int depth,
    std::mt19937& rng,
    SearchTree& tree)
{
    SearchTreeNode node;
    if ((int)indices.size() <= leaf_size || depth >= max_depth) {
        node.points = indices;
        int node_id = (int)tree.nodes.size();
        tree.nodes.push_back(std::move(node));
        return node_id;
    }

    std::uniform_int_distribution<int> uid(0, (int)indices.size() - 1);
    int ai = uid(rng), bi = uid(rng);
    while (bi == ai) bi = uid(rng);
    int pa = indices[ai], pb = indices[bi];

    node.normal.resize(dim);
    for (int d = 0; d < dim; d++) {
        node.normal[d] = data[pa][d] - data[pb][d];
        node.hyperplane_offset -= node.normal[d] * (data[pa][d] + data[pb][d]) * 0.5f;
    }

    std::vector<int> left, right;
    left.reserve(indices.size() / 2);
    right.reserve(indices.size() / 2);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int idx : indices) {
        float margin = node.hyperplane_offset;
        for (int d = 0; d < dim; d++)
            margin += node.normal[d] * data[idx][d];

        if (std::fabs(margin) < 1e-8f) {
            if (coin(rng) == 0) left.push_back(idx);
            else                right.push_back(idx);
        } else if (margin > 0.0f) {
            left.push_back(idx);
        } else {
            right.push_back(idx);
        }
    }

    if (left.empty() || right.empty()) {
        left.clear();
        right.clear();
        for (int idx : indices) {
            if (coin(rng) == 0) left.push_back(idx);
            else                right.push_back(idx);
        }
        if (left.empty() || right.empty()) {
            node.points = indices;
            int node_id = (int)tree.nodes.size();
            tree.nodes.push_back(std::move(node));
            return node_id;
        }
    }

    int node_id = (int)tree.nodes.size();
    tree.nodes.push_back(std::move(node));
    tree.nodes[node_id].left = build_search_tree(data, dim, left, leaf_size, max_depth, depth + 1, rng, tree);
    tree.nodes[node_id].right = build_search_tree(data, dim, right, leaf_size, max_depth, depth + 1, rng, tree);
    return node_id;
}

const std::vector<SearchTree>& get_search_forest(
    const std::vector<std::vector<float>>& data,
    int num_trees,
    int leaf_size)
{
    if (g_search_forest_cache.data_ptr == &data &&
        g_search_forest_cache.num_trees == num_trees &&
        g_search_forest_cache.leaf_size == leaf_size) {
        return g_search_forest_cache.forest;
    }

    int n = (int)data.size();
    int dim = (int)data[0].size();
    int max_depth = 200;

    g_search_forest_cache.data_ptr = &data;
    g_search_forest_cache.num_trees = num_trees;
    g_search_forest_cache.leaf_size = leaf_size;
    g_search_forest_cache.forest.clear();
    g_search_forest_cache.forest.reserve(num_trees);

    std::mt19937 rng(12345);
    for (int t = 0; t < num_trees; t++) {
        SearchTree tree;
        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;
        tree.root = build_search_tree(data, dim, indices, leaf_size, max_depth, 0, rng, tree);
        g_search_forest_cache.forest.push_back(std::move(tree));
    }

    return g_search_forest_cache.forest;
}

const std::vector<int>& descend_tree(
    const SearchTree& tree,
    const float* query,
    int dim)
{
    int node_id = tree.root;
    while (!tree.nodes[node_id].is_leaf()) {
        const SearchTreeNode& node = tree.nodes[node_id];
        float margin = node.hyperplane_offset;
        for (int d = 0; d < dim; d++)
            margin += node.normal[d] * query[d];
        node_id = (margin >= 0.0f) ? node.left : node.right;
    }
    return tree.nodes[node_id].points;
}

} // namespace

SearchResult graph_search(
    const KNNGraph& graph,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    int search_k, int ef, float epsilon,
    int num_entry_points,
    DistFunc dist_fn)
{
    int n = graph.n;
    int dim = (int)data[0].size();
    int nq = (int)queries.size();

    if (ef < search_k) ef = search_k;
    if (num_entry_points < 1) num_entry_points = 1;

    int leaf_size = std::max(10, std::min(256, 5 * std::max(graph.k, search_k)));
    const auto& forest = get_search_forest(data, num_entry_points, leaf_size);

    SearchResult result;
    result.indices.resize(nq);
    result.distances.resize(nq);
    result.total_dist_comps = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < nq; q++) {
        const float* query = queries[q].data();
        long long q_dist_comps = 0;

        std::vector<bool> visited(n, false);

        using MinPair = std::pair<float, int>;
        std::priority_queue<MinPair, std::vector<MinPair>, std::greater<MinPair>> candidates;

        using MaxPair = std::pair<float, int>;
        std::priority_queue<MaxPair> result_heap;

        std::unordered_set<int> initial_nodes;
        initial_nodes.reserve((size_t)num_entry_points * (size_t)leaf_size);

        for (const auto& tree : forest) {
            const auto& leaf = descend_tree(tree, query, dim);
            for (int idx : leaf)
                initial_nodes.insert(idx);
        }

        std::vector<std::pair<float, int>> init_pool;
        init_pool.reserve(initial_nodes.size());
        for (int idx : initial_nodes) {
            float d = dist_fn(query, data[idx].data(), dim);
            q_dist_comps++;
            init_pool.push_back({d, idx});
        }

        if (init_pool.empty()) {
            int fallback = 0;
            float d = dist_fn(query, data[fallback].data(), dim);
            q_dist_comps++;
            init_pool.push_back({d, fallback});
        }

        std::sort(init_pool.begin(), init_pool.end());
        int seed_count = std::min((int)init_pool.size(), std::max(ef, search_k));
        for (int i = 0; i < seed_count; i++) {
            int idx = init_pool[i].second;
            float d = init_pool[i].first;
            if (visited[idx]) continue;
            visited[idx] = true;
            candidates.push({d, idx});
            result_heap.push({d, idx});
            if ((int)result_heap.size() > ef)
                result_heap.pop();
        }

        while (!candidates.empty()) {
            auto [c_dist, c_idx] = candidates.top();
            candidates.pop();

            float bound = result_heap.top().first;
            if ((int)result_heap.size() >= ef && c_dist > bound * (1.0f + epsilon))
                break;

            for (auto& nb : graph.neighbors[c_idx]) {
                int nb_idx = nb.index;
                if (nb_idx < 0 || visited[nb_idx]) continue;
                visited[nb_idx] = true;

                float d = dist_fn(query, data[nb_idx].data(), dim);
                q_dist_comps++;

                if ((int)result_heap.size() < ef || d < result_heap.top().first) {
                    candidates.push({d, nb_idx});
                    result_heap.push({d, nb_idx});
                    if ((int)result_heap.size() > ef)
                        result_heap.pop();
                }
            }
        }

        std::vector<std::pair<float, int>> res_vec;
        res_vec.reserve(result_heap.size());
        while (!result_heap.empty()) {
            res_vec.push_back({result_heap.top().first, result_heap.top().second});
            result_heap.pop();
        }
        std::sort(res_vec.begin(), res_vec.end());

        int out_k = std::min(search_k, (int)res_vec.size());
        result.indices[q].resize(out_k);
        result.distances[q].resize(out_k);
        for (int i = 0; i < out_k; i++) {
            result.distances[q][i] = res_vec[i].first;
            result.indices[q][i] = res_vec[i].second;
        }

        result.total_dist_comps += q_dist_comps;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_time_sec = std::chrono::duration<double>(t_end - t_start).count();
    result.qps = (result.total_time_sec > 0) ? nq / result.total_time_sec : 0.0;

    return result;
}
