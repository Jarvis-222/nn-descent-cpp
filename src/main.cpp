#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include "nn_descent.h"
#include "recall.h"

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
        << "Data options:\n"
        << "  --data <file>          Input data file (text/CSV or .fvecs)\n"
        << "  --synthetic <n> <dim>  Generate synthetic Gaussian data\n"
        << "\nAlgorithm options:\n"
        << "  --k <int>              Number of neighbors (default: 10)\n"
        << "  --init <method>        Initialization: random|lsh|rptree (default: random)\n"
        << "  --metric <name>        Distance: euclidean|cosine|manhattan (default: euclidean)\n"
        << "  --rho <float>          Sampling fraction (default: 0.5)\n"
        << "  --delta <float>        Convergence threshold (default: 0.001)\n"
        << "  --max-iter <int>       Max iterations (default: 20)\n"
        << "\nCollision filter options:\n"
        << "  --filter               Enable collision-based distance filtering\n"
        << "  --tables <int>         Number of hash tables / RP trees (default: 20)\n"
        << "  --hash-functions <int> Hash functions per table, K (default: 4)\n"
        << "  --probes <int>         Multi-probe count for init (default: 5)\n"
        << "  --margin <int>         Safety margin for filter (default: 0)\n"
        << "\nProjection filter options (LSH-APG style):\n"
        << "  --proj-filter          Enable projected-distance filter\n"
        << "  --num-projections <int> Number of random projections, m (default: 16)\n"
        << "  --filter-confidence <f> Chi-squared confidence pτ (default: 0.95)\n"
        << "\nGround truth options:\n"
        << "  --save-gt <file>       Compute and save ground truth to file\n"
        << "  --load-gt <file>       Load precomputed ground truth from file\n"
        << "  --no-gt                Skip ground-truth computation\n"
        << "\nOutput options:\n"
        << "  --output <file>        Output report file (default: results/report.txt)\n"
        << "\nExamples:\n"
        << "  " << prog << " --data mnist.idx --limit 10000 --k 10 --save-gt data/mnist10k_gt.bin\n"
        << "  " << prog << " --data mnist.idx --limit 10000 --k 10 --load-gt data/mnist10k_gt.bin --init lsh\n"
        << "  " << prog << " --synthetic 5000 128 --k 10 --init random\n";
}

int main(int argc, char** argv) {
    NNDescentConfig config;
    std::string data_file;
    std::string save_gt_file;
    std::string load_gt_file;
    int syn_n = 0, syn_dim = 0;
    int data_limit = 0;
    bool compute_gt = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--synthetic" && i + 2 < argc) {
            syn_n = std::stoi(argv[++i]);
            syn_dim = std::stoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            config.k = std::stoi(argv[++i]);
        } else if (arg == "--init" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "lsh") config.init_method = InitMethod::LSH;
            else if (m == "rptree") config.init_method = InitMethod::RP_TREE;
            else config.init_method = InitMethod::RANDOM;
        } else if (arg == "--metric" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "cosine") config.metric = DistanceMetric::COSINE;
            else if (m == "manhattan") config.metric = DistanceMetric::MANHATTAN;
            else config.metric = DistanceMetric::EUCLIDEAN;
        } else if (arg == "--rho" && i + 1 < argc) {
            config.rho = std::stof(argv[++i]);
        } else if ((arg == "--max-candidates" || arg == "--mc") && i + 1 < argc) {
            config.max_candidates = std::stoi(argv[++i]);
        } else if (arg == "--delta" && i + 1 < argc) {
            config.delta = std::stof(argv[++i]);
        } else if (arg == "--max-iter" && i + 1 < argc) {
            config.max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--filter") {
            config.use_collision_filter = true;
        } else if (arg == "--tables" && i + 1 < argc) {
            config.num_tables = std::stoi(argv[++i]);
        } else if (arg == "--hash-functions" && i + 1 < argc) {
            config.num_hash_functions = std::stoi(argv[++i]);
        } else if (arg == "--probes" && i + 1 < argc) {
            config.num_probes = std::stoi(argv[++i]);
        } else if (arg == "--margin" && i + 1 < argc) {
            config.margin = std::stoi(argv[++i]);
        } else if (arg == "--proj-filter") {
            config.use_projection_filter = true;
        } else if (arg == "--num-projections" && i + 1 < argc) {
            config.num_projections = std::stoi(argv[++i]);
        } else if (arg == "--filter-confidence" && i + 1 < argc) {
            config.filter_confidence = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--limit" && i + 1 < argc) {
            data_limit = std::stoi(argv[++i]);
        } else if (arg == "--save-gt" && i + 1 < argc) {
            save_gt_file = argv[++i];
        } else if (arg == "--load-gt" && i + 1 < argc) {
            load_gt_file = argv[++i];
        } else if (arg == "--no-gt") {
            compute_gt = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::vector<std::vector<float>> data;
    if (!data_file.empty()) {
        if (data_file.size() >= 6 && data_file.substr(data_file.size() - 6) == ".fvecs")
            data = load_fvecs(data_file);
        else if (data_file.find("idx") != std::string::npos || data_file.find("ubyte") != std::string::npos)
            data = load_mnist_idx(data_file);
        else
            data = load_data(data_file);
    } else if (syn_n > 0 && syn_dim > 0) {
        std::cout << "[data] Generating synthetic data: n=" << syn_n << " dim=" << syn_dim << "\n";
        std::mt19937 rng(12345);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        data.resize(syn_n, std::vector<float>(syn_dim));
        for (int i = 0; i < syn_n; i++)
            for (int d = 0; d < syn_dim; d++)
                data[i][d] = normal(rng);
    } else {
        std::cerr << "No data specified. Use --data or --synthetic.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (data.empty()) { std::cerr << "ERROR: No data loaded.\n"; return 1; }

    if (data_limit > 0 && data_limit < (int)data.size()) {
        data.resize(data_limit);
        std::cout << "[data] Truncated to first " << data_limit << " points\n";
    }

    config.n = (int)data.size();
    config.dim = (int)data[0].size();

    std::cout << "\n========================================\n";
    std::cout << "  NN-Descent C++ Experiment\n";
    std::cout << "========================================\n";
    std::cout << "n=" << config.n << " dim=" << config.dim << " k=" << config.k << "\n";
    std::cout << "init=" << (config.init_method == InitMethod::LSH ? "LSH" :
                              config.init_method == InitMethod::RP_TREE ? "RP-Tree" :
                              "Random") << "\n";
    std::cout << "filter=" << (config.use_collision_filter ? "collision" :
                                config.use_projection_filter ? "projection" : "OFF") << "\n\n";

    // Ground truth: load, compute, or skip
    std::vector<std::vector<int>> ground_truth;
    if (!load_gt_file.empty()) {
        ground_truth = load_ground_truth(load_gt_file);
        if (ground_truth.empty()) {
            std::cerr << "ERROR: Failed to load ground truth.\n";
            return 1;
        }
    } else if (compute_gt) {
        auto t0 = std::chrono::high_resolution_clock::now();
        DistFunc dist_fn = get_distance_function(config.metric);
        ground_truth = compute_ground_truth(data, config.k, dist_fn);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "[ground-truth] Time: " << std::chrono::duration<double>(t1 - t0).count() << "s\n\n";
        if (!save_gt_file.empty()) {
            save_ground_truth(save_gt_file, ground_truth);
        }
    }

    NNDescentResult result = run_nn_descent(data, config, ground_truth);

    double final_recall = 0.0;
    auto predicted = result.graph.get_index_matrix();
    if (!ground_truth.empty()) {
        final_recall = compute_recall(predicted, ground_truth);
        std::cout << "\n[final] Recall = " << final_recall << "\n";
    }
    std::cout << "[final] Total dist_comps = " << result.total_dist_comps << "\n";
    std::cout << "[final] Total time = " << result.total_time_sec << "s\n";

    write_report(config.output_file, config, result.iter_log,
                 result.init_dist_comps, result.init_time_sec,
                 result.total_time_sec, final_recall, predicted, ground_truth);

    std::cout << "\nDone!\n";
    return 0;
}
