"""
PyNNDescent: Construction Recall vs Search (Query) Recall benchmark.

Usage:
  python pynndescent_recall_bench.py --dataset gist1m
  python pynndescent_recall_bench.py --dataset sift1m
  python pynndescent_recall_bench.py --dataset gist100k

Measures:
  1. Construction recall  — how good is the k-NN graph vs brute-force GT
  2. Search recall@k      — how good are query results at various ef values
  3. Timing               — construction time, queries per second
"""

import argparse
import struct
import time
import os
import sys

import numpy as np
import pynndescent

# ─── I/O helpers ─────────────────────────────────────────────────────────

def load_fvecs(path):
    """Load .fvecs file -> np.ndarray (n, d), float32."""
    with open(path, "rb") as f:
        buf = f.read()
    offset = 0
    vecs = []
    while offset < len(buf):
        (dim,) = struct.unpack_from("<i", buf, offset)
        offset += 4
        vec = np.frombuffer(buf, dtype=np.float32, count=dim, offset=offset)
        vecs.append(vec)
        offset += dim * 4
    return np.array(vecs)


def load_ivecs(path):
    """Load .ivecs file -> np.ndarray (n, d), int32."""
    with open(path, "rb") as f:
        buf = f.read()
    offset = 0
    vecs = []
    while offset < len(buf):
        (dim,) = struct.unpack_from("<i", buf, offset)
        offset += 4
        vec = np.frombuffer(buf, dtype=np.int32, count=dim, offset=offset)
        vecs.append(vec.copy())
        offset += dim * 4
    return np.array(vecs)


def load_bin_gt(path):
    """Load construction ground truth in your C++ .bin format -> np.ndarray (n, k), int32."""
    with open(path, "rb") as f:
        n, k = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(n * k * 4), dtype=np.int32).reshape(n, k)
    print(f"[gt] Loaded {path}: n={n}, k={k}")
    return data


# ─── Recall computation ─────────────────────────────────────────────────

def recall_at_k(predicted, ground_truth, k=None):
    """Compute recall@k. predicted/ground_truth: (n, k') arrays of indices."""
    if k is None:
        k = ground_truth.shape[1]
    correct = 0
    total = 0
    for i in range(predicted.shape[0]):
        gt_set = set(ground_truth[i, :k].tolist())
        pred_k = predicted[i, :k]
        correct += len(gt_set.intersection(pred_k.tolist()))
        total += k
    return correct / total if total > 0 else 0.0


# ─── Dataset configs ─────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

DATASETS = {
    "sift1m": {
        "base":       os.path.join(DATA_DIR, "sift/sift_base.fvecs"),
        "query":      os.path.join(DATA_DIR, "sift/sift_query.fvecs"),
        "search_gt":  os.path.join(DATA_DIR, "sift/sift_groundtruth.ivecs"),
        "build_gt":   os.path.join(DATA_DIR, "sift1m_l2_gt.bin"),
        "metric":     "euclidean",
    },
    "gist1m": {
        "base":       os.path.join(DATA_DIR, "gist/gist_base.fvecs"),
        "query":      os.path.join(DATA_DIR, "gist/gist_query_heldout_1k.fvecs"),
        "search_gt":  os.path.join(DATA_DIR, "gist/gist_groundtruth.ivecs"),
        "build_gt":   os.path.join(DATA_DIR, "gist100k_l2_gt.bin"),  # only 100k available
        "metric":     "euclidean",
    },
    "gist100k": {
        "base":       os.path.join(DATA_DIR, "gist/gist_base_100k.fvecs"),
        "query":      os.path.join(DATA_DIR, "gist/gist_query_heldout_1k.fvecs"),
        "search_gt":  os.path.join(DATA_DIR, "gist100k_search_gt_heldout_1k.ivecs"),
        "build_gt":   os.path.join(DATA_DIR, "gist100k_l2_gt.bin"),
        "metric":     "euclidean",
    },
}

# ─── Main benchmark ─────────────────────────────────────────────────────

def run_benchmark(dataset_name, k=10, efs=None, n_trees_list=None):
    if efs is None:
        efs = [10, 20, 50, 100, 200, 400]
    if n_trees_list is None:
        n_trees_list = [None]  # None = PyNNDescent default

    cfg = DATASETS[dataset_name]
    print(f"\n{'='*70}")
    print(f"  PyNNDescent Benchmark: {dataset_name}")
    print(f"{'='*70}")

    # --- Load data ---
    print(f"\n[1/4] Loading data...")
    base = load_fvecs(cfg["base"])
    print(f"  Base vectors: {base.shape}")

    queries = None
    if os.path.exists(cfg["query"]):
        queries = load_fvecs(cfg["query"])
        print(f"  Query vectors: {queries.shape}")

    search_gt = None
    if os.path.exists(cfg["search_gt"]):
        search_gt = load_ivecs(cfg["search_gt"])
        print(f"  Search GT: {search_gt.shape}")

    build_gt = None
    if os.path.exists(cfg["build_gt"]):
        build_gt = load_bin_gt(cfg["build_gt"])
        # Trim to min(k, gt_k)
        if build_gt.shape[1] > k:
            build_gt = build_gt[:, :k]
    else:
        print(f"  [warn] No construction GT found at {cfg['build_gt']}")
        print(f"         Construction recall will be skipped.")

    results = []

    for n_trees in n_trees_list:
        label = f"n_trees={n_trees}" if n_trees else "default"
        print(f"\n{'─'*60}")
        print(f"  Config: k={k}, {label}, metric={cfg['metric']}")
        print(f"{'─'*60}")

        # --- Build index ---
        print(f"\n[2/4] Building PyNNDescent index...")
        build_kwargs = dict(
            n_neighbors=k,
            metric=cfg["metric"],
            low_memory=False,
            verbose=True,
        )
        if n_trees is not None:
            build_kwargs["n_trees"] = n_trees

        t0 = time.time()
        index = pynndescent.NNDescent(base, **build_kwargs)
        build_time = time.time() - t0
        print(f"  Construction time: {build_time:.2f}s")

        # --- Construction recall ---
        print(f"\n[3/4] Evaluating construction recall...")
        # neighbor_graph returns (indices, distances) for all base points
        graph_indices, graph_distances = index.neighbor_graph

        if build_gt is not None:
            # Make sure shapes are compatible
            n_eval = min(build_gt.shape[0], graph_indices.shape[0])
            construction_recall = recall_at_k(
                graph_indices[:n_eval, :k],
                build_gt[:n_eval, :k],
                k=k,
            )
            print(f"  Construction recall@{k}: {construction_recall:.4f}")
        else:
            construction_recall = None
            print(f"  Construction recall: SKIPPED (no GT)")

        # --- Search recall at various ef ---
        print(f"\n[4/4] Search recall sweep...")
        if queries is not None and search_gt is not None:
            # Prepare the index for querying
            index.prepare()

            search_k = min(k, search_gt.shape[1])

            print(f"  {'ef':<8} {'recall@'+str(search_k):<14} {'QPS':<12} {'time (s)':<10}")
            print(f"  {'─'*48}")

            for ef in efs:
                # Run queries
                t0 = time.time()
                query_indices, query_distances = index.query(
                    queries, k=search_k, epsilon=ef / search_k - 1.0
                )
                query_time = time.time() - t0

                qps = queries.shape[0] / query_time
                search_recall = recall_at_k(query_indices, search_gt, k=search_k)

                print(f"  {ef:<8} {search_recall:<14.4f} {qps:<12.1f} {query_time:<10.3f}")

                results.append({
                    "dataset": dataset_name,
                    "config": label,
                    "k": k,
                    "build_time": build_time,
                    "construction_recall": construction_recall,
                    "ef": ef,
                    "search_recall": search_recall,
                    "qps": qps,
                })
        else:
            print(f"  SKIPPED (no queries or search GT)")
            results.append({
                "dataset": dataset_name,
                "config": label,
                "k": k,
                "build_time": build_time,
                "construction_recall": construction_recall,
                "ef": None,
                "search_recall": None,
                "qps": None,
            })

    # --- Write CSV ---
    csv_path = os.path.join(SCRIPT_DIR, "results",
                            f"pynndescent_{dataset_name}_recall.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("dataset,config,k,build_time,construction_recall,ef,search_recall,qps\n")
        for r in results:
            f.write(f"{r['dataset']},{r['config']},{r['k']},"
                    f"{r['build_time']:.2f},"
                    f"{r['construction_recall'] if r['construction_recall'] is not None else ''},"
                    f"{r['ef'] if r['ef'] is not None else ''},"
                    f"{r['search_recall'] if r['search_recall'] is not None else ''},"
                    f"{r['qps']:.1f}" if r['qps'] is not None else "")
            f.write("\n")
    print(f"\nResults saved to: {csv_path}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {dataset_name}")
    print(f"{'='*70}")
    print(f"  Build time:          {build_time:.2f}s")
    if construction_recall is not None:
        print(f"  Construction recall:  {construction_recall:.4f}")
    if results and results[0]["search_recall"] is not None:
        for r in results:
            if r["ef"] is not None:
                print(f"  Search recall (ef={r['ef']:<3}): {r['search_recall']:.4f}  "
                      f"({r['qps']:.0f} QPS)")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyNNDescent construction recall vs search recall benchmark")
    parser.add_argument("--dataset", type=str, default="gist1m",
                        choices=list(DATASETS.keys()),
                        help="Dataset to benchmark (default: gist1m)")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of neighbors (default: 10)")
    parser.add_argument("--ef", type=str, default="10,20,50,100,200,400",
                        help="Comma-separated ef values for search (default: 10,20,50,100,200,400)")
    args = parser.parse_args()

    efs = [int(x) for x in args.ef.split(",")]
    run_benchmark(args.dataset, k=args.k, efs=efs)
