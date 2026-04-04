#!/usr/bin/env python3
"""Fast FAISS ground-truth computation for construction and search evaluation.

Examples:
  Self-search / construction GT (backward-compatible):
    python compute_gt.py data/gist/gist_base.fvecs data/gist1m_l2_gt.bin 10

  Query-to-base / search GT:
    python compute_gt.py --base data/gist/gist_base_100k.fvecs \
        --query data/gist/gist_query_heldout.fvecs \
        --output data/gist100k_search_gt.ivecs --k 10
"""

import argparse
import os
import struct
import sys
from time import time

import faiss
import numpy as np


def load_fvecs_fast(fname):
    """Load .fvecs file using numpy; returns contiguous float32 array."""
    fsize = os.path.getsize(fname)
    with open(fname, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    record_bytes = (dim + 1) * 4
    n = fsize // record_bytes
    print(f"  File: {fname} -> n={n}, dim={dim}")
    raw = np.fromfile(fname, dtype=np.float32).reshape(n, dim + 1)
    return raw[:, 1:].copy()


def save_bin_gt(out_file, indices):
    n, k = indices.shape
    with open(out_file, "wb") as f:
        f.write(struct.pack("ii", n, k))
        indices.astype(np.int32, copy=False).tofile(f)
    print(f"Saved binary GT to {out_file} (n={n}, k={k})")


def save_ivecs(out_file, indices):
    n, k = indices.shape
    with open(out_file, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", k))
            indices[i].astype(np.int32, copy=False).tofile(f)
    print(f"Saved ivecs GT to {out_file} (n={n}, k={k})")


def infer_format(out_file, requested):
    if requested != "auto":
        return requested
    return "ivecs" if out_file.endswith(".ivecs") else "bin"


def build_index(dim):
    index_flat = faiss.IndexFlatL2(dim)
    ngpu = faiss.get_num_gpus()

    if ngpu > 1:
        print(f"Using {ngpu} GPUs")
        return faiss.index_cpu_to_all_gpus(index_flat)
    if ngpu == 1:
        print("Using 1 GPU")
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index_flat)

    nthreads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(nthreads)
    print(f"No GPU found — using CPU ({nthreads} threads)")
    return index_flat


def default_batch(dim):
    if dim <= 200:
        return 100000
    if dim <= 500:
        return 50000
    return 10000


def compute_gt(base, queries, k, self_search):
    nb, dim = base.shape
    nq = queries.shape[0]

    index = build_index(dim)
    index.add(base)

    batch = default_batch(dim)
    gt = np.zeros((nq, k), dtype=np.int32)
    t0 = time()
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        search_k = k + 1 if self_search else k
        _, I = index.search(queries[s:e], search_k)

        if self_search:
            for i in range(e - s):
                nbs = I[i][I[i] != (s + i)][:k]
                gt[s + i] = nbs
        else:
            gt[s:e] = I[:, :k]

        elapsed = time() - t0
        speed = e / elapsed if elapsed > 0 else 0.0
        eta = (nq - e) / speed if speed > 0 else 0.0
        print(f"  {e:>8d}/{nq} | {elapsed:6.1f}s | {speed:,.0f} vec/s | ETA {eta:.0f}s")

    total = time() - t0
    print(f"\nDone in {total:.1f}s ({nq / total:,.0f} vec/s)")
    return gt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Base .fvecs file for backward-compatible self-search mode")
    parser.add_argument("output_pos", nargs="?", help="Output file for backward-compatible self-search mode")
    parser.add_argument("k_pos", nargs="?", type=int, help="K for backward-compatible self-search mode")
    parser.add_argument("--base", help="Base dataset .fvecs file")
    parser.add_argument("--query", help="Query .fvecs file; if omitted, run self-search on base")
    parser.add_argument("--output", help="Output GT path (.bin or .ivecs)")
    parser.add_argument("--k", type=int, default=None, help="Number of neighbors")
    parser.add_argument("--format", choices=["auto", "bin", "ivecs"], default="auto",
                        help="Output format; auto uses .ivecs suffix, else binary [n,k,data]")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.base or args.output or args.query or args.k is not None:
        base_file = args.base
        query_file = args.query
        out_file = args.output
        k = args.k if args.k is not None else 10
    else:
        if args.input is None or args.output_pos is None:
            print(f"Usage: {sys.argv[0]} <input.fvecs> <output_gt.bin> [K=10]")
            sys.exit(1)
        base_file = args.input
        query_file = None
        out_file = args.output_pos
        k = args.k_pos if args.k_pos is not None else 10

    if not base_file or not out_file:
        print("ERROR: base input and output path are required.")
        sys.exit(1)

    print(f"Loading base {base_file}...")
    t0 = time()
    base = load_fvecs_fast(base_file)
    nb, dim = base.shape
    print(f"Loaded base {nb} x {dim} in {time() - t0:.1f}s")

    if query_file:
        print(f"Loading queries {query_file}...")
        t0 = time()
        queries = load_fvecs_fast(query_file)
        nq, qdim = queries.shape
        if qdim != dim:
            print(f"ERROR: Query dim {qdim} does not match base dim {dim}")
            sys.exit(1)
        print(f"Loaded queries {nq} x {qdim} in {time() - t0:.1f}s")
        self_search = False
    else:
        queries = base
        nq = nb
        self_search = True
        print("Mode: self-search (construction GT)")

    gt = compute_gt(base, queries, k, self_search)

    out_format = infer_format(out_file, args.format)
    if out_format == "ivecs":
        save_ivecs(out_file, gt)
    else:
        save_bin_gt(out_file, gt)


if __name__ == "__main__":
    main()
