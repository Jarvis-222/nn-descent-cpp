#!/usr/bin/env python3
"""Fast GPU ground truth computation using FAISS.

Usage:
  python compute_gt.py <input.fvecs> <output_gt.bin> [K=10]

Examples:
  python compute_gt.py data/sift/sift_base.fvecs data/sift1m_l2_gt.bin 10
  python compute_gt.py data/gist/gist_base.fvecs data/gist1m_l2_gt.bin 10
"""
import struct, sys, os
import numpy as np
import faiss
from time import time


def load_fvecs_fast(fname):
    """Load .fvecs file using memory-mapped numpy — much faster than record-by-record."""
    fsize = os.path.getsize(fname)
    with open(fname, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    # Each record: 4 bytes (dim) + dim*4 bytes (floats) = (dim+1)*4 bytes
    record_bytes = (dim + 1) * 4
    n = fsize // record_bytes
    print(f"  File: {fname} -> n={n}, dim={dim}")
    raw = np.fromfile(fname, dtype=np.float32).reshape(n, dim + 1)
    return raw[:, 1:].copy()  # skip the dim field in each row


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.fvecs> <output_gt.bin> [K=10]")
        sys.exit(1)

    data_file = sys.argv[1]
    out_file = sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Load data
    print(f"Loading {data_file}...")
    t0 = time()
    data = load_fvecs_fast(data_file)
    n, d = data.shape
    print(f"Loaded {n} x {d} in {time() - t0:.1f}s")

    # Build FAISS index — try multi-GPU, then single GPU, then CPU
    index_flat = faiss.IndexFlatL2(d)
    ngpu = faiss.get_num_gpus()

    if ngpu > 1:
        print(f"Using {ngpu} GPUs")
        index = faiss.index_cpu_to_all_gpus(index_flat)
    elif ngpu == 1:
        print("Using 1 GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    else:
        nthreads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(nthreads)
        print(f"No GPU found — using CPU ({nthreads} threads)")
        index = index_flat

    index.add(data)

    # Search in batches — larger for low-dim, smaller for high-dim
    if d <= 200:
        batch = 100000  # SIFT 128-dim: big batches
    elif d <= 500:
        batch = 50000
    else:
        batch = 10000   # GIST 960-dim: smaller batches

    gt = np.zeros((n, K), dtype=np.int32)
    t0 = time()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        D, I = index.search(data[s:e], K + 1)
        # Remove self from results
        for i in range(e - s):
            nbs = I[i][I[i] != (s + i)][:K]
            gt[s + i] = nbs
        elapsed = time() - t0
        speed = e / elapsed
        eta = (n - e) / speed if speed > 0 else 0
        print(f"  {e:>8d}/{n} | {elapsed:6.1f}s | {speed:,.0f} vec/s | ETA {eta:.0f}s")

    total = time() - t0
    print(f"\nDone in {total:.1f}s ({n/total:,.0f} vec/s)")

    # Save in our binary format: [n, K, then n*K int32s]
    with open(out_file, "wb") as f:
        f.write(struct.pack("ii", n, K))
        gt.tofile(f)
    print(f"Saved to {out_file} (n={n}, K={K})")


if __name__ == "__main__":
    main()
