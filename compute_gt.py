#!/usr/bin/env python3
"""Compute all-pairs K-NN ground truth using FAISS (GPU if available, else CPU)."""
import struct, numpy as np, faiss, sys
from time import time


def load_fvecs(fname):
    with open(fname, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
        f.seek(0)
        vecs = []
        while True:
            buf = f.read(4)
            if len(buf) < 4:
                break
            dim = struct.unpack("i", buf)[0]
            vecs.append(np.frombuffer(f.read(4 * dim), np.float32).copy())
    return np.array(vecs)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.fvecs> <output_gt.bin> [K=10]")
        sys.exit(1)

    data_file = sys.argv[1]
    out_file = sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print(f"Loading {data_file}...")
    t0 = time()
    data = load_fvecs(data_file)
    n, d = data.shape
    print(f"Loaded {n} x {d} in {time() - t0:.1f}s")

    # Build index — GPU if available, else CPU
    index = faiss.IndexFlatL2(d)
    use_gpu = False
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        use_gpu = True
        print("Using GPU")
    except Exception:
        faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        print(f"Using CPU ({faiss.omp_get_max_threads()} threads)")

    index.add(data)

    # Smaller batches for high-dim data
    batch = 50000 if d < 200 else 10000
    gt = np.zeros((n, K), dtype=np.int32)
    t0 = time()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        D, I = index.search(data[s:e], K + 1)
        for i in range(e - s):
            nbs = [x for x in I[i] if x != s + i][:K]
            gt[s + i] = nbs
        elapsed = time() - t0
        eta = elapsed / e * (n - e)
        print(f"  {e}/{n} ({elapsed:.1f}s, ETA {eta:.0f}s)")

    print(f"Done in {time() - t0:.1f}s")

    with open(out_file, "wb") as f:
        f.write(struct.pack("ii", n, K))
        gt.tofile(f)
    print(f"Saved to {out_file} (n={n}, K={K})")


if __name__ == "__main__":
    main()