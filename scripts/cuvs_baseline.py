#!/usr/bin/env python3
"""cuVS baseline: NVIDIA's IVF-PQ configured as a flat PQ scan (one list, one probe).

This is the same computation VecKern performs (LUT + ADC over every vector + top-k), done by a production
library, so its throughput is the bar VecKern's kernels are measured against. cuVS trains its own codebooks,
so recall can differ slightly; both are reported.

usage (cuvs_env):
  python scripts/cuvs_baseline.py --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs --m 16
  python scripts/cuvs_baseline.py --n 10000000 --d 128 --m 16 --nq 1000
"""
import argparse
import csv
import os
import time

import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force, ivf_pq


def read_vecs(path, dtype, limit=None):
    raw = np.fromfile(path, dtype=np.int32 if dtype != np.uint8 else np.uint8)
    if dtype == np.uint8:
        d = int(np.frombuffer(raw[:4].tobytes(), dtype=np.int32)[0])
        rows = raw.reshape(-1, d + 4)[:, 4:]
    else:
        d = int(raw[0])
        rows = raw.reshape(-1, d + 1)[:, 1:]
        if dtype == np.float32:
            rows = rows.view(np.float32)
    return np.ascontiguousarray(rows[:limit] if limit else rows)


def synthetic(n, d, clusters, latent_dim, seed, offset=0):
    """Same distribution as VecKern's SyntheticSource: x = center + A z + noise, centers U(-1,1),
    A ~ N(0, 1/latent_dim), z ~ N(0, 0.5^2), noise N(0, 0.05^2). Different random stream, same distribution."""
    g = np.random.default_rng(seed)
    centers = g.uniform(-1, 1, size=(clusters, d)).astype(np.float32)
    basis = (g.normal(size=(d, latent_dim)) / np.sqrt(latent_dim)).astype(np.float32)
    rng = np.random.default_rng(seed + 1 + offset)
    out = np.empty((n, d), dtype=np.float32)
    chunk = 1 << 20
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        z = rng.normal(0, 0.5, size=(e - s, latent_dim)).astype(np.float32)
        out[s:e] = centers[rng.integers(0, clusters, size=e - s)] + z @ basis.T
        out[s:e] += rng.normal(0, 0.05, size=(e - s, d)).astype(np.float32)
    return out


def recall_at_k(found, truth, k):
    return float(np.mean([len(set(found[i, :k]) & set(truth[i, :k])) / k for i in range(len(found))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"), ap.add_argument("--query"), ap.add_argument("--gt")
    ap.add_argument("--n", type=int, default=1_000_000), ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--clusters", type=int, default=100), ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--m", type=int, default=16), ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=1000), ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--eval-queries", type=int, default=100)
    ap.add_argument("--train-fraction", type=float, default=0.1)
    ap.add_argument("--csv", default="results/cuvs_baseline.csv"), ap.add_argument("--label", default="cuvs")
    a = ap.parse_args()

    if a.base:
        base = read_vecs(a.base, np.float32, a.n)
        queries = read_vecs(a.query, np.float32, a.nq)
    else:
        base = synthetic(a.n, a.d, a.clusters, a.latent_dim, 1)
        queries = synthetic(a.nq, a.d, a.clusters, a.latent_dim, 1, offset=12345)
    n, d = base.shape
    nq = len(queries)
    print(f"data: n={n} d={d} nq={nq} | pq_dim={a.m} pq_bits=8 | k={a.k}")

    d_base, d_queries = cp.asarray(base), cp.asarray(queries)
    ne = min(a.eval_queries, nq)
    if a.gt:
        truth = read_vecs(a.gt, np.int32)[:ne]
    else:
        bf = brute_force.build(d_base, metric="sqeuclidean")
        _, nbrs = brute_force.search(bf, d_queries[:ne], a.k)
        truth = cp.asnumpy(cp.asarray(nbrs))

    t0 = time.time()
    params = ivf_pq.IndexParams(n_lists=1, metric="sqeuclidean", pq_bits=8, pq_dim=a.m, codebook_kind="subspace",
                                kmeans_trainset_fraction=a.train_fraction, force_random_rotation=False)
    index = ivf_pq.build(params, d_base)
    cp.cuda.runtime.deviceSynchronize()
    print(f"cuVS build (1 list, PQ{a.m}): {time.time() - t0:.1f} s")

    rows = []
    for lut_name, lut_dtype in [("fp32", np.float32), ("fp16", np.float16)]:
        sp = ivf_pq.SearchParams(n_probes=1, lut_dtype=lut_dtype, internal_distance_dtype=np.float32)
        ivf_pq.search(sp, index, d_queries, a.k)  # warm-up
        times = []
        for _ in range(a.reps):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            dist, nbrs = ivf_pq.search(sp, index, d_queries, a.k)
            stop.record()
            stop.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, stop))
        ms = float(np.median(times))
        found = cp.asnumpy(cp.asarray(nbrs))[:ne]
        rec = recall_at_k(found, truth, a.k)
        qps = nq / (ms / 1e3)
        print(f"cuvs_ivfpq_flat lut={lut_name}: {ms:9.2f} ms  {qps:10.0f} QPS  recall@{a.k}={rec:.3f}")
        rows.append(dict(label=a.label, variant=f"cuvs_ivfpq_flat_lut_{lut_name}", n=n, d=d, m=a.m, nq=nq, k=a.k,
                         total_ms=ms, qps=qps, recall=rec))

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
