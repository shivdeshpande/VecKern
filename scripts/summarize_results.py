#!/usr/bin/env python3
"""Turn the campaign CSVs into the markdown tables used in docs/RESULTS.md, so no number is copied by hand.

usage: python3 scripts/summarize_results.py results/final > docs/results_tables.md
Requires pandas.
"""
import sys
from pathlib import Path

import pandas as pd

ORDER = ["cpu_reference", "v0_global_lut", "v1_shared_lut", "v2_shared_lut_vec", "v3_interleaved", "v4_fused",
         "v5_fused_reg", "v6_fused_vec", "v7_fused_vec_mq", "v8_fused_warp"]


def load(res, name):
    p = res / name
    return pd.read_csv(p) if p.exists() else None


def table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(c(r) if callable(c) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def ordered(df):
    df = df.copy()
    df["_o"] = df.variant.map({v: i for i, v in enumerate(ORDER)}).fillna(99)
    return df.sort_values(["_o"]).drop(columns="_o")


def fmt(x, nd=1):
    return "" if pd.isna(x) or x == 0 else f"{x:,.{nd}f}"


def section_variants(res):
    df = load(res, "scaling.csv")
    if df is None:
        return
    n = df.n.min()
    d = ordered(df[df.n == n].drop_duplicates("variant", keep="last"))
    v0 = d[d.variant == "v0_global_lut"].total_ms
    base = float(v0.iloc[0]) if len(v0) else None
    print(f"### Kernel versions, synthetic {n/1e6:g}M x {int(d.d.iloc[0])}, PQ{int(d.m.iloc[0])}, "
          f"{int(d.nq.iloc[0])} queries, k={int(d.k.iloc[0])}\n")
    print(table(d, ["variant", lambda r: fmt(r.scan_ms, 2), lambda r: fmt(r.select_ms, 2),
                    lambda r: fmt(r.total_ms, 2), lambda r: fmt(r.qps, 0),
                    lambda r: f"{base / r.total_ms:.2f}x" if base and r.variant != "cpu_reference" else "",
                    lambda r: f"{r.recall:.3f}" if r.recall >= 0 else ""],
                ["variant", "scan ms", "select ms", "total ms", "queries/s", "vs v0", "recall@10"]))
    print()


def section_scaling(res):
    df = load(res, "scaling.csv")
    if df is None:
        return
    piv = df.pivot_table(index="variant", columns="n", values="qps", aggfunc="last")
    piv = piv.reindex([v for v in ORDER if v in piv.index])
    cuvs = load(res, "cuvs.csv")
    print("### Throughput (queries/s) vs database size, synthetic, PQ16, 1,000 queries, k=10\n")
    headers = ["variant"] + [f"{int(n/1e6)}M" if n >= 1e6 else f"{int(n/1e3)}K" for n in piv.columns]
    rows = [[v] + [fmt(x, 0) for x in piv.loc[v]] for v in piv.index]
    if cuvs is not None:
        c = cuvs[cuvs.label.str.startswith("synthetic") & cuvs.variant.str.endswith("fp32")]
        rows.append(["cuVS IVF-PQ (1 list, FP32 LUT)"] +
                    [fmt(c[c.n == n].qps.iloc[-1], 0) if (c.n == n).any() else "" for n in piv.columns])
        c16 = cuvs[cuvs.label.str.startswith("synthetic") & cuvs.variant.str.endswith("fp16")]
        rows.append(["cuVS IVF-PQ (1 list, FP16 LUT)"] +
                    [fmt(c16[c16.n == n].qps.iloc[-1], 0) if (c16.n == n).any() else "" for n in piv.columns])
    print("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def section_sift(res):
    a, b, cuvs = load(res, "sift1m.csv"), load(res, "sift1m_rerank.csv"), load(res, "cuvs.csv")
    if a is None:
        return
    print("### SIFT1M (official ground truth, recall@10)\n")
    rows = []
    for m in sorted(a.m.unique()):
        d = a[a.m == m]
        for v in ["cpu_reference", "v4_fused", "v7_fused_vec_mq", "v8_fused_warp"]:
            r = d[d.variant == v]
            if len(r):
                r = r.iloc[-1]
                rows.append([str(int(m)), v, "PQ only", fmt(r.total_ms, 2), fmt(r.qps, 0), f"{r.recall:.3f}"])
        if b is not None:
            for v in ["v8_fused_warp"]:
                r = b[(b.m == m) & (b.variant == v)]
                if len(r):
                    r = r.iloc[-1]
                    rows.append([str(int(m)), v, "+ exact rerank of 100 (CPU, not timed)", fmt(r.total_ms, 2),
                                 fmt(r.qps, 0), f"{r.recall:.3f}"])
        if cuvs is not None:
            for _, r in cuvs[cuvs.label.str.startswith("sift") & (cuvs.m == m)].iterrows():
                rows.append([str(int(m)), r.variant, "PQ only (own codebooks)", fmt(r.total_ms, 2), fmt(r.qps, 0),
                             f"{r.recall:.3f}"])
    print("| m | variant | mode | total ms | queries/s | recall@10 |\n| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def section_latency(res):
    df = load(res, "latency.csv")
    if df is None:
        return
    df = df.assign(us_per_query=df.total_ms / df.nq * 1e3)
    piv = df.pivot_table(index="variant", columns="nq", values="total_ms", aggfunc="last")
    piv = piv.reindex([v for v in ORDER if v in piv.index])
    print(f"### Latency: total GPU time (ms) for a batch of nq queries, n={df.n.iloc[0]/1e6:g}M\n")
    headers = ["variant"] + [f"nq={int(c)}" for c in piv.columns]
    print("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |")
    for v in piv.index:
        print("| " + " | ".join([v] + [fmt(x, 3) for x in piv.loc[v]]) + " |")
    print()


def section_sweeps(res):
    q = load(res, "sweep_q_ilp.csv")
    if q is not None:
        q = q.assign(q_req=q.label.str.extract(r"_q(\d+)_")[0].astype(int))
        print("### Sweep: queries per block (Q) x vectors per step (ILP), scan ms (best per row in bold)\n")
        for n in sorted(q.n.unique()):
            for v in ["v7_fused_vec_mq", "v8_fused_warp"]:
                d = q[(q.n == n) & (q.variant == v)]
                if not len(d):
                    continue
                piv = d.pivot_table(index="ilp", columns="q_req", values="scan_ms", aggfunc="last")
                best = piv.values.min()
                print(f"{v}, n={n/1e6:g}M (effective Q may be capped; see blocks/registers in the CSV)\n")
                headers = ["ILP \\ Q"] + [str(c) for c in piv.columns]
                print("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |")
                for ilp in piv.index:
                    cells = [f"**{x:.2f}**" if x == best else f"{x:.2f}" for x in piv.loc[ilp]]
                    print("| " + " | ".join([str(ilp)] + cells) + " |")
                print()
    b = load(res, "sweep_blocks.csv")
    if b is not None:
        b = b.assign(bpq=b.label.str.extract(r"bpq(\d+)_")[0].astype(int))
        print("### Sweep: blocks per query (0 = automatic wave model), scan ms\n")
        for n in sorted(b.n.unique()):
            piv = b[b.n == n].pivot_table(index="variant", columns="bpq", values="scan_ms", aggfunc="last")
            auto_bx = b[(b.n == n) & (b.bpq == 0)].groupby("variant").blocks_x.last()
            headers = ["variant (n=%gM)" % (n / 1e6)] + ["auto" if c == 0 else str(c) for c in piv.columns]
            print("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |")
            for v in piv.index:
                cells = [fmt(x, 2) for x in piv.loc[v]]
                if v in auto_bx:
                    cells[0] += f" (bx={int(auto_bx[v])})"
                print("| " + " | ".join([v] + cells) + " |")
            print()
    t = load(res, "sweep_threads.csv")
    if t is not None:
        t = t.assign(threads_req=t.label.str.extract(r"threads(\d+)")[0].astype(int))
        print("### Sweep: v2 scan block size (1M vectors)\n")
        print("| threads per block | scan ms |\n| --- | --- |")
        for _, r in t.sort_values("threads_req").iterrows():
            print(f"| {r.threads_req} | {r.scan_ms:.2f} |")
        print()


def section_ncu(res):
    for name in ["ncu_1m.csv", "ncu_10m.csv"]:
        df = load(res, name)
        if df is None:
            continue
        df = df[df.kernel.str.contains("scan|fused")]
        piv = df.pivot_table(index="variant", columns="metric", values="value", aggfunc="first")
        get = lambda r, m: r.get(m, float("nan"))
        print(f"### Nsight Compute, main kernel per variant (n={df.n.iloc[0]/1e6:g}M, nq={df.nq.iloc[0]})\n")
        print("| variant | kernel ms | regs | achieved occ. % | local-mem loads GB | smem wavefronts (G) | "
              "bank conflicts (G) | L1/smem pipe % | DRAM read GB | L2 hit % |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for v in sorted(piv.index):
            r = piv.loc[v]
            print(f"| {v} | {get(r, 'gpu__time_duration.sum') / 1e6:.2f} | {get(r, 'launch__registers_per_thread'):.0f} | "
                  f"{get(r, 'sm__warps_active.avg.pct_of_peak_sustained_active'):.1f} | "
                  f"{get(r, 'l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum') / 1e9:.2f} | "
                  f"{get(r, 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum') / 1e9:.2f} | "
                  f"{get(r, 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum') / 1e9:.2f} | "
                  f"{get(r, 'l1tex__throughput.avg.pct_of_peak_sustained_active'):.1f} | "
                  f"{get(r, 'dram__bytes_read.sum') / 1e9:.2f} | {get(r, 'lts__t_sector_hit_rate.pct'):.1f} |")
        print("\nNote: unfused variants (v0-v3) launch once per batch of queries; their row is the first batch only.\n")


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/final")
    for f in (section_variants, section_scaling, section_sift, section_latency, section_sweeps, section_ncu):
        try:
            f(res)
        except Exception as e:
            print(f"<!-- {f.__name__} skipped: {e} -->\n")


if __name__ == "__main__":
    main()
