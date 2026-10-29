#!/usr/bin/env python3
"""Figures for docs/RESULTS.md from the CSVs written by scripts/run_all.sh.

usage: python3 scripts/plot_results.py results/final --out docs/figures
Requires matplotlib and pandas (pip install matplotlib pandas). Missing CSVs are skipped.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ACCENT = "#2f6fdb"
GREY = "#9a9a9a"
LIGHT = "#d4d4d4"
STAGES = [("transfer_ms", "copies", LIGHT), ("lut_ms", "LUT", "#bcbcbc"), ("scan_ms", "scan", GREY),
          ("select_ms", "select", ACCENT)]
FUSED = ["v4_fused", "v5_fused_reg", "v6_fused_vec", "v7_fused_vec_mq", "v8_fused_warp"]


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)


def load(path):
    return pd.read_csv(path) if path.exists() else None


def short(v):
    return v.replace("_shared_lut", " shared LUT").replace("_global_lut", " global LUT").replace("_fused", " fused") \
            .replace("_vec_mq", " vec+MQ").replace("_vec", " vec").replace("_interleaved", " interleaved") \
            .replace("_reg", " reg").replace("_warp", " warp").replace(" vec", " +16B").replace("_", " ")


def fig_variants(res, out):
    df = load(res / "scaling.csv")
    if df is None:
        return
    n = df.n.min()
    part = df[(df.n == n) & (df.variant != "cpu_reference")].drop_duplicates("variant", keep="last")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(part) + 1.4))
    left = pd.Series(0.0, index=part.index)
    for col, name, color in STAGES:
        ax.barh(part.variant.map(short), part[col], left=left, color=color, label=name)
        left += part[col]
    # Label at the end of the stacked bar (sum of per-stage medians), showing the median total.
    for y, (end, total, qps) in enumerate(zip(left, part.total_ms, part.qps)):
        ax.text(end + left.max() * 0.01, y, f"{total:.1f} ms  ({qps:,.0f} q/s)", va="center", fontsize=8)
    ax.set_xlim(0, left.max() * 1.35)
    cuvs = load(res / "cuvs.csv")
    if cuvs is not None:
        c = cuvs[(cuvs.n == n) & cuvs.variant.str.endswith("fp32")]
        if len(c):
            ax.axvline(c.total_ms.iloc[-1], color="#444444", ls="--", lw=1)
            ax.text(c.total_ms.iloc[-1], len(part) - 0.45, f" cuVS (flat IVF-PQ, FP32 LUT) {c.total_ms.iloc[-1]:.1f} ms",
                    fontsize=8, color="#444444", va="top")
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower right")
    style(ax, f"Time per stage, {n/1e6:g}M vectors, {int(part.nq.iloc[0])} queries, k={int(part.k.iloc[0])}",
          "GPU time for all queries (ms)")
    fig.tight_layout()
    fig.savefig(out / "variants.png", dpi=170)
    plt.close(fig)


def fig_scaling(res, out):
    df = load(res / "scaling.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for v in ["cpu_reference", "v2_shared_lut_vec"] + FUSED:
        d = df[df.variant == v].sort_values("n")
        if len(d) < 2:
            continue
        best = v == "v8_fused_warp"
        shade = {"cpu_reference": "#cfcfcf", "v2_shared_lut_vec": "#b5b5b5", "v4_fused": "#9a9a9a",
                 "v5_fused_reg": "#808080", "v6_fused_vec": "#666666", "v7_fused_vec_mq": "#4d4d4d"}
        ax.plot(d.n, d.qps, marker="o", lw=2.2 if best else 1.2, color=ACCENT if best else shade.get(v, GREY),
                label=short(v))
    cuvs = load(res / "cuvs.csv")
    if cuvs is not None:
        c = cuvs[cuvs.label.str.startswith("synthetic") & cuvs.variant.str.endswith("fp32")].sort_values("n")
        if len(c):
            ax.plot(c.n, c.qps, marker="s", ls="--", color="#222222", lw=1, label="cuVS IVF-PQ (1 list)")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "Throughput vs database size (PQ16, k=10)", "vectors in the database", "queries per second")
    ax.grid(axis="y", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=170)
    plt.close(fig)


def fig_sweep(res, out):
    df = load(res / "sweep_q_ilp.csv")
    if df is None:
        return
    for n in sorted(df.n.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for ax, v in zip(axes, ["v7_fused_vec_mq", "v8_fused_warp"]):
            d = df[(df.n == n) & (df.variant == v)]
            # rows: requested Q (label) vs effective Q may differ; use the label's request
            d = d.assign(q_req=d.label.str.extract(r"_q(\d+)_")[0].astype(int))
            grid = d.pivot_table(index="ilp", columns="q_req", values="scan_ms", aggfunc="last")
            im = ax.imshow(grid.values, cmap="Blues_r", aspect="auto")
            ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
            ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
            best = grid.values.min()
            for (r, c), val in pd.DataFrame(grid.values).stack().items():
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if val == best else "normal",
                        color="white" if val < grid.values.mean() else "black")
            ax.set_xlabel("queries per block requested (Q)")
            ax.set_ylabel("vectors per step (ILP)")
            ax.set_title(f"{short(v)}: scan ms, n={n/1e6:g}M", fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(out / f"sweep_q_ilp_{int(n/1e6)}m.png", dpi=170)
        plt.close(fig)


def fig_blocks(res, out):
    df = load(res / "sweep_blocks.csv")
    if df is None:
        return
    for n in sorted(df.n.unique()):
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for v in ["v4_fused", "v6_fused_vec", "v8_fused_warp"]:
            d = df[(df.n == n) & (df.variant == v)].copy()
            d["bpq"] = d.label.str.extract(r"bpq(\d+)_")[0].astype(int)
            auto = d[d.bpq == 0]
            d = d[d.bpq > 0].sort_values("bpq")
            color = ACCENT if v == "v8_fused_warp" else GREY
            ax.plot(d.bpq, d.scan_ms, marker="o", color=color, label=short(v))
            if len(auto):
                ax.scatter([auto.blocks_x.iloc[-1]], [auto.scan_ms.iloc[-1]], marker="*", s=140, color=color,
                           zorder=5, edgecolor="black", linewidth=0.5)
        ax.set_xscale("log", base=2)
        ax.legend(frameon=False, fontsize=8)
        style(ax, f"Blocks per query, n={n/1e6:g}M (star = automatic choice)", "blocks per query", "scan ms")
        fig.tight_layout()
        fig.savefig(out / f"sweep_blocks_{int(n/1e6)}m.png", dpi=170)
        plt.close(fig)


def fig_latency(res, out):
    df = load(res / "latency.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for v in FUSED:
        d = df[df.variant == v].sort_values("nq")
        if not len(d):
            continue
        best = v == "v8_fused_warp"
        ax.plot(d.nq, d.total_ms / d.nq * 1e3, marker="o", color=ACCENT if best else GREY,
                lw=2.2 if best else 1.2, label=short(v))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    style(ax, f"Latency per query vs batch size (n={df.n.iloc[0]/1e6:g}M)", "queries per batch",
          "µs per query (total GPU time / queries)")
    ax.grid(axis="y", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=170)
    plt.close(fig)


def fig_sift(res, out):
    a, b = load(res / "sift1m.csv"), load(res / "sift1m_rerank.csv")
    if a is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for df, marker, tag in [(a, "o", "PQ only"), (b, "^", "rerank 100")]:
        if df is None:
            continue
        for v, color in [("cpu_reference", GREY), ("v4_fused", "#777777"), ("v8_fused_warp", ACCENT)]:
            d = df[df.variant == v].sort_values("m")
            if not len(d):
                continue
            ax.plot(d.recall, d.qps, marker=marker, color=color, lw=1, label=f"{short(v)}, {tag}")
            for _, r in d.iterrows():
                ax.annotate(f"m={int(r.m)}", (r.recall, r.qps), fontsize=7, xytext=(3, 3), textcoords="offset points")
    cuvs = load(res / "cuvs.csv")
    if cuvs is not None:
        c = cuvs[cuvs.label.str.startswith("sift") & cuvs.variant.str.endswith("fp32")]
        ax.scatter(c.recall, c.qps, marker="s", color="#444444", label="cuVS flat IVF-PQ, PQ only", zorder=5)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)
    style(ax, "SIFT1M: throughput vs recall@10", "recall@10", "queries per second")
    ax.grid(axis="y", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(out / "sift1m_recall.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results/final")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_variants, fig_scaling, fig_sweep, fig_blocks, fig_latency, fig_sift):
        try:
            f(res, out)
        except Exception as e:  # one bad CSV should not stop the other figures
            print(f"skipped {f.__name__}: {e}")
    print(f"figures in {out}/: {sorted(p.name for p in out.glob('*.png'))}")


if __name__ == "__main__":
    main()
