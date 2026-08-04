#!/usr/bin/env python3
"""Generate the project's reproducible benchmark visualizations."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CSV = ROOT / "docs/data/historical_benchmarks.csv"
SUMMARY_CSV = ROOT / "runpod_results/l4_eval_2026-08-04/summary.csv"
NSYS_DIR = ROOT / "runpod_results/l4_eval_2026-08-04/nsys"
DEFAULT_OUTPUT_DIR = ROOT / "docs/assets/performance"

BG = "#0B0F14"
PANEL = "#111820"
GRID = "#2A3441"
TEXT = "#F2F5F7"
MUTED = "#9AA7B4"
CYAN = "#35C7D4"
CORAL = "#FF6B6B"
AMBER = "#F2B84B"
GREEN = "#66D17A"
GRAY = "#7D8996"

COLORS = {
    "legacy": GRAY,
    "untiled": CORAL,
    "block-thread": CYAN,
    "vendor": AMBER,
    "direct": GREEN,
    "pytorch": GREEN,
}

FIGURES = (
    "01_optimization_milestones",
    "02_gemm_scaling",
    "03_operator_latency",
    "04_vendor_efficiency",
    "05_mixed_programs",
    "06_parameter_tuning",
    "07_nsys_bottlenecks",
)

BMM_NAMES = (
    ("bmm_bert", "BERT\n12x128x64x128"),
    ("bmm_long", "Long\n12x512x64x512"),
    ("bmm_value", "Value\n32x128x64x64"),
    ("bmm_irregular", "Irregular\n7x129x65x127"),
)

CONV_NAMES = (
    ("conv_small", "Small\n3x3"),
    ("conv_resnet_stem", "ResNet stem\n7x7"),
    ("conv_resnet_block", "ResNet block\n3x3"),
    ("conv_pointwise", "Pointwise\n1x1"),
    ("conv_irregular", "Irregular\n3x5"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if not value:
        raise ValueError(f"Missing {key!r} in row: {row}")
    return float(value)


class BenchmarkData:
    def __init__(self) -> None:
        self.historical = read_csv(HISTORICAL_CSV)
        self.summary = read_csv(SUMMARY_CSV)
        self.timing = [row for row in self.summary if row["series"] == "timing"]
        self.sweeps = [row for row in self.summary if row["series"] == "sweep"]

    def history(self, name: str, backend: str) -> dict[str, str]:
        rows = [
            row
            for row in self.historical
            if row["name"] == name and row["backend"] == backend
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected one historical row for {name}/{backend}, got {len(rows)}")
        return rows[0]

    def canonical(self, family: str, name: str, backend: str) -> dict[str, str]:
        source_mode = {
            "cublas-pedantic-fp32": "vendor",
            "cudnn-fp32": "vendor",
        }.get(backend)
        rows = [
            row
            for row in self.timing
            if row["family"] == family
            and row["name"] == name
            and row["backend"] == backend
            and (source_mode is None or row["source_mode"] == source_mode)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one timing row for {family}/{name}/{backend}, got {len(rows)}"
            )
        return rows[0]

    def sweep(self, family: str, name: str, source_mode: str) -> dict[str, str]:
        rows = [
            row
            for row in self.sweeps
            if row["family"] == family
            and row["name"] == name
            and row["backend"] == "block-thread"
            and row["source_mode"] == source_mode
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one sweep row for {family}/{name}/{source_mode}, got {len(rows)}"
            )
        return rows[0]


def normalize_xyz(value: str) -> str:
    return " ".join(value.split())


def nsys_profile(label: str) -> dict[str, float]:
    kernel_rows = read_csv(NSYS_DIR / f"{label}_stats_cuda_gpu_kern_gb_sum.csv")
    api_rows = read_csv(NSYS_DIR / f"{label}_stats_cuda_api_sum.csv")
    total_ns = sum(as_float(row, "Total Time (ns)") for row in kernel_rows)
    kernel_count = sum(int(row["Instances"]) for row in kernel_rows)

    if label == "attention_block-thread":
        operator_ns = sum(
            as_float(row, "Total Time (ns)")
            for row in kernel_rows
            if row["Name"] == "attention_block_kernel"
            and normalize_xyz(row["BlockXYZ"]) != "1 1 1"
        )
    elif label == "attention_vendor":
        operator_ns = sum(
            as_float(row, "Total Time (ns)")
            for row in kernel_rows
            if row["Name"] != "attention_block_kernel"
        )
    elif label == "residual_conv_block-thread":
        operator_ns = sum(
            as_float(row, "Total Time (ns)")
            for row in kernel_rows
            if row["Name"] == "residual_conv_block_kernel"
            and normalize_xyz(row["BlockXYZ"]) == "256 1 1"
        )
    elif label == "residual_conv_vendor":
        operator_ns = sum(
            as_float(row, "Total Time (ns)")
            for row in kernel_rows
            if row["Name"] != "residual_conv_block_kernel"
        )
    else:
        raise ValueError(f"Unsupported Nsight profile: {label}")

    sync_ns = sum(
        as_float(row, "Total Time (ns)")
        for row in api_rows
        if row["Name"] == "cuStreamSynchronize"
    )
    return {
        "operator_ms": operator_ns / 1e6,
        "generic_ms": (total_ns - operator_ns) / 1e6,
        "total_ms": total_ns / 1e6,
        "sync_ms": sync_ns / 1e6,
        "kernel_count": float(kernel_count),
    }


def validate_data(data: BenchmarkData) -> None:
    history_keys = [(row["run"], row["name"], row["backend"]) for row in data.historical]
    if len(history_keys) != len(set(history_keys)):
        raise ValueError("Historical benchmark keys are not unique")
    for row in data.historical:
        if as_float(row, "p50_ms") <= 0:
            raise ValueError(f"Historical timing must be positive: {row}")

    summary_keys = [
        (
            row["series"],
            row["family"],
            row["name"],
            row["backend"],
            row["source_mode"],
        )
        for row in data.summary
    ]
    if len(summary_keys) != len(set(summary_keys)):
        raise ValueError("Unified benchmark keys are not unique")
    for row in data.summary:
        p10 = as_float(row, "p10_ms")
        p50 = as_float(row, "p50_ms")
        p90 = as_float(row, "p90_ms")
        if not 0 < p10 <= p50 <= p90:
            raise ValueError(f"Invalid percentile ordering: {row}")

    for name, _ in BMM_NAMES:
        for backend in ("untiled", "block-thread", "vendor", "cublas-pedantic-fp32"):
            data.canonical("bmm", name, backend)
    for name, _ in CONV_NAMES:
        for backend in ("untiled", "block-thread", "vendor", "cudnn-fp32"):
            data.canonical("conv", name, backend)
    for name in ("attention_block", "residual_conv_block"):
        family = "attention" if name == "attention_block" else "residual"
        for backend in ("untiled", "block-thread", "vendor", "pytorch-eager-cuda"):
            data.canonical(family, name, backend)

    cpu_speedup = as_float(data.history("mobilenet_3136x32x64", "untiled-cpu"), "p50_ms") / as_float(
        data.history("mobilenet_3136x32x64", "tiled-cpu"), "p50_ms"
    )
    gemm_speedup = as_float(data.history("gemm_512", "untiled"), "p50_ms") / as_float(
        data.history("gemm_512", "block-thread"), "p50_ms"
    )
    bmm_speedup = as_float(data.canonical("bmm", "bmm_long", "untiled"), "p50_ms") / as_float(
        data.canonical("bmm", "bmm_long", "block-thread"), "p50_ms"
    )
    conv_speedup = as_float(
        data.canonical("conv", "conv_resnet_block", "untiled"), "p50_ms"
    ) / as_float(data.canonical("conv", "conv_resnet_block", "block-thread"), "p50_ms")
    expected = ((cpu_speedup, 1.82), (gemm_speedup, 50.1), (bmm_speedup, 68.72), (conv_speedup, 38.63))
    for actual, target in expected:
        if not math.isclose(actual, target, rel_tol=0.01):
            raise ValueError(f"Headline speedup changed: expected {target}, got {actual}")

    for label in (
        "attention_block-thread",
        "attention_vendor",
        "residual_conv_block-thread",
        "residual_conv_vendor",
    ):
        metrics = nsys_profile(label)
        if not 0 < metrics["operator_ms"] <= metrics["total_ms"]:
            raise ValueError(f"Invalid Nsight breakdown for {label}: {metrics}")


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.6,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
            "legend.labelcolor": TEXT,
            "svg.hashsalt": "mlir-project-performance",
            "svg.fonttype": "none",
        }
    )


def make_figure(*, ncols: int = 1, width_ratios: list[float] | None = None):
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(10, 5.625),
        dpi=160,
        facecolor=BG,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
    )
    if ncols == 1:
        axes = [axes]
    for axis in axes:
        axis.set_facecolor(PANEL)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)
        axis.grid(axis="y", zorder=0)
        axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.20, wspace=0.25)
    return fig, axes


def heading(fig, title: str, subtitle: str, source: str) -> None:
    fig.text(0.05, 0.94, title, fontsize=22, fontweight="bold", color=TEXT)
    fig.text(0.05, 0.885, subtitle, fontsize=10.5, color=MUTED)
    fig.text(0.05, 0.045, source, fontsize=7.5, color=MUTED)


def format_ms(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def save_figure(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{name}.svg",
        format="svg",
        facecolor=BG,
        edgecolor=BG,
        transparent=False,
        metadata={"Date": None, "Creator": "plot_project_performance.py"},
    )
    fig.savefig(
        output_dir / f"{name}.png",
        format="png",
        dpi=160,
        facecolor=BG,
        edgecolor=BG,
        transparent=False,
        metadata={"Software": "plot_project_performance.py"},
    )
    plt.close(fig)


def plot_milestones(data: BenchmarkData, output_dir: Path) -> None:
    cpu = as_float(data.history("mobilenet_3136x32x64", "untiled-cpu"), "p50_ms") / as_float(
        data.history("mobilenet_3136x32x64", "tiled-cpu"), "p50_ms"
    )
    gemm = as_float(data.history("gemm_512", "untiled"), "p50_ms") / as_float(
        data.history("gemm_512", "block-thread"), "p50_ms"
    )
    bmm = as_float(data.canonical("bmm", "bmm_long", "untiled"), "p50_ms") / as_float(
        data.canonical("bmm", "bmm_long", "block-thread"), "p50_ms"
    )
    conv = as_float(data.canonical("conv", "conv_resnet_block", "untiled"), "p50_ms") / as_float(
        data.canonical("conv", "conv_resnet_block", "block-thread"), "p50_ms"
    )
    attention = as_float(data.canonical("attention", "attention_block", "untiled"), "p50_ms") / as_float(
        data.canonical("attention", "attention_block", "block-thread"), "p50_ms"
    )
    residual = as_float(data.canonical("residual", "residual_conv_block", "untiled"), "p50_ms") / as_float(
        data.canonical("residual", "residual_conv_block", "block-thread"), "p50_ms"
    )
    labels = [
        "CPU cache tiling",
        "GPU GEMM mapping",
        "GPU batch matmul",
        "GPU convolution",
        "Attention block",
        "Residual block",
    ]
    values = [cpu, gemm, bmm, conv, attention, residual]
    colors = [AMBER, CYAN, CYAN, CYAN, GREEN, GREEN]

    fig, (ax,) = make_figure()
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.58, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.9, 110)
    ax.set_xlabel("Speedup over the stage-specific untiled baseline (log scale)")
    ax.axvline(1, color=GRAY, linewidth=1.2)
    ax.axvline(10, color=AMBER, linewidth=1, linestyle="--", alpha=0.75)
    ax.text(10.4, 5.35, "10x operator gate", color=AMBER, fontsize=8)
    for index, value in enumerate(values):
        detail = f"{value:.2f}x"
        if index == 1:
            detail += "  (410.6x vs legacy)"
        ax.text(value * 1.08, index, detail, va="center", color=TEXT, fontweight="bold", fontsize=9)
    heading(
        fig,
        "From correct lowering to meaningful GPU speedups",
        "Representative gains at each project stage; workloads and harnesses differ across stages.",
        "Sources: PR1 CPU/OpenBLAS and PR2 GEMM reports; unified NVIDIA L4 run for BMM, convolution, and mixed programs.",
    )
    save_figure(fig, output_dir, FIGURES[0])


def plot_gemm_scaling(data: BenchmarkData, output_dir: Path) -> None:
    names = ("gemm_512", "gemm_1024", "gemm_2048", "gemm_mobilenet", "gemm_irregular")
    labels = ("512x256x512", "1024^3", "2048^3", "3136x32x64", "513x257x509")
    series = (
        ("legacy", "Legacy", "o", "--"),
        ("untiled", "Untiled", "s", "-"),
        ("block-thread", "Block/thread", "D", "-"),
        ("cublas", "cuBLAS", "^", "-"),
    )
    x = np.arange(len(names))
    fig, (ax,) = make_figure()
    for backend, label, marker, linestyle in series:
        values = []
        for name in names:
            rows = [
                row
                for row in data.historical
                if row["name"] == name and row["backend"] == backend
            ]
            values.append(as_float(rows[0], "p50_ms") if rows else np.nan)
        ax.plot(
            x,
            values,
            label=label,
            marker=marker,
            linestyle=linestyle,
            color=COLORS.get(backend, GREEN),
            linewidth=2.2,
            markersize=7,
            zorder=3,
        )
    for index, name in enumerate(names):
        untiled = as_float(data.history(name, "untiled"), "p50_ms")
        custom = as_float(data.history(name, "block-thread"), "p50_ms")
        ax.text(index, custom * 1.8, f"{untiled / custom:.1f}x", color=CYAN, ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Device execution p50 (ms, log scale)")
    ax.set_xticks(x, labels)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    heading(
        fig,
        "Block/thread mapping scales across GEMM shapes",
        "One custom launch replaces the pathological launch topology while cuBLAS remains the optimization ceiling.",
        "Source: PR2 NVIDIA L4 run, CUDA 12.8.1. Cyan labels show block/thread speedup over untiled lowering.",
    )
    save_figure(fig, output_dir, FIGURES[1])


def plot_operator_latency(data: BenchmarkData, output_dir: Path) -> None:
    fig, axes = make_figure(ncols=2)
    configurations = (
        ("untiled", "Untiled", "untiled"),
        ("block-thread", "Block/thread", "block-thread"),
        ("vendor", "Vendor call", "vendor"),
        ("direct", "Direct library", None),
    )
    panels = (
        (axes[0], "bmm", BMM_NAMES, "cublas-pedantic-fp32", "Batch matmul"),
        (axes[1], "conv", CONV_NAMES, "cudnn-fp32", "Convolution"),
    )
    for ax, family, names, direct_backend, panel_title in panels:
        x = np.arange(len(names))
        width = 0.19
        for series_index, (key, label, backend) in enumerate(configurations):
            actual_backend = direct_backend if key == "direct" else backend
            rows = [data.canonical(family, name, actual_backend) for name, _ in names]
            values = np.array([as_float(row, "p50_ms") for row in rows])
            lower = values - np.array([as_float(row, "p10_ms") for row in rows])
            upper = np.array([as_float(row, "p90_ms") for row in rows]) - values
            positions = x + (series_index - 1.5) * width
            ax.bar(
                positions,
                values,
                width,
                label=label,
                color=COLORS[key],
                zorder=3,
                yerr=np.vstack([lower, upper]),
                error_kw={"ecolor": TEXT, "elinewidth": 0.7, "capsize": 1.5, "alpha": 0.8},
            )
        for index, (name, _) in enumerate(names):
            untiled = as_float(data.canonical(family, name, "untiled"), "p50_ms")
            custom = as_float(data.canonical(family, name, "block-thread"), "p50_ms")
            ax.text(index, untiled * 1.35, f"{untiled / custom:.1f}x", color=CYAN, ha="center", fontsize=7.5)
        ax.set_yscale("log")
        ax.set_title(panel_title, loc="left", fontweight="bold")
        ax.set_xticks(x, [label for _, label in names], fontsize=7.5)
        ax.set_ylabel("p50 latency (ms, log scale)")
    axes[0].legend(ncols=2, loc="upper left", fontsize=8)
    heading(
        fig,
        "Custom GPU mapping delivers broad operator gains",
        "Whiskers show canonical p10-p90 device time; cyan labels are speedup over untiled lowering.",
        "Source: unified 2026-08-04 NVIDIA L4 evaluation, three process-level trials per mode.",
    )
    save_figure(fig, output_dir, FIGURES[2])


def plot_vendor_efficiency(data: BenchmarkData, output_dir: Path) -> None:
    entries: list[tuple[str, float, str]] = []
    for family, names, direct_backend in (
        ("bmm", BMM_NAMES, "cublas-pedantic-fp32"),
        ("conv", CONV_NAMES, "cudnn-fp32"),
    ):
        for name, label in names:
            custom = as_float(data.canonical(family, name, "block-thread"), "gflops")
            direct = as_float(data.canonical(family, name, direct_backend), "gflops")
            entries.append((label.replace("\n", " "), 100 * custom / direct, family))
    labels = [entry[0] for entry in entries]
    values = [entry[1] for entry in entries]
    colors = [CYAN if entry[2] == "bmm" else AMBER for entry in entries]
    hatches = ["//" if entry[2] == "bmm" else "" for entry in entries]

    fig, (ax,) = make_figure()
    y = np.arange(len(entries))
    bars = ax.barh(y, values, color=colors, height=0.62, zorder=3)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        bar.set_edgecolor(BG)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_xlabel("Custom throughput as a percentage of direct vendor-library throughput")
    ax.axvline(100, color=GREEN, linewidth=1.4, linestyle="--")
    ax.text(101, len(entries) - 0.2, "vendor parity", color=GREEN, fontsize=8)
    for index, value in enumerate(values):
        ax.text(value + 2, index, f"{value:.0f}%", va="center", fontsize=8, fontweight="bold")
    ax.text(0.72, 1.04, "BMM", color=CYAN, transform=ax.transAxes, fontweight="bold")
    ax.text(0.82, 1.04, "CONV", color=AMBER, transform=ax.transAxes, fontweight="bold")
    heading(
        fig,
        "The vendor gap depends strongly on workload shape",
        "Small and pointwise convolutions reach or exceed cuDNN FP32 throughput; compute-heavy kernels still have headroom.",
        "Source: unified NVIDIA L4 GFLOP/s. Direct baselines use the vendor-mode cuBLAS/cuDNN reference measurement.",
    )
    save_figure(fig, output_dir, FIGURES[3])


def plot_mixed_programs(data: BenchmarkData, output_dir: Path) -> None:
    programs = (
        ("attention", "attention_block", "Attention block"),
        ("residual", "residual_conv_block", "Residual block"),
    )
    series = (
        ("untiled", "Untiled", "untiled"),
        ("block-thread", "Block/thread", "block-thread"),
        ("vendor", "Vendor calls", "vendor"),
        ("pytorch", "PyTorch eager", "pytorch-eager-cuda"),
    )
    x = np.arange(len(programs))
    width = 0.19
    fig, (ax,) = make_figure()
    for series_index, (key, label, backend) in enumerate(series):
        rows = [data.canonical(family, name, backend) for family, name, _ in programs]
        values = np.array([as_float(row, "p50_ms") for row in rows])
        lower = values - np.array([as_float(row, "p10_ms") for row in rows])
        upper = np.array([as_float(row, "p90_ms") for row in rows]) - values
        ax.bar(
            x + (series_index - 1.5) * width,
            values,
            width,
            label=label,
            color=COLORS[key],
            zorder=3,
            yerr=np.vstack([lower, upper]),
            error_kw={"ecolor": TEXT, "elinewidth": 0.8, "capsize": 2, "alpha": 0.8},
        )
    for index, (family, name, _) in enumerate(programs):
        untiled = as_float(data.canonical(family, name, "untiled"), "p50_ms")
        custom = as_float(data.canonical(family, name, "block-thread"), "p50_ms")
        ax.text(index, untiled * 1.12, f"custom: {untiled / custom:.2f}x", color=CYAN, ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 6)
    ax.set_ylabel("p50 latency (ms, log scale)")
    ax.set_xticks(x, [label for _, _, label in programs])
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
    heading(
        fig,
        "Isolated operator gains only partially transfer to mixed programs",
        "Generic surrounding kernels dominate both blocks; PyTorch eager remains a fused-runtime reference, not an equivalent lowering path.",
        "Source: unified NVIDIA L4 evaluation. Whiskers show canonical p10-p90 device time.",
    )
    save_figure(fig, output_dir, FIGURES[4])


def plot_parameter_tuning(data: BenchmarkData, output_dir: Path) -> None:
    fig, axes = make_figure(ncols=2)
    cmap = LinearSegmentedColormap.from_list("dark_cyan", [PANEL, "#164D57", CYAN])
    panels = (
        (
            axes[0],
            "bmm",
            ("bmm_bert", "bmm_long"),
            ("BERT BMM", "Long BMM"),
            ("tile-8x32", "tile-16x16", "tile-16x32", "tile-32x8"),
            ("8x32", "16x16", "16x32", "32x8"),
            "BMM tile",
        ),
        (
            axes[1],
            "conv",
            ("conv_resnet_block", "conv_pointwise"),
            ("ResNet 3x3", "Pointwise 1x1"),
            ("threads-64", "threads-128", "threads-256", "threads-512"),
            ("64", "128", "256", "512"),
            "Threads",
        ),
    )
    for ax, family, names, column_labels, modes, row_labels, title in panels:
        actual = np.array(
            [
                [as_float(data.sweep(family, name, mode), "p50_ms") for name in names]
                for mode in modes
            ]
        )
        normalized = actual / actual.min(axis=0)
        ax.imshow(normalized, cmap=cmap, vmin=1.0, vmax=max(1.15, float(normalized.max())), aspect="auto")
        ax.set_xticks(np.arange(len(names)), column_labels)
        ax.set_yticks(np.arange(len(modes)), row_labels)
        ax.set_title(title, loc="left", fontweight="bold")
        for row_index in range(actual.shape[0]):
            for column_index in range(actual.shape[1]):
                best = math.isclose(normalized[row_index, column_index], 1.0, rel_tol=1e-9)
                label = f"{actual[row_index, column_index]:.3f} ms\n{normalized[row_index, column_index]:.2f}x"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=BG if best else TEXT,
                    fontweight="bold" if best else "normal",
                )
        ax.grid(False)
    heading(
        fig,
        "Tuning is shape-dependent, but defaults remain robust",
        "Each cell shows p50 latency and slowdown relative to the best configuration for that workload.",
        "Source: unified NVIDIA L4 tuning sweep, 5 warmups and 20 timed samples per configuration.",
    )
    save_figure(fig, output_dir, FIGURES[5])


def plot_nsys_bottlenecks(output_dir: Path) -> None:
    profiles = (
        ("attention_block-thread", "Attention\ncustom"),
        ("attention_vendor", "Attention\nvendor"),
        ("residual_conv_block-thread", "Residual\ncustom"),
        ("residual_conv_vendor", "Residual\nvendor"),
    )
    metrics = [nsys_profile(label) for label, _ in profiles]
    labels = [label for _, label in profiles]
    operator = np.array([entry["operator_ms"] for entry in metrics])
    generic = np.array([entry["generic_ms"] for entry in metrics])
    total = operator + generic
    operator_percent = 100 * operator / total
    generic_percent = 100 - operator_percent

    fig, axes = make_figure(ncols=2, width_ratios=[1.05, 1.0])
    y = np.arange(len(profiles))
    axes[0].barh(y, operator_percent, color=CYAN, label="Mapped operator", zorder=3)
    axes[0].barh(y, generic_percent, left=operator_percent, color=CORAL, label="Generic kernels", zorder=3)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Share of GPU kernel time")
    axes[0].set_title("Where GPU time goes", loc="left", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8)
    for index, value in enumerate(operator_percent):
        axes[0].text(max(value + 1.5, 7), index, f"operator {value:.1f}%", va="center", fontsize=8, fontweight="bold")

    axes[1].barh(y, total, color=GRAY, height=0.55, label="GPU kernels", zorder=3)
    sync = np.array([entry["sync_ms"] for entry in metrics])
    axes[1].scatter(sync, y, color=AMBER, marker="D", s=45, label="Stream sync", zorder=4)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("One-shot traced time (ms)")
    axes[1].set_title("Serialization tracks GPU work", loc="left", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8)
    for index, entry in enumerate(metrics):
        axes[1].text(
            entry["total_ms"] + 0.05,
            index,
            f"{int(entry['kernel_count'])} kernels",
            va="center",
            fontsize=8,
            color=TEXT,
        )
    heading(
        fig,
        "Nsight Systems reveals the next whole-model bottleneck",
        "Mapped BMM/convolution kernels are a small fraction of mixed-program GPU time; synchronization serializes the remainder.",
        "Source: Nsight Systems 2024.6.2 correctness-only traces. Diagnostic one-shot values are not formal benchmark timings.",
    )
    save_figure(fig, output_dir, FIGURES[6])


def verify_outputs(output_dir: Path) -> None:
    expected = {output_dir / f"{name}.{extension}" for name in FIGURES for extension in ("png", "svg")}
    missing = [path for path in sorted(expected) if not path.is_file()]
    if missing:
        raise ValueError(f"Missing generated figures: {missing}")
    for name in FIGURES:
        png = output_dir / f"{name}.png"
        with png.open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG header: {png}")
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        if (width, height) != (1600, 900):
            raise ValueError(f"Unexpected PNG dimensions for {png}: {(width, height)}")
        if png.stat().st_size < 20_000:
            raise ValueError(f"PNG appears blank or truncated: {png}")
        ET.parse(output_dir / f"{name}.svg")


def generate(output_dir: Path) -> None:
    configure_style()
    data = BenchmarkData()
    validate_data(data)
    plot_milestones(data, output_dir)
    plot_gemm_scaling(data, output_dir)
    plot_operator_latency(data, output_dir)
    plot_vendor_efficiency(data, output_dir)
    plot_mixed_programs(data, output_dir)
    plot_parameter_tuning(data, output_dir)
    plot_nsys_bottlenecks(output_dir)
    verify_outputs(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source data without generating figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = BenchmarkData()
    validate_data(data)
    if args.check_only:
        print("benchmark visualization data: valid")
        return
    generate(args.output_dir.resolve())
    print(f"generated {len(FIGURES)} SVG/PNG figure pairs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
