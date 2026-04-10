#!/usr/bin/env python3
"""Generate publication-quality paper figures from real experiment CSV files."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "Final Results"
OUTPUT_DIR = ROOT / "docs" / "figures"
SYSTEMS = {
    "qmix": {"label": "QMIX", "color": "#005f73", "ls": "-"},
    "hpa": {"label": "HPA", "color": "#bb3e03", "ls": "--"},
    "baseline": {"label": "Baseline", "color": "#6c757d", "ls": ":"},
}


def _to_minutes(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.notna().all():
        return ((ts - ts.min()).dt.total_seconds() / 60.0).to_numpy()
    return np.arange(len(series), dtype=float) * 0.5


def _add_phase_bands(ax: plt.Axes) -> None:
    phases = [
        (0, 10, "#f8f9fa", "Ramp"),
        (10, 20, "#eef5ff", "Steady"),
        (20, 25, "#fff3e6", "Spike"),
        (25, 30, "#f2fbe9", "Cooldown"),
    ]
    for start, end, color, _ in phases:
        ax.axvspan(start, end, color=color, zorder=0)


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Figure {name} was not generated correctly")
    print(f"[ok] {out.relative_to(ROOT)}")


def _plot_lines(ax: plt.Axes, source: dict, y_col: str, ylabel: str, step: bool = False) -> None:
    plotted = 0
    for key, cfg in SYSTEMS.items():
        df = source[key].copy()
        if "timestamp" not in df.columns or y_col not in df.columns:
            continue
        x = _to_minutes(df["timestamp"])
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        if np.isfinite(y).sum() == 0:
            continue
        if step:
            ax.step(x, y, where="post", label=cfg["label"], color=cfg["color"], lw=1.8)
        else:
            ax.plot(x, y, label=cfg["label"], color=cfg["color"], lw=1.9, ls=cfg["ls"])
        plotted += 1

    if plotted == 0:
        raise RuntimeError(f"No valid series found for column: {y_col}")

    _add_phase_bands(ax)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.92)


def load_data() -> dict:
    data = {}
    for system in SYSTEMS:
        data[system] = {
            "replicas": pd.read_csv(DATA_DIR / f"replicas_over_time_{system}.csv"),
            "p99": pd.read_csv(DATA_DIR / f"p99_over_time_{system}.csv"),
            "cpu": pd.read_csv(DATA_DIR / f"cpu_usage_over_time_{system}.csv"),
        }
    return data


def plot_api_replicas(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    _plot_lines(ax, {k: v["replicas"] for k, v in data.items()}, "api_replicas", "API Replicas", step=True)
    ax.set_ylim(0.8, 5.3)
    _save(fig, "api_replicas_comparison.pdf")


def plot_app_replicas(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    _plot_lines(ax, {k: v["replicas"] for k, v in data.items()}, "app_replicas", "APP Replicas", step=True)
    ax.set_ylim(0.8, 5.3)
    _save(fig, "app_replicas_comparison.pdf")


def plot_api_p99_latency(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    _plot_lines(ax, {k: v["p99"] for k, v in data.items()}, "api_p99_ms", "API P99 Latency (ms)")
    ax.set_ylim(0, 180)
    _save(fig, "api_p99_latency_comparison.pdf")


def plot_app_p99_latency(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    _plot_lines(ax, {k: v["p99"] for k, v in data.items()}, "app_p99_ms", "APP P99 Latency (ms)")
    ax.set_ylim(0, 1300)
    _save(fig, "app_p99_latency_comparison.pdf")


def plot_total_cpu_usage(data: dict) -> None:
    cpu_frames = {}
    for k, v in data.items():
        df = v["cpu"].copy()
        cols = ["api_cpu_cores", "app_cpu_cores", "db_cpu_cores"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing CPU columns for {k}: {missing}")
        df["total_cpu_cores"] = df[cols].sum(axis=1)
        cpu_frames[k] = df

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    _plot_lines(ax, cpu_frames, "total_cpu_cores", "Total CPU Usage (cores)")
    ax.set_ylim(0, 3.2)
    _save(fig, "total_cpu_usage_comparison.pdf")


def plot_combined_replicas(data: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True)
    for ax, service, label in zip(axes, ["api", "app", "db"], ["API", "APP", "DB"]):
        _plot_lines(
            ax,
            {k: v["replicas"] for k, v in data.items()},
            f"{service}_replicas",
            f"{label} Replicas",
            step=True,
        )
        ax.set_ylim(0.8, 5.3)
    axes[-1].set_xlabel("Time (minutes)")
    _save(fig, "all_replicas_comparison.pdf")


def plot_qmix_action_profile() -> None:
    path = DATA_DIR / "qmix_decisions_20260218_171840.csv"
    df = pd.read_csv(path, sep="|", engine="python")
    df.columns = [c.strip() for c in df.columns]
    for col in ["svc", "Δ"]:
        df[col] = df[col].astype(str).str.strip()

    service_order = ["api", "app", "db"]
    delta_order = ["-1", "+0", "+1"]
    pivot = (
        df[df["svc"].isin(service_order) & df["Δ"].isin(delta_order)]
        .groupby(["svc", "Δ"])  # type: ignore[arg-type]
        .size()
        .unstack(fill_value=0)
        .reindex(index=service_order, columns=delta_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    x = np.arange(len(service_order))
    width = 0.24
    colors = {"-1": "#ae2012", "+0": "#6c757d", "+1": "#0a9396"}
    for i, d in enumerate(delta_order):
        ax.bar(x + (i - 1) * width, pivot[d].to_numpy(), width, color=colors[d], label=f"Action {d}")

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in service_order])
    ax.set_ylabel("Decision Count")
    ax.set_xlabel("Service")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    _save(fig, "qmix_action_profile.pdf")


def main() -> None:
    print("Loading experiment data...")
    data = load_data()

    print("Generating upgraded figures...")
    plot_api_replicas(data)
    plot_app_replicas(data)
    plot_api_p99_latency(data)
    plot_app_p99_latency(data)
    plot_total_cpu_usage(data)
    plot_combined_replicas(data)
    plot_qmix_action_profile()

    print(f"All figures generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
