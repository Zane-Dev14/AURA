#!/usr/bin/env python3
"""Summarize local AURA benchmark runs.

This tool scans benchmark JSON files produced by the local baseline, QMIX,
and HPA test scripts, then reports per-mode descriptive statistics,
bootstrap confidence intervals, and a permutation-based significance test
for pairwise comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


MODE_LABELS = {
    "baseline": "Baseline",
    "qmix": "QMIX",
    "hpa": "HPA",
}

METRICS = {
    "cluster.total_rps": "Throughput (RPS)",
    "services.api.p99_ms": "API P99 (ms)",
    "services.app.p99_ms": "APP P99 (ms)",
    "services.app.error_rate": "APP error rate",
    "cluster.total_cpu_used_cores": "Total CPU used (cores)",
    "services.api.replica_hours": "API replica-hours",
    "services.app.replica_hours": "APP replica-hours",
}


@dataclass
class TrialRecord:
    path: Path
    mode: str
    timestamp: str
    payload: dict


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dotted_get(payload: dict, dotted_path: str):
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def trial_value(trial: dict, dotted_path: str):
    value = dotted_get(trial, dotted_path)
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def gather_trials(root: Path):
    trials: dict[str, list[TrialRecord]] = defaultdict(list)

    for path in sorted(root.rglob("*_metrics_*.json")):
        if "combined_" in path.name:
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue

        mode = str(payload.get("mode", "")).lower().strip()
        if mode not in MODE_LABELS:
            continue

        trials[mode].append(
            TrialRecord(
                path=path,
                mode=mode,
                timestamp=str(payload.get("timestamp", "")),
                payload=payload,
            )
        )

    return trials


def bootstrap_mean_ci(values: list[float], iterations: int = 10_000, seed: int = 42):
    if len(values) < 2:
        return None

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = rng.choice(arr, size=(iterations, arr.size), replace=True)
    means = samples.mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def permutation_p_value(sample_a: list[float], sample_b: list[float], iterations: int = 20_000, seed: int = 42):
    if not sample_a or not sample_b:
        return None

    rng = np.random.default_rng(seed)
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    size_a = a.size

    more_extreme = 0
    for _ in range(iterations):
        permuted = rng.permutation(pooled)
        diff = float(permuted[:size_a].mean() - permuted[size_a:].mean())
        if abs(diff) >= abs(observed):
            more_extreme += 1

    return (more_extreme + 1) / (iterations + 1)


def describe(values: list[float]):
    if not values:
        return None

    avg = mean(values)
    spread = pstdev(values) if len(values) > 1 else 0.0
    ci = bootstrap_mean_ci(values)
    return {
        "n": len(values),
        "mean": avg,
        "std": spread,
        "ci": ci,
        "min": min(values),
        "max": max(values),
    }


def format_ci(ci):
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def print_mode_summary(mode: str, trials: list[TrialRecord]):
    print(f"\n{MODE_LABELS[mode]} ({len(trials)} runs)")
    print("-" * (len(MODE_LABELS[mode]) + 12))

    for metric_path, label in METRICS.items():
        values = []
        for trial in trials:
            value = trial_value(trial.payload, metric_path)
            if value is not None:
                values.append(value)

        stats = describe(values)
        if stats is None:
            continue

        print(
            f"{label:<24} n={stats['n']:<2d} "
            f"mean={stats['mean']:.3f} std={stats['std']:.3f} "
            f"ci95={format_ci(stats['ci'])}"
        )


def print_pairwise_summary(left: str, right: str, trials_by_mode: dict[str, list[TrialRecord]]):
    left_trials = trials_by_mode.get(left, [])
    right_trials = trials_by_mode.get(right, [])
    if not left_trials or not right_trials:
        return

    print(f"\n{MODE_LABELS[left]} vs {MODE_LABELS[right]}")
    print("-" * (len(MODE_LABELS[left]) + len(MODE_LABELS[right]) + 4))

    for metric_path, label in METRICS.items():
        left_values = [trial_value(trial.payload, metric_path) for trial in left_trials]
        right_values = [trial_value(trial.payload, metric_path) for trial in right_trials]
        left_values = [value for value in left_values if value is not None]
        right_values = [value for value in right_values if value is not None]

        if not left_values or not right_values:
            continue

        diff = mean(left_values) - mean(right_values)
        p_value = permutation_p_value(left_values, right_values)
        print(
            f"{label:<24} Δmean={diff:+.3f} "
            f"p≈{p_value:.4f} "
            f"({MODE_LABELS[left]} n={len(left_values)}, {MODE_LABELS[right]} n={len(right_values)})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="docs/Final Results",
        help="Directory to scan for benchmark JSON files.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    trials_by_mode = gather_trials(root)

    if not trials_by_mode:
        raise SystemExit(f"No benchmark JSON files found under {root}")

    print(f"Scanning {root}")
    print("=" * (len(str(root)) + 9))

    for mode in ("baseline", "qmix", "hpa"):
        if mode in trials_by_mode:
            print_mode_summary(mode, trials_by_mode[mode])

    print_pairwise_summary("qmix", "baseline", trials_by_mode)
    print_pairwise_summary("qmix", "hpa", trials_by_mode)


if __name__ == "__main__":
    main()
