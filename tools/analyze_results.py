#!/usr/bin/env python3
"""
FGCS Results Analysis Tool

Performs statistical analysis on benchmark results:
- Computes mean ± std for all metrics
- Performs paired t-tests
- Computes effect sizes (Cohen's d)
- Generates comparison tables
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd

def load_trial_metrics(results_dir, mode, trial):
    """Load metrics from a single trial"""
    path = Path(results_dir) / mode / f"trial_{trial}" / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_all_metrics(results_dir, mode):
    """Load all trials for a mode"""
    metrics = []
    for trial in [1, 2, 3]:
        m = load_trial_metrics(results_dir, mode, trial)
        if m:
            metrics.append(m)
    return metrics

def extract_metric(metrics, service, metric_name):
    """Extract a specific metric across all trials"""
    values = []
    for m in metrics:
        if service in m["services"]:
            values.append(m["services"][service][metric_name])
    return np.array(values)

def cohens_d(group1, group2):
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_comparison(baseline, qmix, hpa, metric_name, service):
    """Perform statistical comparison"""
    results = {}
    
    # QMIX vs Baseline
    if len(baseline) > 0 and len(qmix) > 0:
        t_stat, p_value = stats.ttest_rel(qmix, baseline)
        effect_size = cohens_d(baseline, qmix)
        improvement = ((np.mean(baseline) - np.mean(qmix)) / np.mean(baseline)) * 100
        
        results['qmix_vs_baseline'] = {
            'baseline_mean': np.mean(baseline),
            'baseline_std': np.std(baseline),
            'qmix_mean': np.mean(qmix),
            'qmix_std': np.std(qmix),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': effect_size,
            'improvement_pct': improvement,
            'significant': p_value < 0.05
        }
    
    # QMIX vs HPA
    if len(hpa) > 0 and len(qmix) > 0:
        t_stat, p_value = stats.ttest_rel(qmix, hpa)
        effect_size = cohens_d(hpa, qmix)
        improvement = ((np.mean(hpa) - np.mean(qmix)) / np.mean(hpa)) * 100
        
        results['qmix_vs_hpa'] = {
            'hpa_mean': np.mean(hpa),
            'hpa_std': np.std(hpa),
            'qmix_mean': np.mean(qmix),
            'qmix_std': np.std(qmix),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': effect_size,
            'improvement_pct': improvement,
            'significant': p_value < 0.05
        }
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found")
        sys.exit(1)
    
    print("=" * 80)
    print("FGCS STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Results Directory: {results_dir}\n")
    
    # Load all metrics
    baseline_metrics = load_all_metrics(results_dir, "baseline")
    hpa_metrics = load_all_metrics(results_dir, "hpa")
    qmix_metrics = load_all_metrics(results_dir, "qmix")
    
    print(f"Loaded trials: Baseline={len(baseline_metrics)}, HPA={len(hpa_metrics)}, QMIX={len(qmix_metrics)}\n")
    
    if len(baseline_metrics) == 0 or len(qmix_metrics) == 0:
        print("Error: Need at least baseline and QMIX results")
        sys.exit(1)
    
    # Analyze each service and metric
    services = ["api", "app"]
    metrics_to_analyze = [
        ("p99_ms", "P99 Latency (ms)", "lower_is_better"),
        ("rps", "Throughput (RPS)", "higher_is_better"),
        ("error_rate", "Error Rate", "lower_is_better"),
        ("cpu_cores", "CPU Usage (cores)", "lower_is_better")
    ]
    
    all_results = {}
    
    for service in services:
        print(f"\n{'='*80}")
        print(f"{service.upper()} SERVICE ANALYSIS")
        print(f"{'='*80}\n")
        
        for metric_key, metric_name, direction in metrics_to_analyze:
            baseline_vals = extract_metric(baseline_metrics, service, metric_key)
            hpa_vals = extract_metric(hpa_metrics, service, metric_key)
            qmix_vals = extract_metric(qmix_metrics, service, metric_key)
            
            print(f"\n{metric_name}:")
            print(f"  Baseline: {np.mean(baseline_vals):.3f} ± {np.std(baseline_vals):.3f} (n={len(baseline_vals)})")
            if len(hpa_vals) > 0:
                print(f"  HPA:      {np.mean(hpa_vals):.3f} ± {np.std(hpa_vals):.3f} (n={len(hpa_vals)})")
            print(f"  QMIX:     {np.mean(qmix_vals):.3f} ± {np.std(qmix_vals):.3f} (n={len(qmix_vals)})")
            
            # Statistical tests
            analysis = analyze_comparison(baseline_vals, qmix_vals, hpa_vals, metric_key, service)
            
            if 'qmix_vs_baseline' in analysis:
                comp = analysis['qmix_vs_baseline']
                print(f"\n  QMIX vs Baseline:")
                print(f"    Improvement: {comp['improvement_pct']:.2f}%")
                print(f"    t-statistic: {comp['t_statistic']:.3f}")
                print(f"    p-value: {comp['p_value']:.4f} {'✓ significant' if comp['significant'] else '✗ not significant'}")
                print(f"    Cohen's d: {comp['cohens_d']:.3f} ", end="")
                if abs(comp['cohens_d']) > 0.8:
                    print("(large effect)")
                elif abs(comp['cohens_d']) > 0.5:
                    print("(medium effect)")
                else:
                    print("(small effect)")
            
            if 'qmix_vs_hpa' in analysis:
                comp = analysis['qmix_vs_hpa']
                print(f"\n  QMIX vs HPA:")
                print(f"    Improvement: {comp['improvement_pct']:.2f}%")
                print(f"    t-statistic: {comp['t_statistic']:.3f}")
                print(f"    p-value: {comp['p_value']:.4f} {'✓ significant' if comp['significant'] else '✗ not significant'}")
                print(f"    Cohen's d: {comp['cohens_d']:.3f} ", end="")
                if abs(comp['cohens_d']) > 0.8:
                    print("(large effect)")
                elif abs(comp['cohens_d']) > 0.5:
                    print("(medium effect)")
                else:
                    print("(small effect)")
            
            all_results[f"{service}_{metric_key}"] = analysis
    
    # Generate LaTeX table
    print(f"\n{'='*80}")
    print("LATEX TABLE FOR PAPER")
    print(f"{'='*80}\n")
    
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Performance Comparison: QMIX vs Baselines (n=3 trials, mean ± std)}")
    print("\\label{tab:results}")
    print("\\begin{tabular}{llrrr}")
    print("\\toprule")
    print("Service & Metric & Baseline & HPA & QMIX \\\\")
    print("\\midrule")
    
    for service in services:
        for metric_key, metric_name, _ in metrics_to_analyze:
            baseline_vals = extract_metric(baseline_metrics, service, metric_key)
            hpa_vals = extract_metric(hpa_metrics, service, metric_key)
            qmix_vals = extract_metric(qmix_metrics, service, metric_key)
            
            baseline_str = f"{np.mean(baseline_vals):.2f} ± {np.std(baseline_vals):.2f}"
            hpa_str = f"{np.mean(hpa_vals):.2f} ± {np.std(hpa_vals):.2f}" if len(hpa_vals) > 0 else "N/A"
            qmix_str = f"{np.mean(qmix_vals):.2f} ± {np.std(qmix_vals):.2f}"
            
            print(f"{service.upper()} & {metric_name} & {baseline_str} & {hpa_str} & {qmix_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save detailed results
    output_file = Path(results_dir) / "statistical_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY FOR PAPER")
    print(f"{'='*80}\n")
    
    api_p99_baseline = extract_metric(baseline_metrics, "api", "p99_ms")
    api_p99_qmix = extract_metric(qmix_metrics, "api", "p99_ms")
    
    if len(api_p99_baseline) > 0 and len(api_p99_qmix) > 0:
        improvement = ((np.mean(api_p99_baseline) - np.mean(api_p99_qmix)) / np.mean(api_p99_baseline)) * 100
        analysis = all_results.get('api_p99_ms', {}).get('qmix_vs_baseline', {})
        
        print(f"Key Finding:")
        print(f"QMIX achieves {improvement:.1f}% lower API P99 latency compared to baseline")
        print(f"({np.mean(api_p99_qmix):.2f}±{np.std(api_p99_qmix):.2f}ms vs {np.mean(api_p99_baseline):.2f}±{np.std(api_p99_baseline):.2f}ms,")
        if 'p_value' in analysis:
            print(f"paired t-test: t={analysis['t_statistic']:.2f}, p={analysis['p_value']:.4f}, Cohen's d={analysis['cohens_d']:.2f})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

# Made with Bob
