#!/usr/bin/env python3
"""
Generate figures for AURA paper from CSV data.
Outputs publication-ready PDF figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# Use publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 11

# Paths
DATA_DIR = "docs/Final Results"
OUTPUT_DIR = "docs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load all CSV files."""
    data = {}
    for system in ['qmix', 'hpa', 'baseline']:
        data[system] = {
            'replicas': pd.read_csv(f"{DATA_DIR}/replicas_over_time_{system}.csv"),
            'p99': pd.read_csv(f"{DATA_DIR}/p99_over_time_{system}.csv"),
            'cpu': pd.read_csv(f"{DATA_DIR}/cpu_usage_over_time_{system}.csv")
        }
    return data

def plot_api_replicas(data):
    """Figure: API replica count over time."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    # Convert timestamps to minutes (assuming they're in seconds or similar)
    for system, label, style in [
        ('qmix', 'QMIX', '-'),
        ('hpa', 'HPA', '--'),
        ('baseline', 'Baseline', ':')
    ]:
        df = data[system]['replicas'].copy()
        if 'timestamp' in df.columns and 'api_replicas' in df.columns:
            # Convert timestamp to datetime and then to minutes
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
            ax.plot(time, df['api_replicas'], label=label,
                   linestyle=style, linewidth=2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('API Replicas')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/api_replicas_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: api_replicas_comparison.pdf")

def plot_app_replicas(data):
    """Figure: APP replica count over time."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    for system, label, style in [
        ('qmix', 'QMIX', '-'),
        ('hpa', 'HPA', '--'),
        ('baseline', 'Baseline', ':')
    ]:
        df = data[system]['replicas'].copy()
        if 'timestamp' in df.columns and 'app_replicas' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
            ax.plot(time, df['app_replicas'], label=label,
                   linestyle=style, linewidth=2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('APP Replicas')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/app_replicas_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: app_replicas_comparison.pdf")

def plot_api_p99_latency(data):
    """Figure: API P99 latency over time."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    for system, label, style in [
        ('qmix', 'QMIX', '-'),
        ('hpa', 'HPA', '--'),
        ('baseline', 'Baseline', ':')
    ]:
        df = data[system]['p99'].copy()
        if 'timestamp' in df.columns and 'api_p99' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
            ax.plot(time, df['api_p99'], label=label,
                   linestyle=style, linewidth=2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('API P99 Latency (ms)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/api_p99_latency_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: api_p99_latency_comparison.pdf")

def plot_app_p99_latency(data):
    """Figure: APP P99 latency over time."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    for system, label, style in [
        ('qmix', 'QMIX', '-'),
        ('hpa', 'HPA', '--'),
        ('baseline', 'Baseline', ':')
    ]:
        df = data[system]['p99'].copy()
        if 'timestamp' in df.columns and 'app_p99' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
            ax.plot(time, df['app_p99'], label=label,
                   linestyle=style, linewidth=2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('APP P99 Latency (ms)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/app_p99_latency_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: app_p99_latency_comparison.pdf")

def plot_total_cpu_usage(data):
    """Figure: Total CPU usage over time."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    for system, label, style in [
        ('qmix', 'QMIX', '-'),
        ('hpa', 'HPA', '--'),
        ('baseline', 'Baseline', ':')
    ]:
        df = data[system]['cpu'].copy()
        if 'timestamp' in df.columns and 'total_cpu' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
            ax.plot(time, df['total_cpu'], label=label,
                   linestyle=style, linewidth=2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Total CPU Usage (cores)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/total_cpu_usage_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: total_cpu_usage_comparison.pdf")

def plot_combined_replicas(data):
    """Figure: All replicas in subplots."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    
    services = [('api', 'API', ax1), ('app', 'APP', ax2), ('db', 'DB', ax3)]
    
    for service, title, ax in services:
        for system, label, style in [
            ('qmix', 'QMIX', '-'),
            ('hpa', 'HPA', '--'),
            ('baseline', 'Baseline', ':')
        ]:
            df = data[system]['replicas'].copy()
            col_name = f'{service}_replicas'
            if 'timestamp' in df.columns and col_name in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                time = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
                ax.plot(time, df[col_name], label=label,
                       linestyle=style, linewidth=1.5)
        
        ax.set_ylabel(f'{title} Replicas')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_ylim(bottom=0)
    
    ax3.set_xlabel('Time (minutes)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/all_replicas_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: all_replicas_comparison.pdf")

def main():
    print("Loading data from CSV files...")
    data = load_data()
    
    print("\nGenerating figures...")
    plot_api_replicas(data)
    plot_app_replicas(data)
    plot_api_p99_latency(data)
    plot_app_p99_latency(data)
    plot_total_cpu_usage(data)
    plot_combined_replicas(data)
    
    print(f"\n✓ All figures saved to {OUTPUT_DIR}/")
    print("\nTo include in LaTeX paper:")
    print("  \\includegraphics[width=\\columnwidth]{figures/api_replicas_comparison.pdf}")

if __name__ == "__main__":
    main()

# Made with Bob
