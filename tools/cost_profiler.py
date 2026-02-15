#!/usr/bin/env python3
"""
AURA GKE Cost Profiler

Computes theoretical GKE costs based on collected metrics.
Uses resource REQUESTS (not usage) for accurate node-based pricing.

IMPORTANT:
- Cost estimation is based on Kubernetes resource REQUESTS (scheduler guarantees),
  NOT instantaneous usage metrics.
- GKE charges for entire nodes, not container utilization.
- Node count estimation uses ceil(total_cpu_requested / cpu_per_node).

Pricing Model (n2-standard-4):
- Machine type: n2-standard-4
- vCPUs per node: 4
- RAM per node: 16 GB
- Price: $0.189 per hour (us-central1, on-demand)
- Cluster management fee: $0.10 per hour (GKE Standard)

Usage:
    python tools/cost_profiler.py --mode baseline --metrics-file "docs/Final Results/baseline_metrics_*.json"
    python tools/cost_profiler.py --mode qmix --metrics-file "docs/Final Results/qmix_metrics_*.json"

Output:
    Appends cost_model section to metrics and saves as:
    docs/Final Results/<mode>_cost_<TIMESTAMP>.json
"""

import argparse
import datetime
import json
import math
from pathlib import Path
from typing import Optional


# ============================================================
# GKE PRICING CONFIGURATION
# ============================================================

# n2-standard-4 (as specified in thesis requirements)
GKE_MACHINE_TYPE = "n2-standard-4"
GKE_VCPU_PER_NODE = 4
GKE_RAM_GB_PER_NODE = 16
GKE_NODE_PRICE_PER_HOUR = 0.189  # USD, us-central1, on-demand

# GKE Standard cluster management fee
GKE_CLUSTER_MANAGEMENT_FEE = 0.10  # USD per hour

# Monthly hours (24 hours × 30 days)
HOURS_PER_MONTH = 24 * 30  # 720 hours


# ============================================================
# COST MODEL FUNCTIONS
# ============================================================

def estimate_node_count(
    total_cpu_requested_cores: float,
    total_memory_requested_gb: float,
    cpu_per_node: float = GKE_VCPU_PER_NODE,
    ram_per_node: float = GKE_RAM_GB_PER_NODE
) -> dict:
    """
    Estimate required node count based on resource requests.
    
    Uses ceiling of max(CPU-based nodes, memory-based nodes) to ensure
    workloads can be scheduled.
    
    Returns dict with node count and which resource is the bottleneck.
    """
    # CPU-based node count
    cpu_nodes = math.ceil(total_cpu_requested_cores / cpu_per_node)
    
    # Memory-based node count
    mem_nodes = math.ceil(total_memory_requested_gb / ram_per_node)
    
    # Take the maximum (either CPU or memory constrains scheduling)
    node_count = max(cpu_nodes, mem_nodes, 1)  # Minimum 1 node
    
    bottleneck = "cpu" if cpu_nodes >= mem_nodes else "memory"
    
    return {
        "node_count": node_count,
        "cpu_based_estimate": cpu_nodes,
        "memory_based_estimate": mem_nodes,
        "bottleneck": bottleneck
    }


def calculate_hourly_cost(
    node_count: int,
    node_price: float = GKE_NODE_PRICE_PER_HOUR,
    cluster_fee: float = GKE_CLUSTER_MANAGEMENT_FEE
) -> float:
    """
    Calculate hourly GKE cost.
    
    Formula: (node_price × node_count) + cluster_management_fee
    """
    return (node_price * node_count) + cluster_fee


def calculate_monthly_cost(hourly_cost: float) -> float:
    """Calculate monthly cost from hourly."""
    return hourly_cost * HOURS_PER_MONTH


def calculate_cost_per_rps(
    hourly_cost: float,
    total_rps: Optional[float]
) -> Optional[float]:
    """
    Calculate cost per RPS (request per second).
    
    This is a performance-normalized metric useful for comparing
    autoscaler efficiency.
    """
    if total_rps is None or total_rps <= 0:
        return None
    return hourly_cost / total_rps


def calculate_cost_per_1k_requests(cost_per_rps: Optional[float]) -> Optional[float]:
    """Calculate cost per 1000 requests."""
    if cost_per_rps is None:
        return None
    return cost_per_rps * 1000


def calculate_cost_per_1k_users(
    monthly_cost: float,
    peak_users: int
) -> Optional[float]:
    """
    Calculate cost per 1000 users (secondary metric).
    
    Less reliable than cost_per_rps because "users" in Locust
    are virtual users, not real users.
    """
    if peak_users <= 0:
        return None
    return monthly_cost / (peak_users / 1000)


def calculate_total_replica_hours(replica_hours_dict: dict) -> float:
    """
    Calculate total replica-hours across all services.
    """
    total = 0.0
    for service, data in replica_hours_dict.items():
        rh = data.get("replica_hours")
        if rh is not None:
            total += rh
    return round(total, 4)


# ============================================================
# MAIN COST PROFILER
# ============================================================

def compute_cost_model(metrics: dict) -> dict:
    """
    Compute the full cost model from collected metrics.
    
    Parameters:
        metrics: Dict loaded from baseline_metrics_*.json
    
    Returns:
        Dict with complete cost model
    """
    # Extract data from metrics
    resource_requests = metrics.get("resource_requests", {})
    cluster = metrics.get("cluster", {})
    replica_hours = metrics.get("replica_hours", {})
    test_duration = metrics.get("test_duration_minutes", 60)
    
    total_cpu_requested = resource_requests.get("total_cpu_requested_cores", 0)
    total_memory_requested = resource_requests.get("total_memory_requested_gb", 0)
    total_rps = cluster.get("total_rps")
    
    # Estimate node count
    node_estimate = estimate_node_count(total_cpu_requested, total_memory_requested)
    node_count = node_estimate["node_count"]
    
    # Calculate costs
    hourly_cost = calculate_hourly_cost(node_count)
    monthly_cost = calculate_monthly_cost(hourly_cost)
    cost_per_rps = calculate_cost_per_rps(hourly_cost, total_rps)
    cost_per_1k_requests = calculate_cost_per_1k_requests(cost_per_rps)
    
    # Replica-hours calculation
    total_replica_hours = calculate_total_replica_hours(replica_hours)
    
    # Build cost model
    cost_model = {
        # Machine configuration
        "machine_type": GKE_MACHINE_TYPE,
        "vcpu_per_node": GKE_VCPU_PER_NODE,
        "ram_gb_per_node": GKE_RAM_GB_PER_NODE,
        "node_price_per_hour_usd": GKE_NODE_PRICE_PER_HOUR,
        "cluster_management_fee_per_hour_usd": GKE_CLUSTER_MANAGEMENT_FEE,
        
        # Node estimation
        "node_count_estimated": node_count,
        "node_estimate_details": node_estimate,
        
        # Cost calculations
        "hourly_cost_usd": round(hourly_cost, 4),
        "monthly_cost_usd": round(monthly_cost, 2),
        
        # Performance-normalized metrics
        "cost_per_rps_usd": round(cost_per_rps, 6) if cost_per_rps else None,
        "cost_per_1k_requests_usd": round(cost_per_1k_requests, 4) if cost_per_1k_requests else None,
        
        # Replica accounting
        "total_replica_hours": total_replica_hours,
        "cost_per_replica_hour_usd": round(hourly_cost / max(total_replica_hours, 1), 4) if total_replica_hours > 0 else None,
        
        # Input summary
        "input_summary": {
            "total_cpu_requested_cores": total_cpu_requested,
            "total_memory_requested_gb": total_memory_requested,
            "total_rps": total_rps,
            "test_duration_minutes": test_duration
        },
        
        # Metadata
        "pricing_notes": [
            "Cost based on Kubernetes resource REQUESTS, not actual usage",
            "GKE charges for entire nodes regardless of utilization",
            "Prices are for us-central1 on-demand (adjust for other regions)",
            f"Monthly estimate assumes {HOURS_PER_MONTH} hours (24×30 days)"
        ]
    }
    
    return cost_model


def load_metrics(filepath: str) -> dict:
    """Load metrics JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(path, "r") as f:
        return json.load(f)


def save_cost_report(metrics: dict, cost_model: dict, output_dir: Path, mode: str) -> Path:
    """Save combined metrics + cost model to JSON."""
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Combine metrics with cost model
    combined = {**metrics, "cost_model": cost_model}
    
    # Save
    output_file = output_dir / f"{mode}_cost_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)
    
    return output_file


def print_cost_summary(cost_model: dict, mode: str):
    """Print human-readable cost summary."""
    print(f"\n{'='*60}")
    print(f"GKE COST MODEL — Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    print(f"\n[Machine Configuration]")
    print(f"  Machine type:         {cost_model['machine_type']}")
    print(f"  vCPUs per node:       {cost_model['vcpu_per_node']}")
    print(f"  RAM per node:         {cost_model['ram_gb_per_node']} GB")
    print(f"  Node price:           ${cost_model['node_price_per_hour_usd']:.3f}/hour")
    print(f"  Cluster fee:          ${cost_model['cluster_management_fee_per_hour_usd']:.2f}/hour")
    
    print(f"\n[Node Estimation]")
    details = cost_model.get("node_estimate_details", {})
    print(f"  CPU-based estimate:   {details.get('cpu_based_estimate', 'N/A')} nodes")
    print(f"  Memory-based estimate: {details.get('memory_based_estimate', 'N/A')} nodes")
    print(f"  Bottleneck:           {details.get('bottleneck', 'N/A')}")
    print(f"  FINAL NODE COUNT:     {cost_model['node_count_estimated']}")
    
    print(f"\n[Cost Calculations]")
    print(f"  Hourly cost:          ${cost_model['hourly_cost_usd']:.4f}")
    print(f"  Monthly cost:         ${cost_model['monthly_cost_usd']:.2f}")
    
    print(f"\n[Performance-Normalized Metrics]")
    if cost_model.get("cost_per_rps_usd"):
        print(f"  Cost per RPS:         ${cost_model['cost_per_rps_usd']:.6f}")
    else:
        print(f"  Cost per RPS:         N/A (no RPS data)")
    
    if cost_model.get("cost_per_1k_requests_usd"):
        print(f"  Cost per 1K requests: ${cost_model['cost_per_1k_requests_usd']:.4f}")
    else:
        print(f"  Cost per 1K requests: N/A")
    
    print(f"\n[Replica Accounting]")
    print(f"  Total replica-hours:  {cost_model['total_replica_hours']}")
    if cost_model.get("cost_per_replica_hour_usd"):
        print(f"  Cost per replica-hour: ${cost_model['cost_per_replica_hour_usd']:.4f}")
    
    print(f"\n[Input Summary]")
    inputs = cost_model.get("input_summary", {})
    print(f"  CPU requested:        {inputs.get('total_cpu_requested_cores', 'N/A')} cores")
    print(f"  Memory requested:     {inputs.get('total_memory_requested_gb', 'N/A')} GB")
    print(f"  Total RPS:            {inputs.get('total_rps', 'N/A')}")
    print(f"  Test duration:        {inputs.get('test_duration_minutes', 'N/A')} minutes")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AURA GKE Cost Profiler")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "qmix", "hpa"],
        help="Experiment mode (default: baseline)"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        required=True,
        help="Path to metrics JSON file from gke_cost_report.py"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/Final Results",
        help="Output directory (default: docs/Final Results)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metrics from: {args.metrics_file}")
    metrics = load_metrics(args.metrics_file)
    
    print("Computing cost model...")
    cost_model = compute_cost_model(metrics)
    
    # Print summary
    print_cost_summary(cost_model, args.mode)
    
    # Save
    output_file = save_cost_report(metrics, cost_model, output_dir, args.mode)
    print(f"\n[OUTPUT] Cost report saved to: {output_file}")
    
    return cost_model


if __name__ == "__main__":
    main()
