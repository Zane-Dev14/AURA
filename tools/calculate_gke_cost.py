#!/usr/bin/env python3
"""
GKE Cost Calculator for AURA Baseline Test

Calculates exact GKE costs based on your cluster configuration.
Accounts for node pricing + cluster management fee.

Usage:
    python3 tools/calculate_gke_cost.py

Or customize:
    python3 tools/calculate_gke_cost.py --num-nodes 3 --machine-type n2-standard-4 --hours 24
"""

import argparse
from typing import Dict, Tuple

# ============================================================
# GKE Machine Type Pricing (us-central1, on-demand, per hour)
# Source: https://cloud.google.com/compute/all-pricing
# ============================================================

GKE_MACHINE_PRICES = {
    # e2 series (budget, burstable performance)
    "e2-standard-2": 0.067011,    # 2 vCPU, 8 GB
    "e2-standard-4": 0.134023,    # 4 vCPU, 16 GB
    "e2-standard-8": 0.268046,    # 8 vCPU, 32 GB
    
    # n2 series (balanced, standard)
    "n2-standard-2": 0.097118,    # 2 vCPU, 8 GB
    "n2-standard-4": 0.194236,    # 4 vCPU, 16 GB
    "n2-standard-8": 0.388472,    # 8 vCPU, 32 GB
    "n2-standard-16": 0.776944,   # 16 vCPU, 64 GB
    
    # n1 series (older standard)
    "n1-standard-1": 0.047500,    # 1 vCPU, 3.75 GB
    "n1-standard-2": 0.095000,    # 2 vCPU, 7.5 GB
    "n1-standard-4": 0.189999,    # 4 vCPU, 15 GB
    "n1-standard-8": 0.379998,    # 8 vCPU, 30 GB
}

# GKE cluster management fee (fixed, per cluster)
CLUSTER_MANAGEMENT_FEE = 0.10  # $0.10 per hour

# ============================================================
# COST CALCULATION
# ============================================================

def calculate_gke_cost(
    machine_type: str,
    num_nodes: int,
    hours: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate exact GKE cost.
    
    Args:
        machine_type: Machine type name (e.g., "n2-standard-4")
        num_nodes: Number of nodes in cluster
        hours: Duration in hours
        
    Returns:
        Dict with cost breakdown
    """
    
    if machine_type not in GKE_MACHINE_PRICES:
        raise ValueError(
            f"Unknown machine type: {machine_type}\n"
            f"Available types: {', '.join(GKE_MACHINE_PRICES.keys())}"
        )
    
    node_price_per_hour = GKE_MACHINE_PRICES[machine_type]
    
    # Calculate costs
    node_cost_per_hour = node_price_per_hour * num_nodes
    total_hourly = node_cost_per_hour + CLUSTER_MANAGEMENT_FEE
    total_cost = total_hourly * hours
    
    return {
        "machine_type": machine_type,
        "node_price_per_hour": node_price_per_hour,
        "num_nodes": num_nodes,
        "node_cost_per_hour": node_cost_per_hour,
        "cluster_fee_per_hour": CLUSTER_MANAGEMENT_FEE,
        "total_per_hour": total_hourly,
        "hours": hours,
        "total_cost": total_cost,
    }


def format_cost_report(cost_dict: Dict[str, float]) -> str:
    """Format cost data for display."""
    
    report = []
    report.append("╔════════════════════════════════════════════════════════════════╗")
    report.append("║              GKE Cost Calculator - AURA Baseline               ║")
    report.append("╚════════════════════════════════════════════════════════════════╝")
    report.append("")
    
    report.append("CLUSTER CONFIGURATION:")
    report.append(f"  Machine Type:        {cost_dict['machine_type']}")
    report.append(f"  Number of Nodes:     {int(cost_dict['num_nodes'])}")
    report.append(f"  Test Duration:       {cost_dict['hours']:.2f} hours")
    report.append("")
    
    report.append("PRICING (us-central1, on-demand):")
    report.append(f"  Single Node/Hour:    ${cost_dict['node_price_per_hour']:.6f}")
    report.append(f"  All Nodes/Hour:      ${cost_dict['node_cost_per_hour']:.6f}")
    report.append(f"  Cluster Mgmt Fee:    ${cost_dict['cluster_fee_per_hour']:.2f}/hour (fixed)")
    report.append(f"  TOTAL/Hour:          ${cost_dict['total_per_hour']:.6f}")
    report.append("")
    
    report.append("COST BREAKDOWN:")
    hours = cost_dict['hours']
    base_hourly = cost_dict['total_per_hour']
    
    report.append(f"  For {hours:.2f} hour(s):     ${cost_dict['total_cost']:.2f}")
    report.append(f"  Per Day (24h):       ${base_hourly * 24:.2f}")
    report.append(f"  Per Week (7d):       ${base_hourly * 24 * 7:.2f}")
    report.append(f"  Per Month (730h):    ${base_hourly * 730:.2f}")
    report.append(f"  Per Year (8,760h):   ${base_hourly * 8760:.2f}")
    report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate exact GKE costs for AURA baseline test"
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default="n2-standard-4",
        help="Machine type (default: n2-standard-4)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=2,
        help="Number of nodes (default: 2)"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Test duration in hours (default: 1.0 = 60 min)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show costs for all machine types (2 nodes)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        print("\n" + "═" * 70)
        print("GKE COST COMPARISON (2 nodes, 1 hour)")
        print("═" * 70 + "\n")
        
        comparisons = []
        for machine_type in sorted(GKE_MACHINE_PRICES.keys()):
            cost_dict = calculate_gke_cost(machine_type, 2, 1.0)
            comparisons.append((machine_type, cost_dict['total_per_hour']))
        
        for machine_type, hourly_cost in comparisons:
            daily = hourly_cost * 24
            monthly = hourly_cost * 730
            print(f"  {machine_type:20} → ${hourly_cost:7.4f}/hr | ${daily:7.2f}/day | ${monthly:8.2f}/month")
        
        print("\n" + "═" * 70 + "\n")
    else:
        cost_dict = calculate_gke_cost(args.machine_type, args.num_nodes, args.hours)
        print("\n" + format_cost_report(cost_dict) + "\n")


if __name__ == "__main__":
    main()
