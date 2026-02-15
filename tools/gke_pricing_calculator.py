"""
GKE Exact Pricing Calculator - Different Scenarios
Run: python3 gke_pricing_calculator.py
"""

import json
from datetime import datetime

# Machine type prices (us-central1 on-demand)
MACHINE_TYPES = {
    "e2-standard-2": {"vcpu": 4, "memory": "8GB", "price": 0.067011},
    "e2-standard-4": {"vcpu": 4, "memory": "16GB", "price": 0.134023},
    "n2-standard-2": {"vcpu": 2, "memory": "8GB", "price": 0.097118},
    "n2-standard-4": {"vcpu": 4, "memory": "16GB", "price": 0.194236},
    "n2-highmem-4": {"vcpu": 4, "memory": "32GB", "price": 0.262658},
    "n2-highmem-8": {"vcpu": 8, "memory": "64GB", "price": 0.525316},
    "n1-standard-4": {"vcpu": 4, "memory": "15GB", "price": 0.189999},
}

# GKE cluster fee (always $0.10/hour)
CLUSTER_MANAGEMENT_FEE = 0.10

def calculate_cost(machine_type: str, num_nodes: int, hours: float = 1.0) -> dict:
    """Calculate exact GKE cost for a configuration."""
    if machine_type not in MACHINE_TYPES:
        raise ValueError(f"Unknown machine type: {machine_type}")
    
    machine = MACHINE_TYPES[machine_type]
    node_price_per_hour = machine["price"]
    total_hourly = (node_price_per_hour * num_nodes) + CLUSTER_MANAGEMENT_FEE
    total_cost = total_hourly * hours
    
    # Calculate monthly (730 hours)
    monthly_cost = total_hourly * 730
    
    return {
        "machine_type": machine_type,
        "vcpu_per_node": machine["vcpu"],
        "memory_per_node": machine["memory"],
        "node_price_per_hour": node_price_per_hour,
        "num_nodes": num_nodes,
        "cluster_management_fee": CLUSTER_MANAGEMENT_FEE,
        "total_hourly_cost": total_hourly,
        "cost_per_hour": total_hourly,
        "cost_per_day": total_hourly * 24,
        "cost_per_month": monthly_cost,
        "calculation": f"({node_price_per_hour} × {num_nodes}) + {CLUSTER_MANAGEMENT_FEE} = {total_hourly}/hr"
    }

def main():
    print("=" * 70)
    print("GKE EXACT PRICING CALCULATOR")
    print("=" * 70)
    print(f"Generated: {datetime.now().isoformat()}\n")
    
    # Scenarios
    scenarios = [
        ("k3d Local (reference)", "n2-standard-2", 2),
        ("Tiny Production", "e2-standard-2", 1),
        ("Small Production", "n2-standard-2", 2),
        ("Standard Production", "n2-standard-4", 2),
        ("HA Cluster", "n2-standard-4", 3),
        ("Memory-Heavy Workload", "n2-highmem-4", 2),
        ("Large HA Cluster", "n2-standard-4", 5),
    ]
    
    results = []
    
    for name, machine_type, num_nodes in scenarios:
        calc = calculate_cost(machine_type, num_nodes)
        results.append(calc)
        
        print(f"\n{name}")
        print("-" * 70)
        print(f"  Machine type: {machine_type} ({calc['vcpu_per_node']} vCPU, {calc['memory_per_node']})")
        print(f"  Node count: {num_nodes}")
        print(f"  Calculation: {calc['calculation']}")
        print(f"  ")
        print(f"  Cost per hour:  ${calc['cost_per_hour']:.4f}")
        print(f"  Cost per day:   ${calc['cost_per_day']:.2f}")
        print(f"  Cost per month: ${calc['cost_per_month']:.2f}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("COST COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Machine Type':<20} {'Nodes':<6} {'Hourly':<10} {'Monthly':<10}")
    print("-" * 70)
    
    for name, machine_type, num_nodes in scenarios:
        calc = calculate_cost(machine_type, num_nodes)
        print(f"{name:<25} {machine_type:<20} {num_nodes:<6} ${calc['cost_per_hour']:<9.2f} ${calc['cost_per_month']:<9.2f}")
    
    print("\n" + "=" * 70)
    print("HOW TO USE THIS WITH YOUR CLUSTER")
    print("=" * 70)
    print("""
1. Find your node count:
   kubectl get nodes

2. Find your machine type:
   kubectl get nodes -L node.kubernetes.io/instance-type

3. Look up hourly price (us-central1):
   https://cloud.google.com/compute/all-pricing

4. Update tools/gke_cost_report.py:
   NODE_PRICE_PER_HOUR = <price>
   NUM_NODES = <count>

5. Run cost report:
   python3 tools/gke_cost_report.py
""")

if __name__ == "__main__":
    main()

