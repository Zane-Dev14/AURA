#!/usr/bin/env python3
"""
Find your exact GKE cluster costs based on your machine type and region.

Usage:
1. Run: kubectl get nodes -o wide
   Look for the machine type (e.g., n2-standard-2)

2. Visit https://cloud.google.com/compute/pricing and find your region/machine combo

3. Update gke_cost_report.py with your values:
   NODE_PRICE_PER_HOUR = <your machine hourly cost>
   CLUSTER_MANAGEMENT_FEE = 0.10  # Standard GKE fee
"""

# Common GKE Machine Types (US-Central1 region, as of 2026):
GKE_PRICING = {
    # n2-standard series
    "n2-standard-2": 0.0791,      # 2 vCPU, 8 GB
    "n2-standard-4": 0.1582,      # 4 vCPU, 16 GB
    "n2-standard-8": 0.3164,      # 8 vCPU, 32 GB
    "n2-standard-16": 0.6328,     # 16 vCPU, 64 GB
    "n2-standard-32": 1.2657,     # 32 vCPU, 128 GB
    
    # e2-standard series (cheaper, good for general workloads)
    "e2-standard-2": 0.0336,      # 2 vCPU, 8 GB
    "e2-standard-4": 0.0672,      # 4 vCPU, 16 GB
    "e2-standard-8": 0.1344,      # 8 vCPU, 32 GB
    
    # n1-standard series (legacy, more expensive)
    "n1-standard-1": 0.0475,      # 1 vCPU, 3.75 GB
    "n1-standard-2": 0.0950,      # 2 vCPU, 7.5 GB
    "n1-standard-4": 0.1900,      # 4 vCPU, 15 GB
}

CLUSTER_MANAGEMENT_FEE = 0.10  # Same for all GKE clusters

print("=== GKE PRICING REFERENCE (US-Central1, 2026) ===\n")
print("Machine Type              | vCPU | Memory | Cost/hour")
print("-" * 55)

for machine, price in GKE_PRICING.items():
    # Get vCPU from machine type
    parts = machine.split("-")[-1]
    vcpu = int(parts) if parts.isdigit() else "?"
    
    # Total cost with cluster management
    total = price + CLUSTER_MANAGEMENT_FEE
    print(f"{machine:23} | {vcpu:4} | varies | ${total:.4f}")

print(f"\nAll totals include ${CLUSTER_MANAGEMENT_FEE}/h cluster management fee")

print("\n=== HOW TO SET UP YOUR CLUSTER ===")
print("""
1. Determine your machine type:
   kubectl get nodes -o wide
   
2. Find the machine type column (e.g., "n2-standard-2")

3. Update tools/gke_cost_report.py:
   NODE_PRICE_PER_HOUR = 0.0791  # Example: n2-standard-2
   CLUSTER_MANAGEMENT_FEE = 0.10

4. Run the cost report:
   python3 tools/gke_cost_report.py

EXAMPLE OUTPUT:
- If you have 1 n2-standard-2 node in us-central1:
  - Node cost: $0.0791/hour
  - Cluster management: $0.10/hour
  - Total: $0.1791/hour
  
- Your containers use: 0.19 vCPU-hours, 0.22 GB-hours
- Cost for that usage: $0.1791 (for the full node, regardless of container usage)
""")
