# GKE EXACT PRICING GUIDE

## TL;DR: The Formula

**GKE Standard Cluster Cost Per Hour:**
```
cost = (machine_type_hourly_price × num_nodes) + $0.10
```

That's it. GKE bills for **entire nodes**, regardless of their utilization.

---

## Key Facts

1. **GKE charges for NODES, not container utilization**
   - You pay for the full machine type whether containers use 10% or 100% of resources
   - Container metrics are useful for optimization, but they don't affect billing

2. **Cluster Management Fee** 
   - $0.10 per hour (applies once per cluster, not per node)
   - Charged for: monitoring, control plane, lifecycle management

3. **Per-Second Billing**
   - Minimum 1-minute charge per node
   - Rounded to nearest second

4. **Regions Matter**
   - Pricing varies by region (us-central1, us-east1, europe-west1, etc.)
   - This guide uses **us-central1** pricing

---

## Step 1: Detect Your Current Cluster Setup

### Find Your Nodes
```bash
kubectl get nodes -o wide
```

Output will show:
```
NAME                    STATUS   ROLES           VERSION
k3d-aura-server-0       Ready    control-plane   v1.30.0
k3d-aura-agent-0        Ready    worker          v1.30.0
```

### For k3d (Your Current Setup)
k3d runs locally on your machine and has **zero cost** (it's on your hardware).

To calculate equivalent GKE cost, you need to:
1. Determine what machine type your k3d would map to
2. Count the nodes

**Your k3d cluster:**
- 2 nodes: 1 control-plane + 1 worker
- Default k3d: modest resources (typically 1-2 vCPU, 1-2GB RAM per node)

---

## Step 2: Look Up Machine Types

### Common GKE Machine Types (us-central1, on-demand)

| Machine Type | vCPU | Memory | Hourly Cost |
|---|---|---|---|
| e2-standard-2 | 2 | 8 GB | $0.067011 |
| e2-standard-4 | 4 | 16 GB | $0.134023 |
| n2-standard-2 | 2 | 8 GB | $0.097118 |
| n2-standard-4 | 4 | 16 GB | $0.194236 |
| n1-standard-4 | 4 | 15 GB | $0.189999 |

Full pricing table: https://cloud.google.com/compute/all-pricing

### Example Mapping
- **Smallest:** e2-standard-2 = $0.067011/hr
- **Budget:** n2-standard-2 = $0.097118/hr (2 vCPU, 8GB)
- **Standard:** n2-standard-4 = $0.194236/hr (4 vCPU, 16GB)

---

## Step 3: Calculate Your Cluster Cost

### Formula Setup

```
NODE_PRICE_PER_HOUR = cost of single machine type
NUM_NODES = number of nodes in cluster
CLUSTER_MANAGEMENT_FEE = $0.10

Hourly Cost = (NODE_PRICE_PER_HOUR × NUM_NODES) + CLUSTER_MANAGEMENT_FEE
```

### Examples

**Example 1: 2 nodes, n2-standard-2 each**
```
Hourly = ($0.097118 × 2) + $0.10
       = $0.194236 + $0.10
       = $0.294236 per hour
       = $7.06 per day
       = $211.77 per month
```

**Example 2: 1 node, n2-standard-4**
```
Hourly = ($0.194236 × 1) + $0.10
       = $0.294236 per hour
       = $7.06 per day
       = $211.77 per month
```

**Example 3: 3 nodes, n1-standard-4 each**
```
Hourly = ($0.189999 × 3) + $0.10
       = $0.569997 + $0.10
       = $0.669997 per hour
       = $16.08 per day
       = $481.50 per month
```

---

## Step 4: Configure Your Cost Report

Edit `/Users/eric/Documents/GitHub/AURA/tools/gke_cost_report.py`:

```python
# Update these THREE values:
NODE_PRICE_PER_HOUR = 0.097118  # Your machine type price
NUM_NODES = 2                   # Your actual node count
CLUSTER_MANAGEMENT_FEE = 0.10   # GKE standard - don't change
```

Then run:
```bash
python3 tools/gke_cost_report.py
```

Output will show:
```
=== EXACT GKE Cost Report ===

Node Configuration:
  Machine type hourly price: $0.097118/hour
  Number of nodes: 2
  Cluster management fee: $0.10/hour
  Total per hour: $0.297118
  Window duration: 1.00 hours

=== FINAL COST ===
$0.30 for 1.00 hours
```

---

## Why This Is Different From Container Pricing

### Container Pricing (Inaccurate)
- Measures: actual vCPU + memory used
- Result: $0.006 for 1-hour window ❌ UNDERESTIMATES by 50x+

### Node Pricing (Accurate)
- Measures: machine type allocation
- Result: $0.30 for 1-hour window ✓ CORRECT

**The difference:** System pods, OS overhead, and unallocated resources all count as "node cost" in GKE.

---

## Step 5: When Deploying to Real GKE

1. **Create cluster** with your chosen machine type:
   ```bash
   gcloud container clusters create my-cluster \
     --machine-type n2-standard-4 \
     --num-nodes 3 \
     --zone us-central1-a
   ```

2. **Get actual machine type:**
   ```bash
   kubectl get nodes -o jsonpath='{.items[0].metadata.labels.node\.kubernetes\.io\/instance-type}'
   ```

3. **Look up price** from https://cloud.google.com/compute/all-pricing

4. **Update script:**
   ```python
   NODE_PRICE_PER_HOUR = 0.194236  # Your machine type
   NUM_NODES = 3                   # Your actual count
   ```

5. **Run cost report:**
   ```bash
   python3 tools/gke_cost_report.py
   ```

---

## Autopilot Mode (Alternative)

GKE Autopilot uses **pod-based billing** (different pricing model):
- vCPU: $0.0445/hour
- Memory: $0.0049225/GB/hour
- $0.10 cluster management fee

This is better for **small workloads** that don't fill entire nodes. Use Standard mode for consistent traffic.

---

## Summary

| Aspect | Value |
|--------|-------|
| **What GKE charges for** | Entire nodes |
| **Ignored/Free** | Container utilization, system pods, unused resources |
| **Minimum charge** | 1 minute per node |
| **Cluster fee** | $0.10/hour (one-time) |
| **Billing precision** | Per-second |
| **Best for** | Stable workloads, full node utilization |

