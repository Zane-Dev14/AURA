import requests
import datetime
import json
import time

# Prometheus config
PROM_URL = "http://localhost:9090"
PROM_BASES = ["http://127.0.0.1:9090", "http://localhost:9090", "http://[::1]:9090"]
WINDOW_MINUTES = 60
STEP = 60  # seconds
COLLECT_METRICS = False  # Set True to query Prometheus for per-service metrics

# ===== EXACT GKE PRICING CONFIGURATION =====
# GKE charges for ACTUAL NODES, not container utilization
# Formula: (machine_type_hourly_price × num_nodes) + $0.10 (cluster management) per hour
#
# Machine Type Prices (us-central1, on-demand):
#   n2-standard-2:  $0.097118/hr   (2 vCPU, 8GB)
#   n2-standard-4:  $0.194236/hr   (4 vCPU, 16GB)
#   n1-standard-4:  $0.189999/hr   (4 vCPU, 15GB)
#   e2-standard-2:  $0.067011/hr   (2 vCPU, 8GB)
#   e2-standard-4:  $0.134023/hr   (4 vCPU, 16GB)
# Full pricing: https://cloud.google.com/compute/all-pricing
#
# NODE_PRICE = price per node per hour (from machine type)
# NUM_NODES = number of nodes in your cluster
# CLUSTER_MANAGEMENT_FEE = $0.10 per cluster (GKE standard)
#
# Example for k3d with 2 nodes (if they were n2-standard-2):
#   Cost = ($0.097118 × 2) + $0.10 = $0.294236/hour

TARGET_REGION = "asia-south1"  # Mumbai (India)
PRICE_REGION = "us-central1"   # Pricing source (placeholder)
MACHINE_TYPE = "e2-standard-2"  # Closest match for 2 vCPU / ~6 GiB nodes
NODE_PRICE_PER_HOUR = 0.067011  # e2-standard-2 price in us-central1 (placeholder)
NUM_NODES = 2  # k3d shows 2 nodes
CLUSTER_MANAGEMENT_FEE = 0.10  # GKE standard - DO NOT CHANGE
GKE_MODE = "Standard"  # Standard (node-based) vs Autopilot (pod-based)

SERVICES = ["api", "app", "db"]
NAMESPACE = "default"


def get_time_range():
    """Get time window ending now."""
    now = int(time.time())
    start = now - (WINDOW_MINUTES * 60)
    start_str = datetime.datetime.utcfromtimestamp(start).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = datetime.datetime.utcfromtimestamp(now).strftime('%Y-%m-%dT%H:%M:%SZ')
    return start_str, end_str


def prom_query_range(query: str, start: str, end: str):
    """Query Prometheus with range."""
    last_err = None
    for base in PROM_BASES:
        try:
            url = f"{base}/api/v1/query_range"
            r = requests.get(
                url,
                params={"query": query, "start": start, "end": end, "step": STEP},
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "success":
                last_err = f"Prometheus returned: {data.get('error', 'unknown')}"
                continue
            return data["data"]["result"]
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"Prometheus query failed. Last error: {last_err}")


def calculate_cost():
    """
    Calculate EXACT GKE costs based on node pricing.
    
    GKE Standard Mode Cost = (NODE_PRICE_PER_HOUR × NUM_NODES) + CLUSTER_MANAGEMENT_FEE
    
    Container metrics are collected for visibility only - they do NOT affect billing.
    GKE charges for nodes regardless of utilization.
    """
    start, end = get_time_range()
    print(f"Cost window: {start} to {end}")
    print(f"GKE Mode: {GKE_MODE}")

    total_vcpu_seconds = 0.0
    total_gb_seconds = 0.0
    per_service = {}

    # Collect metrics for visibility (not for billing)
    if COLLECT_METRICS:
        for svc in SERVICES:
            # CPU in cores
            cpu_q = f'''
                sum(rate(container_cpu_usage_seconds_total{{
                    namespace="{NAMESPACE}",
                    pod=~"{svc}-.*",
                    container="{svc}",
                    container!="POD"
                }}[1m]))
            '''

            # Memory in bytes
            mem_q = f'''
                sum(container_memory_working_set_bytes{{
                    namespace="{NAMESPACE}",
                    pod=~"{svc}-.*",
                    container="{svc}"
                }})
            '''

            # Pod count
            pod_q = f'kube_deployment_spec_replicas{{deployment="{svc}"}}'

            cpu_data = prom_query_range(cpu_q, start, end)
            mem_data = prom_query_range(mem_q, start, end)
            pod_data = prom_query_range(pod_q, start, end)

            svc_vcpu = 0.0
            svc_gb = 0.0
            svc_pods = 0.0

            # Sum CPU across time (each point is cores, multiply by STEP to get core-seconds)
            for series in cpu_data:
                for ts, v in series.get('values', []):
                    try:
                        svc_vcpu += float(v) * STEP
                    except (ValueError, TypeError):
                        pass

            # Sum memory across time (average working set * STEP in seconds)
            for series in mem_data:
                for ts, v in series.get('values', []):
                    try:
                        svc_gb += float(v) * STEP / 1e9
                    except (ValueError, TypeError):
                        pass

            # Max pod count during window
            for series in pod_data:
                for ts, v in series.get('values', []):
                    try:
                        svc_pods = max(svc_pods, float(v))
                    except (ValueError, TypeError):
                        pass

            total_vcpu_seconds += svc_vcpu
            total_gb_seconds += svc_gb

            per_service[svc] = {
                "vcpu_seconds": svc_vcpu,
                "gb_seconds": svc_gb,
                "pods": svc_pods,
            }

    # Convert to hours for reference
    vcpu_hours = total_vcpu_seconds / 3600.0
    gb_hours = total_gb_seconds / 3600.0
    window_hours = WINDOW_MINUTES / 60.0
    
    # ===== EXACT GKE COST CALCULATION =====
    # GKE Standard clusters charge for nodes: COST = (node_price × num_nodes + cluster_fee) × hours
    total_node_cost_per_hour = (NODE_PRICE_PER_HOUR * NUM_NODES) + CLUSTER_MANAGEMENT_FEE
    cost = total_node_cost_per_hour * window_hours
    
    # Print report
    print("\n=== EXACT GKE Cost Report ===")
    print(f"\nNode Configuration:")
    print(f"  Target region: {TARGET_REGION}")
    print(f"  Price source region: {PRICE_REGION}")
    print(f"  Machine type: {MACHINE_TYPE}")
    print(f"  Machine type hourly price: ${NODE_PRICE_PER_HOUR:.6f}/hour")
    print(f"  Number of nodes: {NUM_NODES}")
    print(f"  Cluster management fee: ${CLUSTER_MANAGEMENT_FEE:.2f}/hour")
    print(f"  Total per hour: ${total_node_cost_per_hour:.6f}")
    print(f"  Window duration: {window_hours:.2f} hours")
    
    if COLLECT_METRICS:
        print(f"\nContainer Metrics (reference only - does NOT affect GKE billing):")
        for svc, data in per_service.items():
            vcpu_h = data["vcpu_seconds"] / 3600.0
            gb_h = data["gb_seconds"] / 3600.0
            print(f"  {svc}: {vcpu_h:.4f} vCPU-h, {gb_h:.4f} GB-h, {int(data['pods'])} pods")

        print(f"\n  Total: {vcpu_hours:.4f} vCPU-h, {gb_hours:.4f} GB-h")
    else:
        print("\nContainer Metrics: skipped (COLLECT_METRICS=False)")
    print(f"\n=== FINAL COST ===")
    print(f"${cost:.2f} for {window_hours:.2f} hours")
    print(f"\nNote: GKE charges for nodes regardless of container utilization.")
    print(f"You pay for {NUM_NODES} node(s) @ ${NODE_PRICE_PER_HOUR:.6f}/hr + ${CLUSTER_MANAGEMENT_FEE}/hr management")
    if TARGET_REGION != PRICE_REGION:
        print("WARNING: Pricing uses a placeholder region. Exact asia-south1 price requires GCP SKU data.")

    return {
        "window": {"start": start, "end": end, "minutes": WINDOW_MINUTES},
        "services": per_service,
        "node_config": {
            "target_region": TARGET_REGION,
            "price_region": PRICE_REGION,
            "machine_type": MACHINE_TYPE,
            "machine_type_hourly_price": NODE_PRICE_PER_HOUR,
            "num_nodes": NUM_NODES,
            "cluster_management_fee": CLUSTER_MANAGEMENT_FEE,
            "total_hourly_cost": total_node_cost_per_hour
        },
        "metrics": {
            "vcpu_hours": vcpu_hours,
            "gb_hours": gb_hours,
            "note": "Metrics shown for reference only. GKE charges nodes, not container utilization."
        },
        "total_cost": cost,
        "gke_mode": GKE_MODE
    }


if __name__ == "__main__":
    try:
        result = calculate_cost()
        # Save diagnostics
        with open('/tmp/gke_cost_report.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\nDiagnostics saved to /tmp/gke_cost_report.json")
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
