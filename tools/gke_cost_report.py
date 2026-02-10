import requests
import datetime
import json
import sys

PROM_URL = "http://localhost:9090"
START = "2026-02-10T10:00:00Z"  # Set your test start time (UTC)
END = "2026-02-10T11:00:00Z"    # Set your test end time (UTC)
STEP = 15  # seconds

# GKE e2-standard-2 (2 vCPU, 8GB RAM) 2026 price (us-central1)
GKE_NODE_VCPU_PRICE = 0.0338  # per vCPU-hour
GKE_NODE_MEM_PRICE = 0.0042375  # per GB-hour

SERVICES = ["api", "app", "db"]

# Helper to query Prometheus range

def prom_range(query):
    resp = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": START,
            "end": END,
            "step": STEP,
        },
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus error: {data}")
    return data["data"]["result"]

# Aggregate vCPU-seconds and GB-seconds for all pods

def main():
    print(f"Test window: {START} to {END}")
    total_vcpu_seconds = 0.0
    total_gb_seconds = 0.0
    max_pods = 0
    user_count = None

    for svc in SERVICES:
        # CPU usage per pod (rate in cores)
        cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{container="{svc}"}}[{STEP}s])) by (pod)'
        cpu_data = prom_range(cpu_query)
        # Memory usage per pod (bytes)
        mem_query = f'sum(container_memory_working_set_bytes{{container="{svc}"}}) by (pod)'
        mem_data = prom_range(mem_query)
        # Pod count
        pod_query = f'kube_deployment_status_replicas{{deployment="{svc}"}}'
        pod_data = prom_range(pod_query)

        # Aggregate over all pods and time
        for series in cpu_data:
            for t, v in series["values"]:
                total_vcpu_seconds += float(v) * STEP
        for series in mem_data:
            for t, v in series["values"]:
                total_gb_seconds += float(v) / 1e9 * STEP
        # Track max pod count
        for series in pod_data:
            max_pods = max(max_pods, max(float(v) for t, v in series["values"]))

    # Try to get user count from Locust metrics (if exposed to Prometheus)
    try:
        user_query = 'locust_user_count'  # adjust if your metric name differs
        user_data = prom_range(user_query)
        if user_data:
            user_count = max(float(v) for series in user_data for t, v in series["values"])
    except Exception:
        pass

    vcpu_hours = total_vcpu_seconds / 3600
    gb_hours = total_gb_seconds / 3600
    cost = vcpu_hours * GKE_NODE_VCPU_PRICE + gb_hours * GKE_NODE_MEM_PRICE

    print(f"\n=== GKE Cost Report ===")
    print(f"Total vCPU-hours: {vcpu_hours:.2f}")
    print(f"Total GB-hours: {gb_hours:.2f}")
    print(f"Estimated GKE cost: ${cost:.4f}")
    print(f"Max pod count: {max_pods}")
    if user_count is not None:
        print(f"Max user count (from Locust): {user_count}")
    else:
        print("User count: (not found in Prometheus, check Locust logs)")

if __name__ == "__main__":
    main()
