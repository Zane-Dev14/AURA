#!/usr/bin/env python3
"""
AURA Local Metrics Collector (legacy filename: gke_cost_report.py)

Collects Prometheus and Kubernetes metrics from a local k3d deployment for
reproducible baseline, QMIX, and HPA comparisons.

IMPORTANT DESIGN DECISIONS:
- Measurements are computed from in-cluster telemetry only.
- DB service uses TCP proxy so HTTP latency histograms are unavailable.
- All PromQL queries are logged for reproducibility.
- Script execution is blocked unless kubectl context is k3d-*.

Usage:
    python tools/gke_cost_report.py --mode baseline
    python tools/gke_cost_report.py --mode qmix
    python tools/gke_cost_report.py --mode hpa

Output:
    docs/Final Results/<mode>_metrics_<TIMESTAMP>.json
    docs/Final Results/<mode>_timeseries.csv
"""

import argparse
import datetime
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import requests

# ============================================================
# CONFIGURATION
# ============================================================

_PROM_OVERRIDE = (
    os.environ.get("PROMETHEUS_URL")
    or os.environ.get("AURA_PROMETHEUS_URL")
    or ""
).strip().rstrip("/")

PROM_BASES = [
    # Prefer stable NodePort from local kube-prometheus-stack values.
    "http://127.0.0.1:30090",
    "http://localhost:30090",

    # Backwards-compatible fallbacks (port-forward).
    "http://127.0.0.1:9090",
    "http://localhost:9090",
]

if _PROM_OVERRIDE:
    PROM_BASES.insert(0, _PROM_OVERRIDE)

SERVICES = ["api", "app", "db"]
NAMESPACE = "default"

# Test duration and collection settings
TEST_DURATION_MINUTES = 30
QUERY_RANGE_STEP = "15s"  # 15-second intervals for time-series

# Retry settings for Prometheus queries
PROM_MAX_RETRIES = 3
PROM_RETRY_DELAY = 2  # seconds between retries
ENFORCE_K3D_CONTEXT = os.environ.get("AURA_ENFORCE_K3D_CONTEXT", "true").lower() == "true"

# Output directory
OUTPUT_DIR = Path("docs/Final Results")

# ============================================================
# PROMETHEUS QUERY HELPERS
# ============================================================

# Store all queries for reproducibility logging
QUERIES_LOG = []


def get_current_kube_context() -> Optional[str]:
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        context = result.stdout.strip()
        return context or None
    except Exception:
        return None


def assert_k3d_context():
    if not ENFORCE_K3D_CONTEXT:
        return

    context = get_current_kube_context()
    if not context:
        raise RuntimeError("kubectl current-context is empty. Refusing to run metrics collector.")
    if not context.startswith("k3d-"):
        raise RuntimeError(f"Refusing to run on non-k3d context: {context}")


def prom_query(query: str, base_url: Optional[str] = None) -> Any:
    """
    Execute instant Prometheus query with retry logic.
    Returns the first result value, or None if empty/error.
    Logs all queries for reproducibility.
    """
    QUERIES_LOG.append({"type": "instant", "query": query})
    
    bases = [base_url] if base_url else PROM_BASES
    last_err = None
    
    for attempt in range(PROM_MAX_RETRIES):
        if attempt > 0:
            time.sleep(PROM_RETRY_DELAY)
        
        for base in bases:
            try:
                r = requests.get(
                    f"{base}/api/v1/query",
                    params={"query": query},
                    timeout=30
                )
                r.raise_for_status()
                data = r.json()
                
                if data.get("status") != "success":
                    last_err = f"Prometheus error: {data.get('error', 'unknown')}"
                    continue
                
                results = data.get("data", {}).get("result", [])
                if not results:
                    return None
                
                # Extract first result value
                value = results[0].get("value", [None, None])[1]
                if value is None:
                    return None
                
                v = float(value)
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
                
            except requests.exceptions.ConnectionError as e:
                last_err = f"Connection error (attempt {attempt+1}/{PROM_MAX_RETRIES}): {e}"
                break  # break inner loop to trigger retry
            except Exception as e:
                last_err = str(e)
                continue
    
    print(f"WARNING: Query failed after {PROM_MAX_RETRIES} attempts: {query[:80]}... Error: {last_err}")
    return None


def prom_query_range(query: str, start: str, end: str, step: str = QUERY_RANGE_STEP) -> list:
    """
    Execute range Prometheus query with retry logic.
    Returns list of (timestamp, value) tuples.
    Logs all queries for reproducibility.
    """
    QUERIES_LOG.append({"type": "range", "query": query, "start": start, "end": end, "step": step})
    
    last_err = None
    for attempt in range(PROM_MAX_RETRIES):
        if attempt > 0:
            time.sleep(PROM_RETRY_DELAY)
        
        for base in PROM_BASES:
            try:
                r = requests.get(
                    f"{base}/api/v1/query_range",
                    params={"query": query, "start": start, "end": end, "step": step},
                    timeout=60
                )
                r.raise_for_status()
                data = r.json()
                
                if data.get("status") != "success":
                    last_err = f"Prometheus error: {data.get('error', 'unknown')}"
                    continue
                
                results = data.get("data", {}).get("result", [])
                if not results:
                    return []
                
                # Extract values from first result
                values = results[0].get("values", [])
                parsed = []
                for ts, val in values:
                    try:
                        v = float(val)
                        if not (math.isnan(v) or math.isinf(v)):
                            parsed.append((ts, v))
                    except (ValueError, TypeError):
                        pass
                return parsed
                
            except requests.exceptions.ConnectionError as e:
                last_err = f"Connection error (attempt {attempt+1}/{PROM_MAX_RETRIES}): {e}"
                break  # break inner loop to trigger retry
            except Exception as e:
                last_err = str(e)
                continue
    
    print(f"WARNING: Range query failed after {PROM_MAX_RETRIES} attempts: {query[:80]}...")
    return []


def get_time_range(duration_minutes: int = TEST_DURATION_MINUTES) -> tuple:
    """Get time window ending now."""
    now = int(time.time())
    start = now - (duration_minutes * 60)
    start_str = datetime.datetime.fromtimestamp(start, tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return start_str, end_str, start, now


# ============================================================
# CLUSTER HARDWARE COLLECTION
# ============================================================

def collect_cluster_hardware() -> dict:
    """
    Collect node metadata from kubectl for reproducibility.
    Returns cluster_hardware dict.
    """
    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"WARNING: kubectl get nodes failed: {result.stderr}")
            return {}
        
        nodes = json.loads(result.stdout)
        items = nodes.get("items", [])
        
        node_count = len(items)
        total_cpu = 0.0
        total_memory_gb = 0.0
        allocatable_cpu = 0.0
        allocatable_memory_gb = 0.0
        
        for node in items:
            status = node.get("status", {})
            capacity = status.get("capacity", {})
            allocatable = status.get("allocatable", {})
            
            # Parse CPU (can be "4" or "4000m")
            cpu_cap = capacity.get("cpu", "0")
            cpu_alloc = allocatable.get("cpu", "0")
            total_cpu += parse_cpu(cpu_cap)
            allocatable_cpu += parse_cpu(cpu_alloc)
            
            # Parse memory (in Ki, Mi, Gi, or bytes)
            mem_cap = capacity.get("memory", "0")
            mem_alloc = allocatable.get("memory", "0")
            total_memory_gb += parse_memory_gb(mem_cap)
            allocatable_memory_gb += parse_memory_gb(mem_alloc)
        
        return {
            "node_count": node_count,
            "cpu_per_node_cores": round(total_cpu / max(node_count, 1), 2),
            "memory_per_node_gb": round(total_memory_gb / max(node_count, 1), 2),
            "allocatable_cpu_total_cores": round(allocatable_cpu, 2),
            "allocatable_memory_total_gb": round(allocatable_memory_gb, 2)
        }
    except Exception as e:
        print(f"WARNING: Could not collect cluster hardware: {e}")
        return {}


def parse_cpu(val: str) -> float:
    """Parse Kubernetes CPU value to cores."""
    val = str(val).strip()
    if val.endswith("m"):
        return float(val[:-1]) / 1000.0
    return float(val)


def parse_memory_gb(val: str) -> float:
    """Parse Kubernetes memory value to GB."""
    val = str(val).strip()
    if val.endswith("Ki"):
        return float(val[:-2]) / (1024 * 1024)
    elif val.endswith("Mi"):
        return float(val[:-2]) / 1024
    elif val.endswith("Gi"):
        return float(val[:-2])
    elif val.endswith("Ti"):
        return float(val[:-2]) * 1024
    elif val.endswith("K"):
        return float(val[:-1]) / (1000 * 1000)
    elif val.endswith("M"):
        return float(val[:-1]) / 1000
    elif val.endswith("G"):
        return float(val[:-1])
    else:
        # Assume bytes
        return float(val) / (1024 ** 3)


# ============================================================
# RESOURCE REQUESTS COLLECTION (FOR COST ESTIMATION)
# ============================================================

def collect_resource_requests() -> dict:
    """
    Collect resource REQUESTS from deployments.
    Cost estimation is based on REQUESTS, not actual usage.
    """
    try:
        result = subprocess.run(
            ["kubectl", "get", "deployments", "-n", NAMESPACE, "-o", "json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"WARNING: kubectl get deployments failed: {result.stderr}")
            return {}
        
        deploys = json.loads(result.stdout)
        items = deploys.get("items", [])
        
        total_cpu_requested = 0.0
        total_memory_requested_gb = 0.0
        services_requests = {}
        
        for deploy in items:
            name = deploy.get("metadata", {}).get("name", "unknown")
            if name not in SERVICES:
                continue
            
            replicas = deploy.get("spec", {}).get("replicas", 1)
            containers = deploy.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
            
            deploy_cpu = 0.0
            deploy_mem = 0.0
            
            for container in containers:
                resources = container.get("resources", {})
                requests = resources.get("requests", {})
                
                cpu_req = requests.get("cpu", "0")
                mem_req = requests.get("memory", "0")
                
                deploy_cpu += parse_cpu(cpu_req)
                deploy_mem += parse_memory_gb(mem_req)
            
            # Multiply by replica count
            total_deploy_cpu = deploy_cpu * replicas
            total_deploy_mem = deploy_mem * replicas
            
            total_cpu_requested += total_deploy_cpu
            total_memory_requested_gb += total_deploy_mem
            
            services_requests[name] = {
                "replicas": replicas,
                "cpu_request_per_pod_cores": round(deploy_cpu, 3),
                "memory_request_per_pod_gb": round(deploy_mem, 3),
                "total_cpu_requested_cores": round(total_deploy_cpu, 3),
                "total_memory_requested_gb": round(total_deploy_mem, 3)
            }
        
        return {
            "total_cpu_requested_cores": round(total_cpu_requested, 3),
            "total_memory_requested_gb": round(total_memory_requested_gb, 3),
            "per_service": services_requests
        }
    except Exception as e:
        print(f"WARNING: Could not collect resource requests: {e}")
        return {}


# ============================================================
# PROMETHEUS METRICS COLLECTION
# ============================================================

def collect_service_metrics(service: str) -> dict:
    """
    Collect metrics for a single service from Prometheus.
    """
    metrics = {}
    
    # CPU usage (cores)
    cpu_query = f'''
        sum(rate(container_cpu_usage_seconds_total{{
            namespace="{NAMESPACE}",
            pod=~"{service}-.*",
            container="{service}",
            container!="POD"
        }}[2m]))
    '''
    metrics["cpu_used_cores"] = prom_query(cpu_query)
    
    # Memory usage (bytes → GB)
    mem_query = f'''
        sum(container_memory_working_set_bytes{{
            namespace="{NAMESPACE}",
            pod=~"{service}-.*",
            container="{service}"
        }})
    '''
    mem_bytes = prom_query(mem_query)
    metrics["memory_used_gb"] = round(mem_bytes / (1024**3), 4) if mem_bytes else None
    
    # Replica count
    replica_query = f'kube_deployment_spec_replicas{{deployment="{service}"}}'
    metrics["replicas"] = prom_query(replica_query)
    
    # RPS (requests per second)
    if service != "db":
        rps_query = f'''
            sum(rate(envoy_http_downstream_rq_total{{
                namespace="{NAMESPACE}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress"
            }}[2m]))
        '''
        metrics["rps"] = prom_query(rps_query)
    else:
        metrics["rps"] = None
        metrics["rps_note"] = "DB is TCP proxy - no HTTP RPS"
    
    # Error rate
    if service != "db":
        error_query = f'''
            sum(rate(envoy_http_downstream_rq_xx{{
                namespace="{NAMESPACE}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress",
                envoy_response_code_class="5"
            }}[2m]))
        '''
        metrics["error_rate"] = prom_query(error_query)
    else:
        metrics["error_rate"] = None
    
    # Latency percentiles (HTTP services only)
    if service != "db":
        for pct, label in [(0.50, "p50_ms"), (0.95, "p95_ms"), (0.99, "p99_ms")]:
            latency_query = f'''
                histogram_quantile(
                    {pct},
                    sum by (le)(
                        rate(envoy_http_downstream_rq_time_bucket{{
                            namespace="{NAMESPACE}",
                            job="{service}",
                            envoy_http_conn_manager_prefix="ingress"
                        }}[2m])
                    )
                )
            '''
            val = prom_query(latency_query)
            metrics[label] = round(val, 2) if val else None
    else:
        metrics["p50_ms"] = None
        metrics["p95_ms"] = None
        metrics["p99_ms"] = None
        metrics["latency_note"] = "TCP proxy — HTTP histograms unavailable"
    
    return metrics


def collect_cluster_usage() -> dict:
    """
    Collect cluster-wide resource usage from Prometheus.
    """
    # Total CPU used across all services
    cpu_query = f'''
        sum(rate(container_cpu_usage_seconds_total{{
            namespace="{NAMESPACE}",
            container!="POD",
            container!=""
        }}[2m]))
    '''
    total_cpu = prom_query(cpu_query)
    
    # Total memory used
    mem_query = f'''
        sum(container_memory_working_set_bytes{{
            namespace="{NAMESPACE}",
            container!="POD",
            container!=""
        }})
    '''
    total_mem = prom_query(mem_query)
    
    # Total RPS (api + app combined)
    rps_query = f'''
        sum(rate(envoy_http_downstream_rq_total{{
            namespace="{NAMESPACE}",
            envoy_http_conn_manager_prefix="ingress"
        }}[2m]))
    '''
    total_rps = prom_query(rps_query)
    
    return {
        "total_cpu_used_cores": round(total_cpu, 4) if total_cpu else None,
        "total_memory_used_gb": round(total_mem / (1024**3), 4) if total_mem else None,
        "total_rps": round(total_rps, 2) if total_rps else None
    }


def check_sla_violations() -> dict:
    """
    Check for SLA violations (p99 > 2000ms for any service).
    """
    violations = {}
    for service in ["api", "app"]:
        p99_query = f'''
            histogram_quantile(
                0.99,
                sum by (le)(
                    rate(envoy_http_downstream_rq_time_bucket{{
                        namespace="{NAMESPACE}",
                        job="{service}",
                        envoy_http_conn_manager_prefix="ingress"
                    }}[2m])
                )
            )
        '''
        p99 = prom_query(p99_query)
        if p99 and p99 > 2000:
            violations[service] = {
                "p99_ms": round(p99, 2),
                "exceeded_threshold_ms": 2000
            }
    return violations


# ============================================================
# TIME SERIES COLLECTION
# ============================================================

def collect_timeseries(start: str, end: str) -> dict:
    """
    Collect time series data for graphing.
    Returns dict with per-service time series.
    """
    timeseries = {}
    
    for service in SERVICES:
        timeseries[service] = {}
        
        # Replica count over time
        replica_query = f'kube_deployment_status_replicas{{deployment="{service}"}}'
        timeseries[service]["replicas"] = prom_query_range(replica_query, start, end)
        
        # CPU usage over time
        cpu_query = f'''
            sum(rate(container_cpu_usage_seconds_total{{
                namespace="{NAMESPACE}",
                pod=~"{service}-.*",
                container="{service}",
                container!="POD"
            }}[2m]))
        '''
        timeseries[service]["cpu_cores"] = prom_query_range(cpu_query, start, end)
        
        # Memory usage over time
        mem_query = f'''
            sum(container_memory_working_set_bytes{{
                namespace="{NAMESPACE}",
                pod=~"{service}-.*",
                container="{service}"
            }}) / 1024 / 1024 / 1024
        '''
        timeseries[service]["memory_gb"] = prom_query_range(mem_query, start, end)
        
        # p99 latency over time (HTTP services only)
        if service != "db":
            p99_query = f'''
                histogram_quantile(
                    0.99,
                    sum by (le)(
                        rate(envoy_http_downstream_rq_time_bucket{{
                            namespace="{NAMESPACE}",
                            job="{service}",
                            envoy_http_conn_manager_prefix="ingress"
                        }}[2m])
                    )
                )
            '''
            timeseries[service]["p99_ms"] = prom_query_range(p99_query, start, end)
    
    return timeseries


def calculate_replica_hours(timeseries: dict, duration_minutes: int) -> dict:
    """
    Calculate replica-hours for each service.
    replica_hours = average_replicas * (duration_hours)
    """
    replica_hours = {}
    duration_hours = duration_minutes / 60.0
    
    for service in SERVICES:
        replicas_ts = timeseries.get(service, {}).get("replicas", [])
        if replicas_ts:
            avg_replicas = sum(v for _, v in replicas_ts) / len(replicas_ts)
            replica_hours[service] = {
                "avg_replicas": round(avg_replicas, 2),
                "replica_hours": round(avg_replicas * duration_hours, 4)
            }
        else:
            replica_hours[service] = {"avg_replicas": None, "replica_hours": None}
    
    return replica_hours


# ============================================================
# CSV EXPORT
# ============================================================

def export_timeseries_csv(timeseries: dict, mode: str, output_dir: Path):
    """
    Export time series data to CSV files for graphing.
    """
    import csv
    
    # Replicas over time
    replicas_file = output_dir / f"replicas_over_time_{mode}.csv"
    with open(replicas_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "api_replicas", "app_replicas", "db_replicas"])
        
        # Align timestamps across services
        all_ts = set()
        for service in SERVICES:
            for ts, _ in timeseries.get(service, {}).get("replicas", []):
                all_ts.add(ts)
        
        for ts in sorted(all_ts):
            row = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()]
            for service in SERVICES:
                replicas = timeseries.get(service, {}).get("replicas", [])
                val = next((v for t, v in replicas if t == ts), "")
                row.append(val)
            writer.writerow(row)
    
    print(f"  Exported: {replicas_file}")
    
    # P99 latency over time
    p99_file = output_dir / f"p99_over_time_{mode}.csv"
    with open(p99_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "api_p99_ms", "app_p99_ms"])
        
        all_ts = set()
        for service in ["api", "app"]:
            for ts, _ in timeseries.get(service, {}).get("p99_ms", []):
                all_ts.add(ts)
        
        for ts in sorted(all_ts):
            row = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()]
            for service in ["api", "app"]:
                p99_ts = timeseries.get(service, {}).get("p99_ms", [])
                val = next((round(v, 2) for t, v in p99_ts if t == ts), "")
                row.append(val)
            writer.writerow(row)
    
    print(f"  Exported: {p99_file}")
    
    # CPU usage over time
    cpu_file = output_dir / f"cpu_usage_over_time_{mode}.csv"
    with open(cpu_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "api_cpu_cores", "app_cpu_cores", "db_cpu_cores"])
        
        all_ts = set()
        for service in SERVICES:
            for ts, _ in timeseries.get(service, {}).get("cpu_cores", []):
                all_ts.add(ts)
        
        for ts in sorted(all_ts):
            row = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()]
            for service in SERVICES:
                cpu_ts = timeseries.get(service, {}).get("cpu_cores", [])
                val = next((round(v, 4) for t, v in cpu_ts if t == ts), "")
                row.append(val)
            writer.writerow(row)
    
    print(f"  Exported: {cpu_file}")


# ============================================================
# MAIN COLLECTION FUNCTION
# ============================================================

def collect_all_metrics(mode: str) -> dict:
    """
    Collect all metrics for the baseline/QMIX/HPA experiment.
    """
    print(f"\n{'='*60}")
    print(f"AURA Metrics Collection — Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    cluster_context = get_current_kube_context()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    start_str, end_str, start_ts, end_ts = get_time_range(TEST_DURATION_MINUTES)
    
    print(f"\nCollection window: {start_str} to {end_str}")
    print(f"Duration: {TEST_DURATION_MINUTES} minutes")
    
    # 1. Cluster hardware
    print("\n[1/6] Collecting cluster hardware metadata...")
    cluster_hardware = collect_cluster_hardware()
    if cluster_hardware:
        print(f"  Nodes: {cluster_hardware.get('node_count', 'N/A')}")
        print(f"  Total allocatable CPU: {cluster_hardware.get('allocatable_cpu_total_cores', 'N/A')} cores")
        print(f"  Total allocatable memory: {cluster_hardware.get('allocatable_memory_total_gb', 'N/A')} GB")
    
    # 2. Resource requests
    print("\n[2/6] Collecting resource requests (for cost estimation)...")
    resource_requests = collect_resource_requests()
    if resource_requests:
        print(f"  Total CPU requested: {resource_requests.get('total_cpu_requested_cores', 'N/A')} cores")
        print(f"  Total memory requested: {resource_requests.get('total_memory_requested_gb', 'N/A')} GB")
    
    # 3. Per-service metrics
    print("\n[3/6] Collecting per-service Prometheus metrics...")
    services = {}
    for service in SERVICES:
        print(f"  Querying {service}...")
        services[service] = collect_service_metrics(service)
    
    # 4. Cluster-wide usage
    print("\n[4/6] Collecting cluster-wide usage...")
    cluster_usage = collect_cluster_usage()
    if cluster_usage:
        print(f"  Total CPU used: {cluster_usage.get('total_cpu_used_cores', 'N/A')} cores")
        print(f"  Total memory used: {cluster_usage.get('total_memory_used_gb', 'N/A')} GB")
        print(f"  Total RPS: {cluster_usage.get('total_rps', 'N/A')}")
    
    # 5. SLA violations
    print("\n[5/6] Checking SLA violations (p99 > 2000ms)...")
    sla_violations = check_sla_violations()
    if sla_violations:
        for svc, data in sla_violations.items():
            print(f"  WARNING: {svc} p99={data['p99_ms']}ms exceeds {data['exceeded_threshold_ms']}ms")
    else:
        print("  No SLA violations detected")
    
    # 6. Time series
    print("\n[6/6] Collecting time series for graphs...")
    timeseries = collect_timeseries(start_str, end_str)
    replica_hours = calculate_replica_hours(timeseries, TEST_DURATION_MINUTES)
    
    for service, data in replica_hours.items():
        if data.get("avg_replicas") is not None:
            print(f"  {service}: avg {data['avg_replicas']} replicas, {data['replica_hours']} replica-hours")
    
    # Build final result
    result = {
        "timestamp": timestamp,
        "mode": mode,
        "cluster_context": cluster_context,
        "test_duration_minutes": TEST_DURATION_MINUTES,
        "collection_window": {
            "start": start_str,
            "end": end_str
        },
        "cluster_hardware": cluster_hardware,
        "resource_requests": resource_requests,
        "cluster": {
            "total_cpu_used_cores": cluster_usage.get("total_cpu_used_cores"),
            "total_cpu_requested_cores": resource_requests.get("total_cpu_requested_cores"),
            "total_memory_used_gb": cluster_usage.get("total_memory_used_gb"),
            "total_memory_requested_gb": resource_requests.get("total_memory_requested_gb"),
            "total_rps": cluster_usage.get("total_rps")
        },
        "services": {},
        "replica_hours": replica_hours,
        "sla_violations": sla_violations,
        "queries_log": QUERIES_LOG.copy()
    }
    
    # Merge service metrics with replica hours
    for service in SERVICES:
        result["services"][service] = {
            **services.get(service, {}),
            **replica_hours.get(service, {})
        }
    
    return result, timeseries


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AURA Metrics Collector")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "qmix", "hpa"],
        help="Experiment mode (default: baseline)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in minutes (default: 30)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/Final Results",
        help="Output directory (default: docs/Final Results)"
    )
    args = parser.parse_args()

    try:
        assert_k3d_context()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
    
    # Use args values (don't modify globals)
    test_duration = args.duration
    output_dir = Path(args.output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Override global for collection function
    global TEST_DURATION_MINUTES
    TEST_DURATION_MINUTES = test_duration
    
    # Collect metrics
    result, timeseries = collect_all_metrics(args.mode)
    
    # Generate timestamp for filenames
    file_ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_file = output_dir / f"{args.mode}_metrics_{file_ts}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OUTPUT] Metrics saved to: {json_file}")
    
    # Export CSVs
    print("\n[OUTPUT] Exporting time series CSVs...")
    export_timeseries_csv(timeseries, args.mode, output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Duration: {test_duration} minutes")
    print(f"Total queries executed: {len(QUERIES_LOG)}")
    print(f"Output directory: {output_dir}")
    
    # Quick validation
    print("\n[VALIDATION]")
    errors = []
    if not result.get("cluster_hardware"):
        errors.append("Missing cluster hardware metadata")
    if not result.get("resource_requests"):
        errors.append("Missing resource requests")
    for svc in ["api", "app"]:
        if result["services"].get(svc, {}).get("p99_ms") is None:
            errors.append(f"{svc} p99 latency is empty — check Prometheus scraping")
    
    if errors:
        print("  WARNINGS:")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  All critical metrics collected successfully")
    
    return result


if __name__ == "__main__":
    main()
