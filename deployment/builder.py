import numpy as np
import requests
from collections import deque

PROM_URL = "http://localhost:9090"
OBS_DIM = 16

CPU_HISTORY = {}
RPS_HISTORY = {}

def _hist(store, key):
    if key not in store:
        store[key] = deque([0.0, 0.0], maxlen=2)
    return store[key]

def q(query: str) -> float:
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=5).json()
        if r["data"]["result"]:
            return float(r["data"]["result"][0]["value"][1])
    except:
        pass
    return 0.0

def collect_metrics(service: str, ns="default"):
    return {
        "cpu": q(f'''
          sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",pod=~"{service}-.*",container="{service}"}}[1m]))
          /
          sum(kube_pod_container_resource_limits{{namespace="{ns}",pod=~"{service}-.*",container="{service}",resource="cpu"}})
        '''),
        "memory": q(f'''
          sum(container_memory_working_set_bytes{{namespace="{ns}",pod=~"{service}-.*",container="{service}"}})
          /
          sum(kube_pod_container_resource_limits{{namespace="{ns}",pod=~"{service}-.*",container="{service}",resource="memory"}})
        '''),
        "rps": q(f'''
          sum(rate(envoy_http_downstream_rq_total{{namespace="{ns}",envoy_cluster_name="{service}"}}[1m]))
        '''),
        "queue": q(f'''
          avg(envoy_http_downstream_rq_active{{namespace="{ns}",envoy_cluster_name="{service}"}})
        '''),
        "p50": q(f'''
          histogram_quantile(0.50,
            sum(rate(envoy_http_downstream_rq_time_bucket{{namespace="{ns}",envoy_cluster_name="{service}"}}[1m])) by (le)
          ) * 1000
        '''),
        "p95": q(f'''
          histogram_quantile(0.95,
            sum(rate(envoy_http_downstream_rq_time_bucket{{namespace="{ns}",envoy_cluster_name="{service}"}}[1m])) by (le)
          ) * 1000
        '''),
        "p99": q(f'''
          histogram_quantile(0.99,
            sum(rate(envoy_http_downstream_rq_time_bucket{{namespace="{ns}",envoy_cluster_name="{service}"}}[1m])) by (le)
          ) * 1000
        '''),
        "error": q(f'''
          sum(rate(envoy_http_downstream_rq_xx{{namespace="{ns}",envoy_cluster_name="{service}",envoy_response_code_class="5"}}[1m]))
        '''),
        "desired": q(f'kube_deployment_spec_replicas{{deployment="{service}"}}'),
        "ready": q(f'sum(kube_pod_status_ready{{pod=~"{service}-.*",condition="true"}})')
    }

def build_observation(service: str, m: dict, max_rep=10):
    cpu_h = _hist(CPU_HISTORY, service)
    rps_h = _hist(RPS_HISTORY, service)

    cpu_d = m["cpu"] - cpu_h[-1]
    rps_d = m["rps"] - rps_h[-1]

    cpu_h.append(m["cpu"])
    rps_h.append(m["rps"])

    return np.array([
        min(m["cpu"]/2,1),
        min(m["memory"]/2,1),
        min(m["p50"]/100,2),
        min(m["p95"]/500,2),
        min(m["p99"]/1000,2),
        min(m["rps"]/500,2),
        min(m["error"],1),
        min(m["queue"]/100,2),
        np.tanh(rps_d/50),
        m["desired"]/max_rep,
        m["ready"]/max_rep,
        m["ready"]/max(m["desired"],1),
        min(cpu_h[-2]/2,1),
        np.tanh(cpu_d/0.5),
        0.0,
        0.0
    ], dtype=np.float32)
