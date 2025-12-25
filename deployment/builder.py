import numpy as np
import requests
from collections import deque
import os
import math

PROM_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
OBS_DIM = 16

CPU_HISTORY = {}
RPS_HISTORY = {}

LAST_Q = {}

def _hist(store, key):
    if key not in store:
        store[key] = deque([0.0, 0.0], maxlen=2)
    return store[key]

def q(query: str) -> float:
    try:
        r = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        ).json()

        if r.get("data", {}).get("result"):
            v = float(r["data"]["result"][0]["value"][1])
            if not (math.isnan(v) or math.isinf(v)):
                LAST_Q[query] = v
                return v
    except Exception:
        pass

    # fallback to last known value
    return LAST_Q.get(query, 0.0)



def collect_metrics(service: str, ns="default"):
    rps = q(f'''
      sum(rate(envoy_http_downstream_rq_total{{
        namespace="{ns}",
        job="{service}",
        envoy_http_conn_manager_prefix="ingress"
      }}[1m]))
    ''')

    avg_latency = q(f'''
      sum(rate(envoy_http_downstream_rq_time_sum{{
        namespace="{ns}",
        job="{service}",
        envoy_http_conn_manager_prefix="ingress"
      }}[1m]))
      /
      sum(rate(envoy_http_downstream_rq_total{{
        namespace="{ns}",
        job="{service}",
        envoy_http_conn_manager_prefix="ingress"
      }}[1m]))
    ''')

    if avg_latency == 0.0:
        avg_latency = 5.0

    return {
        "cpu": q(f'''
          sum(rate(container_cpu_usage_seconds_total{{
            namespace="{ns}",
            pod=~"{service}-.*",
            container="{service}"
          }}[1m]))
          /
          sum(kube_pod_container_resource_limits{{
            namespace="{ns}",
            pod=~"{service}-.*",
            container="{service}",
            resource="cpu"
          }})
        '''),

        "memory": q(f'''
          sum(container_memory_working_set_bytes{{
            namespace="{ns}",
            pod=~"{service}-.*",
            container="{service}"
          }})
          /
          sum(kube_pod_container_resource_limits{{
            namespace="{ns}",
            pod=~"{service}-.*",
            container="{service}",
            resource="memory"
          }})
        '''),

        "rps": rps,

        "queue": q(f'''
          avg(envoy_http_downstream_rq_active{{
            namespace="{ns}",
            job="{service}",
            envoy_http_conn_manager_prefix="ingress"
          }})
        '''),

        # Latencies in ms
        "p50": q(f'''
          histogram_quantile(
            0.50,
            sum by (le) (
              increase(envoy_http_downstream_rq_time_bucket{{
                namespace="{ns}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress"
              }}[1m])
            )
          )
        '''),


        "p95": q(f'''
          histogram_quantile(
            0.95,
            sum by (le) (
              increase(envoy_http_downstream_rq_time_bucket{{
                namespace="{ns}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress"
              }}[1m])
            )
          )
        '''),


        "p99": q(f'''
          histogram_quantile(
            0.99,
            sum by (le) (
              increase(envoy_http_downstream_rq_time_bucket{{
                namespace="{ns}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress"
              }}[1m])
            )
          )
        '''),


        "error": q(f'''
          sum(rate(envoy_http_downstream_rq_xx{{
            namespace="{ns}",
            job="{service}",
            envoy_http_conn_manager_prefix="ingress",
            envoy_response_code_class="5"
          }}[1m]))
        '''),

        "desired": q(f'kube_deployment_spec_replicas{{deployment="{service}"}}'),

        "ready": q(f'''
          sum(kube_pod_status_ready{{
            namespace="{ns}",
            pod=~"{service}-.*",
            condition="true"
          }})
        ''')
    }


def get_upstream_latency(service: str, metrics_cache: dict) -> float:
    UPSTREAM = {
        "app": ["api"],
        "api": ["db"],
        "db": [],
    }

    ups = UPSTREAM.get(service, [])
    vals = [metrics_cache[u]["p95"] for u in ups if u in metrics_cache]
    return float(np.mean(vals)) if vals else 0.0


def build_observation(service: str, m: dict, metrics_cache: dict, max_rep=20):
    cpu_h = _hist(CPU_HISTORY, service)
    rps_h = _hist(RPS_HISTORY, service)

    cpu_d = m["cpu"] - cpu_h[-1]
    rps_d = m["rps"] - rps_h[-1]

    cpu_h.append(m["cpu"])
    rps_h.append(m["rps"])

    DOWNSTREAM = {
        "api": [],
        "app": ["api"],
        "db": [],
    }

    downstream_queue = sum(
        metrics_cache[d]["queue"]
        for d in DOWNSTREAM.get(service, [])
        if d in metrics_cache
    )

    upstream_latency = get_upstream_latency(service, metrics_cache)

    return np.array([
        min(m["cpu"] / 2.0, 1.0),
        min(m["memory"] / 2.0, 1.0),

        min(m["p50"] / 100.0, 2.0),
        min(m["p95"] / 500.0, 2.0),
        min(m["p99"] / 1000.0, 2.0),

        min(m["rps"] / 500.0, 2.0),
        min(m["error"], 1.0),

        min(m["queue"] / 100.0, 2.0),
        np.tanh(rps_d / 50.0),

        m["desired"] / max_rep,
        m["ready"] / max_rep,
        m["ready"] / max(m["desired"], 1),

        min(cpu_h[-2] / 2.0, 1.0),
        np.tanh(cpu_d / 0.5),

        min(downstream_queue / 500.0, 1.0),
        min(upstream_latency / 200.0, 2.0),
    ], dtype=np.float32)
