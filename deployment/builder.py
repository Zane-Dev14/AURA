import numpy as np
import requests
from collections import deque
import os

PROM_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
OBS_DIM = 16

CPU_HISTORY = {}
RPS_HISTORY = {}

def _hist(store, key):
    if key not in store:
        store[key] = deque([0.0, 0.0], maxlen=2)
    return store[key]

import math

def q(query: str) -> float:
    try:
        r = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        ).json()

        if r["data"]["result"]:
            v = float(r["data"]["result"][0]["value"][1])
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return v
    except Exception:
        pass
    return 0.0


def collect_metrics(service: str, ns="default"):
    # Ingress RPS (ground truth)
    rps = q(f'''
      sum(rate(envoy_http_downstream_rq_total{{
        namespace="{ns}",
        job="{service}",
        envoy_http_conn_manager_prefix="ingress"
      }}[1m]))
    ''')

    # Always-safe average latency (ms)
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
    
    # ✅ FIX: Safety floor
    if avg_latency == 0.0:
        avg_latency = 5.0  # reasonable default: 5ms

    # Quantiles only if traffic is sufficient
    if rps > 50:
        p50 = q(f'''
          histogram_quantile(
            0.50,
            sum by (le) (
              increase(envoy_http_downstream_rq_time_bucket{{
                namespace="{ns}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress"
              }}[2m])
            )
          )
        ''')

        p95 = q(f'''
          histogram_quantile(
            0.95,
            sum by (le) (
              increase(envoy_http_downstream_rq_time_bucket{{
                namespace="{ns}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress"
              }}[2m])
            )
          )
        ''')

        p99 = q(f'''
          histogram_quantile(
            0.99,
            sum by (le) (
              increase(envoy_http_downstream_rq_time_bucket{{
                namespace="{ns}",
                job="{service}",
                envoy_http_conn_manager_prefix="ingress"
              }}[2m])
            )
          )
        ''')
    else:
        # Low-traffic safe approximation
        p50 = avg_latency
        p95 = avg_latency * 2.5
        p99 = avg_latency * 4.0

    # NaN → sane fallback
    if p50 == 0.0: p50 = avg_latency
    if p95 == 0.0: p95 = avg_latency * 2.5
    if p99 == 0.0: p99 = avg_latency * 4.0

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

        # Latency in ms — NO MULTIPLIERS
        "p50": p50,
        "p95": p95,
        "p99": p99,

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


def build_observation(service: str, m: dict, max_rep=10):
    cpu_h = _hist(CPU_HISTORY, service)
    rps_h = _hist(RPS_HISTORY, service)

    cpu_d = m["cpu"] - cpu_h[-1]
    rps_d = m["rps"] - rps_h[-1]

    cpu_h.append(m["cpu"])
    rps_h.append(m["rps"])

    p50_clamped = min(m["p50"], 5000)
    p95_clamped = min(m["p95"], 10000)
    p99_clamped = min(m["p99"], 15000)

    return np.array([
        min(m["cpu"]/2, 1),
        min(m["memory"]/2, 1),
        np.log1p(p50_clamped) / np.log1p(5000),
        np.log1p(p95_clamped) / np.log1p(10000),
        np.log1p(p99_clamped) / np.log1p(15000),
        min(m["rps"]/500, 2),
        min(m["error"], 1),
        min(m["queue"]/100, 2),
        np.tanh(rps_d/50),
        m["desired"]/max_rep,
        m["ready"]/max_rep,
        m["ready"]/max(m["desired"], 1),
        min(cpu_h[-2]/2, 1),
        np.tanh(cpu_d/0.5),
        0.0,
        0.0
    ], dtype=np.float32)