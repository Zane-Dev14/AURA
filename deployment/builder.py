
import numpy as np
import requests
from collections import deque
import os

PROM_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:30090")
OBS_DIM = 16

CPU_HISTORY = {}
RPS_HISTORY = {}

def _hist(store, key):
    if key not in store:
        store[key] = deque([0.0] * 20, maxlen=20)  # 20 samples = 100 seconds of history
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
    # Ingress RPS (ground truth) - FIXED: use _completed not _total
    rps = q(f'''
      sum(rate(envoy_http_downstream_rq_completed{{
        namespace="{ns}",
        job="{service}",
        envoy_http_conn_manager_prefix="ingress"
      }}[1m]))
    ''')

    # Always-safe average latency (ms) - FIXED: use _completed
    avg_latency = q(f'''
      sum(rate(envoy_http_downstream_rq_time_sum{{
        namespace="{ns}",
        job="{service}",
        envoy_http_conn_manager_prefix="ingress"
      }}[1m]))
      /
      sum(rate(envoy_http_downstream_rq_completed{{
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

    # Unified queue metric: downstream active requests (normalized)
    if service == "db":
        # DB is TCP proxy, use TCP connection metrics
        queue = q(f'''
          avg(envoy_tcp_downstream_cx_active{{
            namespace="{ns}",
            job="{service}"
          }})
        ''')
    else:
        queue = q(f'''
          avg(envoy_http_downstream_rq_active{{
            namespace="{ns}",
            job="{service}",
            envoy_http_conn_manager_prefix="ingress"
          }})
        ''')

    return {
        "cpu": q(f'''
          sum(rate(container_cpu_usage_seconds_total{{
            namespace="{ns}",
            pod=~"{service}-.*",
            container="{service}",
            container!="POD"
          }}[1m]))
          /
          sum(kube_pod_container_resource_requests{{
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
        "queue": queue,
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

    # === FIX 1: MATCH TRAINING DISTRIBUTION ===
    p50_clamped = min(m["p50"], 500)
    p95_clamped = min(m["p95"], 500)
    p99_clamped = min(m["p99"], 1000)

    # === FIX 2: GENERIC QUEUE PRESSURE (DOWNSTREAM) ===
    # envoy_http_downstream_rq_active already IS the queue
    downstream_pressure = min(m["queue"] / 500.0, 1.0)

    # === FIX 3: PREDICTIVE RPS DERIVATIVE ===
    if rps_d < 0 and m["error"] > 0.05:
        rps_signal = abs(rps_d) * 2
    else:
        rps_signal = rps_d

    return np.array([
        min(m["cpu"] / 2, 1),
        min(m["memory"] / 2, 1),
        np.log1p(p50_clamped) / np.log1p(500),
        np.log1p(p95_clamped) / np.log1p(500),
        np.log1p(p99_clamped) / np.log1p(1000),
        min(m["rps"] / 500, 2),
        min(m["error"], 1),
        min(m["queue"] / 100, 2),
        np.tanh(rps_signal / 50),
        m["desired"] / max_rep,
        m["ready"] / max_rep,
        m["ready"] / max(m["desired"], 1),
        min(cpu_h[-2] / 2, 1),
        np.tanh(cpu_d / 0.5),
        downstream_pressure,
        np.log1p(m["p95"]) / np.log1p(500)
    ], dtype=np.float32)
