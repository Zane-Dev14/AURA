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
<<<<<<< Updated upstream
=======
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
              }}[1m])
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
              }}[1m])
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
              }}[1m])
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

>>>>>>> Stashed changes
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

        # ✅ REAL ingress RPS
        "rps": q(f'''
          sum(rate(envoy_http_downstream_rq_total{{
            namespace="{ns}",
            job="{service}",
            envoy_http_conn_manager_prefix="ingress"
          }}[1m]))
        '''),

        # ✅ Active ingress queue
        "queue": q(f'''
          avg(envoy_http_downstream_rq_active{{
            namespace="{ns}",
            job="{service}",
            envoy_http_conn_manager_prefix="ingress"
          }})
        '''),

        # ✅ Latencies in ms
        "p50": q(f'''
          histogram_quantile(
            0.50,
            sum(rate(envoy_http_downstream_rq_time_bucket{{
              namespace="{ns}",
              job="{service}",
              envoy_http_conn_manager_prefix="ingress"
            }}[1m])) by (le)
          ) * 1000
        '''),

        "p95": q(f'''
          histogram_quantile(
            0.95,
            sum(rate(envoy_http_downstream_rq_time_bucket{{
              namespace="{ns}",
              job="{service}",
              envoy_http_conn_manager_prefix="ingress"
            }}[1m])) by (le)
          ) * 1000
        '''),

        "p99": q(f'''
          histogram_quantile(
            0.99,
            sum(rate(envoy_http_downstream_rq_time_bucket{{
              namespace="{ns}",
              job="{service}",
              envoy_http_conn_manager_prefix="ingress"
            }}[1m])) by (le)
          ) * 1000
        '''),

        "error": q(f'''
          sum(rate(envoy_http_downstream_rq_xx{{
            namespace="{ns}",
            job="{service}",
            envoy_http_conn_manager_prefix="ingress",
            envoy_response_code_class="5"
          }}[1m]))
        '''),

        "desired": q(
          f'kube_deployment_spec_replicas{{deployment="{service}"}}'
        ),

        "ready": q(f'''
          sum(kube_pod_status_ready{{
            namespace="{ns}",
            pod=~"{service}-.*",
            condition="true"
          }})
        ''')
    }

def get_upstream_latency(service: str, metrics_cache: dict) -> float:
    """
    Match simulator upstream_latency:
    mean p95 latency of upstream services
    """
    UPSTREAM = {
        "app": ["api"],
        "api": ["db"],
        "db": [],
    }

    ups = UPSTREAM.get(service, [])
    if not ups:
        return 0.0

    vals = [metrics_cache[u]["p95"] for u in ups if u in metrics_cache]
    return float(np.mean(vals)) if vals else 0.0


def build_observation(service: str, m: dict, metrics_cache: dict, max_rep=20):
    cpu_h = _hist(CPU_HISTORY, service)
    rps_h = _hist(RPS_HISTORY, service)

    cpu_d = m["cpu"] - cpu_h[-1]
    rps_d = m["rps"] - rps_h[-1]

    cpu_h.append(m["cpu"])
    rps_h.append(m["rps"])

<<<<<<< Updated upstream
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
=======
    # === Latencies (NO CLAMP, LINEAR SCALE) ===
    p50 = m["p50"]
    p95 = m["p95"]
    p99 = m["p99"]

    # === Queue signals ===
    local_queue = m["queue"]

    # Downstream queue = sum of queues of downstream services
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

    # === Upstream latency (p95 mean) ===
    upstream_latency = get_upstream_latency(service, metrics_cache)

    # === Observation vector (SIMULATOR-ALIGNED) ===
    return np.array([
        # [0-1] Resource utilization
        min(m["cpu"] / 2.0, 1.0),
        min(m["memory"] / 2.0, 1.0),

        # [2-4] Latency metrics (LINEAR, SAME DIVISORS)
        min(p50 / 100.0, 2.0),
        min(p95 / 500.0, 2.0),
        min(p99 / 1000.0, 2.0),

        # [5-6] Throughput
        min(m["rps"] / 500.0, 2.0),
        min(m["error"], 1.0),

        # [7-8] Queue state
        min(local_queue / 100.0, 2.0),
        np.tanh(rps_d / 50.0),

        # [9-11] Pod state
        m["desired"] / max_rep,
        m["ready"] / max_rep,
        m["ready"] / max(m["desired"], 1),

        # [12-13] Temporal features
        min(cpu_h[-2] / 2.0, 1.0),
        np.tanh(cpu_d / 0.5),

        # [14-15] Cross-service coordination
        min(downstream_queue / 500.0, 1.0),
        min(upstream_latency / 200.0, 2.0),
>>>>>>> Stashed changes
    ], dtype=np.float32)
