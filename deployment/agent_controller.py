#!/usr/bin/env python3
import sys
import os
import time
import subprocess
import requests
import csv
import numpy as np
from datetime import datetime

# -------------------------------------------------
# Force project root into PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# Imports
# -------------------------------------------------
from marl.inference import AuraInference
from deployment.builder import collect_metrics, build_observation

# -------------------------------------------------
# Config
# -------------------------------------------------
CHECKPOINT_DIR = os.environ.get(
    "AURA_CHECKPOINT_DIR",
    "marl/qmix_trained"
)

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
NAMESPACE = "default"

SERVICES = ["api", "app","db"]

MIN_REPLICAS = 1
MAX_REPLICAS = 10
COOLDOWN_SEC = 30

SHADOW_MODE = os.environ.get("AURA_SHADOW_MODE", "true").lower() == "true"

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "shadow_decisions.csv")
def log_scale_decision(svc, m, current, delta, target, shadow):
    print(
        f"[{svc.upper():<4}] "
        f"Δ={delta:+d} "
        f"{current}→{target} | "
        f"p95={m.get('p95', 0):.2f} "
        f"p99={m.get('p99', 0):.2f} | "
        f"cpu={m.get('cpu', 0)*100:.1f}% "
        f"rps={m.get('rps', 0):.1f} | "
        f"{'SHADOW' if shadow else 'LIVE'}"
    )

# -------------------------------------------------
# Prometheus helper
# -------------------------------------------------
# def prom(query: str) -> float:
#     try:
#         r = requests.get(
#             f"{PROMETHEUS_URL}/api/v1/query",
#             params={"query": query},
#             timeout=5,
#         ).json()
#         if r.get("data", {}).get("result"):
#             return float(r["data"]["result"][0]["value"][1])
#     except Exception as e:
#         print("⚠️ Prometheus error:", e)
#     return 0.0

# -------------------------------------------------
# Metric collection (raw Prometheus → dict)
# -------------------------------------------------
# def collect_metrics(service: str) -> dict:
#     return {
#         "cpu": prom(
#             f'rate(container_cpu_usage_seconds_total{{pod=~"{service}-.*"}}[1m])'
#         ),
#         "memory": prom(
#             f'container_memory_working_set_bytes{{pod=~"{service}-.*"}}'
#         ) / 1e9,
#         "rps": prom(
#             f'rate(http_requests_total{{service="{service}"}}[1m])'
#         ),
#         "error_rate": prom(
#             f'rate(http_requests_total{{service="{service}",status=~"5.."}}[1m])'
#         ),
#         "p50": prom(
#             f'histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
#         ),
#         "p95": prom(
#             f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
#         ),
#         "p99": prom(
#             f'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
#         ),
#         "desired": prom(
#             f'kube_deployment_spec_replicas{{deployment="{service}"}}'
#         ),
#         "ready": prom(
#             f'kube_deployment_status_replicas_available{{deployment="{service}"}}'
#         ),
#     }

# -------------------------------------------------
# Scaling helper
# -------------------------------------------------
def scale(service: str, replicas: int):
    subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "scale",
            "deployment",
            service,
            f"--replicas={replicas}",
        ],
        check=False,
    )



# -------------------------------------------------
# Main loop
# -------------------------------------------------
def main():
    print("✅ AURA Agent Controller Started")
    print("🎭 Shadow mode:", SHADOW_MODE)
    print("📦 Checkpoints:", CHECKPOINT_DIR)

    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "service",
                "current_replicas",
                "action_delta",
                "target_replicas",
                "p95_latency"
            ])

    agent = AuraInference(CHECKPOINT_DIR)
    print("✅ MARL inference loaded")

    last_action_time = 0.0

    while True:
        time.sleep(5)

        if time.time() - last_action_time < COOLDOWN_SEC:
            continue

        # -------------------------------------------------
        # Build observations (SIMULATOR-EQUIVALENT)
        # -------------------------------------------------
        obs = {}
        metrics_cache = {}

        for svc in SERVICES:
            metrics = collect_metrics(svc)
            metrics_cache[svc] = metrics
            obs[svc] = build_observation(svc, metrics, metrics_cache)


        # -------------------------------------------------
        # Inference
        # -------------------------------------------------
        try:
            actions = agent.predict(obs)  # returns replica deltas
        except Exception as e:
            print("❌ Inference failed:", e)
            continue

        # -------------------------------------------------
        # Apply / log actions
        # -------------------------------------------------
        for svc in SERVICES:
            m = metrics_cache[svc]

            current = int(m["desired"])
            delta = max(-1, min(1, actions[svc]))
            target = max(MIN_REPLICAS, min(MAX_REPLICAS, current + delta))

            # 🔥 ONE-LINE IMPORTANT METRICS
            log_scale_decision(
                svc=svc,
                m=m,
                current=current,
                delta=delta,
                target=target,
                shadow=SHADOW_MODE
            )

            if not SHADOW_MODE:
                scale(svc, target)


        last_action_time = time.time()

# -------------------------------------------------
if __name__ == "__main__":
    main()
