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

MIN_REPLICAS = {
    "api": 2,
    "app": 3,
    "db": 1
}

MAX_REPLICAS = 10
COOLDOWN_SEC = 30

SHADOW_MODE = os.environ.get("AURA_SHADOW_MODE", "true").lower() == "true"

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "shadow_decisions.csv")

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
    min_r = MIN_REPLICAS.get(service, 1)
    max_r = MAX_REPLICAS

    clamped = max(min_r, min(max_r, replicas))

    if clamped != replicas:
        print(f"⚠️  Clamped {service} replicas {replicas} → {clamped}")

    subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "scale",
            "deployment",
            service,
            f"--replicas={clamped}",
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
            obs[svc] = build_observation(svc, metrics)

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
        # -------------------------------------------------
        # Apply / log actions
        # -------------------------------------------------
        print("\n" + "="*80)
        print(f"⏰ DECISION CYCLE @ {datetime.utcnow().isoformat()}")
        print("="*80)

        for svc in SERVICES:
            m = metrics_cache[svc]
            current = int(m["ready"])
            delta = actions[svc]
            delta = max(-1, min(1, delta))
            target = current + delta
            target = max(MIN_REPLICAS[svc], min(MAX_REPLICAS, target))


            print(f"\n🔷 [{svc.upper()}]")
            print(f"  📊 METRICS:")
            print(f"     CPU:       {m['cpu']:.3f} ({m['cpu']*100:.1f}%)")
            print(f"     Memory:    {m['memory']:.3f} ({m['memory']*100:.1f}%)")
            print(f"     RPS:       {m['rps']:.2f} req/s")
            print(f"     Queue:     {m['queue']:.1f} active")
            print(f"     Errors:    {m['error']:.4f} err/s")
            print(f"     P50:       {m['p50']:.2f} ms")
            print(f"     P95:       {m['p95']:.2f} ms")
            print(f"     P99:       {m['p99']:.2f} ms")
            print(f"  🎯 REPLICAS:")
            print(f"     Current:   {current}")
            print(f"     Desired:   {int(m['desired'])}")
            print(f"     Ready:     {int(m['ready'])}")
            print(f"  🤖 ACTION:")
            print(f"     Delta:     {delta:+d}")
            print(f"     Target:    {target}")
            print(f"     Mode:      {'🔒 SHADOW' if SHADOW_MODE else '⚡ LIVE'}")

            # Show observation vector
            obs_vec = obs[svc]
            print(f"  🧮 OBSERVATION VECTOR (16-D):")
            print(f"     {obs_vec}")

            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    svc,
                    current,
                    delta,
                    target,
                    m["p95"]
                ])

            if not SHADOW_MODE:
                print(f"  ⚡ EXECUTING: kubectl scale deployment {svc} --replicas={target}")
                scale(svc, target)

        print("\n" + "="*80 + "\n")
        last_action_time = time.time()

# -------------------------------------------------
if __name__ == "__main__":
    main()
