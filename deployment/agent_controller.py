
#!/usr/bin/env python3
import sys
import os
import time
import subprocess
import requests
import csv
import numpy as np
from datetime import datetime, timedelta, timezone
P99_SLO = float(os.environ.get("AURA_P99_SLO", "500"))  # ms
P99_WINDOW = os.environ.get("AURA_P99_WINDOW", "5m")
QUEUE_METRIC_ENABLED = os.environ.get("AURA_QUEUE_METRIC_ENABLED", "true").lower() == "true"

IST = timezone(timedelta(hours=5, minutes=30))

def ist_time_str():
    return datetime.now(IST).strftime("%H:%M:%S")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from marl.inference import AuraInference
from deployment.builder import collect_metrics, build_observation, _hist, RPS_HISTORY

CHECKPOINT_DIR = os.environ.get("AURA_CHECKPOINT_DIR", "marl/qmix_trained")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
NAMESPACE = "default"

SERVICES = ["api", "app", "db"]

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


def log_scale_decision(svc, m, current, delta, target, shadow):
    print(
        f"[{svc.upper():<4}] "
        f"Δ={delta:+d} "
        f"{current}→{target} | "
        f"p95={m.get('p95', 0):.3f} "
        f"p99={m.get('p99', 0):.3f} | "
        f"cpu={m.get('cpu', 0)*100:.1f}% "
        f"rps={m.get('rps', 0):.1f} | "
        f"{'SHADOW' if shadow else 'LIVE'}",
        flush=True
    )

def scale(service: str, replicas: int):
    min_r = MIN_REPLICAS.get(service, 1)
    max_r = MAX_REPLICAS

    clamped = max(min_r, min(max_r, replicas))

    if clamped != replicas:
        print(f"⚠️  Clamped {service} replicas {replicas} → {clamped}")

    subprocess.run(
        ["kubectl", "-n", NAMESPACE, "scale", "deployment", service, f"--replicas={clamped}"],
        check=False,
    )

def main():
    print("✅ AURA Agent Controller Started")
    print("🎭 Shadow mode:", SHADOW_MODE)
    print("📦 Checkpoints:", CHECKPOINT_DIR)

    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOG_FILE, "w") as f:
        f.write("time     | svc | cur | Δ  | tgt | p99\n")

    agent = AuraInference(CHECKPOINT_DIR)
    print("✅ MARL inference loaded")


    last_action_time = 0.0

    while True:
        time.sleep(5)

        if time.time() - last_action_time < COOLDOWN_SEC:
            continue

        obs = {}
        metrics_cache = {}

        for svc in SERVICES:
            metrics = collect_metrics(svc)
            metrics_cache[svc] = metrics

            # Optionally override queue metric
            if QUEUE_METRIC_ENABLED:
                obs[svc] = build_observation(svc, metrics)
            else:
                # fallback: set queue to 0
                metrics_copy = dict(metrics)
                metrics_copy["queue"] = 0.0
                obs[svc] = build_observation(svc, metrics_copy)


        try:
            actions = agent.predict(obs)
        except Exception as e:
            print(" Inference failed:", e)
            continue

        # ==============================
        # VETOES (No new observations)
        # ==============================

        for svc in SERVICES:
            m = metrics_cache[svc]

            # Veto 1: Elasticity check (don't scale if not helping)
            if actions[svc] > 0:  # Trying to scale up
                rps_h = _hist(RPS_HISTORY, svc)
                if len(rps_h) >= 2:
                    rps_gain = m["rps"] - rps_h[-2]
                    if rps_gain <= 0 and m["rps"] > 100:
                        print(f"⚠️ ELASTICITY VETO: {svc} scaling futile (Δrps={rps_gain:.1f})")
                        actions[svc] = 0

            # Veto 2: Tier-coupled (don't scale APP if API is bottleneck)
            if svc == "app" and actions[svc] > 0:
                if "api" in metrics_cache:
                    api_queue = metrics_cache["api"]["queue"]
                    api_replicas = metrics_cache["api"]["desired"]
                    if api_replicas >= MAX_REPLICAS and api_queue > 500:
                        print(f"⚠️ TIER VETO: API bottleneck (q={api_queue:.0f}), blocking APP scale-up")
                        actions[svc] = 0

            # Veto 3: Smoothed p99 SLO (scale-down only if below SLO)
            if actions[svc] < 0:
                # Smoothed p99: avg_over_time(histogram_quantile(0.99, ...)[P99_WINDOW])
                try:
                    prom_query = f"avg_over_time(histogram_quantile(0.99, sum by (le) (increase(envoy_http_downstream_rq_time_bucket{{namespace=\"{NAMESPACE}\",job=\"{svc}\",envoy_http_conn_manager_prefix=\"ingress\"}}[2m])))[{P99_WINDOW}])"
                    smoothed_p99 = requests.get(
                        f"{PROMETHEUS_URL}/api/v1/query",
                        params={"query": prom_query},
                        timeout=5
                    ).json()
                    if smoothed_p99["data"]["result"]:
                        p99_val = float(smoothed_p99["data"]["result"][0]["value"][1])
                    else:
                        p99_val = m["p99"]
                except Exception:
                    p99_val = m["p99"]
                if p99_val > P99_SLO:
                    print(f"⚠️ SLO VETO: {svc} smoothed p99={p99_val:.0f}ms > SLO={P99_SLO}ms, blocking scale-down")
                    actions[svc] = 0

        # ==============================
        # Apply actions
        # ==============================

        for svc in SERVICES:
            m = metrics_cache[svc]

            current = int(m["desired"])
            delta = max(-1, min(1, actions[svc]))
            min_r = MIN_REPLICAS.get(svc, 1)
            target = max(min_r, min(MAX_REPLICAS, current + delta))

            log_scale_decision(
                svc=svc,
                m=m,
                current=current,
                delta=delta,
                target=target,
                shadow=SHADOW_MODE
            )

            with open(LOG_FILE, "a") as f:
                f.write(
                    f"{ist_time_str():<8} | "
                    f"{svc:<3} | "
                    f"{current:>3} | "
                    f"{delta:+2} | "
                    f"{target:>3} | "
                    f"{m.get('p99', 0.0):>7.4f}\n"
                )

            if not SHADOW_MODE and target != current:
                scale(svc, target)

        print("\n" + "="*80 + "\n")
        last_action_time = time.time()

if __name__ == "__main__":
    main()
