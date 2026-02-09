
#!/usr/bin/env python3
import sys
import os
import time
import subprocess
import requests
import csv
import numpy as np

from datetime import datetime, timedelta, timezone

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

            obs[svc] = build_observation(svc, metrics)


        try:
            actions = agent.predict(obs)
        except Exception as e:
            print("❌ Inference failed:", e)
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

                    # If last 2 cycles didn't increase RPS, scaling is futile
                    if rps_gain <= 0 and m["rps"] > 100:
                        print(f"⚠️ ELASTICITY VETO: {svc} scaling futile (Δrps={rps_gain:.1f})")
                        actions[svc] = 0

            # Veto 2: Tier-coupled (don't scale APP if API is bottleneck)
            if svc == "app" and actions[svc] > 0:
                if "api" in metrics_cache:
                    api_queue = metrics_cache["api"]["queue"]
                    api_replicas = metrics_cache["api"]["desired"]

                    # API maxed + high queue = bottleneck upstream
                    if api_replicas >= MAX_REPLICAS and api_queue > 500:
                        print(f"⚠️ TIER VETO: API bottleneck (q={api_queue:.0f}), blocking APP scale-up")
                        actions[svc] = 0
            if actions[svc] < 0:  # Trying to scale down
                if m["p99"] > 500:  # 500ms threshold
                    print(f"⚠️ LATENCY VETO: {svc} p99={m['p99']:.0f}ms, blocking scale-down")
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
