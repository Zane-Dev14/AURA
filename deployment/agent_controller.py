#!/usr/bin/env python3
import sys
import os
import time
import subprocess
import requests
import csv
import numpy as np

#Making logs better
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

def ist_time_str():
    return datetime.now(IST).strftime("%H:%M:%S")

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
        f"p95={m.get('p95', 0):.3f} "
        f"p99={m.get('p99', 0):.3f} | "
        f"cpu={m.get('cpu', 0)*100:.1f}% "
        f"rps={m.get('rps', 0):.1f} | "
        f"{'SHADOW' if shadow else 'LIVE'}",
        flush=True
    )


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

    with open(LOG_FILE, "w") as f:
        f.write(
            "time     | svc | cur | Δ  | tgt | p99\n"
        )



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


        last_action_time = time.time()

# -------------------------------------------------
if __name__ == "__main__":
    main()
