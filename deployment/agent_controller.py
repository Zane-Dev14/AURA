
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


def get_current_kube_context():
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        context = result.stdout.strip()
        return context or None
    except Exception:
        return None


def assert_k3d_context():
    context = get_current_kube_context()
    if not context:
        raise RuntimeError("kubectl current-context is empty. Refusing to run controller.")
    if not context.startswith("k3d-"):
        raise RuntimeError(f"Refusing to run controller on non-k3d context: {context}")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from marl.inference import AuraInference
from deployment.builder import collect_metrics, build_observation, _hist, RPS_HISTORY

CHECKPOINT_DIR = os.environ.get("AURA_CHECKPOINT_DIR", "marl/qmix_trained")
# Prefer the kube-prometheus-stack NodePort exposed by local k3d setup.
# You can override via PROMETHEUS_URL if using a different topology.
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:30090")
NAMESPACE = "default"

SERVICES = ["api", "app", "db"]

MIN_REPLICAS = {
    "api": 1,
    "app": 1,
    "db": 1
}

MAX_REPLICAS = 5
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


def app_needs_recovery(m):
    """
    Determine if APP tier needs recovery/scale-up.
    Uses more sensitive thresholds to prevent stuck-at-1-replica scenarios.
    """
    return (
        m.get("p99", 0) > P99_SLO * 0.7  # 350ms threshold (70% of 500ms SLO)
        or m.get("error", 0) > 0.015      # 1.5% error rate
        or m.get("queue", 0) > 15         # Lower queue threshold
        or (m.get("p99", 0) > 300 and m.get("rps", 0) > 100)  # Combined pressure signal
    )


def api_is_bottleneck(m):
    return m.get("desired", 0) >= MAX_REPLICAS and m.get("queue", 0) > 500

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
    assert_k3d_context()

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
            
            # Veto 1b: Aggressive scale-down override (for overtrained agents)
            # If agent wants to scale up/maintain but conditions are excellent, force scale-down
            if actions[svc] >= 0:  # Agent wants to maintain or scale up
                current_replicas = int(m["desired"])
                current_p99 = m["p99"]
                current_rps = m.get("rps", 0) or 0
                current_cpu = m.get("cpu", 0)
                
                # Force scale-down if: excellent latency + low load + high replicas
                if (current_p99 < 50 and current_cpu < 0.6 and current_replicas > 2):
                    print(f"🔧 OVERRIDE: {svc} forcing scale-down (p99={current_p99:.0f}ms, cpu={current_cpu*100:.0f}%, replicas={current_replicas} > 2)")
                    actions[svc] = -1

            # Veto 2: Tier-coupled (don't scale APP if API is bottleneck)
            if svc == "app" and actions[svc] > 0:
                api_metrics = metrics_cache.get("api", {})
                if api_is_bottleneck(api_metrics) and not app_needs_recovery(m):
                    print(
                        f"⚠️ TIER VETO: API bottleneck (q={api_metrics.get('queue', 0):.0f}), "
                        "blocking APP scale-up until APP shows pressure"
                    )
                    actions[svc] = 0

            # Recovery override: never leave APP stuck at 1 replica when it is
            # already breaching its own SLO or showing sustained queue/error pressure.
            if svc == "app" and actions[svc] <= 0 and app_needs_recovery(m):
                print(
                    f"↺ APP RECOVERY OVERRIDE: p99={m.get('p99', 0):.0f}ms, "
                    f"err={m.get('error', 0):.3f}, queue={m.get('queue', 0):.1f}"
                )
                actions[svc] = 1

            # Veto 3: Multi-tier SLO check with escape hatches (intelligent scale-down)
            if actions[svc] < 0:
                current_p99 = m["p99"]
                current_rps = m.get("rps", 0) or 0
                
                # Tier A - ESCAPE HATCH 1: No traffic (allow scale-down immediately)
                if current_rps < 50:
                    print(f"✓ SCALE-DOWN ALLOWED: {svc} - Low traffic (rps={current_rps:.0f} < 50)")
                
                # Tier B - ESCAPE HATCH 2: Current state excellent (allow scale-down)
                elif current_p99 < 100:
                    print(f"✓ SCALE-DOWN ALLOWED: {svc} - Excellent latency (p99={current_p99:.0f}ms < 100ms)")
                
                # Tier C - MARGINAL ZONE: Allow with warning (100 ≤ p99 < 500)
                elif current_p99 < 500:
                    print(f"⚠️  SCALE-DOWN MARGINAL: {svc} - Acceptable latency (p99={current_p99:.0f}ms in [100,500)ms)")
                
                # Tier D - SLO BREACH ZONE: Check smoothed metrics before blocking
                else:  # current_p99 >= 500
                    try:
                        prom_query = f"avg_over_time(histogram_quantile(0.99, sum by (le) (increase(envoy_http_downstream_rq_time_bucket{{namespace=\"{NAMESPACE}\",job=\"{svc}\",envoy_http_conn_manager_prefix=\"ingress\"}}[2m])))[{P99_WINDOW}])"
                        smoothed_p99 = requests.get(
                            f"{PROMETHEUS_URL}/api/v1/query",
                            params={"query": prom_query},
                            timeout=5
                        ).json()
                        if smoothed_p99["data"]["result"]:
                            smoothed_p99_val = float(smoothed_p99["data"]["result"][0]["value"][1])
                        else:
                            smoothed_p99_val = current_p99
                    except Exception:
                        smoothed_p99_val = current_p99
                    
                    # Only block if smoothed average confirms sustained violation
                    if smoothed_p99_val > P99_SLO:
                        print(f"🚫 SLO VETO: {svc} - Sustained violation (smoothed={smoothed_p99_val:.0f}ms, current={current_p99:.0f}ms, SLO={P99_SLO}ms)")
                        actions[svc] = 0
                    else:
                        print(f"✓ SCALE-DOWN ALLOWED: {svc} - Escape Hatch 3 (spike recovered: smoothed={smoothed_p99_val:.0f}ms ≤ SLO, current={current_p99:.0f}ms)")

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
