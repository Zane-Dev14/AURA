
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
APP_RECOVERY_P99 = float(os.environ.get("AURA_APP_RECOVERY_P99", "150"))  # ms
APP_RECOVERY_RPS = float(os.environ.get("AURA_APP_RECOVERY_RPS", "120"))
APP_RECOVERY_ERROR = float(os.environ.get("AURA_APP_RECOVERY_ERROR", "0.01"))
APP_RECOVERY_QUEUE = float(os.environ.get("AURA_APP_RECOVERY_QUEUE", "10"))

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
# Prometheus is accessible on localhost:30090 (NodePort mapping)
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
# Asymmetric cooldowns: keep scale-up responsive, slow down scale-down
SCALE_UP_COOLDOWN_SEC = int(os.environ.get("AURA_SCALE_UP_COOLDOWN_SEC", "60"))  # increased to 60s for stabilization
SCALE_DOWN_COOLDOWN_SEC = int(os.environ.get("AURA_SCALE_DOWN_COOLDOWN_SEC", "30"))

# Minimal scale-up assist (keeps QMIX as primary, only resolves obvious suppression)
SCALEUP_P99_TRIGGER_MS = float(os.environ.get("AURA_SCALEUP_P99_TRIGGER_MS", "250"))
SCALEUP_CPU_FLOOR = float(os.environ.get("AURA_SCALEUP_CPU_FLOOR", "0.40"))
SCALEUP_TREND_TRIGGER = float(os.environ.get("AURA_SCALEUP_TREND_TRIGGER", "200"))  # RPS/min
ELASTICITY_VETO_GRACE_SEC = int(os.environ.get("AURA_ELASTICITY_VETO_GRACE_SEC", "300"))

SHADOW_MODE = os.environ.get("AURA_SHADOW_MODE", "true").lower() == "true"

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "shadow_decisions.csv")


def log_scale_decision(svc, m, current, delta, target, shadow, rps_trend=0.0):
    """Enhanced logging to show predictive behavior"""
    trend_arrow = "↑" if rps_trend > 10 else ("↓" if rps_trend < -10 else "→")
    trend_str = f"{trend_arrow}{abs(rps_trend):.0f}" if abs(rps_trend) > 1 else "~0"
    
    print(
        f"[{svc.upper():<4}] "
        f"Δ={delta:+d} "
        f"{current}→{target} | "
        f"rps={m.get('rps', 0):.1f} (trend:{trend_str} RPS/min) | "
        f"p99={m.get('p99', 0):.1f}ms "
        f"cpu={m.get('cpu', 0)*100:.0f}% | "
        f"{'SHADOW' if shadow else 'LIVE'}",
        flush=True
    )


def app_needs_recovery(m):
    """
    Determine if APP tier needs recovery/scale-up.
    Uses aggressive but still load-based thresholds to prevent stuck-at-1-replica
    scenarios when the app is visibly overloaded.
    """
    return (
        m.get("p99", 0) >= APP_RECOVERY_P99
        or m.get("rps", 0) >= APP_RECOVERY_RPS
        or m.get("error", 0) >= APP_RECOVERY_ERROR
        or m.get("queue", 0) >= APP_RECOVERY_QUEUE
        or (m.get("p99", 0) >= 100 and m.get("rps", 0) >= 80 and m.get("cpu", 0) >= 0.75)
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


    controller_start_time = time.time()
    last_up_action_time = {}
    last_down_action_time = {}
    trend_strikes = {}
    last_scale_state = {}

    while True:
        time.sleep(5)

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
        # PREDICTIVE AUGMENTATION & VETOES
        # ==============================

        # ==============================
        # FUSED CONTROL LOGIC (Target Replicas & Action Deduplication)
        # ==============================

        import math
        
        for svc in SERVICES:
            m = metrics_cache[svc]
            current = int(m.get("desired", 1))
            elapsed = time.time() - controller_start_time
            rps_h = _hist(RPS_HISTORY, svc)
            
            # Feature extraction
            rps_trend = 0.0
            if len(rps_h) >= 3:
                rps_trend = (m.get("rps", 0) - rps_h[-2]) * 12

            # 1. MARL Target
            marl_delta = max(-1, min(1, actions[svc]))
            marl_target = current + marl_delta

            # 2. Predictive Target (Capacity-based + Hysteresis)
            predictive_target = 0
            CAPACITY_PER_POD = 120.0  # safe heuristic
            
            if rps_trend > 150 and m.get("cpu", 0) > 0.40:
                trend_strikes[svc] = trend_strikes.get(svc, 0) + 1
            else:
                trend_strikes[svc] = max(0, trend_strikes.get(svc, 0) - 1)

            if trend_strikes.get(svc, 0) >= 2:
                # Forecast load 1 minute ahead based on trend, calculate total capacity needed
                predicted_rps = m.get("rps", 0) + rps_trend
                predictive_target = int(math.ceil(predicted_rps / CAPACITY_PER_POD))
                print(f"🚀 PREDICTIVE TARGET: {svc} (trend={rps_trend:.0f} RPS/m, target_reps={predictive_target})")
                trend_strikes[svc] = 0  # reset hysteresis 

            # 3. Reactive Safety Target (Overrides)
            override_target = 0
            if m.get("p99", 0) > 400.0 and m.get("cpu", 0) > 0.60:
                print(f"↺ SAFETY NET: {svc} forcing +1 (p99 > 400ms)")
                override_target = current + 1
            if svc == "app" and app_needs_recovery(m):
                print(f"↺ APP RECOVERY OVERRIDE activated")
                override_target = current + 1

            # 4. Merge Controllers BEFORE actuation (Max of all models)
            desired_target = max(marl_target, predictive_target, override_target)

            # 5. Enforce Elasticity Veto, Downscale Guards & Bottleneck Detection
            if desired_target > current:
                # A) Bottleneck Classifier: if High Latency + Low CPU + Stable RPS ⇒ External Bottleneck
                if m.get("p99", 0) > 300.0 and m.get("cpu", 0) < 0.45 and abs(rps_trend) < 50:
                    print(f"🛑 EXTERNAL BOTTLENECK DETECTED: {svc} (p99={m.get('p99',0):.0f}ms, cpu={m.get('cpu',0)*100:.0f}%, flat RPS). Blocking scale-up.")
                    desired_target = current
                
                # B) Convergence Guard (Marginal Gain Check)
                if desired_target > current and svc in last_scale_state:
                    prev_state = last_scale_state[svc]
                    # if we scaled up recently and p99 didn't improve by at least 5%
                    if prev_state["dir"] > 0 and m.get("p99", 0) >= prev_state["p99"] * 0.95 and m.get("p99", 0) > 150:
                        print(f"🛑 CONVERGENCE GUARD: {svc} previous +1 replica yielded <5% latency improvement (was {prev_state['p99']:.0f}ms). Stopping runaway scaling.")
                        desired_target = current

                # C) Elasticity veto: scaling up but not generating more RPS
                if desired_target > current and len(rps_h) >= 2:
                    rps_gain = m.get("rps", 0) - rps_h[-2]
                    if (elapsed >= ELASTICITY_VETO_GRACE_SEC and rps_gain <= 0 
                        and m.get("rps", 0) > 100 and m.get("p99", 0) < SCALEUP_P99_TRIGGER_MS 
                        and m.get("cpu", 0) < SCALEUP_CPU_FLOOR):
                        print(f"⚠️ ELASTICITY VETO: {svc} scaling futile (Δrps={rps_gain:.1f})")
                        desired_target = current
                        
                # D) Tier-coupled veto (don't scale APP if API is bottleneck)
                if svc == "app" and desired_target > current:
                    api_metrics = metrics_cache.get("api", {})
                    if api_is_bottleneck(api_metrics) and not app_needs_recovery(m):
                        print(f"⚠️ TIER VETO: API bottleneck, blocking APP scale-up")
                        desired_target = current

            elif desired_target < current:
                # Strict downscale blocked check
                cpu_pct = m.get("cpu", 0) * 100
                current_rps = m.get("rps", 0)
                if cpu_pct > 70 or current_rps > 50 or rps_trend > 0:
                    print(f"🚫 DOWNSCALE BLOCKED: {svc} (cpu={cpu_pct:.0f}%, rps={current_rps:.1f}, trend={rps_trend:.1f})")
                    desired_target = current

            # 6. Target clamping and Action Deduplication (Per-Service Cooldown)
            min_r = MIN_REPLICAS.get(svc, 1)
            target = max(min_r, min(MAX_REPLICAS, desired_target))
            delta = target - current

            now = time.time()
            if delta > 0 and (now - last_up_action_time.get(svc, 0.0)) < SCALE_UP_COOLDOWN_SEC:
                delta = 0
                target = current
            if delta < 0 and (now - last_down_action_time.get(svc, 0.0)) < SCALE_DOWN_COOLDOWN_SEC:
                delta = 0
                target = current

            # Apply
            log_scale_decision(
                svc=svc,
                m=m,
                current=current,
                delta=delta,
                target=target,
                shadow=SHADOW_MODE,
                rps_trend=rps_trend
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
                last_scale_state[svc] = {
                    "time": now, 
                    "dir": delta, 
                    "p99": m.get("p99", 0), 
                    "target": target
                }
                if delta > 0:
                    last_up_action_time[svc] = now
                elif delta < 0:
                    last_down_action_time[svc] = now

        print("\n" + "="*80 + "\n")
        # loop cadence controlled by sleep(5); per-direction cooldowns gate action frequency

if __name__ == "__main__":
    main()
