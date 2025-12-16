import numpy as np
from collections import deque

OBS_DIM = 16

# Keep short history for derivatives
CPU_HISTORY = {}
RPS_HISTORY = {}

def _get_hist(store, key, default=0.0):
    if key not in store:
        store[key] = deque([default, default], maxlen=2)
    return store[key]

def build_observation(
    service: str,
    metrics: dict,
    max_replicas: int = 10
) -> np.ndarray:
    """
    Reconstructs simulator-style observation vector
    using Prometheus metrics
    """

    cpu = metrics["cpu"]
    mem = metrics["memory"]
    rps = metrics["rps"]
    err = metrics["error_rate"]
    p50 = metrics["p50"]
    p95 = metrics["p95"]
    p99 = metrics["p99"]
    desired = metrics["desired"]
    ready = metrics["ready"]

    # Histories
    cpu_hist = _get_hist(CPU_HISTORY, service)
    rps_hist = _get_hist(RPS_HISTORY, service)

    cpu_deriv = cpu - cpu_hist[-1]
    rps_deriv = rps - rps_hist[-1]

    cpu_hist.append(cpu)
    rps_hist.append(rps)

    queue_proxy = min((p95 / 1000.0) * rps / 100.0, 2.0)

    obs = np.array([
        min(cpu / 2.0, 1.0),                     # 0 CPU
        min(mem / 2.0, 1.0),                     # 1 Memory
        min(p50 / 100.0, 2.0),                   # 2 p50
        min(p95 / 500.0, 2.0),                   # 3 p95
        min(p99 / 1000.0, 2.0),                  # 4 p99
        min(rps / 500.0, 2.0),                   # 5 RPS
        min(err, 1.0),                           # 6 Error rate
        queue_proxy,                             # 7 Queue proxy
        np.tanh(rps_deriv / 50.0),               # 8 ΔRPS
        desired / max_replicas,                  # 9 Desired replicas
        ready / max_replicas,                    #10 Ready replicas
        ready / max(desired, 1),                 #11 Readiness ratio
        min(cpu_hist[-2] / 2.0, 1.0),             #12 CPU history
        np.tanh(cpu_deriv / 0.5),                 #13 ΔCPU
        0.0,                                     #14 Downstream proxy (optional)
        0.0,                                     #15 Upstream proxy (optional)
    ], dtype=np.float32)

    return obs
