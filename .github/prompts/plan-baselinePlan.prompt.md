Plan: Baseline Cost-Performance Measurement (No Autoscalers)

TL;DR: Collect a clean baseline measurement of the 3-tier app under realistic load with fixed replicas (api=2, app=3, db=1) and no autoscaler. Fix the production locustfile to use real endpoints, rebuild and redeploy it, implement a Prometheus-based metrics collector, add a theoretical GKE cost model (n2-standard-4), run a 60-minute load test, and save structured JSON + CSV outputs to `docs/Final Results/` for reproducibility.

PHASE 0 — PRELIMS

- Ensure Prometheus port-forward is active:
  kubectl port-forward -n monitoring pod/prometheus-kube-prom-kube-prometheus-prometheus-0 9090:9090

- Ensure agent controller is NOT running; ensure no HPAs exist for api/app.

- Create output directory:
  mkdir -p "docs/Final Results"

PHASE 1 — CLUSTER INSPECTION & NORMALIZATION

1. Inspect current cluster capacity (manual commands):
   kubectl get nodes -o wide
   kubectl describe nodes
   kubectl top nodes

   Record per-node:
   - CPU cores (capacity & allocatable)
   - Memory (capacity & allocatable)

2. Normalize pod requests/limits target (n2-standard-4 baseline):
   Target per-pod requests (fixed for scheduling):
   - api: 500m CPU, 512Mi RAM
   - app: 500m CPU, 512Mi RAM
   - db: 1000m CPU, 1Gi RAM

   Files to edit:
   - infra/manifests/three-tier/api.yaml
   - infra/manifests/three-tier/app.yaml
   - infra/manifests/three-tier/db.yaml

   Re-apply manifests with kubectl apply -f <file> or kubectl rollout restart.

3. Set fixed replicas (no autoscaler):
   kubectl scale deployment api --replicas=2
   kubectl scale deployment app --replicas=3
   kubectl scale deployment db --replicas=1

   Verify:
   kubectl describe deployment api
   kubectl describe deployment app
   kubectl describe deployment db

PHASE 2 — LOCUST: FIX & DEPLOY PRODUCTION PROFILE

Problem: `locustfile_production.py` uses non-existent endpoints (/api/items, /api/search, /api/action). Those yield 404s and corrupt metrics.

1. Update `microservices/locust/locustfile_production.py`:
   - Keep `UniversityUser` types and `ProductionDayShape` phases.
   - Replace invalid endpoints with real endpoints supported by the app/api services:
     - GET /  (homepage)
     - GET /api/quotes  (list)
     - POST /api/quotes  (create) — send JSON {"text":"...","author":"..."}
   - Adjust weights/gating logic to use these endpoints. Keep the same wait times and spawn shape.

2. Update Dockerfile (`microservices/locust/Dockerfile`):
   - Copy the production locustfile as `/locustfile.py` (or modify args to point specifically to `locustfile_production.py`). Example:
     COPY locustfile_production.py /locustfile.py
     COPY locustfile.py /locustfile_simple.py

3. Rebuild & import into k3d:
   docker build -t project-locust:local microservices/locust/
   k3d image import project-locust:local -c aura
   kubectl rollout restart deployment/locust

4. Start Locust UI (port-forward):
   kubectl port-forward deployment/locust 8089:8089
   Open http://localhost:8089 and Start (ProductionDayShape will manage ramping)

PHASE 3 — PROMETHEUS SCRAPING MODULE (tools/gke_cost_report.py)

Rewrite `tools/gke_cost_report.py` as a deterministic, Prometheus-HTTP-API-only metrics collector. Requirements:

1. Use HTTP API endpoints:
   - Instant queries: /api/v1/query
   - Range queries: /api/v1/query_range (for time-series CSVs)
   - Use PROM_BASES fallback: 127.0.0.1, localhost, [::1]

2. Per-service metrics (namespace=default):
   For svc in [api, app, db]:
   - replicas: kube_deployment_spec_replicas{deployment="<svc>"}
   - cpu_avg (cores): sum(rate(container_cpu_usage_seconds_total{pod=~"<svc>-.*", container="<svc>", container!="POD"}[2m]))
   - memory_bytes: sum(container_memory_working_set_bytes{pod=~"<svc>-.*", container="<svc>"})
   - rps: sum(rate(envoy_http_downstream_rq_total{job="<svc>", envoy_http_conn_manager_prefix="ingress"}[2m]))
   - p50/p95/p99 (ms): histogram_quantile(0.50/0.95/0.99, sum by (le)(rate(envoy_http_downstream_rq_time_bucket{job="<svc>"}[2m])))
     - Note: db is TCP proxy — latency histograms not available (set null)
   - error_rate: sum(rate(envoy_http_downstream_rq_xx{job="<svc>", envoy_response_code_class="5"}[2m]))

3. Cluster-level totals:
   - total_cpu_used = sum of per-service cpu_avg
   - total_memory_used_gb = sum of per-service memory_bytes / 1e9
   - total_replicas = sum of per-service replicas
   - sla_violations = count of services where p99_ms > 2000

4. Time-series collection (query_range):
   - Step: 15s
   - Window: full test duration (e.g., 60m)
   - Export CSVs: replicas_over_time, p99_over_time, cpu_usage_over_time

5. Logging & reproducibility:
   - Save all raw PromQL queries in `queries_log` inside the JSON output
   - Use deterministic ISO8601 timestamps
   - Save JSON to `docs/Final Results/baseline_metrics_<TIMESTAMP>.json`

PHASE 4 — THEORETICAL GKE COST MODEL (tools/cost_profiler.py)

Implement `tools/cost_profiler.py` to consume the metrics JSON and compute estimates using n2-standard-4:

Assumptions:
- n2-standard-4 = 4 vCPU, 16GB RAM
- price_per_hour = $0.189
- cluster_fee_per_hour = $0.10
- month_hours = 730

Compute:
1. total_requested_cpu_cores = sum for each svc (replicas * cpu_request_in_cores)
   - Use pod request targets from PHASE 1
2. node_count_estimated = ceil(total_requested_cpu_cores / 4)
3. hourly_cost = (node_count_estimated * price_per_hour) + cluster_fee_per_hour
4. monthly_compute_usd = hourly_cost * 730
5. cost_per_replica_hour = hourly_cost / total_replicas
6. cost_per_1k_users_usd = monthly_compute_usd / (peak_users / 1000)  # peak_users from Locust shape

Append `cost_estimate` to the metrics JSON and write `docs/Final Results/baseline_cost_<TIMESTAMP>.json`.

PHASE 5 — RUN BASELINE LOAD TEST

1. Confirm Prometheus port-forwarding and that collector can query Prometheus.
2. Confirm agent controller not running and no HPA present.
3. Confirm fixed replicas: api=2, app=3, db=1.
4. Start Locust UI and Start test. Let the `ProductionDayShape` run the full 60-minute profile.
5. Immediately after steady-state/peak windows, run:
   python tools/gke_cost_report.py --mode baseline --out docs/Final\ Results/baseline_metrics_<TS>.json
   python tools/cost_profiler.py --mode baseline --metrics-file docs/Final\ Results/baseline_metrics_<TS>.json

PHASE 6 — EXPORTS & CSVs

Outputs to generate and save under `docs/Final Results/`:
- baseline_metrics_<TS>.json  (full structured JSON)
- baseline_cost_<TS>.json     (cost_estimate appended)
- replicas_over_time_baseline.csv
- p99_over_time_baseline.csv
- cpu_usage_over_time_baseline.csv
- cost_estimates_baseline.csv

CSV column formats:
- replicas_over_time: timestamp, api_replicas, app_replicas, db_replicas
- p99_over_time: timestamp, api_p99_ms, app_p99_ms
- cpu_usage_over_time: timestamp, api_cpu, app_cpu, db_cpu
- cost_estimates: timestamp, node_count_estimated, hourly_cost_usd, monthly_compute_usd, cost_per_replica_hour

PHASE 7 — VERIFICATION & SANITY CHECKS

- Validate JSON files parse: python -m json.tool <file>
- Spot-check PromQL queries logged in JSON and run these manually in browser/Prometheus UI.
- Confirm p99 values are in ms and that db p99 is null (TCP).
- Confirm node_count_estimated is computed from REQUESTS, not actual usage.

ASSUMPTIONS & NOTES (to include as comments in created scripts)

- All scraping uses Prometheus HTTP API only; no curl inside containers.
- Deterministic timestamps (ISO8601 UTC) used for filenames and inside JSON.
- All raw PromQL queries saved to `queries_log` for reproducibility.
- Use 15s step for time-series range queries to balance fidelity & query cost.
- Memory sanity-check: ensure total requested memory fits into estimated nodes (16GB/node).
- Cost model uses on-demand pricing and omits sustained discounts, preemptibles, or committed use savings.

NEXT STEPS (after plan refinement)

- Implement `tools/gke_cost_report.py` and `tools/cost_profiler.py`.
- Make the locustfile changes and Dockerfile edit, then rebuild image.
- Run the baseline test and collect outputs.

---

Timestamp: 2026-02-11T
Mode: baseline

End of plan.
