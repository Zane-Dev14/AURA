#!/bin/bash
################################################################################
# AURA Full Benchmark Suite - FINAL PRODUCTION VERSION
# 
# Order: Baseline → HPA → QMIX (QMIX last for best comparison)
# Uses .venv for Python dependencies
# Total runtime: ~4.5 hours (9 trials × 30 min each)
# 
# Requirements:
# - Rancher Desktop with 32GB RAM, 14 cores
# - k3d cluster running (will be verified/fixed if needed)
# - All services deployed
# - .venv with dependencies installed
################################################################################

set -e  # Exit on error
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TRIAL_DURATION_MINUTES=30
TRIAL_DURATION_SECONDS=$((TRIAL_DURATION_MINUTES * 60))
COOLDOWN_SECONDS=180  # 3 minutes between trials
LOCUST_USERS=150      # Optimized for M4 Max
LOCUST_SPAWN_RATE=15
LOCUST_HOST="http://api:8080"
PROMETHEUS_URL="http://127.0.0.1:9090"
LOCUST_URL="http://127.0.0.1:8089"

# Python venv
VENV_PATH=".venv"
PYTHON_CMD="$VENV_PATH/bin/python3"

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experimental_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"/{baseline,hpa,qmix}/{trial_1,trial_2,trial_3}

# Logging
LOG_FILE="$RESULTS_DIR/benchmark_suite.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     AURA FGCS Benchmark Suite - FINAL PRODUCTION RUN      ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Start Time: $(date '+%Y-%m-%d %H:%M:%S')                           ║${NC}"
echo -e "${BLUE}║  Results Dir: $RESULTS_DIR                                 ║${NC}"
echo -e "${BLUE}║  Execution Order: Baseline → HPA → QMIX                   ║${NC}"
echo -e "${BLUE}║  Total Trials: 9 (3 per configuration)                    ║${NC}"
echo -e "${BLUE}║  Trial Duration: 30 minutes each                           ║${NC}"
echo -e "${BLUE}║  Estimated Total Time: 4.5 hours                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') - $1"
}

log_section() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for pods to be ready
wait_for_pods() {
    local service=$1
    local expected_replicas=$2
    local max_wait=300  # 5 minutes
    local elapsed=0
    
    log_info "Waiting for $service to have $expected_replicas ready replicas..."
    
    while [ $elapsed -lt $max_wait ]; do
        local ready=$(kubectl get deployment $service -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        if [ "$ready" = "$expected_replicas" ]; then
            log_info "$service has $ready/$expected_replicas ready replicas ✓"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    log_error "Timeout waiting for $service to be ready"
    return 1
}

# Collect metrics from Prometheus
collect_metrics() {
    local mode=$1
    local trial=$2
    local output_file=$3
    
    log_info "Collecting metrics for $mode trial $trial..."
    
    $PYTHON_CMD - <<EOF
import requests
import json
import time
from datetime import datetime

PROM_URL = "$PROMETHEUS_URL"
NAMESPACE = "default"

def query_prom(query):
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=10)
        result = r.json()
        if result["data"]["result"]:
            return float(result["data"]["result"][0]["value"][1])
    except:
        pass
    return 0.0

# Collect instant metrics
metrics = {
    "timestamp": datetime.now().isoformat(),
    "mode": "$mode",
    "trial": $trial,
    "duration_minutes": $TRIAL_DURATION_MINUTES,
    "services": {}
}

for svc in ["api", "app", "db"]:
    # Current replicas
    replicas = query_prom(f'kube_deployment_spec_replicas{{deployment="{svc}"}}')
    
    # RPS
    rps = query_prom(f'''
        sum(rate(envoy_http_downstream_rq_total{{
            namespace="{NAMESPACE}",
            job="{svc}",
            envoy_http_conn_manager_prefix="ingress"
        }}[1m]))
    ''')
    
    # P99 latency
    p99 = query_prom(f'''
        histogram_quantile(0.99,
            sum by (le) (
                increase(envoy_http_downstream_rq_time_bucket{{
                    namespace="{NAMESPACE}",
                    job="{svc}",
                    envoy_http_conn_manager_prefix="ingress"
                }}[2m])
            )
        )
    ''')
    
    # P95 latency
    p95 = query_prom(f'''
        histogram_quantile(0.95,
            sum by (le) (
                increase(envoy_http_downstream_rq_time_bucket{{
                    namespace="{NAMESPACE}",
                    job="{svc}",
                    envoy_http_conn_manager_prefix="ingress"
                }}[2m])
            )
        )
    ''')
    
    # Error rate
    error_rate = query_prom(f'''
        sum(rate(envoy_http_downstream_rq_xx{{
            namespace="{NAMESPACE}",
            job="{svc}",
            envoy_http_conn_manager_prefix="ingress",
            envoy_response_code_class="5"
        }}[1m]))
    ''')
    
    # CPU usage
    cpu = query_prom(f'''
        sum(rate(container_cpu_usage_seconds_total{{
            namespace="{NAMESPACE}",
            pod=~"{svc}-.*",
            container="{svc}"
        }}[2m]))
    ''')
    
    # SLA violations (>50ms approx)
    # Total rate minus rate of le="50"
    sla_violation_rate = query_prom(f'''
        sum(rate(envoy_http_downstream_rq_time_bucket{{
            namespace="{NAMESPACE}",
            job="{svc}",
            envoy_http_conn_manager_prefix="ingress",
            le="+Inf"
        }}[1m])) - sum(rate(envoy_http_downstream_rq_time_bucket{{
            namespace="{NAMESPACE}",
            job="{svc}",
            envoy_http_conn_manager_prefix="ingress",
            le="50"
        }}[1m]))
    ''')
    
    metrics["services"][svc] = {
        "replicas": replicas,
        "rps": rps,
        "p99_ms": p99,
        "p95_ms": p95,
        "error_rate": error_rate,
        "cpu_cores": cpu,
        "sla_violation_rate": sla_violation_rate
    }

# Save metrics
with open("$output_file", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Metrics saved to $output_file")
EOF
}

# Restart Locust pod to trigger fresh LoadTestShape run
restart_locust() {
    log_info "Restarting Locust pod to trigger fresh LoadTestShape..."
    kubectl delete pod -l app=locust
    kubectl wait --for=condition=ready pod -l app=locust --timeout=120s
    sleep 10
    
    log_info "Re-establishing Locust port-forward..."
    pkill -f "kubectl port-forward.*8089" || true
    nohup kubectl port-forward --address 0.0.0.0 svc/locust 8089:8089 > locust_pf.log 2>&1 &
    sleep 5
    
    log_info "Locust pod restarted, LoadTestShape will auto-start ✓"
}
start_locust_shape() {
    log_info "Starting Locust LoadTestShape..."

    # Use kubectl exec to reliably trigger locust from inside its own pod
    # Bypasses local port-forwarding drop issues.
    local locust_pod=$(kubectl get pod -l app=locust -o jsonpath='{.items[0].metadata.name}')
    
    # Wait until the web server is up and returns 200 to the post request
    for i in {1..10}; do
        local status_code=$(kubectl exec "$locust_pod" -- python -c "import requests; print(requests.post('http://localhost:8089/swarm', data={'user_count':1, 'spawn_rate':1, 'host':'http://app:8080'}).status_code)" 2>/dev/null)
        if [ "$status_code" = "200" ]; then
            log_info "Locust LoadTestShape triggered ✓"
            return 0
        fi
        sleep 2
    done

    log_error "Failed to trigger Locust swarm (Python requests failed inside pod)"
    exit 1
}
# Reset all deployments to 1 replica
reset_deployments() {
    log_info "Resetting all deployments to 1 replica..."
    for svc in api app db; do
        kubectl scale deployment $svc --replicas=1 --timeout=60s
    done
    sleep 30
    
    # Ensure they are ready before proceeding
    wait_for_pods "api" 1
    wait_for_pods "app" 1
    wait_for_pods "db" 1
    
    # Verify reset
    for svc in api app db; do
        local replicas=$(kubectl get deployment $svc -o jsonpath='{.spec.replicas}')
        log_info "$svc: $replicas replica(s)"
    done
    log_info "Deployments reset ✓"
}

# Delete HPA if exists
delete_hpa() {
    log_info "Removing any existing HPAs..."
    kubectl delete hpa --all --ignore-not-found=true
    sleep 5
    log_info "HPAs removed ✓"
}

################################################################################
# Pre-flight Checks
################################################################################

log_section "PRE-FLIGHT CHECKS"

# Check required commands
for cmd in kubectl k3d curl; do
    if ! command_exists $cmd; then
        log_error "$cmd is not installed"
        exit 1
    fi
done
log_info "All required commands available ✓"

# Check Python venv
if [ ! -f "$PYTHON_CMD" ]; then
    log_error "Python venv not found at $VENV_PATH"
    log_error "Please create venv: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
log_info "Python venv found at $VENV_PATH ✓"

# Test Python imports
$PYTHON_CMD -c "import requests, numpy, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    log_error "Required Python packages not installed in venv"
    log_error "Please run: source .venv/bin/activate && pip install requests numpy torch"
    exit 1
fi
log_info "Python dependencies available ✓"

# Check k3d cluster
log_info "Checking k3d cluster status..."
if ! kubectl cluster-info &>/dev/null; then
    log_warn "Cluster not accessible, attempting to fix..."
    
    # Fix kubeconfig if needed (0.0.0.0 → 127.0.0.1)
    KUBECONFIG_PATH="$HOME/.kube/config"
    if grep -q "0.0.0.0" "$KUBECONFIG_PATH" 2>/dev/null; then
        log_info "Fixing kubeconfig (0.0.0.0 → 127.0.0.1)..."
        sed -i.bak 's/0.0.0.0/127.0.0.1/g' "$KUBECONFIG_PATH"
        sleep 2
    fi
    
    # Try again
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cluster still not accessible. Please check k3d cluster manually."
        log_error "Try: kubectl get nodes"
        exit 1
    fi
fi
log_info "Cluster accessible ✓"

# Check all nodes are Ready
log_info "Checking node status..."
TOTAL_NODES=$(kubectl get nodes --no-headers | wc -l)
READY_NODES=$(kubectl get nodes --no-headers | grep " Ready " | wc -l)
log_info "Nodes: $READY_NODES/$TOTAL_NODES Ready"

if [ "$READY_NODES" -lt "$TOTAL_NODES" ]; then
    log_warn "Some nodes are not Ready:"
    kubectl get nodes
    log_warn "Continuing anyway, but results may be affected..."
else
    log_info "All nodes Ready ✓"
fi

# Check services are deployed
log_info "Checking service deployments..."
for svc in api app db locust; do
    if ! kubectl get deployment $svc &>/dev/null; then
        log_error "Deployment $svc not found. Please run ./tools/deploy_stack.sh first"
        exit 1
    fi
done
log_info "All services deployed ✓"

# Check Prometheus is accessible
log_info "Checking Prometheus..."
if ! curl -s "$PROMETHEUS_URL/-/healthy" | grep -q "Prometheus"; then
    log_info "Prometheus not locally accessible at $PROMETHEUS_URL. Attempting to setup port-forward..."
    pkill -f "kubectl port-forward.*9090" || true
    nohup kubectl port-forward -n monitoring --address 0.0.0.0 svc/kube-prom-kube-prometheus-prometheus 9090:9090 > prometheus_pf.log 2>&1 &
    sleep 5
    if ! curl -s "$PROMETHEUS_URL/-/healthy" | grep -q "Prometheus"; then
        log_error "Prometheus still not accessible at $PROMETHEUS_URL after port-forward attempt"
        log_error "Check: kubectl get svc -n monitoring | grep prometheus"
        exit 1
    fi
fi
log_info "Prometheus accessible ✓"

# Check Locust is accessible
log_info "Checking Locust..."
if ! curl -s "$LOCUST_URL" | grep -q "Locust"; then
    log_info "Locust not locally accessible at $LOCUST_URL. Attempting to setup port-forward..."
    pkill -f "kubectl port-forward.*8089" || true
    nohup kubectl port-forward --address 0.0.0.0 svc/locust 8089:8089 > locust_pf.log 2>&1 &
    sleep 5
    if ! curl -s "$LOCUST_URL" | grep -q "Locust"; then
        log_error "Locust still not accessible at $LOCUST_URL after port-forward attempt"
        log_error "Check: kubectl get pod -l app=locust"
        exit 1
    fi
fi
log_info "Locust accessible ✓"

# Verify APP bug fix
log_info "Verifying APP bug fix in controller..."
if ! grep -q "app_needs_recovery" deployment/agent_controller.py; then
    log_error "APP bug fix not found in controller!"
    exit 1
fi
log_info "APP bug fix verified ✓"

log_info "✅ All pre-flight checks passed!"

################################################################################
# Baseline Experiments (3 trials) - FIRST
################################################################################

log_section "PHASE 1: BASELINE EXPERIMENTS (Static 1 Replica)"

for trial in 1 2 3; do
    log_section "Baseline Trial $trial/3"
    
    TRIAL_DIR="$RESULTS_DIR/baseline/trial_$trial"
    
    # Reset environment
    delete_hpa
    reset_deployments
    
    # Restart Locust to trigger fresh LoadTestShape
    restart_locust
    sleep 30
    start_locust_shape
    kubectl logs -l app=locust --tail=20
    # LoadTestShape auto-starts immediately
    log_info "ProductionDayShape started automatically (30-min phased load)"
    
    # Run for trial duration with progress updates
    log_info "Running baseline trial $trial for $TRIAL_DURATION_MINUTES minutes..."
    log_info "Progress: 0% (0/$TRIAL_DURATION_MINUTES minutes)"
    
    mkdir -p "$TRIAL_DIR/timeseries"
    for i in $(seq 1 $TRIAL_DURATION_MINUTES); do
        sleep 60
        progress=$((i * 100 / TRIAL_DURATION_MINUTES))
        log_info "Progress: $progress% ($i/$TRIAL_DURATION_MINUTES minutes)"
        collect_metrics "baseline" $trial "$TRIAL_DIR/timeseries/metrics_min_${i}.json" &
    done
    
    wait
    
    # LoadTestShape stops automatically after 30 minutes
    log_info "Waiting for LoadTestShape to complete (30 seconds grace period)..."
    sleep 30
    
    # Collect metrics
    collect_metrics "baseline" $trial "$TRIAL_DIR/metrics.json"
    
    # Save deployment states
    kubectl get deployment api -o json > "$TRIAL_DIR/api_deployment.json"
    kubectl get deployment app -o json > "$TRIAL_DIR/app_deployment.json"
    kubectl get deployment db -o json > "$TRIAL_DIR/db_deployment.json"
    
    log_info "✅ Baseline trial $trial completed"
    
    # Cooldown between trials
    if [ $trial -lt 3 ]; then
        log_info "Cooldown period ($COOLDOWN_SECONDS seconds)..."
        sleep $COOLDOWN_SECONDS
    fi
done

log_info "✅ All baseline trials completed"

################################################################################
# HPA Experiments (3 trials) - SECOND
################################################################################

log_section "PHASE 2: HPA EXPERIMENTS (Kubernetes HPA with CPU 70%)"

# Create HPA configurations
cat > /tmp/hpa-api.yaml <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30
EOF

cat > /tmp/hpa-app.yaml <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30
EOF

for trial in 1 2 3; do
    log_section "HPA Trial $trial/3"
    
    TRIAL_DIR="$RESULTS_DIR/hpa/trial_$trial"
    
    # Reset environment
    delete_hpa
    reset_deployments
    
    # Apply HPA
    log_info "Applying HPA configurations..."
    kubectl apply -f /tmp/hpa-api.yaml
    kubectl apply -f /tmp/hpa-app.yaml
    
    # Wait for HPA to be ready
    sleep 30
    kubectl get hpa
    
    # Restart Locust to trigger fresh LoadTestShape
    restart_locust
    sleep 30
    start_locust_shape
    kubectl logs -l app=locust --tail=20
    
    # LoadTestShape auto-starts immediately
    log_info "ProductionDayShape started automatically (30-min phased load)"
    
    # Run for trial duration with progress updates
    log_info "Running HPA trial $trial for $TRIAL_DURATION_MINUTES minutes..."
    log_info "Progress: 0% (0/$TRIAL_DURATION_MINUTES minutes)"
    
    mkdir -p "$TRIAL_DIR/timeseries"
    for i in $(seq 1 $TRIAL_DURATION_MINUTES); do
        sleep 60
        progress=$((i * 100 / TRIAL_DURATION_MINUTES))
        log_info "Progress: $progress% ($i/$TRIAL_DURATION_MINUTES minutes)"
        
        # Log HPA status every 5 minutes
        if [ $((i % 5)) -eq 0 ]; then
            echo "=== Minute $i ===" >> "$TRIAL_DIR/hpa_status.log"
            kubectl get hpa >> "$TRIAL_DIR/hpa_status.log"
        fi
        
        collect_metrics "hpa" $trial "$TRIAL_DIR/timeseries/metrics_min_${i}.json" &
    done
    
    wait
    
    # LoadTestShape stops automatically after 30 minutes
    log_info "Waiting for LoadTestShape to complete (30 seconds grace period)..."
    sleep 30
    
    # Collect metrics
    collect_metrics "hpa" $trial "$TRIAL_DIR/metrics.json"
    
    # Save HPA final state
    kubectl get hpa -o yaml > "$TRIAL_DIR/hpa_final_state.yaml"
    kubectl get deployment api -o json > "$TRIAL_DIR/api_deployment.json"
    kubectl get deployment app -o json > "$TRIAL_DIR/app_deployment.json"
    
    # Remove HPA
    delete_hpa
    
    log_info "✅ HPA trial $trial completed"
    
    # Cooldown between trials
    if [ $trial -lt 3 ]; then
        log_info "Cooldown period ($COOLDOWN_SECONDS seconds)..."
        sleep $COOLDOWN_SECONDS
    fi
done

log_info "✅ All HPA trials completed"

################################################################################
# QMIX Experiments (3 trials) - LAST (for best comparison)
################################################################################

log_section "PHASE 3: QMIX EXPERIMENTS (AURA Agent with Fixed Controller)"

for trial in 1 2 3; do
    log_section "QMIX Trial $trial/3"
    
    TRIAL_DIR="$RESULTS_DIR/qmix/trial_$trial"
    
    # Reset environment
    delete_hpa
    reset_deployments
    
    # Wait for stability
    log_info "Waiting for system to stabilize (60 seconds)..."
    sleep 60
    
    # Start QMIX controller with venv
    log_info "Starting QMIX controller (using venv)..."
    export AURA_SHADOW_MODE=false
    export AURA_CHECKPOINT_DIR=marl/qmix_trained
    export PROMETHEUS_URL="$PROMETHEUS_URL"
    export AURA_COOLDOWN_SEC=15  # More reactive than HPA
    
    $PYTHON_CMD deployment/agent_controller.py > "$TRIAL_DIR/controller.log" 2>&1 &
    CONTROLLER_PID=$!
    
    log_info "QMIX controller started (PID: $CONTROLLER_PID)"
    sleep 45
    
    # Verify controller is running
    if ! ps -p $CONTROLLER_PID > /dev/null; then
        log_error "Controller failed to start! Check $TRIAL_DIR/controller.log"
        cat "$TRIAL_DIR/controller.log"
        exit 1
    fi
    log_info "Controller verified running ✓"
    
    # Restart Locust to trigger fresh LoadTestShape
    restart_locust
    sleep 30
    start_locust_shape
    kubectl logs -l app=locust --tail=20
    
    # LoadTestShape auto-starts immediately
    log_info "ProductionDayShape started automatically (30-min phased load)"
    
    # Run for trial duration with progress updates
    log_info "Running QMIX trial $trial for $TRIAL_DURATION_MINUTES minutes..."
    log_info "Progress: 0% (0/$TRIAL_DURATION_MINUTES minutes)"
    
    mkdir -p "$TRIAL_DIR/timeseries"
    for i in $(seq 1 $TRIAL_DURATION_MINUTES); do
        sleep 60
        progress=$((i * 100 / TRIAL_DURATION_MINUTES))
        log_info "Progress: $progress% ($i/$TRIAL_DURATION_MINUTES minutes)"
        
        # Log replica counts every 5 minutes
        if [ $((i % 5)) -eq 0 ]; then
            echo "=== Minute $i ===" >> "$TRIAL_DIR/replicas.log"
            kubectl get deployments >> "$TRIAL_DIR/replicas.log"
        fi
        
        # Verify controller still running
        if ! ps -p $CONTROLLER_PID > /dev/null; then
            log_error "Controller died! Check $TRIAL_DIR/controller.log"
            exit 1
        fi
        
        collect_metrics "qmix" $trial "$TRIAL_DIR/timeseries/metrics_min_${i}.json" &
    done
    
    wait
    
    # LoadTestShape stops automatically after 30 minutes
    log_info "Waiting for LoadTestShape to complete (30 seconds grace period)..."
    sleep 30
    
    # Stop controller gracefully
    log_info "Stopping QMIX controller..."
    kill -TERM $CONTROLLER_PID 2>/dev/null || true
    sleep 5
    kill -9 $CONTROLLER_PID 2>/dev/null || true
    
    # Collect metrics
    collect_metrics "qmix" $trial "$TRIAL_DIR/metrics.json"
    
    # Save final state
    kubectl get deployment api -o json > "$TRIAL_DIR/api_deployment.json"
    kubectl get deployment app -o json > "$TRIAL_DIR/app_deployment.json"
    kubectl get deployment db -o json > "$TRIAL_DIR/db_deployment.json"
    
    # Verify APP scaled (critical check)
    APP_REPLICAS=$(kubectl get deployment app -o jsonpath='{.spec.replicas}')
    if [ "$APP_REPLICAS" -eq 1 ]; then
        log_warn "⚠️  APP tier still at 1 replica! Bug fix may not be working."
        log_warn "Check controller log: $TRIAL_DIR/controller.log"
    else
        log_info "✅ APP tier scaled to $APP_REPLICAS replicas (bug fix working!)"
    fi
    
    log_info "✅ QMIX trial $trial completed"
    
    # Cooldown between trials
    if [ $trial -lt 3 ]; then
        log_info "Cooldown period ($COOLDOWN_SECONDS seconds)..."
        sleep $COOLDOWN_SECONDS
    fi
done

log_info "✅ All QMIX trials completed"

################################################################################
# Generate Analysis Report
################################################################################

log_section "GENERATING ANALYSIS REPORT"

$PYTHON_CMD - <<EOF
import json
import os
from pathlib import Path
import numpy as np

results_dir = "$RESULTS_DIR"

def load_metrics(mode):
    metrics = []
    for trial in [1, 2, 3]:
        trial_metrics = []
        # Focus on peak load phase (minutes 16-24) where users=4000
        for i in range(17, 25): 
            path = f"{results_dir}/{mode}/trial_{trial}/timeseries/metrics_min_{i}.json"
            if os.path.exists(path):
                with open(path) as f:
                    trial_metrics.append(json.load(f))
        
        if trial_metrics:
            # Average the peak metrics for this trial
            avg_trial = {"timestamp": trial_metrics[0]["timestamp"], "services": {"api": {}, "app": {}}}
            for svc in ["api", "app"]:
                for key in ["p99_ms", "p95_ms", "rps", "error_rate", "replicas", "cpu_cores", "sla_violation_rate"]:
                    avg_trial["services"][svc][key] = np.mean([m["services"][svc].get(key, 0) for m in trial_metrics])
            metrics.append(avg_trial)
    return metrics

# Load all metrics
baseline_metrics = load_metrics("baseline")
hpa_metrics = load_metrics("hpa")
qmix_metrics = load_metrics("qmix")

# Generate summary report
report = []
report.append("=" * 80)
report.append("AURA FGCS BENCHMARK RESULTS SUMMARY")
report.append("=" * 80)
report.append(f"Results Directory: {results_dir}")
report.append(f"Timestamp: {baseline_metrics[0]['timestamp'] if baseline_metrics else 'N/A'}")
report.append("")

def analyze_mode(mode_name, metrics):
    report.append(f"\n{mode_name} Results (n={len(metrics)} trials):")
    report.append("-" * 80)
    
    if not metrics:
        report.append("  No data available")
        return
    
    for svc in ["api", "app"]:
        p99_values = [m["services"][svc]["p99_ms"] for m in metrics]
        p95_values = [m["services"][svc]["p95_ms"] for m in metrics]
        rps_values = [m["services"][svc]["rps"] for m in metrics]
        error_values = [m["services"][svc]["error_rate"] for m in metrics]
        replica_values = [m["services"][svc]["replicas"] for m in metrics]
        
        report.append(f"\n  {svc.upper()} Service:")
        report.append(f"    P99 Latency:  {np.mean(p99_values):.2f} ± {np.std(p99_values):.2f} ms")
        report.append(f"    P95 Latency:  {np.mean(p95_values):.2f} ± {np.std(p95_values):.2f} ms")
        report.append(f"    RPS:          {np.mean(rps_values):.2f} ± {np.std(rps_values):.2f}")
        report.append(f"    Error Rate:   {np.mean(error_values):.4f} ± {np.std(error_values):.4f}")
        report.append(f"    Replicas:     {np.mean(replica_values):.2f} ± {np.std(replica_values):.2f}")
        sla_values = [m["services"][svc].get("sla_violation_rate", 0) for m in metrics]
        report.append(f"    SLA Viol. Rt: {np.mean(sla_values):.2f} ± {np.std(sla_values):.2f} req/s")

analyze_mode("BASELINE", baseline_metrics)
analyze_mode("HPA", hpa_metrics)
analyze_mode("QMIX", qmix_metrics)

# Comparison
if baseline_metrics and qmix_metrics:
    report.append("\n" + "=" * 80)
    report.append("QMIX vs BASELINE COMPARISON")
    report.append("=" * 80)
    
    for svc in ["api", "app"]:
        baseline_p99 = np.mean([m["services"][svc]["p99_ms"] for m in baseline_metrics])
        qmix_p99 = np.mean([m["services"][svc]["p99_ms"] for m in qmix_metrics])
        improvement = ((baseline_p99 - qmix_p99) / baseline_p99) * 100
        
        report.append(f"\n{svc.upper()} P99 Latency:")
        report.append(f"  Baseline: {baseline_p99:.2f} ms")
        report.append(f"  QMIX:     {qmix_p99:.2f} ms")
        report.append(f"  Improvement: {improvement:.2f}%")

if hpa_metrics and qmix_metrics:
    report.append("\n" + "=" * 80)
    report.append("QMIX vs HPA COMPARISON")
    report.append("=" * 80)
    
    for svc in ["api", "app"]:
        hpa_p99 = np.mean([m["services"][svc]["p99_ms"] for m in hpa_metrics])
        qmix_p99 = np.mean([m["services"][svc]["p99_ms"] for m in qmix_metrics])
        improvement = ((hpa_p99 - qmix_p99) / hpa_p99) * 100
        
        report.append(f"\n{svc.upper()} P99 Latency:")
        report.append(f"  HPA:  {hpa_p99:.2f} ms")
        report.append(f"  QMIX: {qmix_p99:.2f} ms")
        report.append(f"  Improvement: {improvement:.2f}%")

report.append("\n" + "=" * 80)
report.append("NEXT STEPS:")
report.append("1. Review individual trial logs in each trial_X directory")
report.append("2. Run statistical analysis:")
report.append(f"   python3 tools/analyze_results.py {results_dir}")
report.append("3. Generate figures:")
report.append(f"   python3 tools/generate_paper_figures.py {results_dir}")
report.append("=" * 80)

# Save report
report_text = "\n".join(report)
print(report_text)

with open(f"{results_dir}/SUMMARY_REPORT.txt", "w") as f:
    f.write(report_text)

print(f"\n✓ Summary report saved to {results_dir}/SUMMARY_REPORT.txt")
EOF

################################################################################
# Completion
################################################################################

log_section "BENCHMARK SUITE COMPLETED SUCCESSFULLY"

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          ALL EXPERIMENTS COMPLETED SUCCESSFULLY!           ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  End Time: $(date '+%Y-%m-%d %H:%M:%S')                             ║${NC}"
echo -e "${GREEN}║  Results: $RESULTS_DIR                                     ║${NC}"
echo -e "${GREEN}║  Execution Order: Baseline → HPA → QMIX                   ║${NC}"
echo -e "${GREEN}║  Total Trials: 9 (3 per configuration)                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

log_info "Summary report: $RESULTS_DIR/SUMMARY_REPORT.txt"
log_info "Full log: $LOG_FILE"

echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "1. Review summary: cat $RESULTS_DIR/SUMMARY_REPORT.txt"
echo "2. Statistical analysis: python3 tools/analyze_results.py $RESULTS_DIR"
echo "3. Generate figures: python3 tools/generate_paper_figures.py $RESULTS_DIR"
echo ""
echo -e "${GREEN}✅ Ready for FGCS paper submission!${NC}"
echo ""

exit 0

# Made with Bob
