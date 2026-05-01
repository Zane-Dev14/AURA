#!/bin/bash
# ============================================================
# AURA Baseline Test Automation Script (30-minute test)
#
# This script:
# 1. Rebuilds the Locust Docker image (picks up locustfile changes)
# 2. Redeploys Locust + ensures service labels are correct
# 3. Port-forwards Prometheus (9090) and Locust (8089)
# 4. Starts the 30-minute baseline load test via Locust web API
# 5. Collects metrics at T+5min, T+15min, T+25min, and final
# 6. Monitors test progress in real-time
# 7. Cleanup
#
# Usage:
#   bash tools/run_baseline_test.sh
#   bash tools/run_baseline_test.sh --skip-build   # skip image rebuild
#   bash tools/run_baseline_test.sh --duration 30 --output-dir docs/Final Results
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/k3d_guard.sh"

assert_k3d_context

KUBE_CONTEXT="$(kubectl config current-context)"
K3D_CLUSTER="${KUBE_CONTEXT#k3d-}"

VENV_PYTHON="$WORKSPACE_DIR/.venv/bin/python"
NAMESPACE="default"
TEST_DURATION_MIN=30
TEST_DURATION_SEC=$((TEST_DURATION_MIN * 60))
OUTPUT_DIR="$WORKSPACE_DIR/docs/Final Results"
SKIP_BUILD=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --duration)
            TEST_DURATION_MIN="$2"
            TEST_DURATION_SEC=$((TEST_DURATION_MIN * 60))
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if ! [[ "$TEST_DURATION_MIN" =~ ^[0-9]+$ ]] || (( TEST_DURATION_MIN < 2 )); then
    echo "[ERROR] --duration must be an integer >= 2"
    exit 1
fi

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$WORKSPACE_DIR/$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# PIDs to track for cleanup
PROM_PF_PID=""
LOCUST_PF_PID=""
BG_PIDS=()

# Prefer stable NodePorts (k3d port mappings) instead of brittle port-forwards.
PROM_URL="${PROM_URL:-http://127.0.0.1:30090}"
PROM_NAMESPACE="monitoring"
PROM_STATEFULSET="prometheus-kube-prom-kube-prometheus-prometheus"
LOCUST_URL="${LOCUST_URL:-http://127.0.0.1:30089}"
USING_PROM_PORT_FORWARD=false
USING_LOCUST_PORT_FORWARD=false

cleanup() {
    echo ""
    log_info "Cleaning up..."
    # Stop Locust test
    curl -s -X POST "${LOCUST_URL}/stop" &>/dev/null || true
    # Kill port-forwards
    [[ -n "$PROM_PF_PID" ]]   && kill "$PROM_PF_PID"   2>/dev/null || true
    [[ -n "$LOCUST_PF_PID" ]] && kill "$LOCUST_PF_PID" 2>/dev/null || true
    # Kill background metric collectors
    for pid in "${BG_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # General cleanup of stale port-forwards
    pkill -f "kubectl port-forward.*9090" 2>/dev/null || true
    pkill -f "kubectl port-forward.*8089" 2>/dev/null || true
    log_ok "Cleanup done"
}
trap cleanup EXIT

# ────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────

wait_for_url() {
    local url=$1 label=$2 retries=${3:-15}
    for i in $(seq 1 "$retries"); do
        if curl -sf "$url" &>/dev/null; then
            return 0
        fi
        sleep 2
    done
    log_err "$label not reachable at $url after $((retries*2))s"
    return 1
}

start_prom_portforward() {
    # Kill any stale process
    [[ -n "$PROM_PF_PID" ]] && kill "$PROM_PF_PID" 2>/dev/null || true
    pkill -f "kubectl port-forward.*9090" 2>/dev/null || true
    sleep 1
    kubectl port-forward -n monitoring svc/kube-prom-kube-prometheus-prometheus 9090:9090 \
        &>/tmp/prom-portforward.log &
    PROM_PF_PID=$!
    sleep 3
    if ! wait_for_url "http://127.0.0.1:9090/api/v1/query?query=up" "Prometheus" 10; then
        log_err "Cannot start Prometheus port-forward"
        exit 1
    fi
    log_ok "Prometheus port-forward active (PID: $PROM_PF_PID)"
    PROM_URL="http://127.0.0.1:9090"
    USING_PROM_PORT_FORWARD=true
}

start_locust_portforward() {
    [[ -n "$LOCUST_PF_PID" ]] && kill "$LOCUST_PF_PID" 2>/dev/null || true
    pkill -f "kubectl port-forward.*8089" 2>/dev/null || true
    sleep 1
    kubectl port-forward deployment/locust 8089:8089 \
        &>/tmp/locust-portforward.log &
    LOCUST_PF_PID=$!
    sleep 3
    if ! wait_for_url "http://127.0.0.1:8089/stats/requests" "Locust" 10; then
        log_err "Cannot start Locust port-forward"
        exit 1
    fi
    log_ok "Locust port-forward active (PID: $LOCUST_PF_PID)"
    LOCUST_URL="http://127.0.0.1:8089"
    USING_LOCUST_PORT_FORWARD=true
}

ensure_prom_endpoint() {
    if curl -sf "${PROM_URL}/api/v1/query?query=up" &>/dev/null; then
        return 0
    fi
    log_warn "Prometheus not reachable at ${PROM_URL}; attempting local restart..."
    kubectl -n "$PROM_NAMESPACE" rollout restart "statefulset/${PROM_STATEFULSET}" >/dev/null 2>&1 || true
    if kubectl -n "$PROM_NAMESPACE" rollout status "statefulset/${PROM_STATEFULSET}" --timeout=180s >/dev/null 2>&1; then
        for _ in $(seq 1 30); do
            if curl -sf "${PROM_URL}/api/v1/query?query=up" &>/dev/null; then
                log_ok "Prometheus endpoint recovered"
                return 0
            fi
            sleep 2
        done
    fi
    log_warn "Prometheus restart did not recover ${PROM_URL}; falling back to port-forward..."
    start_prom_portforward
}

ensure_locust_endpoint() {
    if curl -sf "${LOCUST_URL}/stats/requests" &>/dev/null; then
        return 0
    fi
    log_warn "Locust not reachable at ${LOCUST_URL}; falling back to port-forward..."
    start_locust_portforward
}

ensure_prom_alive() {
    if curl -sf "${PROM_URL}/api/v1/query?query=up" &>/dev/null; then
        return 0
    fi
    if [[ "$USING_PROM_PORT_FORWARD" == "true" ]]; then
        log_warn "Prometheus port-forward died — restarting..."
        start_prom_portforward
        return 0
    fi
    log_warn "Prometheus became unreachable at ${PROM_URL}; attempting local restart..."
    kubectl -n "$PROM_NAMESPACE" rollout restart "statefulset/${PROM_STATEFULSET}" >/dev/null 2>&1 || true
    if kubectl -n "$PROM_NAMESPACE" rollout status "statefulset/${PROM_STATEFULSET}" --timeout=180s >/dev/null 2>&1; then
        for _ in $(seq 1 30); do
            if curl -sf "${PROM_URL}/api/v1/query?query=up" &>/dev/null; then
                log_ok "Prometheus endpoint recovered"
                return 0
            fi
            sleep 2
        done
    fi
    log_err "Prometheus became unreachable at ${PROM_URL}"
    return 1
}

ensure_locust_alive() {
    if curl -sf "${LOCUST_URL}/stats/requests" &>/dev/null; then
        return 0
    fi
    if [[ "$USING_LOCUST_PORT_FORWARD" == "true" ]]; then
        log_warn "Locust port-forward died — restarting..."
        start_locust_portforward
        return 0
    fi
    log_err "Locust became unreachable at ${LOCUST_URL}"
    return 1
}

get_locust_stats() {
    # Returns: total_requests current_rps total_failures
    local stats
    stats=$(curl -sf "${LOCUST_URL}/stats/requests" 2>/dev/null || echo "")
    if [[ -z "$stats" ]]; then
        echo "0 0 0"
        return
    fi
    echo "$stats" | "$VENV_PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    stats = data.get('stats', [])
    total_rq = 0; current_rps = 0; total_fail = 0
    for s in stats:
        if s.get('name') == 'Aggregated':
            total_rq = s.get('num_requests', 0)
            current_rps = s.get('current_rps', 0)
            total_fail = s.get('num_failures', 0)
            break
    if total_rq == 0:
        total_rq = sum(s.get('num_requests', 0) for s in stats)
        current_rps = sum(s.get('current_rps', 0) for s in stats)
        total_fail = sum(s.get('num_failures', 0) for s in stats)
    print(f'{int(total_rq)} {current_rps:.0f} {int(total_fail)}')
except Exception:
    print('0 0 0')
" 2>/dev/null || echo "0 0 0"
}

reset_replicas_to_single() {
    log_info "Setting initial replicas to 1 for api/app/db (fair start state)..."
    kubectl scale deployment api -n "$NAMESPACE" --replicas=1
    kubectl scale deployment app -n "$NAMESPACE" --replicas=1
    kubectl scale deployment db -n "$NAMESPACE" --replicas=1

    kubectl rollout status deployment/api -n "$NAMESPACE" --timeout=120s
    kubectl rollout status deployment/app -n "$NAMESPACE" --timeout=120s
    kubectl rollout status deployment/db -n "$NAMESPACE" --timeout=120s
    log_ok "All services at 1 replica"
}

collect_metrics() {
    local phase_name=$1
    local elapsed_min=$2

    ensure_prom_alive

    log_info "━━━ Collecting metrics: $phase_name (T+${elapsed_min}min) ━━━"
    cd "$WORKSPACE_DIR"
    "$VENV_PYTHON" tools/gke_cost_report.py \
        --mode baseline \
        --duration "$TEST_DURATION_MIN" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "/tmp/metrics_${phase_name}.log"
    log_ok "Metrics collected: $phase_name"
}

# ────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  AURA Baseline Test — ${TEST_DURATION_MIN}-minute automated run${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Step 1: Verify cluster ──
log_info "Verifying cluster connectivity..."
if ! kubectl cluster-info &>/dev/null; then
    log_err "Cannot connect to Kubernetes cluster"
    exit 1
fi
log_ok "Cluster connected"

# ── Step 2: Rebuild & redeploy Locust image ──
if [[ "$SKIP_BUILD" == "false" ]]; then
    log_info "Rebuilding Locust Docker image..."
    cd "$WORKSPACE_DIR/microservices/locust"
    docker build -t project-locust:local . 2>&1 | tail -3
    log_ok "Image rebuilt"
    kubectl apply -f $WORKSPACE_DIR/microservices/locust/locust.yaml
    log_info "Importing image into k3d cluster..."
    k3d image import project-locust:local -c "$K3D_CLUSTER" 2>&1 | tail -2
    log_ok "Image imported"

    log_info "Restarting Locust deployment..."
    kubectl rollout restart deployment/locust -n "$NAMESPACE"
    kubectl rollout status deployment/locust -n "$NAMESPACE" --timeout=90s
    log_ok "Locust redeployed with updated locustfile"
    cd "$WORKSPACE_DIR"
else
    log_warn "Skipping image rebuild (--skip-build)"
fi

# ── Step 3: Ensure service labels match ServiceMonitor selector ──
log_info "Ensuring service labels for Prometheus scraping..."
for svc in api app db; do
    kubectl label service "$svc" -n "$NAMESPACE" metrics=envoy --overwrite 2>/dev/null || true
done
log_ok "Service labels verified"

# ── Step 4: Set fair initial replicas (1/1/1) ──
reset_replicas_to_single

# ── Step 5: Ensure endpoints (prefer NodePorts) ──
log_info "Checking Prometheus/Locust endpoints (prefer NodePorts)..."
ensure_prom_endpoint
ensure_locust_endpoint

# ── Step 6: Start load test ──
TEST_START=$(date +%s)
TEST_START_READABLE=$(date)

log_info "Starting ${TEST_DURATION_MIN}-minute baseline load test..."
echo "  Host: http://app:8080"
echo "  Shape: ProductionDayShape"
echo "  Start: $TEST_START_READABLE"

# LoadTestShape controls user count; swarm params are overridden by the shape.
SWARM_RESP=$(curl -sf -X POST "${LOCUST_URL}/swarm" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "user_count=1&spawn_rate=1&host=http://app:8080" 2>&1 || echo "FAIL")

if [[ "$SWARM_RESP" == "FAIL" ]]; then
    log_err "Failed to start Locust swarm"
    exit 1
fi
log_ok "Test started"

# ── Step 7: Schedule background metrics collections ──
PHASE1_MIN=$(( TEST_DURATION_MIN / 6 ))
PHASE2_MIN=$(( TEST_DURATION_MIN / 2 ))
PHASE3_MIN=$(( (TEST_DURATION_MIN * 5) / 6 ))

(( PHASE1_MIN < 1 )) && PHASE1_MIN=1
(( PHASE2_MIN < PHASE1_MIN + 1 )) && PHASE2_MIN=$((PHASE1_MIN + 1))
(( PHASE3_MIN < PHASE2_MIN + 1 )) && PHASE3_MIN=$((PHASE2_MIN + 1))

(( PHASE1_MIN >= TEST_DURATION_MIN )) && PHASE1_MIN=0
(( PHASE2_MIN >= TEST_DURATION_MIN )) && PHASE2_MIN=0
(( PHASE3_MIN >= TEST_DURATION_MIN )) && PHASE3_MIN=0

if (( PHASE1_MIN > 0 )); then
    (
        sleep $((PHASE1_MIN * 60))
        collect_metrics "phase1" "$PHASE1_MIN"
    ) &
    BG_PIDS+=($!)
fi

if (( PHASE2_MIN > 0 )); then
    (
        sleep $((PHASE2_MIN * 60))
        collect_metrics "phase2" "$PHASE2_MIN"
    ) &
    BG_PIDS+=($!)
fi

if (( PHASE3_MIN > 0 )); then
    (
        sleep $((PHASE3_MIN * 60))
        collect_metrics "phase3" "$PHASE3_MIN"
    ) &
    BG_PIDS+=($!)
fi

# ── Step 8: Monitor loop ──
log_info "Monitoring test progress (${TEST_DURATION_MIN} minutes)..."
echo "  Live dashboard: ${LOCUST_URL}/"
LAST_PF_CHECK=0

while true; do
    ELAPSED=$(( $(date +%s) - TEST_START ))
    ELAPSED_MIN=$(( ELAPSED / 60 ))

    # Health check every 45 seconds
    if (( ELAPSED - LAST_PF_CHECK > 45 )); then
        ensure_prom_alive
        ensure_locust_alive
        LAST_PF_CHECK=$ELAPSED
    fi

    read -r TOTAL_RQ CURRENT_RPS TOTAL_FAIL <<< "$(get_locust_stats)"

    printf "\r  ⏱ %2d / %2d min | Requests: %-8s | RPS: %-6s | Fails: %-6s" \
        "$ELAPSED_MIN" "$TEST_DURATION_MIN" "$TOTAL_RQ" "$CURRENT_RPS" "$TOTAL_FAIL"

    if (( ELAPSED > TEST_DURATION_SEC + 30 )); then
        echo ""
        log_ok "Test completed after ${ELAPSED_MIN} minutes"
        break
    fi

    sleep 15
done

# ── Step 9: Final metrics collection ──
log_info "Running final metrics collection..."
ensure_prom_alive
collect_metrics "final" "$TEST_DURATION_MIN"

# ── Step 10: Stop Locust ──
log_info "Stopping Locust test..."
curl -sf -X POST "${LOCUST_URL}/stop" &>/dev/null || true

# ── Step 11: Wait for background collectors ──
log_info "Waiting for background collectors to finish..."
for pid in "${BG_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# ── Step 12: Summary ──
TEST_END=$(date +%s)
TOTAL_MIN=$(( (TEST_END - TEST_START) / 60 ))

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  BASELINE TEST COMPLETE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Start:    $TEST_START_READABLE"
echo "  End:      $(date)"
echo "  Duration: ${TOTAL_MIN} minutes"
echo ""
echo -e "${YELLOW}Output files:${NC}"
ls -lh "$OUTPUT_DIR"/baseline_metrics_*.json 2>/dev/null | tail -5 || echo "  (none found)"
echo ""
echo -e "${YELLOW}CSV time-series:${NC}"
ls -lh "$OUTPUT_DIR"/*_over_time_baseline.csv 2>/dev/null | sed 's/^/  /' || echo "  (none found)"
echo ""
log_ok "All done!"
