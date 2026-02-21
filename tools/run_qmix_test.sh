#!/bin/bash
# ============================================================
# AURA QMIX Test Automation Script (30-minute test)
#
# Runs the same workload as baseline but with the QMIX MARL
# agent controller actively scaling replicas.
#
# This script:
# 1. Optionally rebuilds Locust image (same workload as baseline)
# 2. Sets initial replicas to QMIX minimums (api=2, app=3, db=1)
# 3. Ensures service labels for Prometheus scraping
# 4. Starts port-forwards (Prometheus + Locust)
# 5. Starts the QMIX agent_controller.py in LIVE mode
# 6. Starts the 30-minute load test via Locust web API
# 7. Collects metrics at T+5, T+15, T+25, and final
# 8. Stops agent + cleanup
#
# Usage:
#   bash tools/run_qmix_test.sh
#   bash tools/run_qmix_test.sh --skip-build   # skip image rebuild
# ============================================================
set -euo pipefail

WORKSPACE_DIR="/Users/eric/Documents/GitHub/AURA"
VENV_PYTHON="$WORKSPACE_DIR/.venv/bin/python"
NAMESPACE="default"
TEST_DURATION_SEC=1800   # 30 minutes
SKIP_BUILD=false

# QMIX replica minimums (must match agent_controller.py MIN_REPLICAS)
API_MIN_REPLICAS=1
APP_MIN_REPLICAS=1
DB_MIN_REPLICAS=1

# Parse args
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_qmix()  { echo -e "${CYAN}[QMIX]${NC} $*"; }

# PIDs to track for cleanup
PROM_PF_PID=""
LOCUST_PF_PID=""
AGENT_PID=""
BG_PIDS=()

cleanup() {
    echo ""
    log_info "Cleaning up..."
    # Stop QMIX agent
    if [[ -n "$AGENT_PID" ]]; then
        log_info "Stopping QMIX agent (PID: $AGENT_PID)..."
        kill "$AGENT_PID" 2>/dev/null || true
        wait "$AGENT_PID" 2>/dev/null || true
    fi
    # Stop Locust test
    curl -s -X POST http://127.0.0.1:8089/stop &>/dev/null || true
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
}

ensure_prom_alive() {
    if ! curl -sf "http://127.0.0.1:9090/api/v1/query?query=up" &>/dev/null; then
        log_warn "Prometheus port-forward died — restarting..."
        start_prom_portforward
    fi
}

ensure_locust_alive() {
    if ! curl -sf "http://127.0.0.1:8089/stats/requests" &>/dev/null; then
        log_warn "Locust port-forward died — restarting..."
        start_locust_portforward
    fi
}

get_locust_stats() {
    local stats
    stats=$(curl -sf http://127.0.0.1:8089/stats/requests 2>/dev/null || echo "")
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

get_replica_counts() {
    # Returns: api_replicas app_replicas db_replicas
    local api app db
    api=$(kubectl get deployment api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "?")
    app=$(kubectl get deployment app -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "?")
    db=$(kubectl get deployment db -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "?")
    echo "$api $app $db"
}

collect_metrics() {
    local phase_name=$1
    local elapsed_min=$2

    ensure_prom_alive

    log_info "━━━ Collecting metrics: $phase_name (T+${elapsed_min}min) ━━━"
    cd "$WORKSPACE_DIR"
    "$VENV_PYTHON" tools/gke_cost_report.py --mode qmix --duration 30 2>&1 | tee "/tmp/metrics_qmix_${phase_name}.log"
    log_ok "Metrics collected: $phase_name"
}

# ────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  AURA QMIX Test — 30-minute automated run${NC}"
echo -e "${CYAN}  (Same workload as baseline, QMIX agent active)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Step 1: Verify cluster ──
log_info "Verifying cluster connectivity..."
if ! kubectl cluster-info &>/dev/null; then
    log_err "Cannot connect to Kubernetes cluster"
    exit 1
fi
log_ok "Cluster connected"

# ── Step 2: Rebuild & redeploy Locust image (same workload) ──
if [[ "$SKIP_BUILD" == "false" ]]; then
    log_info "Rebuilding Locust Docker image (same workload as baseline)..."
    cd "$WORKSPACE_DIR/microservices/locust"
    docker build -t project-locust:local . 2>&1 | tail -3
    log_ok "Image rebuilt"

    log_info "Importing image into k3d cluster..."
    k3d image import project-locust:local -c aura 2>&1 | tail -2
    log_ok "Image imported"

    log_info "Restarting Locust deployment..."
    kubectl rollout restart deployment/locust -n "$NAMESPACE"
    kubectl rollout status deployment/locust -n "$NAMESPACE" --timeout=90s
    log_ok "Locust redeployed"
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

# ── Step 4: Set QMIX initial replicas ──
log_qmix "Letting QMIX agent determine initial replicas (no forced scaling)..."
log_info "Current replicas will be used as starting point"
# Don't force initial replicas - let the agent scale naturally from current state
log_ok "Ready to start with current replica state"

# ── Step 5: Start port-forwards ──
log_info "Starting port-forwards..."
pkill -f "kubectl port-forward.*9090" 2>/dev/null || true
pkill -f "kubectl port-forward.*8089" 2>/dev/null || true
sleep 2

start_prom_portforward
start_locust_portforward

# ── Step 6: Start QMIX agent controller ──
log_qmix "Starting QMIX agent controller (LIVE mode)..."
cd "$WORKSPACE_DIR"

AURA_SHADOW_MODE=false \
AURA_CHECKPOINT_DIR=marl/qmix_trained \
PROMETHEUS_URL=http://localhost:9090 \
"$VENV_PYTHON" deployment/agent_controller.py \
    > /tmp/qmix_agent.log 2>&1 &
AGENT_PID=$!
sleep 3

# Verify agent started
if kill -0 "$AGENT_PID" 2>/dev/null; then
    log_ok "QMIX agent running (PID: $AGENT_PID)"
    log_info "Agent log: /tmp/qmix_agent.log"
else
    log_err "QMIX agent failed to start. Check /tmp/qmix_agent.log"
    cat /tmp/qmix_agent.log
    exit 1
fi

# ── Step 7: Start 30-min test ──
TEST_START=$(date +%s)
TEST_START_READABLE=$(date)

log_info "Starting 30-minute QMIX load test..."
echo "  Host: http://app:8080"
echo "  Shape: ProductionDayShape (30 min phased)"
echo "  Agent: QMIX LIVE (scaling enabled)"
echo "  Start: $TEST_START_READABLE"

SWARM_RESP=$(curl -sf -X POST http://127.0.0.1:8089/swarm \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "user_count=1&spawn_rate=1&host=http://app:8080" 2>&1 || echo "FAIL")

if [[ "$SWARM_RESP" == "FAIL" ]]; then
    log_err "Failed to start Locust swarm"
    exit 1
fi
log_ok "Test started"

# ── Step 8: Schedule background metrics collections ──
(
    sleep 300    # T+5min — ramp/hold phase
    collect_metrics "05min_ramp" 5
) &
BG_PIDS+=($!)

(
    sleep 900    # T+15min — hold/spike transition
    collect_metrics "15min_spike" 15
) &
BG_PIDS+=($!)

(
    sleep 1500   # T+25min — peak/cooldown
    collect_metrics "25min_peak" 25
) &
BG_PIDS+=($!)

# ── Step 9: Monitor loop ──
log_info "Monitoring test progress (30 minutes)..."
echo "  Live dashboard: http://localhost:8089/"
echo "  Agent log:      tail -f /tmp/qmix_agent.log"
LAST_PF_CHECK=0

while true; do
    ELAPSED=$(( $(date +%s) - TEST_START ))
    ELAPSED_MIN=$(( ELAPSED / 60 ))

    # Health check every 45 seconds
    if (( ELAPSED - LAST_PF_CHECK > 45 )); then
        ensure_prom_alive
        ensure_locust_alive

        # Check if agent is still alive
        if ! kill -0 "$AGENT_PID" 2>/dev/null; then
            log_warn "QMIX agent died! Restarting..."
            cd "$WORKSPACE_DIR"
            AURA_SHADOW_MODE=false \
            AURA_CHECKPOINT_DIR=marl/qmix_trained \
            PROMETHEUS_URL=http://localhost:9090 \
            "$VENV_PYTHON" deployment/agent_controller.py \
                >> /tmp/qmix_agent.log 2>&1 &
            AGENT_PID=$!
            sleep 2
            if kill -0 "$AGENT_PID" 2>/dev/null; then
                log_ok "QMIX agent restarted (PID: $AGENT_PID)"
            else
                log_err "Agent cannot restart — continuing without scaling"
            fi
        fi

        LAST_PF_CHECK=$ELAPSED
    fi

    read -r TOTAL_RQ CURRENT_RPS TOTAL_FAIL <<< "$(get_locust_stats)"
    read -r R_API R_APP R_DB <<< "$(get_replica_counts)"

    printf "\r  ⏱ %2d / 30 min | RPS: %-6s | Fails: %-5s | Replicas: api=%s app=%s db=%s" \
        "$ELAPSED_MIN" "$CURRENT_RPS" "$TOTAL_FAIL" "$R_API" "$R_APP" "$R_DB"

    if (( ELAPSED > TEST_DURATION_SEC + 30 )); then
        echo ""
        log_ok "Test completed after ${ELAPSED_MIN} minutes"
        break
    fi

    sleep 15
done

# ── Step 10: Final metrics collection ──
log_info "Running final metrics collection..."
ensure_prom_alive
collect_metrics "final" 30

# ── Step 11: Stop QMIX agent ──
log_qmix "Stopping QMIX agent..."
kill "$AGENT_PID" 2>/dev/null || true
wait "$AGENT_PID" 2>/dev/null || true
AGENT_PID=""
log_ok "Agent stopped"

# ── Step 12: Stop Locust ──
log_info "Stopping Locust test..."
curl -sf -X POST http://127.0.0.1:8089/stop &>/dev/null || true

# ── Step 13: Wait for background collectors ──
log_info "Waiting for background collectors to finish..."
for pid in "${BG_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# ── Step 14: Copy agent decisions log ──
if [[ -f "$WORKSPACE_DIR/logs/shadow_decisions.csv" ]]; then
    DECISIONS_DEST="$WORKSPACE_DIR/docs/Final Results/QMIX"
    mkdir -p "$DECISIONS_DEST"
    cp "$WORKSPACE_DIR/logs/shadow_decisions.csv" "$DECISIONS_DEST/qmix_decisions_$(date +%Y%m%d_%H%M%S).csv"
    log_ok "Agent decisions log copied to docs/Final Results/QMIX/"
fi

# ── Step 15: Summary ──
TEST_END=$(date +%s)
TOTAL_MIN=$(( (TEST_END - TEST_START) / 60 ))

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  QMIX TEST COMPLETE${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Start:    $TEST_START_READABLE"
echo "  End:      $(date)"
echo "  Duration: ${TOTAL_MIN} minutes"
echo ""

# Show final replica counts
read -r R_API R_APP R_DB <<< "$(get_replica_counts)"
echo -e "${CYAN}Final replica counts:${NC}"
echo "  api: $R_API  |  app: $R_APP  |  db: $R_DB"
echo ""

echo -e "${YELLOW}Output files:${NC}"
ls -lh "$WORKSPACE_DIR/docs/Final Results"/qmix_metrics_*.json 2>/dev/null | tail -5 || echo "  (none found)"
echo ""
echo -e "${YELLOW}CSV time-series:${NC}"
ls -lh "$WORKSPACE_DIR/docs/Final Results"/*_over_time_qmix.csv 2>/dev/null | sed 's/^/  /' || echo "  (none found)"
echo ""
echo -e "${YELLOW}Agent decisions log:${NC}"
ls -lh "$WORKSPACE_DIR/docs/Final Results/QMIX"/qmix_decisions_*.csv 2>/dev/null | tail -1 || echo "  (none found)"
echo ""
echo -e "${YELLOW}Full agent log:${NC} /tmp/qmix_agent.log"
echo ""
log_ok "All done!"
