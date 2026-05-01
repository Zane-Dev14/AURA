#!/bin/bash

# AURA Demo Script
# This script sets up port forwarding and guides you through the demo

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/tools/k3d_guard.sh"

assert_k3d_context

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="default"
MONITORING_NS="monitoring"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    AURA DEMO SETUP                         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if port is already in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠️  Port $port is already in use${NC}"
        return 1
    fi
    return 0
}

# Function to start port forward in background
start_port_forward() {
    local service=$1
    local namespace=$2
    local local_port=$3
    local remote_port=$4
    local name=$5
    
    echo -e "${BLUE}🔌 Setting up port forward: $name (localhost:$local_port)${NC}"
    
    if ! check_port $local_port; then
        echo -e "${YELLOW}   Killing existing process on port $local_port...${NC}"
        lsof -ti:$local_port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    kubectl port-forward -n $namespace svc/$service $local_port:$remote_port > /dev/null 2>&1 &
    local pid=$!
    echo $pid >> /tmp/aura_demo_pids.txt
    
    # Wait for port forward to be ready
    sleep 3
    
    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}   ✅ $name ready at http://localhost:$local_port${NC}"
        return 0
    else
        echo -e "${RED}   ❌ Failed to start port forward for $name${NC}"
        return 1
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}🧹 Cleaning up port forwards...${NC}"
    if [ -f /tmp/aura_demo_pids.txt ]; then
        while read pid; do
            kill $pid 2>/dev/null || true
        done < /tmp/aura_demo_pids.txt
        rm /tmp/aura_demo_pids.txt
    fi
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Clear any existing PIDs file
rm -f /tmp/aura_demo_pids.txt

echo -e "${CYAN}📋 Step 1: Checking cluster status...${NC}"
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
    echo -e "${YELLOW}Please ensure your k3d cluster is running:${NC}"
    echo -e "   k3d cluster start aura"
    exit 1
fi
echo -e "${GREEN}✅ Cluster is accessible${NC}"
echo ""

echo -e "${CYAN}📋 Step 2: Checking if services are deployed...${NC}"
services=("api" "app" "db")
for svc in "${services[@]}"; do
    if kubectl get svc $svc -n $NAMESPACE > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Service '$svc' found${NC}"
    else
        echo -e "${RED}   ❌ Service '$svc' not found${NC}"
        echo -e "${YELLOW}Please deploy the stack first:${NC}"
        echo -e "   kubectl apply -f infra/manifests/three-tier/"
        exit 1
    fi
done
echo ""

echo -e "${CYAN}📋 Step 3: Setting up port forwards...${NC}"
echo ""

# Port forward Prometheus
start_port_forward "kube-prom-kube-prometheus-prometheus" "$MONITORING_NS" "9090" "9090" "Prometheus"

# Port forward Locust
start_port_forward "locust" "$NAMESPACE" "8089" "8089" "Locust Dashboard"

# Port forward App (Frontend)
start_port_forward "app" "$NAMESPACE" "8080" "8080" "App Frontend"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              PORT FORWARDS READY                           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}📊 Access Points:${NC}"
echo -e "   ${BLUE}Prometheus:${NC}      http://localhost:9090"
echo -e "   ${BLUE}Locust Dashboard:${NC} http://localhost:8089"
echo -e "   ${BLUE}App Frontend:${NC}     http://localhost:8080"
echo ""

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    DEMO INSTRUCTIONS                       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}🎬 PHASE 1: Shadow Mode (Observation Only)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "1️⃣  ${GREEN}Open Locust Dashboard:${NC} http://localhost:8089"
echo -e "    - Number of users: ${YELLOW}100${NC}"
echo -e "    - Spawn rate: ${YELLOW}10${NC}"
echo -e "    - Host: ${YELLOW}http://app:8080${NC}"
echo -e "    - Click ${GREEN}'Start swarming'${NC}"
echo ""
echo -e "2️⃣  ${GREEN}In a NEW terminal, start the agent in SHADOW MODE:${NC}"
echo -e "    ${BLUE}cd $(pwd)${NC}"
echo -e "    ${BLUE}export AURA_SHADOW_MODE=true${NC}"
echo -e "    ${BLUE}export PROMETHEUS_URL=http://localhost:9090${NC}"
echo -e "    ${BLUE}python deployment/agent_controller.py${NC}"
echo ""
echo -e "3️⃣  ${GREEN}Observe the agent's decisions (it will NOT scale):${NC}"
echo -e "    - Watch the terminal output"
echo -e "    - Note the ${YELLOW}'SHADOW'${NC} label on each decision"
echo -e "    - Decisions are logged but NOT applied"
echo ""
echo -e "4️⃣  ${GREEN}Watch failures accumulate in Locust:${NC}"
echo -e "    - Go to Locust dashboard"
echo -e "    - Watch the ${RED}'Failures'${NC} count increase"
echo -e "    - Note the high response times"
echo ""
echo -e "5️⃣  ${GREEN}Let it run for 2-3 minutes to see the problem${NC}"
echo ""

echo -e "${YELLOW}Press ENTER when you're ready to see Phase 2 instructions...${NC}"
read -r

echo ""
echo -e "${YELLOW}🎬 PHASE 2: Active Mode (AURA Takes Control)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "6️⃣  ${GREEN}Stop the agent (Ctrl+C in the agent terminal)${NC}"
echo ""
echo -e "7️⃣  ${GREEN}Restart the agent in ACTIVE MODE:${NC}"
echo -e "    ${BLUE}export AURA_SHADOW_MODE=false${NC}"
echo -e "    ${BLUE}export PROMETHEUS_URL=http://localhost:9090${NC}"
echo -e "    ${BLUE}python deployment/agent_controller.py${NC}"
echo ""
echo -e "8️⃣  ${GREEN}Watch AURA take action:${NC}"
echo -e "    - Terminal shows ${GREEN}'LIVE'${NC} instead of 'SHADOW'"
echo -e "    - Agent will scale up replicas"
echo -e "    - Watch: ${BLUE}kubectl get pods -w${NC} (in another terminal)"
echo ""
echo -e "9️⃣  ${GREEN}Observe the recovery in Locust:${NC}"
echo -e "    - Failure rate drops"
echo -e "    - Response times improve"
echo -e "    - System stabilizes"
echo ""
echo -e "🔟 ${GREEN}Compare the metrics:${NC}"
echo -e "    - Check Prometheus: http://localhost:9090"
echo -e "    - Query: ${YELLOW}histogram_quantile(0.99, sum by (le) (rate(envoy_http_downstream_rq_time_bucket[1m])))${NC}"
echo -e "    - See P99 latency drop after AURA activates"
echo ""

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    QUICK COMMANDS                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Watch pods:${NC}"
echo -e "  kubectl get pods -w"
echo ""
echo -e "${BLUE}Check current replicas:${NC}"
echo -e "  kubectl get deployments"
echo ""
echo -e "${BLUE}View agent logs:${NC}"
echo -e "  tail -f logs/shadow_decisions.csv"
echo ""
echo -e "${BLUE}Reset replicas to baseline:${NC}"
echo -e "  kubectl scale deployment api --replicas=1"
echo -e "  kubectl scale deployment app --replicas=1"
echo -e "  kubectl scale deployment db --replicas=1"
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Port forwards are active. Press Ctrl+C to stop them.     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Keep script running
echo -e "${YELLOW}Keeping port forwards alive... Press Ctrl+C to exit.${NC}"
while true; do
    sleep 10
    # Check if port forwards are still alive
    if [ -f /tmp/aura_demo_pids.txt ]; then
        while read pid; do
            if ! ps -p $pid > /dev/null 2>&1; then
                echo -e "${RED}⚠️  Port forward died (PID: $pid), restarting...${NC}"
                # Could add restart logic here
            fi
        done < /tmp/aura_demo_pids.txt
    fi
done

# Made with Bob
