#!/bin/bash
################################################################################
# PRE-EXPERIMENT FIX - FORCE CLUSTER RECREATION
# 
# This script ALWAYS recreates the cluster to ensure clean state
# Fixes all containerd, image, and pod issues
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     AURA Pre-Experiment Fix - CLUSTER RECREATION          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

log_warn "This script will RECREATE the k3d cluster to ensure clean state"
log_warn "All existing pods and data will be lost"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Aborted by user"
    exit 0
fi

# Step 1: Delete existing cluster
log_info "Step 1: Deleting existing k3d cluster..."
k3d cluster delete aura 2>/dev/null || true
sleep 5
log_info "✓ Cluster deleted"

# Step 2: Create fresh cluster
log_info "Step 2: Creating fresh k3d cluster..."
if [ ! -f "infra/k3d-cluster.yaml" ]; then
    log_error "infra/k3d-cluster.yaml not found!"
    exit 1
fi

k3d cluster create aura --config infra/k3d-cluster.yaml
sleep 10
log_info "✓ Cluster created"

# Step 3: Fix kubeconfig (0.0.0.0 → 127.0.0.1)
log_info "Step 3: Fixing kubeconfig..."
KUBECONFIG_PATH="$HOME/.kube/config"
if grep -q "0.0.0.0" "$KUBECONFIG_PATH" 2>/dev/null; then
    sed -i.bak 's/0.0.0.0/127.0.0.1/g' "$KUBECONFIG_PATH"
    log_info "✓ Kubeconfig fixed (0.0.0.0 → 127.0.0.1)"
else
    log_info "✓ Kubeconfig already correct"
fi

sleep 5

# Step 4: Verify cluster is accessible
log_info "Step 4: Verifying cluster accessibility..."
if ! kubectl get nodes &>/dev/null; then
    log_error "Cluster not accessible after creation"
    exit 1
fi
log_info "✓ Cluster accessible"

# Step 5: Check all nodes are Ready
log_info "Step 5: Waiting for all nodes to be Ready..."
MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    NOT_READY=$(kubectl get nodes --no-headers | grep -v " Ready " | wc -l)
    if [ $NOT_READY -eq 0 ]; then
        log_info "✓ All nodes Ready"
        break
    fi
    log_info "Waiting for nodes... ($NOT_READY not ready)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    log_error "Timeout waiting for nodes to be Ready"
    kubectl get nodes
    exit 1
fi

kubectl get nodes

# Step 6: Deploy services
log_info "Step 6: Deploying services..."
if [ ! -f "tools/deploy_stack.sh" ]; then
    log_error "tools/deploy_stack.sh not found!"
    exit 1
fi

bash tools/deploy_stack.sh
log_info "✓ Services deployed"

log_info "Waiting 60 seconds for initial deployment..."
sleep 60

# Step 7: Import Docker images into fresh cluster
log_info "Step 7: Importing Docker images..."

IMAGES_NEEDED=("project-api:local" "project-app:local" "project-db:local")

# Check images exist locally
for img in "${IMAGES_NEEDED[@]}"; do
    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${img}$"; then
        log_error "Image $img not found locally!"
        log_error "Please build images first or check microservices/three-tier/"
        exit 1
    fi
    log_info "✓ Found locally: $img"
done

# Import into fresh cluster (should work now)
for img in "${IMAGES_NEEDED[@]}"; do
    log_info "Importing $img..."
    k3d image import $img -c aura
    if [ $? -eq 0 ]; then
        log_info "✓ Imported: $img"
    else
        log_warn "Import had issues for $img, but continuing..."
    fi
done

# Step 8: Delete all pods to force recreation with images
log_info "Step 8: Forcing pod recreation..."
kubectl delete pod --all --namespace=default
log_info "Waiting 45 seconds for pods to recreate..."
sleep 45

# Step 9: Wait for all pods to be ready
log_info "Step 9: Waiting for all pods to be ready..."

MAX_WAIT=300
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Count pods that are Running with all containers ready
    TOTAL_PODS=$(kubectl get pods --no-headers 2>/dev/null | wc -l)
    RUNNING_PODS=$(kubectl get pods --no-headers 2>/dev/null | grep "Running" | wc -l)
    READY_PODS=$(kubectl get pods --no-headers 2>/dev/null | awk '{if ($2 ~ /^[0-9]+\/[0-9]+$/) {split($2, a, "/"); if (a[1] == a[2]) print}}' | wc -l)
    
    if [ $READY_PODS -eq $TOTAL_PODS ] && [ $TOTAL_PODS -gt 0 ]; then
        log_info "✓ All $TOTAL_PODS pods are ready!"
        break
    fi
    
    log_info "Waiting... ($READY_PODS/$TOTAL_PODS pods ready)"
    
    # Show any problematic pods
    kubectl get pods --no-headers 2>/dev/null | grep -v "Running" | head -3
    
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    log_error "Timeout waiting for pods to be ready"
    log_error "Current pod status:"
    kubectl get pods
    exit 1
fi

echo ""
kubectl get pods
echo ""

# Step 10: Verify each service
log_info "Step 10: Verifying services..."

ALL_READY=true
for svc in api app db locust; do
    READY=$(kubectl get deployment $svc -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    if [ "$READY" -ge 1 ]; then
        log_info "✓ $svc: $READY replica(s) ready"
    else
        log_error "✗ $svc: No ready replicas"
        ALL_READY=false
    fi
done

if [ "$ALL_READY" = false ]; then
    log_error "Some services are not ready"
    exit 1
fi

# Step 11: Test Prometheus
log_info "Step 11: Testing Prometheus..."
PROM_ATTEMPTS=0
while [ $PROM_ATTEMPTS -lt 30 ]; do
    if curl -s http://127.0.0.1:30090/-/healthy 2>/dev/null | grep -q "Prometheus"; then
        log_info "✓ Prometheus is accessible"
        break
    fi
    sleep 2
    PROM_ATTEMPTS=$((PROM_ATTEMPTS + 1))
done

if [ $PROM_ATTEMPTS -ge 30 ]; then
    log_error "✗ Prometheus not accessible at http://127.0.0.1:30090"
    exit 1
fi

# Step 12: Test Locust
log_info "Step 12: Testing Locust..."
LOCUST_ATTEMPTS=0
while [ $LOCUST_ATTEMPTS -lt 30 ]; do
    if curl -s http://127.0.0.1:30089 2>/dev/null | grep -q "Locust"; then
        log_info "✓ Locust is accessible"
        break
    fi
    sleep 2
    LOCUST_ATTEMPTS=$((LOCUST_ATTEMPTS + 1))
done

if [ $LOCUST_ATTEMPTS -ge 30 ]; then
    log_error "✗ Locust not accessible at http://127.0.0.1:30089"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         CLUSTER RECREATED AND READY FOR EXPERIMENTS!      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Cluster Status:${NC}"
kubectl get nodes
echo ""
kubectl get pods
echo ""
echo -e "${BLUE}Next step:${NC}"
echo "  ./run_full_benchmark_suite.sh"
echo ""

exit 0

# Made with Bob
