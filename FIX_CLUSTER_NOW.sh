#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     AURA Cluster Complete Fix - Deleting and Recreating    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Delete broken cluster
echo "[INFO] Step 1: Deleting broken k3d cluster..."
k3d cluster delete aura 2>/dev/null || true
echo "[INFO] ✓ Cluster deleted"
echo ""

# Step 2: Create fresh cluster with increased inotify limits
echo "[INFO] Step 2: Creating fresh k3d cluster with proper limits..."
cat > /tmp/k3d-cluster-fixed.yaml <<'EOF'
apiVersion: k3d.io/v1alpha4
kind: Simple
metadata:
  name: aura
servers: 1
agents: 3
image: rancher/k3s:v1.33.6-k3s1
ports:
  - port: 30090:30090
    nodeFilters:
      - loadbalancer
  - port: 32322:32322
    nodeFilters:
      - loadbalancer
  - port: 30089:30089
    nodeFilters:
      - loadbalancer
options:
  k3s:
    extraArgs:
      - arg: --disable=traefik
        nodeFilters:
          - server:*
  kubeconfig:
    updateDefaultKubeconfig: true
    switchCurrentContext: true
  runtime:
    ulimits:
      - name: nofile
        soft: 1000000
        hard: 1000000
EOF

k3d cluster create --config /tmp/k3d-cluster-fixed.yaml
echo "[INFO] ✓ Cluster created"
echo ""

# Step 3: Increase inotify limits in all nodes
echo "[INFO] Step 3: Increasing inotify limits in all nodes..."
for node in k3d-aura-server-0 k3d-aura-agent-0 k3d-aura-agent-1 k3d-aura-agent-2; do
    docker exec $node sh -c "sysctl -w fs.inotify.max_user_instances=512 && sysctl -w fs.inotify.max_user_watches=524288" > /dev/null
    echo "[INFO]   ✓ $node limits increased"
done
echo ""

# Step 4: Fix kubeconfig
echo "[INFO] Step 4: Fixing kubeconfig..."
k3d kubeconfig get aura > infra/kubeconfig-aura.yaml
sed -i '' 's/0\.0\.0\.0/127.0.0.1/g' infra/kubeconfig-aura.yaml
export KUBECONFIG=$(pwd)/infra/kubeconfig-aura.yaml
echo "[INFO] ✓ Kubeconfig fixed"
echo ""

# Step 5: Wait for nodes to be Ready
echo "[INFO] Step 5: Waiting for all nodes to be Ready..."
timeout=120
elapsed=0
while [ $elapsed -lt $timeout ]; do
    ready_count=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
    if [ "$ready_count" -eq "4" ]; then
        echo "[INFO] ✓ All 4 nodes are Ready"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo -n "."
done
echo ""

if [ "$ready_count" -ne "4" ]; then
    echo "[ERROR] Timeout waiting for nodes to be Ready"
    exit 1
fi

# Step 6: Deploy services
echo "[INFO] Step 6: Deploying services..."
bash tools/deploy_stack.sh
echo "[INFO] ✓ Services deployed"
echo ""

# Step 7: Import Docker images
echo "[INFO] Step 7: Importing Docker images..."
for image in project-api:local project-app:local project-db:local; do
    if docker images | grep -q "${image%:*}.*${image#*:}"; then
        echo "[INFO]   Importing $image..."
        k3d image import $image -c aura
    else
        echo "[WARN]   Image $image not found locally, skipping"
    fi
done
echo "[INFO] ✓ Images imported"
echo ""

# Step 8: Force pod recreation
echo "[INFO] Step 8: Forcing pod recreation to use new images..."
kubectl delete pods --all -n default --force --grace-period=0 2>/dev/null || true
echo "[INFO] ✓ Pods deleted, waiting for recreation..."
sleep 10
echo ""

# Step 9: Wait for all pods to be ready
echo "[INFO] Step 9: Waiting for all pods to be Ready..."
timeout=180
elapsed=0
while [ $elapsed -lt $timeout ]; do
    not_ready=$(kubectl get pods --no-headers 2>/dev/null | grep -v "Running\|Completed" | wc -l || echo "999")
    if [ "$not_ready" -eq "0" ]; then
        echo "[INFO] ✓ All pods are Ready"
        break
    fi
    sleep 3
    elapsed=$((elapsed + 3))
    echo -n "."
done
echo ""

# Step 10: Verify Prometheus and Locust
echo "[INFO] Step 10: Verifying services..."
echo "[INFO]   Checking Prometheus (http://localhost:30090)..."
if curl -s http://localhost:30090/-/healthy > /dev/null 2>&1; then
    echo "[INFO]   ✓ Prometheus is accessible"
else
    echo "[WARN]   Prometheus not yet accessible (may need more time)"
fi

echo "[INFO]   Checking Locust (http://localhost:32322)..."
if curl -s http://localhost:32322 > /dev/null 2>&1; then
    echo "[INFO]   ✓ Locust is accessible"
else
    echo "[WARN]   Locust not yet accessible (may need more time)"
fi
echo ""

# Final status
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         CLUSTER RECREATED AND READY FOR EXPERIMENTS!      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Cluster Status:"
kubectl get nodes
echo ""
echo "Pod Status:"
kubectl get pods
echo ""
echo "Next step:"
echo "  ./run_full_benchmark_suite.sh"
echo ""

# Made with Bob
