#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/tools/k3d_guard.sh"

assert_k3d_context

KUBE_CONTEXT="$(kubectl config current-context)"
K3D_CLUSTER="${KUBE_CONTEXT#k3d-}"
K3D_SERVER_CONTAINER="k3d-${K3D_CLUSTER}-server-0"

echo "🔍 AURA Setup Verification"
echo "=========================="
echo ""

# Check cluster
echo "📦 Cluster Status:"
kubectl cluster-info | head -2
echo ""

# Check namespaces
echo "📂 Namespaces:"
kubectl get ns | grep -E "NAME|default|monitoring"
echo ""

# Check all pods
echo "🐳 All Pods:"
kubectl get pods -A -o wide
echo ""

# Check services
echo "🌐 Services:"
kubectl get svc -A | grep -E "NAME|prometheus|grafana|locust|api|app|db"
echo ""

# Check deployments
echo "📊 Deployments:"
kubectl get deployments -A
echo ""

# Check if images are loaded
echo "🖼️  Docker Images in k3d:"
docker exec "$K3D_SERVER_CONTAINER" crictl images | grep -E "project-|locust"
echo ""

# Test Prometheus
echo "🔬 Testing Prometheus (port 30090):"
if curl -s http://localhost:30090/-/healthy | grep -q "Prometheus"; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus not responding"
fi
echo ""

# Test Locust
echo "🦗 Testing Locust (port 30089):"
if curl -s http://localhost:30089 | grep -q "Locust"; then
    echo "✅ Locust is accessible"
else
    echo "❌ Locust not responding"
fi
echo ""

# Check node ports
echo "🔌 NodePort Services:"
kubectl get svc -A | grep NodePort
echo ""

echo "✅ Verification Complete!"
echo ""
echo "📝 Next Steps:"
echo "  1. Run: ./run_demo.sh"
echo "  2. Open Locust: http://localhost:30089"
echo "  3. Open Prometheus: http://localhost:30090"
echo "  4. Open Grafana: http://localhost:32322 (admin/admin)"

# Made with Bob
