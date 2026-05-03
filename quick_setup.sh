#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/tools/k3d_guard.sh"

assert_k3d_context

KUBE_CONTEXT="$(kubectl config current-context)"
K3D_CLUSTER="${KUBE_CONTEXT#k3d-}"

echo "🚀 AURA Quick Setup Script"
echo "=========================="

# Wait for cluster creation to complete
echo "⏳ Waiting for k3d cluster creation..."
while ! kubectl cluster-info &>/dev/null; do
    sleep 2
done
echo "✅ Cluster is ready!"

# Docker builds: don't block on dangling (<none>) images.
echo "⏳ Checking Docker image state..."
DANGLING_COUNT=$(docker images --filter dangling=true -q | wc -l | tr -d ' ')
if [[ "${DANGLING_COUNT}" != "0" ]]; then
  echo "⚠️  Found ${DANGLING_COUNT} dangling (<none>) images; continuing anyway."
else
  echo "✅ No dangling Docker images detected."
fi

# Import images to k3d
echo "📦 Importing images to k3d cluster..."
k3d image import project-api:local project-app:local project-db:local project-locust:local -c "$K3D_CLUSTER"

# Install Prometheus
echo "📊 Installing Prometheus stack..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update
helm upgrade --install kube-prom prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values metrics/prometheus/prometheus-values.yaml \
  --wait --timeout 5m

# Deploy ConfigMaps
echo "⚙️  Deploying ConfigMaps..."
kubectl apply -f infra/manifests/three-tier/envoy-config-api.yaml
kubectl apply -f infra/manifests/three-tier/envoy-config-app.yaml
kubectl apply -f infra/manifests/three-tier/envoy-config-db.yaml
kubectl apply -f infra/manifests/three-tier/mysql-init-script.yaml

# Deploy three-tier app
echo "🏗️  Deploying three-tier application..."
kubectl apply -f infra/manifests/three-tier/db.yaml
kubectl apply -f infra/manifests/three-tier/api.yaml
kubectl apply -f infra/manifests/three-tier/app.yaml

# Deploy ServiceMonitors
echo "📈 Deploying ServiceMonitors..."
# Ensure ServiceMonitor CRD exists (installed by Prometheus Operator)
if ! kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1; then
  echo "⚠️  ServiceMonitor CRD not found — installing Prometheus Operator CRDs..."
  kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
  echo "⏳ Waiting for CRDs to be registered..."
  sleep 5
fi
kubectl apply -f infra/manifests/three-tier/servicemonitor.yaml

# Deploy Locust
echo "🦗 Deploying Locust..."
kubectl apply -f microservices/locust/locust.yaml

# Wait for pods
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=db --timeout=180s
kubectl wait --for=condition=ready pod -l app=api --timeout=180s
kubectl wait --for=condition=ready pod -l app=app --timeout=180s
kubectl wait --for=condition=ready pod -l app=locust --timeout=180s

echo ""
echo "✅ AURA Setup Complete!"
echo "======================="
echo ""
echo "📊 Access Points:"
echo "  Prometheus:  http://localhost:30090"
echo "  Grafana:     http://localhost:32322 (admin/admin)"
echo "  Locust:      http://localhost:30089"
echo ""
echo "🔍 Check status:"
echo "  kubectl get pods -A"
echo ""
echo "🎬 Run demo:"
echo "  ./run_demo.sh"

# Made with Bob
