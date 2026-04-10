#!/bin/bash
set -e

echo "🚀 AURA Quick Setup Script"
echo "=========================="

# Wait for cluster creation to complete
echo "⏳ Waiting for k3d cluster creation..."
while ! kubectl cluster-info &>/dev/null; do
    sleep 2
done
echo "✅ Cluster is ready!"

# Wait for Locust build to complete
echo "⏳ Waiting for Docker builds to complete..."
while docker images | grep -q "<none>"; do
    sleep 2
done
echo "✅ Docker images built!"

# Import images to k3d
echo "📦 Importing images to k3d cluster..."
k3d image import project-api:local project-app:local project-db:local project-locust:local -c aura

# Install Prometheus
echo "📊 Installing Prometheus stack..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update
helm install kube-prom prometheus-community/kube-prometheus-stack \
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
