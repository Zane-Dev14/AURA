#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/k3d_guard.sh"

assert_k3d_context

kubectl apply -f "$ROOT_DIR/infra/manifests/three-tier/envoy-config-api.yaml"
kubectl apply -f "$ROOT_DIR/infra/manifests/three-tier/envoy-config-app.yaml"
kubectl apply -f "$ROOT_DIR/infra/manifests/three-tier/envoy-config-db.yaml"
kubectl apply -f "$ROOT_DIR/infra/manifests/three-tier/mysql-init-script.yaml"
kubectl apply -f "$ROOT_DIR/infra/manifests/three-tier/db.yaml"
kubectl apply -f "$ROOT_DIR/infra/manifests/three-tier/api.yaml"
kubectl apply -f "$ROOT_DIR/infra/manifests/three-tier/app.yaml"
kubectl apply -f "$ROOT_DIR/microservices/locust/locust.yaml"
