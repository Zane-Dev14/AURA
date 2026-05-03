#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

echo "Collecting basic Kubernetes diagnostics for AURA..."

kubectl get nodes -o wide
kubectl get pods -o wide
kubectl get svc -A | grep -E "(api|app|db|locust|prometheus|kube-prom)" || true

# Describe problematic pods
for P in $(kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do
  case "$P" in
    *api*|*app*|*db*|*locust*)
      echo "\n---- describe $P ----"
      kubectl describe pod "$P" || true
      echo "\n---- logs (api/app/db: envoy & app) for $P ----"
      kubectl logs "$P" -c envoy || true
      kubectl logs "$P" -c api || true
      ;;
  esac
done

# Check Envoy admin endpoints by port-forwarding one replica of each service (short lived)
echo "\nAttempting to curl Envoy admin endpoints (9901) for api/app/db pods..."
for SVC in api app db; do
  POD=$(kubectl get pod -l app=${SVC} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -n "$POD" ]]; then
    echo "\n-- $SVC pod: $POD --"
    echo "Port-forwarding pod $POD:9901 -> localhost:9901 (background)..."
    kubectl port-forward "$POD" 9901:9901 >/dev/null 2>&1 &
    PF_PID=$!
    sleep 1
    echo "curl -sS http://localhost:9901/server_info || true"
    curl -sS http://localhost:9901/server_info || true
    curl -sS http://localhost:9901/ready || true
    kill $PF_PID 2>/dev/null || true
  else
    echo "$SVC: no pod found"
  fi
done

# Check for ServiceMonitor CRD
if kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1; then
  echo "\nServiceMonitor CRD present"
else
  echo "\nServiceMonitor CRD NOT present"
fi

# Prometheus targets
if kubectl get svc -n monitoring >/dev/null 2>&1; then
  echo "\nPrometheus services in monitoring namespace:"
  kubectl get pods -n monitoring -o wide || true
  echo "\nPort-forward Prometheus to localhost:9090 and print /targets (brief)"
  kubectl port-forward -n monitoring svc/kube-prom-kube-prometheus-prometheus 9090:9090 >/dev/null 2>&1 &
  PF=$!
  sleep 2
  curl -sS http://localhost:9090/targets || true
  kill $PF 2>/dev/null || true
else
  echo "\nNo monitoring namespace found"
fi

echo "\nDiagnostics collection complete."
