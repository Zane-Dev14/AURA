# AURA Quick Fix Guide

## If Setup Fails

### 1. Check Cluster Status
```bash
kubectl cluster-info
k3d cluster list
```

### 2. Restart Cluster
```bash
k3d cluster stop aura
k3d cluster start aura
```

### 3. Re-import Images
```bash
k3d image import project-api:local project-app:local project-db:local project-locust:local -c aura
```

### 4. Check Pod Status
```bash
kubectl get pods -A
kubectl describe pod <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace>
```

### 5. Restart Failed Pods
```bash
kubectl delete pod <pod-name> -n <namespace>
```

### 6. Check Services
```bash
kubectl get svc -A
kubectl get endpoints -A
```

### 7. Port Forward Manually (if NodePort fails)
```bash
# Prometheus
kubectl port-forward -n monitoring svc/kube-prom-kube-prometheus-prometheus 9090:9090 &

# Grafana
kubectl port-forward -n monitoring svc/kube-prom-grafana 3000:80 &

# Locust
kubectl port-forward svc/locust 8089:8089 &
```

### 8. Check Prometheus Targets
```bash
# Open http://localhost:30090/targets
# All targets should be UP
```

### 9. Reinstall Prometheus
```bash
helm uninstall kube-prom -n monitoring
helm install kube-prom prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values metrics/prometheus/prometheus-values.yaml \
  --wait --timeout 5m
```

### 10. Redeploy Applications
```bash
kubectl delete -f infra/manifests/three-tier/
kubectl apply -f infra/manifests/three-tier/
```

## Common Issues

### Issue: Pods stuck in ImagePullBackOff
**Fix:** Images not imported to k3d
```bash
k3d image import project-api:local project-app:local project-db:local project-locust:local -c aura
kubectl delete pod -l app=api
kubectl delete pod -l app=app
kubectl delete pod -l app=db
kubectl delete pod -l app=locust
```

### Issue: Prometheus not scraping metrics
**Fix:** Check ServiceMonitor
```bash
kubectl get servicemonitor -A
kubectl apply -f infra/manifests/three-tier/servicemonitor.yaml
```

### Issue: DB pod CrashLoopBackOff
**Fix:** Check MySQL initialization
```bash
kubectl logs -l app=db --tail=50
kubectl delete pod -l app=db
```

### Issue: Cannot access NodePort services
**Fix:** Check k3d port mappings
```bash
k3d cluster delete aura
k3d cluster create --config infra/k3d-cluster.yaml
# Re-run setup
```

## Quick Commands

```bash
# Full reset
k3d cluster delete aura
./quick_setup.sh

# Check everything
./verify_setup.sh

# Watch pods
watch kubectl get pods -A

# Stream logs
kubectl logs -f -l app=api
kubectl logs -f -l app=app
kubectl logs -f -l app=db

# Scale manually
kubectl scale deployment api --replicas=3
kubectl scale deployment app --replicas=3
kubectl scale deployment db --replicas=2
```

## Demo Ready Checklist

- [ ] Cluster running: `kubectl cluster-info`
- [ ] All pods ready: `kubectl get pods -A`
- [ ] Prometheus accessible: http://localhost:30090
- [ ] Grafana accessible: http://localhost:32322
- [ ] Locust accessible: http://localhost:30089
- [ ] ServiceMonitors created: `kubectl get servicemonitor`
- [ ] Metrics flowing: Check Prometheus targets
- [ ] Agent controller ready: `python deployment/agent_controller.py`

## Time-Saving Tips

1. **Keep terminal open** with `./run_demo.sh` running
2. **Pre-open browser tabs** for all services
3. **Test Locust** with small load first (10 users)
4. **Monitor logs** in separate terminal: `kubectl logs -f -l app=api`
5. **Have backup**: Keep `kubectl get pods -A` output ready