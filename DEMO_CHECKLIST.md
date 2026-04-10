# AURA Demo Checklist - 10 Minute Setup

## ✅ Pre-Demo Setup (Complete)

- [x] Deleted old aura cluster
- [x] Created new k3d cluster with config
- [x] Built Docker images (api, app, db, locust)
- [x] Running automated setup script

## 🔄 Current Status

The `quick_setup.sh` script is running and will:
1. Import Docker images to k3d
2. Install Prometheus + Grafana
3. Deploy three-tier application (api, app, db)
4. Deploy Locust load generator
5. Wait for all pods to be ready

## 📋 Verification Steps (Run after setup completes)

```bash
# 1. Run verification script
./verify_setup.sh

# 2. Check all pods are running
kubectl get pods -A

# Expected output:
# - monitoring namespace: prometheus, grafana, operators
# - default namespace: api, app, db, locust (all Running)
```

## 🎬 Demo Flow

### Phase 1: Shadow Mode (Observation)
1. Open Locust: http://localhost:30089
2. Start load: 100 users, spawn rate 10, host: http://app:8080
3. In new terminal:
   ```bash
   export AURA_SHADOW_MODE=true
   export PROMETHEUS_URL=http://localhost:9090
   python deployment/agent_controller.py
   ```
4. Watch failures accumulate in Locust
5. Agent logs show "SHADOW" decisions (not applied)

### Phase 2: Active Mode (AURA Takes Control)
1. Stop agent (Ctrl+C)
2. Restart in active mode:
   ```bash
   export AURA_SHADOW_MODE=false
   export PROMETHEUS_URL=http://localhost:9090
   python deployment/agent_controller.py
   ```
3. Watch agent scale up replicas (logs show "LIVE")
4. Monitor recovery in Locust dashboard
5. Check Prometheus metrics

## 🔗 Access Points

- **Prometheus**: http://localhost:30090
- **Grafana**: http://localhost:32322 (admin/admin)
- **Locust**: http://localhost:30089
- **App Frontend**: http://localhost:8080 (via port-forward)

## 🚨 Quick Fixes

If something fails:
```bash
# Check status
./verify_setup.sh

# View logs
kubectl logs -l app=api --tail=50
kubectl logs -l app=app --tail=50
kubectl logs -l app=db --tail=50

# Restart pod
kubectl delete pod -l app=<service-name>

# Full reset
k3d cluster delete aura
./quick_setup.sh
```

## 📊 Key Metrics to Show

1. **P99 Latency**: Should drop when AURA activates
   ```
   histogram_quantile(0.99, sum by (le) (rate(envoy_http_downstream_rq_time_bucket[1m])))
   ```

2. **Replica Count**: Watch it scale up
   ```bash
   kubectl get deployments
   ```

3. **Error Rate**: Should decrease
   ```
   sum(rate(envoy_http_downstream_rq_xx{envoy_response_code_class="5"}[1m]))
   ```

## ⏱️ Timing

- Setup: ~5-7 minutes (automated)
- Demo Phase 1: 2-3 minutes
- Demo Phase 2: 2-3 minutes
- Total: ~10 minutes

## 🎯 Success Criteria

- [ ] All pods Running
- [ ] Prometheus scraping metrics
- [ ] Locust accessible
- [ ] Agent can query Prometheus
- [ ] Agent can scale deployments
- [ ] Metrics show improvement after AURA activation

## 📝 Talking Points

1. **Problem**: Static scaling can't handle dynamic workloads
2. **Solution**: MARL-based autoscaling with QMIX
3. **Shadow Mode**: Safe observation before deployment
4. **Active Mode**: Intelligent scaling decisions
5. **Results**: Lower latency, fewer errors, optimal resource usage

## 🔧 Troubleshooting Reference

See `QUICK_FIX.md` for detailed troubleshooting steps.

## ✨ Demo Tips

1. Keep `kubectl get pods -A -w` running in a terminal
2. Have Prometheus and Locust open in browser tabs
3. Prepare to explain QMIX architecture if asked
4. Show the agent logs clearly (large font)
5. Emphasize the "SHADOW" vs "LIVE" distinction