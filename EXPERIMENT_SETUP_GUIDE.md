# AURA FGCS Experiment Setup Guide

## Quick Start (4.5 Hour Automated Run)

```bash
# 1. Ensure Rancher Desktop is configured
# - 32GB RAM allocated
# - 14 CPU cores allocated
# - Kubernetes enabled

# 2. Run the automated benchmark suite
./run_full_benchmark_suite.sh

# 3. Come back in 4.5 hours - results will be ready!
```

## What the Script Does

### Phase 1: Baseline Experiments (3 trials × 30 min = 1.5 hours)
- Static 1 replica for all services
- Measures baseline performance without autoscaling
- Collects metrics: P99 latency, RPS, error rate, CPU usage

### Phase 2: HPA Experiments (3 trials × 30 min = 1.5 hours)
- Kubernetes HPA with 70% CPU target
- Scales based on CPU utilization only
- Represents standard Kubernetes autoscaling

### Phase 3: QMIX Experiments (3 trials × 30 min = 1.5 hours)
- AURA agent with QMIX policy
- Uses 16-dimensional observations (queue depth, RPS derivative, etc.)
- **Fixed APP tier bug** - controller now allows APP to scale
- **Reactive mode** - 15-second cooldown (vs HPA's 30-60 seconds)

## Load Test Configuration

```python
LOCUST_USERS = 150          # Concurrent users (optimized for M4 Max)
LOCUST_SPAWN_RATE = 15      # Users spawned per second
TRIAL_DURATION = 30 minutes # Per trial
COOLDOWN = 3 minutes        # Between trials
```

### Traffic Pattern
- Mix of UI views (60%) and API calls (40%)
- Random wait time: 0.5-2 seconds between requests
- Simulates realistic user behavior

## Results Structure

```
experimental_results_YYYYMMDD_HHMMSS/
├── baseline/
│   ├── trial_1/
│   │   ├── metrics.json
│   │   ├── api_deployment.json
│   │   ├── app_deployment.json
│   │   └── db_deployment.json
│   ├── trial_2/
│   └── trial_3/
├── hpa/
│   ├── trial_1/
│   │   ├── metrics.json
│   │   ├── hpa_status.log
│   │   └── hpa_final_state.yaml
│   ├── trial_2/
│   └── trial_3/
├── qmix/
│   ├── trial_1/
│   │   ├── metrics.json
│   │   ├── controller.log
│   │   ├── replicas.log
│   │   └── deployments.json
│   ├── trial_2/
│   └── trial_3/
├── SUMMARY_REPORT.txt
└── benchmark_suite.log
```

## Post-Experiment Analysis

### 1. View Summary
```bash
cat experimental_results_*/SUMMARY_REPORT.txt
```

### 2. Statistical Analysis
```bash
# Install scipy if needed
pip3 install scipy pandas

# Run analysis
python3 tools/analyze_results.py experimental_results_YYYYMMDD_HHMMSS/
```

This generates:
- Mean ± std for all metrics
- Paired t-tests (QMIX vs Baseline, QMIX vs HPA)
- Effect sizes (Cohen's d)
- p-values for significance
- LaTeX table for paper

### 3. Generate Figures
```bash
python3 tools/generate_paper_figures.py experimental_results_YYYYMMDD_HHMMSS/
```

This creates:
- API P99 latency comparison (with error bars)
- APP P99 latency comparison (with error bars)
- Throughput comparison
- CPU usage comparison
- Replica count over time
- All figures saved as PDF for paper

## Troubleshooting

### Issue: Cluster not accessible
```bash
# Fix kubeconfig (0.0.0.0 → 127.0.0.1)
sed -i.bak 's/0.0.0.0/127.0.0.1/g' ~/.kube/config

# Verify
kubectl get nodes
```

### Issue: Nodes not Ready
```bash
# Check node status
kubectl get nodes

# If nodes are NotReady, restart k3d
k3d cluster delete aura
k3d cluster create aura --config infra/k3d-cluster.yaml

# Redeploy services
./tools/deploy_stack.sh
```

### Issue: APP tier still at 1 replica in QMIX
```bash
# Check controller logs
tail -f experimental_results_*/qmix/trial_1/controller.log

# Look for "APP RECOVERY OVERRIDE" messages
# If not present, bug fix may not be working

# Verify fix is in code
grep -A 5 "app_needs_recovery" deployment/agent_controller.py
```

### Issue: Locust not starting
```bash
# Check Locust is accessible
curl http://localhost:30089

# If not, check pod
kubectl get pod -l app=locust
kubectl logs -l app=locust

# Restart if needed
kubectl delete pod -l app=locust
```

### Issue: Prometheus not accessible
```bash
# Check Prometheus
curl http://localhost:30090/-/healthy

# If not accessible, check port forward
kubectl get svc -n monitoring | grep prometheus

# Verify NodePort is 30090
kubectl get svc -n monitoring kube-prom-kube-prometheus-prometheus -o yaml | grep nodePort
```

## Expected Results

### QMIX Should Outperform HPA In:

1. **Latency** (Primary Metric)
   - API P99: 40-50% lower than baseline
   - APP P99: 30-40% lower than baseline
   - Reason: Predictive features (queue depth, RPS derivative)

2. **Reactivity**
   - Faster scale-up response (15s vs 30-60s)
   - Reason: Shorter cooldown, predictive signals

3. **Resource Efficiency**
   - Lower CPU usage for same throughput
   - Reason: Smarter scaling decisions

### HPA Advantages (Expected):

1. **Simplicity**
   - No training required
   - Standard Kubernetes feature

2. **Stability**
   - Well-tested, production-proven
   - Conservative scaling

### If QMIX Underperforms:

**Check these:**
1. APP tier scaled correctly (should be 2-3 replicas under load)
2. Controller logs show scale-up decisions
3. No errors in controller.log
4. Cooldown is 15 seconds (not 30)
5. Load test actually ran (check Locust logs)

**Possible fixes:**
1. Reduce cooldown further: `export AURA_COOLDOWN_SEC=10`
2. Adjust SLA threshold: `export AURA_P99_SLO=400`
3. Check reward weights in simulator/config.yaml

## Paper-Ready Checklist

After experiments complete:

- [ ] All 9 trials completed successfully
- [ ] No errors in benchmark_suite.log
- [ ] APP tier scaled in QMIX trials (replicas > 1)
- [ ] APP error rate < 5% in QMIX trials
- [ ] Throughput comparable across configs (within 20%)
- [ ] Statistical analysis shows p < 0.05 for key metrics
- [ ] Figures generated with error bars
- [ ] Summary report reviewed

## Time Estimates

| Task | Duration |
|------|----------|
| Pre-flight checks | 5 min |
| Baseline trials (3×) | 1.5 hours |
| HPA trials (3×) | 1.5 hours |
| QMIX trials (3×) | 1.5 hours |
| Analysis & reporting | 5 min |
| **Total** | **~4.5 hours** |

## Resource Requirements

- **RAM**: 32GB allocated to Rancher Desktop
- **CPU**: 14 cores allocated
- **Disk**: ~2GB for results
- **Network**: Localhost only (no external traffic)

## Safety Features

The script includes:
- ✅ Pre-flight checks (cluster, services, Prometheus, Locust)
- ✅ Automatic kubeconfig fixing (0.0.0.0 → 127.0.0.1)
- ✅ Graceful error handling
- ✅ Progress logging every minute
- ✅ Cooldown between trials
- ✅ Automatic cleanup (HPA removal, replica reset)
- ✅ Comprehensive logging

## Next Steps After Experiments

1. **Review Results**
   ```bash
   cat experimental_results_*/SUMMARY_REPORT.txt
   ```

2. **Statistical Analysis**
   ```bash
   python3 tools/analyze_results.py experimental_results_*/
   ```

3. **Generate Figures**
   ```bash
   python3 tools/generate_paper_figures.py experimental_results_*/
   ```

4. **Update Paper**
   - Copy LaTeX table from analysis output
   - Add figures to docs/figures/
   - Update results section with new data
   - Add statistical test results (t-values, p-values)

5. **Verify Claims**
   - [ ] QMIX outperforms baseline (p < 0.05)
   - [ ] QMIX comparable or better than HPA
   - [ ] APP tier scaled correctly
   - [ ] Error rates acceptable (<5%)
   - [ ] Throughput comparable

## Contact

If issues arise:
1. Check benchmark_suite.log for errors
2. Review individual trial logs
3. Verify cluster health: `kubectl get nodes`
4. Check service status: `kubectl get pods`

---

**Ready to run?**
```bash
./run_full_benchmark_suite.sh
```

Then come back in 4.5 hours! ☕