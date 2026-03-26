# AURA Paper Corrections Required

## Executive Summary

The original `docs/aura.tex` (1147 lines) is **excellent quality** with proper TikZ diagrams, professional IEEE formatting, and comprehensive structure. However, it contains **incorrect performance data** that doesn't match the actual experimental results in the repository.

## Critical Data Corrections Needed

### 1. Observation Space (CRITICAL)

**Current (WRONG in aura.tex line 363-373):**
- Claims 6-dimensional state space
- Missing predictive features

**Correct (from `deployment/builder.py` lines 216-233):**
- **16-dimensional state space** with predictive features:
  1. CPU utilization (normalized)
  2. Memory utilization (normalized)
  3. P50 latency (log-normalized)
  4. P95 latency (log-normalized)
  5. P99 latency (log-normalized)
  6. RPS (normalized)
  7. Error rate
  8. Queue depth (envoy_http_downstream_rq_active)
  9. **RPS derivative** (predictive)
  10. Desired replicas
  11. Ready replicas
  12. Ready/desired ratio
  13. **CPU history t-1** (predictive)
  14. **CPU derivative** (predictive)
  15. **Downstream pressure** (predictive)
  16. P95 latency (duplicate for emphasis)

### 2. Reward Function Weights (CRITICAL)

**Current (WRONG in aura.tex line 434):**
```
α = 10, β = 0.5, γ = 1.0
```

**Correct (from `simulator/config.yaml` lines 52-56):**
```
α (SLA weight) = 2.0
β (cost weight) = 2.5
γ (flapping weight) = 1.5
SLA threshold = 350ms (not 500ms)
```

### 3. Experimental Results (CRITICAL)

**Current (WRONG in aura.tex):**
- Claims QMIX reduces SLA violations by 65%
- Claims better performance than HPA
- No mention of error rates or failures

**Correct (from `docs/Final Results/combined_*.json`):**

#### QMIX Results:
- CPU requested: 0.90 cores
- Total RPS: 662.98
- API P99: **23.13ms** (GOOD)
- APP P99: **780.81ms** (BAD - bottleneck!)
- APP Error Rate: **20.95%** (CRITICAL FAILURE!)
- Avg API replicas: 3.76
- Avg APP replicas: 1.0 (NOT SCALED - root cause)

#### HPA Results:
- CPU requested: 2.00 cores
- Total RPS: 2375.33 (3.6× higher than QMIX!)
- API P99: 99.87ms
- APP P99: 369.51ms (2.1× better than QMIX)
- Error Rate: 0.0%
- Avg API replicas: 3.34
- Avg APP replicas: 2.59 (properly scaled)

#### Baseline Results:
- CPU requested: 0.60 cores
- Total RPS: 448.34
- API P99: 43.13ms
- APP P99: 48.59ms
- Error Rate: 0.0%
- All replicas: 1.0 (static)

### 4. Cost Analysis (CRITICAL)

**Correct calculations:**
- QMIX: $348.12/month (0.90 cores) - 13.1% MORE expensive than baseline
- HPA: $643.50/month (2.00 cores) - 85.0% MORE expensive than QMIX
- Baseline: $307.70/month (0.60 cores) - CHEAPEST

### 5. Training Hyperparameters (from `marl/policies/qmix.py`)

**Correct values:**
- OBS_DIM = 16 (not 6)
- ACTION_DIM = 10 (not 3)
- EPOCHS = 200
- STEPS_PER_EPOCH = 1000
- BATCH_SIZE = 256
- LR = 5e-4
- GAMMA = 0.98
- EPS_START = 0.10, EPS_END = 0.02
- REPLAY_SIZE = 400,000
- MIN_REPLICAS = 1, MAX_REPLICAS = 10

### 6. Pod Startup Times (from `simulator/config.yaml`)

**Correct values:**
- API: 25s total (3s pending + 12s container + 10s warmup)
- APP: 21s total (3s pending + 10s container + 8s warmup)
- DB: 15s total (2s pending + 8s container + 5s warmup)

### 7. Decision Interval

**Correct:** 30 seconds (not 15 seconds as claimed in some places)

## What to Keep from aura.tex

✅ **Keep these (they're excellent):**
1. All TikZ diagrams (Fig 1 - system architecture)
2. IEEE formatting and structure
3. Related work section
4. QMIX algorithm explanation
5. Table formatting style
6. Bibliography
7. Overall paper organization

## What to Fix

❌ **Fix these sections:**
1. **Section 4.1 (State Space):** Update to 16 dimensions with predictive features
2. **Section 4.3 (Reward Function):** Update weights to (2.0, 2.5, 1.5)
3. **Section 6 (Experimental Evaluation):** Replace ALL performance numbers with actual data
4. **Section 6.2 (Results):** Add honest analysis of QMIX failures:
   - 20.95% APP error rate
   - APP tier bottleneck (not scaled)
   - Lower throughput than HPA
   - Higher cost than baseline
5. **Section 7 (Discussion):** Add limitations section discussing:
   - Incomplete multi-tier coordination
   - Performance degradation under high load
   - Need for error-rate penalties in reward function
6. **Abstract:** Rewrite to reflect actual results (both successes and failures)
7. **Conclusion:** Honest assessment of trade-offs

## Recommended Approach

**Option 1: Edit aura.tex directly**
- Keep the excellent structure and diagrams
- Update only the incorrect data sections
- Add honest limitations discussion
- This preserves the high-quality formatting

**Option 2: Use aura.tex as template**
- Copy the structure and TikZ code
- Rewrite content sections with correct data
- Maintain the professional appearance

## Key Message for Paper

The paper should honestly present:

**QMIX Strengths:**
- 46% better API P99 latency than baseline (23.13ms vs 43.13ms)
- Predictive features enable proactive API scaling
- Lower cost than HPA (45.9% cheaper)

**QMIX Weaknesses:**
- 20.95% APP error rate (CRITICAL FAILURE)
- APP tier not scaled (bottleneck)
- 72% lower throughput than HPA (662.98 vs 2375.33 RPS)
- 13.1% more expensive than baseline

**Lessons Learned:**
- Reward function must explicitly penalize error rates
- Training distribution must stress all tiers
- Multi-tier coordination requires cross-tier queue metrics
- Simple solutions (baseline) can be effective for predictable workloads

## Files to Reference

- `deployment/builder.py` (lines 189-233): Observation space
- `marl/policies/qmix.py` (lines 28-70): Hyperparameters
- `simulator/config.yaml`: Reward weights, pod startup times
- `docs/Final Results/combined_*.json`: Actual experimental data
- `docs/QMIX_vs_Baseline_Detailed_Analysis.md`: Comprehensive analysis
- `QMIX_vs_Baseline_FINAL_ANALYSIS.md`: Final analysis with insights

## Action Items

1. ✅ Keep `aura.tex` structure and diagrams
2. ❌ Update observation space to 16 dimensions
3. ❌ Update reward weights to (2.0, 2.5, 1.5)
4. ❌ Replace all experimental results with actual data
5. ❌ Add honest limitations section
6. ❌ Rewrite abstract to reflect actual results
7. ❌ Add error rate analysis
8. ❌ Document APP tier bottleneck
9. ❌ Compare against HPA honestly (HPA wins on throughput and errors)
10. ❌ Emphasize predictive features as key innovation