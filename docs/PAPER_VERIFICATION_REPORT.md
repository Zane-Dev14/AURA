# AURA Paper Verification Report

## Executive Summary
This document verifies all values, tables, code blocks, and formatting issues in the AURA IEEE paper (`docs/aura.tex`).

---

## 1. CRITICAL ISSUES FOUND

### 1.1 RBAC Code Block (Lines 1128-1140) - **NOT FOUND IN REPOSITORY**

**Paper Claims:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: aura-controller
rules:
- apiGroups: ["apps"]
  resources: ["deployments/scale"]
  verbs: ["get", "patch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["list"]
```

**Verification Result:** ❌ **DOES NOT EXIST**
- Searched entire repository for `rbac.authorization.k8s.io` - **0 results**
- No RBAC manifests found in `infra/manifests/` directory
- This is **HALLUCINATED** content

**Action Required:** REMOVE this code block entirely or add disclaimer that RBAC is not yet implemented.

---

### 1.2 Python Code Blocks (Lines 999-1026) - **SHOULD BE REMOVED**

**Paper Contains:**
```python
import pandas as pd
import matplotlib.pyplot as plt
# ... plotting code ...
```

**Issue:** User explicitly stated "I DO NOT WANT CODE IN THE PAPER. RUN THAT CODE, GET THE DIAGRAMS."

**Action Required:** 
- REMOVE all Python code blocks
- Generate actual diagrams from the CSV data
- Insert generated diagrams as figures

---

### 1.3 Missing Table I - Observation Space Table

**User Requested:**
```
TABLE I
AURA OBSERVATION SPACE (16 DIMENSIONS PER AGENT)
```

**Current Status:** ❌ **NOT IN PAPER**

**Action Required:** Add this table after the observation space equation (after line 402)

---

## 2. NUMERICAL VALUE VERIFICATION

### 2.1 Resource Requests - **INCORRECT VALUES**

**Paper Claims (Table, Line 1067-1073):**
| Service | CPU Request | Memory Request |
|---------|-------------|----------------|
| API | 150m | 312 MB |
| APP | 200m | 375 MB |
| DB | 250m | 562 MB |

**Actual Values from Manifests:**
- **API** (`infra/manifests/three-tier/api.yaml:21-23`):
  - CPU: **100m** (NOT 150m) ❌
  - Memory: **256Mi** (NOT 312MB) ❌

- **APP** (`infra/manifests/three-tier/app.yaml:26-28`):
  - CPU: **100m** (NOT 200m) ❌
  - Memory: **256Mi** (NOT 375MB) ❌

- **DB** (`infra/manifests/three-tier/db.yaml:58-60`):
  - CPU: **200m** (NOT 250m) ❌
  - Memory: **512Mi** (NOT 562MB) ❌

**Action Required:** Correct Table "Per-Pod Resource Requests" with actual manifest values.

---

### 2.2 Performance Metrics - **VERIFIED CORRECT** ✅

**Paper Values vs. Actual Data:**

From `combined_qmix.json`:
- API P99: **23.13ms** ✅ (line 59)
- APP P99: **780.81ms** ✅ (line 71)
- APP Error Rate: **20.95%** ✅ (line 68: 0.2095238095238095)
- Total CPU: **0.90 cores** ✅ (calculated from line 17: 0.9)
- Total RPS: **663 RPS** ✅ (line 48: 662.98, rounded)
- API avg replicas: **3.76** ✅ (line 60)

From `combined_baseline.json`:
- API P99: **43.13ms** ✅ (line 59)
- APP P99: **48.59ms** ✅ (line 71)
- Total CPU: **0.60 cores** ✅ (line 17)
- Total RPS: **448 RPS** ✅ (line 48: 448.34, rounded)

From `combined_hpa.json`:
- API P99: **99.87ms** ✅ (line 59)
- APP P99: **369.51ms** ✅ (line 71)
- Total CPU: **2.00 cores** ✅ (line 17)
- Total RPS: **2375 RPS** ✅ (line 48: 2375.33, rounded)
- API avg replicas: **3.34** ✅ (line 60)
- APP avg replicas: **2.59** ✅ (line 72)

**All performance metrics in Tables are CORRECT** ✅

---

### 2.3 Startup Times - **VERIFIED CORRECT** ✅

**Paper Claims (Line 468-471):**
- API: 25s (3s pending + 12s container + 10s warmup)
- APP: 21s (3s pending + 10s container + 8s warmup)
- DB: 15s (2s pending + 8s container + 5s warmup)

**Actual from `simulator/config.yaml`:**
- API: ready: **25.0** ✅ (line 10)
- APP: ready: **21.0** ✅ (line 20)
- DB: ready: **15.0** ✅ (line 30)

**Startup times are CORRECT** ✅

---

### 2.4 Reward Weights - **VERIFIED CORRECT** ✅

**Paper Claims (Line 456):**
- α = 2.0, β = 2.5, γ = 1.5

**Actual from `simulator/config.yaml` (lines 52-56):**
```yaml
reward:
  cost_weight: 2.5      # β
  sla_weight: 2.0       # α
  flapping_weight: 1.5  # γ
  sla_threshold_ms: 350
```

**Reward weights are CORRECT** ✅

---

## 3. FORMATTING ISSUES

### 3.1 Long Filenames Breaking Margins

**Issue:** Lines with filenames in monospace font exceed margins:
- Line 1086: `(\texttt{infra/manifests/three-tier/envoy-config-*.yaml})`
- Line 1099: `(\texttt{infra/manifests/three-tier/*-servicemonitor.yaml})`
- Line 1105: `(\texttt{deployment/agent\_controller.py})`
- Line 1149: `(\texttt{microservices/locust/locustfile\_production.py})`

**Action Required:** Use line breaks or abbreviate paths:
```latex
(\texttt{infra/manifests/three-tier/\\envoy-config-*.yaml})
```

---

### 3.2 Placeholder Text

**Line 937-939:**
```
Figure~\ref{fig:replicas-timeseries} (not shown due to space constraints,
but can be generated from \texttt{replicas\_over\_time\_*.csv})
```

**Issue:** User stated: "DO NOT have things like 'Figure ?? (not shown due to space constraints...'"

**Action Required:** Either:
1. Generate and include the figure, OR
2. Remove the reference entirely

---

### 3.3 List Formatting Issues

**Line 941:** `\begin{itemize}[leftmargin=*,noitemsep,topsep=2pt]`

**Issue:** User stated this causes formatting issues.

**Action Required:** Simplify to `\begin{itemize}` or use standard spacing.

---

## 4. MISSING DIAGRAMS

### 4.1 Replica Scaling Time Series
**Status:** Referenced but not shown (line 937)
**Data Available:** `docs/Final Results/replicas_over_time_*.csv`
**Action:** Generate diagram

### 4.2 P99 Latency Time Series
**Status:** Referenced (line 959)
**Data Available:** `docs/Final Results/p99_over_time_*.csv`
**Action:** Generate diagram

### 4.3 CPU Usage Time Series
**Status:** Referenced (line 977)
**Data Available:** `docs/Final Results/cpu_usage_over_time_*.csv`
**Action:** Generate diagram

---

## 5. CORRECT ELEMENTS ✅

### 5.1 System Architecture Diagram
- Lines 231-292: TikZ diagram is present and well-formatted ✅

### 5.2 Training Convergence Plot
- Lines 862-898: TikZ plot with actual data points ✅

### 5.3 Hyperparameters Table
- Lines 559-587: All values match code ✅

### 5.4 Results Tables
- Lines 739-754: Performance comparison - all values correct ✅
- Lines 786-801: Resource utilization - all values correct ✅

---

## 6. SUMMARY OF REQUIRED CORRECTIONS

### CRITICAL (Must Fix):
1. ❌ **REMOVE RBAC code block** (lines 1128-1140) - NOT IN REPOSITORY
2. ❌ **REMOVE Python plotting code** (lines 999-1026)
3. ❌ **ADD Table I** - Observation Space (16 dimensions)
4. ❌ **CORRECT resource requests** in Table (lines 1067-1073)

### HIGH PRIORITY:
5. ⚠️ **Fix filename formatting** to prevent margin overflow
6. ⚠️ **Remove placeholder text** about figures "not shown"
7. ⚠️ **Generate missing diagrams** from CSV data

### MEDIUM PRIORITY:
8. 📝 **Simplify list formatting** (remove complex itemize parameters)

---

## 7. DATA INTEGRITY VERIFICATION

✅ **All performance metrics verified against actual JSON data**
✅ **All simulator parameters verified against config.yaml**
✅ **All startup times verified**
✅ **All reward weights verified**
❌ **Resource requests DO NOT match manifests**
❌ **RBAC configuration DOES NOT EXIST**

---

## CONCLUSION

The paper contains **mostly accurate data** from experiments, but has:
1. **Hallucinated RBAC code** that doesn't exist
2. **Incorrect resource request values** 
3. **Code blocks that should be diagrams**
4. **Missing observation space table**
5. **Formatting issues** with long filenames

**Recommendation:** Fix critical issues before publication.