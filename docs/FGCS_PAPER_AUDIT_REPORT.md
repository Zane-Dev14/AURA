# AURA Paper Audit Report: FGCS Readiness Assessment

**Date:** 2026-04-27  
**Auditor:** Bob (Plan Mode)  
**Scope:** Paper claims vs. verified local evidence  
**Target:** FGCS journal submission readiness  
**Constraint:** No GCP usage, no pepsib2bi path changes

---

## EXECUTIVE SUMMARY

This audit evaluates the AURA paper's claims against verified local evidence from k3d cluster experiments. The paper contains **mostly accurate numerical data** from actual experiments, but suffers from **critical methodological weaknesses** that must be addressed for FGCS publication:

### Critical Issues for FGCS:
1. ❌ **Single-run experiments** (n=1) with no statistical significance testing
2. ⚠️ **APP tier bug** confounds results interpretation (20.95% error rate)
3. ⚠️ **Unfair throughput comparison** (QMIX 663 RPS vs HPA 2375 RPS due to bug)
4. ⚠️ **k3d local cluster** limitations vs. production claims
5. ⚠️ **Overstated generalizability** from limited experimental scope

### Strengths:
- ✅ All numerical values verified against actual data files
- ✅ Honest disclosure of APP tier issues (in IEEE version)
- ✅ Predictive features (queue depth, RPS derivative) demonstrated
- ✅ API tier results are solid and reproducible

---

## 1. SUPPORTED CLAIMS (Backed by Local Evidence)

### 1.1 API Tier Performance ✅
**Claim:** "QMIX achieves 46% lower API P99 latency (23.13ms vs 43.13ms baseline)"

**Evidence:**
- Source: [`docs/Final Results/combined_qmix.json`](docs/Final Results/combined_qmix.json:59) - API P99: 23.13ms
- Source: [`docs/Final Results/combined_baseline.json`](docs/Final Results/combined_baseline.json:59) - API P99: 43.13ms
- Calculation: (43.13 - 23.13) / 43.13 = 46.37% improvement
- **Status:** ✅ VERIFIED - Accurate to actual data

**Limitations:**
- Single 30-minute run (n=1)
- No confidence intervals
- No statistical significance testing

### 1.2 Resource Efficiency ✅
**Claim:** "QMIX uses 55% less CPU than HPA (0.90 vs 2.00 cores)"

**Evidence:**
- Source: [`docs/Final Results/combined_qmix.json`](docs/Final Results/combined_qmix.json:17) - CPU: 0.90 cores
- Source: [`docs/Final Results/combined_hpa.json`](docs/Final Results/combined_hpa.json:17) - CPU: 2.00 cores
- Calculation: (2.00 - 0.90) / 2.00 = 55% reduction
- **Status:** ✅ VERIFIED - Accurate to actual data

### 1.3 Predictive Features ✅
**Claim:** "16-dimensional observation space including queue depth, RPS derivative, CPU history"

**Evidence:**
- Source: [`deployment/builder.py`](deployment/builder.py:189-233) - 16-dimensional observation implementation
- Source: [`marl/policies/qmix.py`](marl/policies/qmix.py:28-70) - OBS_DIM=16, ACTION_DIM=10
- Source: [`simulator/config.yaml`](simulator/config.yaml:52-56) - Reward weights α=2.0, β=2.5, γ=1.5
- **Status:** ✅ VERIFIED - Implementation matches claims

### 1.4 Cluster Configuration ✅
**Claim:** "2-node k3d cluster, 8 cores total, 15.5GB memory"

**Evidence:**
- Source: [`docs/Final Results/combined_qmix.json`](docs/Final Results/combined_qmix.json:9-14) - Cluster hardware specs
- Source: [`infra/k3d-cluster.yaml`](infra/k3d-cluster.yaml) - Cluster configuration
- **Status:** ✅ VERIFIED - Accurate cluster specs

### 1.5 Startup Times ✅
**Claim:** "API: 25s, APP: 21s, DB: 15s"

**Evidence:**
- Source: [`simulator/config.yaml`](simulator/config.yaml:10,20,30) - Ready times match
- **Status:** ✅ VERIFIED - Accurate to configuration

---

## 2. UNSUPPORTED/OVERSTATED CLAIMS

### 2.1 APP Tier Performance ❌
**Claim (aura.tex):** "APP tier shows elevated P99 latency (780.81ms) and 20.95% error rate... This is a reward function tuning issue"

**Evidence:**
- Source: [`APP_TIER_FIX_VERIFICATION_REPORT.md`](APP_TIER_FIX_VERIFICATION_REPORT.md:1-319) - Documents controller bug
- Source: [`docs/AURA_IEEE_Paper.tex`](docs/AURA_IEEE_Paper.tex:788-794) - Acknowledges "control-path bug"
- **Reality:** APP scale-up was blocked by tier-coupled guard logic bug, not reward function

**Problem:**
- [`aura.tex`](docs/aura.tex:800-807) attributes issue to "reward function tuning"
- [`AURA_IEEE_Paper.tex`](docs/AURA_IEEE_Paper.tex:788-794) correctly identifies "control-path bug"
- **Inconsistency:** Two versions tell different stories

**FGCS Correction Needed:**
```
The APP tier limitation (780.81ms P99, 20.95% error rate) was caused by a 
controller implementation bug where tier-coupled veto logic incorrectly 
suppressed APP scale-up actions under API bottleneck conditions. This bug 
has been identified and fixed (see APP_TIER_FIX_VERIFICATION_REPORT.md), 
but the reported results reflect the buggy controller behavior. Future work 
will re-evaluate with the corrected controller to assess true APP-tier 
performance.
```

### 2.2 Throughput Comparison Fairness ⚠️
**Claim:** "HPA achieves 3.6× higher throughput (2375 vs 663 RPS)"

**Evidence:**
- QMIX: 663 RPS with APP at 1 replica (due to bug)
- HPA: 2375 RPS with APP at 2.59 avg replicas
- **Problem:** Comparison is unfair because QMIX's APP was artificially constrained

**FGCS Correction Needed:**
```
The throughput comparison (QMIX 663 RPS vs HPA 2375 RPS) is confounded by 
the APP-tier controller bug. With APP artificially pinned at 1 replica, 
QMIX's throughput was bottlenecked. This comparison should be re-evaluated 
after bug resolution to provide a fair assessment of QMIX's throughput 
capabilities.
```

### 2.3 Statistical Rigor ❌
**Claim:** Implicit claim of reproducibility and significance

**Evidence:**
- All results from **single 30-minute run** (n=1)
- No repeated trials
- No confidence intervals
- No statistical significance testing
- No variance/standard deviation reported

**FGCS Correction Needed:**
```
LIMITATIONS: All reported results are from single 30-minute experimental 
runs (n=1) on a local k3d cluster. No statistical significance testing was 
performed. The observed performance differences (e.g., 46% API latency 
improvement) should be interpreted as preliminary findings requiring 
validation through multi-trial experiments with proper statistical analysis 
before drawing definitive conclusions.
```

### 2.4 Production Generalizability ⚠️
**Claim:** "evaluated on a live k3d Kubernetes cluster" (implies production-readiness)

**Evidence:**
- k3d is a **local development cluster** running in Docker containers
- Single host, 8 cores, 15.5GB RAM
- No network latency variability
- No multi-datacenter considerations
- No real production workload patterns

**FGCS Correction Needed:**
```
DEPLOYMENT CONTEXT: Experiments were conducted on a k3d local development 
cluster (2 Docker-based nodes on a single host). While k3d provides a 
realistic Kubernetes API surface, it lacks production characteristics such 
as network latency variability, multi-host scheduling constraints, and 
real-world failure modes. Generalization to production GKE/EKS/AKS clusters 
requires additional validation.
```

### 2.5 "Production Runs" Language ⚠️
**Claim (Line 765-766):** "All results are from actual production runs on the live cluster"

**Evidence:**
- k3d is **not a production cluster**
- It's a local development environment
- Misleading terminology

**FGCS Correction Needed:**
```
Replace "production runs" with "experimental runs" or "live cluster runs"
Add footnote: "k3d is a local Kubernetes development cluster, not a 
production cloud environment"
```

---

## 3. STATISTICAL WEAKNESSES

### 3.1 Single-Run Experiments (Critical)
**Issue:** All results from n=1 experiments

**Impact on Claims:**
- API latency improvement: 46% ± **unknown variance**
- Resource efficiency: 55% ± **unknown variance**
- Throughput difference: 3.6× ± **unknown variance**

**FGCS Requirement:**
- Minimum n=3 trials for reproducibility
- Report mean ± standard deviation
- Perform t-tests or Mann-Whitney U tests
- Report p-values and confidence intervals

**Recommended Addition:**
```latex
\subsection{Statistical Methodology}
Each experimental configuration (Baseline, QMIX, HPA) was evaluated in a 
single 30-minute trial due to resource constraints. We acknowledge this 
limits statistical rigor and plan multi-trial validation in future work. 
The reported metrics should be interpreted as point estimates rather than 
statistically validated findings.
```

### 3.2 No Variance Reporting
**Issue:** No error bars, confidence intervals, or standard deviations

**FGCS Correction:**
- Add error bars to all figures (even if estimated from time-series variance)
- Report within-run variance from 15-second Prometheus samples
- Acknowledge lack of between-run variance

### 3.3 No Significance Testing
**Issue:** No statistical tests for claimed improvements

**FGCS Correction:**
```
Without multi-trial data, we cannot perform between-group significance 
testing. The 46% API latency improvement and 55% CPU reduction should be 
considered preliminary findings pending statistical validation.
```

---

## 4. K3D LOCAL LIMITATIONS

### 4.1 Infrastructure Constraints
**k3d Characteristics:**
- Single-host Docker-based cluster
- No real network latency (loopback only)
- No node failures or network partitions
- Limited to 8 cores, 15.5GB RAM
- No persistent storage considerations

**Production Differences:**
- Multi-host clusters with network latency
- Node failures and pod rescheduling
- Larger resource pools (100s of cores)
- Persistent volume management
- Multi-AZ/region considerations

**FGCS Correction:**
```latex
\subsection{Experimental Limitations}
Our evaluation used a k3d local cluster (2 Docker-based nodes on a single 
host) rather than a production cloud environment. This choice enabled rapid 
iteration but introduces limitations:
\begin{itemize}
\item No network latency variability between nodes
\item No node failure scenarios
\item Limited resource pool (8 cores vs. production 100s)
\item Single-host scheduling (no cross-datacenter considerations)
\end{itemize}
Future work will validate AURA on production GKE/EKS clusters to assess 
generalizability.
```

### 4.2 Workload Limitations
**Current Workload:**
- Locust synthetic load generator
- 30-minute duration
- Single traffic pattern

**Missing Validation:**
- Real production traces (e.g., Google cluster traces)
- Multi-day sustained load
- Diurnal traffic patterns
- Flash crowd scenarios
- Gradual load ramps

**FGCS Correction:**
```
The evaluation used synthetic Locust-generated traffic over 30 minutes. 
Real production workloads exhibit more complex patterns (diurnal cycles, 
flash crowds, gradual trends) that may challenge QMIX's predictive 
capabilities differently. Extended validation with production traces is 
needed.
```

---

## 5. APP TIER BUG RESOLUTION STATUS

### 5.1 Bug Description
**Source:** [`APP_TIER_FIX_VERIFICATION_REPORT.md`](APP_TIER_FIX_VERIFICATION_REPORT.md:85-138)

**Root Cause:**
- Tier-coupled veto logic in [`deployment/agent_controller.py`](deployment/agent_controller.py:192-199)
- APP scale-up blocked when API bottlenecked AND APP healthy
- Missing recovery override for APP under pressure

**Fix Applied:**
- Added `app_needs_recovery()` function with sensitive thresholds
- Modified veto condition: `if api_is_bottleneck(api_metrics) and not app_needs_recovery(m)`
- Added recovery override: forces APP scale-up when breaching SLO

**Verification Status:**
- ✅ Integration tests passed (6/6 scenarios)
- ✅ Live controller tests passed (5/5 scenarios)
- ❌ Full end-to-end load test NOT performed
- ❌ Multi-trial statistical validation NOT performed

### 5.2 Impact on Paper Claims
**Problem:** Paper results reflect **buggy controller** behavior

**Affected Claims:**
1. APP P99 latency (780.81ms) - artificially inflated
2. APP error rate (20.95%) - artificially inflated
3. Throughput comparison (663 vs 2375 RPS) - unfair
4. "Reward function tuning issue" explanation - incorrect

**FGCS Correction Required:**
```latex
\textbf{Post-Publication Note:} After paper submission, we identified a 
controller implementation bug that artificially constrained APP-tier 
scaling. The bug has been fixed and verified through unit tests 
(APP_TIER_FIX_VERIFICATION_REPORT.md), but the reported experimental 
results reflect the buggy behavior. We are conducting follow-up experiments 
to assess QMIX's true APP-tier performance with the corrected controller.
```

---

## 6. THROUGHPUT FAIRNESS ISSUES

### 6.1 Unfair Comparison
**Current Comparison:**
- QMIX: 663 RPS (APP at 1 replica due to bug)
- HPA: 2375 RPS (APP at 2.59 avg replicas)
- Ratio: 3.58× difference

**Problem:** QMIX artificially handicapped by bug

**Fair Comparison Would Require:**
- QMIX with fixed controller
- APP allowed to scale to 2-3 replicas
- Re-run experiments with identical conditions

### 6.2 Paper Language Issues
**Current (Line 809-813):**
> "HPA achieves 3.6× higher throughput (2375 vs. 663 RPS) by scaling both 
> API and APP tiers uniformly. QMIX's lower throughput is a direct 
> consequence of the APP tier remaining at 1 replica."

**Problem:** Implies intentional design choice, not bug

**FGCS Correction:**
```
HPA achieves 3.58× higher throughput (2375 vs 663 RPS) by scaling both API 
and APP tiers. QMIX's lower throughput is a consequence of the APP tier 
remaining at 1 replica due to a controller bug (subsequently fixed). This 
throughput comparison should be re-evaluated with the corrected controller 
to provide a fair assessment of QMIX's capabilities.
```

---

## 7. RECOMMENDED CORRECTIONS FOR FGCS

### 7.1 Abstract Changes
**Current:**
> "Experimental results demonstrate AURA's predictive features enable 46% 
> better API P99 latency"

**Recommended:**
> "Experimental results from a local k3d cluster (n=1 trial) demonstrate 
> AURA's predictive features enable 46% better API P99 latency (23.13ms vs 
> 43.13ms baseline), though APP-tier results were confounded by a controller 
> bug subsequently identified and fixed."

### 7.2 Add Limitations Section
**New Section (before Conclusion):**
```latex
\section{Limitations and Future Work}
\label{sec:limitations}

\subsection{Experimental Limitations}
Our evaluation has several limitations that should be addressed in future 
work:

\textbf{Single-Trial Experiments:} All results are from single 30-minute 
runs (n=1) without statistical significance testing. Multi-trial validation 
with proper statistical analysis is needed to establish reproducibility and 
confidence intervals.

\textbf{Local Cluster Environment:} Experiments used a k3d local 
development cluster rather than a production cloud environment. This lacks 
network latency variability, node failures, and multi-datacenter 
considerations present in real deployments.

\textbf{Controller Bug Impact:} The reported APP-tier results (780.81ms 
P99, 20.95% error rate) reflect a controller implementation bug that 
artificially constrained APP scaling. The bug has been fixed and verified, 
but re-evaluation is needed to assess true APP-tier performance.

\textbf{Synthetic Workload:} The evaluation used Locust-generated synthetic 
traffic over 30 minutes. Real production workloads exhibit more complex 
patterns (diurnal cycles, flash crowds) requiring extended validation.

\subsection{Future Work}
\begin{itemize}
\item Multi-trial experiments (n≥3) with statistical significance testing
\item Production cloud deployment (GKE/EKS) validation
\item Extended evaluation with real production traces
\item Per-service SLA thresholds and adaptive reward weights
\item Comparison with additional baselines (KEDA, Predictive HPA)
\end{itemize}
```

### 7.3 Results Section Corrections
**Line 788-794 (APP Tier Challenge):**

**Current (aura.tex):**
> "This is a reward function tuning issue, not a fundamental limitation"

**Recommended:**
> "Post-hoc analysis revealed this was caused by a controller implementation 
> bug where tier-coupled veto logic incorrectly suppressed APP scale-up 
> actions. The bug has been fixed (see APP_TIER_FIX_VERIFICATION_REPORT.md), 
> but the reported results reflect the buggy behavior. Re-evaluation with 
> the corrected controller is needed to assess true APP-tier performance."

### 7.4 Methodology Section Additions
**Add to Section 5.1 (Setup):**
```latex
\textbf{Statistical Methodology:} Each configuration was evaluated in a 
single 30-minute trial due to resource constraints. We acknowledge this 
limits statistical rigor and prevents significance testing. The reported 
metrics should be interpreted as point estimates from preliminary 
experiments rather than statistically validated findings. Future work will 
conduct multi-trial validation (n≥3) with proper statistical analysis.
```

---

## 8. FGCS READINESS ASSESSMENT

### 8.1 Current Status: ⚠️ NOT READY

**Blocking Issues:**
1. ❌ No statistical significance testing (n=1 experiments)
2. ❌ APP tier bug confounds results interpretation
3. ❌ Overstated generalizability from k3d to production
4. ❌ Missing limitations section
5. ❌ Inconsistent bug explanation between paper versions

### 8.2 Required Changes for FGCS Submission

**CRITICAL (Must Fix):**
1. Add comprehensive Limitations section
2. Correct APP tier bug explanation throughout
3. Add statistical methodology disclaimer
4. Replace "production runs" with "experimental runs"
5. Add k3d vs production environment clarification

**HIGH PRIORITY (Should Fix):**
6. Add error bars to figures (from within-run variance)
7. Acknowledge throughput comparison unfairness
8. Add future work section with multi-trial validation plan
9. Tone down generalizability claims
10. Add "preliminary findings" qualifiers

**MEDIUM PRIORITY (Nice to Have):**
11. Conduct n=3 trials and add statistical tests
12. Re-run with fixed APP controller
13. Add production GKE/EKS validation
14. Compare with additional baselines (KEDA, Predictive HPA)

### 8.3 Estimated Timeline to FGCS-Ready

**Option A: Minimal Corrections (2-3 days)**
- Fix critical issues 1-5
- Add limitations section
- Revise claims to be more conservative
- **Result:** Submittable but weak on rigor

**Option B: Statistical Validation (2-3 weeks)**
- Conduct n=3 trials for each configuration
- Add statistical tests and confidence intervals
- Re-run with fixed APP controller
- **Result:** Strong submission with proper rigor

**Option C: Full Production Validation (2-3 months)**
- Deploy to GKE/EKS production cluster
- Multi-trial validation with real workloads
- Extended evaluation (multi-day runs)
- **Result:** Publication-ready with strong impact

---

## 9. SPECIFIC DOCUMENT CORRECTIONS

### 9.1 docs/aura.tex
**Lines to Modify:**

**Line 765-766:**
```latex
% BEFORE:
All results are from actual production runs on the live cluster

% AFTER:
All results are from experimental runs on a local k3d development cluster
```

**Line 791-807 (APP Tier Challenge):**
```latex
% BEFORE:
This is a \textit{reward function tuning issue}, not a fundamental 
limitation of the QMIX approach.

% AFTER:
Post-hoc analysis revealed this was caused by a controller implementation 
bug where tier-coupled veto logic incorrectly suppressed APP scale-up 
actions under API bottleneck conditions. The bug has been identified and 
fixed (see APP_TIER_FIX_VERIFICATION_REPORT.md), but the reported results 
reflect the buggy controller behavior. Re-evaluation with the corrected 
controller is needed to assess true APP-tier performance.
```

**Add before Section 7 (Conclusion):**
```latex
\section{Limitations and Future Work}
[Insert full limitations section from 7.2 above]
```

### 9.2 docs/AURA_IEEE_Paper.tex
**Lines to Modify:**

**Line 777-783:**
```latex
% CURRENT (mostly correct):
QMIX achieves \textbf{46.37\% lower API P99 latency}

% ADD QUALIFIER:
QMIX achieves \textbf{46.37\% lower API P99 latency} in a single 30-minute 
trial (n=1)
```

**Line 788-794 (APP Tier Challenge):**
- Already correctly identifies "control-path bug"
- ✅ No changes needed (this version is more accurate)

### 9.3 docs/CORRECTIONS_APPLIED.md
**Add Section:**
```markdown
## FGCS Readiness Issues (Identified 2026-04-27)

### Statistical Rigor
- All results from n=1 experiments
- No significance testing performed
- No confidence intervals reported
- **Action:** Add limitations section acknowledging this

### APP Tier Bug Impact
- Results reflect buggy controller behavior
- Bug fixed but experiments not re-run
- **Action:** Clarify in paper that results are pre-fix

### k3d vs Production
- Local development cluster, not production
- **Action:** Tone down generalizability claims
```

---

## 10. CONCLUSION

### Summary of Findings

**Supported Claims (✅):**
- API tier latency improvement (46%) - numerically accurate
- Resource efficiency vs HPA (55% less CPU) - numerically accurate
- Predictive features implementation - verified in code
- Cluster configuration - accurate specs

**Unsupported/Overstated Claims (❌):**
- APP tier "reward function tuning issue" - actually controller bug
- Throughput comparison fairness - confounded by bug
- Statistical significance - no testing performed (n=1)
- Production generalizability - k3d is local dev environment
- "Production runs" language - misleading terminology

**Critical Gaps for FGCS:**
1. No statistical rigor (n=1, no significance tests)
2. APP tier bug confounds interpretation
3. Overstated generalizability from k3d to production
4. Missing limitations section
5. Inconsistent bug explanations between versions

### Recommendation

**Status:** ⚠️ **NOT READY for FGCS submission in current form**

**Path Forward:**
1. **Immediate (2-3 days):** Apply critical corrections from Section 7
2. **Short-term (2-3 weeks):** Conduct n=3 trials with statistical tests
3. **Long-term (2-3 months):** Production GKE/EKS validation

**Minimum for Submission:**
- Add comprehensive limitations section
- Correct APP tier bug explanation
- Add statistical methodology disclaimer
- Tone down generalizability claims
- Replace "production" with "experimental" language

**Ideal for Strong Submission:**
- Complete Option B (Statistical Validation)
- Re-run with fixed APP controller
- Add multi-trial results with confidence intervals
- Conduct at least preliminary GKE validation

---

## APPENDIX: Evidence Cross-Reference

### Verified Data Sources
- [`docs/Final Results/combined_qmix.json`](docs/Final Results/combined_qmix.json) - QMIX metrics
- [`docs/Final Results/combined_hpa.json`](docs/Final Results/combined_hpa.json) - HPA metrics
- [`docs/Final Results/combined_baseline.json`](docs/Final Results/combined_baseline.json) - Baseline metrics
- [`APP_TIER_FIX_VERIFICATION_REPORT.md`](APP_TIER_FIX_VERIFICATION_REPORT.md) - Bug documentation
- [`QMIX_vs_Baseline_FINAL_ANALYSIS.md`](QMIX_vs_Baseline_FINAL_ANALYSIS.md) - Detailed analysis
- [`docs/PAPER_VERIFICATION_REPORT.md`](docs/PAPER_VERIFICATION_REPORT.md) - Previous audit
- [`docs/CORRECTIONS_APPLIED.md`](docs/CORRECTIONS_APPLIED.md) - Applied corrections

### Code Verification
- [`deployment/agent_controller.py`](deployment/agent_controller.py:85-95,192-208) - Controller logic
- [`deployment/builder.py`](deployment/builder.py:189-233) - Observation space
- [`marl/policies/qmix.py`](marl/policies/qmix.py:28-70) - QMIX architecture
- [`simulator/config.yaml`](simulator/config.yaml:52-56) - Reward weights
- [`infra/k3d-cluster.yaml`](infra/k3d-cluster.yaml) - Cluster config

---

**Report Generated:** 2026-04-27  
**Next Review:** After corrections applied  
**FGCS Submission Target:** After Option A (minimal) or Option B (statistical) completion