# AURA Local-Only Fix Checklist

Scope constraints:
- No cloud operations.
- No non-k3d cluster operations.
- All automation must fail closed unless current kubectl context is k3d-*.

## 1) Cluster safety guard
- [x] Add shared k3d context guard helper.
- [x] Source and enforce guard in all operational shell scripts.
- [x] Enforce k3d context in Python collectors/controllers that execute kubectl.

How:
- `tools/k3d_guard.sh` provides `assert_k3d_context`.
- Called at startup from setup/deploy/demo/benchmark scripts.
- `tools/gke_cost_report.py` and `deployment/agent_controller.py` now refuse non-k3d contexts.

## 2) APP-tier bug (controller deadlock)
- [x] Narrow tier-coupled veto so APP scale-up is blocked only when APP is healthy.
- [x] Add APP recovery override when APP is breaching latency/error/queue pressure.
- [x] Add a local sanity-check script for regression protection.

How:
- `deployment/agent_controller.py` now uses `app_needs_recovery` and `api_is_bottleneck`.
- APP unhealthy state now forces scale-up (`actions['app'] = 1`) instead of staying pinned.
- `tools/validate_controller_fix.py` verifies guard behavior.

## 3) Fair experiment initialization
- [x] Force baseline and QMIX runs to start from 1/1/1 replicas.
- [x] Wait for rollout readiness before load injection.

How:
- `tools/run_baseline_test.sh` and `tools/run_qmix_test.sh` now call `reset_replicas_to_single`.
- HPA already had this behavior and was retained.

## 4) Repeatable multi-run workflow
- [x] Add orchestrator for repeated local trials (default: 5).
- [x] Keep per-trial outputs isolated by run/trial folders.

How:
- `tools/run_local_trials.sh` runs baseline -> QMIX -> HPA for each trial.
- Outputs written under `docs/Final Results/trials/local_trials_<timestamp>/...`.

## 5) Statistical validation from local runs
- [x] Summarize means/std/95% CI and significance-like p-values.
- [x] Integrate summarization into multi-trial workflow.

How:
- `tools/summarize_trials.py` computes per-mode stats and pairwise permutation p-values.
- `tools/run_local_trials.sh` writes `summary.txt` automatically.

## 6) Local resource scaling
- [x] Replace minimal setup script with configurable k3d provisioning.
- [x] Increase default local topology/resources.

How:
- `tools/setup_k3d.sh` now supports `--agents`, `--servers`, memory sizing, and `--recreate`.
- Defaults now target a larger local cluster (1 server, 3 agents).

## 7) Path/portability hardening
- [x] Remove hardcoded absolute workspace paths.
- [x] Derive active k3d cluster name from current context for image import.

How:
- Benchmark scripts now resolve workspace from script location.
- `k3d image import` now uses context-derived cluster name instead of hardcoded `aura`.

## 8) Validation pass
- [ ] Run full 5-trial local benchmark and inspect APP error-rate reduction.
- [ ] Re-check throughput gap after APP fix.
- [ ] Regenerate final comparative summary for newest trial batch.

How:
- Run `bash tools/run_local_trials.sh --trials 5 --duration 30`.
- Review generated `summary.txt` in run output directory.
