# AURA Presentation-Ready Diagrams

## Overview
This document describes the two high-quality, presentation-ready diagrams created for the AURA paper and presentations. Both diagrams are grounded in actual repository data and verified against the paper.

## Generated Files

### Diagram 1: AURA Predictive Multi-Agent Autoscaling Architecture
- **Files**: 
  - `AURA_Architecture_Diagram.png` (300 DPI, high-resolution)
  - `AURA_Architecture_Diagram.pdf` (vector format, publication-ready)
- **Generator Script**: `diagram1_architecture.py`
- **Purpose**: Shows the production system architecture and real-time control loop

### Diagram 2: AURA Simulation-to-Production Pipeline
- **Files**:
  - `AURA_Pipeline_Diagram.png` (300 DPI, high-resolution)
  - `AURA_Pipeline_Diagram.pdf` (vector format, publication-ready)
- **Generator Script**: `diagram2_pipeline.py`
- **Purpose**: Shows training, validation, and production deployment lifecycle

## Diagram 1 Specifications

### Architecture Components Shown:
1. **Kubernetes Cluster Boundary**
   - Three-tier microservice chain: API → APP → DB
   - Each service with Envoy sidecar for metrics

2. **Monitoring Stack**
   - Prometheus with 15s scrape interval
   - Metrics collection from Envoy sidecars

3. **AURA Controller** (External to cluster)
   - 3 independent agents (API, APP, DB)
   - QMIX mixing network for global coordination
   - Safety layer with guard rails

4. **Control Loop**
   - 30s decision interval
   - Observation construction from metrics
   - kubectl scaling commands back to services

### Verified Quantitative Annotations:
- ✅ **15s scrape interval** (Prometheus default)
- ✅ **30s control loop** (from `deployment/agent_controller.py:39`)
- ✅ **16-dimensional observation** (from `marl/trainer/train_qmix.py:49`)
- ✅ **3 agents** (API, APP, DB)
- ✅ **10-action discrete space** (from `marl/trainer/train_qmix.py:50`)
- ✅ **Guard rails min=1, max=5** (from `deployment/agent_controller.py:32-38`)
- ✅ **API→APP→DB chain** (verified in manifests)

### Data Sources:
- `deployment/agent_controller.py` - Control loop timing, safety parameters
- `marl/trainer/train_qmix.py` - Observation/action space dimensions
- `infra/manifests/three-tier/*.yaml` - Service configuration
- `simulator/config.yaml` - Service dependencies

## Diagram 2 Specifications

### Pipeline Phases Shown:

#### Phase 1: Training (Offline)
- **Simulator**: 8 parallel environments, synthetic workload
- **Pod Lifecycle Model**: 25s total startup (API:25s, APP:21s, DB:15s)
- **QMIX Training**: 200 epochs × 1000 steps
- **Reward Weights**: α=2.0 (SLA), β=2.5 (cost), γ=1.5 (flapping)

#### Phase 2: Validation
- **Policy Checkpoint**: Trained model (qmix_best.pth)
- **Local Cluster**: K3d/Minikube testing
- **Evaluation Metrics**: P99 latency, CPU usage, cost vs HPA

#### Phase 3: Production Deployment
- **AURA Controller**: 3 QMIX agents, 30s control loop
- **Safety Mechanisms**: Guard rails, cooldown, vetoes
- **Production Results**: 55% cost reduction, 77% latency improvement

#### Continuous Feedback Loop
- Production metrics monitoring
- Performance analysis
- Retraining triggers

### Verified Quantitative Annotations:
- ✅ **25s pod startup** (from `simulator/config.yaml:10,20,30`)
- ✅ **200 epochs × 1000 steps** (from `marl/trainer/train_qmix.py:58-59`)
- ✅ **8 parallel environments** (from `marl/trainer/train_qmix.py:59` comment)
- ✅ **30s production control loop** (from `deployment/agent_controller.py:39`)
- ✅ **Reward weights α=2.0, β=2.5, γ=1.5** (from `simulator/config.yaml:52-56`)
- ✅ **55% cost reduction** (from `docs/PAPER_VERIFICATION_REPORT.md:119`)
- ✅ **77% latency improvement** (from `docs/PAPER_VERIFICATION_REPORT.md:119`)

### Data Sources:
- `simulator/config.yaml` - Pod startup times, reward weights
- `marl/trainer/train_qmix.py` - Training hyperparameters
- `deployment/agent_controller.py` - Production control loop
- `docs/Final Results/combined_*.json` - Performance metrics
- `docs/PAPER_VERIFICATION_REPORT.md` - Verified results

## Design Principles

### Visual Quality:
- **Resolution**: 300 DPI for both PNG and PDF formats
- **Color Scheme**: Professional, color-blind friendly palette
- **Typography**: Clean sans-serif fonts (Arial/Helvetica)
- **Layout**: Clear information hierarchy with proper spacing

### Accuracy:
- All numerical values verified against repository code
- No hallucinated or unsupported claims
- Direct traceability to source files
- Consistent with paper verification report

### Presentation-Ready Features:
- High contrast for projection
- Clear labels and annotations
- Legend and key parameters included
- Suitable for both print and digital use

## Usage Instructions

### For Paper/Publication:
1. Use PDF versions for LaTeX/Overleaf inclusion
2. Reference as figures in IEEE format
3. Include in supplementary materials if needed

### For Presentations:
1. Use PNG versions for PowerPoint/Keynote
2. High resolution ensures clarity on large screens
3. Can be cropped or zoomed without quality loss

### Regeneration:
If updates are needed, run the Python scripts:
```bash
python3 docs/diagram1_architecture.py
python3 docs/diagram2_pipeline.py
```

Both scripts are self-contained and use only matplotlib (no external dependencies beyond standard scientific Python stack).

## Verification Checklist

### Diagram 1 - Architecture:
- [x] Kubernetes cluster boundary shown
- [x] Three-tier chain (API→APP→DB) depicted
- [x] Envoy sidecars on all services
- [x] Prometheus with 15s scrape interval
- [x] AURA controller outside cluster
- [x] 3 agents clearly labeled
- [x] QMIX mixing network shown
- [x] Safety layer included
- [x] kubectl scaling path indicated
- [x] 30s control loop annotated
- [x] 16-dim observation noted
- [x] 10-action space noted
- [x] Guard rails (min=1, max=5) shown

### Diagram 2 - Pipeline:
- [x] Simulator with 8 parallel envs
- [x] 25s pod startup model (breakdown shown)
- [x] QMIX training (200 epochs × 1000 steps)
- [x] Policy checkpoint/trained model
- [x] Validation on local cluster
- [x] Production deployment with safety
- [x] Feedback/retraining loop
- [x] 30s production control loop
- [x] Reward weights (α=2.0, β=2.5, γ=1.5)
- [x] Performance outcomes (55% cost, 77% latency)
- [x] Timeline visualization

## Notes

- Both diagrams avoid deep training hyperparameter details (as requested)
- No excessive model internals shown
- All claims are supported by repository data
- Diagrams complement each other: one shows runtime architecture, the other shows development lifecycle
- Color coding is consistent across both diagrams for related concepts

## File Locations

```
docs/
├── AURA_Architecture_Diagram.png    # Diagram 1 (PNG)
├── AURA_Architecture_Diagram.pdf    # Diagram 1 (PDF)
├── AURA_Pipeline_Diagram.png        # Diagram 2 (PNG)
├── AURA_Pipeline_Diagram.pdf        # Diagram 2 (PDF)
├── diagram1_architecture.py         # Generator script 1
├── diagram2_pipeline.py             # Generator script 2
└── DIAGRAM_SPECIFICATIONS.md        # This file
```

---

**Created**: 2026-04-06  
**Verified Against**: AURA repository commit as of 2026-04-06  
**Quality**: Production-ready, 300 DPI, presentation-grade