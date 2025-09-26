#!/usr/bin/env bash
# =======================================================
# Project: AURA - MARL Autoscaler
# Description: Bootstrap script to create initial folder
#              structure and placeholder files.
# Usage:
#   chmod +x tools/setup_project.sh
#   ./tools/setup_project.sh
# =======================================================

ROOT="/mnt/Shared/NerdyShi/Projects/Final Project/AURA"

echo "🚀 Setting up project in $ROOT ..."
mkdir -p $ROOT

# Core files
touch $ROOT/docker-compose.yml
mkdir -p $ROOT/docs
echo "# AURA Project Documentation" > $ROOT/docs/README.md

# Infrastructure
mkdir -p $ROOT/infra/{manifests/three-tier,helm}
touch $ROOT/infra/k3d-setup.md
echo "# K3d Setup Instructions" > $ROOT/infra/k3d-setup.md

# Microservices (3-tier demo app + load testing)
mkdir -p $ROOT/microservices/{three-tier,locust}
echo "# 3-tier demo app source here" > $ROOT/microservices/three-tier/README.md
echo "# Locust load tests here" > $ROOT/microservices/locust/README.md

# Metrics
mkdir -p $ROOT/metrics/{prometheus,grafana/dashboards}
touch $ROOT/metrics/prometheus/prometheus-values.yaml
echo "# Grafana dashboards JSON" > $ROOT/metrics/grafana/dashboards/.gitkeep

# Simulator
mkdir -p $ROOT/simulator
touch $ROOT/simulator/{Dockerfile,simulator.py,api_spec.md}
echo "# Simulator API spec" > $ROOT/simulator/api_spec.md

# MARL
mkdir -p $ROOT/marl/{env,trainer,policies}
echo "# PettingZoo environment" > $ROOT/marl/env/README.md
echo "# QMIX training loop" > $ROOT/marl/trainer/README.md
echo "# Policy checkpoints" > $ROOT/marl/policies/.gitkeep

# Deployment
mkdir -p $ROOT/deployment
touch $ROOT/deployment/agent-controller.py

# Experiments
mkdir -p $ROOT/experiments/{scripts,results,plots}
echo "# Experiment scripts" > $ROOT/experiments/scripts/README.md

# Tools
mkdir -p $ROOT/tools
touch $ROOT/tools/{setup_k3d.sh,deploy_stack.sh,run_training.sh}

echo "✅ Project structure initialized!"
