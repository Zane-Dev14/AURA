#!/bin/bash
set -e

kubectl apply -f infra/manifests/three-tier/envoy-config-api.yaml
kubectl apply -f infra/manifests/three-tier/envoy-config-app.yaml
kubectl apply -f infra/manifests/three-tier/envoy-config-db.yaml
kubectl apply -f infra/manifests/three-tier/mysql-init-script.yaml
kubectl apply -f infra/manifests/three-tier/db.yaml
kubectl apply -f infra/manifests/three-tier/api.yaml
kubectl apply -f infra/manifests/three-tier/app.yaml
kubectl apply -f microservices/locust/locust.yaml
