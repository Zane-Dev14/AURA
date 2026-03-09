#!/usr/bin/env bash
# Helper: set LOCUST_USERS / LOCUST_SPAWN on the running Locust deployment
# Usage: tools/set_locust_users.sh <users> [spawn_rate] [namespace]

set -euo pipefail

USERS=${1:-}
SPAWN=${2:-}
NS=${3:-default}

if [ -z "$USERS" ]; then
  echo "Usage: $0 <users> [spawn_rate] [namespace]"
  exit 1
fi

echo "Patching deployment/locust in namespace '$NS' -> LOCUST_USERS=$USERS LOCUST_SPAWN=$SPAWN"
kubectl set env deployment/locust LOCUST_USERS=${USERS} -n ${NS}

if [ -n "$SPAWN" ]; then
  kubectl set env deployment/locust LOCUST_SPAWN=${SPAWN} -n ${NS}
fi

echo "Triggering rollout restart to apply env changes..."
kubectl rollout restart deployment/locust -n ${NS}
kubectl rollout status deployment/locust -n ${NS}

echo "Done. New LOCUST_USERS=${USERS} LOCUST_SPAWN=${SPAWN}"
