#!/usr/bin/env bash

set -euo pipefail

CLUSTER_NAME="aura"
SERVERS=1
AGENTS=3
SERVER_MEMORY="8192m"
AGENT_MEMORY="6144m"
RECREATE=false

usage() {
	cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --cluster-name NAME     k3d cluster name (default: aura)
  --servers N             Number of server nodes (default: 1)
  --agents N              Number of agent nodes (default: 3)
  --server-memory SIZE    Server node memory, e.g. 8192m (default: 8192m)
  --agent-memory SIZE     Agent node memory, e.g. 6144m (default: 6144m)
  --recreate              Delete existing cluster with same name first
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--cluster-name)
			CLUSTER_NAME="$2"
			shift 2
			;;
		--servers)
			SERVERS="$2"
			shift 2
			;;
		--agents)
			AGENTS="$2"
			shift 2
			;;
		--server-memory)
			SERVER_MEMORY="$2"
			shift 2
			;;
		--agent-memory)
			AGENT_MEMORY="$2"
			shift 2
			;;
		--recreate)
			RECREATE=true
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "[ERROR] Unknown argument: $1"
			usage
			exit 1
			;;
	esac
done

if k3d cluster list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$CLUSTER_NAME"; then
	if [[ "$RECREATE" == "true" ]]; then
		echo "[INFO] Deleting existing cluster: $CLUSTER_NAME"
		k3d cluster delete "$CLUSTER_NAME"
	else
		echo "[INFO] Cluster '$CLUSTER_NAME' already exists."
		echo "[INFO] Use --recreate to recreate it with updated sizing."
		kubectl config use-context "k3d-$CLUSTER_NAME" >/dev/null 2>&1 || true
		kubectl get nodes
		exit 0
	fi
fi

echo "[INFO] Creating k3d cluster '$CLUSTER_NAME' (servers=$SERVERS, agents=$AGENTS)"
k3d cluster create "$CLUSTER_NAME" \
	--servers "$SERVERS" \
	--agents "$AGENTS" \
	--servers-memory "$SERVER_MEMORY" \
	--agents-memory "$AGENT_MEMORY" \
	--k3s-arg "--disable=traefik@server:0" \
	-p "30090:30090@loadbalancer" \
	-p "32322:32322@loadbalancer" \
	-p "30089:30089@loadbalancer" \
	--wait

kubectl config use-context "k3d-$CLUSTER_NAME"
echo "[OK] Active context: $(kubectl config current-context)"
kubectl get nodes
