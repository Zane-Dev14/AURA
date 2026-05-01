#!/usr/bin/env bash

# Shared safety guard for all cluster-facing scripts.
# Refuses execution unless kubectl current-context targets k3d.

assert_k3d_context() {
    local context
    context="$(kubectl config current-context 2>/dev/null || true)"

    if [[ -z "$context" ]]; then
        echo "[ERROR] kubectl current-context is empty. Refusing to run."
        return 1
    fi

    if [[ "$context" != k3d-* ]]; then
        echo "[ERROR] Refusing to run on non-k3d context: $context"
        echo "[ERROR] Switch to your local k3d context first (expected prefix: k3d-)."
        return 1
    fi

    return 0
}
