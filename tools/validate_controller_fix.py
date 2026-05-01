#!/usr/bin/env python3
"""Sanity checks for APP recovery and bottleneck guard logic.

Run:
    .venv/bin/python tools/validate_controller_fix.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from deployment.agent_controller import app_needs_recovery, api_is_bottleneck


def main():
    # APP should recover when it is already unhealthy.
    assert app_needs_recovery({"p99": 780.0, "error": 0.2095, "queue": 5.0}) is True

    # Healthy APP should not trigger recovery override.
    assert app_needs_recovery({"p99": 40.0, "error": 0.0, "queue": 1.0}) is False

    # API bottleneck detection should only trigger at saturation + high queue.
    assert api_is_bottleneck({"desired": 5, "queue": 800}) is True
    assert api_is_bottleneck({"desired": 4, "queue": 800}) is False

    print("controller_guard_checks=ok")


if __name__ == "__main__":
    main()
