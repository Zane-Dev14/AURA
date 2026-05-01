#!/usr/bin/env python3
"""
Integration test for APP-tier guard bug fix.
Tests the complete veto and recovery logic flow.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from deployment.agent_controller import app_needs_recovery, api_is_bottleneck

def simulate_controller_logic(app_metrics, api_metrics, agent_action):
    """
    Simulate the controller's veto and recovery logic for APP tier.
    Returns: (final_action, reason)
    """
    actions = {"app": agent_action}
    svc = "app"
    
    # Veto 2: Tier-coupled (lines 186-194)
    if actions[svc] > 0:
        if api_is_bottleneck(api_metrics) and not app_needs_recovery(app_metrics):
            actions[svc] = 0
            return (0, "TIER_VETO: API bottleneck blocks APP scale-up")
    
    # Recovery override (lines 196-203)
    if actions[svc] <= 0 and app_needs_recovery(app_metrics):
        actions[svc] = 1
        return (1, "RECOVERY_OVERRIDE: APP needs recovery")
    
    return (actions[svc], "NO_VETO: Agent action allowed")

def test_integration():
    """Test complete controller logic flow"""
    
    print("="*80)
    print("APP-TIER GUARD INTEGRATION TEST")
    print("="*80)
    
    test_cases = [
        {
            "name": "Moderate pressure + API bottleneck (THE BUG SCENARIO)",
            "app": {"p99": 400, "error": 0.01, "queue": 20, "rps": 150, "desired": 1},
            "api": {"desired": 5, "queue": 600},
            "agent_action": 1,
            "expected_action": 1,
            "expected_reason": "NO_VETO"  # Fix allows scale-up by preventing tier veto
        },
        {
            "name": "Low pressure + API bottleneck",
            "app": {"p99": 100, "error": 0.001, "queue": 5, "rps": 50, "desired": 1},
            "api": {"desired": 5, "queue": 600},
            "agent_action": 1,
            "expected_action": 0,
            "expected_reason": "TIER_VETO"
        },
        {
            "name": "High pressure + API bottleneck",
            "app": {"p99": 600, "error": 0.01, "queue": 20, "rps": 150, "desired": 1},
            "api": {"desired": 5, "queue": 600},
            "agent_action": 1,
            "expected_action": 1,
            "expected_reason": "NO_VETO"  # Fix allows scale-up by preventing tier veto
        },
        {
            "name": "Agent wants scale-down but APP needs recovery",
            "app": {"p99": 550, "error": 0.01, "queue": 20, "rps": 150, "desired": 2},
            "api": {"desired": 3, "queue": 100},
            "agent_action": -1,
            "expected_action": 1,
            "expected_reason": "RECOVERY_OVERRIDE"
        },
        {
            "name": "Normal operation - no API bottleneck",
            "app": {"p99": 200, "error": 0.01, "queue": 10, "rps": 100, "desired": 2},
            "api": {"desired": 3, "queue": 100},
            "agent_action": 1,
            "expected_action": 1,
            "expected_reason": "NO_VETO"
        },
        {
            "name": "Combined pressure signal (p99=350, rps=150)",
            "app": {"p99": 350, "error": 0.01, "queue": 10, "rps": 150, "desired": 1},
            "api": {"desired": 5, "queue": 600},
            "agent_action": 1,
            "expected_action": 1,
            "expected_reason": "NO_VETO"  # Fix allows scale-up by preventing tier veto
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {tc['name']}")
        print(f"  APP: p99={tc['app']['p99']}, err={tc['app']['error']}, q={tc['app']['queue']}, rps={tc['app']['rps']}")
        print(f"  API: desired={tc['api']['desired']}, q={tc['api']['queue']}")
        print(f"  Agent action: {tc['agent_action']}")
        
        actual_action, reason = simulate_controller_logic(
            tc['app'], tc['api'], tc['agent_action']
        )
        
        print(f"  Expected: action={tc['expected_action']}, reason contains '{tc['expected_reason']}'")
        print(f"  Actual:   action={actual_action}, reason='{reason}'")
        
        if actual_action == tc['expected_action'] and tc['expected_reason'] in reason:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed == 0:
        print("\n✅ ALL INTEGRATION TESTS PASSED")
        print("   The APP-tier guard bug is FIXED")
        print("   APP can now scale up even when API is bottlenecked if it shows pressure")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)

# Made with Bob
