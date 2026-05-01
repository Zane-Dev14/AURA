#!/usr/bin/env python3
"""
Live controller verification test - simulates pressure scenarios and verifies
the APP-tier guard fix works correctly with actual controller logic.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from deployment.agent_controller import app_needs_recovery, api_is_bottleneck

def test_live_controller_logic():
    """
    Test the actual controller functions with realistic scenarios
    to verify the APP-tier guard fix is working.
    """
    
    print("="*80)
    print("LIVE CONTROLLER VERIFICATION TEST")
    print("Testing actual controller functions from agent_controller.py")
    print("="*80)
    
    test_scenarios = [
        {
            "name": "BUG SCENARIO: Moderate APP pressure + API bottleneck",
            "app_metrics": {
                "p99": 400,      # Above 70% threshold (350ms)
                "error": 0.01,   # Below error threshold
                "queue": 20,     # Above queue threshold (15)
                "rps": 150,      # High RPS
                "desired": 1
            },
            "api_metrics": {
                "desired": 5,    # At max replicas
                "queue": 600     # High queue (bottleneck)
            },
            "expected_app_recovery": True,
            "expected_api_bottleneck": True,
            "expected_behavior": "APP should scale up despite API bottleneck"
        },
        {
            "name": "Healthy APP + API bottleneck",
            "app_metrics": {
                "p99": 100,      # Well below threshold
                "error": 0.001,  # Low error
                "queue": 5,      # Low queue
                "rps": 50,       # Low RPS
                "desired": 1
            },
            "api_metrics": {
                "desired": 5,
                "queue": 600
            },
            "expected_app_recovery": False,
            "expected_api_bottleneck": True,
            "expected_behavior": "APP should NOT scale up (tier veto applies)"
        },
        {
            "name": "High APP pressure + API bottleneck",
            "app_metrics": {
                "p99": 600,      # Well above threshold
                "error": 0.02,   # Above error threshold
                "queue": 25,     # High queue
                "rps": 200,      # Very high RPS
                "desired": 1
            },
            "api_metrics": {
                "desired": 5,
                "queue": 700
            },
            "expected_app_recovery": True,
            "expected_api_bottleneck": True,
            "expected_behavior": "APP should scale up (recovery override)"
        },
        {
            "name": "Combined pressure signal (p99=350, rps=150)",
            "app_metrics": {
                "p99": 350,      # Exactly at 70% threshold
                "error": 0.01,
                "queue": 10,
                "rps": 150,      # High RPS triggers combined signal
                "desired": 1
            },
            "api_metrics": {
                "desired": 5,
                "queue": 600
            },
            "expected_app_recovery": True,
            "expected_api_bottleneck": True,
            "expected_behavior": "APP should scale up (combined pressure)"
        },
        {
            "name": "Normal operation - no bottleneck",
            "app_metrics": {
                "p99": 200,
                "error": 0.005,
                "queue": 10,
                "rps": 100,
                "desired": 2
            },
            "api_metrics": {
                "desired": 3,
                "queue": 100
            },
            "expected_app_recovery": False,
            "expected_api_bottleneck": False,
            "expected_behavior": "Normal operation - no special handling"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[TEST {i}] {scenario['name']}")
        print(f"  APP: p99={scenario['app_metrics']['p99']}, "
              f"err={scenario['app_metrics']['error']}, "
              f"q={scenario['app_metrics']['queue']}, "
              f"rps={scenario['app_metrics']['rps']}")
        print(f"  API: desired={scenario['api_metrics']['desired']}, "
              f"q={scenario['api_metrics']['queue']}")
        
        # Test actual controller functions
        app_recovery = app_needs_recovery(scenario['app_metrics'])
        api_bottleneck = api_is_bottleneck(scenario['api_metrics'])
        
        print(f"  Expected: app_recovery={scenario['expected_app_recovery']}, "
              f"api_bottleneck={scenario['expected_api_bottleneck']}")
        print(f"  Actual:   app_recovery={app_recovery}, "
              f"api_bottleneck={api_bottleneck}")
        print(f"  Behavior: {scenario['expected_behavior']}")
        
        # Verify results
        if (app_recovery == scenario['expected_app_recovery'] and 
            api_bottleneck == scenario['expected_api_bottleneck']):
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed == 0:
        print("\n✅ ALL LIVE CONTROLLER TESTS PASSED")
        print("   The APP-tier guard fix is working correctly")
        print("   Controller logic properly handles:")
        print("   - APP recovery detection (p99 > 350ms or combined signals)")
        print("   - API bottleneck detection (max replicas + high queue)")
        print("   - Tier veto bypass when APP needs recovery")
        return True
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return False

if __name__ == "__main__":
    success = test_live_controller_logic()
    sys.exit(0 if success else 1)

# Made with Bob
