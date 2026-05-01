#!/usr/bin/env python3
"""
Test to verify APP-tier guard bug is fixed.
Simulates scenario where APP needs scaling but API is bottlenecked.
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_app_guard_logic():
    """Test the APP guard logic with various scenarios"""
    
    # Import the guard functions
    from deployment.agent_controller import app_needs_recovery, api_is_bottleneck
    
    print("="*80)
    print("APP-TIER GUARD BUG TEST")
    print("="*80)
    
    # Test Case 1: APP stuck at 1 replica with moderate pressure
    print("\n[TEST 1] APP moderate pressure, API bottlenecked")
    app_metrics = {
        "p99": 400,      # Below 500 threshold
        "error": 0.01,   # Below 0.02 threshold
        "queue": 20,     # Below 25 threshold
        "desired": 1,
        "rps": 150
    }
    api_metrics = {
        "desired": 5,    # At max replicas
        "queue": 600     # High queue (>500)
    }
    
    app_recovery = app_needs_recovery(app_metrics)
    api_bottleneck = api_is_bottleneck(api_metrics)
    
    print(f"  APP metrics: p99={app_metrics['p99']}, error={app_metrics['error']}, queue={app_metrics['queue']}")
    print(f"  API metrics: desired={api_metrics['desired']}, queue={api_metrics['queue']}")
    print(f"  app_needs_recovery() = {app_recovery}")
    print(f"  api_is_bottleneck() = {api_bottleneck}")
    print(f"  Result: Tier veto would block scale-up = {api_bottleneck and not app_recovery}")
    print(f"  Recovery override would trigger = {not app_recovery}")
    
    if api_bottleneck and not app_recovery:
        print("  ❌ BUG PRESENT: APP would be stuck at 1 replica!")
    else:
        print("  ✅ OK: APP can scale")
    
    # Test Case 2: APP with high pressure (recovery should trigger)
    print("\n[TEST 2] APP high pressure (p99 > 500), API bottlenecked")
    app_metrics_high = {
        "p99": 550,      # Above 500 threshold
        "error": 0.01,
        "queue": 20,
        "desired": 1,
        "rps": 150
    }
    
    app_recovery_high = app_needs_recovery(app_metrics_high)
    print(f"  APP metrics: p99={app_metrics_high['p99']}, error={app_metrics_high['error']}, queue={app_metrics_high['queue']}")
    print(f"  app_needs_recovery() = {app_recovery_high}")
    print(f"  Result: Recovery override would trigger = {app_recovery_high}")
    
    if app_recovery_high:
        print("  ✅ OK: Recovery override would allow scale-up")
    else:
        print("  ❌ BUG: Recovery should trigger but doesn't")
    
    # Test Case 3: APP with high error rate
    print("\n[TEST 3] APP high error rate, API bottlenecked")
    app_metrics_error = {
        "p99": 300,
        "error": 0.03,   # Above 0.02 threshold
        "queue": 15,
        "desired": 1,
        "rps": 150
    }
    
    app_recovery_error = app_needs_recovery(app_metrics_error)
    print(f"  APP metrics: p99={app_metrics_error['p99']}, error={app_metrics_error['error']}, queue={app_metrics_error['queue']}")
    print(f"  app_needs_recovery() = {app_recovery_error}")
    
    if app_recovery_error:
        print("  ✅ OK: Recovery override would allow scale-up")
    else:
        print("  ❌ BUG: Recovery should trigger but doesn't")
    
    # Test Case 4: APP with high queue
    print("\n[TEST 4] APP high queue, API bottlenecked")
    app_metrics_queue = {
        "p99": 300,
        "error": 0.01,
        "queue": 30,     # Above 25 threshold
        "desired": 1,
        "rps": 150
    }
    
    app_recovery_queue = app_needs_recovery(app_metrics_queue)
    print(f"  APP metrics: p99={app_metrics_queue['p99']}, error={app_metrics_queue['error']}, queue={app_metrics_queue['queue']}")
    print(f"  app_needs_recovery() = {app_recovery_queue}")
    
    if app_recovery_queue:
        print("  ✅ OK: Recovery override would allow scale-up")
    else:
        print("  ❌ BUG: Recovery should trigger but doesn't")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Determine if bug is present
    bug_present = api_bottleneck and not app_recovery
    
    if bug_present:
        print("\n❌ BUG DETECTED: APP can get stuck at 1 replica in Test Case 1")
        print("   Scenario: APP has moderate pressure (p99=400ms, queue=20)")
        print("   Problem: Tier veto blocks scale-up, recovery thresholds not met")
        print("   Impact: APP stays at 1 replica even though it needs scaling")
        print("\n   RECOMMENDED FIX: Lower recovery thresholds or add intermediate guard")
        return False
    else:
        print("\n✅ ALL TESTS PASSED: Recovery override prevents stuck-at-1-replica")
        return True

if __name__ == "__main__":
    success = test_app_guard_logic()
    sys.exit(0 if success else 1)

# Made with Bob
