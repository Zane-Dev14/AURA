import numpy as np
from simulator import K8sSimulator, Service, Pod, PodState
import yaml


def test_pod_lifecycle():
    """Test pod goes through PENDING → READY"""
    print("Testing pod lifecycle...")
    
    startup_times = {'pending': 3, 'container': 10, 'warmup': 8, 'ready': 21}
    pod = Pod(PodState.PENDING)
    
    assert pod.state == PodState.PENDING
    
    # Advance time
    pod.update(2.0, startup_times)
    assert pod.state == PodState.PENDING
    
    pod.update(5.0, startup_times)  # Now at 7s
    assert pod.state == PodState.CONTAINER_CREATING
    
    pod.update(10.0, startup_times)  # Now at 17s
    assert pod.state == PodState.RUNNING
    assert pod.warmup_capacity > 0 and pod.warmup_capacity < 1.0
    
    pod.update(10.0, startup_times)  # Now at 27s
    assert pod.state == PodState.READY
    assert pod.warmup_capacity == 1.0
    
    print("  ✓ Pod lifecycle correct")


def test_service_scaling():
    """Test service scales up/down correctly"""
    print("Testing service scaling...")
    
    config = {
        'base_latency_ms': 10,
        'capacity_rps': 100,
        'startup_times': {'pending': 3, 'container': 10, 'warmup': 8, 'ready': 21},
        'max_queue_size': 1000
    }
    svc = Service('test', config)
    
    assert len(svc.pods) == 1
    assert svc.replicas_desired == 1
    
    # Scale up
    svc.scale(5)
    assert svc.replicas_desired == 5
    assert len(svc.pods) == 5
    assert sum(1 for p in svc.pods if p.state == PodState.PENDING) == 4
    
    # Scale down
    svc.scale(2)
    assert svc.replicas_desired == 2
    assert len(svc.pods) == 2
    
    print("  ✓ Service scaling correct")


def test_queueing():
    """Test queue builds under high load"""
    print("Testing queueing dynamics...")
    
    config = {
        'base_latency_ms': 10,
        'capacity_rps': 100,
        'startup_times': {'pending': 3, 'container': 10, 'warmup': 8, 'ready': 21},
        'max_queue_size': 1000
    }
    svc = Service('test', config)
    
    # Make pod ready
    svc.pods[0].state = PodState.READY
    svc.pods[0].warmup_capacity = 1.0
    
    rand = {'capacity_variance': 1.0, 'network_jitter_ms': 0}
    
    # Low load - queue stays small
    for _ in range(5):
        svc.tick(1.0, 50, rand)  # 50 RPS < 100 capacity
    
    assert svc.queue < 10
    
    # High load - queue builds
    for _ in range(10):
        svc.tick(1.0, 200, rand)  # 200 RPS > 100 capacity
    
    assert svc.queue > 50
    assert svc.p95_latency_ms > svc.base_latency_ms * 2  # Latency degraded
    
    print("  ✓ Queueing dynamics correct")


def test_simulator_reset_step():
    """Test full simulator reset and step"""
    print("Testing simulator reset/step...")
    
    sim = K8sSimulator('config.yaml')
    
    # Reset
    obs = sim.reset(seed=42)
    assert len(obs) == len(sim.agent_ids)
    
    for agent, o in obs.items():
        assert o.shape == (16,)
        assert o.dtype == np.float32
    
    # Step
    actions = {agent: 3 for agent in sim.agent_ids}
    obs, reward, done, info = sim.step(actions)
    
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert 'metrics' in info
    assert len(info['metrics']) == len(sim.agent_ids)
    
    print("  ✓ Simulator reset/step correct")


def test_reward_calculation():
    """Test reward balances cost vs SLA"""
    print("Testing reward function...")
    
    sim = K8sSimulator('config.yaml')
    
    # High replicas = high cost, low SLA violations
    sim.reset(seed=100)
    actions_high = {agent: 9 for agent in sim.agent_ids}  # 10 replicas
    _, reward_high, _, _ = sim.step(actions_high)
    
    # Low replicas = low cost, high SLA violations
    sim.reset(seed=100)
    actions_low = {agent: 0 for agent in sim.agent_ids}  # 1 replica
    _, reward_low, _, _ = sim.step(actions_low)
    
    # Rewards should differ
    assert reward_high != reward_low
    print(f"  High replicas: {reward_high:.3f} (cost penalty)")
    print(f"  Low replicas: {reward_low:.3f} (SLA penalty)")
    print("  ✓ Reward function correct")


def test_observation_features():
    """Test observation vector has correct features"""
    print("Testing observation features...")
    
    sim = K8sSimulator('config.yaml')
    obs = sim.reset(seed=200)
    
    # Run a few steps to populate history
    for _ in range(300):
        actions = {agent: 2 for agent in sim.agent_ids}
        obs, _, _, _ = sim.step(actions)
    
    agent = sim.agent_ids[0]
    o = obs[agent]
    
    assert o.shape == (16,)
    assert o[0] >= 0 and o[0] <= 2  # CPU
    assert o[3] >= 0  # P95 latency
    assert o[7] >= 0  # Queue
    assert o[9] > 0  # Desired replicas
    assert o[11] >= 0 and o[11] <= 1  # Ready ratio
    
    print(f"  Sample observation: CPU={o[0]:.3f}, P95={o[3]:.3f}, Queue={o[7]:.3f}")
    print("  ✓ Observation features correct")


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    print()
    
    test_pod_lifecycle()
    test_service_scaling()
    test_queueing()
    test_simulator_reset_step()
    test_reward_calculation()
    test_observation_features()
    
    print()
    print("=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)