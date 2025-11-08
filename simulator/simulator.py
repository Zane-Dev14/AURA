import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque
import yaml
import json


# POD LIFECYCLE - Exact K8s Behavior

class PodState(Enum):
    """Mirrors real K8s pod lifecycle"""
    PENDING = 0           # Scheduling, image pull
    CONTAINER_CREATING = 1
    RUNNING = 2           # App starting, not ready
    READY = 3             # Health checks pass, serving traffic
    TERMINATING = 4       # Graceful shutdown


@dataclass
class Pod:
    """Single pod instance"""
    state: PodState
    age: float = 0.0
    warmup_capacity: float = 0.0  # 0.0 → 1.0 during warmup
    
    def update(self, dt: float, startup_times: dict) -> None:
        """Update pod state machine"""
        self.age += dt
        
        if self.state == PodState.PENDING:
            if self.age >= startup_times['pending']:
                self.state = PodState.CONTAINER_CREATING
        
        elif self.state == PodState.CONTAINER_CREATING:
            if self.age >= startup_times['container']:
                self.state = PodState.RUNNING
                self.warmup_capacity = 0.2  # Start at 20% capacity
        
        elif self.state == PodState.RUNNING:
            # Gradual warmup
            warmup_progress = (self.age - startup_times['container']) / startup_times['warmup']
            self.warmup_capacity = min(1.0, 0.2 + 0.8 * warmup_progress)
            
            if self.age >= startup_times['ready']:
                self.state = PodState.READY
                self.warmup_capacity = 1.0
    
    @property
    def is_ready(self) -> bool:
        return self.state == PodState.READY
    
    @property
    def is_serving(self) -> bool:
        """Can serve traffic (RUNNING or READY)"""
        return self.state in [PodState.RUNNING, PodState.READY]


# SERVICE - Single Microservice Model

class Service:
    """
    Models a single microservice with:
    - Pod lifecycle management
    - Queueing dynamics (M/M/c)
    - Prometheus-style metrics
    - Dependency-aware load
    """
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        
        # Service characteristics
        self.base_latency_ms = config['base_latency_ms']
        self.capacity_per_pod = config['capacity_rps']
        self.startup_times = config['startup_times']
        
        # Pod management
        self.pods: List[Pod] = [self._create_ready_pod()]
        self.replicas_desired = 1
        
        # Queueing state
        self.queue = 0.0
        self.max_queue = config.get('max_queue_size', 1000)
        self.incoming_rps = 0.0
        
        # Metrics (current)
        self.cpu_util = 0.0
        self.memory_util = 0.3
        self.p50_latency_ms = self.base_latency_ms
        self.p95_latency_ms = self.base_latency_ms
        self.p99_latency_ms = self.base_latency_ms
        self.error_rate = 0.0
        
        # Counters
        self.requests_served = 0
        self.requests_dropped = 0
        
        # History for temporal features
        self.cpu_history = deque([0.0] * 4, maxlen=4)  # Last 2 minutes
        self.rps_history = deque([0.0] * 4, maxlen=4)
        self.latency_history = deque([self.base_latency_ms] * 4, maxlen=4)
        
        # Dependencies
        self.downstream_services: List[str] = []
        self.upstream_services: List[str] = []
        
        # Last action
        self.last_action = 1
        
    def _create_ready_pod(self) -> Pod:
        """Create a pod that's already ready (for initialization)"""
        return Pod(PodState.READY, age=100.0, warmup_capacity=1.0)
    
    def scale(self, desired: int) -> int:
        """
        Scale to desired replicas (like kubectl scale)
        Returns: flapping count (for penalty)
        """
        desired = max(1, min(desired, 20))  # Clamp to [1, 20]
        old_desired = self.replicas_desired
        self.replicas_desired = desired
        
        # Scale up: create new pods
        if desired > len(self.pods):
            for _ in range(desired - len(self.pods)):
                self.pods.append(Pod(PodState.PENDING))
        
        # Scale down: remove pods (newest first, PENDING first)
        elif desired < len(self.pods):
            # Sort: PENDING first, then by age descending
            self.pods.sort(key=lambda p: (p.state.value, -p.age))
            self.pods = self.pods[:desired]
        
        self.last_action = desired
        return abs(desired - old_desired)  # Flapping metric
    
    def tick(self, dt: float, incoming_rps: float, randomization: dict) -> None:
        """
        Advance simulation by dt seconds
        
        Args:
            dt: Time step (typically 1.0 second)
            incoming_rps: Request rate from load generator/upstream
            randomization: Domain randomization params
        """
        self.incoming_rps = incoming_rps
        
        # Update all pods
        for pod in self.pods:
            pod.update(dt, self.startup_times)
        
        # Calculate capacity
        total_capacity = 0.0
        for pod in self.pods:
            if pod.is_serving:
                total_capacity += pod.warmup_capacity * self.capacity_per_pod
        
        # Apply capacity variance (domain randomization)
        total_capacity *= randomization['capacity_variance']
        
        # === QUEUEING DYNAMICS ===
        # Requests arrive
        self.queue += incoming_rps * dt
        
        # Process queue up to capacity
        can_serve = min(self.queue, total_capacity * dt)
        self.queue -= can_serve
        self.requests_served += can_serve
        
        # Drop overflow
        if self.queue > self.max_queue:
            dropped = self.queue - self.max_queue
            self.requests_dropped += dropped
            self.queue = self.max_queue
            self.error_rate = dropped / max(incoming_rps * dt, 1e-6)
        else:
            self.error_rate = 0.0
        
        # Compute metrics
        self._compute_metrics(total_capacity, randomization)
        
        # Update history
        self.cpu_history.append(self.cpu_util)
        self.rps_history.append(incoming_rps)
        self.latency_history.append(self.p95_latency_ms)
    
    def _compute_metrics(self, capacity: float, rand: dict) -> None:
        """Compute CPU, latency using queueing theory"""
        
        if capacity <= 0:
            self.cpu_util = 0.0
            self.memory_util = 0.3
            self.p95_latency_ms = 9999.0
            return
        
        # CPU utilization
        utilization = self.incoming_rps / capacity if capacity > 0 else 0
        self.cpu_util = min(utilization, 2.0)  # Can spike above 100%
        
        # Memory (grows with queue)
        self.memory_util = 0.3 + 0.5 * min(self.queue / 100.0, 1.0)
        
        # === LATENCY CALCULATION ===
        if self.incoming_rps > 0 and utilization < 0.99:
            base = self.base_latency_ms
            
            # Queue delay using Little's Law: W = L / λ
            if self.queue > 1:
                avg_queue_time_s = self.queue / self.incoming_rps
                queue_delay_ms = avg_queue_time_s * 1000
            else:
                queue_delay_ms = 0
            
            # Tail latency multiplier (nonlinear near capacity)
            if utilization > 0.7:
                tail_multiplier = 1 + (utilization - 0.7) * 8  # Exponential growth
            else:
                tail_multiplier = 1.0
            
            # Percentiles
            self.p50_latency_ms = base + queue_delay_ms * 0.3
            self.p95_latency_ms = base + queue_delay_ms * tail_multiplier
            self.p99_latency_ms = base + queue_delay_ms * tail_multiplier * 2.5
            
            # Add network jitter
            jitter = rand['network_jitter_ms']
            self.p95_latency_ms += jitter
            self.p99_latency_ms += jitter * 2
        else:
            # Overloaded or no traffic
            if utilization >= 0.99:
                self.p95_latency_ms = 9999.0
                self.p99_latency_ms = 9999.0
            else:
                self.p95_latency_ms = self.base_latency_ms
                self.p99_latency_ms = self.base_latency_ms
    
    @property
    def ready_replicas(self) -> int:
        """Count pods in READY state"""
        return sum(1 for p in self.pods if p.is_ready)
    
    @property
    def running_replicas(self) -> int:
        """Count pods that can serve traffic"""
        return sum(1 for p in self.pods if p.is_serving)
    
    def get_observation(self, all_services: Dict[str, 'Service']) -> np.ndarray:
        """
        Return 16-dim observation vector for RL agent
        
        This is what makes agents smarter than HPA!
        """
        # Downstream pressure
        downstream_queue = sum(
            all_services[dep].queue
            for dep in self.downstream_services
            if dep in all_services
        )
        
        # Upstream latency
        upstream_latency = 0.0
        if self.upstream_services:
            upstream_latencies = [
                all_services[up].p95_latency_ms
                for up in self.upstream_services
                if up in all_services
            ]
            upstream_latency = np.mean(upstream_latencies) if upstream_latencies else 0.0
        
        # Traffic derivative
        traffic_derivative = 0.0
        if len(self.rps_history) >= 2:
            traffic_derivative = self.rps_history[-1] - self.rps_history[-2]
        
        # CPU derivative
        cpu_derivative = 0.0
        if len(self.cpu_history) >= 2:
            cpu_derivative = self.cpu_history[-1] - self.cpu_history[-2]
        
        # Build observation vector
        obs = np.array([
            # [0-1] Resource utilization
            min(self.cpu_util / 2.0, 1.0),
            min(self.memory_util / 2.0, 1.0),
            
            # [2-4] Latency metrics
            min(self.p50_latency_ms / 100.0, 2.0),
            min(self.p95_latency_ms / 500.0, 2.0),
            min(self.p99_latency_ms / 1000.0, 2.0),
            
            # [5-6] Throughput
            min(self.incoming_rps / 500.0, 2.0),
            min(self.error_rate, 1.0),
            
            # [7-8] Queue state (HPA can't see this!)
            min(self.queue / 100.0, 2.0),
            np.tanh(traffic_derivative / 50.0),  # Normalized
            
            # [9-11] Pod state
            self.replicas_desired / 20.0,
            self.ready_replicas / 20.0,
            self.ready_replicas / max(self.replicas_desired, 1),
            
            # [12-13] Temporal features
            min(self.cpu_history[-2] / 2.0, 1.0) if len(self.cpu_history) >= 2 else 0.0,
            np.tanh(cpu_derivative / 0.5),
            
            # [14-15] Cross-service coordination signals
            min(downstream_queue / 500.0, 1.0),
            min(upstream_latency / 200.0, 2.0),
        ], dtype=np.float32)
        
        return obs


# K8S CLUSTER SIMULATOR

class K8sSimulator:
    """
    Digital twin of your K3d cluster
    
    Configurable for 3-tier (api, app, db) or 11-service (Online Boutique)
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Load configuration and initialize services"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        
        # Create services
        self.services: Dict[str, Service] = {}
        for svc_config in config['services']:
            name = svc_config['name']
            self.services[name] = Service(name, svc_config)
        
        # Set dependencies
        dependency_graph = config.get('dependencies', {})
        for service_name, deps in dependency_graph.items():
            if service_name in self.services:
                self.services[service_name].downstream_services = deps
                for dep in deps:
                    if dep in self.services:
                        self.services[dep].upstream_services.append(service_name)
        
        self.agent_ids = list(self.services.keys())
        
        # Simulation params
        sim_config = config['simulation']
        self.decision_interval = sim_config['decision_interval_s']
        self.max_steps = sim_config['max_steps']
        
        # Load pattern
        load_config = config['load_pattern']
        self.base_rps = load_config['base_rps']
        self.amplitude = load_config['amplitude']
        self.period = load_config['period_s']
        self.load_type = load_config.get('type', 'sine')
        
        # Reward weights
        reward_config = config['reward']
        self.alpha = reward_config['cost_weight']
        self.beta = reward_config['sla_weight']
        self.gamma = reward_config['flapping_weight']
        self.sla_threshold_ms = reward_config['sla_threshold_ms']
        
        # State
        self.time = 0.0
        self.step_count = 0
        self.randomization = {}
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset simulator to initial state
        
        Includes domain randomization for robustness
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.time = 0.0
        self.step_count = 0
        
        # Domain randomization (makes agents robust)
        self.randomization = {
            'capacity_variance': np.random.uniform(0.85, 1.15),
            'network_jitter_ms': np.random.uniform(0, 15),
            'load_multiplier': np.random.uniform(0.8, 1.2),
        }
        
        # Reset all services
        for svc in self.services.values():
            svc.scale(1)  # Start with 1 replica
            svc.queue = 0
            svc.requests_served = 0
            svc.requests_dropped = 0
            svc.cpu_history.clear()
            svc.rps_history.clear()
            svc.latency_history.clear()
        
        return {
            name: svc.get_observation(self.services)
            for name, svc in self.services.items()
        }
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, float, bool, dict]:
        """
        Execute one decision interval
        
        Args:
            actions: {service_name: desired_replicas}
        
        Returns:
            observations, reward, done, info
        """
        # Apply scaling actions
        total_flapping = 0
        for name, desired in actions.items():
            if name in self.services:
                flapping = self.services[name].scale(desired)
                total_flapping += flapping
        
        # Simulate in 1-second substeps
        substeps = int(self.decision_interval)
        for _ in range(substeps):
            # Generate load
            current_rps = self._generate_load(self.time)
            
            # Propagate load through services
            service_loads = self._compute_service_loads(current_rps)
            
            # Tick all services
            for name, svc in self.services.items():
                svc.tick(1.0, service_loads[name], self.randomization)
            
            self.time += 1.0
        
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward(total_flapping)
        
        # Collect observations
        observations = {
            name: svc.get_observation(self.services)
            for name, svc in self.services.items()
        }
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        # Build info dict
        info = {
            'time': self.time,
            'step': self.step_count,
            'total_cost': sum(len(svc.pods) for svc in self.services.values()) * self.decision_interval * 0.001,
            'sla_violations': sum(
                1 for svc in self.services.values()
                if svc.p95_latency_ms > self.sla_threshold_ms
            ),
            'metrics': {
                name: {
                    'p95_ms': svc.p95_latency_ms,
                    'cpu_util': svc.cpu_util,
                    'memory_util': svc.memory_util,
                    'ready_replicas': svc.ready_replicas,
                    'desired_replicas': svc.replicas_desired,
                    'queue': svc.queue,
                    'error_rate': svc.error_rate,
                    'requests_served': svc.requests_served,
                }
                for name, svc in self.services.items()
            }
        }
        
        return observations, reward, done, info
    
    def _generate_load(self, t: float) -> float:
        """Generate realistic load pattern"""
        if self.load_type == 'sine':
            # Sine wave with noise
            phase = 2 * np.pi * t / self.period
            sine_factor = 0.5 + 0.5 * np.sin(phase)  # 0-1
            
            traffic = self.base_rps + self.amplitude * sine_factor
            traffic *= self.randomization['load_multiplier']
            traffic *= np.random.uniform(0.95, 1.05)  # ±5% noise
            
            # Random spikes (0.1% chance per second)
            if np.random.random() < 0.001:
                traffic += np.random.uniform(50, 200)
            
            return max(0, traffic)
        
        elif self.load_type == 'constant':
            return self.base_rps * self.randomization['load_multiplier']
        
        else:
            return self.base_rps
    
    def _compute_service_loads(self, frontend_rps: float) -> Dict[str, float]:
        """
        Propagate load through dependency graph
        
        For 3-tier: api → app → db
        For Online Boutique: complex dependencies
        """
        loads = {name: 0.0 for name in self.services.keys()}
        
        # Find entry point (service with no upstream)
        entry_services = [
            name for name, svc in self.services.items()
            if not svc.upstream_services
        ]
        
        if not entry_services:
            # Fallback: assume first service is entry
            entry_services = [self.agent_ids[0]]
        
        # Apply load to entry services
        for entry in entry_services:
            loads[entry] = frontend_rps
        
        # Propagate through dependencies
        # Simple model: each downstream call happens once per request
        for name, svc in self.services.items():
            for dep in svc.downstream_services:
                if dep in loads:
                    loads[dep] += loads[name]
        
        return loads
    
    def _compute_reward(self, flapping: float) -> float:
        """
        Reward function: -(cost + SLA_violations + flapping + over-scaling)
        Agents learn to minimize this
        """
        # Cost: replica-seconds
        total_replicas = sum(len(svc.pods) for svc in self.services.values())
        cost = total_replicas * self.decision_interval * 0.001  # $0.001 per replica per step

        # SLA violations: how much over threshold
        sla_violations = 0.0
        for svc in self.services.values():
            if svc.p95_latency_ms > self.sla_threshold_ms:
                violation = (svc.p95_latency_ms - self.sla_threshold_ms) / self.sla_threshold_ms
                sla_violations += violation

        # Over-scaling penalty: penalize if replicas > 6 (mid-range)
        over_scaling_penalty = 0.0
        for svc in self.services.values():
            if svc.replicas_desired > 6:  # mid-range threshold
                over_scaling_penalty += (svc.replicas_desired - 6) * 0.7

        reward = -(
            self.alpha * cost +
            self.beta * sla_violations +
            self.gamma * flapping +
            0.3 * over_scaling_penalty   # NEW penalty
        )

        return reward



# QUICK TEST

if __name__ == '__main__':
    import os
    
    # Check if config exists
    if not os.path.exists('config.yaml'):
        print("Error: config.yaml not found!")
        print("Please create config.yaml first (see artifact for template)")
        exit(1)
    
    print("=" * 70)
    print("K8s SIMULATOR - QUICK TEST")
    print("=" * 70)
    
    sim = K8sSimulator('config.yaml')
    
    print(f"\nServices: {sim.agent_ids}")
    print(f"Observation dim: {len(sim.services[sim.agent_ids[0]].get_observation(sim.services))}")
    print(f"Decision interval: {sim.decision_interval}s")
    print(f"Max steps: {sim.max_steps}")
    
    # Reset
    obs = sim.reset(seed=42)
    print(f"\nInitial observations: {len(obs)} agents")
    
    # Run 3 steps
    print("\n=== Running 3 steps ===")
    for i in range(3):
        actions = {name: 3 for name in sim.agent_ids}
        obs, reward, done, info = sim.step(actions)
        
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Cost: ${info['total_cost']:.4f}")
        print(f"  SLA violations: {info['sla_violations']}")
        
        for name in sim.agent_ids[:2]:  # Show first 2 services
            m = info['metrics'][name]
            print(f"  {name}: P95={m['p95_ms']:.0f}ms, "
                  f"Ready={m['ready_replicas']}/{m['desired_replicas']}, "
                  f"Queue={m['queue']:.1f}")
    
    print("\n✅ Simulator working!")