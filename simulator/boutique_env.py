# boutique_env.py
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from simulator import K8sSimulator


class K8sAutoscaleEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for K8s simulator.
    Works with centralized training (QMIX / MADDPG etc.)
    """

    metadata = {'name': 'k8s_autoscale_v0', 'is_parallelizable': True}

    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__()
        self.simulator = K8sSimulator(config_path)

        self.possible_agents = self.simulator.agent_ids
        self.agents = self.possible_agents[:]

        # Observation space: 16-dim normalized vector
        self._observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=2.0,
                shape=(16,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Action space: 0-9 → 1-10 replicas
        self._action_spaces = {
            agent: spaces.Discrete(10)
            for agent in self.possible_agents
        }

    def get_global_state(self, observations=None):
        """
        Concatenate all agent observations into one global state vector.
        """
        if observations is None:
            observations = {
                agent: np.zeros(16, dtype=np.float32)
                for agent in self.possible_agents
            }
        return np.concatenate(
            [observations[agent] for agent in self.possible_agents], axis=0
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        self.agents = self.possible_agents[:]
        observations = self.simulator.reset(seed)
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """
        Step environment with actions.
        Args:
            actions: dict {agent_name: action_index}
        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        # Convert 0–9 actions to 1–10 replicas
        replica_actions = {agent: actions[agent] + 1 for agent in actions}

        observations, reward, done, info = self.simulator.step(replica_actions)

        # Shared reward for all agents
        rewards = {agent: reward for agent in self.agents}
        terminateds = {agent: done for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        if done:
            self.agents = []

        return observations, rewards, terminateds, truncateds, infos

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]


class HPAPolicy:
    """
    This is used to compare HPA to our MARL agents. 
    Horizontal Pod Autoscaler baseline - mimics K8s HPA behavior.
    
    Scales based on CPU utilization with configurable thresholds.
    """
    def __init__(self, target_cpu=0.7, scale_up_threshold=0.8, scale_down_threshold=0.5):
        self.target_cpu = target_cpu
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.last_actions = {}  # Track last action per agent
        
    def get_actions(self, observations):
        """
        Compute HPA actions from observations.
        
        HPA logic (simplified K8s algorithm):
        - Scale up if CPU > scale_up_threshold
        - Scale down if CPU < scale_down_threshold
        - Otherwise maintain current replicas
        """
        actions = {}
        
        for agent, obs in observations.items():
            # obs[0] is cpu_util (normalized to 0-1 in your observations)
            cpu_util = obs[0] * 2.0  # Denormalize (was divided by 2.0)
            
            # obs[9] is replicas_desired / 20.0
            current_replicas = int(obs[9] * 20.0)
            
            if agent not in self.last_actions:
                self.last_actions[agent] = max(1, current_replicas)
            
            desired = self.last_actions[agent]
            
            # HPA decision logic
            if cpu_util > self.scale_up_threshold:
                # Scale up: add 1-2 replicas based on how much over threshold
                if cpu_util > 1.2:
                    desired = min(desired + 2, 10)  # Max 10 replicas
                else:
                    desired = min(desired + 1, 10)
            elif cpu_util < self.scale_down_threshold:
                # Scale down: remove 1 replica
                desired = max(desired - 1, 1)  # Min 1 replica
            # else: maintain current
            
            # Convert to action space (1-10 replicas -> 0-9 actions)
            actions[agent] = max(0, min(desired - 1, 9))
            self.last_actions[agent] = desired
            
        return actions