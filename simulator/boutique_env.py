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
