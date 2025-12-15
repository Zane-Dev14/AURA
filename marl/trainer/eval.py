"""
Evaluation script for trained QMIX agents.
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl.env import BoutiqueEnv
from marl.policies import QMIXAgent


def evaluate_policy(config_path='simulator/config.yaml', 
                    model_dir='simulator/qmix_trained',
                    num_episodes=10):
    """
    Evaluate trained QMIX policy.
    
    Args:
        config_path: Path to config.yaml
        model_dir: Directory containing trained models
        num_episodes: Number of episodes to evaluate
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment
    env = BoutiqueEnv(config)
    
    # Load trained agent
    agent = QMIXAgent(
        num_agents=3,
        obs_dim=config['training']['obs_dim'],
        action_dim=config['training']['action_dim'],
        global_state_dim=config['training']['global_state_dim']
    )
    
    # Load model weights
    model_dir = Path(model_dir)
    agent.actors[0].load_state_dict(torch.load(model_dir / 'api_actor_best.pth'))
    agent.actors[1].load_state_dict(torch.load(model_dir / 'app_actor_best.pth'))
    agent.actors[2].load_state_dict(torch.load(model_dir / 'db_actor_best.pth'))
    agent.mixer.load_state_dict(torch.load(model_dir / 'mixing_best.pth'))
    
    print(f"Loaded models from {model_dir}")
    
    # Rest of evaluation code...


if __name__ == "__main__":
    evaluate_policy()