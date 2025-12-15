"""
AURA Multi-Agent Reinforcement Learning module.
"""

__version__ = '0.1.0'

from .env import BoutiqueEnv
from .policies import ActorNetwork, QMixer, QMIXAgent

__all__ = ['BoutiqueEnv', 'ActorNetwork', 'QMixer', 'QMIXAgent']