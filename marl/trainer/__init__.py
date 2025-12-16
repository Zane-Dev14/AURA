"""
Training and evaluation utilities.
"""

from .train_qmix import train_qmix
from .eval import evaluate_policy

__all__ = ['train_qmix', 'evaluate_policy']