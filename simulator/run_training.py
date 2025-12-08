"""
Training launcher for AURA.
Stays in simulator/ but imports from marl/.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import training function
from marl.trainer import train_qmix

if __name__ == "__main__":
    # Config is in same directory as this script
    config_path = Path(__file__).parent / "config.yaml"
    
    print("="*60)
    print("AURA QMIX Training")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Project root: {PROJECT_ROOT}")
    print("="*60)
    
    # Run training
    train_qmix(config_path=str(config_path))