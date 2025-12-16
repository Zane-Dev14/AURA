import os
import torch
import torch.nn as nn
import numpy as np


# -------------------------------------------------
# QMIX ACTOR NETWORK (MATCHES TRAINING EXACTLY)
# -------------------------------------------------
class MLPQNetwork(nn.Module):
    def __init__(self, obs_dim=16, act_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)  # MUST be 10
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# AURA INFERENCE WRAPPER
# -------------------------------------------------
class AuraInference:
    """
    Loads trained QMIX actor networks and performs inference.

    Action space (trained):
        0..9

    Action semantics (example mapping):
        0..3  -> scale down
        4..5  -> no-op
        6..9  -> scale up
    """

    def __init__(self, checkpoint_dir: str):
        self.device = torch.device("cpu")

        self.api_net = MLPQNetwork()
        self.app_net = MLPQNetwork()
        self.db_net  = MLPQNetwork()

        self._load(self.api_net, os.path.join(checkpoint_dir, "api_actor_best.pth"))
        self._load(self.app_net, os.path.join(checkpoint_dir, "app_actor_best.pth"))
        self._load(self.db_net,  os.path.join(checkpoint_dir, "db_actor_best.pth"))

        self.api_net.eval()
        self.app_net.eval()
        self.db_net.eval()

        # -------------------------------------------------
        # ACTION INDEX → REPLICA DELTA MAPPING
        # ⚠️ MUST MATCH TRAINING LOGIC
        # -------------------------------------------------
        self.action_map = {
            0: -2,
            1: -2,
            2: -1,
            3: -1,
            4:  0,
            5:  0,
            6: +1,
            7: +1,
            8: +2,
            9: +2,
        }

    def _load(self, model, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state, strict=True)

    def _predict_single(self, model, obs: np.ndarray) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        qvals = model(obs_t)
        act_idx = int(torch.argmax(qvals, dim=1).item())
        return self.action_map[act_idx]

    def predict(self, obs_dict: dict) -> dict:
        """
        Returns replica deltas:
            -2, -1, 0, +1, +2
        """
        return {
            "api": self._predict_single(self.api_net, obs_dict["api"]),
            "app": self._predict_single(self.app_net, obs_dict["app"]),
            "db":  self._predict_single(self.db_net,  obs_dict["db"]),
        }
