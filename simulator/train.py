
    #!/usr/bin/env python3
"""
Stable Single-Agent DQN Training for K8sAutoscaleEnv
- Matches pretrained 256-256-128 + GELU architecture
- Loads warm-start weights if available
- Contains safety fixes: reward sanitization, obs clamping, longer warmup
- Uses Tianshou DQNPolicy + DummyVectorEnv
"""
import argparse #I added this so we can use train the servcie via CLI. example: python train.py --agent=api , or python train.py --agent=app
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import gymnasium
from tianshou.env import DummyVectorEnv,SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer, Batch

# Import your environment
from boutique_env import K8sAutoscaleEnv

# =============================================================================
# Configuration (Optimized)
# =============================================================================
ENV_CONFIG = "config.yaml"
OBS_DIM = 16
ACTION_DIM = 10

# Core learning params
LR = 5e-4
BATCH_SIZE = 1024
GAMMA = 0.98
REPLAY_BUFFER_SIZE = 100_000
EPOCHS = 400
STEP_PER_EPOCH = 2000
STEP_PER_COLLECT = 200
UPDATE_PER_STEP = 1.0

# Exploration
EPSILON_START = 0.1
EPSILON_END = 0.01
EPSILON_DECAY_EPOCHS = 200

# DQN specifics
TARGET_UPDATE_FREQ = 200
ESTIMATION_STEP = 1

# Regularization & logging
SEED = 42
SAVE_DIR = "./trained_agents"
PRETRAINED_PATH = "agent.pth"
LOG_INTERVAL = 5
LR_DECAY = True
GRAD_CLIP = None  # set a float (e.g. 5.0) to enable clipping if you change update flow

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set up CSV logging
log_path = os.path.join(SAVE_DIR, "train_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eps", "loss", "train_mean", "test_mean", "test_std"])

# =============================================================================
# 1. Single-Agent Wrapper
# =============================================================================
class SingleAgentWrapper(gymnasium.Env):
    """
    Wraps PettingZoo ParallelEnv to train one agent at a time.
    Sanitizes reward to avoid NaN/Inf and returns (obs, reward, done, truncated, info)
    """
    def __init__(self, config_path, agent_id):
        super().__init__()
        self.env = K8sAutoscaleEnv(config_path)
        self.agent_id = agent_id
        self.agents = list(self.env.possible_agents)
        self.action_space = gymnasium.spaces.Discrete(ACTION_DIM)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed, options=options)
        obs = obs_dict[self.agent_id]
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # Build actions for all agents (default dummy=2 -> 3 replicas)
        actions = {agent: 2 for agent in self.agents} # <-- FIX
        
        actions[self.agent_id] = int(action)
        obs_dict, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # Extract done/reward for this agent safely
        done = bool(terminateds.get(self.agent_id, False) or truncateds.get(self.agent_id, False))
        raw_reward = rewards.get(self.agent_id, 0.0)

        # === SANITIZE reward: remove nan/inf and clamp ===
        reward = float(np.nan_to_num(raw_reward, nan=0.0, posinf=1e6, neginf=-1e6))
        reward = float(np.clip(reward, -1e6, 1e6))

        obs = np.zeros_like(obs_dict[self.agent_id]) if done else obs_dict[self.agent_id]
        return np.array(obs, dtype=np.float32), float(reward), done, False, infos.get(self.agent_id, {})

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()

    def close(self):
        self.env.close()

# =============================================================================
# 2. Q-Network (Matches pretrained architecture)
# =============================================================================
class QNetwork(nn.Module):
    """MLP that matches 256-256-128 + GELU architecture used in your pretrained model"""
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs, state=None, info=None):
        device = next(self.net.parameters()).device
        # convert numpy -> tensor and send to device
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        elif isinstance(obs, (list, tuple)):
            obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device)
        elif torch.is_tensor(obs) and obs.device != device:
            obs = obs.to(device)

        # clamp observations to prevent extreme activations
        # (keeps numerical stability if simulator returns spikes)
        obs = torch.clamp(obs, -20.0, 20.0)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        q_values = self.net(obs)
        return q_values, state

if __name__=="__main__":
        # =============================================================================
    # 3. Environment and Policy setup
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    parser = argparse.ArgumentParser(description="Train a single DQN agent.")
    parser.add_argument("--agent", type=str, required=True, help="Name of the agent to train (e.g., 'api', 'app', 'db')")
    args = parser.parse_args()

    temp_env = K8sAutoscaleEnv(ENV_CONFIG)
    agent_names = list(temp_env.possible_agents)
    print(f"Agents detected: {agent_names}")

    if args.agent not in agent_names:
        print(f"Error: Agent '{args.agent}' not found in possible agents: {agent_names}")
        exit(1)

    TRAIN_AGENT = args.agent
    print(f"Training agent: {TRAIN_AGENT}\n")

    def make_env():
        return SingleAgentWrapper(ENV_CONFIG, TRAIN_AGENT)

    # Use 4 train envs and 2 test envs (same as before)
    train_envs = SubprocVectorEnv([make_env for _ in range(4)], context='spawn')
    test_envs = SubprocVectorEnv([make_env for _ in range(2)], context='spawn')

    # =============================================================================
    # 4. Initialize network, optimizer, policy
    # =============================================================================
    q_net = QNetwork(OBS_DIM, ACTION_DIM).to(device)
    optim = torch.optim.Adam(q_net.parameters(), lr=LR)

    # LR scheduler (optional)
    if LR_DECAY:
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.2, total_iters=EPOCHS)
    else:
        scheduler = None

    if os.path.exists(PRETRAINED_PATH):
        try:
            pretrained_dict = torch.load(PRETRAINED_PATH, map_location=device)
            q_net.load_state_dict(pretrained_dict, strict=True)
            print(f"✓ Loaded pretrained weights from '{PRETRAINED_PATH}'")
        except Exception as e:
            print(f"✗ Could not load pretrained weights ({e}) — starting fresh.")
    else:
        print(f"⚠ No pretrained file '{PRETRAINED_PATH}' found, training from scratch.")

    policy = DQNPolicy(
        model=q_net,
        optim=optim,
        action_space=gymnasium.spaces.Discrete(ACTION_DIM),
        discount_factor=GAMMA,
        estimation_step=ESTIMATION_STEP,
        target_update_freq=TARGET_UPDATE_FREQ,
    ).to(device)

    # Double DQN target initialization
    policy._target_q = QNetwork(OBS_DIM, ACTION_DIM).to(device)
    policy.sync_weight()

    # =============================================================================
    # 5. Replay buffer + collectors
    # =============================================================================
    buffer = VectorReplayBuffer(
        total_size=REPLAY_BUFFER_SIZE,
        buffer_num=train_envs.env_num,  # DummyVectorEnv property
    )

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    policy.set_eps(EPSILON_START)

    # =============================================================================
    # 6. Training loop (with warmup and safer reward extraction)
    # =============================================================================
    best_reward = -float("inf")
    print("\nStarting training loop...\n")

    # --- longer warmup collection so buffer isn't tiny before first update ---
    warmup_steps = BATCH_SIZE * 50
    print(f"Warmup collecting {warmup_steps} steps into replay buffer...")
    train_collector.collect(n_step=warmup_steps, reset_before_collect=True)

    for epoch in range(1, EPOCHS + 1):
        # decay eps linearly over EPSILON_DECAY_EPOCHS
        eps = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * epoch / EPSILON_DECAY_EPOCHS)
        policy.set_eps(eps)

        collect_result = train_collector.collect(n_step=STEP_PER_COLLECT)
        n_updates = max(1, int(STEP_PER_COLLECT * UPDATE_PER_STEP))
        losses = []

        # Manually set within training step flag so update() works
        policy.is_within_training_step = True
        for _ in range(n_updates):
            # policy.update performs the backward + optimizer.step internally
            loss_batch = policy.update(BATCH_SIZE, train_collector.buffer)
            if isinstance(loss_batch, dict) and "loss" in loss_batch:
                losses.append(loss_batch["loss"])
            elif hasattr(loss_batch, "loss"):
                losses.append(loss_batch.loss)
        policy.is_within_training_step = False

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # periodic evaluation & logging
        if epoch % LOG_INTERVAL == 0 or epoch == EPOCHS:
            policy.set_eps(0.0)  # greedy for evaluation
            # ensure test collector reset
            try:
                test_collector.reset_env()
            except Exception:
                pass
            test_result = test_collector.collect(n_episode=10)
            policy.set_eps(eps)

            # === robust extraction of reward statistics ===
            mean_reward, std_reward = 0.0, 0.0
            try:
                # Tianshou >=0.5 returns an object with 'returns' attribute
                if hasattr(test_result, "returns"):
                    returns = test_result.returns
                    # If returns is a dict mapping metric -> array
                    if isinstance(returns, dict):
                        # find a sensible reward key
                        if "n/ep_reward" in returns:
                            arr = np.asarray(returns["n/ep_reward"])
                        else:
                            # fallback to first item
                            arr = np.asarray(list(returns.values())[0])
                    else:
                        arr = np.asarray(returns)
                    if arr.size:
                        mean_reward = float(np.mean(arr))
                        std_reward = float(np.std(arr))
                    else:
                        mean_reward, std_reward = 0.0, 0.0
                else:
                    # older interface: try dict-like access
                    if isinstance(test_result, dict) and "rews" in test_result:
                        arr = np.asarray(test_result["rews"])
                        mean_reward = float(np.mean(arr)) if arr.size else 0.0
                        std_reward = float(np.std(arr)) if arr.size else 0.0
            except Exception:
                mean_reward, std_reward = 0.0, 0.0

            print(f"Test reward: {mean_reward:.2f} ± {std_reward:.2f}")

            # checkpoint best
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(q_net.state_dict(), os.path.join(SAVE_DIR, f"{TRAIN_AGENT}_best.pth"))
                print(f"✓ Saved new best policy for {TRAIN_AGENT}: reward {mean_reward:.2f}")

            # training rollout stats (safe)
            train_reward = 0.0
            try:
                if hasattr(collect_result, "returns"):
                    cr = collect_result.returns
                    if isinstance(cr, dict):
                        if "n/ep_reward" in cr:
                            arr = np.asarray(cr["n/ep_reward"])
                        else:
                            arr = np.asarray(list(cr.values())[0])
                    else:
                        arr = np.asarray(cr)
                    train_reward = float(np.mean(arr)) if arr.size else 0.0
                elif isinstance(collect_result, dict) and "rews" in collect_result:
                    arr = np.asarray(collect_result["rews"])
                    train_reward = float(np.mean(arr)) if arr.size else 0.0
            except Exception:
                train_reward = 0.0

            print(f"Epoch {epoch:03d} | Eps: {eps:.3f} | Loss: {avg_loss:.4f} | "
                f"TrainR: {train_reward:.2f} | TestR: {mean_reward:.2f} | Best: {best_reward:.2f}")

            # append to CSV
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, eps, avg_loss, train_reward, mean_reward, std_reward])

        # optional LR scheduler step
        if scheduler is not None:
            scheduler.step()

    # final save
    torch.save(q_net.state_dict(), os.path.join(SAVE_DIR, f"{TRAIN_AGENT}_final.pth"))
    print(f"\nTraining complete for {TRAIN_AGENT}! Final policy saved in {SAVE_DIR}/\n")
