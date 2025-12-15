#!/usr/bin/env python3
"""
FAST Stable Single-Agent DQN Training for K8sAutoscaleEnv
- Same architecture & logic
- Same reward logic
- Same policy update logic
- Only optimized for SPEED (≈ 15 minutes)
"""
import argparse
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import gymnasium
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer

from boutique_env import K8sAutoscaleEnv

# =============================================================================
# Configuration (SPEED OPTIMIZED)
# =============================================================================
ENV_CONFIG = "config.yaml"
OBS_DIM = 16
ACTION_DIM = 10

# ---- SPEED TUNED ----
LR = 5e-4
BATCH_SIZE = 256              # was 1024
GAMMA = 0.98
REPLAY_BUFFER_SIZE = 50_000   # was 100k
EPOCHS = 120                  # was 400
STEP_PER_EPOCH = 1000         # was 2000
STEP_PER_COLLECT = 100        # was 200
UPDATE_PER_STEP = 2.0         # was 1.0

EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY_EPOCHS = EPOCHS    # match faster training

TARGET_UPDATE_FREQ = 200
ESTIMATION_STEP = 1

SEED = 42
SAVE_DIR = "./trained_agents"
PRETRAINED_PATH = "agent.pth"
LOG_INTERVAL = 10             # less eval frequency
LR_DECAY = True

TEST_EPISODES = 3             # was 10
TRAIN_ENVS = 8                # was 4
TEST_ENVS = 4                 # was 2

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
def print_scaling_info(obs, rewards, infos):
    """
    Pretty print:
    - replicas per service
    - reward contributions
    """
    replicas = {k: infos[k].get("replicas_desired", None) for k in infos.keys()}
    latencies = {k: infos[k].get("p95_latency_ms", None) for k in infos.keys()}
    print("Replicas:", replicas)
    print("P95 Latency:", latencies)
    print("Reward:", rewards)

# CSV Logging
log_path = os.path.join(SAVE_DIR, "train_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "eps", "loss", "train_mean", "test_mean", "test_std"])

# =============================================================================
# 1. Single-Agent Wrapper
# =============================================================================
class SingleAgentWrapper(gymnasium.Env):
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
        return np.array(obs_dict[self.agent_id], dtype=np.float32), {}

    def step(self, action):
        actions = {ag: 2 for ag in self.agents}
        # Clip actions to 0-5 (1-6 replicas) and penalize max replica action in reward
        clipped_action = max(0, min(int(action), 5))
        actions[self.agent_id] = clipped_action


        obs_dict, rewards, terms, truns, infos = self.env.step(actions)
        done = bool(terms.get(self.agent_id, False) or truns.get(self.agent_id, False))

        raw_reward = rewards.get(self.agent_id, 0.0)
        reward = float(np.nan_to_num(raw_reward, nan=0.0, posinf=1e6, neginf=-1e6))
        reward = float(np.clip(reward, -1e6, 1e6))

        if done:
            obs = np.zeros_like(obs_dict[self.agent_id])
        else:
            obs = obs_dict[self.agent_id]

        return np.array(obs, dtype=np.float32), reward, done, False, infos.get(self.agent_id, {})

    def close(self):
        self.env.close()

# =============================================================================
# 2. Q-Network
# =============================================================================
class QNetwork(nn.Module):
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
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        elif isinstance(obs, (list, tuple)):
            obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device)
        elif torch.is_tensor(obs) and obs.device != device:
            obs = obs.to(device)

        obs = torch.clamp(obs, -20.0, 20.0)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return self.net(obs), state

# =============================================================================
# 3. Main Training
# =============================================================================
if __name__ == "__main__":
    # Device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True)
    args = parser.parse_args()

    temp_env = K8sAutoscaleEnv(ENV_CONFIG)
    agent_names = list(temp_env.possible_agents)

    TRAIN_AGENT = args.agent
    print(f"Training agent: {TRAIN_AGENT}")

    def make_env():
        return SingleAgentWrapper(ENV_CONFIG, TRAIN_AGENT)

    # ---- MUCH FASTER ----
    train_envs = SubprocVectorEnv([make_env for _ in range(TRAIN_ENVS)])
    test_envs = SubprocVectorEnv([make_env for _ in range(TEST_ENVS)])

    # Model
    q_net = QNetwork(OBS_DIM, ACTION_DIM).to(device)
    # try:
    #     q_net = torch.compile(q_net)
    # except:
    #     pass

    optim = torch.optim.Adam(q_net.parameters(), lr=LR)

    scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1.0, end_factor=0.2, total_iters=EPOCHS
        )
        if LR_DECAY else None
    )

    # Optional pretrained load
    if os.path.exists(PRETRAINED_PATH):
        try:
            ck = torch.load(PRETRAINED_PATH, map_location=device)
            q_net.load_state_dict(ck, strict=True)
            print(f"Loaded pretrained weights.")
        except Exception as e:
            print(f"Failed to load pretrained: {e}")

    policy = DQNPolicy(
        model=q_net,
        optim=optim,
        action_space=gymnasium.spaces.Discrete(ACTION_DIM),
        discount_factor=GAMMA,
        estimation_step=ESTIMATION_STEP,
        target_update_freq=TARGET_UPDATE_FREQ,
    ).to(device)

    # Target
    policy._target_q = QNetwork(OBS_DIM, ACTION_DIM).to(device)
    policy.sync_weight()

    # Replay buffer
    buffer = VectorReplayBuffer(REPLAY_BUFFER_SIZE, buffer_num=train_envs.env_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    policy.set_eps(EPSILON_START)

    # ---- Warmup (much smaller) ----
    warmup_steps = BATCH_SIZE * 10
    print(f"Warming up with {warmup_steps} steps...")
    train_collector.collect(n_step=warmup_steps, reset_before_collect=True)

    # Training loop
    best_reward = -float("inf")

    for epoch in range(1, EPOCHS + 1):
        last_best_info = None

        eps = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * epoch / EPSILON_DECAY_EPOCHS)
        policy.set_eps(eps)

        collect_result = train_collector.collect(n_step=STEP_PER_COLLECT)
        n_updates = max(1, int(STEP_PER_COLLECT * UPDATE_PER_STEP))

        policy.is_within_training_step = True
        losses = []
        for _ in range(n_updates):
            out = policy.update(BATCH_SIZE, train_collector.buffer)
            if isinstance(out, dict) and "loss" in out:
                losses.append(out["loss"])
        policy.is_within_training_step = False

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # ---- Periodic eval ----
        if epoch % LOG_INTERVAL == 0 or epoch == EPOCHS:
            policy.set_eps(0.0)
            try:
                test_collector.reset_env()
            except:
                pass
            eval_result = test_collector.collect(n_episode=TEST_EPISODES)
            policy.set_eps(eps)

            mean_reward, std_reward = 0.0, 0.0
            if hasattr(eval_result, "returns"):
                ret = eval_result.returns
                if isinstance(ret, dict):
                    arr = np.asarray(list(ret.values())[0])
                else:
                    arr = np.asarray(ret)
                mean_reward = float(np.mean(arr)) if arr.size else 0.0
                std_reward = float(np.std(arr)) if arr.size else 0.0

            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(q_net.state_dict(), os.path.join(SAVE_DIR, f"{TRAIN_AGENT}_best.pth"))
                print(f"✓ New best reward {mean_reward:.2f}")

                # -----------------------
                # SHOW SCALING INFO
                # -----------------------
                policy.set_eps(0.0)  # greedy
                obs_episode, _ = test_envs.reset()
                done_episode = [False] * test_envs.env_num
                step_count = 0
                episode_infos = []

                while not all(done_episode) and step_count < 20:
                    # Get actions from policy
                    logits, _ = policy.model(obs_episode)
                    actions = logits.argmax(dim=1).cpu().numpy()  # greedy actions

                    # Step env
                    obs_episode, rewards_step, terminateds, truncateds, infos_step = test_envs.step(actions)

                    episode_infos.append((actions.copy(), rewards_step.copy(), infos_step.copy()))
                    done_episode = [t or d for t, d in zip(done_episode, terminateds)]
                    step_count += 1

                # Show comparison with previous best
                if last_best_info:
                    print("--- Previous best snapshot ---")
                    prev_actions = last_best_info["actions"]
                    prev_rewards = last_best_info["rewards"]
                    prev_infos = last_best_info["infos"]
                    for i, agent_id in enumerate([TRAIN_AGENT]):
                        print(f"{agent_id} | action: {prev_actions[i]} | reward: {prev_rewards[i]} | info: {prev_infos[i] if isinstance(prev_infos, list) else prev_infos}")

                # Show new best snapshot
                last_actions, last_rewards, last_infos = episode_infos[-1]
                print(f"--- New best snapshot ---")
                for i, agent_id in enumerate([TRAIN_AGENT]):
                    print(f"{agent_id} | action: {last_actions[i]} | reward: {last_rewards[i]} | info: {last_infos[i] if isinstance(last_infos, list) else last_infos}")

                # Update last_best_info
                last_best_info = {"actions": last_actions, "rewards": last_rewards, "infos": last_infos}


            # log
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, eps, avg_loss, 0, mean_reward, std_reward])

            print(f"Epoch {epoch}/{EPOCHS} | eps {eps:.3f} | loss {avg_loss:.4f} | testR {mean_reward:.2f}")

        if scheduler:
            scheduler.step()

    torch.save(q_net.state_dict(), os.path.join(SAVE_DIR, f"{TRAIN_AGENT}_final.pth"))
    print("Training complete.")

