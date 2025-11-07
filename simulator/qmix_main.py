#!/usr/bin/env python3
"""
QMIX Trainer (fixed): CTDE with real global state, replay stores state/next_state,
mixing network uses state, warm-start from single-agent checkpoint.

Usage:
    python qmix_train.py
"""

import os
import time
import math
import random
from collections import deque, namedtuple
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Replace these imports with your actual env module path
from boutique_env import K8sAutoscaleEnv  # PettingZoo ParallelEnv wrapper you provided
from pettingzoo import ParallelEnv

# ----------------------------
# Hyperparameters (tweakable)
# ----------------------------
ENV_CONFIG = "config.yaml"
SEED = 42

N_AGENTS = None  # auto-detected below
OBS_DIM_PER_AGENT = 16
ACTION_DIM = 10   # 0..9 -> replicas 1..10

# QMIX training params
LR = 5e-4
BATCH_SIZE = 64
GAMMA = 0.98
REPLAY_SIZE = 200_000
MIN_REPLAY_SIZE = 5_000
EPOCHS = 400
STEPS_PER_EPOCH = 2000    # environment steps PER epoch (collect)
EPS_PER_STEP = 0.01       # tiny epsilon per step for eps-greedy increment if desired
EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY_EPOCHS = 200

TARGET_UPDATE_FREQ = 200  # in gradient steps
MIXING_HIDDEN = 32        # hidden units in mixing net
HYPERNET_HIDDEN = 64

GPU_COOLDOWN_FREQ = 30    # epochs; will sleep GPU_COOLDOWN_SECONDS
GPU_COOLDOWN_SECONDS = 60

PRETRAINED_PATH = "agent.pth"
SAVE_DIR = "./qmix_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Determinism
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Utilities & Replay Buffer
# ----------------------------
Transition = namedtuple("Transition", [
    "obs",         # np array shape (n_agents, obs_dim)
    "state",       # np array shape (state_dim,)
    "actions",     # np array shape (n_agents,)
    "reward",      # float (shared reward)
    "next_obs",    # np array shape (n_agents, obs_dim)
    "next_state",  # np array shape (state_dim,)
    "done"         # bool
])

class JointReplayBuffer:
    def _init_(self, capacity):
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def _len_(self):
        return len(self.buf)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        # convert to numpy arrays
        obs = np.stack([t.obs for t in batch])             # (B, n_agents, obs_dim)
        state = np.stack([t.state for t in batch])         # (B, state_dim)
        actions = np.stack([t.actions for t in batch])     # (B, n_agents)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)  # (B,)
        next_obs = np.stack([t.next_obs for t in batch])
        next_state = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return dict(obs=obs, state=state, actions=actions,
                    rewards=rewards, next_obs=next_obs, next_state=next_state, dones=dones)

# ----------------------------
# Environment wrapper fixes
# ----------------------------
class K8sAutoscaleEnvFixed(K8sAutoscaleEnv):
    """
    Extends your PettingZoo ParallelEnv wrapper to expose a global state vector
    suitable for QMIX. Replace field names inside get_global_state() if your
    simulator stores differently.
    """

    def get_global_state(self):
        """
        Build a deterministic global state vector from all services.
        Example per-service features: cpu_util, memory_util, p95_latency_ms,
        queue, replicas_desired, ready_replicas, incoming_rps.
        Adjust to match your K8sSimulator internal API.
        """
        state = []
        # assume self.simulator.services is an OrderedDict or dict of service objects
        for name, svc in self.simulator.services.items():
            # Replace these attribute names if your simulator uses different naming
            cpu = getattr(svc, "cpu_util", 0.0)
            mem = getattr(svc, "memory_util", 0.0)
            latency = getattr(svc, "p95_latency_ms", 0.0)
            queue = getattr(svc, "queue", 0.0)
            replicas_desired = getattr(svc, "replicas_desired", 0.0)
            ready_replicas = getattr(svc, "ready_replicas", 0.0)
            incoming = getattr(svc, "incoming_rps", 0.0)
            state.extend([cpu, mem, latency, queue, replicas_desired, ready_replicas, incoming])
        return np.array(state, dtype=np.float32)

    # override reset to return obs, infos, state for convenience (we'll call env.reset() and then get_global_state())
    def reset_with_state(self, seed=None, options=None):
        obs, infos = self.reset(seed=seed, options=options)
        state = self.get_global_state()
        return obs, infos, state

    # similarly for stepping: step(action_dict) returns (obs, rewards, term, trunc, infos, state)
    def step_with_state(self, actions):
        obs, rewards, terminateds, truncateds, infos = self.step(actions)
        state = self.get_global_state()
        return obs, rewards, terminateds, truncateds, infos, state

# ----------------------------
# Neural nets
# ----------------------------
class AgentNet(nn.Module):
    """Per-agent Q-network (256-256-128 + GELU)"""
    def _init_(self, obs_dim, action_dim):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs):  # obs: (B, obs_dim) -> q: (B, action_dim)
        return self.net(obs)

class MixingNetwork(nn.Module):
    """
    QMIX mixing net implemented with hypernetworks.
    Produces q_tot scalar from agent_qs (B, n_agents) and state (B, state_dim).
    """
    def _init_(self, n_agents, state_dim, mixing_hidden=MIXING_HIDDEN, hyper_hidden=HYPERNET_HIDDEN):
        super()._init_()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden = mixing_hidden

        # hypernet W1: state -> (n_agents * mixing_hidden)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden)
        )
        # hypernet b1: state -> mixing_hidden
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden)
        )
        # hypernet W2: state -> (mixing_hidden * 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden)
        )
        # hypernet b2: state -> 1
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1)
        )

        self.elu = nn.ELU()

    def forward(self, agent_qs, state):
        """
        agent_qs: tensor (B, n_agents)
        state: tensor (B, state_dim)
        returns: q_tot (B, 1)
        """
        B = agent_qs.size(0)
        # compute W1
        w1 = self.hyper_w1(state)  # (B, n_agents * mixing_hidden)
        w1 = w1.view(B, self.n_agents, self.mixing_hidden)  # (B, n_agents, H)
        b1 = self.hyper_b1(state).view(B, 1, self.mixing_hidden)  # (B,1,H)

        # (B, n_agents) x (B, n_agents, H) -> (B, H)
        agent_qs = agent_qs.view(B, 1, self.n_agents)  # (B,1,n_agents)
        hidden = torch.bmm(agent_qs, w1).squeeze(1) + b1.squeeze(1)  # (B, H)
        hidden = self.elu(hidden)

        # W2: produce (B, H) -> (B, 1)
        w2 = self.hyper_w2(state).view(B, self.mixing_hidden, 1)  # (B,H,1)
        b2 = self.hyper_b2(state).view(B, 1)                      # (B,1)
        q_tot = torch.bmm(hidden.view(B,1,self.mixing_hidden), w2).squeeze(2) + b2  # (B,1)
        return q_tot  # shape (B,1)

# ----------------------------
# Trainer
# ----------------------------
def build_env_and_dims():
    env = K8sAutoscaleEnvFixed(ENV_CONFIG)  # uses your provided K8sAutoscaleEnv subclass
    possible_agents = list(env.possible_agents)
    n_agents = len(possible_agents)
    # compute state dim using get_global_state()
    state = env.get_global_state()
    state_dim = int(state.size)
    return env, possible_agents, n_agents, state_dim

def sanitize_obs(obs_array):
    # obs_array: array of per-agent observations (n_agents, obs_dim)
    # clip/replace nan/inf
    obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1e6, neginf=-1e6)
    obs_array = np.clip(obs_array, -1e6, 1e6)
    return obs_array.astype(np.float32)

def main():
    # build env & dims
    env, agent_ids, n_agents, state_dim = build_env_and_dims()
    print(f"Detected agents: {agent_ids}, n_agents={n_agents}, state_dim={state_dim}")

    # set global constants based on env
    global OBS_DIM_PER_AGENT, N_AGENTS
    N_AGENTS = n_agents
    # assume all agents share same obs shape:
    OBS_DIM_PER_AGENT = int(env.observation_space(agent_ids[0]).shape[0])

    # models
    agent_nets = [AgentNet(OBS_DIM_PER_AGENT, ACTION_DIM).to(device) for _ in range(n_agents)]
    target_agent_nets = [AgentNet(OBS_DIM_PER_AGENT, ACTION_DIM).to(device) for _ in range(n_agents)]
    for tnet, net in zip(target_agent_nets, agent_nets):
        tnet.load_state_dict(net.state_dict())

    mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    # optimizer for all params
    params = []
    for net in agent_nets:
        params += list(net.parameters())
    params += list(mixer.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    # optionally warm-start each agent net from single-agent checkpoint if matches
    if os.path.exists(PRETRAINED_PATH):
        try:
            ck = torch.load(PRETRAINED_PATH, map_location=device)
            # attempt to load into each agent net (architectures must match)
            for i, net in enumerate(agent_nets):
                try:
                    net.load_state_dict(ck, strict=True)
                    target_agent_nets[i].load_state_dict(net.state_dict())
                    print(f"Loaded pretrained weights into agent {i}")
                except Exception as e:
                    print(f"Could not load pretrained into agent {i}: {e}")
        except Exception as e:
            print(f"Failed to load pretrained {PRETRAINED_PATH}: {e}")

    # replay buffer
    buffer = JointReplayBuffer(REPLAY_SIZE)

    # epsilon schedule
    def epsilon_for_epoch(ep):
        t = min(ep / max(1, EPS_DECAY_EPOCHS), 1.0)
        return EPS_START + (EPS_END - EPS_START) * t

    # ----------------------------------------------------------
    # Collection routine: runs episodes until we collect 'steps' env steps
    # ----------------------------------------------------------
    def collect_steps(num_steps, epsilon):
        steps_collected = 0
        while steps_collected < num_steps:
            # reset env + get initial state
            obs_dict, infos, state = env.reset_with_state()
            # obs_dict: dict agent->obs
            obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)  # (n_agents, obs_dim)
            done = False
            while not done and steps_collected < num_steps:
                # select joint actions with epsilon-greedy
                actions = {}
                action_list = []
                for i, aid in enumerate(agent_ids):
                    obs_i = torch.tensor(obs_arr[i:i+1], dtype=torch.float32, device=device)  # (1,obs_dim)
                    with torch.no_grad():
                        qvals = agent_nets[i](obs_i)  # (1, action_dim)
                    if random.random() < epsilon:
                        act = random.randrange(ACTION_DIM)
                    else:
                        act = int(qvals.argmax(dim=1).item())
                    actions[aid] = act
                    action_list.append(act)
                # step env
                next_obs_dict, rewards, terminateds, truncateds, infos, next_state = env.step_with_state(actions)
                # sanitized arrays
                next_obs_arr = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                # shared reward - pick first if dict provides single reward scalar or average across agents
                if isinstance(rewards, dict):
                    # if rewards are per-agent but your env uses shared reward, you can average or pick first
                    try:
                        r = float(list(rewards.values())[0])
                    except Exception:
                        r = float(0.0)
                else:
                    r = float(rewards)
                # sanitize reward
                r = float(np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6))
                r = float(np.clip(r, -1e6, 1e6))

                done_flag = any(terminateds.values()) if isinstance(terminateds, dict) else bool(terminateds)
                # store transition: obs_arr (n_agents, obs_dim), state, actions, reward, next_obs_arr, next_state, done
                buffer.push(sanitize_obs(obs_arr), state.astype(np.float32),
                            np.array(action_list, dtype=np.int64),
                            r, sanitize_obs(next_obs_arr), next_state.astype(np.float32), done_flag)
                steps_collected += 1
                obs_arr = next_obs_arr
                state = next_state
                if done_flag:
                    break
        return

    # ----------------------------------------------------------
    # Training sampling & update step
    # ----------------------------------------------------------
    gradient_steps = 0
    best_test = -1e18

    for epoch in range(1, EPOCHS + 1):
        eps = epsilon_for_epoch(epoch)
        # collect env interactions
        collect_steps(STEPS_PER_EPOCH, eps)

        # Do several SGD updates (updates proportional to collected steps)
        # We'll perform (STEPS_PER_EPOCH * UPDATE_PER_STEP) gradient steps
        n_grad_steps = max(1, int(STEPS_PER_EPOCH * 0.1))  # fixed fraction; adjust as needed
        for gs in range(n_grad_steps):
            if len(buffer) < MIN_REPLAY_SIZE:
                break
            batch = buffer.sample(BATCH_SIZE)
            # convert to tensors
            obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)          # (B, n_agents, obs_dim)
            state_b = torch.tensor(batch["state"], dtype=torch.float32, device=device)      # (B, state_dim)
            actions_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)     # (B, n_agents)
            rewards_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)  # (B,)
            next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
            next_state_b = torch.tensor(batch["next_state"], dtype=torch.float32, device=device)
            dones_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

            B = obs_b.size(0)

            # Compute per-agent Q(s,a) for taken actions
            agent_qs_taken = []
            for i in range(n_agents):
                obs_i = obs_b[:, i, :]             # (B, obs_dim)
                q_i = agent_nets[i](obs_i)         # (B, action_dim)
                a_i = actions_b[:, i].unsqueeze(1) # (B,1)
                q_taken = q_i.gather(1, a_i).squeeze(1)  # (B,)
                agent_qs_taken.append(q_taken)
            # stack agent q values -> (B, n_agents)
            agent_qs_taken = torch.stack(agent_qs_taken, dim=1)

            # compute target: max_a' Q_target(next_obs, a')
            with torch.no_grad():
                agent_qs_next = []
                for i in range(n_agents):
                    next_obs_i = next_obs_b[:, i, :]
                    q_next = target_agent_nets[i](next_obs_i)  # (B, action_dim)
                    # double-dqn style: use target nets' greedy next-value (we're not using online for selection to keep implementation simple)
                    max_q_next, _ = q_next.max(dim=1)          # (B,)
                    agent_qs_next.append(max_q_next)
                agent_qs_next = torch.stack(agent_qs_next, dim=1)  # (B, n_agents)
                # mix next qs with target mixer using next_state
                q_tot_next = target_mixer(agent_qs_next, next_state_b).squeeze(1)  # (B,)
                # TD target
                td_target = rewards_b + (1.0 - dones_b) * (GAMMA * q_tot_next)

            # current total Q
            q_tot = mixer(agent_qs_taken, state_b).squeeze(1)  # (B,)

            loss = F.mse_loss(q_tot, td_target)

            optimizer.zero_grad()
            loss.backward()
            # optional gradient clipping
            if False:
                torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            gradient_steps += 1
            # soft/hard target update
            if gradient_steps % TARGET_UPDATE_FREQ == 0:
                # hard sync
                for tnet, net in zip(target_agent_nets, agent_nets):
                    tnet.load_state_dict(net.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())

        # ----------------------------------------------------
        # Evaluate policy periodically (greedy)
        # ----------------------------------------------------
        if epoch % 5 == 0 or epoch == EPOCHS:
            # run N eval episodes
            eval_episodes = 10
            rewards_list = []
            for _ in range(eval_episodes):
                obs_dict, infos, state = env.reset_with_state()
                obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                done = False
                total_r = 0.0
                while not done:
                    actions = {}
                    for i, aid in enumerate(agent_ids):
                        obs_i = torch.tensor(obs_arr[i:i+1], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            qvals = agent_nets[i](obs_i)
                            act = int(qvals.argmax(dim=1).item())
                        actions[aid] = act
                    next_obs_dict, rewards, terminateds, truncateds, infos, next_state = env.step_with_state(actions)
                    # accumulate shared reward
                    if isinstance(rewards, dict):
                        r = float(list(rewards.values())[0])
                    else:
                        r = float(rewards)
                    total_r += r
                    obs_arr = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                    done = any(terminateds.values()) if isinstance(terminateds, dict) else bool(terminateds)
                rewards_list.append(total_r)
            mean_eval = float(np.mean(rewards_list))
            std_eval = float(np.std(rewards_list))
            print(f"[Epoch {epoch:04d}] Eval reward: {mean_eval:.2f} ± {std_eval:.2f}  buffer_len={len(buffer)} loss={loss.item():.4f}")

            # checkpoint best
            if mean_eval > best_test:
                best_test = mean_eval
                torch.save({
                    "agent_nets": [net.state_dict() for net in agent_nets],
                    "mixer": mixer.state_dict()
                }, os.path.join(SAVE_DIR, f"qmix_best_epoch{epoch}.pth"))
                print("Saved best checkpoint.")

        # GPU cooldown (throttle) as requested
        if epoch % GPU_COOLDOWN_FREQ == 0 and epoch > 0:
            print(f"GPU cooldown: sleeping for {GPU_COOLDOWN_SECONDS}s to throttle GPU...")
            time.sleep(GPU_COOLDOWN_SECONDS)

    # final save
    torch.save({
        "agent_nets": [net.state_dict() for net in agent_nets],
        "mixer": mixer.state_dict()
    }, os.path.join(SAVE_DIR, "qmix_final.pth"))
    print("Training finished. Final model saved.")

if __name__ == "_main_":
    main()