#!/usr/bin/env python3
"""
qmixv1.py - QMIX trainer, GPU-focused, stable, delta-actions +/-2, PER, n-step, mixer-first.
Replaces previous qmixv1.py (drop-in). Uses your simulator (no changes required).

Run:
    python qmixv1.py
"""
import os
import time
import math
import random
from collections import deque, namedtuple
from typing import List, Dict, Tuple
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your simulator env wrapper
from boutique_env import K8sAutoscaleEnv  # PettingZoo ParallelEnv wrapper you provided

# ----------------------------
# Hyperparameters (tweakable)
# ----------------------------
ENV_CONFIG = "config.yaml"
SEED = 42

# Delta actions: -2, -1, 0, +1, +2  -> 5 discrete outputs per agent
DELTA_ACTIONS = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
N_DELTAS = len(DELTA_ACTIONS)
MIN_REPLICAS = 1
MAX_REPLICAS = 20

# model / training
LR = 5e-4
BATCH_SIZE = 2048           # large batch to utilize GPU (tune if OOM)
GAMMA = 0.98
REPLAY_SIZE = 600_000
MIN_REPLAY_SIZE = 5_000
EPOCHS = 200
STEPS_PER_EPOCH = 1000      # env steps per epoch (per env if vectorized)
N_ENVS = 8                  # number of parallel envs (multiprocess)
N_STEP = 3                  # n-step returns

EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY_EPOCHS = 150

TARGET_UPDATE_FREQ = 200
MIXING_HIDDEN = 32
HYPERNET_HIDDEN = 64

MIXER_ONLY_EPOCHS = 20

# prioritized replay
PRIO_ALPHA = 0.6
PRIO_BETA_START = 0.4
PRIO_BETA_FRAMES = EPOCHS * (STEPS_PER_EPOCH // 10 + 1)

# reward scaling/clipping
REWARD_CLIP = 10.0         # clip per-step reward magnitude after normalization
REWARD_SCALE = 10.0        # divide raw reward by this (tune as needed)

# optimization & stability
GRAD_CLIP = 5.0
TD_CLIP = 1e3
WARMUP_STEPS = 1000

# checkpointing
DQN_AGENT_DIR = "./trained_agents"
ALTERNATE_SINGLE_FILE = "agent.pth"
SAVE_DIR = "./qmix_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DQN_AGENT_DIR, exist_ok=True)

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Helper datatypes
# ----------------------------
Transition = namedtuple("Transition", ["obs", "state", "actions", "reward", "next_obs", "next_state", "done"])

# ----------------------------
# Vectorized environment (simple subprocess workers)
# ----------------------------
# Each worker runs a K8sAutoscaleEnv instance and communicates via Pipes.
def _env_worker(remote, parent_remote, env_config):
    parent_remote.close()
    env = K8sAutoscaleEnv(env_config)
    # Create deterministic order of agents
    possible_agents = list(env.possible_agents)
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            obs_dict, infos = env.reset(seed=None, options=None)
            # return obs dict and global state
            state = env.get_global_state()
            remote.send((obs_dict, state))
        elif cmd == "reset_with_seed":
            seed = data
            obs_dict, infos = env.reset(seed=seed, options=None)
            state = env.get_global_state()
            remote.send((obs_dict, state))
        elif cmd == "step":
            actions = data  # dict agent->replicas
            obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
            state = env.get_global_state()
            remote.send((obs_dict, rewards, terminateds, truncateds, infos, state))
        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            remote.send(("unknown", None))

class SubprocVecEnv:
    """Minimal subprocess vector env to run multiple K8sAutoscaleEnv instances in parallel."""
    def __init__(self, config_path: str, n_envs: int):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=_env_worker, args=(work_remote, remote, config_path))
            p.daemon = True
            p.start()
            self.ps.append(p)
            work_remote.close()

        # probe agent ids from first env
        self.remotes[0].send(("reset", None))
        obs_dict, state = self.remotes[0].recv()
        self.agent_ids = list(obs_dict.keys())
        self.n_agents = len(self.agent_ids)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        results = [r.recv() for r in self.remotes]
        obs_dicts, states = zip(*results)
        return list(obs_dicts), list(states)

    def reset_with_seed(self, seed: int):
        for i, r in enumerate(self.remotes):
            r.send(("reset_with_seed", seed + i))
        results = [r.recv() for r in self.remotes]
        obs_dicts, states = zip(*results)
        return list(obs_dicts), list(states)

    def step(self, actions_list: List[Dict[str, int]]):
        # actions_list: list of dict per env
        for r, a in zip(self.remotes, actions_list):
            r.send(("step", a))
        results = [r.recv() for r in self.remotes]
        # each result: obs_dict, rewards, terminateds, truncateds, infos, state
        obs_dicts, rewards, terms, truncs, infos, states = zip(*results)
        return list(obs_dicts), list(rewards), list(terms), list(truncs), list(infos), list(states)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except:
                pass
        for p in self.ps:
            p.join(timeout=1)

# ----------------------------
# Prioritized replay (proportional)
# ----------------------------
class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.pos = 0
        self.full = False
        self.buffer = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.eps = 1e-6

    def __len__(self):
        return self.capacity if self.full else self.pos

    def push(self, *args):
        transition = Transition(*args)
        self.buffer[self.pos] = transition
        max_prio = self.priorities.max() if (self.pos > 0 or self.full) else 1.0
        if max_prio <= 0:
            max_prio = 1.0
        self.priorities[self.pos] = max_prio
        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, beta: float = 0.4):
        N = len(self)
        assert N > 0, "Buffer empty"
        prios = self.priorities[:N].astype(np.float64)
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / N
        else:
            probs = probs / probs_sum
        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        # convert to arrays
        obs = np.stack([s.obs for s in samples]).astype(np.float32)          # (B, n_agents, obs_dim)
        state = np.stack([s.state for s in samples]).astype(np.float32)      # (B, state_dim)
        actions = np.stack([s.actions for s in samples]).astype(np.int64)    # (B, n_agents)
        rewards = np.array([s.reward for s in samples], dtype=np.float32)    # (B,)
        next_obs = np.stack([s.next_obs for s in samples]).astype(np.float32)
        next_state = np.stack([s.next_state for s in samples]).astype(np.float32)
        dones = np.array([s.done for s in samples], dtype=np.float32)
        return {
            "obs": obs, "state": state, "actions": actions, "rewards": rewards,
            "next_obs": next_obs, "next_state": next_state, "dones": dones,
            "indices": indices, "weights": weights
        }

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = max(p, self.eps)

# ----------------------------
# Networks
# ----------------------------
class AgentNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class MixingNetwork(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_hidden=MIXING_HIDDEN, hyper_hidden=HYPERNET_HIDDEN):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden = mixing_hidden
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, n_agents * mixing_hidden))
        self.hyper_b1 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, mixing_hidden))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, mixing_hidden))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, 1))
        self.elu = nn.ELU()
    def forward(self, agent_qs, state):
        B = agent_qs.size(0)
        w1 = self.hyper_w1(state).view(B, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(state).view(B, 1, self.mixing_hidden)
        agent_qs_ = agent_qs.view(B, 1, self.n_agents)
        hidden = torch.bmm(agent_qs_, w1).squeeze(1) + b1.squeeze(1)
        hidden = self.elu(hidden)
        w2 = self.hyper_w2(state).view(B, self.mixing_hidden, 1)
        b2 = self.hyper_b2(state).view(B, 1)
        q_tot = torch.bmm(hidden.view(B,1,self.mixing_hidden), w2).squeeze(2) + b2
        return q_tot  # (B,1)

# ----------------------------
# N-step helper
# ----------------------------
class NStepBuffer:
    def __init__(self, n_step: int, gamma: float):
        self.n = n_step
        self.gamma = gamma
        self.buf = deque()

    def push(self, obs, state, actions, reward, next_obs, next_state, done):
        self.buf.append((obs, state, actions, reward, next_obs, next_state, done))
        if len(self.buf) < self.n:
            return None
        # compute n-step return
        ret = 0.0
        for i in range(self.n):
            r = self.buf[i][3]
            ret += (self.gamma ** i) * r
        obs0, state0, actions0 = self.buf[0][0], self.buf[0][1], self.buf[0][2]
        next_obs_n, next_state_n, done_n = self.buf[-1][4], self.buf[-1][5], self.buf[-1][6]
        self.buf.popleft()
        return (obs0, state0, actions0, ret, next_obs_n, next_state_n, done_n)

    def flush(self):
        outs = []
        while self.buf:
            ret = 0.0
            for i in range(len(self.buf)):
                ret += (self.gamma ** i) * self.buf[i][3]
            obs0, state0, actions0 = self.buf[0][0], self.buf[0][1], self.buf[0][2]
            next_obs_n, next_state_n, done_n = self.buf[-1][4], self.buf[-1][5], self.buf[-1][6]
            outs.append((obs0, state0, actions0, ret, next_obs_n, next_state_n, done_n))
            self.buf.popleft()
        return outs

# ----------------------------
# Utility functions
# ----------------------------
def sanitize(a: np.ndarray, clip: float = 1e6) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(a, -clip, clip).astype(np.float32)

def clamp_replicas(prev: int, delta: int) -> int:
    new = int(prev + int(delta))
    return max(MIN_REPLICAS, min(MAX_REPLICAS, new))

# ----------------------------
# Build env & dims
# ----------------------------
def build_envs(n_envs: int):
    venv = SubprocVecEnv(ENV_CONFIG, n_envs)
    agent_ids = venv.agent_ids
    n_agents = venv.n_agents
    # compute state dim by resetting
    obs_dicts, states = venv.reset()
    state_dim = int(states[0].size)
    obs_dim = int(next(iter(obs_dicts[0].values())).shape[0])
    return venv, agent_ids, n_agents, state_dim, obs_dim

# ----------------------------
# Warm-start loader
# ----------------------------
def try_load_per_agent(agent_nets: List[nn.Module], agent_ids: List[str]) -> int:
    loaded = 0
    for i, aid in enumerate(agent_ids):
        fn = os.path.join(DQN_AGENT_DIR, f"{aid}_best.pth")
        if os.path.exists(fn):
            try:
                ck = torch.load(fn, map_location=device)
                if isinstance(ck, dict) and all(k in ck for k in agent_nets[i].state_dict().keys()):
                    agent_nets[i].load_state_dict(ck, strict=True)
                elif isinstance(ck, dict) and "state_dict" in ck:
                    agent_nets[i].load_state_dict(ck["state_dict"], strict=False)
                else:
                    agent_nets[i].load_state_dict(ck, strict=False)
                loaded += 1
                print(f"Loaded per-agent checkpoint for {aid} from {fn}")
            except Exception as e:
                print(f"Failed loading {fn}: {e}")
    # fallback single-file
    if loaded == 0:
        single = os.path.join(DQN_AGENT_DIR, ALTERNATE_SINGLE_FILE) if os.path.exists(os.path.join(DQN_AGENT_DIR, ALTERNATE_SINGLE_FILE)) else ALTERNATE_SINGLE_FILE
        if os.path.exists(single):
            try:
                ck = torch.load(single, map_location=device)
                if isinstance(ck, dict):
                    for i, aid in enumerate(agent_ids):
                        if f"agent_{aid}" in ck:
                            try:
                                agent_nets[i].load_state_dict(ck[f"agent_{aid}"], strict=False)
                                loaded += 1
                            except Exception:
                                pass
                    if loaded == 0:
                        for i in range(len(agent_nets)):
                            try:
                                agent_nets[i].load_state_dict(ck, strict=False)
                                loaded = len(agent_nets)
                            except Exception:
                                pass
                        if loaded > 0:
                            print("Loaded single-file checkpoint into all agents")
            except Exception as e:
                print(f"Failed loading single-file checkpoint {single}: {e}")
    return loaded

# ----------------------------
# TRAIN
# ----------------------------
def main():
    # build vectorized envs
    venv, agent_ids, n_agents, state_dim, obs_dim = build_envs(N_ENVS)
    print(f"Agents detected: {agent_ids} | n_agents={n_agents} | state_dim={state_dim} | obs_dim={obs_dim}")

    # models on device: output n_actions = N_DELTAS
    agent_nets = [AgentNet(obs_dim, N_DELTAS).to(device) for _ in range(n_agents)]
    target_agent_nets = [AgentNet(obs_dim, N_DELTAS).to(device) for _ in range(n_agents)]
    for t, s in zip(target_agent_nets, agent_nets):
        t.load_state_dict(s.state_dict())

    mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    # warm-start
    loaded = try_load_per_agent(agent_nets, agent_ids)
    if loaded > 0:
        for i in range(n_agents):
            target_agent_nets[i].load_state_dict(agent_nets[i].state_dict())
        print(f"Warm-started {loaded}/{n_agents} agents from DQN checkpoints.")
    else:
        print("No per-agent warm-start found. Training from scratch.")

    # optimizer: mixer-first curriculum
    mixer_params = list(mixer.parameters())
    agent_params = [p for net in agent_nets for p in net.parameters()]
    use_mixer_only = MIXER_ONLY_EPOCHS > 0 and loaded > 0
    if use_mixer_only:
        optimizer = torch.optim.Adam(mixer_params, lr=LR)
        print(f"Using mixer-only curriculum for first {MIXER_ONLY_EPOCHS} epochs")
    else:
        optimizer = torch.optim.Adam(agent_params + mixer_params, lr=LR)

    # LR warmup scheduler
    def lr_lambda(step):
        return min(1.0, float(step) / max(1, WARMUP_STEPS))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # replay
    buffer = PrioritizedReplay(REPLAY_SIZE, PRIO_ALPHA)

    # per-env n-step buffers
    nstep_buffers = [NStepBuffer(N_STEP, GAMMA) for _ in range(N_ENVS)]

    # training bookkeeping
    gradient_steps = 0
    total_steps = 0
    best_eval = -1e18
    prev_best_actions = None

    # target beta schedule
    def beta_by_frame(frame):
        return min(1.0, PRIO_BETA_START + frame * (1.0 - PRIO_BETA_START) / max(1, PRIO_BETA_FRAMES))

    print("Starting training loop...")
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        # linear epsilon schedule (shared for simplicity)
        eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * (epoch / EPS_DECAY_EPOCHS))
        steps_needed = STEPS_PER_EPOCH
        steps_collected = 0

        # collect in parallel: run each env until we've collected STEPS_PER_EPOCH total (summed across venvs)
        # we'll step each env once per loop so that all envs proceed roughly in lockstep
        # Initialize env states if first epoch or after episodes ended
        # Use reset for each env if needed
        obs_dicts, states = venv.reset()  # list length N_ENVS
        # For each env, maintain current obs_arr and current replicas (we will query from env via infos occasionally)
        # We will infer current replicas from agent observations: obs[9] = replicas_desired/20.0 (index used in simulator)
        current_replicas = []
        for env_obs in obs_dicts:
            # take replicas_desired from observations
            replicas = []
            for aid in agent_ids:
                o = env_obs[aid]
                # field index 9 is replicas_desired/20.0 as per simulator
                rep = int(round(o[9] * MAX_REPLICAS))
                rep = max(MIN_REPLICAS, min(MAX_REPLICAS, rep))
                replicas.append(rep)
            current_replicas.append(replicas)

        while steps_collected < steps_needed:
            # build actions_list for all envs
            actions_list = []
            delta_list_per_env = []  # store chosen deltas for logging
            for env_idx in range(N_ENVS):
                obs_dict = obs_dicts[env_idx]
                obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)  # (n_agents, obs_dim)
                deltas = []
                action_dict = {}
                for i, aid in enumerate(agent_ids):
                    obs_i = torch.tensor(obs_arr[i:i+1], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        qvals = agent_nets[i](obs_i)  # (1, N_DELTAS)
                        if random.random() < eps:
                            a_idx = random.randrange(N_DELTAS)
                        else:
                            a_idx = int(qvals.argmax(dim=1).item())
                    delta = int(DELTA_ACTIONS[a_idx])
                    prev_rep = current_replicas[env_idx][i]
                    desired = clamp_replicas(prev_rep, delta)
                    action_dict[aid] = int(desired)
                    deltas.append(delta)
                actions_list.append(action_dict)
                delta_list_per_env.append(deltas)

            # step all envs in parallel
            next_obs_dicts, rewards_list, terms_list, truncs_list, infos_list, next_states = venv.step(actions_list)

            # push transitions (n-step)
            for env_idx in range(N_ENVS):
                obs_dict = obs_dicts[env_idx]
                state = states[env_idx]
                next_obs = next_obs_dicts[env_idx]
                next_state = next_states[env_idx]
                rewards = rewards_list[env_idx]
                # shared reward extraction
                if isinstance(rewards, dict):
                    r_raw = float(list(rewards.values())[0])
                else:
                    r_raw = float(rewards)
                # reward scaling + clipping
                r_scaled = float(np.clip(r_raw / REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
                done_flag = any(terms_list[env_idx].values()) if isinstance(terms_list[env_idx], dict) else bool(terms_list[env_idx])

                # build arrays for obs/next_obs (n_agents, obs_dim)
                obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                next_obs_arr = np.stack([next_obs[a] for a in agent_ids], axis=0).astype(np.float32)
                # actions taken are absolute replicas in action_list[env_idx] (1..20), but we store them as ints
                actions_array = np.array([actions_list[env_idx][aid] for aid in agent_ids], dtype=np.int64)

                # push into that env's n-step buffer; when it returns an n-step output, push to PER buffer
                out = nstep_buffers[env_idx].push(sanitize(obs_arr), state.astype(np.float32),
                                                  actions_array, r_scaled,
                                                  sanitize(next_obs_arr), next_state.astype(np.float32), done_flag)
                if out is not None:
                    buffer.push(*out)
                # if episode ended, flush buffer
                if done_flag:
                    for rem in nstep_buffers[env_idx].flush():
                        buffer.push(*rem)
                # update pointers
                obs_dicts[env_idx] = next_obs
                states[env_idx] = next_state
                # update current replicas for delta smoothing use
                current_replicas[env_idx] = [int(round(next_obs[a][9] * MAX_REPLICAS)) for a in agent_ids]

                steps_collected += 1
                total_steps += 1
                if steps_collected >= steps_needed:
                    break

        # switch optimizer when unfreezing agents
        if use_mixer_only and epoch == MIXER_ONLY_EPOCHS + 1:
            for net in agent_nets:
                for p in net.parameters():
                    p.requires_grad = True
            optimizer = torch.optim.Adam(agent_params + mixer_params, lr=LR)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            use_mixer_only = False
            print("Unfroze agents; optimizer now updates agents + mixer")

        # TRAIN: perform gradient updates
        if len(buffer) >= MIN_REPLAY_SIZE:
            n_updates = max(1, int(STEPS_PER_EPOCH * 0.1))
            total_loss = 0.0
            for _ in range(n_updates):
                beta = beta_by_frame(total_steps)
                batch = buffer.sample(BATCH_SIZE, beta=beta)
                # convert to device tensors
                obs_b = torch.tensor(batch["obs"], device=device)            # (B, n_agents, obs_dim)
                state_b = torch.tensor(batch["state"], device=device)        # (B, state_dim)
                actions_b = torch.tensor(batch["actions"], device=device)    # (B, n_agents) absolute replicas
                rewards_b = torch.tensor(batch["rewards"], device=device)    # (B,)
                next_obs_b = torch.tensor(batch["next_obs"], device=device)
                next_state_b = torch.tensor(batch["next_state"], device=device)
                dones_b = torch.tensor(batch["dones"], device=device)
                weights_b = torch.tensor(batch["weights"], device=device)
                indices = batch["indices"]

                B = obs_b.size(0)

                # per-agent Q of taken actions: first compute action indices (map absolute replicas back to delta index)
                # Need agent q-values for each action candidate (delta). We'll compute q_i for each agent for obs.
                agent_qs_taken = []
                for i in range(n_agents):
                    obs_i = obs_b[:, i, :]
                    q_i = agent_nets[i](obs_i)      # (B, N_DELTAS)
                    # compute index of delta that was chosen: based on previous replicas stored in obs vector (field idx 9)
                    # prev_replicas in obs = obs[:,9] * MAX_REPLICAS
                    prev_replicas = torch.clamp((obs_i[:, 9] * MAX_REPLICAS).round().long(), MIN_REPLICAS, MAX_REPLICAS)
                    # actions_b stores absolute desired replicas (1..20). delta = desired - prev
                    desired = actions_b[:, i].long()
                    delta = desired - prev_replicas
                    # map delta to delta index
                    # compute index by finding where DELTA_ACTIONS == delta (vectorized)
                    # create a mapping dict on CPU
                    delta_idx = []
                    delta_cpu = delta.detach().cpu().numpy()
                    # map each element
                    for val in delta_cpu:
                        # clamp val to DELTA_ACTIONS range
                        if val in DELTA_ACTIONS:
                            idx = int(np.where(DELTA_ACTIONS == val)[0][0])
                        else:
                            # if out of range (shouldn't), clip to nearest
                            diffs = np.abs(DELTA_ACTIONS - val)
                            idx = int(diffs.argmin())
                        delta_idx.append(idx)
                    idx_t = torch.tensor(delta_idx, dtype=torch.long, device=device).unsqueeze(1)  # (B,1)
                    q_taken = q_i.gather(1, idx_t).squeeze(1)  # (B,)
                    agent_qs_taken.append(q_taken)
                agent_qs_taken = torch.stack(agent_qs_taken, dim=1)  # (B, n_agents)

                # Double-DQN n-step target: use online argmax on next_obs, evaluate with target nets
                with torch.no_grad():
                    agent_qs_next_online = []
                    for i in range(n_agents):
                        q_online_next = agent_nets[i](next_obs_b[:, i, :])  # (B, N_DELTAS)
                        agent_qs_next_online.append(q_online_next)
                    next_actions_idx = [q.argmax(dim=1) for q in agent_qs_next_online]  # list of (B,)
                    agent_qs_next_eval = []
                    for i in range(n_agents):
                        q_target_next = target_agent_nets[i](next_obs_b[:, i, :])  # (B, N_DELTAS)
                        ai = next_actions_idx[i].unsqueeze(1)
                        q_eval = q_target_next.gather(1, ai).squeeze(1)
                        agent_qs_next_eval.append(q_eval)
                    agent_qs_next_eval = torch.stack(agent_qs_next_eval, dim=1)  # (B, n_agents)
                    q_tot_next = target_mixer(agent_qs_next_eval, next_state_b).squeeze(1)
                    td_target = rewards_b + (1.0 - dones_b) * (GAMMA ** N_STEP) * q_tot_next
                    td_target = torch.clamp(td_target, -TD_CLIP, TD_CLIP)

                q_tot = mixer(agent_qs_taken, state_b).squeeze(1)  # (B,)
                # per-sample absolute TD for priority update
                per_sample_abs = (q_tot - td_target).abs().detach().cpu().numpy()
                # weighted MSE
                loss = (weights_b * F.mse_loss(q_tot, td_target, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(list(mixer.parameters()) + agent_params, GRAD_CLIP)
                optimizer.step()
                scheduler.step()

                total_loss += float(loss.item())
                gradient_steps += 1

                # update priorities
                new_prios = per_sample_abs + 1e-6
                buffer.update_priorities(indices, new_prios)

                if gradient_steps % TARGET_UPDATE_FREQ == 0:
                    for tnet, net in zip(target_agent_nets, agent_nets):
                        tnet.load_state_dict(net.state_dict())
                    target_mixer.load_state_dict(mixer.state_dict())
            avg_loss = total_loss / max(1, n_updates)
        else:
            avg_loss = 0.0

        epoch_time = time.time() - epoch_start

        # EVALUATION (greedy)
        if epoch % 5 == 0 or epoch == EPOCHS:
            eval_episodes = 5
            eval_rewards = []
            last_actions = None
            for _e in range(eval_episodes):
                obs_dicts, states = venv.reset()  # start fresh
                obs_dict = obs_dicts[0]  # take first env for logging
                state = states[0]
                obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                done = False
                total_r = 0.0
                prev_reps = [int(round(obs_arr[i,9] * MAX_REPLICAS)) for i in range(n_agents)]
                while not done:
                    actions = {}
                    actions_delta = {}
                    for i, aid in enumerate(agent_ids):
                        obs_i = torch.tensor(obs_arr[i:i+1], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            qvals = agent_nets[i](obs_i)
                            ai = int(qvals.argmax(dim=1).item())
                            delta = int(DELTA_ACTIONS[ai])
                        desired = clamp_replicas(prev_reps[i], delta)
                        actions[aid] = int(desired)
                        actions_delta[aid] = delta
                        prev_reps[i] = desired
                    next_obs_dicts, rewards, terms, truns, infos, states = venv.step([actions])
                    # single env step returns list; extract index 0
                    next_obs = next_obs_dicts[0]
                    reward_raw = rewards[0] if not isinstance(rewards, list) else rewards[0]
                    if isinstance(reward_raw, dict):
                        r = float(list(reward_raw.values())[0])
                    else:
                        r = float(reward_raw)
                    total_r += float(np.clip(r / REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
                    obs_arr = np.stack([next_obs[a] for a in agent_ids], axis=0).astype(np.float32)
                    done = any(terms[0].values()) if isinstance(terms[0], dict) else bool(terms[0])
                    last_actions = actions.copy()
                eval_rewards.append(total_r)
            mean_eval = float(np.mean(eval_rewards))
            std_eval = float(np.std(eval_rewards))
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
            print(f"[Epoch {epoch:03d} | {epoch_time:.1f}s] eval={mean_eval:.2f}±{std_eval:.2f} | buffer={len(buffer)} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

            if mean_eval > best_eval:
                print("=== New best eval! ===")
                # Show previous best actions vs current greedy (convert to replicas)
                if prev_best_actions is None:
                    print("Prev actions: (none)")
                else:
                    print("Prev best actions (replicas):")
                    for aid in agent_ids:
                        print(f"  {aid}: {prev_best_actions.get(aid, 'N/A')}")
                current_actions = {}
                if last_actions is not None:
                    for aid in agent_ids:
                        current_actions[aid] = last_actions.get(aid, 'N/A')
                print("Current greedy actions (replicas):")
                for aid in agent_ids:
                    print(f"  {aid}: {current_actions.get(aid, 'N/A')}")
                prev_best_actions = current_actions.copy()

                best_eval = mean_eval
                save_data = {
                    "agent_nets": [net.state_dict() for net in agent_nets],
                    "mixer": mixer.state_dict(),
                    "epoch": epoch,
                    "eval": mean_eval,
                }
                # per-agent save
                for i, aid in enumerate(agent_ids):
                    save_data[f"agent_{aid}"] = agent_nets[i].state_dict()
                    torch.save(agent_nets[i].state_dict(), os.path.join(DQN_AGENT_DIR, f"{aid}_best.pth"))
                torch.save(save_data, os.path.join(SAVE_DIR, "qmix_best.pth"))
                print(f"Saved best QMIX checkpoint to {os.path.join(SAVE_DIR, 'qmix_best.pth')} (eval={mean_eval:.2f})")

        # periodic logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} done in {epoch_time:.1f}s | avg_loss={avg_loss:.4f} | buffer={len(buffer)}")

    # final save
    final = {
        "agent_nets": [net.state_dict() for net in agent_nets],
        "mixer": mixer.state_dict(),
    }
    for i, aid in enumerate(agent_ids):
        final[f"agent_{aid}"] = agent_nets[i].state_dict()
    torch.save(final, os.path.join(SAVE_DIR, "qmix_final.pth"))
    print("Training complete. Final saved to:", os.path.join(SAVE_DIR, "qmix_final.pth"))
    venv.close()

if __name__ == "__main__":
    main()
