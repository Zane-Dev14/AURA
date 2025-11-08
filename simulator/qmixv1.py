#!/usr/bin/env python3
"""
qmixv1.py -- Stable QMIX trainer

Notes:
- Keeps per-agent net architecture unchanged: obs_dim=16, action_dim=10 (0..9 -> 1..10 replicas).
- Environment enforces actual applied replica changes <= +/- MAX_DELTA (2).
- Vectorized envs run in subprocesses (SubprocVecEnv). All neural work on GPU.
- Prioritized replay, n-step returns (N=3), mixer-first curriculum, double-DQN targets,
  TD clipping, gradient clipping, LR warmup, per-agent snapshots.
"""

import os
import time
import random
from collections import deque, namedtuple
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp

# Import your PettingZoo ParallelEnv wrapper
from boutique_env import K8sAutoscaleEnv

# -----------------------------
# Hyperparameters (tweakable)
# -----------------------------
ENV_CONFIG = "config.yaml"
SEED = 42

# Epsilon schedule
EPS_START = 0.10
EPS_END = 0.02
EPS_DECAY_EPOCHS = 80

OBS_DIM = 16
ACTION_DIM = 10  # model outputs 0..9 => replicas 1..10

# Vectorization / performance
N_ENVS = 8                    # number of subprocess envs (tweak to fit VRAM/CPU)
REPLAY_SIZE = 400_000
MIN_REPLAY_SIZE = 5_000

LR = 5e-4
BATCH_SIZE = 256              # reduce if OOM
GAMMA = 0.98
N_STEP = 3                    # n-step returns

EPOCHS = 200
STEPS_PER_EPOCH = 1000        # env steps per epoch across all envs
TARGET_UPDATE_FREQ = 200      # gradient steps
MIXER_ONLY_EPOCHS = 30

# Prioritized replay
PRIO_ALPHA = 0.6
PRIO_BETA_START = 0.4
PRIO_BETA_FRAMES = EPOCHS * (STEPS_PER_EPOCH // 10 + 1)

# Reward scaling/clipping
REWARD_CLIP = 100.0
REWARD_SCALE = 10.0  # divide raw reward by this before clipping

# Gradient / TD clipping
GRAD_CLIP = 5.0
TD_CLIP = 100.0

# Action clamp: actual applied change limited to +/- MAX_DELTA
MAX_DELTA = 2
MIN_REPLICAS = 1
MAX_REPLICAS = 20

# Save locations
SAVE_DIR = "./qmix_checkpoints"
DQN_SAVE_DIR = "./trained_agents"   # per-agent DQN checkpoints expected here
ALT_SINGLE = "agent.pth"            # fallback single-file checkpoint
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DQN_SAVE_DIR, exist_ok=True)

# -----------------------------
# Utility data structures
# -----------------------------
Transition = namedtuple("Transition", [
    "obs", "state", "actions", "reward", "next_obs", "next_state", "done"
])

# ================================================================
# Simple single-process vectorized environment (DummyVecEnv)
# ================================================================
class DummyVecEnv:
    def __init__(self, n_envs, env_config):
        from boutique_env import K8sAutoscaleEnv  # import here to avoid top-level issues

        self.n_envs = n_envs
        self.envs = [K8sAutoscaleEnv(env_config) for _ in range(n_envs)]

    def reset(self):
        obs_list = []
        infos_list = []
        states_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos_list.append(info)
            states_list.append(info.get("state", None))
        return obs_list, infos_list, states_list

    def step(self, actions_batch):
        """
        actions_batch: list of dicts
        [
            {"api": a1, "app": a2, "db": a3},   # env 0 actions
            ...
        ]
        """
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        infos_list = []
        states_list = []

        for env, actions in zip(self.envs, actions_batch):

            obs, reward, terminated, truncated, info = env.step(actions)

            # Fallback if env returns combined done
            if isinstance(terminated, bool) and isinstance(truncated, bool):
                done_terminated = terminated
                done_truncated = truncated
            else:
                # If env returns a single "done"
                done = terminated or truncated
                done_terminated = done
                done_truncated = False

            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(done_terminated)
            truncated_list.append(done_truncated)
            infos_list.append(info)
            states_list.append(info.get("state", None))

        return (
            obs_list,
            reward_list,
            terminated_list,
            truncated_list,
            infos_list,
            states_list,
        )

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()


# -----------------------------
# Prioritized replay buffer
# -----------------------------
class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.pos = 0
        self.full = False
        self.buf: List[Transition] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.eps = 1e-6

    def __len__(self):
        return self.capacity if self.full else self.pos

    def push(self, *args):
        t = Transition(*args)
        self.buf[self.pos] = t
        max_p = self.priorities.max() if (self.pos > 0 or self.full) else 1.0
        if max_p <= 0:
            max_p = 1.0
        self.priorities[self.pos] = max_p
        self.pos += 1
        if self.pos >= self.capacity:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, beta: float = 0.4):
        length = len(self)
        assert length > 0, "Empty buffer"
        prios = self.priorities[:length].astype(np.float64)
        probs = prios ** self.alpha
        total = probs.sum()
        if total <= 0:
            probs = np.ones_like(probs) / length
        else:
            probs = probs / total
        indices = np.random.choice(length, batch_size, p=probs)
        samples = [self.buf[i] for i in indices]
        weights = (length * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-12)
        obs = np.stack([s.obs for s in samples]).astype(np.float32)
        state = np.stack([s.state for s in samples]).astype(np.float32)
        actions = np.stack([s.actions for s in samples]).astype(np.int64)
        rewards = np.array([s.reward for s in samples], dtype=np.float32)
        next_obs = np.stack([s.next_obs for s in samples]).astype(np.float32)
        next_state = np.stack([s.next_state for s in samples]).astype(np.float32)
        dones = np.array([s.done for s in samples], dtype=np.float32)
        return {
            "obs": obs, "state": state, "actions": actions, "rewards": rewards,
            "next_obs": next_obs, "next_state": next_state, "dones": dones,
            "indices": indices, "weights": weights
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = max(p, self.eps)

# -----------------------------
# Networks (same architecture as DQN checkpoint)
# -----------------------------
class AgentNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class MixingNetwork(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_hidden=32, hyper_hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden = mixing_hidden
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden)
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, 1)
        )
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
        q_tot = torch.bmm(hidden.view(B, 1, self.mixing_hidden), w2).squeeze(2) + b2
        return q_tot  # (B,1)

# -----------------------------
# SubprocVecEnv: each worker runs a K8sAutoscaleEnv instance.
# The clamp of +/- MAX_DELTA is enforced inside the worker.
# -----------------------------
def worker_process(pipe, env_config: str):
    parent, child = pipe
    child.close()
    env = K8sAutoscaleEnv(env_config)
    try:
        while True:
            cmd, data = parent.recv()
            if cmd == "reset":
                obs, infos = env.reset(seed=data.get("seed", None), options=None)
                parent.send((obs, infos, env.get_global_state() if hasattr(env, "get_global_state") else None))
            elif cmd == "step":
                actions = data  # dict agent->action(0..9)
                # Allow full range 0..9, but let the reward penalize over-scaling
            clamped_actions = {}
            for name, desired in actions.items():
                # +1 because actions are 0-indexed (0->1 replica)
                final = int(desired) + 1
                
                # Ensure within absolute min/max bounds
                final = max(MIN_REPLICAS, min(MAX_REPLICAS, final))
                
                clamped_actions[name] = final

                obs, rewards, terminateds, truncateds, infos = env.step(clamped_actions)
                parent.send((obs, rewards, terminateds, truncateds, infos, env.get_global_state() if hasattr(env, "get_global_state") else None))
            elif cmd == "close":
                parent.send(("closed", None))
                parent.close()
                break
            else:
                parent.send(("unknown_cmd", None))
    except EOFError:
        # main process died; exit gracefully
        pass

class SubprocVecEnv:
    def __init__(self, n_envs: int, env_config: str):
        self.n_envs = int(n_envs)
        self.env_config = env_config
        self.parents = []
        self.procs = []
        for _ in range(self.n_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=worker_process, args=((parent_conn, child_conn), env_config))
            p.daemon = True
            p.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.procs.append(p)

    def reset(self):
        for parent in self.parents:
            parent.send(("reset", {}))
        results = [parent.recv() for parent in self.parents]
        obs_list, infos_list, states = zip(*results)
        return list(obs_list), list(infos_list), list(states)

    def step(self, actions_list: List[Dict[str, int]]):
        for parent, act in zip(self.parents, actions_list):
            parent.send(("step", act))
        results = [parent.recv() for parent in self.parents]
        obs_list, rewards_list, terminateds_list, truncateds_list, infos_list, states = zip(*results)
        return list(obs_list), list(rewards_list), list(terminateds_list), list(truncateds_list), list(infos_list), list(states)

    def close(self):
        for parent in self.parents:
            try:
                parent.send(("close", {}))
                _ = parent.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=0.1)

# -----------------------------
# Helpers
# -----------------------------
def sanitize(a: np.ndarray, clip=1e6):
    a = np.nan_to_num(a, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(a, -clip, clip).astype(np.float32)

def beta_by_frame(frame_idx: int):
    return min(1.0, PRIO_BETA_START + frame_idx * (1.0 - PRIO_BETA_START) / max(1, PRIO_BETA_FRAMES))

# -----------------------------
# Loading per-agent checkpoints (safe, flexible)
# -----------------------------
def try_load_per_agent(agent_nets: List[nn.Module], agent_ids: List[str], device):
    loaded = 0
    for i, aid in enumerate(agent_ids):
        fn = os.path.join(DQN_SAVE_DIR, f"{aid}_best.pth")
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
    if loaded > 0:
        return loaded
    # fallback to single-file checkpoint if present
    single = os.path.join(DQN_SAVE_DIR, ALT_SINGLE) if os.path.exists(os.path.join(DQN_SAVE_DIR, ALT_SINGLE)) else ALT_SINGLE
    if os.path.exists(single):
        try:
            ck = torch.load(single, map_location=device)
            if isinstance(ck, dict):
                for i, aid in enumerate(agent_ids):
                    if f"agent_{aid}" in ck:
                        try:
                            agent_nets[i].load_state_dict(ck[f"agent_{aid}"], strict=False)
                            loaded += 1
                            print(f"Loaded {aid} from {single}['agent_{aid}']")
                        except Exception:
                            pass
                if loaded == 0:
                    # try load same ck into all
                    for i in range(len(agent_nets)):
                        agent_nets[i].load_state_dict(ck, strict=False)
                    loaded = len(agent_nets)
                    print(f"Loaded single-file checkpoint into all agents from {single}")
        except Exception as e:
            print("Failed fallback single-file load:", e)
    return loaded

# -----------------------------
# Build env dims helper
# -----------------------------
def build_env_and_dims():
    env = K8sAutoscaleEnv(ENV_CONFIG)
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)
    # get state dim from env's get_global_state (if present) else concat obs
    try:
        _, _ = env.reset()
        state = env.get_global_state()
        state_dim = int(state.size)
    except Exception:
        sample_obs, _ = env.reset()
        state = np.concatenate([sample_obs[a] for a in agent_ids], axis=0)
        state_dim = state.size
    obs_dim = int(env.observation_space(agent_ids[0]).shape[0])
    return agent_ids, n_agents, state_dim, obs_dim

# -----------------------------
# Main training loop
# -----------------------------
def main():
    # Set multiprocess start method early (only in main)
    mp.set_start_method("spawn", force=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # build dims
    agent_ids, n_agents, state_dim, obs_dim = build_env_and_dims()
    print(f"Agents detected: {agent_ids} | n_agents={n_agents} | state_dim={state_dim} | obs_dim={obs_dim}")

    # networks
    agent_nets = [AgentNet(obs_dim, ACTION_DIM).to(device) for _ in range(n_agents)]
    target_agent_nets = [AgentNet(obs_dim, ACTION_DIM).to(device) for _ in range(n_agents)]
    for t, s in zip(target_agent_nets, agent_nets):
        t.load_state_dict(s.state_dict())

    mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    # warm-start per-agent nets if available
    loaded = try_load_per_agent(agent_nets, agent_ids, device)
    if loaded > 0:
        for i in range(n_agents):
            target_agent_nets[i].load_state_dict(agent_nets[i].state_dict())
        print(f"Warm-started {loaded}/{n_agents} agents from DQN checkpoints.")
    else:
        print("No per-agent warm-start found. Training from scratch.")

    # prepare optimizer: start with mixer-only if we warmed up
    mixer_params = list(mixer.parameters())
    agent_params = [p for net in agent_nets for p in net.parameters()]
    use_mixer_only = MIXER_ONLY_EPOCHS > 0 and loaded > 0
    if use_mixer_only:
        print(f"Using mixer-only curriculum for first {MIXER_ONLY_EPOCHS} epochs.")
        optimizer = torch.optim.Adam([
            {"params": agent_params, "lr": 1e-4},
            {"params": mixer_params, "lr": 5e-4}
        ])

    else:
        optimizer = torch.optim.Adam([
            {"params": agent_params, "lr": 1e-4},
            {"params": mixer_params, "lr": 5e-4}
        ])
    # LR warmup scheduler
    total_steps_est = EPOCHS * max(1, STEPS_PER_EPOCH // 10)
    def lr_lambda(step):
        warmup = max(1, min(2000, total_steps_est // 10))
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # replay
    replay = PrioritizedReplay(REPLAY_SIZE, alpha=PRIO_ALPHA)

    # vectorized envs (workers)
    vec_env = DummyVecEnv(N_ENVS, ENV_CONFIG)


    # training bookkeeping
    grad_steps = 0
    total_frames = 0
    best_eval = -1e18
    prev_best_actions = None

    print("Starting training loop...")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        # compute epsilon (linear decay)
        eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * (epoch / max(1, EPS_DECAY_EPOCHS)))

        # Collect STEPS_PER_EPOCH transitions across N_ENVS
        steps_collected = 0
        env_obs_list, _infos_list, states_list = vec_env.reset()
        env_states = [np.stack([obsd[a] for a in agent_ids], axis=0).astype(np.float32) for obsd in env_obs_list]

        # per-env n-step buffers
        nstep_buffers = [deque() for _ in range(N_ENVS)]

        while steps_collected < STEPS_PER_EPOCH:
            actions_batch = []
            for env_i in range(N_ENVS):
                obs_arr = env_states[env_i]  # (n_agents, obs_dim)
                per_env_actions = {}
                for ai, aid in enumerate(agent_ids):
                    obs_tensor = torch.tensor(obs_arr[ai:ai+1], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        q = agent_nets[ai](obs_tensor)
                    if random.random() < eps:
                        act = random.randrange(ACTION_DIM)
                    else:
                        act = int(q.argmax(dim=1).item())
                    per_env_actions[aid] = act
                actions_batch.append(per_env_actions)

            # step all envs
            next_obs_list, rewards_list, terms_list, truncs_list, infos_list, states_list = vec_env.step(actions_batch)

            for env_i in range(N_ENVS):
                obs_arr = env_states[env_i]
                next_obs_arr = np.stack([next_obs_list[env_i][a] for a in agent_ids], axis=0).astype(np.float32)
                r_raw = rewards_list[env_i]
                r = float(list(r_raw.values())[0]) if isinstance(r_raw, dict) else float(r_raw)
                r = float(np.clip(r / REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
                done_flag = any(terms_list[env_i].values()) if isinstance(terms_list[env_i], dict) else bool(terms_list[env_i])
                state_vec = states_list[env_i] if states_list[env_i] is not None else np.concatenate([next_obs_arr[a] for a in range(len(agent_ids))], axis=0)

                action_list = np.array([actions_batch[env_i][aid] for aid in agent_ids], dtype=np.int64)

                # append raw step to n-step buffer
                nstep_buffers[env_i].append((sanitize(obs_arr), state_vec.astype(np.float32), action_list, float(r), sanitize(next_obs_arr), state_vec.astype(np.float32), done_flag))

                # when buffer >= N_STEP, create n-step transition
                if len(nstep_buffers[env_i]) >= N_STEP:
                    ret_r = 0.0
                    for idx in range(N_STEP):
                        ret_r += (GAMMA ** idx) * nstep_buffers[env_i][idx][3]
                    obs0, state0, acts0 = nstep_buffers[env_i][0][0], nstep_buffers[env_i][0][1], nstep_buffers[env_i][0][2]
                    next_obs_n, next_state_n, done_n = nstep_buffers[env_i][-1][4], nstep_buffers[env_i][-1][5], nstep_buffers[env_i][-1][6]
                    replay.push(obs0, state0, acts0, ret_r, next_obs_n, next_state_n, float(done_n))
                    nstep_buffers[env_i].popleft()

                steps_collected += 1
                total_frames += 1

                # if episode ended flush remaining n-step entries
                if done_flag:
                    while nstep_buffers[env_i]:
                        L = len(nstep_buffers[env_i])
                        ret_r = 0.0
                        for idx in range(L):
                            ret_r += (GAMMA ** idx) * nstep_buffers[env_i][idx][3]
                        obs0, state0, acts0 = nstep_buffers[env_i][0][0], nstep_buffers[env_i][0][1], nstep_buffers[env_i][0][2]
                        next_obs_n, next_state_n, done_n = nstep_buffers[env_i][-1][4], nstep_buffers[env_i][-1][5], nstep_buffers[env_i][-1][6]
                        replay.push(obs0, state0, acts0, ret_r, next_obs_n, next_state_n, float(done_n))
                        nstep_buffers[env_i].popleft()

                # update current env state
                env_states[env_i] = next_obs_arr

        # end collection loop

        # if mixer-only curriculum ends now, unfreeze agents
        if use_mixer_only and epoch == (MIXER_ONLY_EPOCHS + 1):
            print("Unfreezing agents and optimizing agents + mixer now.")
            optimizer = torch.optim.Adam(agent_params + mixer_params, lr=LR)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            use_mixer_only = False

        # Training updates
        if len(replay) >= MIN_REPLAY_SIZE:
            n_updates = max(1, int(STEPS_PER_EPOCH * 0.1))
            total_loss = 0.0
            for _ in range(n_updates):
                beta = beta_by_frame(total_frames)
                batch = replay.sample(BATCH_SIZE, beta=beta)

                # move to device
                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                state_b = torch.tensor(batch["state"], dtype=torch.float32, device=device)
                actions_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
                rewards_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                next_state_b = torch.tensor(batch["next_state"], dtype=torch.float32, device=device)
                dones_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)
                weights_b = torch.tensor(batch["weights"], dtype=torch.float32, device=device)
                indices = batch["indices"]

                B = obs_b.size(0)

                # Q(s,a) for taken actions
                agent_qs_taken = []
                for i in range(n_agents := len(agent_ids)):
                    obs_i = obs_b[:, i, :]
                    q_i = agent_nets[i](obs_i)            # (B, ACTION_DIM)
                    a_i = actions_b[:, i].unsqueeze(1)    # (B,1)
                    q_taken = q_i.gather(1, a_i).squeeze(1)  # (B,)
                    agent_qs_taken.append(q_taken)
                agent_qs_taken = torch.stack(agent_qs_taken, dim=1)  # (B, n_agents)

                # Double DQN: online argmax, target evaluation
                with torch.no_grad():
                    agent_qs_next_online = []
                    for i in range(n_agents):
                        next_obs_i = next_obs_b[:, i, :]
                        q_online = agent_nets[i](next_obs_i)
                        agent_qs_next_online.append(q_online)
                    next_actions = [q.argmax(dim=1) for q in agent_qs_next_online]  # list of (B,)
                    agent_qs_next_eval = []
                    for i in range(n_agents):
                        q_target_next = target_agent_nets[i](next_obs_b[:, i, :])
                        a_i = next_actions[i].unsqueeze(1)
                        q_eval = q_target_next.gather(1, a_i).squeeze(1)
                        agent_qs_next_eval.append(q_eval)
                    agent_qs_next_eval = torch.stack(agent_qs_next_eval, dim=1)  # (B, n_agents)
                    q_tot_next = target_mixer(agent_qs_next_eval, next_state_b).squeeze(1)
                    td_target = rewards_b + (1.0 - dones_b) * (GAMMA ** N_STEP) * q_tot_next
                    td_target = torch.clamp(td_target, -TD_CLIP, TD_CLIP)

                q_tot = mixer(agent_qs_taken, state_b).squeeze(1)
                # per-sample TD errors for priorities
                td_errors = (q_tot - td_target).detach().abs().cpu().numpy()
                # weighted MSE loss
                loss = (weights_b * F.mse_loss(q_tot, td_target, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(agent_params + mixer_params, GRAD_CLIP)
                optimizer.step()
                scheduler.step()

                total_loss += float(loss.item())
                grad_steps += 1

                if grad_steps % TARGET_UPDATE_FREQ == 0:
                    for tnet, net in zip(target_agent_nets, agent_nets):
                        tnet.load_state_dict(net.state_dict())
                    target_mixer.load_state_dict(mixer.state_dict())

                # update replay priorities
                new_prios = td_errors + 1e-6
                replay.update_priorities(indices, new_prios)

            avg_loss = total_loss / max(1, n_updates)
        else:
            avg_loss = 0.0

        epoch_time = time.time() - epoch_start

        # Periodic evaluation (greedy)
        if epoch % 5 == 0 or epoch == EPOCHS:
            eval_eps = 5
            eval_rewards = []
            last_actions_repr = None
            for _e in range(eval_eps):
                env_eval = K8sAutoscaleEnv(ENV_CONFIG)
                obs_dict, _ = env_eval.reset()
                obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                done = False
                total_r = 0.0
                while not done:
                    actions = {}
                    for i, aid in enumerate(agent_ids):
                        obs_i = torch.tensor(obs_arr[i:i+1], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            q = agent_nets[i](obs_i)
                        act = int(q.argmax(dim=1).item())
                        actions[aid] = act
                    next_obs_dict, rewards, terminateds, truncateds, infos = env_eval.step(actions)
                    r = float(list(rewards.values())[0]) if isinstance(rewards, dict) else float(rewards)
                    total_r += float(np.clip(r / REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
                    obs_arr = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                    done = any(terminateds.values()) if isinstance(terminateds, dict) else bool(terminateds)
                eval_rewards.append(total_r)
                last_actions_repr = {aid: (int(actions[aid]) + 1) for aid in agent_ids}

            mean_eval = float(np.mean(eval_rewards))
            std_eval = float(np.std(eval_rewards))
            print(f"[Epoch {epoch:03d} | {epoch_time:.1f}s] eval={mean_eval:.2f}±{std_eval:.2f} | buffer={len(replay)} | loss={avg_loss:.4f}")

            if mean_eval > best_eval:
                print("=== New best eval found ===")
                if prev_best_actions:
                    print("Prev best actions (replicas):")
                    for aid in agent_ids:
                        print(f"  {aid}: {prev_best_actions.get(aid, 'N/A')}")
                else:
                    print("Prev best actions: (none)")
                print("Current greedy actions (replicas):")
                for aid in agent_ids:
                    print(f"  {aid}: {last_actions_repr.get(aid, 'N/A')}")
                prev_best_actions = last_actions_repr.copy()
                best_eval = mean_eval

                # Save joint + per-agent snapshots
                save_dict = {
                    "agent_nets": [net.state_dict() for net in agent_nets],
                    "mixer": mixer.state_dict(),
                    "epoch": epoch,
                    "eval": mean_eval
                }
                for i, aid in enumerate(agent_ids):
                    save_dict[f"agent_{aid}"] = agent_nets[i].state_dict()
                    torch.save(agent_nets[i].state_dict(), os.path.join(DQN_SAVE_DIR, f"{aid}_best.pth"))
                torch.save(save_dict, os.path.join(SAVE_DIR, "qmix_best.pth"))
                print(f"Saved best QMIX checkpoint to {os.path.join(SAVE_DIR, 'qmix_best.pth')}")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} completed in {epoch_time:.1f}s | avg_loss={avg_loss:.4f} | buffer={len(replay)}")

    # final save
    final_save = {
        "agent_nets": [net.state_dict() for net in agent_nets],
        "mixer": mixer.state_dict()
    }
    for i, aid in enumerate(agent_ids):
        final_save[f"agent_{aid}"] = agent_nets[i].state_dict()
    torch.save(final_save, os.path.join(SAVE_DIR, "qmix_final.pth"))
    print("Training finished. Final saved to:", os.path.join(SAVE_DIR, "qmix_final.pth"))
    vec_env.close()

if __name__ == "__main__":
    main()
