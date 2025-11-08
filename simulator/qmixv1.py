#!/usr/bin/env python3
"""
qmixv1.py -- Fixed QMIX trainer with proper logging

Key fixes:
- Removed MAX_DELTA clamping (full action space)
- Fixed worker_process indentation bug
- Proper mixer-only curriculum
- Comprehensive logging for debugging
"""

import os
import time
import random
from collections import deque, namedtuple, Counter
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp

from boutique_env import K8sAutoscaleEnv

# -----------------------------
# Hyperparameters
# -----------------------------
ENV_CONFIG = "config.yaml"
SEED = 42

EPS_START = 0.10
EPS_END = 0.02
EPS_DECAY_EPOCHS = 80

OBS_DIM = 16
ACTION_DIM = 10

N_ENVS = 8
REPLAY_SIZE = 400_000
MIN_REPLAY_SIZE = 5_000

LR = 5e-4
BATCH_SIZE = 256
GAMMA = 0.98
N_STEP = 3

EPOCHS = 200
STEPS_PER_EPOCH = 1000
TARGET_UPDATE_FREQ = 200
MIXER_ONLY_EPOCHS = 30  # Freeze agents, train mixer only

PRIO_ALPHA = 0.6
PRIO_BETA_START = 0.4
PRIO_BETA_FRAMES = EPOCHS * (STEPS_PER_EPOCH // 10 + 1)

REWARD_CLIP = 100.0
REWARD_SCALE = 10.0  # Changed from 50.0

GRAD_CLIP = 5.0
TD_CLIP = 100.0

MIN_REPLICAS = 1
MAX_REPLICAS = 10  # Allow full range

SAVE_DIR = "./qmix_checkpoints"
DQN_SAVE_DIR = "./trained_agents"
ALT_SINGLE = "agent.pth"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DQN_SAVE_DIR, exist_ok=True)

Transition = namedtuple("Transition", [
    "obs", "state", "actions", "reward", "next_obs", "next_state", "done"
])

# -----------------------------
# DummyVecEnv (single-process)
# -----------------------------
class DummyVecEnv:
    def __init__(self, n_envs, env_config):
        self.n_envs = n_envs
        self.envs = [K8sAutoscaleEnv(env_config) for _ in range(n_envs)]

    def reset(self):
        obs_list, infos_list, states_list = [], [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos_list.append(info)
            states_list.append(env.get_global_state() if hasattr(env, "get_global_state") else None)
        return obs_list, infos_list, states_list

    def step(self, actions_batch):
        obs_list, reward_list, term_list, trunc_list, info_list, state_list = [], [], [], [], [], []
        for env, actions in zip(self.envs, actions_batch):
            obs, reward, term, trunc, info = env.step(actions)
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(term if isinstance(term, bool) else any(term.values()))
            trunc_list.append(trunc if isinstance(trunc, bool) else any(trunc.values()))
            info_list.append(info)
            state_list.append(env.get_global_state() if hasattr(env, "get_global_state") else None)
        return obs_list, reward_list, term_list, trunc_list, info_list, state_list

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

# -----------------------------
# Prioritized Replay
# -----------------------------
class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.pos = 0
        self.full = False
        self.buf = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.eps = 1e-6

    def __len__(self):
        return self.capacity if self.full else self.pos

    def push(self, *args):
        self.buf[self.pos] = Transition(*args)
        max_p = self.priorities.max() if (self.pos > 0 or self.full) else 1.0
        self.priorities[self.pos] = max(max_p, 1e-6)
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, beta: float = 0.4):
        length = len(self)
        prios = self.priorities[:length].astype(np.float64)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(length, batch_size, p=probs)
        samples = [self.buf[i] for i in indices]
        weights = (length * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return {
            "obs": np.stack([s.obs for s in samples]).astype(np.float32),
            "state": np.stack([s.state for s in samples]).astype(np.float32),
            "actions": np.stack([s.actions for s in samples]).astype(np.int64),
            "rewards": np.array([s.reward for s in samples], dtype=np.float32),
            "next_obs": np.stack([s.next_obs for s in samples]).astype(np.float32),
            "next_state": np.stack([s.next_state for s in samples]).astype(np.float32),
            "dones": np.array([s.done for s in samples], dtype=np.float32),
            "indices": indices,
            "weights": weights.astype(np.float32)
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = max(p, self.eps)

# -----------------------------
# Networks
# -----------------------------
class AgentNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
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
        hidden = torch.bmm(agent_qs.view(B, 1, self.n_agents), w1).squeeze(1) + b1.squeeze(1)
        hidden = self.elu(hidden)
        w2 = self.hyper_w2(state).view(B, self.mixing_hidden, 1)
        b2 = self.hyper_b2(state).view(B, 1)
        return torch.bmm(hidden.view(B, 1, self.mixing_hidden), w2).squeeze(2) + b2

# -----------------------------
# Helpers
# -----------------------------
def sanitize(a: np.ndarray, clip=1e6):
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=clip, neginf=-clip), -clip, clip).astype(np.float32)

def beta_by_frame(frame_idx: int):
    return min(1.0, PRIO_BETA_START + frame_idx * (1.0 - PRIO_BETA_START) / max(1, PRIO_BETA_FRAMES))

def try_load_per_agent(agent_nets: List[nn.Module], agent_ids: List[str], device):
    loaded = 0
    for i, aid in enumerate(agent_ids):
        fn = os.path.join(DQN_SAVE_DIR, f"{aid}_best.pth")
        if os.path.exists(fn):
            try:
                agent_nets[i].load_state_dict(torch.load(fn, map_location=device), strict=False)
                loaded += 1
                print(f"✓ Loaded {aid} from {fn}")
            except Exception as e:
                print(f"✗ Failed loading {aid}: {e}")
    return loaded

def build_env_and_dims():
    env = K8sAutoscaleEnv(ENV_CONFIG)
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)
    obs_dim = int(env.observation_space(agent_ids[0]).shape[0])
    env.reset()
    state = env.get_global_state() if hasattr(env, "get_global_state") else np.concatenate([np.zeros(obs_dim) for _ in agent_ids])
    state_dim = int(state.size)
    return agent_ids, n_agents, state_dim, obs_dim

# -----------------------------
# Main Training
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    agent_ids, n_agents, state_dim, obs_dim = build_env_and_dims()
    print(f"Agents: {agent_ids} | n_agents={n_agents} | state_dim={state_dim}")

    # Networks
    agent_nets = [AgentNet(obs_dim, ACTION_DIM).to(device) for _ in range(n_agents)]
    target_agent_nets = [AgentNet(obs_dim, ACTION_DIM).to(device) for _ in range(n_agents)]
    for t, s in zip(target_agent_nets, agent_nets):
        t.load_state_dict(s.state_dict())

    mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    # Warm-start
    loaded = try_load_per_agent(agent_nets, agent_ids, device)
    if loaded > 0:
        for i in range(n_agents):
            target_agent_nets[i].load_state_dict(agent_nets[i].state_dict())
        print(f"Warm-started {loaded}/{n_agents} agents")
    
    # Optimizer: FREEZE agents for mixer-only curriculum
    mixer_params = list(mixer.parameters())
    agent_params = [p for net in agent_nets for p in net.parameters()]
    
    if MIXER_ONLY_EPOCHS > 0 and loaded > 0:
        print(f"🔒 FREEZING agents for first {MIXER_ONLY_EPOCHS} epochs (mixer-only curriculum)")
        for p in agent_params:
            p.requires_grad = False
        optimizer = torch.optim.Adam(mixer_params, lr=LR)
        agents_frozen = True
    else:
        print("Training agents + mixer from scratch")
        optimizer = torch.optim.Adam(agent_params + mixer_params, lr=LR)
        agents_frozen = False
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.5, 1.0 - s / (EPOCHS * 100)))

    replay = PrioritizedReplay(REPLAY_SIZE, alpha=PRIO_ALPHA)
    vec_env = DummyVecEnv(N_ENVS, ENV_CONFIG)

    grad_steps = 0
    total_frames = 0
    best_eval = -1e18
    prev_best_actions = None

    print("Starting training...\n")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * (epoch / max(1, EPS_DECAY_EPOCHS)))

        # === UNFREEZE AGENTS AFTER MIXER-ONLY PHASE ===
        if agents_frozen and epoch == (MIXER_ONLY_EPOCHS + 1):
            print(f"\n🔓 UNFREEZING agents at epoch {epoch}")
            for p in agent_params:
                p.requires_grad = True
            optimizer = torch.optim.Adam(agent_params + mixer_params, lr=LR)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.5, 1.0 - s / (EPOCHS * 100)))
            agents_frozen = False

        # === COLLECTION PHASE ===
        action_counter = Counter()
        steps_collected = 0
        env_obs_list, _, states_list = vec_env.reset()
        env_states = [np.stack([obsd[a] for a in agent_ids], axis=0).astype(np.float32) for obsd in env_obs_list]
        nstep_buffers = [deque() for _ in range(N_ENVS)]

        while steps_collected < STEPS_PER_EPOCH:
            actions_batch = []
            for env_i in range(N_ENVS):
                obs_arr = env_states[env_i]
                per_env_actions = {}
                for ai, aid in enumerate(agent_ids):
                    obs_t = torch.tensor(obs_arr[ai:ai+1], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        q = agent_nets[ai](obs_t)
                    if random.random() < eps:
                        act = random.randrange(ACTION_DIM)
                    else:
                        act = int(q.argmax(dim=1).item())
                    per_env_actions[aid] = act
                    action_counter[act] += 1
                actions_batch.append(per_env_actions)

            next_obs_list, rewards_list, terms_list, truncs_list, _, states_list = vec_env.step(actions_batch)

            for env_i in range(N_ENVS):
                obs_arr = env_states[env_i]
                next_obs_arr = np.stack([next_obs_list[env_i][a] for a in agent_ids], axis=0).astype(np.float32)
                r_raw = rewards_list[env_i]
                r = float(list(r_raw.values())[0]) if isinstance(r_raw, dict) else float(r_raw)
                r = float(np.clip(r / REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
                done = terms_list[env_i]
                state_vec = states_list[env_i] if states_list[env_i] is not None else np.concatenate(next_obs_arr)
                action_list = np.array([actions_batch[env_i][aid] for aid in agent_ids], dtype=np.int64)

                nstep_buffers[env_i].append((sanitize(obs_arr), state_vec.astype(np.float32), action_list, float(r), sanitize(next_obs_arr), state_vec.astype(np.float32), done))

                if len(nstep_buffers[env_i]) >= N_STEP:
                    ret_r = sum((GAMMA ** idx) * nstep_buffers[env_i][idx][3] for idx in range(N_STEP))
                    obs0, state0, acts0 = nstep_buffers[env_i][0][:3]
                    next_obs_n, next_state_n, done_n = nstep_buffers[env_i][-1][4:]
                    replay.push(obs0, state0, acts0, ret_r, next_obs_n, next_state_n, float(done_n))
                    nstep_buffers[env_i].popleft()

                steps_collected += 1
                total_frames += 1

                if done:
                    while nstep_buffers[env_i]:
                        L = len(nstep_buffers[env_i])
                        ret_r = sum((GAMMA ** idx) * nstep_buffers[env_i][idx][3] for idx in range(L))
                        obs0, state0, acts0 = nstep_buffers[env_i][0][:3]
                        next_obs_n, next_state_n, done_n = nstep_buffers[env_i][-1][4:]
                        replay.push(obs0, state0, acts0, ret_r, next_obs_n, next_state_n, float(done_n))
                        nstep_buffers[env_i].popleft()

                env_states[env_i] = next_obs_arr

        # === TRAINING PHASE ===
        if len(replay) >= MIN_REPLAY_SIZE:
            n_updates = max(1, int(STEPS_PER_EPOCH * 0.1))
            total_loss = 0.0
            grad_norms = []
            
            for _ in range(n_updates):
                beta = beta_by_frame(total_frames)
                batch = replay.sample(BATCH_SIZE, beta=beta)

                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                state_b = torch.tensor(batch["state"], dtype=torch.float32, device=device)
                actions_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)
                rewards_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                next_state_b = torch.tensor(batch["next_state"], dtype=torch.float32, device=device)
                dones_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)
                weights_b = torch.tensor(batch["weights"], dtype=torch.float32, device=device)
                indices = batch["indices"]

                # Q(s,a)
                agent_qs_taken = []
                for i in range(n_agents):
                    q_i = agent_nets[i](obs_b[:, i, :])
                    agent_qs_taken.append(q_i.gather(1, actions_b[:, i].unsqueeze(1)).squeeze(1))
                agent_qs_taken = torch.stack(agent_qs_taken, dim=1)

                # Double DQN target
                with torch.no_grad():
                    next_actions = [agent_nets[i](next_obs_b[:, i, :]).argmax(dim=1) for i in range(n_agents)]
                    agent_qs_next = [target_agent_nets[i](next_obs_b[:, i, :]).gather(1, next_actions[i].unsqueeze(1)).squeeze(1) for i in range(n_agents)]
                    agent_qs_next = torch.stack(agent_qs_next, dim=1)
                    q_tot_next = target_mixer(agent_qs_next, next_state_b).squeeze(1)
                    td_target = rewards_b + (1.0 - dones_b) * (GAMMA ** N_STEP) * q_tot_next
                    td_target = torch.clamp(td_target, -TD_CLIP, TD_CLIP)

                q_tot = mixer(agent_qs_taken, state_b).squeeze(1)
                td_errors = (q_tot - td_target).detach().abs().cpu().numpy()
                loss = (weights_b * F.mse_loss(q_tot, td_target, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP:
                    grad_norm = torch.nn.utils.clip_grad_norm_(mixer_params if agents_frozen else agent_params + mixer_params, GRAD_CLIP)
                    grad_norms.append(grad_norm.item())
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                grad_steps += 1

                if grad_steps % TARGET_UPDATE_FREQ == 0:
                    for tnet, net in zip(target_agent_nets, agent_nets):
                        tnet.load_state_dict(net.state_dict())
                    target_mixer.load_state_dict(mixer.state_dict())

                replay.update_priorities(indices, td_errors + 1e-6)

            avg_loss = total_loss / n_updates
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        else:
            avg_loss = 0.0
            avg_grad_norm = 0.0

        # === LOGGING ===
        action_dist = {act: action_counter[act] / max(sum(action_counter.values()), 1) for act in range(ACTION_DIM)}
        print(f"\n[Epoch {epoch:03d}/{EPOCHS}] eps={eps:.3f} | loss={avg_loss:.4f} | grad_norm={avg_grad_norm:.3f}")
        print(f"  Buffer: {len(replay)}/{REPLAY_SIZE} | Actions: {dict(sorted(action_dist.items()))}")
        if agents_frozen:
            print(f"  🔒 Agents FROZEN (mixer-only)")

        # === EVALUATION ===
        if epoch % 5 == 0 or epoch == EPOCHS:
            eval_rewards = []
            for _ in range(5):
                env_eval = K8sAutoscaleEnv(ENV_CONFIG)
                obs_dict, _ = env_eval.reset()
                obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                done, total_r = False, 0.0
                while not done:
                    actions = {}
                    for i, aid in enumerate(agent_ids):
                        with torch.no_grad():
                            q = agent_nets[i](torch.tensor(obs_arr[i:i+1], dtype=torch.float32, device=device))
                        actions[aid] = int(q.argmax(dim=1).item())
                    next_obs_dict, rewards, terms, _, _ = env_eval.step(actions)
                    r = float(list(rewards.values())[0]) if isinstance(rewards, dict) else float(rewards)
                    total_r += float(np.clip(r / REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
                    obs_arr = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                    done = any(terms.values()) if isinstance(terms, dict) else bool(terms)
                eval_rewards.append(total_r)
                last_actions = {aid: int(actions[aid]) + 1 for aid in agent_ids}

            mean_eval = float(np.mean(eval_rewards))
            std_eval = float(np.std(eval_rewards))
            print(f"  📊 Eval: {mean_eval:.2f} ± {std_eval:.2f} | Actions: {last_actions}")

            if mean_eval > best_eval:
                best_eval = mean_eval
                print(f"  ⭐ NEW BEST! (prev: {prev_best_actions})")
                prev_best_actions = last_actions
                save_dict = {
                    "agent_nets": [net.state_dict() for net in agent_nets],
                    "mixer": mixer.state_dict(),
                    "epoch": epoch,
                    "eval": mean_eval
                }
                for i, aid in enumerate(agent_ids):
                    save_dict[f"agent_{aid}"] = agent_nets[i].state_dict()
                torch.save(save_dict, os.path.join(SAVE_DIR, f"qmix_best_epoch{epoch}.pth"))

    torch.save({"agent_nets": [net.state_dict() for net in agent_nets], "mixer": mixer.state_dict()}, 
               os.path.join(SAVE_DIR, "qmix_final.pth"))
    print(f"\n✅ Training complete! Best eval: {best_eval:.2f}")
    vec_env.close()

if __name__ == "__main__":
    main()