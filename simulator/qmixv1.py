#!/usr/bin/env python3
"""
QMIX Trainer (fixed & ready)

- Per-agent warm-start (tries ./trained_agents/{agent}_best.pth first)
- Mixer-first curriculum (freeze agent nets for initial epochs)
- Double-DQN selection for targets (online argmax, target evaluation)
- Reward/TD clipping, gradient clipping, numeric sanitization
- Saves both qmix_best.pth and per-agent snapshots for reuse
"""
import os
import time
import random
from collections import deque, namedtuple

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from boutique_env import K8sAutoscaleEnv  # your PettingZoo ParallelEnv wrapper

# ----------------------------
# Hyperparameters (tweakable)
# ----------------------------
ENV_CONFIG = "config.yaml"
SEED = 42

OBS_DIM_PER_AGENT = 16
ACTION_DIM = 10   # 0..9 -> replicas 1..10

LR = 5e-4
BATCH_SIZE = 512
GAMMA = 0.98
REPLAY_SIZE = 400_000
MIN_REPLAY_SIZE = 5_000
EPOCHS = 400
STEPS_PER_EPOCH = 2000

EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY_EPOCHS = 200

TARGET_UPDATE_FREQ = 200
MIXING_HIDDEN = 32
HYPERNET_HIDDEN = 64

GRAD_CLIP = 5.0
TD_CLIP = 1e3        # clip TD targets for stability

MIXER_ONLY_EPOCHS = 20   # freeze agent nets for these epochs so mixer learns first

# Warm-start paths
DQN_AGENT_DIR = "./trained_agents"  # where single-agent DQN saved {agent}_best.pth
ALTERNATE_SINGLE_FILE = "agent.pth" # fallback single-file checkpoint
SAVE_DIR = "./qmix_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Simple joint replay buffer
# ----------------------------
Transition = namedtuple("Transition", [
    "obs", "state", "actions", "reward", "next_obs", "next_state", "done"
])

class JointReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size:int):
        idx = random.sample(range(len(self.buf)), batch_size)
        batch = [self.buf[i] for i in idx]
        obs = np.stack([b.obs for b in batch]).astype(np.float32)         # (B, n, obs)
        state = np.stack([b.state for b in batch]).astype(np.float32)     # (B, state_dim)
        actions = np.stack([b.actions for b in batch]).astype(np.int64)   # (B, n)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)   # (B,)
        next_obs = np.stack([b.next_obs for b in batch]).astype(np.float32)
        next_state = np.stack([b.next_state for b in batch]).astype(np.float32)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return dict(obs=obs, state=state, actions=actions, rewards=rewards,
                    next_obs=next_obs, next_state=next_state, dones=dones)

# ----------------------------
# Agent net + mixer
# ----------------------------
class AgentNet(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
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
    def __init__(self, n_agents:int, state_dim:int, mixing_hidden=MIXING_HIDDEN, hyper_hidden=HYPERNET_HIDDEN):
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
# Helpers
# ----------------------------
def sanitize_array(a: np.ndarray, clip=1e6) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(a, -clip, clip).astype(np.float32)

# ----------------------------
# Environment wrapper helpers
# ----------------------------
class K8sAutoscaleEnvFixed(K8sAutoscaleEnv):
    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        # ensure consistent deterministic ordering of agents
        self.agent_ids = list(self.simulator.services.keys())
    def get_global_state(self):
        # delegate to simulator if available
        # here, concatenate per-agent observations (keeps same as boutique_env.get_global_state)
        obs = {a: self.simulator.services[a].get_observation(self.simulator.services) for a in self.agent_ids}
        return np.concatenate([obs[a] for a in self.agent_ids], axis=0).astype(np.float32)
    def reset_with_state(self, seed=None, options=None):
        obs, infos = self.reset(seed=seed, options=options)
        state = self.get_global_state()
        return obs, infos, state
    def step_with_state(self, actions: Dict[str,int]):
        obs, rewards, terminateds, truncateds, infos = self.step(actions)
        # ensure obs has entries for all agents
        all_obs = {}
        for a in self.agent_ids:
            all_obs[a] = obs.get(a, np.zeros(OBS_DIM_PER_AGENT, dtype=np.float32))
        state = self.get_global_state()
        return all_obs, rewards, terminateds, truncateds, infos, state

# ----------------------------
# Training
# ----------------------------
def build_env_and_dims():
    env = K8sAutoscaleEnvFixed(ENV_CONFIG)
    agent_ids = list(env.possible_agents)
    n_agents = len(agent_ids)
    # get state dim
    _obs, _infos = env.reset()
    state = env.get_global_state()
    state_dim = int(state.size)
    obs_dim = int(env.observation_space(agent_ids[0]).shape[0])
    return env, agent_ids, n_agents, state_dim, obs_dim

def try_load_per_agent_checkpoints(agent_nets: List[nn.Module], agent_ids: List[str]) -> int:
    """
    Tries to load per-agent checkpoints from DQN_AGENT_DIR/{agent}_best.pth
    Returns number of agents successfully loaded.
    If that fails, attempts to load an alternate single-file checkpoint (agent.pth) that may contain keys.
    """
    loaded = 0
    # first try per-agent files
    for i, aid in enumerate(agent_ids):
        fn = os.path.join(DQN_AGENT_DIR, f"{aid}_best.pth")
        if os.path.exists(fn):
            try:
                ck = torch.load(fn, map_location=device)
                # ck might be a full state_dict or raw weights; try both
                if isinstance(ck, dict) and all(k in ck for k in agent_nets[i].state_dict().keys()):
                    agent_nets[i].load_state_dict(ck, strict=True)
                elif "state_dict" in ck and isinstance(ck["state_dict"], dict):
                    agent_nets[i].load_state_dict(ck["state_dict"], strict=False)
                else:
                    # assume ck is state_dict
                    agent_nets[i].load_state_dict(ck, strict=False)
                print(f"Loaded per-agent checkpoint for {aid} from {fn}")
                loaded += 1
            except Exception as e:
                print(f"Failed loading {fn} into {aid}: {e}")
    if loaded > 0:
        return loaded
    # fallback: try single-file checkpoint that might contain agent keys
    single = os.path.join(DQN_AGENT_DIR, ALTERNATE_SINGLE_FILE) if os.path.exists(os.path.join(DQN_AGENT_DIR, ALTERNATE_SINGLE_FILE)) else ALTERNATE_SINGLE_FILE
    if os.path.exists(single):
        try:
            ck = torch.load(single, map_location=device)
            # if ck is dict of agent_x keys
            if isinstance(ck, dict):
                for i, aid in enumerate(agent_ids):
                    # common keys: "agent_{aid}" or f"{aid}" or first entry
                    if f"agent_{aid}" in ck:
                        try:
                            agent_nets[i].load_state_dict(ck[f"agent_{aid}"], strict=False)
                            loaded += 1
                            print(f"Loaded {aid} from {single}['agent_{aid}']")
                        except Exception:
                            pass
                    elif aid in ck:
                        try:
                            agent_nets[i].load_state_dict(ck[aid], strict=False)
                            loaded += 1
                            print(f"Loaded {aid} from {single}['{aid}']")
                        except Exception:
                            pass
                # If ck may be a single-state dict intended for identical agents, attempt to load same dict into all
                if loaded == 0:
                    try:
                        # try using ck as state_dict for all agent nets
                        for i in range(len(agent_nets)):
                            agent_nets[i].load_state_dict(ck, strict=False)
                        loaded = len(agent_nets)
                        print(f"Loaded single-file checkpoint into all agents from {single}")
                    except Exception as e:
                        print(f"Could not apply single-file checkpoint broadly: {e}")
            else:
                print(f"Single-file checkpoint {single} not recognized format.")
        except Exception as e:
            print(f"Failed loading fallback single-file checkpoint {single}: {e}")
    return loaded

def main():
    env, agent_ids, n_agents, state_dim, obs_dim = build_env_and_dims()
    print(f"Agents detected: {agent_ids} | n_agents={n_agents} | state_dim={state_dim} | obs_dim={obs_dim}")

    # build networks
    agent_nets = [AgentNet(obs_dim, ACTION_DIM).to(device) for _ in range(n_agents)]
    target_agent_nets = [AgentNet(obs_dim, ACTION_DIM).to(device) for _ in range(n_agents)]
    for t, s in zip(target_agent_nets, agent_nets):
        t.load_state_dict(s.state_dict())

    mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    # try warm-start from single-agent DQN checkpoints
    loaded = try_load_per_agent_checkpoints(agent_nets, agent_ids)
    if loaded > 0:
        # copy weights to targets
        for i in range(n_agents):
            target_agent_nets[i].load_state_dict(agent_nets[i].state_dict())
        print(f"Warm-started {loaded}/{n_agents} agents from DQN checkpoints.")
    else:
        print("No per-agent warm-start found. Training from scratch.")

    # parameters & optimizer: initially only mixer params (mixer-first curriculum)
    mixer_params = list(mixer.parameters())
    agent_params = []
    for net in agent_nets:
        agent_params += list(net.parameters())

    # start with optimizer only on mixer if MIXER_ONLY_EPOCHS > 0 and we have warm-start
    use_mixer_only = MIXER_ONLY_EPOCHS > 0 and loaded > 0
    if use_mixer_only:
        print(f"Using mixer-only curriculum for first {MIXER_ONLY_EPOCHS} epochs (agents frozen).")
        optimizer = torch.optim.Adam(mixer_params, lr=LR)
    else:
        optimizer = torch.optim.Adam(agent_params + mixer_params, lr=LR)

    buffer = JointReplayBuffer(REPLAY_SIZE)

    def epsilon_for_epoch(ep:int):
        t = min(ep / max(1, EPS_DECAY_EPOCHS), 1.0)
        return EPS_START + (EPS_END - EPS_START) * t

    gradient_steps = 0
    best_eval = -1e18

    print("Starting training loop...")
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        eps = epsilon_for_epoch(epoch)

        # --- Collect STEPS_PER_EPOCH environment steps ---
        steps_collected = 0
        while steps_collected < STEPS_PER_EPOCH:
            obs_dict, infos, state = env.reset_with_state()
            obs_arr = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
            done = False
            while not done and steps_collected < STEPS_PER_EPOCH:
                # select joint actions (epsilon-greedy)
                actions = {}
                action_list = []
                for i, aid in enumerate(agent_ids):
                    obs_i = torch.tensor(obs_arr[i:i+1], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        q = agent_nets[i](obs_i)  # (1, A)
                    if random.random() < eps:
                        act = random.randrange(ACTION_DIM)
                    else:
                        act = int(q.argmax(dim=1).item())
                    actions[aid] = act
                    action_list.append(act)

                next_obs_dict, rewards, terminateds, truncateds, infos, next_state = env.step_with_state(actions)
                next_obs_arr = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)

                # shared reward (sim returns a single float in rewards values)
                if isinstance(rewards, dict):
                    r = float(list(rewards.values())[0])
                else:
                    r = float(rewards)
                r = float(np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6))
                r = float(np.clip(r, -1e6, 1e6))

                done_flag = any(terminateds.values()) if isinstance(terminateds, dict) else bool(terminateds)

                buffer.push(sanitize_array(obs_arr), state.astype(np.float32),
                            np.array(action_list, dtype=np.int64), r,
                            sanitize_array(next_obs_arr), next_state.astype(np.float32), done_flag)
                steps_collected += 1

                obs_arr = next_obs_arr
                state = next_state
                if done_flag:
                    break

        # If using mixer-only and we're at the point to unfreeze, switch optimizer
        if use_mixer_only and epoch == (MIXER_ONLY_EPOCHS + 1):
            # re-create optimizer to include all params
            for net in agent_nets:
                for p in net.parameters():
                    p.requires_grad = True
            optimizer = torch.optim.Adam(agent_params + mixer_params, lr=LR)
            use_mixer_only = False
            print("Unfroze agent nets; optimizer now updates agent + mixer parameters.")

        # --- Perform gradient updates ---
        if len(buffer) >= MIN_REPLAY_SIZE:
            n_grad_steps = max(1, int(STEPS_PER_EPOCH * 0.1))
            total_loss = 0.0
            for _ in range(n_grad_steps):
                batch = buffer.sample(BATCH_SIZE)
                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)        # (B,n,obs)
                state_b = torch.tensor(batch["state"], dtype=torch.float32, device=device)    # (B, state)
                actions_b = torch.tensor(batch["actions"], dtype=torch.long, device=device)   # (B,n)
                rewards_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)# (B,)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                next_state_b = torch.tensor(batch["next_state"], dtype=torch.float32, device=device)
                dones_b = torch.tensor(batch["dones"], dtype=torch.float32, device=device)

                B = obs_b.size(0)

                # Q(s,a) for taken actions
                agent_qs_taken = []
                for i in range(n_agents := len(agent_ids)):
                    obs_i = obs_b[:, i, :]
                    q_i = agent_nets[i](obs_i)                     # (B, A)
                    a_i = actions_b[:, i].unsqueeze(1)             # (B,1)
                    q_taken = q_i.gather(1, a_i).squeeze(1)        # (B,)
                    agent_qs_taken.append(q_taken)
                agent_qs_taken = torch.stack(agent_qs_taken, dim=1)   # (B, n_agents)

                # Double-DQN target: online argmax, target evaluation
                with torch.no_grad():
                    agent_qs_next_online = []
                    for i in range(n_agents):
                        next_obs_i = next_obs_b[:, i, :]
                        q_online = agent_nets[i](next_obs_i)         # online nets
                        agent_qs_next_online.append(q_online)
                    # compute argmax actions from online
                    next_actions = [q.argmax(dim=1) for q in agent_qs_next_online]   # list of (B,)
                    # evaluate those actions with target nets
                    agent_qs_next_eval = []
                    for i in range(n_agents):
                        q_target_next = target_agent_nets[i](next_obs_b[:, i, :])     # (B, A)
                        a_i = next_actions[i].unsqueeze(1)
                        q_eval = q_target_next.gather(1, a_i).squeeze(1)             # (B,)
                        agent_qs_next_eval.append(q_eval)
                    agent_qs_next_eval = torch.stack(agent_qs_next_eval, dim=1)     # (B, n_agents)
                    q_tot_next = target_mixer(agent_qs_next_eval, next_state_b).squeeze(1)  # (B,)
                    td_target = rewards_b + (1.0 - dones_b) * (GAMMA * q_tot_next)
                    # clip td_target
                    td_target = torch.clamp(td_target, -TD_CLIP, TD_CLIP)

                q_tot = mixer(agent_qs_taken, state_b).squeeze(1)   # (B,)
                loss = F.mse_loss(q_tot, td_target)

                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(list(mixer.parameters()) + agent_params, GRAD_CLIP)
                else:
                    # ensure gradients are bounded
                    torch.nn.utils.clip_grad_norm_(list(mixer.parameters()) + agent_params, 5.0)
                optimizer.step()

                total_loss += loss.item()
                gradient_steps += 1

                if gradient_steps % TARGET_UPDATE_FREQ == 0:
                    for tnet, net in zip(target_agent_nets, agent_nets):
                        tnet.load_state_dict(net.state_dict())
                    target_mixer.load_state_dict(mixer.state_dict())

            avg_loss = total_loss / max(1, n_grad_steps)
        else:
            avg_loss = 0.0

        epoch_time = time.time() - epoch_start

        # --- Evaluation (greedy) ---
        if epoch % 5 == 0 or epoch == EPOCHS:
            eval_episodes = 5
            rews = []
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
                            q = agent_nets[i](obs_i)
                            act = int(q.argmax(dim=1).item())
                        actions[aid] = act
                    next_obs_dict, rewards, terminateds, truncateds, infos, next_state = env.step_with_state(actions)
                    if isinstance(rewards, dict):
                        r = float(list(rewards.values())[0])
                    else:
                        r = float(rewards)
                    total_r += r
                    obs_arr = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                    done = any(terminateds.values()) if isinstance(terminateds, dict) else bool(terminateds)
                rews.append(total_r)
            mean_eval = float(np.mean(rews))
            std_eval = float(np.std(rews))
            print(f"[Epoch {epoch:03d} | {epoch_time:.1f}s] eval={mean_eval:.2f}±{std_eval:.2f} | buffer={len(buffer)} | loss={avg_loss:.4f}")

            if mean_eval > best_eval:
                best_eval = mean_eval
                save_path = os.path.join(SAVE_DIR, "qmix_best.pth")
                save_dict = {
                    "agent_nets": [net.state_dict() for net in agent_nets],
                    "mixer": mixer.state_dict(),
                    "epoch": epoch,
                    "eval": mean_eval
                }
                # also save per-agent snapshots usable by DQN warm-start
                for i, aid in enumerate(agent_ids):
                    save_dict[f"agent_{aid}"] = agent_nets[i].state_dict()
                    torch.save(agent_nets[i].state_dict(), os.path.join(DQN_AGENT_DIR, f"{aid}_best.pth"))
                torch.save(save_dict, save_path)
                print(f"Saved best QMIX checkpoint to {save_path} (eval={mean_eval:.2f})")

        # periodic status
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} completed in {epoch_time:.1f}s | avg_loss={avg_loss:.4f} | buffer={len(buffer)}")

    # final save
    final_path = os.path.join(SAVE_DIR, "qmix_final.pth")
    final_dict = {
        "agent_nets": [net.state_dict() for net in agent_nets],
        "mixer": mixer.state_dict()
    }
    for i, aid in enumerate(agent_ids):
        final_dict[f"agent_{aid}"] = agent_nets[i].state_dict()
    torch.save(final_dict, final_path)
    print("Training finished. Final saved to:", final_path)

if __name__ == "__main__":
    main()
