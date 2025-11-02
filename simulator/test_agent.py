import torch
import torch.nn as nn
import numpy as np
from boutique_env import K8sAutoscaleEnv


# =====================================================
# QNetwork (same as training)
# =====================================================
class QNetwork(nn.Module):
    def __init__(self, obs_dim=16, action_dim=10):
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


# =====================================================
# Random policy
# =====================================================
def random_actions(observations):
    actions = {}
    for agent in observations.keys():
        actions[agent] = np.random.randint(0, 10)  # random 0-9
    return actions


# =====================================================
# Agent policy
# =====================================================
def load_agent(path="agent.pth"):
    model = QNetwork().to("cpu")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"✅ Loaded agent from: {path}")
    return model


def agent_actions(model, observations):
    actions = {}
    for agent, obs in observations.items():
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs_t)
            action = torch.argmax(logits, dim=1).item()
        actions[agent] = action
    return actions


# =====================================================
# Run 1 rollout
# =====================================================
def run_episode(title, policy_fn, model=None):
    print(f"\n\n===================== {title} =====================\n")
    env = K8sAutoscaleEnv("config.yaml")
    observations, infos = env.reset(seed=0)

    total_reward = 0
    done = False
    step = 0

    while not done:
        step += 1

        if model is None:
            actions = policy_fn(observations)
        else:
            actions = policy_fn(model, observations)

        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        reward = list(rewards.values())[0]
        done = list(terminateds.values())[0]

        # ✅ Correct way: extract simulator info from any agent
        first_agent = list(infos.keys())[0]
        sim_info = infos[first_agent]

        total_reward += reward

        print(f"Step {step:03d}: Reward={reward:.3f}  "
              f"SLA={sim_info['sla_violations']}  "
              f"Cost={sim_info['total_cost']:.4f}")

        if step % 10 == 0:
            print("--- Metrics snapshot ---")
            for name, m in sim_info["metrics"].items():
                print(f"{name}: p95={m['p95_ms']:.0f}ms "
                      f"ready={m['ready_replicas']}/{m['desired_replicas']} "
                      f"queue={m['queue']:.1f}")
            print()

    print(f"\n✅ Episode finished: Total Reward = {total_reward:.3f}")
    return total_reward


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    # ---------- RUN BASELINE ----------
    random_total = run_episode("BASELINE: RANDOM ACTIONS", random_actions)

    # ---------- RUN AGENT ----------
    model = load_agent("agent.pth")
    agent_total = run_episode("TRAINED AGENT", agent_actions, model)

    # ---------- Comparison ----------
    print("\n\n===================== COMPARISON =====================")
    print(f"Random Policy Total Reward: {random_total:.3f}")
    print(f"Agent Policy Total Reward : {agent_total:.3f}")
    print("======================================================\n")
