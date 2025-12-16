import torch
import torch.nn as nn
import numpy as np
import sys, os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
# HPA Policy (K8s Horizontal Pod Autoscaler baseline)
# =====================================================
from collections import defaultdict, deque

class HPAPolicy:
    def __init__(self,
                 target_cpu=0.7,
                 tolerance=0.1,
                 scale_down_stabilization=300,
                 min_replicas=1,
                 max_replicas=10):
        
        self.target_cpu = target_cpu
        self.tolerance = tolerance
        self.scale_down_stabilization = scale_down_stabilization
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        
        self.history = defaultdict(lambda: deque(maxlen=50))
    
    def get_actions(self, observations):
        now = time.time()
        actions = {}
        
        for agent, obs in observations.items():
            # Extract metrics
            cpu_util = obs[0] * 2.0
            current = int(round(obs[9] * 20.0))
            
            # Step 1: Apply tolerance
            ratio = cpu_util / self.target_cpu
            if abs(1 - ratio) < self.tolerance:
                desired = current
            else:
                desired = int(np.ceil(current * ratio))
            
            # Clamp to allowed range
            desired = int(np.clip(desired, self.min_replicas, self.max_replicas))
            
            # Step 2: Stabilization window
            hist = self.history[agent]
            hist.append((now, desired))
            
            # Filter to window
            cutoff = now - self.scale_down_stabilization
            recent = [d for t, d in hist if t >= cutoff]
            
            if recent:
                if desired < current:
                    # Scale-down: use MAX from window
                    desired = max(recent)
                # Scale-up: immediate (0s stabilization)
            
            # Final clamp
            desired = int(np.clip(desired, self.min_replicas, self.max_replicas))
            
            # Convert to action space [0-9] representing [1-10] replicas
            actions[agent] = desired - 1
        
        return actions
# =====================================================
# Policy Functions
# =====================================================
def random_actions(observations):
    actions = {}
    for agent in observations.keys():
        actions[agent] = np.random.randint(0, 10)
    return actions


def hpa_actions(hpa_policy, observations):
    return hpa_policy.get_actions(observations)


def load_agent(path="../agent.pth"):
    """Load single trained agent"""
    model = QNetwork().to("cpu")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"✅ Loaded agent from: {path}")
    return model


def load_all_agents():
    """Load all per-agent trained models"""
    agents = ["api", "app", "db"]
    models = {}
    
    base_path = "../trained_agents"
    if not os.path.exists(base_path):
        print(f"⚠️  Could not find directory: {base_path}. Skipping Per-Agent DQN.")
        return None

    for agent in agents:
        path = f"{base_path}/{agent}_best.pth"
        try:
            model = QNetwork().to("cpu")
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            models[agent] = model
            print(f"✅ Loaded {agent} from {path}")
        except Exception as e:
            print(f"⚠️  Could not load {agent} from {path}: {e}")
            print("⚠️  Skipping Per-Agent DQN test as not all models are present.")
            return None
    
    return models


def load_qmix():
    """Load QMIX trained models from simulator/qmix_checkpoints/qmix_final.pth"""
    qmix_path = os.path.join("..", "qmix_checkpoints", "qmix_final.pth")

    if not os.path.exists(qmix_path):
        print(f"⚠️  QMIX checkpoint not found at {qmix_path}")
        return None

    try:
        checkpoint = torch.load(qmix_path, map_location="cpu")

        # Load agent networks (stored as list)
        agent_nets_state = checkpoint.get("agent_nets", None)
        if agent_nets_state is None:
            print("⚠️  No agent_nets found in checkpoint.")
            return None

        agents = ["api", "app", "db"]
        models = {}
        for i, agent in enumerate(agents):
            model = QNetwork().to("cpu")
            model.load_state_dict(agent_nets_state[i])
            model.eval()
            models[agent] = model

        print(f"✅ Loaded QMIX joint checkpoint from {qmix_path}")
        return models

    except Exception as e:
        print(f"⚠️  Could not load QMIX checkpoint: {e}")
        return None


def agent_actions(models, observations):
    """Get actions from trained model(s)"""
    actions = {}
    
    if isinstance(models, nn.Module):
        for agent, obs in observations.items():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = models(obs_t)
                action = torch.argmax(logits, dim=1).item()
            actions[agent] = action
    else:
        for agent, obs in observations.items():
            if agent in models:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = models[agent](obs_t)
                    action = torch.argmax(logits, dim=1).item()
                actions[agent] = action
            else:
                actions[agent] = 2
    
    return actions


# =====================================================
# Evaluation Runner
# =====================================================
def run_episode(title, policy_fn, policy_obj=None, seed=0, verbose=False):
    """Run single episode and collect metrics"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")
    
    env = K8sAutoscaleEnv("../config.yaml")
    observations, infos = env.reset(seed=seed)

    total_reward = 0
    total_cost = 0
    total_sla_violations = 0
    latencies = []
    done = False
    step = 0

    while not done:
        step += 1

        # Get actions
        if policy_obj is None:
            actions = policy_fn(observations)
        else:
            actions = policy_fn(policy_obj, observations)

        # --- VERBOSE PRINTING ---
        if verbose:
            # Actions are 0-9, so +1 for replicas 1-10
            action_replicas = {a: act + 1 for a, act in actions.items()}
            print(f"\n--- Step {step:03d} ---")
            print(f"  Actions Taken: {action_replicas}")

        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        reward = list(rewards.values())[0]
        done = list(terminateds.values())[0]

        first_agent = list(infos.keys())[0]
        sim_info = infos[first_agent]

        total_reward += reward
        total_cost += sim_info['total_cost']
        total_sla_violations += sim_info['sla_violations']
        
        for svc, m in sim_info['metrics'].items():
            latencies.append(m['p95_ms'])

        if verbose:
            print(f"  Reward this step: {reward:.3f}")
            print(f"  SLA Violations this step: {sim_info['sla_violations']}")
            print("  --- Metrics Snapshot ---")
            for name, m in sim_info["metrics"].items():
                print(f"    {name:<10}: "
                      f"P95={m['p95_ms']:.0f}ms, "
                      f"Ready={m['ready_replicas']}/{m['desired_replicas']}, "
                      f"Queue={m['queue']:.1f}, "
                      f"CPU={(observations[name][0] * 2.0):.2f}") # De-normalize CPU
            
    avg_latency = np.mean(latencies) if latencies else 0
    
    print(f"\n{'='*70}")
    print(f"📊 EPISODE SUMMARY: {title}")
    print(f"{'='*70}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Total Cost: ${total_cost:.4f}")
    print(f"   SLA Violations: {total_sla_violations}")
    print(f"   Avg P95 Latency: {avg_latency:.1f}ms")
    
    return {
        "reward": total_reward,
        "cost": total_cost,
        "sla_violations": total_sla_violations,
        "avg_latency": avg_latency
    }


def run_multiple_episodes(title, policy_fn, policy_obj=None, n_episodes=5):
    """Run multiple episodes and aggregate results"""
    print(f"\n{'='*70}")
    print(f"RUNNING {n_episodes} EPISODES: {title}")
    print(f"{'='*70}")
    
    results = []
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep+1}/{n_episodes} ---")
        result = run_episode(f"{title} (ep {ep+1})", policy_fn, policy_obj, seed=ep, verbose=False)
        results.append(result)
    
    rewards = [r["reward"] for r in results]
    costs = [r["cost"] for r in results]
    slas = [r["sla_violations"] for r in results]
    latencies = [r["avg_latency"] for r in results]
    
    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS: {title}")
    print(f"{'='*70}")
    print(f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Cost: ${np.mean(costs):.4f} ± {np.std(costs):.4f}")
    print(f"  SLA Violations: {np.mean(slas):.1f} ± {np.std(slas):.1f}")
    print(f"  Avg P9G Latency: {np.mean(latencies):.1f} ± {np.std(latencies):.1f}ms")
    
    return {
        "title": title,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_cost": np.mean(costs),
        "std_cost": np.std(costs),
        "mean_sla": np.mean(slas),
        "std_sla": np.std(slas),
        "mean_latency": np.mean(latencies),
        "std_latency": np.std(latencies),
        "all_rewards": rewards,
        "all_costs": costs,
        "all_slas": slas,
        "all_latencies": latencies
    }


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and compare autoscaling policies")
    parser.add_argument("--mode", type=str, default="single",  # <-- CHANGED DEFAULT
                       choices=["single", "compare", "all"],
                       help="single: run one episode; compare: compare all policies; all: detailed multi-episode comparison")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes for comparison mode")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # ---------- SINGLE EPISODE MODE (VERBOSE) ----------
        print("\n" + "="*70)
        print("SINGLE EPISODE VERBOSE COMPARISON")
        print("="*70)
        
        # Run HPA
        hpa = HPAPolicy()
        run_episode("HPA (K8s Horizontal Pod Autoscaler)", hpa_actions, hpa, seed=0, verbose=True)
        
        # Run Per-Agent DQN (The Winner)
        all_agents = load_all_agents()
        if all_agents:
            run_episode("Per-Agent DQN (The Winner)", agent_actions, all_agents, seed=0, verbose=True)
        else:
            print("\nSkipping Per-Agent DQN (models not found in ../trained_agents/).")
            
        # Run QMIX
        qmix = load_qmix()
        if qmix:
            run_episode("QMIX", agent_actions, qmix, seed=0, verbose=True)
        else:
            print("\nSkipping QMIX (models not found in ../qmix_trained/).")
            
    
    elif args.mode == "compare":
        # ---------- QUICK COMPARISON MODE (1 episode each) ----------
        # ... (no changes here) ...
        
        print("\n" + "="*70)
        print("QUICK COMPARISON MODE (1 episode each)")
        print("="*70)
        
        results = {}
        
        # HPA
        hpa = HPAPolicy()
        results["HPA"] = run_episode("HPA", hpa_actions, hpa, seed=0, verbose=False)
        
        # Single-Agent DQN (Alibaba Baseline)
        model = load_agent()
        results["Single-Agent DQN"] = run_episode("Single-Agent DQN", agent_actions, model, seed=0, verbose=False)
        
        # Per-Agent DQN (I-DQN)
        all_agents = load_all_agents()
        if all_agents:
            results["Per-Agent DQN"] = run_episode("Per-Agent DQN", agent_actions, all_agents, seed=0, verbose=False)
        
        # QMIX
        qmix = load_qmix()
        if qmix:
            results["QMIX"] = run_episode("QMIX", agent_actions, qmix, seed=0, verbose=False)
        
        # Print comparison table
        print("\n" + "="*70)
        print("COMPARISON TABLE (1 EPISODE)")
        print("="*70)
        print(f"{'Policy':<20} {'Reward':<12} {'Cost':<12} {'SLA Viol':<10} {'P95 Lat'}")
        print("-"*70)
        
        for name, res in results.items():
            print(f"{name:<20} {res['reward']:>8.2f}    ${res['cost']:>7.4f}    {res['sla_violations']:>4}      {res['avg_latency']:>6.1f}ms")
        
    
    else:  # args.mode == "all"
        # ---------- COMPREHENSIVE COMPARISON MODE ----------
        # ... (no changes here) ...
        
        print("\n" + "="*70)
        print(f"COMPREHENSIVE COMPARISON ({args.episodes} episodes each)")
        print("="*70)
        
        all_results = {}
        
        # HPA
        hpa = HPAPolicy()
        all_results["HPA"] = run_multiple_episodes("HPA", hpa_actions, hpa, n_episodes=args.episodes)
        
        # Single-Agent DQN (Alibaba Baseline)
        model = load_agent()
        all_results["Single-Agent DQN"] = run_multiple_episodes("Single-Agent DQN", agent_actions, model, n_episodes=args.episodes)
        
        # Per-Agent DQN (I-DQN)
        all_agents = load_all_agents()
        if all_agents:
            all_results["Per-Agent DQN"] = run_multiple_episodes("Per-Agent DQN", agent_actions, all_agents, n_episodes=args.episodes)
        
        # QMIX
        qmix = load_qmix()
        if qmix:
            all_results["QMIX"] = run_multiple_episodes("QMIX", agent_actions, qmix, n_episodes=args.episodes)
        
        # Final comparison
        print("\n" + "="*80)
        print("FINAL COMPARISON TABLE")
        print("="*80)
        print(f"{'Policy':<20} {'Reward':<20} {'Cost':<20} {'SLA Viol':<15} {'Latency'}")
        print("-"*80)
        
        for name, res in all_results.items():
            print(f"{name:<20} {res['mean_reward']:>7.2f} ± {res['std_reward']:<7.2f} "
                  f"${res['mean_cost']:>6.4f} ± {res['std_cost']:<7.4f} "
                  f"{res['mean_sla']:>6.1f}       {res['mean_latency']:>7.1f}ms")
        
        # Improvements vs HPA
        if "HPA" in all_results:
            hpa_cost = all_results["HPA"]["mean_cost"]
            hpa_sla = all_results["HPA"]["mean_sla"]
            
            print("\n" + "="*80)
            print("IMPROVEMENT vs HPA BASELINE")
            print("="*80)
            
            for name, res in all_results.items():
                if name == "HPA":
                    continue
                
                cost_improvement = (1 - res["mean_cost"] / hpa_cost) * 100 if hpa_cost > 0 else 0
                sla_improvement = (hpa_sla - res["mean_sla"]) / max(hpa_sla, 1) * 100
                
                print(f"{name}:")
                print(f"  Cost Reduction: {cost_improvement:+.1f}%")
                print(f"  SLA Improvement: {sla_improvement:+.1f}% (fewer violations)")

        # === NEW CODE: SAVE RESULTS TO JSON ===
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        if all_results: # Only save if we ran this mode
            for policy, stats in all_results.items():
                for key, value in stats.items():
                    if isinstance(value, np.ndarray):
                        stats[key] = value.tolist()
            
            try:
                with open(filename, 'w') as f:
                    json.dump(all_results, f, indent=4)
                print("\n" + "="*80)
                print(f"✅ Successfully saved detailed results to {filename}")
                print("="*80)
            except Exception as e:
                print(f"\n⚠️  Failed to save results to JSON: {e}")