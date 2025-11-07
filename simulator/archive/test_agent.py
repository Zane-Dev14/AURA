import torch
import torch.nn as nn
import numpy as np
import sys, os
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
class HPAPolicy:
    """
    Mimics Kubernetes HPA behavior.
    Only uses CPU utilization (like real HPA) for fair comparison.
    """
    def __init__(self, target_cpu=0.7, scale_up_threshold=0.8, scale_down_threshold=0.5):
        self.target_cpu = target_cpu
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.last_actions = {}
        
    def get_actions(self, observations):
        actions = {}
        
        for agent, obs in observations.items():
            # obs[0] is cpu_util normalized (divided by 2.0 in simulator)
            cpu_util = obs[0] * 2.0  # Denormalize
            
            # obs[9] is replicas_desired / 20.0
            current_replicas = int(obs[9] * 20.0)
            
            if agent not in self.last_actions:
                self.last_actions[agent] = max(1, current_replicas)
            
            desired = self.last_actions[agent]
            
            # HPA logic: scale based on CPU only
            if cpu_util > self.scale_up_threshold:
                if cpu_util > 1.2:
                    desired = min(desired + 2, 10)
                else:
                    desired = min(desired + 1, 10)
            elif cpu_util < self.scale_down_threshold:
                desired = max(desired - 1, 1)
            
            # Convert to action space (0-9)
            actions[agent] = max(0, min(desired - 1, 9))
            self.last_actions[agent] = desired
            
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
    
    for agent in agents:
        path = f"../trained_agents/{agent}_best.pth"
        try:
            model = QNetwork().to("cpu")
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            models[agent] = model
            print(f"✅ Loaded {agent} from {path}")
        except Exception as e:
            print(f"⚠️  Could not load {agent}: {e}")
    
    return models if models else None


def load_qmix():
    """Load QMIX trained models from the 'qmix_trained' directory"""
    agents = ["api", "app", "db"]  # Or all agents from your env
    models = {}
    
    try:
        for agent in agents:
            # Correct path based on your 'ls' output
            path = f"../qmix_trained/{agent}_actor_best.pth" 
            
            if not os.path.exists(path):
                print(f"⚠️  Missing QMIX file for agent: {agent} at {path}")
                return None # Fail if any agent is missing

            model = QNetwork().to("cpu")
            
            # Load the state dict directly from the individual file
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            models[agent] = model
        
        print(f"✅ Loaded QMIX models from ../qmix_trained/")
        return models
        
    except Exception as e:
        print(f"⚠️  Could not load QMIX models: {e}")
        return None


def agent_actions(models, observations):
    """Get actions from trained model(s)"""
    actions = {}
    
    # Handle single model case
    if isinstance(models, nn.Module):
        for agent, obs in observations.items():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = models(obs_t)
                action = torch.argmax(logits, dim=1).item()
            actions[agent] = action
    # Handle multi-agent dict case
    else:
        for agent, obs in observations.items():
            if agent in models:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = models[agent](obs_t)
                    action = torch.argmax(logits, dim=1).item()
                actions[agent] = action
            else:
                actions[agent] = 2  # default
    
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

        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        reward = list(rewards.values())[0]
        done = list(terminateds.values())[0]

        # Extract metrics
        first_agent = list(infos.keys())[0]
        sim_info = infos[first_agent]

        total_reward += reward
        total_cost += sim_info['total_cost']
        total_sla_violations += sim_info['sla_violations']
        
        # Collect latencies
        for svc, m in sim_info['metrics'].items():
            latencies.append(m['p95_ms'])

        if verbose:
            print(f"Step {step:03d}: Reward={reward:.3f}  "
                  f"SLA={sim_info['sla_violations']}  "
                  f"Cost={sim_info['total_cost']:.4f}")

            if step % 10 == 0:
                print("--- Metrics snapshot ---")
                for name, m in sim_info["metrics"].items():
                    print(f"  {name}: p95={m['p95_ms']:.0f}ms "
                          f"ready={m['ready_replicas']}/{m['desired_replicas']} "
                          f"queue={m['queue']:.1f}")
                print()

    avg_latency = np.mean(latencies) if latencies else 0
    
    print(f"\n📊 Episode Summary:")
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
    
    # Aggregate
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
    print(f"  Avg P95 Latency: {np.mean(latencies):.1f} ± {np.std(latencies):.1f}ms")
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_cost": np.mean(costs),
        "std_cost": np.std(costs),
        "mean_sla": np.mean(slas),
        "mean_latency": np.mean(latencies)
    }


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and compare autoscaling policies")
    parser.add_argument("--mode", type=str, default="compare", 
                       choices=["single", "compare", "all"],
                       help="single: run one episode; compare: compare all policies; all: detailed multi-episode comparison")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes for comparison mode")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # ---------- SINGLE EPISODE MODE ----------
        print("\n" + "="*70)
        print("SINGLE EPISODE MODE")
        print("="*70)
        
        # Run Random
        run_episode("BASELINE: RANDOM ACTIONS", random_actions, verbose=True)
        
        # Run HPA
        hpa = HPAPolicy()
        run_episode("HPA (K8s Horizontal Pod Autoscaler)", hpa_actions, hpa, seed=0, verbose=True)
        
        # Run Trained Agent
        model = load_agent()
        run_episode("TRAINED AGENT (Single-Agent DQN)", agent_actions, model, seed=0, verbose=True)
    
    elif args.mode == "compare":
        # ---------- QUICK COMPARISON MODE ----------
        print("\n" + "="*70)
        print("QUICK COMPARISON MODE (1 episode each)")
        print("="*70)
        
        results = {}
        
        # HPA
        hpa = HPAPolicy()
        results["HPA"] = run_episode("HPA", hpa_actions, hpa, seed=0, verbose=False)
        
        # Random
        results["Random"] = run_episode("Random", random_actions, seed=0, verbose=False)
        
        # Single-Agent DQN (single model for all agents)
        model = load_agent()
        results["Single-Agent DQN"] = run_episode("Single-Agent DQN", agent_actions, model, seed=0, verbose=False)
        
        # Per-Agent DQN (if available)
        all_agents = load_all_agents()
        if all_agents:
            results["Per-Agent DQN"] = run_episode("Per-Agent DQN", agent_actions, all_agents, seed=0, verbose=False)
        
        # QMIX (if available)
        qmix = load_qmix()
        if qmix:
            results["QMIX"] = run_episode("QMIX", agent_actions, qmix, seed=0, verbose=False)
        
        # Print comparison table
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(f"{'Policy':<20} {'Reward':<12} {'Cost':<12} {'SLA Viol':<10} {'P95 Lat'}")
        print("-"*70)
        
        for name, res in results.items():
            print(f"{name:<20} {res['reward']:>8.2f}    ${res['cost']:>7.4f}    {res['sla_violations']:>4}      {res['avg_latency']:>6.1f}ms")
        
        # Compute improvements vs HPA
        if "HPA" in results:
            hpa_cost = results["HPA"]["cost"]
            hpa_sla = results["HPA"]["sla_violations"]
            
            print("\n" + "="*70)
            print("IMPROVEMENT vs HPA")
            print("="*70)
            
            for name, res in results.items():
                if name == "HPA":
                    continue
                cost_improvement = (1 - res["cost"] / hpa_cost) * 100 if hpa_cost > 0 else 0
                sla_improvement = (hpa_sla - res["sla_violations"])
                
                print(f"{name}:")
                print(f"  Cost: {cost_improvement:+.1f}%")
                print(f"  SLA Violations: {sla_improvement:+.0f} (fewer)")
    
    else:  # args.mode == "all"
        # ---------- COMPREHENSIVE COMPARISON MODE ----------
        print("\n" + "="*70)
        print(f"COMPREHENSIVE COMPARISON ({args.episodes} episodes each)")
        print("="*70)
        
        all_results = {}
        
        # HPA
        hpa = HPAPolicy()
        all_results["HPA"] = run_multiple_episodes("HPA", hpa_actions, hpa, n_episodes=args.episodes)
        
        # Single-Agent DQN
        model = load_agent()
        all_results["Single-Agent DQN"] = run_multiple_episodes("Single-Agent DQN", agent_actions, model, n_episodes=args.episodes)
        
        # Per-Agent DQN
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