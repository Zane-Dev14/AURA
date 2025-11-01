
"""
Compare simulator performance: Random vs HPA vs (future) MARL
"""

import numpy as np
from boutique_env import K8sAutoscaleEnv
import matplotlib.pyplot as plt


def run_random_policy(env, num_episodes=10):
    """Random baseline - worst case"""
    print("\n" + "=" * 70)
    print("RANDOM POLICY BASELINE")
    print("=" * 70)
    
    episode_rewards = []
    episode_costs = []
    episode_slas = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total_reward = 0
        steps = 50  # 25 minutes
        
        for step in range(steps):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            total_reward += list(rewards.values())[0]
        
        episode_rewards.append(total_reward / steps)
        episode_costs.append(infos[env.possible_agents[0]]['total_cost'])
        episode_slas.append(infos[env.possible_agents[0]]['sla_violations'])
    
    avg_reward = np.mean(episode_rewards)
    avg_cost = np.mean(episode_costs)
    avg_sla = np.mean(episode_slas)
    
    print(f"\n  Average reward: {avg_reward:.2f}")
    print(f"  Average cost: ${avg_cost:.4f}")
    print(f"  Average SLA violations: {avg_sla:.2f}")
    
    return {'reward': avg_reward, 'cost': avg_cost, 'sla': avg_sla}


def run_hpa_policy(env, target_cpu=0.75, num_episodes=10):
    """HPA baseline - what we want to beat"""
    print("\n" + "=" * 70)
    print(f"HPA POLICY BASELINE (target CPU: {target_cpu*100:.0f}%)")
    print("=" * 70)
    
    episode_rewards = []
    episode_costs = []
    episode_slas = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total_reward = 0
        steps = 50
        
        for step in range(steps):
            # HPA formula: desired = current * (current_cpu / target_cpu)
            actions = {}
            for agent in env.agents:
                cpu_util = obs[agent][0] * 2.0  # Denormalize
                current_replicas = int(obs[agent][10] * 20)
                
                # HPA scaling logic
                if current_replicas == 0:
                    current_replicas = 1
                
                desired = int(current_replicas * (cpu_util / target_cpu))
                desired = max(1, min(desired, 10))
                
                actions[agent] = desired - 1
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_reward += list(rewards.values())[0]
        
        episode_rewards.append(total_reward / steps)
        episode_costs.append(infos[env.possible_agents[0]]['total_cost'])
        episode_slas.append(infos[env.possible_agents[0]]['sla_violations'])
        
        if ep == 0:
            print(f"\n  Episode 1 sample metrics:")
            for agent in env.possible_agents:
                m = infos[env.possible_agents[0]]['metrics'][agent]
                print(f"    {agent}: P95={m['p95_ms']:.0f}ms, "
                      f"Replicas={m['ready_replicas']}, CPU={m['cpu_util']:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    avg_cost = np.mean(episode_costs)
    avg_sla = np.mean(episode_slas)
    
    print(f"\n  Average reward: {avg_reward:.2f}")
    print(f"  Average cost: ${avg_cost:.4f}")
    print(f"  Average SLA violations: {avg_sla:.2f}")
    
    return {'reward': avg_reward, 'cost': avg_cost, 'sla': avg_sla}


def compare_baselines():
    """Compare random vs HPA"""
    print("=" * 70)
    print("BASELINE COMPARISON FOR MARL TRAINING")
    print("=" * 70)
    
    env = K8sAutoscaleEnv('config.yaml')
    
    # Run baselines
    random_results = run_random_policy(env, num_episodes=5)
    hpa_results = run_hpa_policy(env, target_cpu=0.75, num_episodes=5)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Policy':<15} {'Reward':<12} {'Cost':<12} {'SLA Violations':<15}")
    print("-" * 70)
    print(f"{'Random':<15} {random_results['reward']:<12.2f} ${random_results['cost']:<11.4f} {random_results['sla']:<15.2f}")
    print(f"{'HPA':<15} {hpa_results['reward']:<12.2f} ${hpa_results['cost']:<11.4f} {hpa_results['sla']:<15.2f}")
    
    # Target for MARL
    improvement = 0.20  # Target 20% improvement
    target_reward = hpa_results['reward'] * (1 + improvement)
    
    print(f"{'MARL Target':<15} {target_reward:<12.2f} {'(+20% vs HPA)':<25}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Train your MARL agent using boutique_env.py")
    print("2. Target: Beat HPA by 15-30% (reward improvement)")
    print("3. Expected training time: ~100k steps")
    print("4. Your agents will learn to:")
    print("   - Scale predictively (traffic derivative signal)")
    print("   - Coordinate across services (downstream queue)")
    print("   - Anticipate pod startup delays")
    print("   - Balance cost vs SLA optimally")
    print("=" * 70)


if __name__ == '__main__':
    compare_baselines()