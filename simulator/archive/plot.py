# FILE: plot_episode.py

import numpy as np
import matplotlib.pyplot as plt
from boutique_env import K8sAutoscaleEnv

# Inside your plotting script (plot.py, demo_baseline.py, etc.)

def run_hpa_policy(env, target_cpu=0.75):
    """
    Runs a full episode with the HPA logic and returns
    a history of all metrics for plotting.
    """
    obs, _ = env.reset(seed=1000) # Use a fixed seed for a repeatable plot
    history = []
    
    # --- THE FIX ---
    # Get a stable agent name *before* the loop.
    # env.agents will be cleared on the last step, so we can't rely on it inside the loop.
    if not env.agents:
        print("Error: Environment has no agents after reset.")
        return []
    agent_key = env.agents[0] # e.g., 'api'. We just need one key to get the shared info.
    # --- END FIX ---
    
    print("Running HPA simulation to generate plot...")
    
    for step in range(env.simulator.max_steps):
        actions = {}
        # Make sure to check if agent is still in obs, as it might be done
        for agent in obs: 
            # Denormalize observation to get real values
            cpu_util = obs[agent][0] * 2.0
            ready_replicas = int(obs[agent][10] * 20.0)
            
            # HPA formula
            desired = int(np.ceil(ready_replicas * (cpu_util / target_cpu)))
            desired = max(1, min(desired, 10)) # Clamp 1-10
            
            actions[agent] = desired - 1 # Convert to 0-9 action space

        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # --- THE FIX ---
        # Use the stable 'agent_key' to access the shared info dictionary.
        # This is safe even if env.agents is now empty.
        history.append(infos[agent_key]['metrics'])
        # --- END FIX ---
        
        # Check for termination
        if terms[agent_key]:
            break # Exit the loop as soon as the episode is done
            
    print(f"Simulation complete. {len(history)} steps recorded.")
    return history
def plot_history(history, agents_to_plot=['api', 'app', 'db']):
    """
    Takes the simulation history and generates a plot.
    """
    print("Generating plot...")
    
    num_agents = len(agents_to_plot)
    fig, axes = plt.subplots(num_agents, 1, figsize=(15, 5 * num_agents), sharex=True)
    if num_agents == 1:
        axes = [axes] # Make it iterable if only one
        
    fig.suptitle("HPA Policy Baseline Performance", fontsize=16, y=1.02)

    for i, agent_name in enumerate(agents_to_plot):
        ax = axes[i]
        
        # Extract data for this agent
        steps = range(len(history))
        p95 = [h[agent_name]['p95_ms'] for h in history]
        cpu = [h[agent_name]['cpu_util'] * 100 for h in history] # As %
        queue = [h[agent_name]['queue'] for h in history]
        ready = [h[agent_name]['ready_replicas'] for h in history]
        desired = [h[agent_name]['desired_replicas'] for h in history]
        
        # --- Left Y-Axis (Latency & Queue) ---
        ax.set_title(f"Service: {agent_name}")
        ax.set_ylabel("Latency (ms) / Queue", color='blue')
        ax.plot(steps, p95, label='P95 Latency (ms)', color='blue', linewidth=2)
        ax.plot(steps, queue, label='Queue Length', color='cyan', linestyle='--')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, linestyle=':')
        
        # Add SLA threshold line
        sla_threshold = env.simulator.sla_threshold_ms
        ax.axhline(y=sla_threshold, color='red', linestyle='--', label=f'SLA Threshold ({sla_threshold}ms)')
        
        # --- Right Y-Axis (Replicas & CPU) ---
        ax2 = ax.twinx()
        ax2.set_ylabel("Replicas / CPU %", color='green')
        ax2.plot(steps, ready, label='Ready Replicas', color='green', linewidth=2, linestyle='-')
        ax2.plot(steps, desired, label='Desired Replicas', color='orange', linestyle=':')
        ax2.plot(steps, cpu, label='CPU Util %', color='purple', linestyle='-.', alpha=0.5)
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

    ax.set_xlabel("Simulation Step (x 30s)")
    plt.tight_layout()
    plt.savefig("hpa_baseline_plot.png")
    print(f"\n✅ Plot saved to hpa_baseline_plot.png")
    print("Run this again with your trained agent to compare!")

if __name__ == "__main__":
    env = K8sAutoscaleEnv()
    hpa_history = run_hpa_policy(env)
    plot_history(hpa_history)
