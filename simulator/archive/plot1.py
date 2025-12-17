import json
import glob
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------
# ✅ Find latest evaluation_results_*.json
# -------------------------------------------------------
json_files = sorted(
    glob.glob("evaluation_results_*.json"),
    key=os.path.getmtime
)

if not json_files:
    raise FileNotFoundError("No evaluation_results_*.json files found")

latest_file = json_files[-1]
print(f"📊 Using results file: {latest_file}")

with open(latest_file, "r") as f:
    results = json.load(f)

# -------------------------------------------------------
# ✅ Methods (ordered for plotting)
# -------------------------------------------------------
methods = [
    "HPA",
    "Single-Agent DQN",
    "Per-Agent DQN",
    "QMIX"
]

display_names = [
    "HPA",
    "Basic Alibaba Model",
    "Per-Agent DQN",
    "QMIX"
]

# -------------------------------------------------------
# ✅ Extract metrics dynamically
# -------------------------------------------------------
mean_reward = [results[m]["mean_reward"] for m in methods]
std_reward  = [results[m]["std_reward"]  for m in methods]

mean_cost = [results[m]["mean_cost"] for m in methods]
std_cost  = [results[m]["std_cost"]  for m in methods]

mean_sla = [results[m]["mean_sla"] for m in methods]
std_sla  = [results[m]["std_sla"]  for m in methods]

mean_latency = [results[m]["mean_latency"] for m in methods]
std_latency  = [results[m]["std_latency"]  for m in methods]

# -------------------------------------------------------
# ✅ Matplotlib styling (unchanged)
# -------------------------------------------------------
plt.style.use("ggplot")

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 13,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2
})

# -------------------------------------------------------
# ✅ Generic bar plot function
# -------------------------------------------------------
def create_bar_plot(title, ylabel, means, stds):
    fig, ax = plt.subplots()

    bars = ax.bar(display_names, means, yerr=stds, capsize=6)

    for b in bars:
        b.set_edgecolor("black")
        b.set_linewidth(1.3)

    ax.set_title(title, pad=20)
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------
# ✅ Generate plots
# -------------------------------------------------------
create_bar_plot(
    "Mean Reward (with Standard Deviation)",
    "Reward",
    mean_reward,
    std_reward
)

create_bar_plot(
    "Mean Cost (with Standard Deviation)",
    "Cost ($)",
    mean_cost,
    std_cost
)

create_bar_plot(
    "Mean SLA Violations (with Standard Deviation)",
    "SLA Violations",
    mean_sla,
    std_sla
)

create_bar_plot(
    "Mean P95 Latency (with Standard Deviation)",
    "Latency (ms)",
    mean_latency,
    std_latency
)
