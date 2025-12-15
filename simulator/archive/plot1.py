import matplotlib.pyplot as plt

# -------------------------------------------------------
# ✅  Optional: use a clean Matplotlib style
# (Not seaborn, but a built-in Matplotlib stylesheet)
# -------------------------------------------------------
plt.style.use("ggplot")

# -------------------------------------------------------
# ✅ Updated Names
# -------------------------------------------------------
methods = ["HPA", "Basic Alibaba Cluster Model", "Per-Agent DQN", "QMIX"]

# Data
mean_reward = [-282.9997, -3996.107, -975.652, -206.747]
std_reward = [62.46, 840.58, 211.38, 10.34]

mean_cost = [26.585, 29.519, 45.98, 20.291]
std_cost = [4.496, 5.595, 0.849, 1.924]

mean_sla = [1.4, 41.9, 2.5, 2.1]
std_sla = [0.91, 20.03, 0.67, 0.3]

mean_latency = [38.87, 615.89, 52.91, 51.90]
std_latency = [11.75, 284.14, 2.93, 3.99]

# -------------------------------------------------------
# ✅ Global Aesthetic Improvements
# -------------------------------------------------------
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
# ✅ Function for clean, modern bar charts
# -------------------------------------------------------
def create_bar_plot(title, ylabel, means, stds):
    fig, ax = plt.subplots()
    
    bars = ax.bar(methods, means, yerr=stds, capsize=6, linewidth=1.2)
    
    # Increase bar edge sharpness
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
# ✅ Generate all improved figures
# -------------------------------------------------------
create_bar_plot("Mean Reward (with Standard Deviation)", "Reward", mean_reward, std_reward)
create_bar_plot("Mean Cost (with Standard Deviation)", "Cost", mean_cost, std_cost)
create_bar_plot("Mean SLA Violations (with Standard Deviation)", "SLA Violations", mean_sla, std_sla)
create_bar_plot("Mean Latency (with Standard Deviation)", "Latency (ms)", mean_latency, std_latency)
