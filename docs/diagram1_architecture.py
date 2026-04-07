#!/usr/bin/env python3
"""
AURA Predictive Multi-Agent Autoscaling Architecture Diagram
High-quality, presentation-ready visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

# Set up high-quality figure
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
COLOR_K8S = '#326CE5'  # Kubernetes blue
COLOR_ENVOY = '#AC6199'  # Envoy purple
COLOR_PROM = '#E6522C'  # Prometheus orange
COLOR_AURA = '#00897B'  # Teal for AURA
COLOR_SERVICE = '#5C6BC0'  # Indigo for services
COLOR_ARROW = '#424242'  # Dark gray for arrows
COLOR_METRICS = '#FF6F00'  # Orange for metrics flow
COLOR_CONTROL = '#1565C0'  # Blue for control flow

def draw_box(ax, x, y, width, height, label, color, alpha=0.15, linewidth=2):
    """Draw a rounded box with label"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor=color, facecolor=color,
                         alpha=alpha, linewidth=linewidth)
    ax.add_patch(box)
    ax.text(x + width/2, y + height - 0.25, label,
            ha='center', va='top', fontsize=10, fontweight='bold', color=color)

def draw_service_pod(ax, x, y, service_name, port, color):
    """Draw a service pod with Envoy sidecar"""
    # Main service container
    service_box = FancyBboxPatch((x, y), 1.8, 0.8,
                                 boxstyle="round,pad=0.03",
                                 edgecolor=color, facecolor=color,
                                 alpha=0.2, linewidth=1.5)
    ax.add_patch(service_box)
    ax.text(x + 0.9, y + 0.55, service_name.upper(),
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x + 0.9, y + 0.25, f':{port}',
            ha='center', va='center', fontsize=7, color='#666')
    
    # Envoy sidecar
    envoy_box = FancyBboxPatch((x + 0.1, y - 0.5), 1.6, 0.4,
                               boxstyle="round,pad=0.02",
                               edgecolor=COLOR_ENVOY, facecolor=COLOR_ENVOY,
                               alpha=0.15, linewidth=1)
    ax.add_patch(envoy_box)
    ax.text(x + 0.9, y - 0.3, 'Envoy :9901',
            ha='center', va='center', fontsize=7, style='italic')

def draw_arrow(ax, x1, y1, x2, y2, label='', color=COLOR_ARROW, style='solid', width=1.5):
    """Draw an arrow with optional label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           color=color, linewidth=width,
                           linestyle=style, alpha=0.8)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label,
                ha='center', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))

# ============= KUBERNETES CLUSTER =============
draw_box(ax, 0.5, 0.5, 10, 8.5, 'Kubernetes Cluster', COLOR_K8S, alpha=0.08)

# ============= SERVICES (Three-Tier Chain) =============
# API Service
draw_service_pod(ax, 2, 6.5, 'API', '5001', COLOR_SERVICE)
# APP Service
draw_service_pod(ax, 5, 6.5, 'APP', '5002', COLOR_SERVICE)
# DB Service
draw_service_pod(ax, 8, 6.5, 'DB', '3306', COLOR_SERVICE)

# Service chain arrows
draw_arrow(ax, 3.9, 6.9, 4.9, 6.9, 'calls', COLOR_SERVICE, width=2)
draw_arrow(ax, 6.9, 6.9, 7.9, 6.9, 'queries', COLOR_SERVICE, width=2)

# ============= PROMETHEUS =============
prom_box = FancyBboxPatch((1.5, 3.5), 2, 1.2,
                          boxstyle="round,pad=0.05",
                          edgecolor=COLOR_PROM, facecolor=COLOR_PROM,
                          alpha=0.2, linewidth=2)
ax.add_patch(prom_box)
ax.text(2.5, 4.3, 'Prometheus', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(2.5, 3.95, ':9090', ha='center', va='center', fontsize=8)
ax.text(2.5, 3.7, '15s scrape', ha='center', va='center', fontsize=7, style='italic', color=COLOR_PROM)

# Metrics collection arrows (dashed)
draw_arrow(ax, 2.9, 6.0, 2.7, 4.8, '', COLOR_METRICS, style='dashed', width=1.2)
draw_arrow(ax, 5.9, 6.0, 3.3, 4.7, '', COLOR_METRICS, style='dashed', width=1.2)
draw_arrow(ax, 8.9, 6.0, 3.5, 4.6, '', COLOR_METRICS, style='dashed', width=1.2)

# ============= KUBERNETES API SERVER =============
k8s_api_box = FancyBboxPatch((7, 3.5), 2.5, 1.2,
                             boxstyle="round,pad=0.05",
                             edgecolor=COLOR_K8S, facecolor=COLOR_K8S,
                             alpha=0.2, linewidth=2)
ax.add_patch(k8s_api_box)
ax.text(8.25, 4.3, 'Kubernetes API', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(8.25, 3.9, 'kubectl scale', ha='center', va='center', fontsize=7, style='italic')

# Scaling control arrows (from K8s API to services)
draw_arrow(ax, 7.8, 4.8, 3.5, 6.3, '', COLOR_CONTROL, style='dotted', width=1.5)
draw_arrow(ax, 8.25, 4.8, 6.5, 6.3, '', COLOR_CONTROL, style='dotted', width=1.5)
draw_arrow(ax, 8.7, 4.8, 8.5, 6.3, '', COLOR_CONTROL, style='dotted', width=1.5)

# ============= AURA CONTROLLER (Outside cluster) =============
draw_box(ax, 11, 2, 2.8, 6.5, 'AURA Controller', COLOR_AURA, alpha=0.12, linewidth=2.5)

# AURA Components
# 3 Agents
agent_y_positions = [7, 5.5, 4]
agent_names = ['API Agent', 'APP Agent', 'DB Agent']
for i, (y_pos, name) in enumerate(zip(agent_y_positions, agent_names)):
    agent_box = FancyBboxPatch((11.3, y_pos), 2.2, 0.6,
                               boxstyle="round,pad=0.03",
                               edgecolor=COLOR_AURA, facecolor=COLOR_AURA,
                               alpha=0.25, linewidth=1.5)
    ax.add_patch(agent_box)
    ax.text(12.4, y_pos + 0.3, name, ha='center', va='center', fontsize=8, fontweight='bold')

# QMIX Mixing Network
mixer_box = FancyBboxPatch((11.3, 3), 2.2, 0.7,
                           boxstyle="round,pad=0.03",
                           edgecolor='#D32F2F', facecolor='#D32F2F',
                           alpha=0.2, linewidth=1.5)
ax.add_patch(mixer_box)
ax.text(12.4, 3.35, 'QMIX Mixer', ha='center', va='center', fontsize=9, fontweight='bold')

# Safety Layer
safety_box = FancyBboxPatch((11.3, 2.2), 2.2, 0.6,
                            boxstyle="round,pad=0.03",
                            edgecolor='#F57C00', facecolor='#F57C00',
                            alpha=0.2, linewidth=1.5)
ax.add_patch(safety_box)
ax.text(12.4, 2.5, 'Safety Guards', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(12.4, 2.3, 'min=1, max=5', ha='center', va='center', fontsize=6, style='italic')

# Arrows within AURA
for y_pos in agent_y_positions:
    draw_arrow(ax, 12.4, y_pos, 12.4, 3.75, '', COLOR_AURA, width=1)
draw_arrow(ax, 12.4, 3.0, 12.4, 2.85, '', COLOR_AURA, width=1.5)

# ============= CONTROL LOOP =============
# Prometheus to AURA
draw_arrow(ax, 3.5, 4.1, 11.0, 5.5, 'Query Metrics', COLOR_METRICS, width=2)

# AURA to K8s API
draw_arrow(ax, 11.3, 2.5, 9.5, 3.8, 'Scale Commands', COLOR_CONTROL, width=2)

# ============= ANNOTATIONS =============
# 30s control loop
loop_annotation = FancyBboxPatch((4.5, 1.2), 3, 0.6,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#1976D2', facecolor='#E3F2FD',
                                 alpha=0.9, linewidth=2)
ax.add_patch(loop_annotation)
ax.text(6, 1.5, '30s Control Loop', ha='center', va='center', fontsize=10, fontweight='bold', color='#1976D2')

# Observation space
obs_annotation = FancyBboxPatch((11.2, 7.8), 2.4, 0.5,
                                boxstyle="round,pad=0.03",
                                edgecolor='#6A1B9A', facecolor='#F3E5F5',
                                alpha=0.9, linewidth=1.5)
ax.add_patch(obs_annotation)
ax.text(12.4, 8.05, '16-dim observation', ha='center', va='center', fontsize=7, fontweight='bold')
ax.text(12.4, 7.9, '10-action space', ha='center', va='center', fontsize=6)

# Title
ax.text(7, 9.3, 'AURA Predictive Multi-Agent Autoscaling Architecture',
        ha='center', va='center', fontsize=14, fontweight='bold')

# Legend
legend_elements = [
    mlines.Line2D([], [], color=COLOR_SERVICE, linewidth=2, label='Service Chain'),
    mlines.Line2D([], [], color=COLOR_METRICS, linewidth=1.5, linestyle='dashed', label='Metrics (15s scrape)'),
    mlines.Line2D([], [], color=COLOR_CONTROL, linewidth=1.5, linestyle='dotted', label='Scaling Control'),
    mlines.Line2D([], [], color=COLOR_AURA, linewidth=2, label='AURA Decision Flow')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.95)

# Key metrics box
metrics_text = (
    "Key Parameters:\n"
    "• 3 Agents (API, APP, DB)\n"
    "• 16-dimensional observation\n"
    "• 10-action discrete space\n"
    "• Guard rails: min=1, max=5\n"
    "• 30s decision interval"
)
ax.text(0.7, 2.5, metrics_text, ha='left', va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', alpha=0.9, edgecolor='#F57F17', linewidth=1.5))

plt.tight_layout()
plt.savefig('docs/AURA_Architecture_Diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('docs/AURA_Architecture_Diagram.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Diagram 1 saved: AURA_Architecture_Diagram.png and .pdf")
plt.close()

# Made with Bob
