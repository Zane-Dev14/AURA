#!/usr/bin/env python3
"""
AURA Simulation-to-Production Pipeline Diagram
High-quality, presentation-ready visualization showing training, validation, and deployment lifecycle
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
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
COLOR_SIM = '#7B1FA2'  # Purple for simulation
COLOR_TRAIN = '#C62828'  # Red for training
COLOR_VAL = '#F57C00'  # Orange for validation
COLOR_PROD = '#2E7D32'  # Green for production
COLOR_FEEDBACK = '#1565C0'  # Blue for feedback loop
COLOR_ARROW = '#424242'  # Dark gray

def draw_phase_box(ax, x, y, width, height, label, color, alpha=0.15, linewidth=2):
    """Draw a phase box with label"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.08",
                         edgecolor=color, facecolor=color,
                         alpha=alpha, linewidth=linewidth)
    ax.add_patch(box)
    ax.text(x + width/2, y + height - 0.3, label,
            ha='center', va='top', fontsize=11, fontweight='bold', color=color)

def draw_component(ax, x, y, width, height, title, details, color, alpha=0.2):
    """Draw a component box with title and details"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor=color, facecolor=color,
                         alpha=alpha, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height - 0.2, title,
            ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Add details
    y_offset = y + height - 0.5
    for detail in details:
        ax.text(x + width/2, y_offset, detail,
                ha='center', va='top', fontsize=7)
        y_offset -= 0.25

def draw_arrow(ax, x1, y1, x2, y2, label='', color=COLOR_ARROW, style='solid', width=2, curve=0):
    """Draw an arrow with optional label and curve"""
    if curve != 0:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=25,
                               color=color, linewidth=width,
                               linestyle=style, alpha=0.8,
                               connectionstyle=f"arc3,rad={curve}")
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=25,
                               color=color, linewidth=width,
                               linestyle=style, alpha=0.8)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label,
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor=color, linewidth=1.5))

# ============= TITLE =============
ax.text(7, 9.5, 'AURA Simulation-to-Production Pipeline',
        ha='center', va='center', fontsize=15, fontweight='bold')

# ============= PHASE 1: SIMULATION & TRAINING =============
draw_phase_box(ax, 0.5, 5.5, 4.5, 3.2, 'Phase 1: Training', COLOR_TRAIN, alpha=0.1)

# Simulator
draw_component(ax, 0.8, 7.2, 1.8, 1.3,
               'Simulator',
               ['Synthetic Env', '8 parallel envs', 'Pod lifecycle'],
               COLOR_SIM)

# Pod lifecycle detail
pod_box = FancyBboxPatch((0.9, 6.5), 1.6, 0.5,
                         boxstyle="round,pad=0.03",
                         edgecolor=COLOR_SIM, facecolor='#F3E5F5',
                         alpha=0.9, linewidth=1)
ax.add_patch(pod_box)
ax.text(1.7, 6.75, '25s Pod Startup', ha='center', va='center', fontsize=7, fontweight='bold')
ax.text(1.7, 6.6, 'API:25s APP:21s DB:15s', ha='center', va='center', fontsize=6)

# QMIX Training
draw_component(ax, 3.0, 7.2, 1.7, 1.3,
               'QMIX Training',
               ['200 epochs', '1000 steps/epoch', 'CTDE learning'],
               COLOR_TRAIN)

# Training metrics
metrics_box = FancyBboxPatch((3.1, 6.5), 1.5, 0.5,
                             boxstyle="round,pad=0.03",
                             edgecolor=COLOR_TRAIN, facecolor='#FFEBEE',
                             alpha=0.9, linewidth=1)
ax.add_patch(metrics_box)
ax.text(3.85, 6.75, 'Reward Weights', ha='center', va='center', fontsize=7, fontweight='bold')
ax.text(3.85, 6.6, 'α=2.0 β=2.5 γ=1.5', ha='center', va='center', fontsize=6)

# Arrow from simulator to training
draw_arrow(ax, 2.7, 7.8, 2.95, 7.8, 'Episodes', COLOR_SIM, width=2)

# ============= PHASE 2: VALIDATION =============
draw_phase_box(ax, 5.5, 5.5, 4, 3.2, 'Phase 2: Validation', COLOR_VAL, alpha=0.1)

# Policy checkpoint
checkpoint_box = FancyBboxPatch((5.8, 7.5), 1.5, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='#6A1B9A', facecolor='#E1BEE7',
                                alpha=0.9, linewidth=2)
ax.add_patch(checkpoint_box)
ax.text(6.55, 7.9, 'Trained Model', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(6.55, 7.65, 'qmix_best.pth', ha='center', va='center', fontsize=7, style='italic')

# Local cluster validation
draw_component(ax, 5.8, 6.2, 1.5, 1.0,
               'Local Cluster',
               ['K3d/Minikube', 'Test scenarios'],
               COLOR_VAL)

# Evaluation metrics
draw_component(ax, 7.6, 6.2, 1.6, 1.6,
               'Evaluation',
               ['P99 latency', 'CPU usage', 'Cost metrics', 'vs HPA baseline'],
               COLOR_VAL)

# Arrows in validation phase
draw_arrow(ax, 6.55, 7.45, 6.55, 7.2, '', COLOR_VAL, width=1.5)
draw_arrow(ax, 6.55, 6.2, 7.55, 6.9, 'Metrics', COLOR_VAL, width=1.5)

# ============= PHASE 3: PRODUCTION DEPLOYMENT =============
draw_phase_box(ax, 10, 5.5, 3.5, 3.2, 'Phase 3: Production', COLOR_PROD, alpha=0.1)

# AURA Controller
draw_component(ax, 10.3, 7.2, 2.9, 1.3,
               'AURA Controller',
               ['3 QMIX agents', '30s control loop', 'Real-time scaling'],
               COLOR_PROD)

# Safety mechanisms
safety_box = FancyBboxPatch((10.4, 6.5), 2.7, 0.5,
                            boxstyle="round,pad=0.03",
                            edgecolor='#D32F2F', facecolor='#FFCDD2',
                            alpha=0.9, linewidth=1.5)
ax.add_patch(safety_box)
ax.text(11.75, 6.75, 'Safety Layer', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(11.75, 6.6, 'Guard rails • Cooldown • Vetoes', ha='center', va='center', fontsize=6)

# Arrows between phases
draw_arrow(ax, 4.8, 7.8, 5.7, 7.8, 'Checkpoint', COLOR_TRAIN, width=2.5)
draw_arrow(ax, 9.3, 7.0, 10.2, 7.5, 'Deploy', COLOR_VAL, width=2.5)

# ============= OUTCOMES BOX =============
outcomes_box = FancyBboxPatch((10.3, 4.5), 2.9, 0.8,
                              boxstyle="round,pad=0.05",
                              edgecolor='#1B5E20', facecolor='#C8E6C9',
                              alpha=0.95, linewidth=2)
ax.add_patch(outcomes_box)
ax.text(11.75, 5.05, 'Production Results', ha='center', va='center', fontsize=9, fontweight='bold', color='#1B5E20')
ax.text(11.75, 4.8, '55% cost reduction • 77% latency improvement', ha='center', va='center', fontsize=7, fontweight='bold')
ax.text(11.75, 4.6, 'vs HPA baseline', ha='center', va='center', fontsize=6, style='italic')

# ============= FEEDBACK LOOP =============
# Monitoring and feedback
feedback_box = FancyBboxPatch((0.8, 3.5), 3.5, 1.3,
                              boxstyle="round,pad=0.05",
                              edgecolor=COLOR_FEEDBACK, facecolor=COLOR_FEEDBACK,
                              alpha=0.15, linewidth=2)
ax.add_patch(feedback_box)
ax.text(2.55, 4.5, 'Continuous Monitoring', ha='center', va='center', fontsize=10, fontweight='bold', color=COLOR_FEEDBACK)
ax.text(2.55, 4.15, 'Production metrics', ha='center', va='center', fontsize=8)
ax.text(2.55, 3.9, 'Performance analysis', ha='center', va='center', fontsize=8)
ax.text(2.55, 3.65, 'Retraining triggers', ha='center', va='center', fontsize=8)

# Feedback arrows
draw_arrow(ax, 11.75, 5.3, 11.75, 3.0, '', COLOR_PROD, width=1.5, style='dashed')
draw_arrow(ax, 11.75, 3.0, 4.5, 3.0, '', COLOR_FEEDBACK, width=2, style='dashed')
draw_arrow(ax, 4.5, 3.0, 2.55, 3.4, '', COLOR_FEEDBACK, width=2, style='dashed')
draw_arrow(ax, 1.5, 4.9, 1.5, 6.4, 'Retrain', COLOR_FEEDBACK, width=2, curve=0.3)

# ============= TIMELINE INDICATOR =============
# Timeline arrow at bottom
timeline_arrow = FancyArrowPatch((1, 2.5), (13, 2.5),
                                arrowstyle='->', mutation_scale=30,
                                color='#616161', linewidth=3, alpha=0.6)
ax.add_patch(timeline_arrow)
ax.text(7, 2.1, 'Time', ha='center', va='center', fontsize=10, style='italic', color='#616161')

# Phase markers on timeline
phase_markers = [
    (2.5, 'Offline Training\n~hours'),
    (7, 'Validation\n~minutes'),
    (11.75, 'Production\nContinuous')
]
for x, label in phase_markers:
    ax.plot(x, 2.5, 'o', markersize=10, color='#616161', zorder=10)
    ax.text(x, 1.7, label, ha='center', va='top', fontsize=7, color='#424242')

# ============= KEY PARAMETERS BOX =============
params_text = (
    "Training Configuration:\n"
    "• 8 parallel environments\n"
    "• 200 epochs × 1000 steps\n"
    "• 25s pod startup model\n"
    "• Reward: α=2.0, β=2.5, γ=1.5\n"
    "\n"
    "Production Configuration:\n"
    "• 30s control loop\n"
    "• 16-dim observation space\n"
    "• 10-action discrete space\n"
    "• Safety: min=1, max=5 replicas"
)
ax.text(5.8, 4.5, params_text, ha='left', va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', alpha=0.95, 
                 edgecolor='#F57F17', linewidth=2))

# ============= LEGEND =============
legend_elements = [
    mlines.Line2D([], [], color=COLOR_TRAIN, linewidth=3, label='Training Flow'),
    mlines.Line2D([], [], color=COLOR_VAL, linewidth=3, label='Validation Flow'),
    mlines.Line2D([], [], color=COLOR_PROD, linewidth=3, label='Production Flow'),
    mlines.Line2D([], [], color=COLOR_FEEDBACK, linewidth=2, linestyle='dashed', label='Feedback Loop')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95)

plt.tight_layout()
plt.savefig('docs/AURA_Pipeline_Diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('docs/AURA_Pipeline_Diagram.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Diagram 2 saved: AURA_Pipeline_Diagram.png and .pdf")
plt.close()

# Made with Bob
