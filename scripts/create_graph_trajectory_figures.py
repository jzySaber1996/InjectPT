#!/usr/bin/env python3
"""Create compact paper-ready PNG diagrams for graph trajectories and threat propagation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT = Path('/root/JZY/agent-threat-mining/artifacts/openclaw_graph_figures')

BLUE = '#5D7896'
BLUE_DARK = '#3B566F'
GRAY = '#B5C0CA'
GRAY_DARK = '#788896'
RED = '#C53D3D'
RED_DARK = '#9E2525'
RED_LIGHT = '#E8A6A6'
WHITE = '#FFFFFF'

# A compact abstract workflow: start -> behaviors -> gates -> terminal.
NODES = {
    # Compact horizontal spacing keeps arrows short and gives the state symbols room.
    'start': (0.48, 3.10),
    'b1': (1.38, 3.10),
    'b2': (2.28, 3.10),
    'g1': (3.22, 3.10),
    'b3': (4.05, 3.95),
    'b4': (4.05, 2.25),
    'g2': (4.98, 3.10),
    'b5': (5.92, 3.10),
    'g3': (6.88, 3.10),
    'b6': (7.82, 3.95),
    'b7': (7.82, 2.25),
    'terminal': (9.15, 3.10),
}

EDGES = [
    ('start', 'b1'),
    ('b1', 'b2'),
    ('b2', 'g1'),
    ('g1', 'b3'),
    ('g1', 'b4'),
    ('b3', 'g2'),
    ('b4', 'g2'),
    ('g2', 'b5'),
    ('b5', 'g3'),
    ('g3', 'b6'),
    ('g3', 'b7'),
    ('b6', 'terminal'),
    ('b7', 'terminal'),
]

BEHAVIOR_NODES = {'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'}
GATE_NODES = {'g1', 'g2', 'g3'}


def arrow(ax, start, end, *, color=GRAY_DARK, width=1.7, alpha=0.9, linestyle='-', mutation_scale=13, rad=0.0, zorder=2):
    x1, y1 = NODES[start]
    x2, y2 = NODES[end]
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>',
        mutation_scale=mutation_scale,
        linewidth=width,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
        connectionstyle=f'arc3,rad={rad}',
        shrinkA=18,
        shrinkB=18,
        zorder=zorder,
    )
    ax.add_patch(patch)


def node(ax, name, *, edge=BLUE_DARK, face=WHITE, radius=0.19, linewidth=2.1, zorder=5):
    x, y = NODES[name]
    ax.add_patch(Circle((x, y), radius=radius, facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder))


def gate(ax, name, *, color=BLUE_DARK, zorder=5):
    x, y = NODES[name]
    # Dashed outer circle keeps the node circular while distinguishing a decision gate.
    ax.add_patch(Circle((x, y), radius=0.225, facecolor=WHITE, edgecolor=color, linewidth=2.1, linestyle=(0, (3, 2)), zorder=zorder))
    ax.add_patch(Circle((x, y), radius=0.060, facecolor=color, edgecolor=color, linewidth=0.0, zorder=zorder + 1))


def terminal(ax, *, color=BLUE_DARK, zorder=5):
    x, y = NODES['terminal']
    ax.add_patch(Circle((x, y), radius=0.115, facecolor=WHITE, edgecolor=color, linewidth=1.8, zorder=zorder))
    ax.add_patch(Circle((x, y), radius=0.078, facecolor=WHITE, edgecolor=color, linewidth=1.35, zorder=zorder + 1))


def start(ax, *, color=BLUE_DARK, zorder=5):
    x, y = NODES['start']
    ax.add_patch(Circle((x, y), radius=0.072, facecolor=color, edgecolor=color, linewidth=0.0, zorder=zorder))


def setup(ax):
    ax.set_xlim(0.12, 9.55)
    ax.set_ylim(1.42, 4.78)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_base(ax, *, muted=False):
    edge = GRAY_DARK if muted else BLUE_DARK
    line = GRAY if muted else BLUE
    for start_name, end_name in EDGES:
        arrow(ax, start_name, end_name, color=line, width=1.35 if muted else 1.85, alpha=0.70 if muted else 0.92, mutation_scale=12)
    start(ax, color=edge)
    for name in BEHAVIOR_NODES:
        node(ax, name, edge=edge, radius=0.155 if muted else 0.19, linewidth=1.6 if muted else 2.1)
    for name in GATE_NODES:
        gate(ax, name, color=edge)
    terminal(ax, color=edge)


def add_subtle_lane_guides(ax):
    ax.plot([0.35, 9.30], [3.10, 3.10], color='#E8EDF1', linewidth=0.8, zorder=0)
    ax.plot([3.22, 4.98], [3.95, 3.10], color='#F0F3F5', linewidth=0.6, zorder=0)
    ax.plot([3.22, 4.98], [2.25, 3.10], color='#F0F3F5', linewidth=0.6, zorder=0)


def draw_trajectory():
    fig, ax = plt.subplots(figsize=(12, 4.1), dpi=240)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    setup(ax)
    add_subtle_lane_guides(ax)
    draw_base(ax)
    fig.savefig(OUT / 'graph_wo_check_state_trajectory.png', dpi=360, transparent=True, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def halo(ax, name, *, color=RED, radius=0.25, alpha=0.18, linewidth=8.0, zorder=3):
    x, y = NODES[name]
    ax.add_patch(Circle((x, y), radius=radius, facecolor='none', edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=zorder))


def draw_propagation():
    fig, ax = plt.subplots(figsize=(12, 4.1), dpi=240)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    setup(ax)
    add_subtle_lane_guides(ax)
    draw_base(ax, muted=True)

    # Threat entry and propagation chain: entry -> carrier -> trust shift -> effect.
    halo(ax, 'b2', radius=0.23, linewidth=7.0)
    halo(ax, 'g1', radius=0.26, linewidth=7.0)
    halo(ax, 'g2', radius=0.26, linewidth=7.0)
    halo(ax, 'b5', radius=0.24, linewidth=7.0)
    halo(ax, 'g3', radius=0.26, linewidth=7.0)
    halo(ax, 'b6', radius=0.26, linewidth=7.0)
    halo(ax, 'terminal', radius=0.30, linewidth=7.0)

    # External untrusted input enters the first affected behavior node.
    ax.add_patch(Circle((0.88, 2.35), radius=0.055, facecolor=RED, edgecolor=RED, linewidth=0.0, zorder=8))
    entry = FancyArrowPatch((0.88, 2.35), NODES['b2'], arrowstyle='-|>', mutation_scale=15, linewidth=2.8, color=RED, zorder=8, shrinkA=1, shrinkB=18)
    ax.add_patch(entry)

    # Solid red chain shows the effective propagation route.
    for start_name, end_name in [('b2', 'g1'), ('g1', 'b3'), ('b3', 'g2'), ('g2', 'b5'), ('b5', 'g3'), ('g3', 'b6'), ('b6', 'terminal')]:
        arrow(ax, start_name, end_name, color=RED, width=3.0, alpha=0.98, mutation_scale=16, zorder=7)

    # Dashed red branch denotes a containment gate / alternate controlled outcome.
    arrow(ax, 'g3', 'b7', color=RED_DARK, width=2.0, alpha=0.82, linestyle=(0, (5, 3)), mutation_scale=14, zorder=7)
    ax.plot([7.73, 7.91], [2.14, 2.36], color=RED_DARK, linewidth=2.3, zorder=9)
    ax.plot([7.73, 7.91], [2.36, 2.14], color=RED_DARK, linewidth=2.3, zorder=9)

    # Redrawn nodes keep the overlay legible without adding text inside circles.
    for name in ['b2', 'b3', 'b5', 'b6']:
        node(ax, name, edge=RED_DARK, face='#FFF7F7', radius=0.19, linewidth=2.4, zorder=10)
    for name in ['g1', 'g2', 'g3']:
        gate(ax, name, color=RED_DARK, zorder=10)
    terminal(ax, color=RED_DARK, zorder=10)

    fig.savefig(OUT / 'threat_propagation_overlay.png', dpi=360, transparent=True, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_trajectory()
    draw_propagation()
    print(OUT / 'graph_wo_check_state_trajectory.png')
    print(OUT / 'threat_propagation_overlay.png')


if __name__ == '__main__':
    main()
