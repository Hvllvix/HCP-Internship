"""
Interactive dependency network graph module.
Simplified overview by default; click a node to highlight only first-degree connections.
"""
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .data_loader import load_encdm_dependencies, load_rgph_dependencies, get_column_label


def build_dependency_graph(dep_data, title="Feature Dependency Network"):
    """Build interactive network graph with click-to-highlight first-degree edges."""
    if dep_data is None:
        return None

    nodes = []
    edges = []
    node_set = set()

    # Extract nodes and edges
    if isinstance(dep_data, dict):
        items = dep_data.items()
    elif isinstance(dep_data, list):
        items = enumerate(dep_data)
    else:
        items = []

    for key, value in items:
        source_name = str(key)[:40]
        if source_name not in node_set:
            display_label = get_column_label(source_name)
            nodes.append({"id": source_name, "label": display_label, "group": "source"})
            node_set.add(source_name)

        if isinstance(value, dict):
            targets = list(value.keys())
            weights = list(value.values()) if len(value) else []
        elif isinstance(value, list):
            targets = [str(v) for v in value[:30]]
            weights = [1] * len(targets)
        else:
            targets = [str(value)]
            weights = [1]

        for i, target in enumerate(targets[:30]):
            target_name = str(target)[:40]
            if target_name not in node_set:
                display_label = get_column_label(target_name)
                nodes.append({"id": target_name, "label": display_label, "group": "target"})
                node_set.add(target_name)

            w = weights[i] if i < len(weights) else 1
            if isinstance(w, dict):
                w = 1
            edges.append({
                "source": source_name,
                "target": target_name,
                "weight": float(w) if isinstance(w, (int, float)) else 1.0
            })

    if len(nodes) == 0:
        return None

    # Limit nodes for readability (keep top connected)
    if len(nodes) > 45:
        source_ids = set(n["id"] for n in nodes if n["group"] == "source")
        target_connections = {}
        for e in edges:
            if e["target"] in target_connections:
                target_connections[e["target"]] += 1
            else:
                target_connections[e["target"]] = 1
        top_targets = set(t[0] for t in sorted(target_connections.items(), key=lambda x: x[1], reverse=True)[:35 - len(source_ids)])
        keep_ids = source_ids | top_targets
        nodes = [n for n in nodes if n["id"] in keep_ids]
        edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]

    # Positions: sources on left, targets on right
    src_nodes = [n for n in nodes if n["group"] == "source"]
    tgt_nodes = [n for n in nodes if n["group"] == "target"]
    positions = {}
    for i, n in enumerate(src_nodes):
        angle = -np.pi / 2 + (i / max(1, len(src_nodes) - 1)) * np.pi
        positions[n["id"]] = {"x": -1.5 + 0.2 * np.cos(angle), "y": 0.9 * np.sin(angle)}
    for i, n in enumerate(tgt_nodes):
        angle = -np.pi / 2 + (i / max(1, len(tgt_nodes) - 1)) * np.pi
        positions[n["id"]] = {"x": 1.5 - 0.2 * np.cos(angle), "y": 0.9 * np.sin(angle)}

    # Build edge traces (all edges as one trace for performance)
    edge_x = []
    edge_y = []
    edge_src = []
    edge_tgt = []
    edge_widths = []
    max_w = max(e["weight"] for e in edges) if edges else 1

    for e in edges:
        if e["source"] in positions and e["target"] in positions:
            x0, y0 = positions[e["source"]]["x"], positions[e["source"]]["y"]
            x1, y1 = positions[e["target"]]["x"], positions[e["target"]]["y"]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_src.append(e["source"])
            edge_tgt.append(e["target"])
            edge_widths.append(max(0.5, e["weight"] / max_w * 3))

    # Create customdata for hover: source -> target
    edge_hover = [f"{get_column_label(s)} -> {get_column_label(t)}" 
                  for s, t in zip(edge_src, edge_tgt)]
    # Repeat each hover 3 times to match [x0, x1, None] pattern
    edge_hover_expanded = []
    for h in edge_hover:
        edge_hover_expanded += [h, h, ""]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#B0BEC5", width=1),
        hoverinfo="text",
        hovertext=edge_hover_expanded if edge_hover_expanded else None,
        showlegend=False,
        name="edges",
    )

    # Node positions
    node_x = [positions[n["id"]]["x"] for n in nodes]
    node_y = [positions[n["id"]]["y"] for n in nodes]
    node_labels = [n["label"] for n in nodes]
    node_ids = [n["id"] for n in nodes]
    node_groups = [n["group"] for n in nodes]
    node_colors = ["#1A5F7A" if g == "source" else "#2C7A7B" for g in node_groups]
    node_sizes = [14 if g == "source" else 10 for g in node_groups]

    # Build adjacency list for highlighting
    adj = {n["id"]: set() for n in nodes}
    for e in edges:
        if e["source"] in adj and e["target"] in adj:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])

    # Node trace - ALL nodes visible by default (click to highlight)
    # Use customdata to store node id for click interaction
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=8, color="#212529"),
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color="#FFFFFF", width=1.5),
        ),
        hoverinfo="text",
        hovertext=[f"<b>{get_column_label(n['id'])}</b><br>Type: {'Dependent Variable' if n['group']=='source' else 'Predictor'}" for n in nodes],
        customdata=node_ids,
        showlegend=False,
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#212529"), x=0.02, xanchor="left"),
        showlegend=False,
        hovermode="closest",
        height=500,
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#F8F9FA",
        font=dict(family="Inter, sans-serif", size=11, color="#212529"),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False, range=[-2.2, 2.2]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
        clickmode="event+select",
    )

    return fig


def build_encdm_network():
    deps = load_encdm_dependencies()
    return build_dependency_graph(deps, "ENCDM Feature Dependency Network (Click a node to highlight connections)")


def build_rgph_network():
    deps = load_rgph_dependencies()
    return build_dependency_graph(deps, "RGPH Feature Dependency Network (Click a node to highlight connections)")


def get_all_networks():
    return [
        ("ENCDM Dependencies", build_encdm_network, "Feature dependency relationships in ENCDM data - click nodes to highlight connections"),
        ("RGPH Dependencies", build_rgph_network, "Feature dependency relationships in RGPH census data - click nodes to highlight connections"),
    ]