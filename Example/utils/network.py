"""
Interactive dependency network graphs with node-focus highlighting.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .data_loader import load_encdm_dependencies, load_rgph_dependencies, get_column_label


def build_dependency_graph(dep_data, title="Feature Dependency Network", highlight_node=None):
    if dep_data is None:
        return None

    nodes, edges, node_set = [], [], set()

    if isinstance(dep_data, dict):
        items = dep_data.items()
    else:
        items = []

    for key, value in items:
        source_name = str(key)[:40]
        if source_name not in node_set:
            nodes.append({"id": source_name, "label": get_column_label(source_name), "group": "source"})
            node_set.add(source_name)

        if isinstance(value, dict):
            targets = list(value.keys())
            weights = list(value.values()) if value else []
        elif isinstance(value, list):
            targets = [str(v) for v in value[:30]]
            weights = [1] * len(targets)
        else:
            targets = [str(value)]
            weights = [1]

        for i, target in enumerate(targets[:30]):
            target_name = str(target)[:40]
            if target_name not in node_set:
                nodes.append({"id": target_name, "label": get_column_label(target_name), "group": "target"})
                node_set.add(target_name)
            w = weights[i] if i < len(weights) else 1
            if isinstance(w, dict):
                w = 1
            edges.append({
                "source": source_name, "target": target_name,
                "weight": float(w) if isinstance(w, (int, float)) else 1.0,
            })

    if not nodes:
        return None

    if len(nodes) > 45:
        source_ids = {n["id"] for n in nodes if n["group"] == "source"}
        target_connections = {}
        for e in edges:
            target_connections[e["target"]] = target_connections.get(e["target"], 0) + 1
        top_targets = {t for t, _ in sorted(target_connections.items(), key=lambda x: x[1], reverse=True)[:35 - len(source_ids)]}
        keep_ids = source_ids | top_targets
        nodes = [n for n in nodes if n["id"] in keep_ids]
        edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]

    adj = {n["id"]: set() for n in nodes}
    for e in edges:
        if e["source"] in adj and e["target"] in adj:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])

    focus_set = None
    if highlight_node and highlight_node in adj:
        focus_set = adj[highlight_node] | {highlight_node}

    src_nodes = [n for n in nodes if n["group"] == "source"]
    tgt_nodes = [n for n in nodes if n["group"] == "target"]
    positions = {}
    for i, n in enumerate(src_nodes):
        angle = -np.pi / 2 + (i / max(1, len(src_nodes) - 1)) * np.pi
        positions[n["id"]] = {"x": -1.5 + 0.2 * np.cos(angle), "y": 0.9 * np.sin(angle)}
    for i, n in enumerate(tgt_nodes):
        angle = -np.pi / 2 + (i / max(1, len(tgt_nodes) - 1)) * np.pi
        positions[n["id"]] = {"x": 1.5 - 0.2 * np.cos(angle), "y": 0.9 * np.sin(angle)}

    edge_x, edge_y, edge_hover = [], [], []
    max_w = max(e["weight"] for e in edges) if edges else 1
    for e in edges:
        if e["source"] not in positions or e["target"] not in positions:
            continue
        faded = focus_set is not None and not (e["source"] in focus_set and e["target"] in focus_set)
        x0, y0 = positions[e["source"]]["x"], positions[e["source"]]["y"]
        x1, y1 = positions[e["target"]]["x"], positions[e["target"]]["y"]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        label = f"{get_column_label(e['source'])} -> {get_column_label(e['target'])}"
        edge_hover += [label, label, ""]

    edge_color = "#DEE2E6" if focus_set else "#B0BEC5"
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=edge_color, width=1.5),
        hoverinfo="text", hovertext=edge_hover, showlegend=False,
    )

    node_x, node_y, node_labels, node_ids = [], [], [], []
    node_colors, node_sizes, node_opacity = [], [], []
    for n in nodes:
        nid = n["id"]
        node_x.append(positions[nid]["x"])
        node_y.append(positions[nid]["y"])
        node_labels.append(n["label"][:18])
        node_ids.append(nid)
        if focus_set is None:
            node_colors.append("#1A3A5C" if n["group"] == "source" else "#2C7A7B")
            node_opacity.append(1.0)
            node_sizes.append(14 if n["group"] == "source" else 10)
        elif nid in focus_set:
            node_colors.append("#1A3A5C" if nid == highlight_node else "#2C7A7B")
            node_opacity.append(1.0)
            node_sizes.append(18 if nid == highlight_node else 12)
        else:
            node_colors.append("#ADB5BD")
            node_opacity.append(0.25)
            node_sizes.append(8)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="middle center",
        textfont=dict(size=8, color="#212529"),
        marker=dict(color=node_colors, size=node_sizes, opacity=node_opacity,
                    line=dict(color="#FFFFFF", width=1.5)),
        customdata=node_ids, hoverinfo="text",
        hovertext=[f"<b>{get_column_label(n['id'])}</b>" for n in nodes],
        showlegend=False,
    )

    subtitle = f" — Focus: {get_column_label(highlight_node)}" if highlight_node else " — Select a node below"
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text=title + subtitle, font=dict(size=13, color="#212529"), x=0.02),
        showlegend=False, hovermode="closest", height=480,
        paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False, range=[-2.2, 2.2]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
    )
    return fig, [n["id"] for n in nodes]


def build_encdm_network(highlight_node=None):
    deps = load_encdm_dependencies()
    return build_dependency_graph(deps, "ENCDM Feature Dependency Network", highlight_node)


def build_rgph_network(highlight_node=None):
    deps = load_rgph_dependencies()
    return build_dependency_graph(deps, "RGPH Feature Dependency Network", highlight_node)


def get_all_networks():
    return [
        ("ENCDM Dependencies", build_encdm_network,
         "Click-select a node to isolate first-degree dependencies"),
        ("RGPH Dependencies", build_rgph_network,
         "Click-select a node to isolate first-degree dependencies"),
    ]
