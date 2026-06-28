"""
Interactive imputation dependency network graphs.
"""
import numpy as np
import plotly.graph_objects as go

from Utils.data_loader import get_label, load_deps_encdm, load_deps_rgph
from Utils.theme import PALETTE, plotly_layout


def _build_graph(dep_data, title, highlight_node=None):
    if not dep_data:
        return None

    nodes, edges, node_set = [], [], set()

    for key, value in dep_data.items():
        source = str(key)[:40]
        if source not in node_set:
            nodes.append({"id": source, "label": get_label(source), "group": "source"})
            node_set.add(source)

        if isinstance(value, dict):
            targets = list(value.keys())
            weights = list(value.values())
        elif isinstance(value, list):
            targets = [str(v) for v in value[:30]]
            weights = [1.0] * len(targets)
        else:
            targets = [str(value)]
            weights = [1.0]

        for i, target in enumerate(targets[:30]):
            target = str(target)[:40]
            if target not in node_set:
                nodes.append({"id": target, "label": get_label(target), "group": "target"})
                node_set.add(target)
            w = weights[i] if i < len(weights) else 1.0
            if isinstance(w, dict):
                w = 1.0
            edges.append({
                "source": source,
                "target": target,
                "weight": float(w) if isinstance(w, (int, float)) else 1.0,
            })

    if not nodes:
        return None

    if len(nodes) > 45:
        source_ids = {n["id"] for n in nodes if n["group"] == "source"}
        target_connections = {}
        for e in edges:
            target_connections[e["target"]] = target_connections.get(e["target"], 0) + 1
        top_targets = {
            t for t, _ in sorted(
                target_connections.items(), key=lambda x: x[1], reverse=True
            )[:35 - len(source_ids)]
        }
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
        nodes = [n for n in nodes if n["id"] in focus_set]
        edges = [e for e in edges if e["source"] in focus_set and e["target"] in focus_set]

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
    for e in edges:
        if e["source"] not in positions or e["target"] not in positions:
            continue
        x0, y0 = positions[e["source"]]["x"], positions[e["source"]]["y"]
        x1, y1 = positions[e["target"]]["x"], positions[e["target"]]["y"]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        label = f"{get_label(e['source'])} → {get_label(e['target'])}"
        edge_hover += [label, label, ""]

    edge_color = "#B0BEC5" if focus_set is None else PALETTE["navy"]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=edge_color, width=2.0 if focus_set else 1.5),
        hoverinfo="text", hovertext=edge_hover, showlegend=False,
    )

    node_x, node_y, node_labels, node_ids = [], [], [], []
    node_colors, node_sizes = [], []
    for n in nodes:
        nid = n["id"]
        node_x.append(positions[nid]["x"])
        node_y.append(positions[nid]["y"])
        node_labels.append(n["label"][:18])
        node_ids.append(nid)
        if focus_set is None:
            node_colors.append(PALETTE["navy"] if n["group"] == "source" else PALETTE["navy_light"])
            node_sizes.append(14 if n["group"] == "source" else 10)
        else:
            node_colors.append(PALETTE["amber"] if nid == highlight_node else PALETTE["navy_light"])
            node_sizes.append(20 if nid == highlight_node else 12)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="middle center",
        textfont=dict(size=9 if focus_set else 8, color=PALETTE["navy"]),
        marker=dict(
            color=node_colors, size=node_sizes, opacity=1.0,
            line=dict(color=PALETTE["white"], width=1.5),
        ),
        customdata=node_ids, hoverinfo="text",
        hovertext=[f"<b>{get_label(n['id'])}</b>" for n in nodes],
        showlegend=False,
    )

    subtitle = f" — focus: {get_label(highlight_node)}" if highlight_node else ""
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        **plotly_layout(height=420, margin=dict(l=10, r=10, t=40, b=10)),
        title=dict(text=title + subtitle, font=dict(size=13), x=0.02),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(visible=False, range=[-2.2, 2.2]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
    )
    return fig, [n["id"] for n in nodes]


def build_encdm_network(highlight_node=None):
    return _build_graph(load_deps_encdm(), "ENCDM Imputation Dependencies", highlight_node)


def build_rgph_network(highlight_node=None):
    return _build_graph(load_deps_rgph(), "RGPH Imputation Dependencies", highlight_node)
