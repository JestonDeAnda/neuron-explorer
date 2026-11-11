"""Render attribution graphs as standalone SVG documents."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

ROW = 22
CELL = 11
PAD_L, PAD_R, PAD_T, PAD_B = 56, 40, 70, 110


def _escape(value: object) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value))
    return html.escape(cleaned)


def _target_label(graph: Mapping[str, Any]) -> str:
    target = graph.get("target", graph["token"])
    if isinstance(target, Mapping) and target.get("kind") == "completion_logprob":
        token_ids = target.get("token_ids")
        count = len(token_ids) if isinstance(token_ids, Sequence) else 0
        return f"completion log-prob ({count} tokens)"
    return repr(target)


def _tokens(graph: Mapping[str, Any]) -> list[str]:
    tokens = graph.get("tokens")
    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
        return [str(token) for token in tokens]
    positions = [
        node.get("pos")
        for node in graph["nodes"]
        if isinstance(node.get("pos"), int) and node["pos"] >= 0
    ]
    return [str(position) for position in range(max(positions, default=-1) + 1)]


def select_graph(
    graph: Mapping[str, Any], max_edges: int = 300, labels: int = 40
) -> tuple[set[int], list[tuple[Sequence[float], float]], set[int]]:
    """Select the highest-influence edges, their nodes, and labeled neurons."""
    nodes, edges = graph["nodes"], graph["edges"]
    incoming = [0.0] * len(nodes)
    for _source, destination, edge_weight in edges:
        incoming[destination] += abs(edge_weight)
    for index, node in enumerate(nodes):
        if node["kind"] == "neuron":
            incoming[index] += abs(node.get("other") or 0)

    def target_weight(index: int) -> float:
        return (
            1.0
            if nodes[index]["kind"] == "logit"
            else nodes[index]["influence"]
        )

    scores = [
        abs(edge_weight) / (incoming[destination] or 1) * target_weight(destination)
        for _source, destination, edge_weight in edges
    ]
    order = sorted(range(len(edges)), key=lambda edge: -scores[edge])[:max_edges]
    keep = {index for edge in order for index in edges[edge][:2]}
    shown = [(edges[edge], scores[edge]) for edge in order]
    named = sorted(
        (
            index
            for index in keep
            if nodes[index]["kind"] == "neuron"
        ),
        key=lambda index: -nodes[index]["influence"],
    )[:labels]
    named += [
        index for index in keep if nodes[index].get("clamped")
    ][:labels]
    return keep, shown, set(named)


def layout_graph(
    graph: Mapping[str, Any], keep: set[int]
) -> tuple[dict[int, tuple[float, float]], list[int], list[int], int]:
    """Lay selected nodes out by token position and model layer."""
    nodes, tokens = graph["nodes"], _tokens(graph)
    top = max(node["layer"] for node in nodes)
    cells: dict[tuple[int, int], list[int]] = {}
    for index in sorted(keep, key=lambda item: -nodes[item]["influence"]):
        node = nodes[index]
        cells.setdefault((node["layer"], node["pos"]), []).append(index)
    widths = [
        max(
            64,
            14
            + CELL
            * max(
                [
                    len(indices)
                    for (_layer, position), indices in cells.items()
                    if position == token_index
                ]
                or [0]
            ),
        )
        for token_index in range(len(tokens))
    ]
    x_origins = [PAD_L]
    for width in widths:
        x_origins.append(x_origins[-1] + width)
    coordinates = {}
    for (layer, position), indices in cells.items():
        y = PAD_T + (top - layer) * ROW
        middle = x_origins[position] + widths[position] / 2
        for offset, index in enumerate(indices):
            coordinates[index] = (
                middle + (offset - (len(indices) - 1) / 2) * CELL,
                y,
            )
    return coordinates, x_origins, widths, top


def render_svg(
    graph: Mapping[str, Any], max_edges: int = 300, labels: int = 40
) -> str:
    """Return a complete standalone SVG representation of *graph*."""
    nodes, tokens = graph["nodes"], _tokens(graph)
    keep, shown, named = select_graph(graph, max_edges, labels)
    coordinates, x_origins, widths, top = layout_graph(graph, keep)
    width = x_origins[-1] + PAD_R + 170
    title = graph.get("title")
    title_padding = 22 if title else 0
    height = PAD_T + (top + 1) * ROW + PAD_B + title_padding
    peak = max((score for _, score in shown), default=1) or 1
    max_influence = (
        max(
            (
                nodes[index]["influence"]
                for index in keep
                if nodes[index]["kind"] != "logit"
            ),
            default=1,
        )
        or 1
    )
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" '
        'font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="11">',
        "<style>.e{fill:none}.e:hover{stroke-opacity:1;stroke-width:3}"
        ".n:hover{stroke:#000;stroke-width:2}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<g transform="translate(0,{title_padding})">',
    ]
    if title:
        output.append(
            f'<text x="{PAD_L}" y="2" font-size="14" fill="#c05621" '
            f'font-weight="600">{_escape(title)}</text>'
        )

    stats = graph["stats"]
    target = _escape(_target_label(graph))
    aggregate_target = isinstance(graph.get("target"), Mapping)
    probability = (
        f"log p={float(graph['completion_logprob']):.3f} · p={float(graph['prob']):.2e}"
        if aggregate_target and "completion_logprob" in graph
        else f"p={float(graph['prob']):.2f}"
    )
    output.append(
        f'<text x="{PAD_L}" y="24" font-size="15" font-weight="600">'
        f"→ {target}  {_escape(probability)}</text>"
    )
    output.append(
        f'<text x="{PAD_L}" y="44" fill="#555">'
        f'{stats.get("kept_neurons", 0)} neurons from '
        f'{stats.get("candidates", 0)} candidates · candidates explain '
        f'{stats.get("logit_from_candidates", 0):.0%} of the '
        f'{"target locally" if aggregate_target else "logit directly"} · '
        f'top {len(shown)} of {stats.get("kept_edges", len(graph["edges"]))} '
        "edges shown</text>"
    )

    for layer in range(-1, top + 1):
        y = PAD_T + (top - layer) * ROW
        if layer % 4 == 3 or layer in (-1, top):
            name = "Emb" if layer < 0 else "Logit" if layer == top else f"L{layer}"
            output.append(
                f'<text x="{PAD_L - 8}" y="{y + 4}" text-anchor="end" '
                f'fill="#888">{name}</text>'
            )
            output.append(
                f'<line x1="{PAD_L}" x2="{x_origins[-1]}" y1="{y}" '
                f'y2="{y}" stroke="#f0f0f0"/>'
            )
    base = PAD_T + (top + 1) * ROW
    for token_index, token in enumerate(tokens):
        center = x_origins[token_index] + widths[token_index] / 2
        if token_index:
            output.append(
                f'<line x1="{x_origins[token_index]}" '
                f'x2="{x_origins[token_index]}" y1="{PAD_T - 10}" '
                f'y2="{base}" stroke="#f4f4f4"/>'
            )
        label = _escape(token.replace("\n", "⏎"))
        output.append(
            f'<text transform="translate({center + 4},{base + 14}) rotate(-45)" '
            f'text-anchor="end" font-size="12" fill="#333">{label}</text>'
        )

    for (source, destination, edge_weight), score in sorted(
        shown, key=lambda edge: edge[1]
    ):
        source_x, source_y = coordinates[source]
        destination_x, destination_y = coordinates[destination]
        alpha = score / peak
        color = "#2b6cb0" if edge_weight > 0 else "#c53030"
        bend = (destination_y - source_y) * 0.45
        tip = (
            f"{_node_name(graph, nodes[source], tokens)} → "
            f"{_node_name(graph, nodes[destination], tokens)}: "
            f"{edge_weight:+.3g}"
        )
        output.append(
            f'<path class="e" d="M{source_x:.1f},{source_y:.1f} '
            f"C{source_x:.1f},{source_y + bend:.1f} "
            f"{destination_x:.1f},{destination_y - bend:.1f} "
            f'{destination_x:.1f},{destination_y:.1f}" stroke="{color}" '
            f'stroke-opacity="{0.15 + 0.75 * alpha:.2f}" '
            f'stroke-width="{0.4 + 3.6 * alpha:.2f}">'
            f"<title>{_escape(tip)}</title></path>"
        )

    labeled = []
    for index in sorted(
        keep,
        key=lambda item: (
            nodes[item]["influence"] if nodes[item]["kind"] != "logit" else 9
        ),
    ):
        node = nodes[index]
        x, y = coordinates[index]
        tip = _escape(_node_tooltip(graph, node, tokens))
        if node["kind"] == "logit":
            points = " ".join(
                f"{x + 9 * math.sin(angle):.1f},{y - 9 * math.cos(angle):.1f}"
                for angle in (2 * math.pi * point / 5 for point in range(5))
            )
            output.append(
                f'<polygon class="n" points="{points}" fill="#d69e2e" '
                f'stroke="#744210"><title>{tip}</title></polygon>'
            )
            output.append(
                f'<text x="{x + 14}" y="{y + 4}" '
                f'font-weight="600">{target}</text>'
            )
            continue
        radius = 3 + 5 * math.sqrt(node["influence"] / max_influence)
        if node["kind"] == "embed":
            output.append(
                f'<rect class="n" x="{x - radius}" y="{y - radius}" '
                f'width="{2 * radius}" height="{2 * radius}" fill="#a0aec0" '
                f'stroke="#4a5568"><title>{tip}</title></rect>'
            )
            continue
        fill = "#9ae6b4" if (node.get("act") or 0) > 0 else "#fbb6ce"
        if node.get("gate") is not None and node["gate"] < -1.278:
            fill = "#e2e8f0"
        ring = (
            'stroke="#dd6b20" stroke-width="2.6"'
            if node.get("clamped")
            else 'stroke="#6b46c1" stroke-width="2.4"'
            if node.get("shared")
            else 'stroke="#2d3748" stroke-width="0.8"'
        )
        output.append(
            f'<circle class="n" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'fill="{fill}" {ring}><title>{tip}</title></circle>'
        )
        if index in named:
            labeled.append((node["influence"], x, y, radius, node))

    taken = [(x - 4, y - 4, 8, 8) for x, y in coordinates.values()]
    for _, x, y, radius, node in sorted(
        labeled, key=lambda item: (not item[4].get("clamped"), -item[0])
    ):
        if node.get("clamped"):
            text = f'L{node["layer"]}.{node["n"]} := {node["act"]:+.2f}'
        else:
            description = node.get("name") or "/".join(node.get("lens", [])[:2])
            text = f'L{node["layer"]}.{node["n"]} {description[:18]}'
        text_width = 5.6 * len(text)
        placements = (
            (x + radius + 2, "start"),
            (x - radius - 2 - text_width, "end"),
        )
        for box_x, anchor in placements:
            box = (box_x, y - radius - 10, text_width, 11)
            if not any(
                _overlap(box, occupied)
                for occupied in taken
                if occupied[2] > 8 or not _near(occupied, x, y)
            ):
                taken.append(box)
                text_x = box_x if anchor == "start" else box_x + text_width
                output.append(
                    f'<text x="{text_x:.1f}" y="{y - radius:.1f}" '
                    f'font-size="9.5" text-anchor="{anchor}" fill="#1a202c" '
                    f'paint-order="stroke" stroke="#fff" stroke-width="2.5">'
                    f"{_escape(text)}</text>"
                )
                break

    legend_y = height - 22 - title_padding
    items = [
        ("#9ae6b4", "circle", "neuron, act > 0"),
        ("#fbb6ce", "circle", "neuron, act < 0"),
        ("#e2e8f0", "circle", "gate < −1.278 (gate path sign unreliable)"),
        ("#a0aec0", "rect", "token embedding"),
    ]
    if any(node.get("clamped") for node in nodes):
        items.append(("#fff", "ring", "clamped by the intervention (no inputs)"))
    if any(node.get("shared") for node in nodes):
        items.append(("#fff", "shared", _shared_label(graph)))
    legend_x = PAD_L
    for color, shape, label in items:
        if shape == "shared":
            output.append(
                f'<circle cx="{legend_x + 5}" cy="{legend_y}" r="5" '
                f'fill="{color}" stroke="#6b46c1" stroke-width="2.4"/>'
            )
        elif shape == "ring":
            output.append(
                f'<circle cx="{legend_x + 5}" cy="{legend_y}" r="5" '
                f'fill="{color}" stroke="#dd6b20" stroke-width="2.6"/>'
            )
        elif shape == "circle":
            output.append(
                f'<circle cx="{legend_x + 5}" cy="{legend_y}" r="5" '
                f'fill="{color}" stroke="#2d3748" stroke-width="0.8"/>'
            )
        else:
            output.append(
                f'<rect x="{legend_x}" y="{legend_y - 5}" width="10" '
                f'height="10" fill="{color}" stroke="#4a5568"/>'
            )
        output.append(
            f'<text x="{legend_x + 14}" y="{legend_y + 4}" fill="#555">'
            f"{_escape(label)}</text>"
        )
        legend_x += 24 + 6.2 * len(label)
    for color, label in (("#2b6cb0", "excites"), ("#c53030", "inhibits")):
        output.append(
            f'<line x1="{legend_x}" x2="{legend_x + 22}" '
            f'y1="{legend_y}" y2="{legend_y}" stroke="{color}" '
            'stroke-width="2.5"/>'
        )
        output.append(
            f'<text x="{legend_x + 28}" y="{legend_y + 4}" fill="#555">'
            f"{label}</text>"
        )
        legend_x += 90
    output.append("</g></svg>")
    return "\n".join(output)


def _shared_label(graph: Mapping[str, Any]) -> str:
    return str(graph.get("shared_label", "shared with the other graphs"))


def _overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        first[0] < second[0] + second[2]
        and second[0] < first[0] + first[2]
        and first[1] < second[1] + second[3]
        and second[1] < first[1] + first[3]
    )


def _near(box: tuple[float, float, float, float], x: float, y: float) -> bool:
    return abs(box[0] + 4 - x) < 0.5 and abs(box[1] + 4 - y) < 0.5


def _node_token(
    graph: Mapping[str, Any], node: Mapping[str, Any], tokens: Sequence[str]
) -> str:
    if "tok" in node:
        return str(node["tok"])
    position = node.get("pos")
    if isinstance(position, int) and 0 <= position < len(tokens):
        return tokens[position]
    return str(position)


def _node_name(
    graph: Mapping[str, Any], node: Mapping[str, Any], tokens: Sequence[str]
) -> str:
    if node["kind"] == "embed":
        return f"emb {_node_token(graph, node, tokens)!r}"
    if node["kind"] == "logit":
        return "logit"
    return f"L{node['layer']} N{node['n']}@{node['pos']}"


def _node_tooltip(
    graph: Mapping[str, Any], node: Mapping[str, Any], tokens: Sequence[str]
) -> str:
    if node["kind"] == "logit":
        if node.get("target_kind") == "completion_logprob":
            return "target scalar: summed completion log-probability"
        return "target logit"
    head = (
        f"{_node_name(graph, node, tokens)}  "
        f"token {_node_token(graph, node, tokens)!r}  "
        f"influence {node['influence']:.3f}"
    )
    if node["kind"] == "embed":
        return head
    extra = f"\nact {node['act']:+.3f}"
    extra += (
        "  (clamped)"
        if node.get("clamped")
        else f"  gate {node.get('gate', float('nan')):+.2f}"
    )
    if node.get("name"):
        extra += f"\nname: {node['name']}"
    if node.get("lens"):
        extra += "\nsteer lens: " + ", ".join(node["lens"])
    return head + extra
