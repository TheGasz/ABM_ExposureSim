"""
viz/obstacle_heatmap.py
=======================
Visualisasi heatmap exposure per sisi obstacle.
Smoothing Viridis dengan midpoint interpolation.
"""

import numpy as np # type: ignore
import plotly.graph_objects as go # type: ignore
from typing import Dict, Tuple, Optional, List
from constants import CELL_OBSTACLE

BASE_LINE = "#2a2a4a"
GRID_SCALE = 12
SIGMA = 0.35
MIDPOINT_DISTANCE = 1.0


def _add_midpoints(points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    points_map = {(x, y): w for x, y, w in points}
    midpoints: Dict[Tuple[float, float], float] = {}
    for (x, y), w in points_map.items():
        for dx, dy in ((MIDPOINT_DISTANCE, 0.0), (0.0, MIDPOINT_DISTANCE)):
            neighbor = (x + dx, y + dy)
            if neighbor in points_map:
                mid = (x + dx * 0.5, y + dy * 0.5)
                mid_w = (w + points_map[neighbor]) * 0.5
                existing = midpoints.get(mid)
                midpoints[mid] = mid_w if existing is None else max(existing, mid_w)
    combined = [(x, y, w) for (x, y), w in points_map.items()]
    combined.extend((x, y, w) for (x, y), w in midpoints.items())
    return combined


def plot_obstacle_heatmap(
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    layout: Dict[Tuple[int, int], int],
    width: int,
    height: int,
    prev_fig: Optional[go.Figure] = None,
) -> go.Figure:
    """Plot heatmap per sisi obstacle (empat sisi).
    
    Parameters
    ----------
    obstacle_heatmap : Dict
        {(col, row): {"top": weight, "right": weight, "bottom": weight, "left": weight}}
    layout : Dict
        Grid layout
    width, height : int
        Grid dimensions
    prev_fig : Optional[go.Figure]
        Parameter legacy (diabaikan), disisakan untuk kompatibilitas.
    
    Returns
    -------
    go.Figure
        Plotly heatmap visualization
    """

    obstacle_cells = [
        (col, row)
        for (col, row), cell_type in layout.items()
        if cell_type == CELL_OBSTACLE and 0 <= row < height and 0 <= col < width
    ]

    side_points: Dict[str, List[Tuple[float, float, float]]] = {
        "top": [],
        "right": [],
        "bottom": [],
        "left": [],
    }

    hover_x = []
    hover_y = []
    hover_text = []
    shapes = []

    for (col, row) in obstacle_cells:
        x0 = float(col)
        y0 = float(row)
        xc = x0 + 0.5
        yc = y0 + 0.5

        weights = obstacle_heatmap.get((col, row), {})
        top_w = float(weights.get("top", 0.0))
        right_w = float(weights.get("right", 0.0))
        bottom_w = float(weights.get("bottom", 0.0))
        left_w = float(weights.get("left", 0.0))

        if top_w > 0:
            side_points["top"].append((xc, y0, top_w))
        if right_w > 0:
            side_points["right"].append((x0 + 1.0, yc, right_w))
        if bottom_w > 0:
            side_points["bottom"].append((xc, y0 + 1.0, bottom_w))
        if left_w > 0:
            side_points["left"].append((x0, yc, left_w))

        total = top_w + right_w + bottom_w + left_w
        sides_str = " | ".join(
            part
            for part in [
                f"top: {top_w:.1f}" if top_w > 0 else "",
                f"right: {right_w:.1f}" if right_w > 0 else "",
                f"bottom: {bottom_w:.1f}" if bottom_w > 0 else "",
                f"left: {left_w:.1f}" if left_w > 0 else "",
            ]
            if part
        ) or "no exposure"

        hover_x.append(xc)
        hover_y.append(yc)
        hover_text.append(
            f"<b>Obstacle ({col}, {row})</b><br>"
            f"Total: {total:.1f}<br>"
            f"{sides_str}"
        )

        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x0 + 1.0,
                y0=y0,
                y1=y0 + 1.0,
                line=dict(color=BASE_LINE, width=1),
                fillcolor="rgba(0,0,0,0)",
            )
        )

    all_points: List[Tuple[float, float, float]] = []
    for points in side_points.values():
        if points:
            all_points.extend(_add_midpoints(points))

    nx = max(2, int(width * GRID_SCALE))
    ny = max(2, int(height * GRID_SCALE))
    x = np.linspace(0.0, float(width), nx)
    y = np.linspace(0.0, float(height), ny)
    X, Y = np.meshgrid(x, y)

    field = np.zeros((ny, nx), dtype=float)
    for px, py, weight in all_points:
        d2 = (X - px) ** 2 + (Y - py) ** 2
        field += weight * np.exp(-d2 / (2.0 * SIGMA ** 2))

    max_val = float(np.max(field)) if field.size else 0.0
    if max_val > 0:
        field = field / max_val

    fig = go.Figure(
        data=go.Heatmap(
            z=field,
            x=x,
            y=y,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="Exposure", font=dict(color="white", size=11)),
                tickfont=dict(color="white"),
                len=0.75,
                thickness=14,
            ),
            hovertemplate="Exposure: %{z:.2f}<extra></extra>",
            showscale=True,
            zsmooth="best",
        )
    )

    fig.update_layout(
        title=dict(
            text="🔥 Obstacle Exposure Heatmap",
            font=dict(color="white", size=13, family="monospace"),
            x=0.02,
        ),
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#666"),
            range=[0, width],
            constrain="domain",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#666"),
            range=[0, height],
            autorange="reversed",
            scaleanchor="x",
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=400,
        shapes=shapes,
    )

    if hover_x:
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker=dict(size=6, color="rgba(0,0,0,0)"),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    return fig