"""
viz/obstacle_heatmap.py
=======================
Visualisasi heatmap exposure per sisi obstacle.
"""

import numpy as np # type: ignore
import plotly.graph_objects as go # type: ignore
from typing import Dict, Tuple
from constants import CELL_OBSTACLE


def plot_obstacle_heatmap(
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    layout: Dict[Tuple[int, int], int],
    width: int,
    height: int,
) -> go.Figure:
    """Plot heatmap per sisi obstacle.
    
    Parameters
    ----------
    obstacle_heatmap : Dict
        {(col, row): {"top": weight, "right": weight, "bottom": weight, "left": weight}}
    layout : Dict
        Grid layout
    width, height : int
        Grid dimensions
    
    Returns
    -------
    go.Figure
        Plotly heatmap visualization
    """

    # Initialize heatmap data dengan NaN (sel non-obstacle = transparan)
    heatmap_data = np.full((height, width), np.nan)

    # Tandai semua obstacle dengan 0 dulu (termasuk yang belum ada di heatmap)
    for (col, row), cell_type in layout.items():
        if cell_type == CELL_OBSTACLE and 0 <= row < height and 0 <= col < width:
            heatmap_data[row, col] = 0.0

    if not obstacle_heatmap:
        # Tampilkan grid dengan obstacle tapi semua weight = 0
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            colorscale=[[0.0, "#2a2a4a"], [1.0, "#2a2a4a"]],
            showscale=False,
            zmin=0,
            zmax=1,
            hovertemplate="Col %{x}, Row %{y}<extra>No data yet</extra>",
        ))
        fig.update_layout(
            title=dict(
                text="🔥 Obstacle Exposure Heatmap",
                font=dict(color="white", size=13, family="monospace"),
            ),
            paper_bgcolor="#0d0d1a",
            plot_bgcolor="#0d0d1a",
            xaxis=dict(showgrid=False, zeroline=False, color="#666"),
            yaxis=dict(showgrid=False, zeroline=False, color="#666", autorange="reversed"),
            margin=dict(l=40, r=20, t=50, b=40),
            height=400,
        )
        return fig

    # Hitung total weight per obstacle (sum semua 4 sisi)
    # Gunakan raw sum agar perubahan kecil tetap terlihat
    all_totals = []
    for (col, row), side_weights in obstacle_heatmap.items():
        if 0 <= row < height and 0 <= col < width and side_weights:
            total = sum(side_weights.values())
            heatmap_data[row, col] = total
            all_totals.append(total)

    max_weight = max(all_totals) if all_totals else 1.0
    max_weight = max(max_weight, 0.01)  # Hindari pembagian nol

    # Buat hover text yang informatif (hanya untuk sel obstacle)
    hover_text = [[""] * width for _ in range(height)]
    for (col, row), side_weights in obstacle_heatmap.items():
        if 0 <= row < height and 0 <= col < width and side_weights:
            total = sum(side_weights.values())
            sides_str = " | ".join(
                f"{s}: {v:.1f}" for s, v in side_weights.items() if v > 0
            ) or "no exposure"
            hover_text[row][col] = (
                f"<b>Obstacle ({col}, {row})</b><br>"
                f"Total: {total:.1f}<br>"
                f"{sides_str}"
            )

    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        colorscale=[
            [0.0, "#1e1e3a"],      # Near-zero — dark navy
            [0.15, "#0f3460"],     # Low — dark blue
            [0.4, "#2980b9"],      # Medium — blue
            [0.65, "#f39c12"],     # High — orange
            [0.85, "#e74c3c"],     # Very high — red
            [1.0, "#c0392b"],      # Max — dark red
        ],
        zmin=0,
        zmax=max_weight,
        colorbar=dict(
            title=dict(text="Exposure", font=dict(color="white", size=11)),
            tickfont=dict(color="white"),
            len=0.75,
            thickness=14,
        ),
        hoverongaps=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
        connectgaps=False,
    ))

    fig.update_layout(
        title=dict(
            text="🔥 Obstacle Exposure Heatmap",
            font=dict(color="white", size=13, family="monospace"),
            x=0.02,
        ),
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        xaxis=dict(
            title=dict(text="Column", font=dict(color="#888")),
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#666"),
        ),
        yaxis=dict(
            title=dict(text="Row", font=dict(color="#888")),
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#666"),
            autorange="reversed",
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=400,
    )

    return fig