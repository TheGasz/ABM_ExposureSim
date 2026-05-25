"""
viz/obstacle_heatmap.py
=======================
Visualisasi heatmap exposure per sisi obstacle.
Optimized untuk menghindari flicker di Streamlit.
"""

import numpy as np # type: ignore
import plotly.graph_objects as go # type: ignore
from typing import Dict, Tuple, Optional
from constants import CELL_OBSTACLE


def plot_obstacle_heatmap(
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    layout: Dict[Tuple[int, int], int],
    width: int,
    height: int,
    prev_fig: Optional[go.Figure] = None,
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
    prev_fig : Optional[go.Figure]
        Figure sebelumnya untuk update (hindari flicker). Jika None, buat baru.
    
    Returns
    -------
    go.Figure
        Plotly heatmap visualization
    """

    # Initialize heatmap data dengan NaN (sel non-obstacle = transparan)
    heatmap_data = np.full((height, width), np.nan)

    # Tandai semua obstacle dengan 0 dulu
    for (col, row), cell_type in layout.items():
        if cell_type == CELL_OBSTACLE and 0 <= row < height and 0 <= col < width:
            heatmap_data[row, col] = 0.0

    if not obstacle_heatmap:
        # Tampilkan grid dengan obstacle tapi semua weight = 0
        if prev_fig is not None:
            # Update existing figure
            flat_data = heatmap_data.flatten()
            flat_data = np.where(np.isnan(flat_data), None, flat_data)
            prev_fig.update_traces(z=heatmap_data)
            return prev_fig
        
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
            # Disable animations for faster rendering
            transition_duration=0,
        )
        return fig

    # Hitung total weight per obstacle
    all_totals = []
    for (col, row), side_weights in obstacle_heatmap.items():
        if 0 <= row < height and 0 <= col < width and side_weights:
            total = sum(side_weights.values())
            heatmap_data[row, col] = total
            all_totals.append(total)

    max_weight = max(all_totals) if all_totals else 1.0
    max_weight = max(max_weight, 1.0)

    # Buat hover text
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

    # Update existing figure jika ada (lebih cepat, hindari flicker)
    if prev_fig is not None:
        prev_fig.update_traces(
            z=heatmap_data,
            customdata=hover_text,
            zmin=0,
            zmax=max_weight,
        )
        # Update colorbar tickvals
        colorbar_ticks = [0, max_weight * 0.25, max_weight * 0.5, max_weight * 0.75, max_weight]
        prev_fig.update_traces(
            colorbar=dict(
                title=dict(text="Exposure", font=dict(color="white", size=11)),
                tickfont=dict(color="white"),
                tickvals=colorbar_ticks,
                ticktext=["0", "25%", "50%", "75%", f"{max_weight:.1f}"],
                len=0.75,
                thickness=14,
            )
        )
        return prev_fig

    # Create new figure (first time)
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        colorscale=[
            [0.0,  "#1e1e3a"],
            [0.05, "#0f3460"],
            [0.25, "#1a6fa8"],
            [0.5,  "#2ecc71"],
            [0.75, "#f39c12"],
            [0.9,  "#e74c3c"],
            [1.0,  "#c0392b"],
        ],
        zmin=0,
        zmax=max_weight,
        colorbar=dict(
            title=dict(text="Exposure", font=dict(color="white", size=11)),
            tickfont=dict(color="white"),
            tickvals=[0, max_weight * 0.25, max_weight * 0.5, max_weight * 0.75, max_weight],
            ticktext=["0", "25%", "50%", "75%", f"{max_weight:.1f}"],
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
        # Disable animations untuk smooth rendering tanpa flicker
        transition_duration=0,
        transition_easing="cubic-in-out",
    )

    return fig