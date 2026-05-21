"""
viz/editor_plot.py
==================
Membangun figure Plotly untuk editor denah ruangan interaktif.

Mendukung: kursi dengan arah hadap, pintu masuk, grid besar.
"""

from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go

from constants import (
    CELL_COLOR, CELL_BORDER, CELL_EMOJI, CELL_EMPTY,
    CELL_CHAIR, CELL_DOOR, CHAIR_DIR_EMOJI,
)


def build_editor_figure(
    grid_state: np.ndarray,
    width: int,
    height: int,
    brush_type: int,
    chair_directions: Dict[Tuple[int, int], str] = None,
) -> go.Figure:
    """
    Bangun figure Plotly interaktif untuk editor denah ruangan.

    Parameter
    ---------
    grid_state : np.ndarray, shape (height, width)
    width, height : int
    brush_type : int — brush aktif (0/1/2/3)
    chair_directions : dict — {(col, row): "up"|"down"|"left"|"right"}
    """
    if chair_directions is None:
        chair_directions = {}

    shapes: List[dict] = []
    annotations: List[dict] = []

    # Adaptive sizing untuk grid besar
    max_dim = max(width, height)
    font_size = max(6, min(18, int(280 / max_dim)))
    dir_font_size = max(5, min(12, int(200 / max_dim)))

    for row in range(height):
        for col in range(width):
            ct = int(grid_state[row, col])

            shapes.append(dict(
                type="rect",
                x0=col - 0.48, x1=col + 0.48,
                y0=row - 0.48, y1=row + 0.48,
                fillcolor=CELL_COLOR[ct],
                line=dict(color=CELL_BORDER[ct], width=1),
                layer="below",
            ))

            emoji = CELL_EMOJI[ct]
            if emoji:
                annotations.append(dict(
                    x=col, y=row, text=emoji, showarrow=False,
                    font=dict(size=font_size, color="white"),
                    xref="x", yref="y",
                ))

            # Tampilkan panah arah untuk kursi
            if ct == CELL_CHAIR and (col, row) in chair_directions:
                dir_emoji = CHAIR_DIR_EMOJI.get(chair_directions[(col, row)], "")
                if dir_emoji:
                    annotations.append(dict(
                        x=col, y=row - 0.32,
                        text=dir_emoji, showarrow=False,
                        font=dict(size=dir_font_size, color="#f5a623"),
                        xref="x", yref="y",
                    ))

    # Garis penanda pintu masuk untuk setiap CELL_DOOR
    door_positions = []
    for row in range(height):
        for col in range(width):
            if int(grid_state[row, col]) == CELL_DOOR:
                door_positions.append((col, row))

    # Jika tidak ada pintu, tampilkan garis fallback di kolom 0
    if not door_positions:
        shapes.append(dict(
            type="line",
            x0=-0.5, x1=-0.5,
            y0=-0.5, y1=height - 0.5,
            line=dict(color="#f5a623", width=2, dash="dot"),
            layer="above",
        ))
        annotations.append(dict(
            x=-0.5, y=-0.75, text="PINTU (fallback)",
            showarrow=False,
            font=dict(size=8, color="#f5a623"),
            xref="x", yref="y",
        ))

    # Scatter invisible untuk menangkap klik
    xs = [col for col in range(width) for _ in range(height)]
    ys = [row for _ in range(width) for row in range(height)]
    marker_size = max(8, int(320 / max_dim))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=marker_size, color="rgba(0,0,0,0)", symbol="square"),
        hovertemplate="Kol %{x}, Baris %{y}<extra></extra>",
        name="cells",
    ))

    brush_labels = {0: "🗑️ Hapus", 1: "🪑 Kursi", 2: "🧱 Halangan", 3: "🚪 Pintu"}
    title_text = (
        f"<b>Editor Denah Ruangan</b>"
        f"  —  Brush: {brush_labels.get(brush_type, '?')}"
    )

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(
            range=[-0.8, width - 0.5], showgrid=False, zeroline=False,
            tickvals=list(range(width)),
            tickfont=dict(color="#666688", size=max(5, 8 - max_dim // 30)),
            fixedrange=True,
        ),
        yaxis=dict(
            range=[height - 0.5, -0.8], showgrid=False, zeroline=False,
            tickvals=list(range(height)),
            tickfont=dict(color="#666688", size=max(5, 8 - max_dim // 30)),
            fixedrange=True, autorange=False,
        ),
        plot_bgcolor="#0d0d1a", paper_bgcolor="#0d0d1a",
        margin=dict(l=30, r=10, t=40, b=30),
        height=max(300, min(700, height * max(15, 42 - max_dim // 5))),
        title=dict(text=title_text, font=dict(color="white", size=13, family="monospace"), x=0.02),
        clickmode="event", dragmode=False, showlegend=False,
    )
    return fig