"""
viz/editor_plot.py
==================
Membangun figure Plotly untuk editor denah ruangan interaktif.

Fungsi utama:
    build_editor_figure() → go.Figure

Cara kerja editor:
  • Grid divisualisasikan sebagai kumpulan shape rect berwarna di Plotly.
  • Di atas semua shape, ditambahkan scatter plot INVISIBLE (opacity=0) —
    ini yang menangkap event klik dari pengguna.
  • Streamlit membaca koordinat klik via `on_select="rerun"` dan
    mengembalikan titik (x=col, y=row) yang diklik.
  • Logika update state grid ada di ui/panel_design.py, bukan di sini.

Tanggung jawab modul ini: HANYA membangun figure, tidak ada state logic.
"""

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go

from constants import CELL_COLOR, CELL_BORDER, CELL_EMOJI, CELL_EMPTY


def build_editor_figure(
    grid_state: np.ndarray,
    width: int,
    height: int,
    brush_type: int,
) -> go.Figure:
    """
    Bangun figure Plotly interaktif untuk editor denah ruangan.

    Parameter
    ---------
    grid_state : np.ndarray, shape (height, width)
        Matriks nilai int: 0=kosong, 1=kursi, 2=halangan.
    width, height : int
        Dimensi grid (kolom dan baris).
    brush_type : int
        Tipe brush aktif (0/1/2) — digunakan hanya untuk label judul figure.

    Returns
    -------
    go.Figure
        Figure Plotly siap ditampilkan dengan st.plotly_chart(..., on_select="rerun").

    Catatan teknis
    --------------
    Setiap sel dirender sebagai go.Figure shape (rect).
    Scatter markers invisible (rgba(0,0,0,0)) diletakkan di tiap koordinat
    sel agar Plotly bisa menangkap event klik dan mengembalikan (x, y).
    """
    shapes: List[dict]      = []
    annotations: List[dict] = []

    # ── Render tiap sel sebagai shape rect ──────────────────────────────────
    font_size = max(8, min(18, int(280 / max(width, height))))

    for row in range(height):
        for col in range(width):
            ct = int(grid_state[row, col])

            shapes.append(dict(
                type      = "rect",
                x0        = col - 0.48,
                x1        = col + 0.48,
                y0        = row - 0.48,
                y1        = row + 0.48,
                fillcolor = CELL_COLOR[ct],
                line      = dict(color=CELL_BORDER[ct], width=1),
                layer     = "below",
            ))

            emoji = CELL_EMOJI[ct]
            if emoji:
                annotations.append(dict(
                    x         = col,
                    y         = row,
                    text      = emoji,
                    showarrow = False,
                    font      = dict(size=font_size, color="white"),
                    xref      = "x",
                    yref      = "y",
                ))

    # ── Garis pintu masuk (kolom 0, batas kiri) ──────────────────────────────
    shapes.append(dict(
        type  = "line",
        x0    = -0.5, x1=-0.5,
        y0    = -0.5, y1=height - 0.5,
        line  = dict(color="#f5a623", width=3, dash="dot"),
        layer = "above",
    ))
    annotations.append(dict(
        x         = -0.5,
        y         = -0.75,
        text      = "PINTU",
        showarrow = False,
        font      = dict(size=9, color="#f5a623"),
        xref      = "x",
        yref      = "y",
    ))

    # ── Scatter invisible: satu titik per sel untuk menangkap klik ───────────
    xs = [col for col in range(width) for _ in range(height)]
    ys = [row for _ in range(width)   for row in range(height)]

    marker_size = max(14, int(320 / max(width, height)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x             = xs,
        y             = ys,
        mode          = "markers",
        marker        = dict(
            size      = marker_size,
            color     = "rgba(0,0,0,0)",
            symbol    = "square",
        ),
        hovertemplate = "Kol %{x}, Baris %{y}<extra></extra>",
        name          = "cells",
    ))

    # ── Label judul menampilkan brush aktif ──────────────────────────────────
    brush_labels = {0: "🗑️ Hapus", 1: "🪑 Kursi", 2: "🧱 Halangan"}
    title_text   = (
        f"<b>Editor Denah Ruangan</b>"
        f"  —  Brush aktif: {brush_labels[brush_type]}"
    )

    fig.update_layout(
        shapes        = shapes,
        annotations   = annotations,
        xaxis         = dict(
            range      = [-0.8, width - 0.5],
            showgrid   = False,
            zeroline   = False,
            tickvals   = list(range(width)),
            tickfont   = dict(color="#666688", size=8),
            fixedrange = True,
        ),
        yaxis         = dict(
            range      = [height - 0.5, -0.8],
            showgrid   = False,
            zeroline   = False,
            tickvals   = list(range(height)),
            tickfont   = dict(color="#666688", size=8),
            fixedrange = True,
            autorange  = False,
        ),
        plot_bgcolor  = "#0d0d1a",
        paper_bgcolor = "#0d0d1a",
        margin        = dict(l=30, r=10, t=40, b=30),
        height        = max(350, min(550, height * 42)),
        title         = dict(
            text  = title_text,
            font  = dict(color="white", size=13, family="monospace"),
            x     = 0.02,
        ),
        clickmode     = "event",
        dragmode      = False,
        showlegend    = False,
    )

    return fig