"""
Modul viz/obstacle_heatmap.py
"""

import io
from typing import Dict, Tuple, Optional

import numpy as np  # type: ignore
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import plotly.graph_objects as go  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore

from constants import CELL_OBSTACLE

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Konstanta rendering
# ---------------------------------------------------------------------------
GRID_SCALE = 4          # resolusi field: 4× ukuran grid (turun dari 12)
BLUR_SIGMA = 1.8        # sigma gaussian_filter dalam pixel field
BG_COLOR = "#0d0d1a"
OBSTACLE_EDGE = "#4a4a7a"
COLORMAP = "viridis"
PLOTLY_COLORSCALE = "Viridis"


# ---------------------------------------------------------------------------
# ExposureField — state yang hidup di dalam model (atau di session_state)
# ---------------------------------------------------------------------------

class ExposureField:
    """Kelas ExposureField."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        ny = max(2, height * GRID_SCALE)
        nx = max(2, width * GRID_SCALE)
        self._field = np.zeros((ny, nx), dtype=np.float32)
        self._dirty = True  # perlu re-blur?
        self._blurred: Optional[np.ndarray] = None
        self._version = 0

    # ------------------------------------------------------------------ #
    # Side → pixel koordinat di dalam field array                         #
    # ------------------------------------------------------------------ #
    def _side_pixel(self, col: int, row: int, side: str) -> Tuple[int, int]:
        """Metode _side_pixel."""
        s = GRID_SCALE
        xc = int((col + 0.5) * s)
        yc = int((row + 0.5) * s)
        x0 = int(col * s)
        y0 = int(row * s)
        x1 = int((col + 1) * s)
        y1 = int((row + 1) * s)
        mapping = {
            "top":    (y0,  xc),
            "bottom": (y1,  xc),
            "left":   (yc,  x0),
            "right":  (yc,  x1),
        }
        iy, ix = mapping.get(side, (yc, xc))
        ny, nx = self._field.shape
        return (min(iy, ny - 1), min(ix, nx - 1))

    def record(self, col: int, row: int, side: str, weight: float = 1.0) -> None:
        """Metode record."""
        iy, ix = self._side_pixel(col, row, side)
        self._field[iy, ix] += weight
        self._dirty = True
        self._version += 1

    def get_version(self) -> int:
        return self._version

    def get_raw_copy(self) -> np.ndarray:
        return self._field.copy()

    def get_blurred(self) -> np.ndarray:
        """Metode get_blurred."""
        if self._dirty or self._blurred is None:
            blurred = gaussian_filter(self._field, sigma=BLUR_SIGMA)
            max_val = blurred.max()
            self._blurred = blurred / max_val if max_val > 0 else blurred
            self._dirty = False
        return self._blurred

    def reset(self) -> None:
        self._field[:] = 0.0
        self._blurred = None
        self._dirty = True
        self._version = 0

    # ------------------------------------------------------------------ #
    # Rebuild dari obstacle_heatmap dict (fallback / backward-compat)     #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_heatmap_dict(
        cls,
        obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
        width: int,
        height: int,
    ) -> "ExposureField":
        """Metode from_heatmap_dict."""
        ef = cls(width, height)
        for (col, row), sides in obstacle_heatmap.items():
            for side, weight in sides.items():
                if weight > 0:
                    ef.record(col, row, side, weight)
        return ef


def _empty_field(width: int, height: int) -> np.ndarray:
    return np.zeros((max(2, height * GRID_SCALE), max(2, width * GRID_SCALE)), dtype=np.float32)


def build_blurred_field_from_raw(raw_field: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(raw_field, sigma=BLUR_SIGMA)
    max_val = blurred.max()
    return blurred / max_val if max_val > 0 else blurred


def build_obstacle_heatmap_field(
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    width: int,
    height: int,
    exposure_field: Optional[ExposureField] = None,
) -> np.ndarray:
    if exposure_field is not None:
        return exposure_field.get_blurred()
    if obstacle_heatmap:
        ef = ExposureField.from_heatmap_dict(obstacle_heatmap, width, height)
        return ef.get_blurred()
    return _empty_field(width, height)


def _format_obstacle_hover(
    col: int,
    row: int,
    sides: Dict[str, float],
) -> str:
    top = sides.get("top", 0.0)
    right = sides.get("right", 0.0)
    bottom = sides.get("bottom", 0.0)
    left = sides.get("left", 0.0)
    return (
        f"Obstacle ({col}, {row})<br>"
        f"Top: {top:.2f}<br>"
        f"Right: {right:.2f}<br>"
        f"Bottom: {bottom:.2f}<br>"
        f"Left: {left:.2f}"
    )


def build_obstacle_heatmap_plotly(
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    layout: Dict[Tuple[int, int], int],
    width: int,
    height: int,
    exposure_field: Optional[ExposureField] = None,
) -> "go.Figure":
    field = build_obstacle_heatmap_field(obstacle_heatmap, width, height, exposure_field)
    ny, nx = field.shape
    xs = np.linspace(0, width, nx)
    ys = np.linspace(0, height, ny)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=field,
            x=xs,
            y=ys,
            colorscale=PLOTLY_COLORSCALE,
            zmin=0.0,
            zmax=1.0,
            showscale=True,
            colorbar=dict(title=dict(text="Exposure"), thickness=12),
            zsmooth="best",
            hoverinfo="skip",
        )
    )

    shapes = []
    hover_x = []
    hover_y = []
    hover_text = []

    for (col, row), cell_type in layout.items():
        if cell_type != CELL_OBSTACLE or not (0 <= row < height and 0 <= col < width):
            continue
        shapes.append(
            dict(
                type="rect",
                x0=col,
                x1=col + 1,
                y0=row,
                y1=row + 1,
                line=dict(color=OBSTACLE_EDGE, width=1),
                fillcolor="rgba(0,0,0,0)",
            )
        )
        hover_x.append(col + 0.5)
        hover_y.append(row + 0.5)
        hover_text.append(_format_obstacle_hover(col, row, obstacle_heatmap.get((col, row), {})))

    if hover_x:
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker=dict(size=24, color="rgba(255,255,255,0.01)", symbol="square"),
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        shapes=shapes,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text="🔥 Obstacle Exposure Heatmap",
            font=dict(color="white", size=12, family="monospace"),
            x=0.02,
        ),
        hovermode="closest",
    )
    fig.update_xaxes(range=[0, width], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[height, 0], showgrid=False, zeroline=False, visible=False)
    return fig


def _add_obstacle_detail_text(
    ax: plt.Axes,
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    width: int,
    height: int,
) -> None:
    for (col, row), sides in obstacle_heatmap.items():
        if not (0 <= col < width and 0 <= row < height):
            continue
        text = (
            f"T:{sides.get('top', 0.0):.2f} R:{sides.get('right', 0.0):.2f}\n"
            f"B:{sides.get('bottom', 0.0):.2f} L:{sides.get('left', 0.0):.2f}"
        )
        ax.text(
            col + 0.5,
            row + 0.5,
            text,
            ha="center",
            va="center",
            fontsize=5.5,
            color="white",
            family="monospace",
        )


class HeatmapRenderer:
    """Kelas HeatmapRenderer."""

    def __init__(self, width: int, height: int, layout: Dict[Tuple[int, int], int]) -> None:
        self.width = width
        self.height = height
        self.layout = layout

        field = _empty_field(width, height)
        self.fig, self.ax = plt.subplots(figsize=(4.2, 4.2), facecolor=BG_COLOR)
        self.ax.set_facecolor(BG_COLOR)

        self._im = self.ax.imshow(
            field,
            origin="upper",
            cmap=COLORMAP,
            vmin=0.0,
            vmax=1.0,
            extent=[0, width, height, 0],
            interpolation="bilinear",
            aspect="equal",
        )

        cbar = self.fig.colorbar(self._im, ax=self.ax, fraction=0.035, pad=0.02)
        cbar.set_label("Exposure", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

        obstacle_cells = [
            (col, row)
            for (col, row), cell_type in layout.items()
            if cell_type == CELL_OBSTACLE and 0 <= row < height and 0 <= col < width
        ]
        for (col, row) in obstacle_cells:
            rect = plt.Rectangle(
                (col, row), 1, 1,
                linewidth=0.7,
                edgecolor=OBSTACLE_EDGE,
                facecolor="none",
            )
            self.ax.add_patch(rect)

        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.tick_params(colors="#555555", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#333355")

        self.ax.set_title(" Obstacle Exposure Heatmap", color="white", fontsize=9,
                          fontfamily="monospace", loc="left", pad=6)

        self.fig.tight_layout(pad=0.4)

    def render(self, field: np.ndarray) -> bytes:
        self._im.set_data(field)
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    def close(self) -> None:
        plt.close(self.fig)


class HeatmapRenderWorker:
    """Kelas HeatmapRenderWorker."""

    def __init__(self, width: int, height: int, layout: Dict[Tuple[int, int], int]) -> None:
        self.width = width
        self.height = height
        self.layout = layout
        self._renderer: Optional[HeatmapRenderer] = None
        self._layout_signature: Optional[int] = None

    def render(self, raw_field: np.ndarray, layout_signature: int) -> bytes:
        if self._renderer is None or layout_signature != self._layout_signature:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = HeatmapRenderer(self.width, self.height, self.layout)
            self._layout_signature = layout_signature

        field = build_blurred_field_from_raw(raw_field)
        return self._renderer.render(field)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def render_heatmap_details_png(
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    layout: Dict[Tuple[int, int], int],
    width: int,
    height: int,
    exposure_field: Optional[ExposureField] = None,
) -> bytes:
    field = build_obstacle_heatmap_field(obstacle_heatmap, width, height, exposure_field)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    im = ax.imshow(
        field,
        origin="upper",
        cmap=COLORMAP,
        vmin=0.0,
        vmax=1.0,
        extent=[0, width, height, 0],
        interpolation="bilinear",
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Exposure", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

    obstacle_cells = [
        (col, row)
        for (col, row), cell_type in layout.items()
        if cell_type == CELL_OBSTACLE and 0 <= row < height and 0 <= col < width
    ]
    for (col, row) in obstacle_cells:
        rect = plt.Rectangle(
            (col, row), 1, 1,
            linewidth=0.7,
            edgecolor=OBSTACLE_EDGE,
            facecolor="none",
        )
        ax.add_patch(rect)

    _add_obstacle_detail_text(ax, obstacle_heatmap, width, height)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.set_title("🔥 Obstacle Exposure Heatmap", color="white", fontsize=9,
                 fontfamily="monospace", loc="left", pad=6)

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fungsi render utama — matplotlib, bukan Plotly
# ---------------------------------------------------------------------------

def plot_obstacle_heatmap(
    obstacle_heatmap: Dict[Tuple[int, int], Dict[str, float]],
    layout: Dict[Tuple[int, int], int],
    width: int,
    height: int,
    exposure_field: Optional[ExposureField] = None,
) -> plt.Figure:
    """Fungsi plot_obstacle_heatmap."""
    # --- Ambil / build blurred field ---
    field = build_obstacle_heatmap_field(obstacle_heatmap, width, height, exposure_field)

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # --- Heatmap ---
    im = ax.imshow(
        field,
        origin="upper",
        cmap=COLORMAP,
        vmin=0.0,
        vmax=1.0,
        extent=[0, width, height, 0],
        interpolation="bilinear",   # ringan, smooth cukup
        aspect="equal",
    )

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Exposure", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

    # --- Overlay obstacle rectangles ---
    obstacle_cells = [
        (col, row)
        for (col, row), cell_type in layout.items()
        if cell_type == CELL_OBSTACLE and 0 <= row < height and 0 <= col < width
    ]
    for (col, row) in obstacle_cells:
        rect = plt.Rectangle(
            (col, row), 1, 1,
            linewidth=0.7,
            edgecolor=OBSTACLE_EDGE,
            facecolor="none",
        )
        ax.add_patch(rect)

    # --- Axes styling ---
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)   # y terbalik: row 0 di atas
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.set_title("🔥 Obstacle Exposure Heatmap", color="white", fontsize=9,
                 fontfamily="monospace", loc="left", pad=6)

    fig.tight_layout(pad=0.4)
    return fig
