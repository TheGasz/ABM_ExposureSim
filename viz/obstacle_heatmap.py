"""
viz/obstacle_heatmap.py
=======================
Visualisasi heatmap exposure per sisi obstacle.

Opsi 2 + 3:
- Incremental exposure field: field numpy diakumulasi langsung dari model,
  bukan di-recompute tiap frame dari obstacle_heatmap dict.
- Render via matplotlib imshow (jauh lebih ringan dari Plotly go.Heatmap).
- Gaussian blur via scipy.ndimage.gaussian_filter (O(n) vs O(n * points)).
"""

import numpy as np  # type: ignore
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore
from typing import Dict, Tuple, Optional

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


# ---------------------------------------------------------------------------
# ExposureField — state yang hidup di dalam model (atau di session_state)
# ---------------------------------------------------------------------------

class ExposureField:
    """Array numpy yang terakumulasi tiap kali agen menyentuh sisi obstacle.

    Cara pakai (di WaitingRoomModel):

        # Inisialisasi (sekali, saat model dibuat)
        self.exposure_field = ExposureField(width, height)

        # Tiap agen lewat sisi obstacle:
        self.exposure_field.record(col, row, side, weight=1.0)

        # obstacle_heatmap dict tetap diupdate seperti biasa untuk hover info.

    Di viz layer, lewatkan `model.exposure_field` ke `plot_obstacle_heatmap`.
    Jika model tidak punya atribut tersebut (backward-compat), fungsi akan
    fall back ke rebuild dari obstacle_heatmap dict.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        ny = max(2, height * GRID_SCALE)
        nx = max(2, width * GRID_SCALE)
        self._field = np.zeros((ny, nx), dtype=np.float32)
        self._dirty = True  # perlu re-blur?
        self._blurred: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Side → pixel koordinat di dalam field array                         #
    # ------------------------------------------------------------------ #
    def _side_pixel(self, col: int, row: int, side: str) -> Tuple[int, int]:
        """Kembalikan (iy, ix) dalam koordinat field array."""
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
        """Tambahkan exposure pada satu sisi obstacle. O(1)."""
        iy, ix = self._side_pixel(col, row, side)
        self._field[iy, ix] += weight
        self._dirty = True

    def get_blurred(self) -> np.ndarray:
        """Return normalized blurred field. Blur hanya dihitung ulang jika dirty."""
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
        """Bangun ExposureField dari obstacle_heatmap dict yang sudah ada.

        Dipakai sebagai fallback jika model lama belum punya ExposureField.
        Lebih lambat dari incremental, tapi jauh lebih cepat dari full meshgrid.
        """
        ef = cls(width, height)
        for (col, row), sides in obstacle_heatmap.items():
            for side, weight in sides.items():
                if weight > 0:
                    ef.record(col, row, side, weight)
        return ef


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
    """Render obstacle exposure heatmap sebagai matplotlib Figure.

    Parameters
    ----------
    obstacle_heatmap : Dict
        {(col, row): {"top": w, "right": w, "bottom": w, "left": w}}
        Dipakai untuk annotasi tooltip/text jika perlu, dan sebagai
        fallback jika exposure_field tidak diberikan.
    layout : Dict
        Grid layout — untuk menggambar overlay obstacle.
    width, height : int
        Dimensi grid.
    exposure_field : ExposureField, optional
        Jika diberikan, field sudah terakumulasi secara incremental (cepat).
        Jika None, di-rebuild dari obstacle_heatmap dict (fallback).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # --- Ambil / build blurred field ---
    if exposure_field is not None:
        field = exposure_field.get_blurred()
    elif obstacle_heatmap:
        ef = ExposureField.from_heatmap_dict(obstacle_heatmap, width, height)
        field = ef.get_blurred()
    else:
        field = np.zeros((max(2, height * GRID_SCALE), max(2, width * GRID_SCALE)), dtype=np.float32)

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
