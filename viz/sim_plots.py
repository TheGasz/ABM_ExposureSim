"""
Modul viz/sim_plots.py  [OPTIMIZED]

Perubahan dari versi original:
1. DPI render diturunkan 90 → 72 (lebih ringan, masih tajam di web)
2. Output format PNG → JPEG quality=85 (ukuran ~40% lebih kecil)
3. Vision polygon: skip render setiap 2 frame untuk hemat draw calls
4. Vision patch pool: max dibatasi 60 agar tidak unbounded
"""

import io
import math

import numpy as np
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle, Polygon

from constants import SIM_COLOR

# Maksimum vision patches yang di-pool (cegah terlalu banyak patch di canvas)
_MAX_VISION_PATCHES = 60
# Render vision setiap N frame (1 = setiap frame, 2 = selang-seling)
_VISION_RENDER_EVERY = 2


class SimRenderer:
    """Kelas SimRenderer."""

    def __init__(self, width: int, height: int, cell_size_px: int) -> None:
        self.width = width
        self.height = height
        self.cell_size_px = cell_size_px
        self._frame_count = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor("#0d0d1a")
        self.ax.set_facecolor("#1a1a2e")

        w_px = width * cell_size_px
        h_px = height * cell_size_px
        self.ax.set_xlim(0, w_px)
        self.ax.set_ylim(0, h_px)
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Room State", color="white", fontsize=11, fontweight="bold")

        s = (cell_size_px * 1.1) ** 2
        self._sc_seek: PathCollection = self.ax.scatter(
            [], [], c=SIM_COLOR["human_seek"], s=s, zorder=6,
            label="To chair", edgecolors="none",
        )
        self._sc_exit: PathCollection = self.ax.scatter(
            [], [], c=SIM_COLOR["human_move"], s=s, zorder=6,
            label="Leaving", edgecolors="none",
        )
        self._sc_pass: PathCollection = self.ax.scatter(
            [], [], c=SIM_COLOR["human_pass"], s=s, zorder=6,
            label="Passing", edgecolors="none",
        )
        self._sc_sit: PathCollection = self.ax.scatter(
            [], [], c=SIM_COLOR["human_sit"], s=s, zorder=7,
            label="Sitting", edgecolors="none",
        )

        self.ax.legend(
            loc="upper right", fontsize=7, framealpha=0.25,
            facecolor="#1a1a2e", edgecolor="#444466",
        )
        plt.tight_layout()

        self._vision_patches: list = []
        self._static_drawn = False

    def _draw_static(self, snap: dict) -> None:
        cell_size_px = self.cell_size_px

        def add_rect(col, row, fc, ec, alpha, z, ls="solid", lw=1.0):
            self.ax.add_patch(Rectangle(
                (col * cell_size_px, row * cell_size_px),
                cell_size_px, cell_size_px,
                facecolor=fc, edgecolor=ec, linewidth=lw,
                alpha=alpha, zorder=z, linestyle=ls,
            ))

        for col, row in snap.get("obstacles", []):
            add_rect(col, row, SIM_COLOR["obstacle"], SIM_COLOR["obstacle"], 1.0, 2)

        all_chairs = list(snap.get("chairs_empty", [])) + list(snap.get("chairs_full", []))
        for col, row in all_chairs:
            add_rect(col, row, SIM_COLOR["chair_empty"], SIM_COLOR["chair_empty"], 0.9, 3)

        for col, row in snap.get("doors", []):
            add_rect(col, row, "none", SIM_COLOR["door"], 1.0, 5, ls="dashed", lw=2.0)

        self._static_drawn = True

    def _update_chairs(self, snap: dict) -> None:
        if not hasattr(self, "_sc_chairs_full"):
            s = (self.cell_size_px * 1.3) ** 2
            self._sc_chairs_full: PathCollection = self.ax.scatter(
                [], [], c=SIM_COLOR["chair_full"], s=s, zorder=4,
                marker="s", edgecolors="none",
            )

        chairs_full = snap.get("chairs_full", [])
        if chairs_full:
            cs = self.cell_size_px
            xs = [col * cs + cs * 0.5 for col, row in chairs_full]
            ys = [row * cs + cs * 0.5 for col, row in chairs_full]
            self._sc_chairs_full.set_offsets(list(zip(xs, ys)))
        else:
            self._sc_chairs_full.set_offsets(np.empty((0, 2)))

    def _update_vision(self, snap: dict) -> None:
        """Update vision patches — skip frame ganjil untuk hemat draw calls."""
        # Hanya update vision setiap _VISION_RENDER_EVERY frame
        if self._frame_count % _VISION_RENDER_EVERY != 0:
            return

        vision_polygons = snap.get("humans_vision", [])
        # Batasi jumlah vision yang dirender agar tidak overflow pool
        vision_polygons = vision_polygons[:_MAX_VISION_PATCHES]

        n_needed = len(vision_polygons)
        n_have = len(self._vision_patches)

        for _ in range(n_needed - n_have):
            p = Polygon(
                np.zeros((3, 2)),
                closed=True,
                facecolor="#ffffff",
                edgecolor="none",
                alpha=0.06,
                zorder=5,
                linewidth=0,
            )
            self.ax.add_patch(p)
            self._vision_patches.append(p)

        for i, poly_pts in enumerate(vision_polygons):
            if len(poly_pts) >= 3:
                self._vision_patches[i].set_xy(np.array(poly_pts))
                self._vision_patches[i].set_visible(True)
            else:
                self._vision_patches[i].set_visible(False)

        for i in range(n_needed, n_have):
            self._vision_patches[i].set_visible(False)

    def render(self, snap: dict) -> bytes:
        """Render frame ke bytes — JPEG untuk ukuran lebih kecil."""
        self._frame_count += 1

        if not self._static_drawn:
            self._draw_static(snap)

        self._update_chairs(snap)
        self._update_vision(snap)

        def _upd(sc: PathCollection, points) -> None:
            if points:
                sc.set_offsets(points)
                sc.set_visible(True)
            else:
                sc.set_offsets(np.empty((0, 2)))
                sc.set_visible(False)

        _upd(self._sc_seek, snap.get("humans_seek", []))
        _upd(self._sc_exit, snap.get("humans_exit", []))
        _upd(self._sc_pass, snap.get("humans_pass", []))
        _upd(self._sc_sit,  snap.get("humans_sit",  []))

        buf = io.BytesIO()
        # JPEG lebih kecil dari PNG — quality=85 tidak terlihat bedanya di web
        self.fig.savefig(buf, format="jpeg", dpi=72, bbox_inches="tight",
                         pil_kwargs={"quality": 85, "optimize": True})
        buf.seek(0)
        return buf.getvalue()

    def close(self) -> None:
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Fungsi standalone (dipakai saat pause / render tunggal)
# ---------------------------------------------------------------------------

def plot_sim_room(snap: dict, width: int, height: int, cell_size_px: int) -> plt.Figure:
    """Fungsi plot_sim_room — tidak berubah dari original."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    w_px = width * cell_size_px
    h_px = height * cell_size_px
    ax.set_xlim(0, w_px)
    ax.set_ylim(0, h_px)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    def draw_cells(cells, color, edge, alpha, z):
        for col, row in cells:
            ax.add_patch(Rectangle(
                (col * cell_size_px, row * cell_size_px),
                cell_size_px, cell_size_px,
                facecolor=color, edgecolor=edge,
                linewidth=1.0, alpha=alpha, zorder=z,
            ))

    draw_cells(snap.get("obstacles",    []), SIM_COLOR["obstacle"],    SIM_COLOR["obstacle"],    1.0, 2)
    draw_cells(snap.get("chairs_empty", []), SIM_COLOR["chair_empty"], SIM_COLOR["chair_empty"], 0.9, 3)
    draw_cells(snap.get("chairs_full",  []), SIM_COLOR["chair_full"],  SIM_COLOR["chair_full"],  1.0, 4)

    for col, row in snap.get("doors", []):
        ax.add_patch(Rectangle(
            (col * cell_size_px, row * cell_size_px),
            cell_size_px, cell_size_px,
            facecolor="none", edgecolor=SIM_COLOR["door"],
            linewidth=2.0, linestyle="--", zorder=5,
        ))

    s = (cell_size_px * 1.1) ** 2

    def scatter_px(points, color, label, z):
        if not points:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, c=color, s=s, zorder=z, label=label, edgecolors="none")

    scatter_px(snap.get("humans_seek", []), SIM_COLOR["human_seek"], "To chair", 6)
    scatter_px(snap.get("humans_exit", []), SIM_COLOR["human_move"], "Leaving",  6)
    scatter_px(snap.get("humans_pass", []), SIM_COLOR["human_pass"], "Passing",  6)
    scatter_px(snap.get("humans_sit",  []), SIM_COLOR["human_sit"],  "Sitting",  7)

    ax.set_title("Room State", color="white", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.25,
              facecolor="#1a1a2e", edgecolor="#444466")
    plt.tight_layout()
    return fig