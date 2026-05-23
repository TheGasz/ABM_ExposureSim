"""
viz/sim_plots.py
================
Matplotlib plots for the simulation view.

Optimasi: pisahkan static layer (obstacles, chairs, doors) dari dynamic layer
(posisi agen). Static layer di-render sekali ke background image, dynamic layer
di-update in-place pada axes yang sama — menghindari pembuatan figure baru
setiap frame sehingga animasi bisa berjalan lancar di fps tinggi.
"""

import io
import math

import numpy as np
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle, Polygon

from constants import SIM_COLOR


class SimRenderer:
    """
    Renderer yang reuse figure/axes yang sama antar frame.

    Alur:
    1. Buat instance sekali saat simulasi dimulai.
    2. Panggil render(snap) setiap frame → kembalikan PNG bytes.
    3. Tampilkan dengan st.image(bytes, ...) — lebih cepat dari st.pyplot.
    """

    def __init__(self, width: int, height: int, cell_size_px: int) -> None:
        self.width = width
        self.height = height
        self.cell_size_px = cell_size_px

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

        # Scatter artists untuk dynamic layer — dibuat sekali, di-update tiap frame
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

        # Pool Polygon patches untuk vision cone — di-reuse antar frame
        # agar tidak ada alokasi baru setiap render.
        self._vision_patches: list = []

        # Static layer belum digambar — akan digambar saat render pertama
        self._static_drawn = False

    def _draw_static(self, snap: dict) -> None:
        """Gambar obstacles, chairs, doors — hanya sekali per instance."""
        cell_size_px = self.cell_size_px

        def add_rect(col: int, row: int, fc: str, ec: str, alpha: float,
                     z: int, ls: str = "solid", lw: float = 1.0) -> None:
            self.ax.add_patch(Rectangle(
                (col * cell_size_px, row * cell_size_px),
                cell_size_px, cell_size_px,
                facecolor=fc, edgecolor=ec, linewidth=lw,
                alpha=alpha, zorder=z, linestyle=ls,
            ))

        for col, row in snap.get("obstacles", []):
            add_rect(col, row, SIM_COLOR["obstacle"], SIM_COLOR["obstacle"], 1.0, 2)

        # Chairs: gambar semua empty dulu; occupied akan di-overlay via scatter
        all_chairs = list(snap.get("chairs_empty", [])) + list(snap.get("chairs_full", []))
        for col, row in all_chairs:
            add_rect(col, row, SIM_COLOR["chair_empty"], SIM_COLOR["chair_empty"], 0.9, 3)

        for col, row in snap.get("doors", []):
            add_rect(col, row, "none", SIM_COLOR["door"], 1.0, 5, ls="dashed", lw=2.0)

        self._static_drawn = True

    def _update_chairs(self, snap: dict) -> None:
        """Update warna chair full/empty dengan menggambar ulang hanya patch kursi."""
        # Hapus patch kursi lama (zorder 3 dan 4), gambar ulang
        # Lebih mudah: simpan referensi patch kursi dan update facecolor-nya.
        # Untuk simplisitas, gunakan scatter overlay untuk chairs_full.
        # Chair penuh ditampilkan via scatter terpisah (zorder 4).
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

    @staticmethod
    def _points_to_offsets(points: List[Tuple[float, float]]):
        if not points:
            return []
        return list(points)

    def _update_vision(self, snap: dict) -> None:
        """Update polygon vision setiap agen bergerak. Reuse patch pool.

        humans_vision sekarang berformat List[List[Tuple[float,float]]]:
        satu polygon (sudah di-ray-cast, obstacle-aware) per agen.
        """
        vision_polygons = snap.get("humans_vision", [])

        n_needed = len(vision_polygons)
        n_have   = len(self._vision_patches)

        # Tambah Polygon baru jika pool kurang
        for _ in range(n_needed - n_have):
            p = Polygon(
                np.zeros((3, 2)),   # placeholder vertices
                closed=True,
                facecolor="#ffffff",
                edgecolor="none",
                alpha=0.06,
                zorder=5,
                linewidth=0,
            )
            self.ax.add_patch(p)
            self._vision_patches.append(p)

        # Update patch yang dipakai
        for i, poly_pts in enumerate(vision_polygons):
            if len(poly_pts) >= 3:
                self._vision_patches[i].set_xy(np.array(poly_pts))
                self._vision_patches[i].set_visible(True)
            else:
                self._vision_patches[i].set_visible(False)

        # Sembunyikan patch yang tidak dipakai
        for i in range(n_needed, n_have):
            self._vision_patches[i].set_visible(False)

    def render(self, snap: dict) -> bytes:
        """
        Update dynamic layer dan kembalikan PNG sebagai bytes.
        Jauh lebih cepat dari membuat figure baru setiap frame.
        """
        if not self._static_drawn:
            self._draw_static(snap)

        self._update_chairs(snap)
        self._update_vision(snap)

        # Update posisi agen — hanya set_offsets, tidak ada alokasi baru
        def _upd(sc: PathCollection, points: List[Tuple[float, float]]) -> None:
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
        self.fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    def close(self) -> None:
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Fungsi standalone (dipakai saat pause / render tunggal)
# ---------------------------------------------------------------------------

def plot_sim_room(snap: dict, width: int, height: int, cell_size_px: int) -> plt.Figure:
    """Buat figure baru sekali pakai. Gunakan SimRenderer untuk animasi."""
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

    draw_cells(snap.get("obstacles", []),    SIM_COLOR["obstacle"],    SIM_COLOR["obstacle"],    1.0, 2)
    draw_cells(snap.get("chairs_empty", []), SIM_COLOR["chair_empty"], SIM_COLOR["chair_empty"], 0.9, 3)
    draw_cells(snap.get("chairs_full", []),  SIM_COLOR["chair_full"],  SIM_COLOR["chair_full"],  1.0, 4)

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