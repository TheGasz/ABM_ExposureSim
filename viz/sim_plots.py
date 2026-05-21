"""
viz/sim_plots.py
================
Fungsi visualisasi matplotlib untuk panel simulasi.

Dua fungsi utama:
    plot_sim_room()      → Figure kondisi ruangan real-time
    plot_dual_heatmap()  → Figure dua heatmap akumulatif (floor + obstacle)

Modul ini murni visualisasi:
  • Menerima data (snapshot dict, numpy array) sebagai input.
  • Mengembalikan plt.Figure sebagai output.
  • Tidak ada akses ke session_state atau model Mesa secara langsung.
"""

from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import CELL_OBSTACLE, Layout, SIM_COLOR


# =============================================================================
# PLOT 1 — KONDISI RUANGAN (REAL-TIME)
# =============================================================================

def plot_sim_room(snap: dict, width: int, height: int) -> plt.Figure:
    """
    Render kondisi ruangan simulasi pada satu step tertentu.

    Elemen yang divisualisasikan:
      • Sorot mata lantai  (hijau neon, alpha rendah)  — background
      • Sorot mata tiang   (oranye, alpha sedang)       — highlight
      • ObstacleAgent      (kotak abu)
      • ChairAgent kosong  (pentagon biru gelap)
      • ChairAgent terisi  (pentagon biru sedang)
      • CustomerAgent SEEKING (lingkaran merah)
      • CustomerAgent MOVING  (lingkaran biru muda)
      • CustomerAgent SITTING (lingkaran oranye)
      • Garis pintu masuk  (kolom 0, oranye putus-putus)

    Parameter
    ---------
    snap : dict
        Output dari WaitingRoomModel.get_grid_snapshot().
    width, height : int
        Dimensi grid untuk mengatur batas axis.

    Returns
    -------
    plt.Figure — tutup dengan plt.close(fig) setelah ditampilkan.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    # Konfigurasi axis
    ax.set_xlim(-0.5, width  - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.tick_params(colors="#444466", labelsize=5)
    ax.grid(color="#252545", linewidth=0.4, zorder=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()   # row 0 di atas (konsisten dengan konvensi grid)

    # ── Helper scatter ────────────────────────────────────────────────────────
    def scatter(
        coords: List[Tuple[int, int]],
        color: str,
        marker: str,
        size: int,
        zorder: int,
        alpha: float = 1.0,
        label: str   = None,
    ):
        """Render titik-titik agen ke axis. Skip jika list kosong."""
        if not coords:
            return
        xs, ys = zip(*coords)
        ax.scatter(
            ys, xs,
            c          = color,
            marker     = marker,
            s          = size,
            zorder     = zorder,
            alpha      = alpha,
            label      = label,
            edgecolors = "none",
        )

    # ── Layer 1: Sorot mata lantai (hijau, paling bawah) ─────────────────────
    for (x, y) in snap["gaze_floor"]:
        ax.add_patch(plt.Rectangle(
            (y - 0.5, x - 0.5), 1, 1,
            color  = SIM_COLOR["gaze_floor"],
            alpha  = 0.13,
            zorder = 1,
        ))

    # ── Layer 2: Sorot mata tiang (oranye, sedikit lebih opaque) ─────────────
    for (x, y) in snap["gaze_obstacle"]:
        ax.add_patch(plt.Rectangle(
            (y - 0.5, x - 0.5), 1, 1,
            color  = SIM_COLOR["gaze_obstacle"],
            alpha  = 0.35,
            zorder = 2,
        ))

    # ── Layer 3: Agen statis & dinamis ───────────────────────────────────────
    scatter(snap["obstacles"],      SIM_COLOR["obstacle"],      "s", 200, 3, label="Tiang/Halangan")
    scatter(snap["chairs_empty"],   SIM_COLOR["chair_empty"],   "p", 160, 3, alpha=0.7, label="Kursi Kosong")
    scatter(snap["chairs_full"],    SIM_COLOR["chair_full"],    "p", 180, 4, label="Kursi Terisi")
    scatter(snap["customers_seek"], SIM_COLOR["customer_seek"], "o", 140, 5, label="Mencari Kursi")
    scatter(snap["customers_move"], SIM_COLOR["customer_move"], "o", 140, 5, label="Bergerak")
    scatter(snap["customers_sit"],  SIM_COLOR["customer_sit"],  "o", 170, 5, label="Duduk")

    # ── Legenda & label ───────────────────────────────────────────────────────
    ax.legend(
        loc        = "upper right",
        fontsize   = 6.5,
        framealpha = 0.25,
        facecolor  = "#1a1a2e",
        edgecolor  = "#444466",
        labelcolor = "white",
    )
    ax.set_title(
        "Kondisi Ruangan (Real-time)",
        color="white", fontsize=10, fontweight="bold", pad=5,
    )
    ax.set_xlabel("Kolom", color="#888899", fontsize=7)
    ax.set_ylabel("Baris",  color="#888899", fontsize=7)

    # ── Penanda pintu masuk (kolom 0) ─────────────────────────────────────────
    ax.axvline(x=-0.5, color="#f5a623", linewidth=2, linestyle=":", alpha=0.8)
    ax.text(-0.48, height - 0.7, "PINTU",
            color="white", fontsize=6, rotation=90, va="top",
            bbox=dict(facecolor="#f5a623", alpha=0.7, pad=1))

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 2 — DUAL HEATMAP AKUMULATIF
# =============================================================================

def plot_dual_heatmap(
    floor_hm: np.ndarray,
    obstacle_hm: np.ndarray,
    layout: Layout,
    width: int,
    height: int,
) -> plt.Figure:
    """
    Render dua heatmap akumulatif secara berdampingan.

    Kiri  — Floor Exposure:
        Menunjukkan area lantai yang paling sering tersorot pandangan pelanggan.
        Rekomendasi: banner lantai, display stand, signage vertikal.

    Kanan — Obstacle Exposure:
        Menunjukkan tiang/halangan yang paling sering "dilihat" pelanggan
        (ray berhenti di sana). Rekomendasi: stiker tiang, banner menempel.

    Anotasi:
      • Posisi obstacle di-overlay dengan emoji 🧱 + border abu.
      • TOP-3 lokasi terpanas diberi border berwarna dan label #1, #2, #3.

    Parameter
    ---------
    floor_hm, obstacle_hm : np.ndarray, shape (width, height)
        Heatmap dari WaitingRoomModel.floor_heatmap / obstacle_heatmap.
    layout : Layout
        Dict layout editor — digunakan untuk mengetahui posisi obstacle.
    width, height : int
        Dimensi grid.

    Returns
    -------
    plt.Figure — tutup dengan plt.close(fig) setelah ditampilkan.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d1a")

    # Konfigurasi per subplot
    panel_configs = [
        {
            "title"     : "🟢 Floor Exposure\n(Banner Lantai / Display Stand)",
            "heatmap"   : floor_hm.T,         # Transpose: seaborn (row=y, col=x)
            "source"    : floor_hm,
            "cmap"      : "YlOrRd",
            "rank_color": "cyan",
            "obs_only"  : False,               # Tampilkan rank di sel lantai
        },
        {
            "title"     : "🟠 Obstacle Exposure\n(Stiker / Banner di Tiang)",
            "heatmap"   : obstacle_hm.T,
            "source"    : obstacle_hm,
            "cmap"      : "plasma",
            "rank_color": "lime",
            "obs_only"  : True,                # Rank hanya di sel obstacle
        },
    ]

    obstacle_positions: Set[Tuple[int, int]] = {
        k for k, v in layout.items() if v == CELL_OBSTACLE
    }

    for cfg, ax in zip(panel_configs, axes):
        ax.set_facecolor("#0d0d1a")

        # Hindari heatmap semua-nol yang menyebabkan error seaborn
        data = cfg["heatmap"].copy()
        if data.max() == 0:
            data += 1e-9

        sns.heatmap(
            data,
            ax       = ax,
            cmap     = cfg["cmap"],
            linewidths= 0,
            square   = True,
            cbar_kws = {"label": "Exposure Hits", "shrink": 0.85},
            xticklabels = False,
            yticklabels = False,
        )

        # ── Overlay posisi obstacle ───────────────────────────────────────────
        for (col, row) in obstacle_positions:
            ax.add_patch(plt.Rectangle(
                (row + 0.05, col + 0.05), 0.9, 0.9,
                fill      = False,
                edgecolor = "#6a6a9a",
                linewidth = 1.5,
                zorder    = 4,
            ))
            ax.text(
                row + 0.5, col + 0.5, "🧱",
                ha="center", va="center", fontsize=7, zorder=5,
            )

        # ── TOP-3 lokasi terpanas ─────────────────────────────────────────────
        src  = cfg["source"]
        tops = _get_top_locations(src, n=3, obstacle_only=cfg["obs_only"],
                                  obstacle_positions=obstacle_positions)

        for rank, (rx, ry, _) in enumerate(tops):
            ax.add_patch(plt.Rectangle(
                (ry + 0.05, rx + 0.05), 0.9, 0.9,
                fill      = False,
                edgecolor = cfg["rank_color"],
                linewidth = 2.5,
                zorder    = 6,
            ))
            ax.text(
                ry + 0.5, rx + 0.5, f"#{rank + 1}",
                color      = cfg["rank_color"],
                fontsize   = 8,
                fontweight = "bold",
                ha         = "center",
                va         = "center",
                zorder     = 7,
            )

        # ── Label axis & colorbar ─────────────────────────────────────────────
        ax.set_title(cfg["title"], color="white", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Kolom", color="#888899", fontsize=7)
        ax.set_ylabel("Baris",  color="#888899", fontsize=7)

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

    fig.suptitle(
        "Heatmap Exposure Akumulatif — Rekomendasi Penempatan Iklan",
        color="white", fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# HELPER PRIVAT
# =============================================================================

def _get_top_locations(
    heatmap: np.ndarray,
    n: int,
    obstacle_only: bool,
    obstacle_positions: Set[Tuple[int, int]],
) -> List[Tuple[int, int, float]]:
    """
    Kembalikan n lokasi dengan nilai heatmap tertinggi.

    Parameter
    ---------
    heatmap          : np.ndarray
    n                : jumlah top lokasi yang diinginkan
    obstacle_only    : jika True, hanya kembalikan sel yang ada di obstacle_positions
    obstacle_positions : set posisi obstacle

    Returns
    -------
    List of (row, col, value) terurut dari terbesar.
    """
    tops = []
    for idx in np.argsort(heatmap.flatten())[::-1]:
        if len(tops) >= n:
            break
        rx, ry = np.unravel_index(idx, heatmap.shape)
        value  = heatmap[rx, ry]
        if value == 0:
            break
        if obstacle_only and (rx, ry) not in obstacle_positions:
            continue
        tops.append((rx, ry, float(value)))
    return tops