"""
viz/sim_plots.py
================
Fungsi visualisasi matplotlib untuk panel simulasi.

Dua fungsi utama:
    plot_sim_room()      → Figure kondisi ruangan real-time
    plot_dual_heatmap()  → Figure dua heatmap akumulatif (floor + obstacle)
"""

from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import CELL_OBSTACLE, Layout, SIM_COLOR, CHAIR_DIR_VECTORS


def plot_sim_room(snap: dict, width: int, height: int) -> plt.Figure:
    """Render kondisi ruangan simulasi pada satu step tertentu."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#1a1a2e")

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.tick_params(colors="#444466", labelsize=5)
    ax.grid(color="#252545", linewidth=0.4, zorder=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    def scatter(coords, color, marker, size, zorder, alpha=1.0, label=None):
        if not coords:
            return
        xs, ys = zip(*coords)
        ax.scatter(ys, xs, c=color, marker=marker, s=size, zorder=zorder,
                   alpha=alpha, label=label, edgecolors="none")

    # Layer 1: Sorot mata lantai
    for (x, y) in snap["gaze_floor"]:
        ax.add_patch(plt.Rectangle(
            (y - 0.5, x - 0.5), 1, 1,
            color=SIM_COLOR["gaze_floor"], alpha=0.13, zorder=1,
        ))

    # Layer 2: Sorot mata tiang
    for (x, y) in snap["gaze_obstacle"]:
        ax.add_patch(plt.Rectangle(
            (y - 0.5, x - 0.5), 1, 1,
            color=SIM_COLOR["gaze_obstacle"], alpha=0.35, zorder=2,
        ))

    # Layer 3: Agen
    scatter(snap["obstacles"], SIM_COLOR["obstacle"], "s", 200, 3, label="Tiang/Halangan")
    scatter(snap.get("doors", []), SIM_COLOR["door"], "D", 180, 3, label="Pintu Masuk")
    scatter(snap["chairs_empty"], SIM_COLOR["chair_empty"], "p", 160, 3, alpha=0.7, label="Kursi Kosong")
    scatter(snap["chairs_full"], SIM_COLOR["chair_full"], "p", 180, 4, label="Kursi Terisi")
    scatter(snap["customers_seek"], SIM_COLOR["customer_seek"], "o", 140, 5, label="Mencari Kursi")
    scatter(snap["customers_move"], SIM_COLOR["customer_move"], "o", 140, 5, label="Bergerak")
    scatter(snap["customers_sit"], SIM_COLOR["customer_sit"], "o", 170, 5, label="Duduk")

    # Layer 4: Panah arah hadap kursi
    chair_facings = snap.get("chair_facings", {})
    for (cx, cy), facing in chair_facings.items():
        dvec = CHAIR_DIR_VECTORS.get(facing, (1, 0))
        ax.annotate("", xy=(cy + dvec[1] * 0.35, cx + dvec[0] * 0.35),
                     xytext=(cy, cx),
                     arrowprops=dict(arrowstyle="->", color="#f5a623",
                                     lw=1.5, mutation_scale=10),
                     zorder=6)

    # Penanda pintu masuk
    door_positions = snap.get("doors", [])
    if door_positions:
        for (dx, dy) in door_positions:
            ax.add_patch(plt.Rectangle(
                (dy - 0.5, dx - 0.5), 1, 1,
                fill=False, edgecolor=SIM_COLOR["door"],
                linewidth=2, linestyle="--", zorder=2,
            ))
    else:
        # Fallback: garis di kolom 0
        ax.axvline(x=-0.5, color="#f5a623", linewidth=2, linestyle=":", alpha=0.8)
        ax.text(-0.48, height - 0.7, "PINTU",
                color="white", fontsize=6, rotation=90, va="top",
                bbox=dict(facecolor="#f5a623", alpha=0.7, pad=1))

    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.25,
              facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")
    ax.set_title("Kondisi Ruangan (Real-time)",
                 color="white", fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("Kolom", color="#888899", fontsize=7)
    ax.set_ylabel("Baris", color="#888899", fontsize=7)

    plt.tight_layout()
    return fig


def plot_dual_heatmap(
    floor_hm: np.ndarray,
    obstacle_hm: np.ndarray,
    layout: Layout,
    width: int,
    height: int,
) -> plt.Figure:
    """Render dua heatmap akumulatif secara berdampingan."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d1a")

    panel_configs = [
        {
            "title": "🟢 Floor Exposure\n(Banner Lantai / Display Stand)",
            "heatmap": floor_hm.T,
            "source": floor_hm,
            "cmap": "YlOrRd",
            "rank_color": "cyan",
            "obs_only": False,
        },
        {
            "title": "🟠 Obstacle Exposure\n(Stiker / Banner di Tiang)",
            "heatmap": obstacle_hm.T,
            "source": obstacle_hm,
            "cmap": "plasma",
            "rank_color": "lime",
            "obs_only": True,
        },
    ]

    obstacle_positions: Set[Tuple[int, int]] = {
        k for k, v in layout.items() if v == CELL_OBSTACLE
    }

    for cfg, ax in zip(panel_configs, axes):
        ax.set_facecolor("#0d0d1a")
        data = cfg["heatmap"].copy()
        if data.max() == 0:
            data += 1e-9

        sns.heatmap(data, ax=ax, cmap=cfg["cmap"], linewidths=0, square=True,
                    cbar_kws={"label": "Exposure Hits", "shrink": 0.85},
                    xticklabels=False, yticklabels=False)

        for (col, row) in obstacle_positions:
            ax.add_patch(plt.Rectangle(
                (row + 0.05, col + 0.05), 0.9, 0.9,
                fill=False, edgecolor="#6a6a9a", linewidth=1.5, zorder=4,
            ))
            ax.text(row + 0.5, col + 0.5, "🧱",
                    ha="center", va="center", fontsize=7, zorder=5)

        src = cfg["source"]
        tops = _get_top_locations(src, n=3, obstacle_only=cfg["obs_only"],
                                  obstacle_positions=obstacle_positions)
        for rank, (rx, ry, _) in enumerate(tops):
            ax.add_patch(plt.Rectangle(
                (ry + 0.05, rx + 0.05), 0.9, 0.9,
                fill=False, edgecolor=cfg["rank_color"], linewidth=2.5, zorder=6,
            ))
            ax.text(ry + 0.5, rx + 0.5, f"#{rank + 1}",
                    color=cfg["rank_color"], fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=7)

        ax.set_title(cfg["title"], color="white", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Kolom", color="#888899", fontsize=7)
        ax.set_ylabel("Baris", color="#888899", fontsize=7)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

    fig.suptitle("Heatmap Exposure Akumulatif — Rekomendasi Penempatan Iklan",
                 color="white", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def _get_top_locations(heatmap, n, obstacle_only, obstacle_positions):
    """Kembalikan n lokasi dengan nilai heatmap tertinggi."""
    tops = []
    for idx in np.argsort(heatmap.flatten())[::-1]:
        if len(tops) >= n:
            break
        rx, ry = np.unravel_index(idx, heatmap.shape)
        value = heatmap[rx, ry]
        if value == 0:
            break
        if obstacle_only and (rx, ry) not in obstacle_positions:
            continue
        tops.append((rx, ry, float(value)))
    return tops