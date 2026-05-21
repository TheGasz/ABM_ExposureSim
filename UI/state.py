"""
ui/state.py
===========
Manajemen Streamlit session_state dan fungsi konversi grid.

Tanggung jawab modul ini:
  • init_session()           — Inisialisasi semua key session_state dengan default.
  • make_empty_grid()        — Buat numpy array kosong untuk grid baru.
  • grid_state_to_layout()   — Konversi numpy grid → layout dict (untuk model).
  • layout_to_grid_state()   — Konversi layout dict → numpy grid (untuk import).

Konvensi penyimpanan grid:
  • Di session_state: np.ndarray shape (height, width), nilai 0/1/2/3.
    → Indeks: grid_state[row][col]  (baris dulu, kolom belakang)
  • Di layout dict  : {(col, row): int}
    → Key adalah tuple (col, row) — kolom dulu, baris belakang
  Ini konsisten dengan konvensi (x=kolom, y=baris) pada plot Plotly/Matplotlib.

  • chair_directions: dict {(col, row): "up"|"down"|"left"|"right"}
    → Menyimpan arah hadap setiap kursi di grid.
"""

import numpy as np
import streamlit as st

from constants import (
    CELL_CHAIR, CELL_EMPTY, CELL_OBSTACLE, CELL_DOOR,
    CHAIR_DIRECTIONS,
)


# =============================================================================
# INISIALISASI SESSION STATE
# =============================================================================

def init_session() -> None:
    """
    Inisialisasi semua key session_state dengan nilai default.

    Dipanggil sekali di awal main() sebelum rendering apapun.
    Menggunakan pola "set only if not exists" agar tidak menimpa
    state yang sudah ada saat reruns.

    Keys yang dikelola:
        grid_w           : int  — lebar grid (kolom)
        grid_h           : int  — tinggi grid (baris)
        grid_state       : ndarray | None — nilai sel grid [row][col]
        chair_directions : dict — arah hadap kursi {(col,row): str}
        brush            : int  — tipe brush aktif (0/1/2/3)
        model            : WaitingRoomModel | None
        running          : bool — apakah simulasi sedang berjalan
        step_count       : int  — step simulasi yang sudah dijalankan
    """
    defaults = {
        "grid_w"           : 15,
        "grid_h"           : 10,
        "grid_state"       : None,
        "chair_directions" : {},     # {(col, row): "up"|"down"|"left"|"right"}
        "brush"            : CELL_CHAIR,
        "model"            : None,
        "running"          : False,
        "step_count"       : 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Inisialisasi atau re-inisialisasi grid jika ukuran belum sesuai
    gs = st.session_state.grid_state
    target_shape = (st.session_state.grid_h, st.session_state.grid_w)
    if gs is None or gs.shape != target_shape:
        st.session_state.grid_state = make_empty_grid(
            st.session_state.grid_w,
            st.session_state.grid_h,
        )
        # Bersihkan arah kursi yang di luar batas baru
        _clean_chair_directions()


# =============================================================================
# GRID HELPER FUNCTIONS
# =============================================================================

def make_empty_grid(width: int, height: int) -> np.ndarray:
    """
    Buat grid kosong sebagai numpy array.

    Returns
    -------
    np.ndarray, shape (height, width), dtype=int, semua nilai 0 (CELL_EMPTY).
    """
    return np.zeros((height, width), dtype=int)


def grid_state_to_layout(grid_state: np.ndarray) -> dict:
    """
    Konversi numpy grid → layout dict untuk WaitingRoomModel.

    Input:  grid_state[row][col] = tipe sel
    Output: {(col, row): tipe_sel} hanya untuk sel NON-EMPTY

    Sel kosong (CELL_EMPTY=0) tidak dimasukkan ke layout dict
    karena model hanya perlu tahu di mana kursi, obstacle, dan pintu berada.
    """
    layout = {}
    H, W   = grid_state.shape
    for row in range(H):
        for col in range(W):
            value = int(grid_state[row, col])
            if value != CELL_EMPTY:
                layout[(col, row)] = value
    return layout


def layout_to_grid_state(layout: dict, width: int, height: int) -> np.ndarray:
    """
    Konversi layout dict → numpy grid state.

    Digunakan saat import layout dari file JSON.
    Sel yang berada di luar batas grid saat ini diabaikan.

    Input:  {(col, row): tipe_sel}
    Output: np.ndarray, shape (height, width)
    """
    gs = make_empty_grid(width, height)
    for (col, row), value in layout.items():
        if 0 <= row < height and 0 <= col < width:
            gs[row, col] = value
    return gs


def get_chair_directions() -> dict:
    """
    Kembalikan dict arah hadap kursi dari session_state.

    Returns
    -------
    dict: {(col, row): "up"|"down"|"left"|"right"}
    """
    return st.session_state.get("chair_directions", {})


def set_chair_direction(col: int, row: int, direction: str) -> None:
    """
    Set arah hadap kursi di posisi tertentu.

    Parameters
    ----------
    col, row : int — posisi kursi
    direction : str — "up", "down", "left", "right"
    """
    if direction not in CHAIR_DIRECTIONS:
        raise ValueError(f"Arah tidak valid: {direction}")
    st.session_state.chair_directions[(col, row)] = direction


def rotate_chair_direction(col: int, row: int) -> str:
    """
    Putar arah hadap kursi ke arah berikutnya (cycle).

    Urutan: up → right → down → left → up → ...

    Returns
    -------
    str — arah baru setelah diputar.
    """
    current = st.session_state.chair_directions.get((col, row), "right")
    idx     = CHAIR_DIRECTIONS.index(current)
    new_dir = CHAIR_DIRECTIONS[(idx + 1) % len(CHAIR_DIRECTIONS)]
    st.session_state.chair_directions[(col, row)] = new_dir
    return new_dir


def remove_chair_direction(col: int, row: int) -> None:
    """Hapus data arah kursi di posisi tertentu (saat kursi dihapus)."""
    st.session_state.chair_directions.pop((col, row), None)


def _clean_chair_directions() -> None:
    """
    Bersihkan entri chair_directions yang di luar batas grid saat ini,
    atau yang selnya bukan CELL_CHAIR lagi.
    """
    W  = st.session_state.grid_w
    H  = st.session_state.grid_h
    gs = st.session_state.grid_state
    dirs = st.session_state.chair_directions

    to_remove = []
    for (col, row) in dirs:
        if not (0 <= col < W and 0 <= row < H):
            to_remove.append((col, row))
        elif gs is not None and int(gs[row, col]) != CELL_CHAIR:
            to_remove.append((col, row))

    for key in to_remove:
        dirs.pop(key, None)