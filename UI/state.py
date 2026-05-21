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
  • Di session_state: np.ndarray shape (height, width), nilai 0/1/2.
    → Indeks: grid_state[row][col]  (baris dulu, kolom belakang)
  • Di layout dict  : {(col, row): int}
    → Key adalah tuple (col, row) — kolom dulu, baris belakang
  Ini konsisten dengan konvensi (x=kolom, y=baris) pada plot Plotly/Matplotlib.
"""

import numpy as np
import streamlit as st

from constants import CELL_CHAIR, CELL_EMPTY, CELL_OBSTACLE


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
        grid_w      : int  — lebar grid (kolom)
        grid_h      : int  — tinggi grid (baris)
        grid_state  : ndarray | None — nilai sel grid [row][col]
        brush       : int  — tipe brush aktif (0/1/2)
        model       : WaitingRoomModel | None
        running     : bool — apakah simulasi sedang berjalan
        step_count  : int  — step simulasi yang sudah dijalankan
    """
    defaults = {
        "grid_w"    : 15,
        "grid_h"    : 10,
        "grid_state": None,
        "brush"     : CELL_CHAIR,
        "model"     : None,
        "running"   : False,
        "step_count": 0,
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
    karena model hanya perlu tahu di mana kursi dan obstacle berada.
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