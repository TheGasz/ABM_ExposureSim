"""
constants.py
============
Konstanta global dan alias tipe yang dipakai bersama oleh seluruh modul.

Aturan: TIDAK ada import silang dari modul lain di sini.
Setiap modul yang butuh konstanta cukup melakukan:
    from constants import CELL_EMPTY, CELL_CHAIR, ...
"""

from typing import Dict, Tuple

# =============================================================================
# TIPE ALIAS
# =============================================================================

# Layout adalah mapping (col, row) → tipe sel
Layout = Dict[Tuple[int, int], int]


# =============================================================================
# TIPE SEL GRID EDITOR
# =============================================================================

CELL_EMPTY    = 0   # Sel kosong / lantai
CELL_CHAIR    = 1   # Kursi
CELL_OBSTACLE = 2   # Tiang / pot / halangan


# =============================================================================
# PALET WARNA — EDITOR PLOTLY
# =============================================================================

CELL_COLOR: Dict[int, str] = {
    CELL_EMPTY   : "#16213e",   # Biru gelap (lantai)
    CELL_CHAIR   : "#0f3460",   # Biru tua (kursi)
    CELL_OBSTACLE: "#4a4a6a",   # Abu ungu (halangan)
}

CELL_BORDER: Dict[int, str] = {
    CELL_EMPTY   : "#252545",
    CELL_CHAIR   : "#1a4a8a",
    CELL_OBSTACLE: "#6a6a9a",
}

CELL_EMOJI: Dict[int, str] = {
    CELL_EMPTY   : "",
    CELL_CHAIR   : "🪑",
    CELL_OBSTACLE: "🧱",
}


# =============================================================================
# PALET WARNA — PLOT SIMULASI (MATPLOTLIB)
# =============================================================================

SIM_COLOR: Dict[str, str] = {
    "obstacle"      : "#6a6a9a",   # Abu ungu
    "chair_empty"   : "#0f3460",   # Biru tua
    "chair_full"    : "#1a5a9a",   # Biru sedang
    "customer_seek" : "#e94560",   # Merah (mencari)
    "customer_move" : "#7ed6df",   # Biru muda (bergerak)
    "customer_sit"  : "#f5a623",   # Oranye (duduk)
    "gaze_floor"    : "#00ff9f",   # Hijau neon (sorot lantai)
    "gaze_obstacle" : "#ff6b35",   # Oranye terang (sorot tiang)
}