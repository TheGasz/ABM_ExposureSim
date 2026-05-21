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

# ChairDirections adalah mapping (col, row) → arah hadap ("up"/"down"/"left"/"right")
ChairDirections = Dict[Tuple[int, int], str]


# =============================================================================
# TIPE SEL GRID EDITOR
# =============================================================================

CELL_EMPTY    = 0   # Sel kosong / lantai
CELL_CHAIR    = 1   # Kursi
CELL_OBSTACLE = 2   # Tiang / pot / halangan
CELL_DOOR     = 3   # Pintu masuk pelanggan


# =============================================================================
# ARAH HADAP KURSI
# =============================================================================

CHAIR_DIRECTIONS = ["up", "right", "down", "left"]

# Vektor arah untuk raycasting sorot mata
CHAIR_DIR_VECTORS: Dict[str, Tuple[int, int]] = {
    "up"   : ( 0, -1),   # y berkurang (ke atas)
    "down" : ( 0,  1),   # y bertambah (ke bawah)
    "left" : (-1,  0),   # x berkurang (ke kiri)
    "right": ( 1,  0),   # x bertambah (ke kanan)
}


# =============================================================================
# PALET WARNA — EDITOR PLOTLY
# =============================================================================

CELL_COLOR: Dict[int, str] = {
    CELL_EMPTY   : "#16213e",   # Biru gelap (lantai)
    CELL_CHAIR   : "#0f3460",   # Biru tua (kursi)
    CELL_OBSTACLE: "#4a4a6a",   # Abu ungu (halangan)
    CELL_DOOR    : "#2d6a4f",   # Hijau tua (pintu masuk)
}

CELL_BORDER: Dict[int, str] = {
    CELL_EMPTY   : "#252545",
    CELL_CHAIR   : "#1a4a8a",
    CELL_OBSTACLE: "#6a6a9a",
    CELL_DOOR    : "#52b788",
}

CELL_EMOJI: Dict[int, str] = {
    CELL_EMPTY   : "",
    CELL_CHAIR   : "🪑",
    CELL_OBSTACLE: "🧱",
    CELL_DOOR    : "🚪",
}

# Emoji arah panah untuk kursi di editor
CHAIR_DIR_EMOJI: Dict[str, str] = {
    "up"   : "⬆️",
    "down" : "⬇️",
    "left" : "⬅️",
    "right": "➡️",
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
    "door"          : "#52b788",   # Hijau (pintu masuk)
}