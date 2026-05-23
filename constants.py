"""
constants.py
============
Global constants and shared type aliases.

Rule: do not import other project modules here.
"""

from typing import Dict, Tuple

# =============================================================================
# TYPE ALIASES
# =============================================================================

Layout = Dict[Tuple[int, int], int]
ChairDirections = Dict[Tuple[int, int], str]


# =============================================================================
# GRID CELL TYPES
# =============================================================================

CELL_EMPTY = 0
CELL_CHAIR = 1
CELL_OBSTACLE = 2
CELL_DOOR = 3


# =============================================================================
# CHAIR FACING
# =============================================================================

CHAIR_DIRECTIONS = ["up", "right", "down", "left"]

CHAIR_DIR_VECTORS: Dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


# =============================================================================
# SPATIAL UNITS
# =============================================================================

# 1 meter = 10 pixels, and 1 cell = 1 meter.
CELL_SIZE_PX = 10
PX_PER_METER = 10

SPEED_M_PER_S = 1.0
SPEED_PX_PER_S = SPEED_M_PER_S * PX_PER_METER

DEFAULT_DT_S = 0.1
DEFAULT_FPSTEP = 10  # Default frames per step
# Speed variation: berapa persen deviasi dari base speed (normal distribution)
# Contoh: 0.2 = 20% deviasi, jadi agent speed berkisar 0.8x - 1.2x dari base
SPEED_VARIATION = 0.2  # 20% variation
ARRIVE_THRESHOLD_PX = CELL_SIZE_PX * 0.35


# =============================================================================
# COLOR PALETTE - EDITOR (PLOTLY)
# =============================================================================

CELL_COLOR: Dict[int, str] = {
    CELL_EMPTY: "#16213e",
    CELL_CHAIR: "#0f3460",
    CELL_OBSTACLE: "#4a4a6a",
    CELL_DOOR: "#2d6a4f",
}

CELL_BORDER: Dict[int, str] = {
    CELL_EMPTY: "#252545",
    CELL_CHAIR: "#1a4a8a",
    CELL_OBSTACLE: "#6a6a9a",
    CELL_DOOR: "#52b788",
}

CELL_EMOJI: Dict[int, str] = {
    CELL_EMPTY: "",
    CELL_CHAIR: "C",
    CELL_OBSTACLE: "O",
    CELL_DOOR: "D",
}

CHAIR_DIR_EMOJI: Dict[str, str] = {
    "up": "^",
    "down": "v",
    "left": "<",
    "right": ">",
}


# =============================================================================
# COLOR PALETTE - SIMULATION (MATPLOTLIB)
# =============================================================================

SIM_COLOR: Dict[str, str] = {
    "obstacle": "#6a6a9a",
    "chair_empty": "#0f3460",
    "chair_full": "#1a5a9a",
    "door": "#52b788",
    "human_seek": "#e94560",
    "human_move": "#7ed6df",
    "human_sit": "#f5a623",
    "human_pass": "#9bdeac",
}
