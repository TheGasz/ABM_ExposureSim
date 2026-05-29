"""
ui/state.py
===========
Session state and grid conversion helpers.

Grid conventions:
- session_state grid_state: numpy array shape (height, width) with [row][col]
- layout dict: {(col, row): cell_type}
"""

import numpy as np
import streamlit as st

from constants import (
    CELL_CHAIR,
    CELL_EMPTY,
    CELL_OBSTACLE,
    CELL_DOOR,
    CHAIR_DIRECTIONS,
)


def init_session() -> None:
    defaults = {
        "grid_w": 13,
        "grid_h": 13,
        "grid_state": None,
        "chair_directions": {},
        "door_probs": {},
        "sim_config": {
            "arrival_rate": 0.6,
            "mean_sitting": 20.0,
            "pass_through_prob": 0.5,
            "fps": 10,
            "max_steps": 200,
            "seed": 42,
        },
        "brush": CELL_CHAIR,
        "model": None,
        "running": False,
        "step_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    gs = st.session_state.grid_state
    target_shape = (st.session_state.grid_h, st.session_state.grid_w)
    if gs is None or gs.shape != target_shape:
        st.session_state.grid_state = make_empty_grid(
            st.session_state.grid_w,
            st.session_state.grid_h,
        )
        _clean_chair_directions()


def make_empty_grid(width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = CELL_OBSTACLE
    grid[height - 1, :] = CELL_OBSTACLE
    grid[:, 0] = CELL_OBSTACLE
    grid[:, width - 1] = CELL_OBSTACLE
    return grid


def grid_state_to_layout(grid_state: np.ndarray) -> dict:
    layout = {}
    height, width = grid_state.shape
    for row in range(height):
        for col in range(width):
            value = int(grid_state[row, col])
            if value != CELL_EMPTY:
                layout[(col, row)] = value
    return layout


def layout_to_grid_state(layout: dict, width: int, height: int) -> np.ndarray:
    grid = make_empty_grid(width, height)
    for (col, row), value in layout.items():
        if 0 <= row < height and 0 <= col < width:
            grid[row, col] = value
    return grid


def get_chair_directions() -> dict:
    return st.session_state.get("chair_directions", {})


def set_chair_direction(col: int, row: int, direction: str) -> None:
    if direction not in CHAIR_DIRECTIONS:
        raise ValueError(f"Invalid chair direction: {direction}")
    st.session_state.chair_directions[(col, row)] = direction


def rotate_chair_direction(col: int, row: int) -> str:
    current = st.session_state.chair_directions.get((col, row), "right")
    idx = CHAIR_DIRECTIONS.index(current)
    new_dir = CHAIR_DIRECTIONS[(idx + 1) % len(CHAIR_DIRECTIONS)]
    st.session_state.chair_directions[(col, row)] = new_dir
    return new_dir


def remove_chair_direction(col: int, row: int) -> None:
    st.session_state.chair_directions.pop((col, row), None)


def _clean_chair_directions() -> None:
    width = st.session_state.grid_w
    height = st.session_state.grid_h
    gs = st.session_state.grid_state
    dirs = st.session_state.chair_directions

    to_remove = []
    for (col, row) in dirs:
        if not (0 <= col < width and 0 <= row < height):
            to_remove.append((col, row))
        elif gs is not None and int(gs[row, col]) != CELL_CHAIR:
            to_remove.append((col, row))

    for key in to_remove:
        dirs.pop(key, None)
