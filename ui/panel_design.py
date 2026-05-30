"""
ui/panel_design.py
==================
Design panel for the room layout editor.
"""

import json
import random

import numpy as np
import streamlit as st

from constants import (
    CELL_CHAIR,
    CELL_EMPTY,
    CELL_OBSTACLE,
    CELL_DOOR,
    DEFAULT_DT_S,
)
from ui.state import (
    grid_state_to_layout,
    make_empty_grid,
    layout_to_grid_state,
    get_chair_directions,
    set_chair_direction,
    rotate_chair_direction,
    remove_chair_direction,
)
from viz.editor_plot import build_editor_figure


def build_sidebar_design() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:8px 0'>"
            "<h2 style='color:#f5a623;margin:0;font-family:monospace'>Design</h2>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("Room Size")
        st.caption("1 cell = 1 meter (10 px)")

        internal_w = max(2, st.session_state.grid_w - 2)
        internal_h = max(2, st.session_state.grid_h - 2)

        new_w_m = st.number_input("Inner width (meter)", 2, 100, internal_w)
        new_h_m = st.number_input("Inner height (meter)", 2, 100, internal_h)

        new_w = new_w_m + 2
        new_h = new_h_m + 2

        size_changed = (
            new_w != st.session_state.grid_w or
            new_h != st.session_state.grid_h
        )

        if size_changed:
            if st.button("Apply Size", use_container_width=True, type="primary"):
                _resize_grid(new_w, new_h)
                st.rerun()
        else:
            st.info(
                f"Inner: {new_w_m}m x {new_h_m}m | Total grid: {new_w} x {new_h} cells"
            )

        st.divider()
        st.subheader("Door Settings")
        gs = st.session_state.grid_state
        door_cells = []
        if gs is not None:
            for row in range(gs.shape[0]):
                for col in range(gs.shape[1]):
                    if gs[row, col] == CELL_DOOR:
                        door_cells.append((col, row))

        if not door_cells:
            st.info("No doors placed yet.")
        else:
            if "door_probs" not in st.session_state:
                st.session_state.door_probs = {}
            for dc in door_cells:
                if dc not in st.session_state.door_probs:
                    st.session_state.door_probs[dc] = 100.0 / len(door_cells)

            st.caption("Spawn share (total 100%)")
            total_prob = 0.0
            for dc in door_cells:
                p = st.number_input(
                    f"Door {dc}",
                    0.0,
                    100.0,
                    float(st.session_state.door_probs.get(dc, 100.0 / len(door_cells))),
                    step=1.0,
                )
                st.session_state.door_probs[dc] = p
                total_prob += p
            if abs(total_prob - 100.0) > 0.001:
                st.error(f"Total: {total_prob}%. Please fix to 100%.")

        st.divider()
        st.markdown(
            """
            **Design Tips:**
            - Place doors on the edge of the grid.
            - Click a chair with the Chair brush to rotate its facing.
            - Obstacles block movement.
            """
        )


def _resize_grid(new_w: int, new_h: int) -> None:
    old_gs = st.session_state.grid_state
    new_gs = make_empty_grid(new_w, new_h)
    old_h, old_w = old_gs.shape
    copy_h = min(old_h, new_h)
    copy_w = min(old_w, new_w)

    for r in range(copy_h):
        for c in range(copy_w):
            val = old_gs[r, c]
            is_old_edge = (r == 0 or r == old_h - 1 or c == 0 or c == old_w - 1)
            is_new_edge = (r == 0 or r == new_h - 1 or c == 0 or c == new_w - 1)

            if is_old_edge and not is_new_edge and val in (CELL_OBSTACLE, CELL_DOOR):
                continue
            if is_new_edge and val == CELL_EMPTY:
                continue

            new_gs[r, c] = val

    st.session_state.grid_w = new_w
    st.session_state.grid_h = new_h
    st.session_state.grid_state = new_gs
    st.session_state.model = None


def panel_design() -> None:
    width = st.session_state.grid_w
    height = st.session_state.grid_h
    gs = st.session_state.grid_state

    if "pending_edits" not in st.session_state:
        st.session_state.pending_edits = {}
    if "last_click_hash" not in st.session_state:
        st.session_state.last_click_hash = None

    _render_info_bar(gs, width, height)
    st.divider()
    _render_brush_selector()

    chair_dirs = get_chair_directions()

    display_gs = gs.copy()
    for (col, row), val in st.session_state.pending_edits.items():
        is_edge = _is_edge_cell(col, row, width, height)
        if is_edge and val not in (CELL_DOOR, CELL_OBSTACLE):
            continue
        display_gs[row, col] = val

    fig = build_editor_figure(display_gs, width, height, st.session_state.brush, chair_dirs)

    st.markdown(
        "Use click or box/lasso select to paint. "
        "Click Save Draft to apply edits to the room."
    )

    click_data = st.plotly_chart(
        fig,
        use_container_width=True,
        key="grid_editor",
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
    )

    if click_data and hasattr(click_data, "selection"):
        pts = click_data.selection.get("points", [])
        import hashlib
        import json as json_lib

        pts_str = json_lib.dumps([{"x": p.get("x"), "y": p.get("y")} for p in pts], sort_keys=True)
        current_hash = hashlib.md5(pts_str.encode()).hexdigest()

        if current_hash != st.session_state.last_click_hash and pts:
            st.session_state.last_click_hash = current_hash
            brush = st.session_state.brush
            for pt in pts:
                col = int(round(pt.get("x", -1)))
                row = int(round(pt.get("y", -1)))
                if 0 <= col < width and 0 <= row < height:
                    current = int(gs[row, col])
                    if current == CELL_CHAIR and brush == CELL_CHAIR:
                        rotate_chair_direction(col, row)
                    else:
                        st.session_state.pending_edits[(col, row)] = brush
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("Save Draft", type="primary", use_container_width=True):
        changes = len(st.session_state.pending_edits)
        for (col, row), val in st.session_state.pending_edits.items():
            is_edge = _is_edge_cell(col, row, width, height)
            if is_edge and val not in (CELL_DOOR, CELL_OBSTACLE):
                continue

            current = int(gs[row, col])
            new_val = CELL_EMPTY if current == val else val
            if is_edge and new_val == CELL_EMPTY:
                new_val = CELL_OBSTACLE

            if current == CELL_CHAIR and new_val != CELL_CHAIR:
                remove_chair_direction(col, row)
            if new_val == CELL_CHAIR and current != CELL_CHAIR:
                set_chair_direction(col, row, "right")

            st.session_state.grid_state[row, col] = new_val

        st.session_state.pending_edits.clear()
        if changes > 0:
            st.success(f"Saved {changes} cells.")
        st.rerun()

    if c2.button("Clear Draft", use_container_width=True):
        st.session_state.pending_edits.clear()
        st.rerun()

    st.divider()
    _render_quick_tools(gs, width, height)


def _render_info_bar(gs, width, height):
    n_chairs = int(np.sum(gs == CELL_CHAIR))
    n_obstacles = int(np.sum(gs == CELL_OBSTACLE))
    n_doors = int(np.sum(gs == CELL_DOOR))
    n_empty = width * height - n_chairs - n_obstacles - n_doors

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Chairs", n_chairs)
    c2.metric("Obstacles", n_obstacles)
    c3.metric("Doors", n_doors)
    c4.metric("Empty", n_empty)
    c5.metric("Grid", f"{width} x {height}")

    if n_doors == 0:
        st.warning("No doors yet. Spawns will use the left edge fallback.")

    used_ratio = (n_chairs + n_obstacles + n_doors) / (width * height)
    if used_ratio >= 1.0:
        st.error("Grid is full. Remove some cells.")
    elif used_ratio > 0.7:
        st.warning("Grid is almost full (>70%).")


def _render_brush_selector():
    st.markdown("**Brush** - Select a type then click the grid")
    cb1, cb2, cb3, cb4, _ = st.columns([1, 1, 1, 1, 3])

    if cb1.button(
        "Chair",
        use_container_width=True,
        type="primary" if st.session_state.brush == CELL_CHAIR else "secondary",
    ):
        st.session_state.brush = CELL_CHAIR
        st.rerun()
    if cb2.button(
        "Obstacle",
        use_container_width=True,
        type="primary" if st.session_state.brush == CELL_OBSTACLE else "secondary",
    ):
        st.session_state.brush = CELL_OBSTACLE
        st.rerun()
    if cb3.button(
        "Door",
        use_container_width=True,
        type="primary" if st.session_state.brush == CELL_DOOR else "secondary",
    ):
        st.session_state.brush = CELL_DOOR
        st.rerun()
    if cb4.button(
        "Erase",
        use_container_width=True,
        type="primary" if st.session_state.brush == CELL_EMPTY else "secondary",
    ):
        st.session_state.brush = CELL_EMPTY
        st.rerun()

    st.caption("Click a chair with Chair brush to rotate its facing.")


def _is_edge_cell(col, row, width, height):
    return col == 0 or col == width - 1 or row == 0 or row == height - 1


def _render_quick_tools(gs, width, height):
    st.markdown("**Quick Tools**")
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)

    if qc1.button("Clear All", use_container_width=True):
        st.session_state.grid_state = make_empty_grid(width, height)
        st.session_state.chair_directions = {}
        st.rerun()

    if qc2.button("Random Chairs (6)", use_container_width=True):
        _add_random_elements(CELL_CHAIR, 6)
        st.rerun()

    if qc3.button("Random Obstacles (4)", use_container_width=True):
        _add_random_elements(CELL_OBSTACLE, 4)
        st.rerun()

    if qc4.button("Demo Layout", use_container_width=True):
        _load_demo_layout()
        st.rerun()

    layout = grid_state_to_layout(gs)
    dirs = get_chair_directions()
    layout_export = {
        "layout": {f"{c},{r}": int(v) for (c, r), v in layout.items()},
        "chair_directions": {f"{c},{r}": d for (c, r), d in dirs.items()},
    }
    full_export = {
        "version": 1,
        "grid": {"width": width, "height": height},
        "layout": layout_export["layout"],
        "chair_directions": layout_export["chair_directions"],
        "door_probs": {
            f"{c},{r}": float(p)
            for (c, r), p in st.session_state.get("door_probs", {}).items()
        },
        "sim_config": st.session_state.get("sim_config", {}),
    }
    layout_json = json.dumps(layout_export, indent=2)
    full_json = json.dumps(full_export, indent=2)
    qc5.download_button(
        "Export Room + Config",
        data=full_json,
        file_name="room_config.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "Export Layout Only",
        data=layout_json,
        file_name="room_layout.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    # Pakai key counter supaya widget reset (file hilang) setelah import berhasil.
    # Tanpa ini, Streamlit menyimpan file di session dan memanggil _import_layout_json
    # lagi di setiap rerun berikutnya (termasuk saat Stop ditekan), yang me-reset model.
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    uploaded = st.file_uploader(
        "Import Room JSON",
        type="json",
        label_visibility="collapsed",
        key=f"room_json_uploader_{st.session_state.file_uploader_key}",
    )
    if uploaded is not None:
        ok = _import_layout_json(uploaded, width, height)
        if ok:
            # Increment key → widget dapat key baru → file otomatis hilang
            st.session_state.file_uploader_key += 1
            st.rerun()


def _add_random_elements(cell_type, n):
    width = st.session_state.grid_w
    height = st.session_state.grid_h
    gs = st.session_state.grid_state
    empty = [(c, r) for c in range(1, width) for r in range(height) if gs[r, c] == CELL_EMPTY]
    random.shuffle(empty)
    for col, row in empty[:n]:
        gs[row, col] = cell_type
        if cell_type == CELL_CHAIR:
            d = random.choice(["up", "down", "left", "right"])
            set_chair_direction(col, row, d)


def _load_demo_layout():
    width = st.session_state.grid_w
    height = st.session_state.grid_h
    gs = make_empty_grid(width, height)
    st.session_state.chair_directions = {}

    mid_row = height // 2
    for dr in range(-1, 2):
        r = mid_row + dr
        if 0 <= r < height:
            gs[r, 0] = CELL_DOOR

    for col in range(2, width - 1, 3):
        if col < width:
            gs[1, col] = CELL_CHAIR
            set_chair_direction(col, 1, "down")
            gs[height - 2, col] = CELL_CHAIR
            set_chair_direction(col, height - 2, "up")

    mid_col = width // 2
    obstacles = [
        (mid_row - 1, mid_col),
        (mid_row + 1, mid_col),
        (mid_row, mid_col - 2),
        (mid_row, mid_col + 2),
    ]
    for row, col in obstacles:
        if 0 < row < height and 0 < col < width:
            gs[row, col] = CELL_OBSTACLE

    for col in [mid_col - 1, mid_col, mid_col + 1]:
        if 0 < col < width:
            gs[mid_row, col] = CELL_CHAIR
            set_chair_direction(col, mid_row, "right")

    st.session_state.grid_state = gs


def _import_layout_json(uploaded_file, width, height) -> bool:
    """Import layout dari JSON. Return True jika berhasil, False jika gagal."""
    try:
        raw = json.load(uploaded_file)
        grid_info = raw.get("grid", {}) if isinstance(raw, dict) else {}
        new_w = int(grid_info.get("width", width))
        new_h = int(grid_info.get("height", height))

        if "layout" in raw:
            layout_data = raw["layout"]
            dir_data = raw.get("chair_directions", {})
            door_data = raw.get("door_probs", {})
            sim_config = raw.get("sim_config", {})
        else:
            layout_data = raw
            dir_data = {}
            door_data = {}
            sim_config = {}

        new_layout = {}
        for key, value in layout_data.items():
            col, row = map(int, key.split(","))
            new_layout[(col, row)] = int(value)

        new_gs = layout_to_grid_state(new_layout, new_w, new_h)
        new_dirs = {}
        for key, d in dir_data.items():
            col, row = map(int, key.split(","))
            if 0 <= row < new_h and 0 <= col < new_w:
                new_dirs[(col, row)] = d

        new_doors = {}
        for key, p in door_data.items():
            col, row = map(int, key.split(","))
            if 0 <= row < new_h and 0 <= col < new_w:
                new_doors[(col, row)] = float(p)

        st.session_state.grid_w = new_w
        st.session_state.grid_h = new_h
        st.session_state.grid_state = new_gs
        st.session_state.chair_directions = new_dirs
        st.session_state.door_probs = new_doors
        if sim_config:
            if "fps" not in sim_config:
                fps_value = None
                if "sim_dt" in sim_config:
                    legacy_dt = float(sim_config.get("sim_dt", DEFAULT_DT_S))
                    if legacy_dt > 1e-9:
                        fps_value = round(1.0 / legacy_dt)
                if fps_value is None and "step_delay" in sim_config:
                    legacy_delay = float(sim_config.get("step_delay", DEFAULT_DT_S))
                    if legacy_delay > 1e-9:
                        fps_value = round(1.0 / legacy_delay)
                if fps_value is None and "motion_level" in sim_config:
                    level = int(sim_config.get("motion_level", 3))
                    if level >= 5:
                        fps_value = 20
                    elif level == 4:
                        fps_value = 12
                    elif level == 3:
                        fps_value = 10
                    elif level == 2:
                        fps_value = 7
                    else:
                        fps_value = 5

                if fps_value is None:
                    fps_value = 10

                fps_value = max(4, min(24, int(fps_value)))
                sim_config = {
                    **sim_config,
                    "fps": fps_value,
                }
                sim_config.pop("motion_level", None)
                sim_config.pop("sim_dt", None)
                sim_config.pop("render_substeps", None)
                sim_config.pop("delay_scale", None)
                sim_config.pop("step_delay", None)

            st.session_state.sim_config = sim_config
        st.session_state.model = None
        # Reset running/paused agar simulasi lama tidak carry-over ke layout baru
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.show_final_heatmap = False
        st.success(f"✅ Layout berhasil diimport ({new_w}x{new_h} cells).")
        return True
    except json.JSONDecodeError:
        st.error("Invalid JSON file.")
        return False
    except Exception as e:
        st.error(f"Gagal import: {e}")
        return False

