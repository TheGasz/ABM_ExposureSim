"""
ui/panel_design.py
==================
Panel mode DESIGN — editor denah ruangan interaktif.

Fitur:
  • Brush: Kursi / Halangan / Pintu Masuk / Hapus
  • Klik kursi yang sudah ada → putar arah hadap (cycle)
  • Grid max 100x100 (1m ≈ 10 kotak)
  • Pintu masuk bisa diletakkan di tepi grid manapun
"""

import json
import random

import numpy as np
import streamlit as st

from constants import (
    CELL_CHAIR, CELL_EMPTY, CELL_OBSTACLE, CELL_DOOR,
    CHAIR_DIR_EMOJI,
)
from ui.state import (
    grid_state_to_layout, make_empty_grid, layout_to_grid_state,
    get_chair_directions, set_chair_direction, rotate_chair_direction,
    remove_chair_direction,
)
from viz.editor_plot import build_editor_figure


def build_sidebar_design() -> None:
    """Render sidebar untuk konfigurasi ukuran grid."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:8px 0'>"
            "<h2 style='color:#f5a623;margin:0;font-family:monospace'>📐 Desain</h2>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("🏗️ Ukuran Ruangan")
        st.caption("1 meter ≈ 10 kotak grid")
        new_w = st.slider("Lebar (kolom)",  5, 100, st.session_state.grid_w)
        new_h = st.slider("Tinggi (baris)", 5, 100, st.session_state.grid_h)

        size_changed = (
            new_w != st.session_state.grid_w or
            new_h != st.session_state.grid_h
        )

        if size_changed:
            if st.button("✅ Terapkan Ukuran", use_container_width=True, type="primary"):
                _resize_grid(new_w, new_h)
                st.rerun()
        else:
            st.info(f"Ukuran aktif: **{new_w}×{new_h}** ({new_w/10:.1f}m × {new_h/10:.1f}m)")

        st.divider()
        st.markdown("""
        **Panduan Desain:**
        - 🚪 Letakkan **Pintu Masuk** di tepi grid
        - 🪑 Klik kursi yang sudah ada → **putar arah**
        - 🧱 Halangan memblokir jalur pelanggan
        - Pelanggan masuk lewat pintu, bukan celah
        """)


def _resize_grid(new_w: int, new_h: int) -> None:
    """Ubah ukuran grid sambil mempertahankan elemen yang masih muat."""
    old_gs = st.session_state.grid_state
    new_gs = make_empty_grid(new_w, new_h)
    old_h, old_w = old_gs.shape
    copy_h = min(old_h, new_h)
    copy_w = min(old_w, new_w)
    new_gs[:copy_h, :copy_w] = old_gs[:copy_h, :copy_w]
    st.session_state.grid_w = new_w
    st.session_state.grid_h = new_h
    st.session_state.grid_state = new_gs
    st.session_state.model = None


def panel_design() -> None:
    """Render panel editor denah ruangan."""
    W = st.session_state.grid_w
    H = st.session_state.grid_h
    gs = st.session_state.grid_state

    _render_info_bar(gs, W, H)
    st.divider()
    _render_brush_selector()

    chair_dirs = get_chair_directions()
    fig = build_editor_figure(gs, W, H, st.session_state.brush, chair_dirs)
    click_data = st.plotly_chart(
        fig, use_container_width=True, key="grid_editor", on_select="rerun",
    )
    _handle_cell_click(click_data, gs, W, H)

    st.divider()
    _render_quick_tools(gs, W, H)


def _render_info_bar(gs, W, H):
    """Tampilkan metrik jumlah elemen dan validasi."""
    n_chairs = int(np.sum(gs == CELL_CHAIR))
    n_obstacles = int(np.sum(gs == CELL_OBSTACLE))
    n_doors = int(np.sum(gs == CELL_DOOR))
    n_empty = W * H - n_chairs - n_obstacles - n_doors

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🪑 Kursi", n_chairs)
    c2.metric("🧱 Halangan", n_obstacles)
    c3.metric("🚪 Pintu", n_doors)
    c4.metric("⬜ Kosong", n_empty)
    c5.metric("📐 Grid", f"{W}×{H}")

    if n_doors == 0:
        st.warning("⚠️ Belum ada pintu masuk! Pelanggan akan spawn di kolom 0 (fallback).")

    used_ratio = (n_chairs + n_obstacles + n_doors) / (W * H)
    if used_ratio >= 1.0:
        st.error("⚠️ Grid penuh! Hapus beberapa elemen.")
    elif used_ratio > 0.7:
        st.warning("⚠️ Grid hampir penuh (>70%).")


def _render_brush_selector():
    """Render tombol pemilih brush."""
    st.markdown("**🖌️ Brush Aktif** — Pilih tipe lalu klik sel pada grid")
    cb1, cb2, cb3, cb4, _ = st.columns([1, 1, 1, 1, 3])

    if cb1.button("🪑 Kursi", use_container_width=True,
                  type="primary" if st.session_state.brush == CELL_CHAIR else "secondary"):
        st.session_state.brush = CELL_CHAIR
        st.rerun()
    if cb2.button("🧱 Halangan", use_container_width=True,
                  type="primary" if st.session_state.brush == CELL_OBSTACLE else "secondary"):
        st.session_state.brush = CELL_OBSTACLE
        st.rerun()
    if cb3.button("🚪 Pintu", use_container_width=True,
                  type="primary" if st.session_state.brush == CELL_DOOR else "secondary"):
        st.session_state.brush = CELL_DOOR
        st.rerun()
    if cb4.button("🗑️ Hapus", use_container_width=True,
                  type="primary" if st.session_state.brush == CELL_EMPTY else "secondary"):
        st.session_state.brush = CELL_EMPTY
        st.rerun()

    st.caption("💡 Klik kursi yang sudah ada (dengan brush Kursi) → putar arah hadap")


def _is_edge_cell(col, row, W, H):
    """Cek apakah sel berada di tepi grid."""
    return col == 0 or col == W - 1 or row == 0 or row == H - 1


def _handle_cell_click(click_data, gs, W, H):
    """Proses event klik dari figure Plotly."""
    if not (click_data and hasattr(click_data, "selection")):
        return
    sel = click_data.selection
    pts = sel.get("points", []) if isinstance(sel, dict) else []
    if not pts:
        return

    pt = pts[0]
    col = int(round(pt.get("x", -1)))
    row = int(round(pt.get("y", -1)))
    if not (0 <= col < W and 0 <= row < H):
        return

    current = int(gs[row, col])
    brush = st.session_state.brush

    # Klik kursi yang sudah ada dengan brush Kursi → putar arah
    if current == CELL_CHAIR and brush == CELL_CHAIR:
        new_dir = rotate_chair_direction(col, row)
        st.toast(f"🔄 Kursi ({col},{row}) diputar ke: {CHAIR_DIR_EMOJI.get(new_dir, new_dir)}")
        st.rerun()
        return

    # Toggle: klik sel yang jenisnya sama dengan brush → hapus
    new_val = CELL_EMPTY if current == brush else brush

    # Pintu hanya boleh di tepi grid
    if new_val == CELL_DOOR and not _is_edge_cell(col, row, W, H):
        st.toast("🚪 Pintu hanya bisa diletakkan di tepi grid!", icon="⚠️")
        return

    # Jika menghapus kursi, hapus juga data arah
    if current == CELL_CHAIR and new_val != CELL_CHAIR:
        remove_chair_direction(col, row)

    # Jika menambah kursi baru, set arah default
    if new_val == CELL_CHAIR and current != CELL_CHAIR:
        set_chair_direction(col, row, "right")

    st.session_state.grid_state[row, col] = new_val
    st.rerun()


def _render_quick_tools(gs, W, H):
    """Quick tools: hapus semua, random, demo, export, import."""
    st.markdown("**⚡ Quick Tools**")
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)

    if qc1.button("🗑 Hapus Semua", use_container_width=True):
        st.session_state.grid_state = make_empty_grid(W, H)
        st.session_state.chair_directions = {}
        st.rerun()

    if qc2.button("🪑 Kursi Acak (6)", use_container_width=True):
        _add_random_elements(CELL_CHAIR, 6)
        st.rerun()

    if qc3.button("🧱 Halangan Acak (4)", use_container_width=True):
        _add_random_elements(CELL_OBSTACLE, 4)
        st.rerun()

    if qc4.button("🏢 Layout Demo", use_container_width=True):
        _load_demo_layout()
        st.rerun()

    # Export JSON
    layout = grid_state_to_layout(gs)
    dirs = get_chair_directions()
    export_data = {
        "layout": {f"{c},{r}": int(v) for (c, r), v in layout.items()},
        "chair_directions": {f"{c},{r}": d for (c, r), d in dirs.items()},
    }
    layout_json = json.dumps(export_data, indent=2)
    qc5.download_button(
        "💾 Export JSON", data=layout_json,
        file_name="room_layout.json", mime="application/json",
        use_container_width=True,
    )

    # Import JSON
    st.divider()
    uploaded = st.file_uploader("📂 Import Layout JSON", type="json", label_visibility="collapsed")
    if uploaded:
        _import_layout_json(uploaded, W, H)


def _add_random_elements(cell_type, n):
    """Tambahkan n elemen ke posisi kosong acak (bukan tepi untuk non-door)."""
    W = st.session_state.grid_w
    H = st.session_state.grid_h
    gs = st.session_state.grid_state
    empty = [(c, r) for c in range(1, W) for r in range(H) if gs[r, c] == CELL_EMPTY]
    random.shuffle(empty)
    for col, row in empty[:n]:
        gs[row, col] = cell_type
        if cell_type == CELL_CHAIR:
            d = random.choice(["up", "down", "left", "right"])
            set_chair_direction(col, row, d)


def _load_demo_layout():
    """Layout demo dengan pintu masuk dan kursi berarah."""
    W = st.session_state.grid_w
    H = st.session_state.grid_h
    gs = make_empty_grid(W, H)
    st.session_state.chair_directions = {}

    # Pintu masuk di tengah kolom 0
    mid_row = H // 2
    for dr in range(-1, 2):
        r = mid_row + dr
        if 0 <= r < H:
            gs[r, 0] = CELL_DOOR

    # Kursi di baris tepian menghadap ke tengah
    for col in range(2, W - 1, 3):
        if col < W:
            gs[1, col] = CELL_CHAIR
            set_chair_direction(col, 1, "down")
            gs[H - 2, col] = CELL_CHAIR
            set_chair_direction(col, H - 2, "up")

    # Tiang di sekitar titik tengah
    mid_col = W // 2
    tiang = [(mid_row - 1, mid_col), (mid_row + 1, mid_col),
             (mid_row, mid_col - 2), (mid_row, mid_col + 2)]
    for row, col in tiang:
        if 0 < row < H and 0 < col < W:
            gs[row, col] = CELL_OBSTACLE

    # Kursi di baris tengah menghadap kanan
    for col in [mid_col - 1, mid_col, mid_col + 1]:
        if 0 < col < W:
            gs[mid_row, col] = CELL_CHAIR
            set_chair_direction(col, mid_row, "right")

    st.session_state.grid_state = gs


def _import_layout_json(uploaded_file, W, H):
    """Parse file JSON dan terapkan ke grid state."""
    try:
        raw = json.load(uploaded_file)
        new_gs = make_empty_grid(W, H)
        new_dirs = {}

        # Support format baru (nested) dan lama (flat)
        if "layout" in raw:
            layout_data = raw["layout"]
            dir_data = raw.get("chair_directions", {})
        else:
            layout_data = raw
            dir_data = {}

        for key, value in layout_data.items():
            col, row = map(int, key.split(","))
            if 0 <= row < H and 0 <= col < W:
                new_gs[row, col] = int(value)

        for key, d in dir_data.items():
            col, row = map(int, key.split(","))
            if 0 <= row < H and 0 <= col < W:
                new_dirs[(col, row)] = d

        st.session_state.grid_state = new_gs
        st.session_state.chair_directions = new_dirs
        st.success("✅ Layout berhasil diimport!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal import layout: {e}")