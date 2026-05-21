"""
ui/panel_design.py
==================
Panel mode DESIGN — editor denah ruangan interaktif.

Fungsi utama yang dipanggil dari app.py:
    panel_design()        — render seluruh UI editor
    build_sidebar_design() — render sidebar ukuran grid

Alur interaksi pengguna:
  1. Atur ukuran ruangan di sidebar → Terapkan Ukuran.
  2. Pilih brush (Kursi / Halangan / Hapus) di panel utama.
  3. Klik sel pada grid Plotly → sel di-update → Streamlit rerun.
  4. Gunakan Quick Tools untuk operasi massal atau demo.
  5. Export/import layout sebagai JSON untuk disimpan atau dibagikan.

Semua perubahan grid disimpan ke st.session_state.grid_state (numpy array).
"""

import json
import random

import numpy as np
import streamlit as st

from constants import CELL_CHAIR, CELL_EMPTY, CELL_OBSTACLE
from ui.state import grid_state_to_layout, make_empty_grid, layout_to_grid_state
from viz.editor_plot import build_editor_figure


# =============================================================================
# SIDEBAR — UKURAN RUANGAN
# =============================================================================

def build_sidebar_design() -> None:
    """
    Render sidebar untuk konfigurasi ukuran grid.

    Perubahan ukuran bersifat non-destructive: elemen yang masih muat
    di ukuran baru akan dipertahankan. Elemen yang terpotong akan hilang.
    Tombol "Terapkan Ukuran" hanya muncul jika ukuran benar-benar berubah.
    """
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:8px 0'>"
            "<h2 style='color:#f5a623;margin:0;font-family:monospace'>📐 Desain</h2>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("🏗️ Ukuran Ruangan")
        new_w = st.slider("Lebar (kolom)",  8, 30, st.session_state.grid_w)
        new_h = st.slider("Tinggi (baris)", 6, 25, st.session_state.grid_h)

        size_changed = (
            new_w != st.session_state.grid_w or
            new_h != st.session_state.grid_h
        )

        if size_changed:
            if st.button("✅ Terapkan Ukuran", use_container_width=True, type="primary"):
                _resize_grid(new_w, new_h)
                st.rerun()
        else:
            st.info(f"Ukuran aktif: **{new_w}×{new_h}**")

        st.divider()
        st.markdown("""
        **Panduan Desain:**
        - 🚪 Kolom 0 = pintu masuk pelanggan
        - 🪑 Pilih brush Kursi lalu klik sel
        - 🧱 Pilih brush Halangan lalu klik sel
        - 🗑️ Pilih brush Hapus lalu klik sel berisi
        - Klik sel yang sama jenisnya untuk menghapus
        """)


def _resize_grid(new_w: int, new_h: int) -> None:
    """
    Ubah ukuran grid sambil mempertahankan elemen yang masih muat.

    Elemen di baris/kolom yang terpotong akan hilang.
    Setelah resize, model lama diinvalidasi karena dimensi berubah.
    """
    old_gs           = st.session_state.grid_state
    new_gs           = make_empty_grid(new_w, new_h)
    old_h, old_w     = old_gs.shape

    # Copy irisan yang masih masuk ke ukuran baru
    copy_h = min(old_h, new_h)
    copy_w = min(old_w, new_w)
    new_gs[:copy_h, :copy_w] = old_gs[:copy_h, :copy_w]

    st.session_state.grid_w    = new_w
    st.session_state.grid_h    = new_h
    st.session_state.grid_state = new_gs
    st.session_state.model     = None   # Invalidasi model lama


# =============================================================================
# PANEL UTAMA — EDITOR
# =============================================================================

def panel_design() -> None:
    """
    Render panel editor denah ruangan:
      • Info bar (jumlah elemen + validasi kapasitas)
      • Brush selector (Kursi / Halangan / Hapus)
      • Grid Plotly interaktif + logika klik
      • Quick Tools (hapus semua, random fill, demo, export/import)
    """
    W  = st.session_state.grid_w
    H  = st.session_state.grid_h
    gs = st.session_state.grid_state

    # ── Info bar ──────────────────────────────────────────────────────────────
    _render_info_bar(gs, W, H)
    st.divider()

    # ── Brush selector ────────────────────────────────────────────────────────
    _render_brush_selector()

    # ── Grid editor Plotly ────────────────────────────────────────────────────
    fig        = build_editor_figure(gs, W, H, st.session_state.brush)
    click_data = st.plotly_chart(
        fig,
        use_container_width = True,
        key                 = "grid_editor",
        on_select           = "rerun",
    )

    # ── Proses klik pada sel grid ─────────────────────────────────────────────
    _handle_cell_click(click_data, gs, W, H)

    # ── Quick Tools ───────────────────────────────────────────────────────────
    st.divider()
    _render_quick_tools(gs, W, H)


# =============================================================================
# SUB-KOMPONEN PANEL DESIGN
# =============================================================================

def _render_info_bar(gs: np.ndarray, W: int, H: int) -> None:
    """Tampilkan metrik jumlah elemen dan validasi kapasitas."""
    n_chairs    = int(np.sum(gs == CELL_CHAIR))
    n_obstacles = int(np.sum(gs == CELL_OBSTACLE))
    n_empty     = W * H - n_chairs - n_obstacles
    total       = W * H

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🪑 Kursi",      n_chairs)
    c2.metric("🧱 Halangan",   n_obstacles)
    c3.metric("⬜ Sel Kosong", n_empty)
    c4.metric("📐 Grid",       f"{W}×{H}")

    used_ratio = (n_chairs + n_obstacles) / total
    if used_ratio >= 1.0:
        st.error("⚠️ Grid penuh! Hapus beberapa elemen terlebih dahulu.")
    elif used_ratio > 0.7:
        st.warning("⚠️ Grid hampir penuh (>70%). Pelanggan mungkin kesulitan bergerak.")


def _render_brush_selector() -> None:
    """
    Render tombol pemilih brush.

    Brush aktif ditampilkan sebagai tombol 'primary' (highlighted).
    Perubahan brush tidak mengubah grid — hanya mengubah session_state.brush.
    """
    st.markdown("**🖌️ Brush Aktif** — Pilih tipe lalu klik sel pada grid")

    col_b1, col_b2, col_b3, _ = st.columns([1, 1, 1, 4])

    if col_b1.button(
        "🪑 Kursi",
        use_container_width = True,
        type = "primary" if st.session_state.brush == CELL_CHAIR else "secondary",
    ):
        st.session_state.brush = CELL_CHAIR
        st.rerun()

    if col_b2.button(
        "🧱 Halangan",
        use_container_width = True,
        type = "primary" if st.session_state.brush == CELL_OBSTACLE else "secondary",
    ):
        st.session_state.brush = CELL_OBSTACLE
        st.rerun()

    if col_b3.button(
        "🗑️ Hapus",
        use_container_width = True,
        type = "primary" if st.session_state.brush == CELL_EMPTY else "secondary",
    ):
        st.session_state.brush = CELL_EMPTY
        st.rerun()


def _handle_cell_click(click_data, gs: np.ndarray, W: int, H: int) -> None:
    """
    Proses event klik dari figure Plotly.

    Streamlit (≥1.33) mengembalikan selection object saat on_select="rerun".
    Klik pada scatter invisible → koordinat (x=col, y=row).

    Logika toggle:
      • Klik sel kosong dengan brush aktif → taruh brush
      • Klik sel yang SAMA jenisnya dengan brush → hapus (toggle)
      • Klik sel kolom 0 dengan brush non-hapus → tolak (pintu masuk)
    """
    if not (click_data and hasattr(click_data, "selection")):
        return

    sel = click_data.selection
    pts = sel.get("points", []) if isinstance(sel, dict) else []
    if not pts:
        return

    pt  = pts[0]
    col = int(round(pt.get("x", -1)))
    row = int(round(pt.get("y", -1)))

    if not (0 <= col < W and 0 <= row < H):
        return   # Klik di luar batas grid

    current = int(gs[row, col])
    brush   = st.session_state.brush

    # Toggle: klik sel yang jenisnya sama dengan brush → hapus
    new_val = CELL_EMPTY if current == brush else brush

    # Proteksi kolom 0 (pintu masuk) — tidak boleh diblokir
    if col == 0 and new_val != CELL_EMPTY:
        st.toast("🚪 Kolom 0 adalah pintu masuk, tidak bisa diblokir.", icon="⚠️")
        return

    st.session_state.grid_state[row, col] = new_val
    st.rerun()


def _render_quick_tools(gs: np.ndarray, W: int, H: int) -> None:
    """
    Render quick tools: hapus semua, random fill, demo, export, import.

    Export: serialize layout ke JSON dengan key "col,row".
    Import: parse JSON dan terapkan ke grid state saat ini.
    """
    st.markdown("**⚡ Quick Tools**")
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)

    if qc1.button("🗑 Hapus Semua", use_container_width=True):
        st.session_state.grid_state = make_empty_grid(W, H)
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
    layout      = grid_state_to_layout(gs)
    layout_json = json.dumps(
        {f"{col},{row}": int(v) for (col, row), v in layout.items()},
        indent=2,
    )
    qc5.download_button(
        "💾 Export JSON",
        data      = layout_json,
        file_name = "room_layout.json",
        mime      = "application/json",
        use_container_width = True,
    )

    # Import JSON
    st.divider()
    uploaded = st.file_uploader(
        "📂 Import Layout JSON",
        type              = "json",
        label_visibility  = "collapsed",
    )
    if uploaded:
        _import_layout_json(uploaded, W, H)


def _add_random_elements(cell_type: int, n: int) -> None:
    """
    Tambahkan n elemen bertipe cell_type ke posisi kosong acak.

    Kolom 0 (pintu masuk) tidak digunakan sebagai lokasi elemen.
    """
    W  = st.session_state.grid_w
    H  = st.session_state.grid_h
    gs = st.session_state.grid_state

    empty_cells = [
        (col, row)
        for col in range(1, W)
        for row in range(H)
        if gs[row, col] == CELL_EMPTY
    ]
    random.shuffle(empty_cells)
    for col, row in empty_cells[:n]:
        gs[row, col] = cell_type


def _load_demo_layout() -> None:
    """
    Terapkan layout demo representatif ke grid saat ini.

    Pola layout:
      • Kursi di baris pertama dan terakhir (tepian atas/bawah) dengan jarak 3
      • Tiang di sekitar tengah ruangan (membentuk pola simetris)
      • Tiga kursi di baris tengah di antara tiang
    """
    W  = st.session_state.grid_w
    H  = st.session_state.grid_h
    gs = make_empty_grid(W, H)

    # Kursi di baris tepian (row 1 dan row H-2)
    for col in range(2, W - 1, 3):
        gs[1,     col] = CELL_CHAIR
        gs[H - 2, col] = CELL_CHAIR

    # Tiang di sekitar titik tengah
    mid_col = W // 2
    mid_row = H // 2
    tiang_positions = [
        (mid_row - 1, mid_col),
        (mid_row + 1, mid_col),
        (mid_row,     mid_col - 2),
        (mid_row,     mid_col + 2),
    ]
    for row, col in tiang_positions:
        if 0 < row < H and 0 < col < W:
            gs[row, col] = CELL_OBSTACLE

    # Kursi di baris tengah (di antara tiang)
    for col in [mid_col - 1, mid_col, mid_col + 1]:
        if 0 < col < W:
            gs[mid_row, col] = CELL_CHAIR

    st.session_state.grid_state = gs


def _import_layout_json(uploaded_file, W: int, H: int) -> None:
    """
    Parse file JSON yang di-upload dan terapkan ke grid state.

    Format JSON yang diterima:
        { "col,row": tipe_sel, ... }
    Sel di luar batas grid saat ini diabaikan secara diam-diam.
    """
    try:
        raw    = json.load(uploaded_file)
        new_gs = make_empty_grid(W, H)
        for key, value in raw.items():
            col, row = map(int, key.split(","))
            if 0 <= row < H and 0 <= col < W:
                new_gs[row, col] = int(value)
        st.session_state.grid_state = new_gs
        st.success("✅ Layout berhasil diimport!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal import layout: {e}")