"""
ui/panel_simulate.py
====================
Panel mode SIMULATE — menjalankan ABM dan menampilkan visualisasi real-time.

Fungsi utama yang dipanggil dari app.py:
    panel_simulate(cfg)        — render seluruh UI simulasi
    build_sidebar_sim()        — render sidebar parameter simulasi

Alur kerja:
  1. Layout dari panel_design (grid_state) dikonversi ke format model.
  2. WaitingRoomModel dibuat dengan parameter dari sidebar.
  3. Loop animasi: setiap iterasi memanggil model.step() lalu memperbarui
     semua placeholder Streamlit (metrik, plot ruangan, dual heatmap).
  4. Setelah selesai (atau stop), tampilkan tabel rekomendasi lokasi iklan.

Penting: model disimpan di session_state agar state tidak hilang saat rerun.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from abm.model import WaitingRoomModel
from constants import CELL_OBSTACLE
from ui.state import grid_state_to_layout
from viz.sim_plots import plot_dual_heatmap, plot_sim_room


# =============================================================================
# SIDEBAR — PARAMETER SIMULASI
# =============================================================================

def build_sidebar_sim() -> dict:
    """
    Render sidebar untuk konfigurasi parameter simulasi.

    Returns
    -------
    dict berisi semua parameter simulasi + state tombol (start/stop/reset).
    """
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:8px 0'>"
            "<h2 style='color:#f5a623;margin:0;font-family:monospace'>⚙️ Simulasi</h2>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("👥 Kedatangan (Poisson)")
        arrival_rate = st.slider(
            "λ (kedatangan/step)", 0.05, 2.0, 0.4, 0.05,
            help="Jumlah pelanggan yang datang per step ~ Poisson(λ)",
        )

        st.subheader("⏱️ Perilaku Pelanggan")
        mean_sitting = st.slider(
            "Rata-rata duduk (step)", 5, 80, 20,
            help="Durasi duduk ~ Eksponensial(mean). Min 3 step.",
        )
        gaze_range = st.slider(
            "Jarak pandang (sel)", 2, 15, 8,
            help="Panjang maksimal ray sorot mata dalam jumlah sel.",
        )

        st.subheader("🔁 Kontrol Simulasi")
        max_steps  = st.number_input("Maks. steps", 20, 1000, 150, 10)
        step_delay = st.slider(
            "Delay/step (dtk)", 0.0, 1.0, 0.08, 0.02,
            help="Jeda antar step untuk keperluan animasi.",
        )
        seed = st.number_input("Random seed", 0, 9999, 42)

        st.divider()
        col_start, col_stop = st.columns(2)
        start = col_start.button("▶ Mulai",     use_container_width=True, type="primary")
        stop  = col_stop.button("⏹ Stop",       use_container_width=True)
        reset = st.button("🔄 Reset Simulasi",  use_container_width=True)

    return dict(
        arrival_rate = arrival_rate,
        mean_sitting = float(mean_sitting),
        gaze_range   = gaze_range,
        max_steps    = int(max_steps),
        step_delay   = step_delay,
        seed         = int(seed),
        start        = start,
        stop         = stop,
        reset_sim    = reset,
    )


# =============================================================================
# PANEL UTAMA — SIMULASI
# =============================================================================

def panel_simulate(cfg: dict) -> None:
    """
    Render panel simulasi ABM.

    Alur:
      1. Validasi: pastikan ada kursi di layout.
      2. Proses tombol (Start / Stop / Reset) → update session_state.model.
      3. Buat placeholder Streamlit untuk metrik dan plot.
      4. Loop animasi (hanya jika running=True).
      5. Setelah selesai: tampilkan rekomendasi lokasi iklan.

    Parameter
    ---------
    cfg : dict
        Output dari build_sidebar_sim().
    """
    W      = st.session_state.grid_w
    H      = st.session_state.grid_h
    gs     = st.session_state.grid_state
    layout = grid_state_to_layout(gs)

    # ── Validasi layout ───────────────────────────────────────────────────────
    n_chairs = sum(1 for v in layout.values() if v != CELL_OBSTACLE)
    if n_chairs == 0:
        st.warning(
            "⚠️ Belum ada kursi di denah. "
            "Buka tab **🏗️ Design Ruangan** untuk menambah kursi."
        )
        return

    # ── Proses kontrol tombol ─────────────────────────────────────────────────
    _handle_controls(cfg, W, H, layout)

    model: WaitingRoomModel = st.session_state.model

    # ── Placeholder Streamlit (diupdate setiap step) ──────────────────────────
    ph_step, ph_active, ph_total, ph_fheat, ph_oheat = _create_metric_placeholders()
    ph_room, ph_hmap = _create_plot_placeholders()
    progress_bar     = st.progress(st.session_state.step_count / max(cfg["max_steps"], 1))
    ph_info          = st.empty()

    # ── Fungsi render satu frame ──────────────────────────────────────────────
    def render_frame():
        """Perbarui semua placeholder dengan data model terkini."""
        snap  = model.get_grid_snapshot()
        n_now = model.count_customers()
        n_tot = model.total_customers
        fmax  = float(model.floor_heatmap.max())
        omax  = float(model.obstacle_heatmap.max())

        ph_step.metric("Step",             st.session_state.step_count)
        ph_active.metric("Aktif",          n_now)
        ph_total.metric("Total Datang",    n_tot)
        ph_fheat.metric("Max Floor Exp.",  f"{fmax:.0f}")
        ph_oheat.metric("Max Obs. Exp.",   f"{omax:.0f}")

        fig_room = plot_sim_room(snap, W, H)
        ph_room.pyplot(fig_room)
        plt.close(fig_room)   # Bebaskan memori matplotlib

        fig_heat = plot_dual_heatmap(
            model.floor_heatmap, model.obstacle_heatmap, layout, W, H
        )
        ph_hmap.pyplot(fig_heat)
        plt.close(fig_heat)

    # ── Loop animasi ──────────────────────────────────────────────────────────
    if st.session_state.running:
        for _ in range(st.session_state.step_count, cfg["max_steps"]):
            if not st.session_state.running:
                break

            model.step()
            st.session_state.step_count += 1

            render_frame()
            progress_bar.progress(st.session_state.step_count / cfg["max_steps"])
            ph_info.caption(
                f"Step {st.session_state.step_count}/{cfg['max_steps']} — "
                f"Aktif: {model.count_customers()} | "
                f"Total: {model.total_customers} pelanggan"
            )
            time.sleep(cfg["step_delay"])

        # Loop selesai
        st.session_state.running = False
        progress_bar.progress(1.0)
        ph_info.success(
            f"✅ Simulasi selesai! {cfg['max_steps']} steps · "
            f"{model.total_customers} pelanggan dilayani."
        )
        st.divider()
        render_recommendations(model, layout)

    else:
        # Simulasi tidak berjalan → tampilkan frame statis
        render_frame()

        if st.session_state.step_count > 0:
            ph_info.caption(
                f"Dijeda di step {st.session_state.step_count}. "
                f"Tekan **▶ Mulai** untuk melanjutkan."
            )
        else:
            ph_info.caption("Tekan **▶ Mulai** untuk memulai simulasi.")

        # Tampilkan rekomendasi jika simulasi sudah selesai
        if st.session_state.step_count >= cfg["max_steps"] and st.session_state.step_count > 0:
            st.divider()
            render_recommendations(model, layout)


# =============================================================================
# SUB-KOMPONEN PANEL SIMULASI
# =============================================================================

def _handle_controls(cfg: dict, W: int, H: int, layout: dict) -> None:
    """
    Proses state tombol Start / Stop / Reset.

    Start  → Buat model baru dari layout terkini, set running=True.
    Stop   → Set running=False (loop akan berhenti di iterasi berikutnya).
    Reset  → Buat model baru, set step_count=0, running=False.
    None   → Jika model belum ada, inisialisasi tanpa menjalankan.
    """
    def _new_model() -> WaitingRoomModel:
        return WaitingRoomModel(
            width        = W,
            height       = H,
            layout       = layout,
            arrival_rate = cfg["arrival_rate"],
            mean_sitting = cfg["mean_sitting"],
            gaze_range   = cfg["gaze_range"],
            seed         = cfg["seed"],
        )

    if cfg["start"]:
        st.session_state.model      = _new_model()
        st.session_state.step_count = 0
        st.session_state.running    = True

    elif cfg["stop"]:
        st.session_state.running = False

    elif cfg["reset_sim"] or st.session_state.model is None:
        st.session_state.model      = _new_model()
        st.session_state.step_count = 0
        st.session_state.running    = False


def _create_metric_placeholders():
    """Buat 5 kolom placeholder untuk metrik simulasi."""
    cols = st.columns(5)
    return (col.empty() for col in cols)


def _create_plot_placeholders():
    """Buat 2 kolom placeholder untuk plot ruangan dan heatmap."""
    col_room, col_heat = st.columns([1, 1], gap="medium")
    with col_room:
        st.markdown("#### 🗺️ Kondisi Ruangan")
        ph_room = st.empty()
    with col_heat:
        st.markdown("#### 🔥 Dual Heatmap Exposure")
        ph_hmap = st.empty()
    return ph_room, ph_hmap


# =============================================================================
# REKOMENDASI PENEMPATAN IKLAN
# =============================================================================

def render_recommendations(model: WaitingRoomModel, layout: dict) -> None:
    """
    Tampilkan tabel rekomendasi lokasi penempatan iklan.

    Tab 1 — Floor Exposure:
        TOP-5 sel lantai dengan akumulasi sorot mata terbanyak.
        Rekomendasi: banner lantai, display stand, totem.

    Tab 2 — Obstacle Exposure:
        TOP-5 tiang/halangan yang paling sering ditatap pelanggan.
        Rekomendasi: stiker tiang, banner menempel, sign holder.

    Parameter
    ---------
    model  : WaitingRoomModel dengan heatmap terisi.
    layout : dict layout untuk mengetahui posisi obstacle.
    """
    st.markdown("### 📍 Rekomendasi Penempatan Iklan")

    obs_positions = {k for k, v in layout.items() if v == CELL_OBSTACLE}
    tab_floor, tab_obstacle = st.tabs([
        "🟢 Banner / Display Stand (Lantai)",
        "🟠 Stiker / Banner di Tiang",
    ])

    with tab_floor:
        _render_top_locations(
            heatmap        = model.floor_heatmap,
            obstacle_only  = False,
            obs_positions  = obs_positions,
            empty_msg      = "Belum ada floor exposure data.",
        )

    with tab_obstacle:
        _render_top_locations(
            heatmap        = model.obstacle_heatmap,
            obstacle_only  = True,
            obs_positions  = obs_positions,
            empty_msg      = (
                "Belum ada obstacle exposure. "
                "Pastikan ada halangan (tiang) di denah."
            ),
        )


def _render_top_locations(
    heatmap: np.ndarray,
    obstacle_only: bool,
    obs_positions: set,
    empty_msg: str,
    n: int = 5,
) -> None:
    """
    Render metrik TOP-n lokasi terpanas dari heatmap.

    Parameter
    ---------
    heatmap       : np.ndarray — floor atau obstacle heatmap
    obstacle_only : bool — jika True, filter hanya sel obstacle
    obs_positions : set  — kumpulan posisi obstacle
    empty_msg     : str  — pesan jika tidak ada data
    n             : int  — jumlah top lokasi yang ditampilkan
    """
    tops = []
    for idx in np.argsort(heatmap.flatten())[::-1]:
        if len(tops) >= n:
            break
        rx, ry = np.unravel_index(idx, heatmap.shape)
        value  = heatmap[rx, ry]
        if value == 0:
            break
        if obstacle_only and (rx, ry) not in obs_positions:
            continue
        tops.append((rx, ry, value))

    if not tops:
        st.info(empty_msg)
        return

    cols = st.columns(len(tops))
    for i, (rx, ry, value) in enumerate(tops):
        label = "Tiang" if obstacle_only else "Sel"
        with cols[i]:
            st.metric(
                label = f"Rank #{i + 1}",
                value = f"{label} ({rx},{ry})",
                delta = f"{value:.0f} hits",
            )