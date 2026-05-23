"""
ui/panel_simulate.py
====================
Simulation panel for running the ABM and visualizing movement.
"""

import time

import matplotlib.pyplot as plt
import streamlit as st

from abm.model import WaitingRoomModel
from constants import CELL_CHAIR, CELL_SIZE_PX, DEFAULT_DT_S
from ui.state import grid_state_to_layout, get_chair_directions
from viz.sim_plots import plot_sim_room, SimRenderer

# Frames per step: berapa frame dirender untuk setiap 1 step (1 detik simulasi)
MIN_FPSTEP = 4
MAX_FPSTEP = 30
DEFAULT_FPSTEP = 10

# Simulation speed multiplier (hanya mempengaruhi wall-clock, bukan kecepatan agen)
MIN_SPEED = 1
MAX_SPEED = 5

LEGACY_MOTION_DT = {
    1: 0.2,
    2: 0.15,
    3: 0.1,
    4: 0.08,
    5: 0.05,
}


def _resolve_default_fpstep(sim_defaults: dict) -> int:
    """Backward-compatible: convert lama fps/sim_dt/motion_level ke fps_step baru."""
    if "fps_step" in sim_defaults:
        val = int(sim_defaults["fps_step"])
        return max(MIN_FPSTEP, min(MAX_FPSTEP, val))

    if "fps" in sim_defaults:
        val = int(float(sim_defaults["fps"]))
        return max(MIN_FPSTEP, min(MAX_FPSTEP, val))

    if "sim_dt" in sim_defaults:
        sim_dt = float(sim_defaults["sim_dt"])
        if sim_dt > 1e-9:
            val = int(round(1.0 / sim_dt))
            return max(MIN_FPSTEP, min(MAX_FPSTEP, val))

    if "motion_level" in sim_defaults:
        level = int(sim_defaults.get("motion_level", 3))
        sim_dt = LEGACY_MOTION_DT.get(level, DEFAULT_DT_S)
        if sim_dt > 1e-9:
            val = int(round(1.0 / sim_dt))
            return max(MIN_FPSTEP, min(MAX_FPSTEP, val))

    return DEFAULT_FPSTEP


def build_sidebar_sim() -> dict:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:8px 0'>"
            "<h2 style='color:#f5a623;margin:0;font-family:monospace'>Simulation</h2>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("Arrivals")
        sim_defaults = st.session_state.get("sim_config", {})
        arrival_rate = st.slider(
            "Lambda (people/sec)",
            0.05,
            3.0,
            float(sim_defaults.get("arrival_rate", 0.6)),
            0.05,
            help="Arrivals per second, sampled using Poisson distribution",
        )

        st.subheader("Human Behavior")
        mean_sitting = st.slider(
            "Average sitting time (sec)",
            5,
            180,
            int(sim_defaults.get("mean_sitting", 20)),
            help="Sitting time follows an exponential distribution",
        )
        pass_prob = st.slider(
            "Pass-through probability",
            0.0,
            1.0,
            float(sim_defaults.get("pass_through_prob", 0.5)),
            0.05,
            help="Chance to pass through without sitting",
        )

        st.subheader("Playback")
        fps_step_default = _resolve_default_fpstep(sim_defaults)
        fps_step = st.slider(
            "Frames per Step",
            MIN_FPSTEP,
            MAX_FPSTEP,
            fps_step_default,
            1,
            help=(
                "Jumlah frame yang dirender per 1 step (= 1 detik simulasi). "
                "Makin tinggi = gerakan agen lebih halus."
            ),
        )
        speed_x = st.select_slider(
            "Simulation Speed",
            options=list(range(MIN_SPEED, MAX_SPEED + 1)),
            value=int(sim_defaults.get("speed_x", 1)),
            format_func=lambda v: f"{v}×",
            help=(
                "Kecepatan playback. Kecepatan jalan agen tidak berubah — "
                "hanya wall-clock yang dipercepat. "
                "5× berarti 200 step selesai 5× lebih cepat dari 1×."
            ),
        )

        # frame_dt: detik simulasi per frame — TIDAK bergantung speed_x
        # agar kecepatan agen (px/step) selalu sama di semua speed level
        frame_dt = 1.0 / float(fps_step)

        # wall-clock delay per frame diperkecil sesuai speed_x
        wall_fps = fps_step * speed_x  # target frame/detik wall-clock

        st.caption(
            f"frame\\_dt = **{frame_dt:.3f} s** · "
            f"wall FPS ≈ **{wall_fps}** · "
            f"1 step = **{fps_step} frames**"
        )

        st.subheader("Simulation Control")
        max_steps = st.number_input(
            "Max steps",
            20,
            2000,
            int(sim_defaults.get("max_steps", 200)),
            10,
        )
        seed = st.number_input(
            "Random seed",
            0,
            9999,
            int(sim_defaults.get("seed", 42)),
        )

        st.divider()
        col_start, col_stop = st.columns(2)
        start = col_start.button("Start", use_container_width=True, type="primary")
        stop = col_stop.button("Stop", use_container_width=True)
        reset = st.button("Reset Simulation", use_container_width=True)

    return dict(
        arrival_rate=arrival_rate,
        mean_sitting=float(mean_sitting),
        pass_through_prob=float(pass_prob),
        fps_step=int(fps_step),
        speed_x=int(speed_x),
        frame_dt=float(frame_dt),
        wall_fps=int(wall_fps),
        max_steps=int(max_steps),
        seed=int(seed),
        start=start,
        stop=stop,
        reset_sim=reset,
    )


def panel_simulate(cfg: dict) -> None:
    width = st.session_state.grid_w
    height = st.session_state.grid_h
    gs = st.session_state.grid_state
    layout = grid_state_to_layout(gs)

    n_chairs = sum(1 for v in layout.values() if v == CELL_CHAIR)
    if n_chairs == 0:
        st.info("No chairs placed. Agents will only pass through.")

    _handle_controls(cfg, width, height, layout)
    model: WaitingRoomModel = st.session_state.model

    st.session_state.sim_config = {
        "arrival_rate": cfg["arrival_rate"],
        "mean_sitting": cfg["mean_sitting"],
        "pass_through_prob": cfg["pass_through_prob"],
        "fps_step": cfg["fps_step"],
        "speed_x": cfg["speed_x"],
        "max_steps": cfg["max_steps"],
        "seed": cfg["seed"],
    }

    ph_step, ph_active, ph_sit, ph_pass, ph_total = _create_metric_placeholders()
    ph_room = _create_plot_placeholders()
    progress_bar = st.progress(st.session_state.step_count / max(cfg["max_steps"], 1))
    ph_info = st.empty()

    def render_static():
        """Render sekali pakai untuk kondisi pause — pakai plot_sim_room biasa."""
        snap = model.get_grid_snapshot()
        n_now = model.count_humans()
        n_sit = model.count_sitting()
        n_pass = model.count_passing()
        n_tot = model.total_customers
        ph_step.metric("Step", st.session_state.step_count)
        ph_active.metric("Active", n_now)
        ph_sit.metric("Sitting", n_sit)
        ph_pass.metric("Passing", n_pass)
        ph_total.metric("Total Arrived", n_tot)
        fig_room = plot_sim_room(snap, width, height, CELL_SIZE_PX)
        ph_room.pyplot(fig_room)
        plt.close(fig_room)
        return n_now

    if st.session_state.running:
        fps_step  = cfg["fps_step"]
        frame_dt  = cfg["frame_dt"]   # 1/fps_step — tidak bergantung speed_x
        speed_x   = cfg["speed_x"]

        # Budget waktu wall-clock per STEP (bukan per frame).
        # Agen tetap di-step fps_step kali per step dengan frame_dt kecil
        # agar fisika halus, tapi render dilakukan sesering yang masih muat
        # dalam budget ini. Dengan cara ini fps_step TIDAK mempengaruhi
        # kecepatan wall-clock — hanya kehalusan fisika.
        wall_step_budget = 1.0 / float(speed_x)

        renderer = SimRenderer(width, height, CELL_SIZE_PX)
        try:
            for _ in range(st.session_state.step_count, cfg["max_steps"]):
                if not st.session_state.running:
                    break

                step_start = time.perf_counter()

                # --- Jalankan semua sub-step fisika untuk 1 step penuh ---
                # Render dilakukan di tengah jika ada sisa waktu, atau minimal
                # sekali di akhir sub-step terakhir.
                next_render_at = 0       # index sub-step kapan render berikutnya
                render_cost    = 0.05    # estimasi awal biaya render (detik), adaptif
                renders_done   = 0

                for sub in range(fps_step):
                    model.step(frame_dt)

                    now = time.perf_counter()
                    elapsed = now - step_start

                    # Sisa budget setelah sub-step ini
                    remaining_budget = wall_step_budget - elapsed

                    # Render jika: ini saatnya render (sub >= next_render_at)
                    # DAN (masih ada cukup budget ATAU ini sub-step terakhir)
                    # BARU - Render lebih sering untuk smooth motion
                    # Render setiap 2 frame, atau minimal di frame terakhir
                    render_interval = max(1, fps_step // 5)  # ~5 render per step
                    should_render = (sub % render_interval == 0) or (sub == fps_step - 1)
                    if should_render:
                        render_start = time.perf_counter()

                        snap  = model.get_grid_snapshot()
                        n_now = model.count_humans()
                        n_sit = model.count_sitting()
                        n_pass = model.count_passing()
                        n_tot = model.total_customers

                        ph_step.metric("Step", st.session_state.step_count)
                        ph_active.metric("Active",        n_now)
                        ph_sit.metric("Sitting",          n_sit)
                        ph_pass.metric("Passing",         n_pass)
                        ph_total.metric("Total Arrived",  n_tot)

                        png_bytes = renderer.render(snap)
                        ph_room.image(png_bytes, use_container_width=True)

                        ph_info.caption(
                            f"Step {st.session_state.step_count + 1}/{cfg['max_steps']} "
                            f"· sub {sub + 1}/{fps_step} "
                            f"· Speed {speed_x}x "
                            f"· Active: {n_now}"
                        )

                        actual_cost = time.perf_counter() - render_start
                        # Update estimasi biaya render (exponential moving average)
                        render_cost = 0.7 * render_cost + 0.3 * actual_cost
                        renders_done += 1

                        # Jadwalkan render berikutnya: lewati sub-step secukupnya
                        # agar render berikutnya tidak langsung melebihi budget
                        remaining_after = wall_step_budget - (time.perf_counter() - step_start)
                        if remaining_after > render_cost and render_cost > 0:
                            skip = max(1, int(render_cost / frame_dt))
                        else:
                            skip = fps_step  # tidak ada waktu lagi, skip semua
                        next_render_at = sub + skip

                # --- Habiskan sisa budget dengan sleep ---
                elapsed = time.perf_counter() - step_start
                leftover = wall_step_budget - elapsed
                if leftover > 0:
                    time.sleep(leftover)

                if not st.session_state.running:
                    break

                st.session_state.step_count += 1
                progress_bar.progress(st.session_state.step_count / cfg["max_steps"])

        finally:
            renderer.close()

        st.session_state.running = False
        progress_bar.progress(1.0)
        ph_info.success(
            f"Simulation finished at {cfg['max_steps']} steps. "
            f"Total arrivals: {model.total_customers}."
        )
    else:
        # Check jika model belum di-initialize sebelum render
        if st.session_state.model is not None:
            render_static()
            if st.session_state.step_count > 0:
                ph_info.caption(
                    f"Paused at step {st.session_state.step_count}. Press Start to resume."
                )
            else:
                ph_info.caption("Press Start to begin the simulation.")
        else:
            ph_info.caption("⚠️ Model belum di-initialize. Tekan **▶ Mulai** untuk memulai simulasi.")

def _handle_controls(cfg, width, height, layout):
    chair_dirs = get_chair_directions()
    door_probs = st.session_state.get("door_probs", {})
    

    def _new_model():
        return WaitingRoomModel(
            width=width,
            height=height,
            layout=layout,
            chair_directions=chair_dirs,
            door_probs=door_probs,
            arrival_rate=cfg["arrival_rate"],
            mean_sitting_s=cfg["mean_sitting"],
            pass_through_prob=cfg["pass_through_prob"],
            fps_step=cfg["fps_step"],  # <-- TAMBAHKAN INI
            seed=cfg["seed"],
        )

    if cfg["start"]:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = True
    elif cfg["stop"]:
        st.session_state.running = False
    elif cfg["reset_sim"] or st.session_state.model is None:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = False


def _create_metric_placeholders():
    cols = st.columns(5)
    return [col.empty() for col in cols]


def _create_plot_placeholders():
    st.markdown("#### Room State")
    return st.empty()