"""
ui/panel_simulate.py
====================
Simulation panel for running the ABM and visualizing movement.

Perubahan vs versi lama:
- Heatmap render via matplotlib (plot_obstacle_heatmap) bukan Plotly.
- ExposureField dipakai sebagai incremental accumulator; diinit ulang tiap
  Start/Reset dan dipassing ke plot_obstacle_heatmap agar blur hanya dihitung
  saat ada perubahan (flag dirty).
- render_heatmap() pakai ph_hmap.pyplot() konsisten dengan render_static().
- Interval refresh heatmap saat live: HEATMAP_REFRESH_INTERVAL detik sim-time
  agar tidak rebuild tiap step.
"""

from viz.sim_plots import plot_sim_room, SimRenderer
from viz.obstacle_heatmap import plot_obstacle_heatmap, ExposureField
import matplotlib.pyplot as plt  # type: ignore
import streamlit as st  # type: ignore

from abm.model import WaitingRoomModel
from constants import CELL_CHAIR, CELL_SIZE_PX, DEFAULT_DT_S
from ui.state import grid_state_to_layout, get_chair_directions
from viz.sim_plots import plot_sim_room, SimRenderer

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
MIN_SPEED = 1
MAX_SPEED = 5
SIM_DT_S = DEFAULT_DT_S

# Seberapa sering heatmap di-refresh saat simulasi berjalan (detik sim-time).
# Naikkan nilai ini jika masih terasa berat; turunkan untuk update lebih sering.
HEATMAP_REFRESH_INTERVAL = 5.0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

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
        speed_x = st.select_slider(
            "Simulation Speed",
            options=list(range(MIN_SPEED, MAX_SPEED + 1)),
            value=int(sim_defaults.get("speed_x", 1)),
            format_func=lambda v: f"{v}×",
            help=(
                "Kecepatan simulasi. Waktu simulasi melompat lebih besar "
                "per update, FPS menyesuaikan performa. "
                "5× berarti 200 step selesai 5× lebih cepat dari 1×."
            ),
        )

        base_dt = SIM_DT_S
        effective_dt = base_dt * speed_x
        st.caption(
            f"base sim\\_dt = **{base_dt:.3f} s** · "
            f"effective dt = **{effective_dt:.3f} s** · FPS auto"
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
        speed_x=int(speed_x),
        max_steps=int(max_steps),
        seed=int(seed),
        start=start,
        stop=stop,
        reset_sim=reset,
    )


# ---------------------------------------------------------------------------
# Panel utama
# ---------------------------------------------------------------------------

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
    max_steps = float(cfg["max_steps"])

    st.session_state.sim_config = {
        "arrival_rate": cfg["arrival_rate"],
        "mean_sitting": cfg["mean_sitting"],
        "pass_through_prob": cfg["pass_through_prob"],
        "speed_x": cfg["speed_x"],
        "max_steps": cfg["max_steps"],
        "seed": cfg["seed"],
    }

    ph_step, ph_active, ph_sit, ph_pass, ph_total = _create_metric_placeholders()

    col_room, col_heat = st.columns([1.2, 1], gap="medium")
    with col_room:
        st.markdown("#### 🗺️ Room State")
        ph_room = st.empty()
    with col_heat:
        st.markdown("#### 🔥 Obstacle Heatmap")
        ph_hmap = st.empty()

    current_time = model.time_s if model is not None else 0.0
    progress_bar = st.progress(min(current_time / max(max_steps, 1.0), 1.0))
    ph_info = st.empty()

    # --- Session state untuk heatmap cache ---
    if "heatmap_cache_time" not in st.session_state:
        st.session_state.heatmap_cache_time = -1.0

    # ExposureField hidup di session_state agar persist antar rerun
    if "exposure_field" not in st.session_state or st.session_state.exposure_field is None:
        st.session_state.exposure_field = ExposureField(width, height)

    ef: ExposureField = st.session_state.exposure_field

    # ------------------------------------------------------------------ #
    # Helper render heatmap                                                #
    # ------------------------------------------------------------------ #

    def render_heatmap(force: bool = False) -> None:
        """Render heatmap ke ph_hmap.

        Hanya rebuild Figure jika:
        - force=True (misal: setelah simulasi selesai), atau
        - sudah lewat HEATMAP_REFRESH_INTERVAL sejak render terakhir.

        ExposureField.get_blurred() sendiri sudah lazy (hanya blur ulang
        jika dirty), jadi aman dipanggil lebih sering.
        """
        last_render = st.session_state.heatmap_cache_time
        elapsed = model.time_s - last_render
        if not force and elapsed < HEATMAP_REFRESH_INTERVAL:
            return

        # Sync ExposureField dari obstacle_heatmap dict jika model tidak
        # punya atribut exposure_field sendiri (backward-compat).
        source_ef = getattr(model, "exposure_field", None)
        if source_ef is not None:
            # Model sudah pakai ExposureField incremental — langsung pakai
            fig = plot_obstacle_heatmap(
                model.obstacle_heatmap,
                layout,
                width,
                height,
                exposure_field=source_ef,
            )
        else:
            # Fallback: rebuild dari dict (lebih lambat, tapi tetap pakai
            # matplotlib + gaussian_filter — jauh lebih cepat dari versi lama)
            fig = plot_obstacle_heatmap(
                model.obstacle_heatmap,
                layout,
                width,
                height,
                exposure_field=ef,
            )

        ph_hmap.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.session_state.heatmap_cache_time = model.time_s

    # ------------------------------------------------------------------ #
    # Render statis (pause / stop)                                         #
    # ------------------------------------------------------------------ #

    def render_static() -> int:
        snap = model.get_grid_snapshot()
        n_now = model.count_humans()
        n_sit = model.count_sitting()
        n_pass = model.count_passing()
        n_tot = model.total_customers
        step_value = int(model.time_s)

        st.session_state.step_count = step_value
        ph_step.metric("Step", step_value)
        ph_active.metric("Active", n_now)
        ph_sit.metric("Sitting", n_sit)
        ph_pass.metric("Passing", n_pass)
        ph_total.metric("Total Arrived", n_tot)

        fig_room = plot_sim_room(snap, width, height, CELL_SIZE_PX)
        ph_room.pyplot(fig_room)
        plt.close(fig_room)
        return n_now

    # ================================================================== #
    # Branch: running                                                      #
    # ================================================================== #

    if st.session_state.running:
        speed_x = cfg["speed_x"]
        sim_dt = SIM_DT_S * speed_x

        renderer = SimRenderer(width, height, CELL_SIZE_PX)
        try:
            while model.time_s < max_steps:
                if not st.session_state.running:
                    break

                model.step(sim_dt)

                # Sync ExposureField jika model tidak punya sendiri
                if not hasattr(model, "exposure_field"):
                    _sync_exposure_field(ef, model.obstacle_heatmap)

                sim_time_s = model.time_s
                step_value = int(sim_time_s)
                st.session_state.step_count = step_value

                snap = model.get_grid_snapshot()
                n_now = model.count_humans()
                n_sit = model.count_sitting()
                n_pass = model.count_passing()
                n_tot = model.total_customers

                ph_step.metric("Step", step_value)
                ph_active.metric("Active", n_now)
                ph_sit.metric("Sitting", n_sit)
                ph_pass.metric("Passing", n_pass)
                ph_total.metric("Total Arrived", n_tot)

                png_bytes = renderer.render(snap)
                ph_room.image(png_bytes, use_container_width=True)

                ph_info.caption(
                    f"Time {sim_time_s:.1f}/{max_steps:.1f} s "
                    f"· Speed {speed_x}x "
                    f"· Active: {n_now}"
                )

                # Heatmap di-render tiap HEATMAP_REFRESH_INTERVAL
                render_heatmap(force=False)

                progress_bar.progress(min(sim_time_s / max_steps, 1.0))

                if not st.session_state.running:
                    break

        finally:
            renderer.close()

        st.session_state.running = False
        progress_bar.progress(1.0)
        render_heatmap(force=True)
        ph_info.success(
            f"Simulation finished at {int(model.time_s)} s. "
            f"Total arrivals: {model.total_customers}."
        )

    # ================================================================== #
    # Branch: paused / stopped                                            #
    # ================================================================== #

    else:
        if st.session_state.model is not None:
            render_static()

            if model.time_s > 0:
                render_heatmap(force=True)

            if st.session_state.get("show_final_heatmap", False) or model.time_s >= max_steps:
                ph_info.success(
                    f"✓ Simulation stopped at {model.time_s:.1f} s. "
                    f"Total arrivals: {st.session_state.model.total_customers}."
                )
            elif model.time_s > 0:
                ph_info.caption(
                    f"Paused at {model.time_s:.1f} s. Press Start to resume."
                )
            else:
                ph_info.caption("Press **Start** to begin the simulation.")
        else:
            ph_info.warning(
                "⚠️ Model belum di-initialize. Tekan **Start** untuk memulai simulasi."
            )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

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
            seed=cfg["seed"],
        )

    def _reset_heatmap_state():
        st.session_state.heatmap_cache_time = -1.0
        st.session_state.exposure_field = ExposureField(width, height)

    if cfg["start"]:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = True
        st.session_state.show_final_heatmap = False
        _reset_heatmap_state()
    elif cfg["stop"]:
        st.session_state.running = False
        st.session_state.show_final_heatmap = True
        st.rerun()
    elif cfg["reset_sim"]:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = False
        st.session_state.show_final_heatmap = False
        _reset_heatmap_state()

    if st.session_state.model is None:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = False
        st.session_state.show_final_heatmap = False
        _reset_heatmap_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_exposure_field(
    ef: ExposureField,
    obstacle_heatmap: dict,
) -> None:
    """Update ExposureField dari seluruh obstacle_heatmap dict.

    Ini adalah fallback O(obstacles) yang dipanggil tiap step jika model
    tidak punya ExposureField internal sendiri.

    Catatan: metode ini me-reset dan rebuild ulang field dari dict — artinya
    masih O(obstacles * sides) per step. Untuk performa optimal, integrasikan
    ExposureField.record() langsung ke WaitingRoomModel.step().
    """
    ef.reset()
    for (col, row), sides in obstacle_heatmap.items():
        for side, weight in sides.items():
            if weight > 0:
                ef.record(col, row, side, weight)


def _create_metric_placeholders():
    cols = st.columns(5)
    return [col.empty() for col in cols]
