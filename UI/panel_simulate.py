"""
ui/panel_simulate.py
====================
Simulation panel for running the ABM and visualizing movement.
"""

from viz.sim_plots import plot_sim_room, SimRenderer
from viz.obstacle_heatmap import plot_obstacle_heatmap
import matplotlib.pyplot as plt # type: ignore
import streamlit as st # type: ignore

from abm.model import WaitingRoomModel
from constants import CELL_CHAIR, CELL_SIZE_PX, DEFAULT_DT_S
from ui.state import grid_state_to_layout, get_chair_directions
from viz.sim_plots import plot_sim_room, SimRenderer

# Simulation speed multiplier (hanya mempengaruhi wall-clock, bukan kecepatan agen)
MIN_SPEED = 1
MAX_SPEED = 5
# Fixed simulation timestep (detik simulasi per update)
SIM_DT_S = DEFAULT_DT_S


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
    
    # Plot placeholders — 2 kolom
    col_room, col_heat = st.columns([1.2, 1], gap="medium")
    
    with col_room:
        st.markdown("####  Room State")
        ph_room = st.empty()
    
    with col_heat:
        st.markdown("####  Obstacle Heatmap")
        ph_hmap = st.empty()
    
    current_time = model.time_s if model is not None else 0.0
    progress_bar = st.progress(min(current_time / max(max_steps, 1.0), 1.0))
    ph_info = st.empty()
    
    
    # Initialize heatmap cache
    if "fig_hmap_cache" not in st.session_state:
        st.session_state.fig_hmap_cache = None
    if "last_heatmap_interval" not in st.session_state:
        st.session_state.last_heatmap_interval = -1

    HEATMAP_UPDATE_INTERVAL = 10  # Update heatmap setiap 10 steps

    def render_heatmap() -> None:
        """Update dan render heatmap"""
        st.session_state.fig_hmap_cache = plot_obstacle_heatmap(
            model.obstacle_heatmap,
            layout,
            width,
            height,
        )
        ph_hmap.plotly_chart(
            st.session_state.fig_hmap_cache,
            use_container_width=True,
        )

    def should_update_heatmap() -> bool:
        """Check jika sudah saatnya update heatmap (setiap 10 step berdasarkan model time)"""
        current_interval = int(model.time_s / HEATMAP_UPDATE_INTERVAL)
        if current_interval > st.session_state.last_heatmap_interval:
            st.session_state.last_heatmap_interval = current_interval
            return True
        return False

    def render_static():
        """Render sekali pakai untuk kondisi pause — pakai plot_sim_room biasa."""
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
        # fig_hmap = plot_obstacle_heatmap(
        #     model.obstacle_heatmap, layout, width, height
        # )
        # ph_hmap.plotly_chart(fig_hmap, use_container_width=True, key=f"heatmap_step_{st.session_state.step_count}")
        plt.close(fig_room)
        return n_now

    if st.session_state.running:
        speed_x = cfg["speed_x"]
        base_dt = SIM_DT_S
        sim_dt = base_dt * speed_x

        renderer = SimRenderer(width, height, CELL_SIZE_PX)
        try:
            while model.time_s < max_steps:
                if not st.session_state.running:
                    break

                model.step(sim_dt)

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

                # Update heatmap HANYA setiap 10 step (bukan setiap frame)
                if should_update_heatmap():
                    render_heatmap()

                ph_info.caption(
                    f"Time {sim_time_s:.1f}/{max_steps:.1f} s "
                    f"· Speed {speed_x}x "
                    f"· Active: {n_now}"
                )

                if not st.session_state.running:
                    break

                progress_bar.progress(min(sim_time_s / max_steps, 1.0))

        finally:
            renderer.close()

        st.session_state.running = False
        progress_bar.progress(1.0)
        # Render final heatmap saat selesai
        render_heatmap()
        ph_info.success(
            f"Simulation finished at {int(model.time_s)} s. "
            f"Total arrivals: {model.total_customers}."
        )
    else:
        # Check jika model belum di-initialize sebelum render
        if st.session_state.model is not None:
            render_static()
            
            # Tampilkan final heatmap saat stop atau selesai
            if model.time_s > 0 and should_update_heatmap():
                render_heatmap()

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
            ph_info.warning("⚠️ Model belum di-initialize. Tekan **Start** untuk memulai simulasi.")
            
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

    if cfg["start"]:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = True
        st.session_state.show_final_heatmap = False
        st.session_state.fig_hmap_cache = None
        st.session_state.last_heatmap_interval = -1  # Reset interval tracker
    elif cfg["stop"]:
        st.session_state.running = False
        st.session_state.show_final_heatmap = True
        st.rerun()
    elif cfg["reset_sim"]:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = False
        st.session_state.show_final_heatmap = False
        st.session_state.fig_hmap_cache = None
        st.session_state.last_heatmap_interval = -1  # Reset interval tracker
    
    # Initialize model hanya jika belum ada
    if st.session_state.model is None:
        st.session_state.model = _new_model()
        st.session_state.step_count = 0
        st.session_state.running = False
        st.session_state.show_final_heatmap = False
        st.session_state.fig_hmap_cache = None
        st.session_state.last_heatmap_interval = -1  # Initialize interval tracker


def _create_metric_placeholders():
    cols = st.columns(5)
    return [col.empty() for col in cols]