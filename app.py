"""
app.py
======
Entry point for the waiting room ABM simulation.

Run with:
    streamlit run app.py
"""

import streamlit as st
from "@vercel/speed-insights/next" import SpeedInsights  # type: ignore

from ui.state import init_session
from ui.panel_design import build_sidebar_design, panel_design
from ui.panel_simulate import build_sidebar_sim, panel_simulate


def main() -> None:
    st.set_page_config(
        page_title="ABM Waiting Room - Random Movement",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .block-container { padding: 1rem 1.8rem; }
        h1, h2, h3 { font-family: 'Space Mono', monospace; }

        .stMetric [data-testid="stMetricValue"] {
            color: #f5a623;
            font-size: 1.3rem;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background: #1a1a2e;
            border-radius: 6px 6px 0 0;
            padding: 6px 16px;
            color: #888;
        }
        .stTabs [aria-selected="true"] {
            background: #0f3460 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "# ABM Waiting Room - Random Movement"
        "<br><small style='color:#666;font-family:monospace'>"
        "v3 - Randomized motion - 1 m = 10 px"
        "</small>",
        unsafe_allow_html=True,
    )

    init_session()

    tab_design, tab_simulate = st.tabs([
        "Design Room",
        "Run Simulation",
    ])

    with tab_design:
        build_sidebar_design()
        panel_design()

    with tab_simulate:
        cfg = build_sidebar_sim()
        panel_simulate(cfg)


if __name__ == "__main__":
    main()
