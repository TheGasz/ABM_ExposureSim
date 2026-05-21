"""
app.py
======
Entry point aplikasi Streamlit ABM Ruang Tunggu.

Jalankan dengan:
    streamlit run app.py

Dependensi:
    pip install "mesa>=3.0" streamlit plotly matplotlib seaborn numpy

─────────────────────────────────────────────────────────────────────────────
STRUKTUR PROYEK
─────────────────────────────────────────────────────────────────────────────

abm_iklan/
│
├── app.py                  ← Entry point (file ini) — hanya orkestrasi
│
├── constants.py            ← Konstanta global & alias tipe
│
├── abm/                    ← Paket logika ABM (Mesa)
│   ├── __init__.py
│   ├── agents.py           ← ObstacleAgent, ChairAgent, CustomerAgent
│   └── model.py            ← WaitingRoomModel (Mesa Model)
│
├── viz/                    ← Paket visualisasi (Plotly + Matplotlib)
│   ├── __init__.py
│   ├── editor_plot.py      ← Figure Plotly untuk editor denah interaktif
│   └── sim_plots.py        ← Plot simulasi: kondisi ruangan + dual heatmap
│
└── ui/                     ← Paket Streamlit UI
    ├── __init__.py
    ├── state.py            ← Session state + konversi grid ↔ layout
    ├── panel_design.py     ← Tab "Design Ruangan" (editor grid)
    └── panel_simulate.py   ← Tab "Jalankan Simulasi" (ABM + visualisasi)

─────────────────────────────────────────────────────────────────────────────
ALUR DATA
─────────────────────────────────────────────────────────────────────────────

  Editor Plotly (panel_design)
      → grid_state (np.ndarray di session_state)
          → grid_state_to_layout() [ui/state.py]
              → layout dict {(col,row): tipe}
                  → WaitingRoomModel.__init__() [abm/model.py]
                      → grid.place_agent(ChairAgent | ObstacleAgent)
                          → CustomerAgent._trace_ray() per step
                              → floor_heatmap / obstacle_heatmap (akumulasi)
                                  → plot_dual_heatmap() [viz/sim_plots.py]
                                      → Streamlit placeholder.pyplot()

─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st

from ui.state import init_session
from ui.panel_design import build_sidebar_design, panel_design
from ui.panel_simulate import build_sidebar_sim, panel_simulate


def main() -> None:
    """
    Fungsi utama aplikasi.

    Tanggung jawab:
      1. Konfigurasi halaman Streamlit (set_page_config, CSS).
      2. Header aplikasi.
      3. Inisialisasi session state.
      4. Render dua tab utama:
           Tab 1 → Design Ruangan  (sidebar design + panel_design)
           Tab 2 → Jalankan Simulasi (sidebar sim + panel_simulate)

    Setiap tab memanggil sidebar-nya masing-masing secara kondisional —
    Streamlit akan menampilkan sidebar sesuai tab yang aktif.
    """
    # ── Konfigurasi halaman ───────────────────────────────────────────────────
    st.set_page_config(
        page_title            = "ABM Ruang Tunggu — Optimasi Iklan",
        page_icon             = "📊",
        layout                = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Global CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"]    { font-family: 'DM Sans', sans-serif; }
    .block-container               { padding: 1rem 1.8rem; }
    h1, h2, h3                     { font-family: 'Space Mono', monospace; }

    .stMetric [data-testid="stMetricValue"] {
        color: #f5a623;
        font-size: 1.3rem;
    }
    .stTabs [data-baseweb="tab-list"]  { gap: 6px; }
    .stTabs [data-baseweb="tab"]       {
        background    : #1a1a2e;
        border-radius : 6px 6px 0 0;
        padding       : 6px 16px;
        color         : #888;
    }
    .stTabs [aria-selected="true"]     {
        background : #0f3460 !important;
        color      : white   !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "# 📊 ABM Ruang Tunggu — Optimasi Penempatan Iklan"
        "<br><small style='color:#666;font-family:monospace'>"
        "v2 · Mesa 3.x · Plotly Editor · Dual Exposure Heatmap"
        "</small>",
        unsafe_allow_html=True,
    )

    # ── Inisialisasi session state ────────────────────────────────────────────
    init_session()

    # ── Tab navigasi utama ────────────────────────────────────────────────────
    tab_design, tab_simulate = st.tabs([
        "🏗️  Design Ruangan",
        "▶️  Jalankan Simulasi",
    ])

    with tab_design:
        build_sidebar_design()   # Sidebar ukuran grid
        panel_design()           # Editor grid interaktif

    with tab_simulate:
        cfg = build_sidebar_sim()   # Sidebar parameter simulasi
        panel_simulate(cfg)         # Loop ABM + visualisasi


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()