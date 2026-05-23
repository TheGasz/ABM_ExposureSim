"""
viz/__init__.py
===============
Public visualization API.
"""

from viz.editor_plot import build_editor_figure
from viz.sim_plots import plot_sim_room

__all__ = [
    "build_editor_figure",
    "plot_sim_room",
]
