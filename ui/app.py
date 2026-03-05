#!/usr/bin/env python3
"""BiomeML Dashboard -- Streamlit entry point."""

import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="BiomeML",
    layout="wide",
    initial_sidebar_state="expanded",
)

dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True)
experiments = st.Page("pages/experiments.py", title="Experiments", icon=":material/science:")
results = st.Page("pages/results.py", title="Results", icon=":material/bar_chart:")
config = st.Page("pages/config_editor.py", title="Config", icon=":material/settings:")
analysis = st.Page("pages/analysis.py", title="Analysis", icon=":material/analytics:")

pg = st.navigation([dashboard, experiments, results, config, analysis])
pg.run()
