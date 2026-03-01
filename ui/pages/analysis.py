import streamlit as st
import subprocess
import sys
import pandas as pd
from pathlib import Path
from components.experiment_io import PROJECT_ROOT, ANALYSIS_DIR

FIGURES_DIR = PROJECT_ROOT / "figures"

st.header("Analysis")

c1, c2 = st.columns(2)

with c1:
    run_analysis = st.button("Run Analysis", type="primary")
with c2:
    run_viz = st.button("Create Static Visualizations")

if run_analysis:
    with st.spinner("Running analysis..."):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "analyze_results.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
    if result.returncode == 0:
        st.success("Analysis complete")
    else:
        st.error(f"Analysis failed (exit {result.returncode})")
    st.code(result.stdout + result.stderr, language="text")

if run_viz:
    with st.spinner("Creating visualizations..."):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "create_visualizations.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
    if result.returncode == 0:
        st.success("Visualizations created in figures/")
    else:
        st.error(f"Failed (exit {result.returncode})")
    st.code(result.stdout + result.stderr, language="text")

# -- Research questions report -------------------------------------------------

st.divider()

report_path = ANALYSIS_DIR / "research_questions_analysis.txt"
if report_path.exists():
    st.subheader("Research Questions Report")
    st.code(report_path.read_text(), language="text")

# -- Summary tables ------------------------------------------------------------

st.divider()

csv_files = {
    "Top Experiments": "top_experiments.csv",
    "By Disease": "results_by_disease.csv",
    "By Architecture": "results_by_architecture.csv",
    "By Category": "results_by_category.csv",
}

tabs = st.tabs(list(csv_files.keys()))
for tab, (label, filename) in zip(tabs, csv_files.items()):
    with tab:
        csv_path = ANALYSIS_DIR / filename
        if csv_path.exists():
            try:
                table_df = pd.read_csv(csv_path)
                st.dataframe(table_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
        else:
            st.info(f"Run analysis first to generate {filename}")

# -- Figures gallery -----------------------------------------------------------

st.divider()
st.subheader("Figures")

if FIGURES_DIR.exists():
    figures = sorted(FIGURES_DIR.glob("*.png"))
    if figures:
        cols_per_row = 2
        for i in range(0, len(figures), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(figures):
                    with col:
                        st.image(
                            str(figures[idx]),
                            caption=figures[idx].stem.replace("_", " ").title(),
                            use_container_width=True,
                        )
    else:
        st.info("No figures found. Click 'Create Static Visualizations' above.")
else:
    st.info("No figures/ directory found. Click 'Create Static Visualizations' above.")
