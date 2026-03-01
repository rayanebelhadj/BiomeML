import streamlit as st
import subprocess
import threading
import sys
from components.experiment_io import (
    load_experiments,
    get_experiment_status,
    get_diseases_for_dataset,
    get_experiments_for_disease,
    get_experiments_for_dataset,
    parse_category,
    DATASETS,
    PROJECT_ROOT,
)

# Maps UI radio labels to the categories returned by parse_category()
TYPE_CATEGORIES = {
    "Architecture": {"Architecture"},
    "Control": {"Control: Random", "Control: Complete", "Control: Shuffled"},
    "Metadata": {"Metadata", "Metadata Only", "GNN + Metadata"},
    "Multi-class": {"Multi-class: Age", "Multi-class: Subtypes"},
    "Edge Weights": {"Edge Weight"},
    "k-NN / Distance": {"k-NN Density", "Distance Matrix"},
    "Hyperparameter": {"Hyperparameter"},
}

st.header("Run Experiments")

experiments = load_experiments()

if not experiments:
    st.warning("No experiments found in `experiments.yaml`.")
    st.stop()

# -- Filters -------------------------------------------------------------------

c1, c2 = st.columns(2)
with c1:
    dataset = st.selectbox(
        "Dataset",
        list(DATASETS.keys()),
        format_func=lambda d: f"{d.upper()} -- {DATASETS[d]}",
    )
with c2:
    known = get_diseases_for_dataset(dataset)
    dataset_exps = get_experiments_for_dataset(dataset, experiments)
    extra = sorted(
        {experiments[e].get("disease", "") for e in dataset_exps}
        - set(known)
        - {""},
    )
    diseases = known + extra
    selected_diseases = st.multiselect("Diseases", diseases, default=diseases[:1])

exp_type = st.radio(
    "Experiment type",
    ["All"] + list(TYPE_CATEGORIES.keys()),
    horizontal=True,
)

# -- Build experiment list -----------------------------------------------------

matched = []
for disease in selected_diseases:
    disease_exps = get_experiments_for_disease(disease, experiments)
    if exp_type == "All":
        matched.extend(disease_exps)
    else:
        allowed = TYPE_CATEGORIES[exp_type]
        matched.extend(e for e in disease_exps if parse_category(e) in allowed)

matched = list(dict.fromkeys(matched))

if matched:
    rows = []
    for exp in matched:
        runs, status = get_experiment_status(exp)
        desc = experiments.get(exp, {}).get("description", "")
        cat = parse_category(exp)
        rows.append({
            "experiment": exp,
            "category": cat,
            "description": desc,
            "runs": runs,
            "status": status,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=250)
else:
    st.info("No experiments match the current filters.")

# -- Launch controls -----------------------------------------------------------

st.divider()
c1, c2, _ = st.columns([1, 1, 2])
with c1:
    num_runs = st.number_input("Runs per experiment", min_value=1, max_value=200, value=50)
with c2:
    seed = st.number_input("Base seed", min_value=0, value=42)

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "process" not in st.session_state:
    st.session_state.process = None


def _stream_output(proc):
    """Background thread that reads subprocess stdout into session state."""
    for line in iter(proc.stdout.readline, ""):
        st.session_state.log_lines.append(line.rstrip("\n"))
    proc.stdout.close()
    proc.wait()


col_launch, col_stop, _ = st.columns([1, 1, 2])

with col_launch:
    launch_disabled = (not matched) or (st.session_state.process is not None)
    if st.button("Launch", type="primary", disabled=launch_disabled):
        st.session_state.log_lines = [
            f"Starting {len(matched)} experiment(s) with {num_runs} runs each..."
        ]
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_experiments.py"),
            "--experiments", *matched,
            "--num-runs", str(num_runs),
            "--seed", str(seed),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        st.session_state.process = proc
        t = threading.Thread(target=_stream_output, args=(proc,), daemon=True)
        t.start()
        st.rerun()

with col_stop:
    if st.button("Stop", disabled=st.session_state.process is None):
        proc = st.session_state.process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            st.session_state.log_lines.append("--- Process terminated by user ---")
        st.session_state.process = None
        st.rerun()

# -- Live log ------------------------------------------------------------------


@st.fragment(run_every="2s")
def show_log():
    lines = st.session_state.get("log_lines", [])
    proc = st.session_state.get("process")

    if proc is not None and proc.poll() is not None:
        st.session_state.log_lines.append(
            f"--- Process exited with code {proc.returncode} ---"
        )
        st.session_state.process = None

    if lines:
        running = proc is not None and proc.poll() is None
        if running:
            st.caption("Running...")
        elif lines[-1].startswith("--- Process"):
            st.caption("Complete")
        st.code("\n".join(lines[-300:]), language="text")


show_log()
