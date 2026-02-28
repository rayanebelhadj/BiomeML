import streamlit as st
import pandas as pd
from components.experiment_io import (
    load_experiments,
    get_experiment_status,
    load_all_results,
)

st.header("Dashboard")

TARGET_RUNS = 50


@st.fragment(run_every="30s")
def dashboard_content():
    experiments = load_experiments()

    if not experiments:
        st.warning("No experiments found in `experiments.yaml`.")
        st.stop()

    # -- Summary metrics ------------------------------------------------------

    statuses = []
    for name in experiments:
        runs, status_text = get_experiment_status(name)
        statuses.append({"experiment": name, "runs": runs, "status": status_text})

    total = len(statuses)
    completed = sum(1 for s in statuses if s["runs"] >= TARGET_RUNS)
    in_progress = sum(1 for s in statuses if 0 < s["runs"] < TARGET_RUNS)
    not_started = sum(1 for s in statuses if s["runs"] == 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Experiments", total)
    c2.metric("Completed", completed)
    c3.metric("In Progress", in_progress)
    c4.metric("Not Started", not_started)

    st.progress(completed / total if total else 0, text=f"{completed}/{total} experiments complete")

    # -- Best-result banner ---------------------------------------------------

    df = load_all_results()

    if not df.empty and "test_accuracy_mean" in df.columns:
        st.divider()
        best_idx = df["test_accuracy_mean"].idxmax()
        best = df.loc[best_idx]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Accuracy", f"{best['test_accuracy_mean']:.1%}")
        c2.metric("Best Experiment", best["experiment"])
        c3.metric("Total Runs", f"{int(df['num_runs'].sum()):,}")

    # -- Per-disease summary --------------------------------------------------

        st.divider()
        st.subheader("Per-Disease Summary")
        disease_summary = (
            df.groupby("disease")
            .agg(
                experiments=("experiment", "count"),
                best_accuracy=("test_accuracy_mean", "max"),
                avg_accuracy=("test_accuracy_mean", "mean"),
            )
            .reset_index()
            .sort_values("best_accuracy", ascending=False)
        )
        disease_summary.columns = ["Disease", "Experiments", "Best Accuracy", "Avg Accuracy"]
        st.dataframe(
            disease_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Best Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Avg Accuracy": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    # -- Full status table with progress bars ---------------------------------

    st.divider()
    st.subheader("Experiment Status")

    status_df = pd.DataFrame(statuses)
    status_df["progress"] = status_df["runs"].apply(lambda r: min(r / TARGET_RUNS, 1.0))
    status_df = status_df[["experiment", "runs", "progress", "status"]]
    status_df.columns = ["Experiment", "Runs", "Progress", "Status"]

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
        height=min(len(status_df) * 35 + 38, 600),
        column_config={
            "Progress": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.0%%",
            ),
        },
    )


dashboard_content()
