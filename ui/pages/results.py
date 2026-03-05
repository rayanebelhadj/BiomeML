import json
import streamlit as st
import pandas as pd
from components.experiment_io import load_all_results, EXPERIMENTS_DIR
from components.charts import (
    accuracy_by_architecture,
    gnn_vs_mlp,
    gnn_vs_cnn,
    edge_weight_comparison,
    summary_heatmap,
)

st.header("Results")

df = load_all_results()

if df.empty:
    st.info("No experiment results found. Run some experiments first.")
    st.stop()

# -- Sidebar filters -----------------------------------------------------------

with st.sidebar:
    st.subheader("Filters")
    all_diseases = sorted(df["disease"].unique())
    selected_diseases = st.multiselect("Disease", all_diseases, default=all_diseases)
    all_archs = sorted(df["model_type"].unique())
    selected_archs = st.multiselect("Architecture", all_archs, default=all_archs)
    all_categories = sorted(df["category"].unique())
    selected_cats = st.multiselect("Category", all_categories, default=all_categories)

filtered = df[
    df["disease"].isin(selected_diseases)
    & df["model_type"].isin(selected_archs)
    & df["category"].isin(selected_cats)
]

# -- Data table ----------------------------------------------------------------

st.subheader(f"All Experiments ({len(filtered)})")

display_cols = ["experiment", "disease", "model_type", "category", "num_runs"]
for col in ("test_accuracy_mean", "test_accuracy_std", "test_auc_mean"):
    if col in filtered.columns:
        display_cols.append(col)

show_df = filtered[display_cols].copy()
if "test_accuracy_mean" in show_df.columns:
    show_df = show_df.sort_values("test_accuracy_mean", ascending=False)

st.dataframe(
    show_df,
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        "test_accuracy_mean": st.column_config.NumberColumn("Accuracy (mean)", format="%.4f"),
        "test_accuracy_std": st.column_config.NumberColumn("Accuracy (std)", format="%.4f"),
        "test_auc_mean": st.column_config.NumberColumn("AUC (mean)", format="%.4f"),
    },
)

st.download_button(
    "Download filtered results as CSV",
    data=show_df.to_csv(index=False),
    file_name="biomeml_results.csv",
    mime="text/csv",
)

# -- Experiment detail ---------------------------------------------------------

st.divider()
st.subheader("Experiment Detail")

exp_names = show_df["experiment"].tolist() if not show_df.empty else []
selected_exp = st.selectbox("Select an experiment", exp_names, key="detail_exp")

if selected_exp:
    agg_path = EXPERIMENTS_DIR / selected_exp / "aggregated_results.json"
    if agg_path.exists():
        with open(agg_path) as f:
            agg_data = json.load(f)

        metrics = agg_data.get("metrics", {})
        config = agg_data.get("config", {})

        col_m, col_c = st.columns(2)
        with col_m:
            st.caption("Metrics")
            metric_rows = []
            for k, v in metrics.items():
                if isinstance(v, dict) and "mean" in v:
                    metric_rows.append({
                        "Metric": k,
                        "Mean": v["mean"],
                        "Std": v.get("std"),
                        "CI low": v.get("ci_lower"),
                        "CI high": v.get("ci_upper"),
                        "N": v.get("n"),
                    })
            if metric_rows:
                st.dataframe(
                    pd.DataFrame(metric_rows),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Mean": st.column_config.NumberColumn(format="%.4f"),
                        "Std": st.column_config.NumberColumn(format="%.4f"),
                        "CI low": st.column_config.NumberColumn(format="%.4f"),
                        "CI high": st.column_config.NumberColumn(format="%.4f"),
                    },
                )

            values = metrics.get("test_accuracy", {}).get("values", [])
            if values:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=values, nbinsx=20, marker_color="#636EFA"))
                fig.update_layout(
                    title="Per-run accuracy distribution",
                    xaxis_title="Accuracy",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_c:
            st.caption("Experiment config overrides")
            if config:
                st.code(json.dumps(config, indent=2), language="json")
            else:
                st.info("No config overrides recorded")
    else:
        st.info(f"No aggregated results found for {selected_exp}")

# -- Charts --------------------------------------------------------------------

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "By Architecture", "GNN vs MLP", "GNN vs CNN", "Edge Weights", "Heatmap",
])

with tab1:
    disease_opt = st.selectbox(
        "Filter by disease",
        [None] + all_diseases,
        format_func=lambda x: "All" if x is None else x.upper(),
        key="arch_disease",
    )
    st.plotly_chart(accuracy_by_architecture(filtered, disease_opt), use_container_width=True)

with tab2:
    st.plotly_chart(gnn_vs_mlp(df), use_container_width=True)

with tab3:
    st.plotly_chart(gnn_vs_cnn(df), use_container_width=True)

with tab4:
    st.plotly_chart(edge_weight_comparison(df), use_container_width=True)

with tab5:
    st.plotly_chart(summary_heatmap(df), use_container_width=True)
