"""Plotly chart builders for the results page.

Each function takes a DataFrame (from load_all_results) and returns a
plotly.graph_objects.Figure ready for st.plotly_chart().
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Optional


TEMPLATE = "plotly_dark"
DEFAULT_HEIGHT = 420


def accuracy_by_architecture(
    df: pd.DataFrame,
    disease_filter: Optional[str] = None,
) -> go.Figure:
    if disease_filter:
        df = df[df["disease"] == disease_filter]
    if df.empty or "test_accuracy_mean" not in df.columns:
        return _empty_figure("No data available")

    arch_df = (
        df.groupby("model_type")
        .agg(
            mean_acc=("test_accuracy_mean", "mean"),
            std_acc=("test_accuracy_mean", "std"),
            count=("experiment", "count"),
        )
        .reset_index()
        .sort_values("mean_acc", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=arch_df["model_type"],
        x=arch_df["mean_acc"],
        error_x=dict(type="data", array=arch_df["std_acc"], visible=True),
        orientation="h",
        text=arch_df["mean_acc"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        marker_color="#636EFA",
    ))
    title = "Accuracy by Architecture"
    if disease_filter:
        title += f" ({disease_filter.upper()})"
    fig.update_layout(
        title=title,
        xaxis_title="Mean Accuracy",
        yaxis_title="Architecture",
        template=TEMPLATE,
        height=DEFAULT_HEIGHT,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def gnn_vs_mlp(df: pd.DataFrame) -> go.Figure:
    if df.empty or "test_accuracy_mean" not in df.columns:
        return _empty_figure("No data available")

    diseases, gnn_acc, mlp_acc = [], [], []
    for disease in df["disease"].unique():
        ddf = df[df["disease"] == disease]
        gnn = ddf[
            ddf["experiment"].str.contains("baseline")
            & ~ddf["experiment"].str.contains("mlp|cnn")
        ]
        mlp = ddf[ddf["experiment"].str.contains("mlp")]
        if not gnn.empty and not mlp.empty:
            diseases.append(disease.upper())
            gnn_acc.append(gnn["test_accuracy_mean"].values[0])
            mlp_acc.append(mlp["test_accuracy_mean"].values[0])

    if not diseases:
        return _empty_figure("No GNN vs MLP comparisons available")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="GNN", x=diseases, y=gnn_acc, marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="MLP", x=diseases, y=mlp_acc, marker_color="#EF553B"))
    fig.update_layout(
        title="Q1: GNN vs MLP -- Do phylogenetic graphs help?",
        xaxis_title="Disease",
        yaxis_title="Accuracy",
        barmode="group",
        template=TEMPLATE,
        height=DEFAULT_HEIGHT,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.3)
    return fig


def gnn_vs_cnn(df: pd.DataFrame) -> go.Figure:
    if df.empty or "test_accuracy_mean" not in df.columns:
        return _empty_figure("No data available")

    diseases, gnn_acc, cnn_acc = [], [], []
    for disease in df["disease"].unique():
        ddf = df[df["disease"] == disease]
        gnn = ddf[
            ddf["experiment"].str.contains("baseline")
            & ~ddf["experiment"].str.contains("mlp|cnn")
        ]
        cnn = ddf[ddf["experiment"].str.contains("cnn")]
        if not gnn.empty and not cnn.empty:
            diseases.append(disease.upper())
            gnn_acc.append(gnn["test_accuracy_mean"].values[0])
            cnn_acc.append(cnn["test_accuracy_mean"].values[0])

    if not diseases:
        return _empty_figure("No GNN vs CNN comparisons available")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="GNN (graphs)", x=diseases, y=gnn_acc, marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="CNN (fixed)", x=diseases, y=cnn_acc, marker_color="#AB63FA"))
    fig.update_layout(
        title="Q2: GNN vs CNN -- Are graphs more flexible?",
        xaxis_title="Disease",
        yaxis_title="Accuracy",
        barmode="group",
        template=TEMPLATE,
        height=DEFAULT_HEIGHT,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.3)
    return fig


def edge_weight_comparison(df: pd.DataFrame) -> go.Figure:
    edge_df = df[df["category"] == "Edge Weight"].sort_values(
        "test_accuracy_mean", ascending=True,
    )
    if edge_df.empty:
        return _empty_figure("No edge weight experiments found")

    labels = [exp.replace("edge_", "") for exp in edge_df["experiment"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=edge_df["test_accuracy_mean"],
        error_x=dict(type="data", array=edge_df["test_accuracy_std"], visible=True),
        orientation="h",
        text=edge_df["test_accuracy_mean"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        marker_color="#00CC96",
    ))
    fig.update_layout(
        title="Q4: Edge Weighting Strategies",
        xaxis_title="Accuracy",
        yaxis_title="Strategy",
        template=TEMPLATE,
        height=DEFAULT_HEIGHT,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def summary_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "test_accuracy_mean" not in df.columns:
        return _empty_figure("No data available")

    models = ["GCN", "GINEConv", "GAT", "GraphSAGE", "EdgeCentricRGCN", "CNN", "MLP"]
    diseases = sorted(df["disease"].unique())

    z, text = [], []
    for disease in diseases:
        row, text_row = [], []
        for model in models:
            mdf = df[(df["disease"] == disease) & (df["model_type"] == model)]
            if not mdf.empty:
                val = mdf["test_accuracy_mean"].values[0]
                row.append(val)
                text_row.append(f"{val:.3f}")
            else:
                row.append(None)
                text_row.append("")
        z.append(row)
        text.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=models,
        y=[d.upper() for d in diseases],
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0.6,
        zmin=0.45,
        zmax=0.75,
        colorbar=dict(title="Accuracy"),
    ))
    fig.update_layout(
        title="Summary: All Diseases x All Architectures",
        xaxis_title="Architecture",
        yaxis_title="Disease",
        template=TEMPLATE,
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template=TEMPLATE,
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
