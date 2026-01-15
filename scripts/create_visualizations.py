#!/usr/bin/env python3
"""
Create visualizations for GNN experiment results

Generates figures for:
- GNN vs MLP comparison
- GNN vs CNN comparison
- Binary vs multi-class classification
- Edge weighting strategies
- Architecture comparison
- Summary heatmap (diseases × architectures)

Outputs saved to: figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    return pd.read_csv(ANALYSIS_DIR / "all_results.csv")


def plot_gnn_vs_mlp(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    diseases, gnn_acc, mlp_acc = [], [], []
    
    for disease in df['disease'].unique():
        disease_df = df[df['disease'] == disease]
        gnn = disease_df[disease_df['experiment'].str.contains('baseline') & 
                        ~disease_df['experiment'].str.contains('mlp|cnn')]
        mlp = disease_df[disease_df['experiment'].str.contains('mlp')]
        
        if not gnn.empty and not mlp.empty:
            diseases.append(disease.upper())
            gnn_acc.append(gnn['test_accuracy_mean'].values[0])
            mlp_acc.append(mlp['test_accuracy_mean'].values[0])
    
    x = range(len(diseases))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], gnn_acc, width, label='GNN', alpha=0.8)
    ax.bar([i + width/2 for i in x], mlp_acc, width, label='MLP', alpha=0.8)
    
    ax.set_xlabel('Disease')
    ax.set_ylabel('Accuracy')
    ax.set_title('Q1: GNN vs MLP - Do phylogenetic graphs help?')
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    ax.set_ylim(0.5, 0.7)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_gnn_vs_mlp.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/01_gnn_vs_mlp.png")


def plot_gnn_vs_cnn(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    diseases, gnn_acc, cnn_acc = [], [], []
    
    for disease in df['disease'].unique():
        disease_df = df[df['disease'] == disease]
        gnn = disease_df[disease_df['experiment'].str.contains('baseline') & 
                        ~disease_df['experiment'].str.contains('mlp|cnn')]
        cnn = disease_df[disease_df['experiment'].str.contains('cnn')]
        
        if not gnn.empty and not cnn.empty:
            diseases.append(disease.upper())
            gnn_acc.append(gnn['test_accuracy_mean'].values[0])
            cnn_acc.append(cnn['test_accuracy_mean'].values[0])
    
    x = range(len(diseases))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], gnn_acc, width, label='GNN (graphs)', alpha=0.8)
    ax.bar([i + width/2 for i in x], cnn_acc, width, label='CNN (fixed neighborhood)', alpha=0.8)
    
    ax.set_xlabel('Disease')
    ax.set_ylabel('Accuracy')
    ax.set_title('Q2: GNN vs CNN - Are graphs more flexible?')
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    ax.set_ylim(0.5, 0.7)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_gnn_vs_cnn.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/02_gnn_vs_cnn.png")


def plot_binary_vs_multiclass(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    diseases, binary_acc, age_acc = [], [], []
    
    for disease in df['disease'].unique():
        disease_df = df[df['disease'] == disease]
        binary = disease_df[disease_df['experiment'].str.contains('baseline') & 
                           ~disease_df['experiment'].str.contains('mlp|cnn|age')]
        age = disease_df[disease_df['experiment'].str.contains('age')]
        
        if not binary.empty and not age.empty:
            diseases.append(disease.upper())
            binary_acc.append(binary['test_accuracy_mean'].values[0])
            age_acc.append(age['test_accuracy_mean'].values[0])
    
    x = range(len(diseases))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], binary_acc, width, label='Binary (disease)', alpha=0.8)
    ax.bar([i + width/2 for i in x], age_acc, width, label='Multi-class (age)', alpha=0.8)
    
    ax.set_xlabel('Disease')
    ax.set_ylabel('Accuracy')
    ax.set_title('Q3: Binary vs Multi-class Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.3)
    ax.set_ylim(0.5, 0.7)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_binary_vs_multiclass.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/03_binary_vs_multiclass.png")


def plot_edge_weights(df):
    edge_df = df[df['category'] == 'Edge Weight'].sort_values('test_accuracy_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    experiments = [exp.replace('edge_', '') for exp in edge_df['experiment']]
    accuracies = edge_df['test_accuracy_mean']
    errors = edge_df['test_accuracy_std']
    
    ax.barh(experiments, accuracies, xerr=errors, alpha=0.8, capsize=5)
    ax.set_xlabel('Accuracy')
    ax.set_title('Q4: Edge Weighting Strategies Comparison')
    ax.axvline(x=0.6, color='r', linestyle='--', alpha=0.3)
    ax.set_xlim(0.55, 0.65)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_edge_weights.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/04_edge_weights.png")


def plot_architectures(df):
    arch_df = df[(df['category'] == 'Architecture') & (df['disease'] == 'ibd')].sort_values('test_accuracy_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    experiments = arch_df['experiment']
    accuracies = arch_df['test_accuracy_mean']
    errors = arch_df['test_accuracy_std']
    colors = ['C1' if 'mlp' in exp or 'cnn' in exp else 'C0' for exp in experiments]
    
    ax.barh(experiments, accuracies, xerr=errors, alpha=0.8, capsize=5, color=colors)
    ax.set_xlabel('Accuracy')
    ax.set_title('Q5: Architecture Comparison')
    ax.axvline(x=0.6, color='r', linestyle='--', alpha=0.3)
    ax.set_xlim(0.5, 0.7)
    
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='C0', label='GNN'), Patch(facecolor='C1', label='Non-graph')])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_architectures.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/05_architectures.png")


def plot_summary_heatmap(df):
    models = ['GCN', 'GINEConv', 'GAT', 'GraphSAGE', 'EdgeCentricRGCN', 'CNN', 'MLP']
    diseases = []
    data = []
    
    for disease in df['disease'].unique():
        row = []
        diseases.append(disease.upper())
        for model in models:
            disease_df = df[df['disease'] == disease]
            model_df = disease_df[disease_df['model_type'] == model]
            row.append(model_df['test_accuracy_mean'].values[0] if not model_df.empty else None)
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    heatmap_df = pd.DataFrame(data, index=diseases, columns=models)
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.6,
                vmin=0.5, vmax=0.7, ax=ax, cbar_kws={'label': 'Accuracy'})
    
    ax.set_title('Summary: All Diseases × All Architectures')
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Disease')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_summary_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/06_summary_heatmap.png")


def main():
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} experiments")
    print()
    
    print("Creating plots...")
    plot_gnn_vs_mlp(df)
    plot_gnn_vs_cnn(df)
    plot_binary_vs_multiclass(df)
    plot_edge_weights(df)
    plot_architectures(df)
    plot_summary_heatmap(df)
    
    print()
    print("=" * 80)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print()
    print("Generated figures:")
    for fig in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  - {fig}")


if __name__ == "__main__":
    main()
