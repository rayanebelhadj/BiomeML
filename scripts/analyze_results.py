#!/usr/bin/env python3
"""
Comprehensive Analysis of GNN Experiments

Analyzes all experiments to answer research questions:
- Q1: Do evolutionary relationships (graphs) help?
- Q2: Are graphs more flexible than CNNs?
- Q3: Binary vs multi-class classification
- Q4: Which edge weighting strategy works best?
- Q5: Which GNN architecture performs best?

Outputs:
- analysis/all_results.csv
- analysis/top_experiments.csv
- analysis/results_by_disease.csv
- analysis/results_by_architecture.csv
- analysis/results_by_category.csv
- analysis/research_questions_analysis.txt
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_results() -> pd.DataFrame:
    results = []
    
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
            
        agg_file = exp_dir / "aggregated_results.json"
        if not agg_file.exists():
            continue
            
        try:
            with open(agg_file) as f:
                data = json.load(f)
            
            metrics = data.get('metrics', {})
            
            result = {
                'experiment': exp_dir.name,
                'num_runs': data.get('num_runs', len(metrics.get('test_accuracy', {}).get('values', []))),
            }
            
            for metric_name in ['test_accuracy', 'test_auc', 'test_f1', 'test_precision', 'test_recall']:
                if metric_name in metrics:
                    result[f'{metric_name}_mean'] = metrics[metric_name].get('mean')
                    result[f'{metric_name}_std'] = metrics[metric_name].get('std')
                    result[f'{metric_name}_ci_lower'] = metrics[metric_name].get('ci_lower')
                    result[f'{metric_name}_ci_upper'] = metrics[metric_name].get('ci_upper')
            
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {exp_dir.name}: {e}")
            continue
    
    df = pd.DataFrame(results)
    df['disease'] = df['experiment'].apply(parse_disease)
    df['model_type'] = df['experiment'].apply(parse_model_type)
    df['category'] = df['experiment'].apply(parse_category)
    
    return df


def parse_disease(exp_name: str) -> str:
    diseases = ['ibd', 'diabetes', 'cancer', 'autoimmune', 'depression', 
                'mental_illness', 'ptsd', 'arthritis', 'asthma', 'stomach_bowel']
    
    for disease in diseases:
        if exp_name.startswith(disease):
            return disease
    
    if exp_name in ['baseline', 'gnn_gcn', 'gnn_gineconv', 'gnn_gat', 
                     'gnn_graphsage', 'gnn_edgecentric', 'cnn_baseline', 
                     'no_graph_mlp', 'random_edges', 'complete_graph', 
                     'shuffled_labels']:
        return 'ibd'
    
    if exp_name.startswith(('edge_', 'k_', 'distance_', 'hp_', 'meta_')):
        return 'ibd'
    
    return 'ibd'


def parse_model_type(exp_name: str) -> str:
    if 'cnn' in exp_name:
        return 'CNN'
    elif 'mlp' in exp_name:
        return 'MLP'
    elif 'gcn' in exp_name and 'gineconv' not in exp_name:
        return 'GCN'
    elif 'gineconv' in exp_name or 'baseline' in exp_name:
        return 'GINEConv'
    elif 'gat' in exp_name:
        return 'GAT'
    elif 'graphsage' in exp_name:
        return 'GraphSAGE'
    elif 'edgecentric' in exp_name:
        return 'EdgeCentricRGCN'
    else:
        return 'Other'


def parse_category(exp_name: str) -> str:
    if exp_name.startswith('k_'):
        return 'k-NN Density'
    elif exp_name.startswith('distance_'):
        return 'Distance Matrix'
    elif exp_name.startswith('edge_'):
        return 'Edge Weight'
    elif exp_name.startswith('hp_'):
        return 'Hyperparameter'
    elif exp_name.startswith('meta_'):
        return 'Metadata'
    elif 'metadata_only' in exp_name:
        return 'Metadata Only'
    elif 'gnn_meta' in exp_name:
        return 'GNN + Metadata'
    elif 'random_edges' in exp_name:
        return 'Control: Random'
    elif 'complete_graph' in exp_name:
        return 'Control: Complete'
    elif 'shuffled_labels' in exp_name:
        return 'Control: Shuffled'
    elif 'age' in exp_name and 'meta_age' not in exp_name:
        return 'Multi-class: Age'
    elif 'subtypes' in exp_name:
        return 'Multi-class: Subtypes'
    elif exp_name.endswith('_baseline'):
        return 'Architecture'
    elif exp_name.startswith('gnn_'):
        return 'Architecture'
    elif exp_name in ['baseline', 'cnn_baseline', 'no_graph_mlp']:
        return 'Architecture'
    else:
        return 'Other'


def compare_two_experiments(df: pd.DataFrame, exp1: str, exp2: str, 
                           metric: str = 'test_accuracy_mean') -> Dict:
    exp1_rows = df[df['experiment'] == exp1]
    exp2_rows = df[df['experiment'] == exp2]
    
    if len(exp1_rows) == 0:
        raise ValueError(f"Experiment '{exp1}' not found")
    if len(exp2_rows) == 0:
        raise ValueError(f"Experiment '{exp2}' not found")
    
    row1, row2 = exp1_rows.iloc[0], exp2_rows.iloc[0]
    
    val1 = row1[metric]
    val2 = row2[metric]
    std1 = row1[metric.replace('_mean', '_std')]
    std2 = row2[metric.replace('_mean', '_std')]
    
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    cohens_d = (val1 - val2) / pooled_std if pooled_std > 0 else 0
    
    n1, n2 = row1['num_runs'], row2['num_runs']
    se = np.sqrt(std1**2/n1 + std2**2/n2)
    t_stat = (val1 - val2) / se if se > 0 else 0
    df_t = n1 + n2 - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_t))
    
    return {
        'exp1': exp1, 'exp2': exp2,
        'mean_diff': val1 - val2,
        'cohens_d': cohens_d,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }


def analyze_research_questions(df: pd.DataFrame) -> str:
    report = []
    
    report.append("=" * 80)
    report.append("ANALYSIS OF RESEARCH QUESTIONS")
    report.append("=" * 80)
    report.append("")
    
    report.append("Q1: Do evolutionary relationships (graphs) help classify diseases?")
    report.append("-" * 80)
    
    for disease in df['disease'].unique():
        disease_df = df[df['disease'] == disease]
        gnn_exp = disease_df[disease_df['experiment'].str.contains('baseline') & 
                            ~disease_df['experiment'].str.contains('mlp|cnn')].head(1)
        mlp_exp = disease_df[disease_df['experiment'].str.contains('mlp')].head(1)
        
        if not gnn_exp.empty and not mlp_exp.empty:
            gnn_acc = gnn_exp['test_accuracy_mean'].values[0]
            mlp_acc = mlp_exp['test_accuracy_mean'].values[0]
            diff = gnn_acc - mlp_acc
            comp = compare_two_experiments(df, gnn_exp['experiment'].values[0], 
                                          mlp_exp['experiment'].values[0])
            sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else "ns"
            report.append(f"  {disease.upper():20s}: GNN={gnn_acc:.3f} vs MLP={mlp_acc:.3f} | "
                         f"Δ={diff:+.3f} (d={comp['cohens_d']:.2f}) {sig}")
    
    report.append("")
    report.append("Q2: Are graphs more flexible than CNNs?")
    report.append("-" * 80)
    
    for disease in df['disease'].unique():
        disease_df = df[df['disease'] == disease]
        gnn_exp = disease_df[disease_df['experiment'].str.contains('baseline') & 
                            ~disease_df['experiment'].str.contains('mlp|cnn')].head(1)
        cnn_exp = disease_df[disease_df['experiment'].str.contains('cnn')].head(1)
        
        if not gnn_exp.empty and not cnn_exp.empty:
            gnn_acc = gnn_exp['test_accuracy_mean'].values[0]
            cnn_acc = cnn_exp['test_accuracy_mean'].values[0]
            diff = gnn_acc - cnn_acc
            comp = compare_two_experiments(df, gnn_exp['experiment'].values[0], 
                                          cnn_exp['experiment'].values[0])
            sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else "ns"
            report.append(f"  {disease.upper():20s}: GNN={gnn_acc:.3f} vs CNN={cnn_acc:.3f} | "
                         f"Δ={diff:+.3f} (d={comp['cohens_d']:.2f}) {sig}")
    
    report.append("")
    report.append("Q3: Is binary classification similar to multi-class?")
    report.append("-" * 80)
    
    for disease in df['disease'].unique():
        disease_df = df[df['disease'] == disease]
        binary_exp = disease_df[disease_df['experiment'].str.contains('baseline') & 
                               ~disease_df['experiment'].str.contains('mlp|cnn|age')].head(1)
        age_exp = disease_df[disease_df['experiment'].str.contains('age')].head(1)
        
        if not binary_exp.empty and not age_exp.empty:
            binary_acc = binary_exp['test_accuracy_mean'].values[0]
            age_acc = age_exp['test_accuracy_mean'].values[0]
            report.append(f"  {disease.upper():20s}: Binary={binary_acc:.3f} vs Age={age_acc:.3f} | Δ={binary_acc - age_acc:+.3f}")
    
    report.append("")
    report.append("Q4: Which edge weighting strategy works best?")
    report.append("-" * 80)
    
    edge_exps = df[df['category'] == 'Edge Weight'].sort_values('test_accuracy_mean', ascending=False)
    for idx, row in edge_exps.head(5).iterrows():
        report.append(f"  {row['experiment']:30s}: {row['test_accuracy_mean']:.3f} ± {row['test_accuracy_std']:.3f}")
    
    report.append("")
    report.append("Q5: Which GNN architecture performs best?")
    report.append("-" * 80)
    
    arch_exps = df[(df['category'] == 'Architecture') & (df['disease'] == 'ibd')].sort_values('test_accuracy_mean', ascending=False)
    for idx, row in arch_exps.iterrows():
        report.append(f"  {row['experiment']:30s}: {row['test_accuracy_mean']:.3f} ± {row['test_accuracy_std']:.3f}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def generate_summary_tables(df: pd.DataFrame):
    top_exp = df.nlargest(20, 'test_accuracy_mean')[['experiment', 'disease', 'model_type', 
                                                       'test_accuracy_mean', 'test_accuracy_std']]
    top_exp.to_csv(OUTPUT_DIR / "top_experiments.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/top_experiments.csv")
    
    by_disease = df.groupby('disease').agg({
        'test_accuracy_mean': ['mean', 'std', 'max'],
        'experiment': 'count'
    }).round(3)
    by_disease.to_csv(OUTPUT_DIR / "results_by_disease.csv")
    print(f"Saved: {OUTPUT_DIR}/results_by_disease.csv")
    
    by_arch = df.groupby('model_type').agg({
        'test_accuracy_mean': ['mean', 'std', 'max'],
        'experiment': 'count'
    }).round(3)
    by_arch.to_csv(OUTPUT_DIR / "results_by_architecture.csv")
    print(f"Saved: {OUTPUT_DIR}/results_by_architecture.csv")
    
    by_category = df.groupby('category').agg({
        'test_accuracy_mean': ['mean', 'std', 'max'],
        'experiment': 'count'
    }).round(3)
    by_category.to_csv(OUTPUT_DIR / "results_by_category.csv")
    print(f"Saved: {OUTPUT_DIR}/results_by_category.csv")


def main():
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS OF GNN EXPERIMENTS")
    print("=" * 80)
    print()
    
    print("Loading results...")
    df = load_all_results()
    print(f"Loaded {len(df)} experiments with {df['num_runs'].sum():.0f} total runs")
    print()
    
    df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/all_results.csv")
    print()
    
    print("Generating summary tables...")
    generate_summary_tables(df)
    print()
    
    print("Analyzing research questions...")
    report = analyze_research_questions(df)
    print(report)
    
    with open(OUTPUT_DIR / "research_questions_analysis.txt", 'w') as f:
        f.write(report)
    print()
    print(f"Saved: {OUTPUT_DIR}/research_questions_analysis.txt")
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
