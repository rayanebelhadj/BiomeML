#!/usr/bin/env python3
"""
Interactive experiment runner for BiomeML project.

Provides a menu-driven interface to:
- Run experiments by disease or type
- View experiment status
- View results and comparisons
- Run analysis and create visualizations

Usage:
    python scripts/run_interactive.py
"""

import os
import sys
import json
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_YAML = PROJECT_ROOT / "experiments.yaml"

DISEASES = [
    "ibd", "diabetes", "cancer", "autoimmune", "depression",
    "mental_illness", "ptsd", "arthritis", "asthma", "stomach_bowel"
]

ARCHITECTURES = {
    "baseline": "GINEConv (baseline)",
    "gcn": "GCN",
    "gat": "GAT", 
    "graphsage": "GraphSAGE",
    "edgecentric": "EdgeCentricRGCN",
    "mlp": "MLP (no graph)",
    "cnn": "CNN"
}

CATEGORIES = {
    "arch": ["baseline", "gcn", "gat", "graphsage", "edgecentric", "mlp", "cnn"],
    "control": ["random_edges", "complete_graph", "shuffled_labels"],
    "metadata": ["gnn_meta", "metadata_only"],
    "multiclass": ["age"]
}


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def load_experiments():
    try:
        import yaml
        with open(EXPERIMENTS_YAML) as f:
            return yaml.safe_load(f)
    except:
        return {}


def get_experiment_status(exp_name):
    exp_dir = EXPERIMENTS_DIR / exp_name
    if not exp_dir.exists():
        return 0, "not started"
    
    runs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    completed = sum(1 for r in runs if (r / "evaluation_results.json").exists())
    
    agg_file = exp_dir / "aggregated_results.json"
    if agg_file.exists():
        try:
            with open(agg_file) as f:
                data = json.load(f)
                acc = data.get("test_accuracy", {}).get("mean", 0)
                return completed, f"{completed} runs, {acc:.1%} acc"
        except:
            pass
    
    return completed, f"{completed} runs"


def get_experiments_for_disease(disease, experiments):
    results = []
    for name, config in experiments.items():
        if isinstance(config, dict) and config.get("disease") == disease:
            results.append(name)
    return sorted(results)


def show_header():
    print("\n" + "=" * 50)
    print("       BiomeML Experiment Runner")
    print("=" * 50)


def show_main_menu():
    show_header()
    print("\n[1] Run experiments")
    print("[2] View experiment status")
    print("[3] View results")
    print("[4] Run analysis & visualizations")
    print("[5] Exit")
    print()
    return input("Select option [1-5]: ").strip()


def parse_selection(choice, max_val):
    indices = set()
    
    for part in choice.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = part.split('-')
                start, end = int(start), int(end)
                for i in range(start, end + 1):
                    if 1 <= i <= max_val:
                        indices.add(i - 1)
            except:
                pass
        else:
            try:
                i = int(part)
                if 1 <= i <= max_val:
                    indices.add(i - 1)
            except:
                pass
    
    return sorted(indices)


def select_from_list(items, prompt, allow_all=True, allow_back=True):
    print(f"\n{prompt}")
    print("(Enter numbers separated by commas, e.g. '1,3,5' or ranges '1-5')\n")
    
    for i, item in enumerate(items, 1):
        print(f"  [{i:2}] {item}")
    
    if allow_all:
        print(f"  [ 0] All")
    if allow_back:
        print(f"  [ b] Back")
    
    print()
    choice = input("Select: ").strip().lower()
    
    if choice == 'b' and allow_back:
        return None
    if choice == '0' and allow_all:
        return items
    
    indices = parse_selection(choice, len(items))
    
    if indices:
        return [items[i] for i in indices]
    
    print("Invalid selection")
    return []


def run_experiments_menu(experiments):
    while True:
        clear_screen()
        show_header()
        print("\n--- Run Experiments ---\n")
        
        selected_diseases = select_from_list(DISEASES, "Select disease:", allow_all=True)
        
        if selected_diseases is None:
            return
        if not selected_diseases:
            continue
        
        clear_screen()
        show_header()
        print("\n--- Select Experiment Type ---")
        print("(Enter numbers separated by commas, e.g. '1,2' or ranges '1-3')\n")
        
        print("  [ 1] Architecture comparison (GCN, GAT, etc.)")
        print("  [ 2] Control experiments (random, complete, shuffled)")
        print("  [ 3] Metadata experiments")
        print("  [ 4] Multi-class (age prediction)")
        print("  [ 0] All experiments for selected disease(s)")
        print("  [ b] Back")
        print()
        
        type_choice = input("Select: ").strip().lower()
        
        if type_choice == 'b':
            continue
        
        type_indices = parse_selection(type_choice, 4) if type_choice != '0' else []
        type_map = {0: "arch", 1: "control", 2: "metadata", 3: "multiclass"}
        selected_types = [type_map[i] for i in type_indices] if type_indices else []
        
        exp_to_run = []
        
        for disease in selected_diseases:
            disease_exps = get_experiments_for_disease(disease, experiments)
            
            if type_choice == '0' or not selected_types:
                if type_choice == '0':
                    exp_to_run.extend(disease_exps)
            else:
                if "arch" in selected_types:
                    for suffix in CATEGORIES["arch"]:
                        name = f"{disease}_{suffix}"
                        if name in experiments:
                            exp_to_run.append(name)
                        if disease == "ibd" and suffix == "baseline":
                            if "baseline" in experiments:
                                exp_to_run.append("baseline")
                
                if "control" in selected_types:
                    for suffix in CATEGORIES["control"]:
                        name = f"{disease}_{suffix}"
                        if name in experiments:
                            exp_to_run.append(name)
                
                if "metadata" in selected_types:
                    for suffix in CATEGORIES["metadata"]:
                        name = f"{disease}_{suffix}"
                        if name in experiments:
                            exp_to_run.append(name)
                
                if "multiclass" in selected_types:
                    name = f"{disease}_age"
                    if name in experiments:
                        exp_to_run.append(name)
        
        if not exp_to_run:
            print("\nNo matching experiments found.")
            input("Press Enter to continue...")
            continue
        
        exp_to_run = list(dict.fromkeys(exp_to_run))
        
        clear_screen()
        show_header()
        print(f"\n--- Selected {len(exp_to_run)} experiments ---\n")
        
        for exp in exp_to_run[:20]:
            runs, status = get_experiment_status(exp)
            print(f"  {exp:<35} [{status}]")
        
        if len(exp_to_run) > 20:
            print(f"  ... and {len(exp_to_run) - 20} more")
        
        print()
        
        num_runs = input("Number of runs per experiment [50]: ").strip()
        num_runs = int(num_runs) if num_runs.isdigit() else 50
        
        print(f"\nWill run {len(exp_to_run)} experiments with {num_runs} runs each.")
        confirm = input("Proceed? [y/N]: ").strip().lower()
        
        if confirm == 'y':
            exp_list = " ".join(exp_to_run)
            cmd = f"python {PROJECT_ROOT / 'scripts' / 'run_experiments.py'} --experiments {exp_list} --num-runs {num_runs}"
            print(f"\nRunning: {cmd}\n")
            os.system(cmd)
            input("\nPress Enter to continue...")
        
        return


def view_status_menu(experiments):
    while True:
        clear_screen()
        show_header()
        print("\n--- Experiment Status ---\n")
        
        print("  [1] View by disease")
        print("  [2] View all experiments")
        print("  [3] View incomplete only")
        print("  [b] Back")
        print()
        
        choice = input("Select: ").strip().lower()
        
        if choice == 'b':
            return
        
        clear_screen()
        show_header()
        
        if choice == '1':
            selected = select_from_list(DISEASES, "Select disease:", allow_all=False)
            if selected is None:
                continue
            
            disease = selected[0]
            disease_exps = get_experiments_for_disease(disease, experiments)
            
            print(f"\n--- {disease.replace('_', ' ').title()} Experiments ---\n")
            
            for exp in disease_exps:
                runs, status = get_experiment_status(exp)
                marker = "+" if runs >= 50 else "-" if runs == 0 else "~"
                print(f"  [{marker}] {exp:<35} {status}")
        
        elif choice == '2':
            print("\n--- All Experiments ---\n")
            
            total = 0
            complete = 0
            
            for name in sorted(experiments.keys()):
                if isinstance(experiments[name], dict):
                    runs, status = get_experiment_status(name)
                    marker = "+" if runs >= 50 else "-" if runs == 0 else "~"
                    print(f"  [{marker}] {name:<35} {status}")
                    total += 1
                    if runs >= 50:
                        complete += 1
            
            print(f"\nTotal: {complete}/{total} complete")
        
        elif choice == '3':
            print("\n--- Incomplete Experiments ---\n")
            
            incomplete = []
            for name in sorted(experiments.keys()):
                if isinstance(experiments[name], dict):
                    runs, status = get_experiment_status(name)
                    if runs < 50:
                        incomplete.append((name, runs, status))
            
            if incomplete:
                for name, runs, status in incomplete:
                    print(f"  {name:<35} {status}")
                print(f"\n{len(incomplete)} experiments incomplete")
            else:
                print("  All experiments complete!")
        
        input("\nPress Enter to continue...")


def view_results_menu(experiments):
    while True:
        clear_screen()
        show_header()
        print("\n--- View Results ---\n")
        
        print("  [1] Top experiments by accuracy")
        print("  [2] Results by disease")
        print("  [3] Results by architecture")
        print("  [4] Compare GNN vs MLP")
        print("  [b] Back")
        print()
        
        choice = input("Select: ").strip().lower()
        
        if choice == 'b':
            return
        
        clear_screen()
        show_header()
        
        results = []
        for name in experiments.keys():
            if isinstance(experiments[name], dict):
                agg_file = EXPERIMENTS_DIR / name / "aggregated_results.json"
                if agg_file.exists():
                    try:
                        with open(agg_file) as f:
                            data = json.load(f)
                            acc = data.get("test_accuracy", {}).get("mean", 0)
                            std = data.get("test_accuracy", {}).get("std", 0)
                            results.append({
                                "name": name,
                                "accuracy": acc,
                                "std": std,
                                "disease": experiments[name].get("disease", "unknown"),
                                "model": experiments[name].get("model_type", "unknown")
                            })
                    except:
                        pass
        
        if not results:
            print("\nNo results found. Run experiments first.")
            input("\nPress Enter to continue...")
            continue
        
        if choice == '1':
            print("\n--- Top Experiments by Accuracy ---\n")
            sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
            
            for i, r in enumerate(sorted_results[:20], 1):
                print(f"  {i:2}. {r['name']:<35} {r['accuracy']:.1%} +/- {r['std']:.1%}")
        
        elif choice == '2':
            print("\n--- Results by Disease ---\n")
            
            for disease in DISEASES:
                disease_results = [r for r in results if r["disease"] == disease]
                if disease_results:
                    best = max(disease_results, key=lambda x: x["accuracy"])
                    avg = sum(r["accuracy"] for r in disease_results) / len(disease_results)
                    print(f"  {disease:<15} best: {best['accuracy']:.1%} ({best['name']})  avg: {avg:.1%}")
        
        elif choice == '3':
            print("\n--- Results by Architecture ---\n")
            
            arch_results = {}
            for r in results:
                model = r["model"]
                if model not in arch_results:
                    arch_results[model] = []
                arch_results[model].append(r["accuracy"])
            
            for model, accs in sorted(arch_results.items(), key=lambda x: -sum(x[1])/len(x[1])):
                avg = sum(accs) / len(accs)
                print(f"  {model:<20} avg: {avg:.1%} ({len(accs)} experiments)")
        
        elif choice == '4':
            print("\n--- GNN vs MLP Comparison ---\n")
            
            for disease in DISEASES:
                gnn_results = [r for r in results if r["disease"] == disease and r["model"] != "MLP"]
                mlp_results = [r for r in results if r["disease"] == disease and r["model"] == "MLP"]
                
                if gnn_results and mlp_results:
                    gnn_avg = sum(r["accuracy"] for r in gnn_results) / len(gnn_results)
                    mlp_acc = mlp_results[0]["accuracy"]
                    diff = gnn_avg - mlp_acc
                    marker = "+" if diff > 0 else ""
                    print(f"  {disease:<15} GNN: {gnn_avg:.1%}  MLP: {mlp_acc:.1%}  diff: {marker}{diff:.1%}")
        
        input("\nPress Enter to continue...")


def run_analysis_menu():
    clear_screen()
    show_header()
    print("\n--- Analysis & Visualizations ---\n")
    
    print("  [1] Run full analysis")
    print("  [2] Create visualizations")
    print("  [3] Both")
    print("  [b] Back")
    print()
    
    choice = input("Select: ").strip().lower()
    
    if choice == 'b':
        return
    
    if choice in ['1', '3']:
        print("\nRunning analysis...")
        os.system(f"python {PROJECT_ROOT / 'scripts' / 'analyze_results.py'}")
    
    if choice in ['2', '3']:
        print("\nCreating visualizations...")
        os.system(f"python {PROJECT_ROOT / 'scripts' / 'create_visualizations.py'}")
    
    input("\nPress Enter to continue...")


def main():
    experiments = load_experiments()
    
    if not experiments:
        print("Error: Could not load experiments.yaml")
        sys.exit(1)
    
    while True:
        clear_screen()
        choice = show_main_menu()
        
        if choice == '1':
            run_experiments_menu(experiments)
        elif choice == '2':
            view_status_menu(experiments)
        elif choice == '3':
            view_results_menu(experiments)
        elif choice == '4':
            run_analysis_menu()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()

