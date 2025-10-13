# PhyloGNN

Classification de maladies a partir du microbiome intestinal en utilisant des reseaux de neurones sur graphes phylogenetiques.

## Installation

```bash
git clone <repo> && cd IBD
pixi install
```

## Utilisation

### Menu interactif (recommande)

```bash
pixi run menu
```

Permet de:
- Selectionner les maladies et experiences a lancer
- Voir le statut des experiences
- Consulter les resultats
- Lancer les analyses

### Ligne de commande

```bash
# Lister les experiences disponibles
pixi run python scripts/run_experiments.py --list

# Lancer des experiences specifiques
pixi run python scripts/run_experiments.py --experiments baseline ibd_gcn --num-runs 50

# Lancer toutes les experiences
pixi run python scripts/run_experiments.py --all --num-runs 50

# Analyser les resultats
pixi run analyze

# Creer les visualisations
pixi run visualize
```

### Autres commandes

```bash
pixi run check      # Verifier l'environnement
pixi run jupyter    # Lancer JupyterLab
```

## Structure

```
IBD/
├── scripts/           # Scripts executables
│   ├── run_interactive.py
│   ├── run_experiments.py
│   ├── analyze_results.py
│   └── create_visualizations.py
│
├── src/               # Modules Python
│   ├── models.py          # Architectures GNN
│   ├── edge_weights.py    # Strategies de ponderation
│   └── graph_utils.py     # Utilitaires graphes
│
├── notebooks/         # Pipeline d'analyse
│   ├── 01_data_extraction.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
│
├── docs/              # Documentation et resultats
├── data/              # Donnees brutes
├── experiments/       # Resultats des experiences
├── figures/           # Visualisations
└── analysis/          # Analyses CSV
```

## Architectures GNN

| Architecture | Description |
|--------------|-------------|
| GCN | Graph Convolutional Network |
| GINEConv | Graph Isomorphism Network avec aretes |
| GAT | Graph Attention Network |
| GraphSAGE | Inductive learning sur graphes |
| EdgeCentricRGCN | Relational GCN |
| MLP | Baseline sans graphe |
| CNN | Baseline convolutionnel |

## Maladies etudiees

IBD, Diabetes, Cancer, Autoimmune, Depression, Mental Illness, PTSD, Arthritis, Asthma, Stomach/Bowel

## Execution sur serveur

```bash
ssh user@server
cd ~/IBD
pixi run menu
```

Pour lancer en arriere-plan:
```bash
nohup pixi run python scripts/run_experiments.py --all --num-runs 50 > logs/exp.log 2>&1 &
```

## Documentation

Voir `docs/` pour:
- Methodologie complete
- Resultats principaux  
- Analyse detaillee
