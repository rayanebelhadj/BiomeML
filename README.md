# BiomeML

Classification de maladies à partir du microbiome intestinal utilisant des réseaux de neurones sur graphes phylogénétiques.

## Description

Ce projet exploite la structure phylogénétique du microbiome pour prédire différentes conditions médicales. L'approche consiste à construire des graphes où les nœuds représentent des taxons microbiens et les arêtes capturent les relations évolutives, puis à appliquer différentes architectures de GNN pour la classification.

Le projet comprend 293 expériences testant 6 architectures GNN avec 9 stratégies de pondération d'arêtes sur 10 maladies du projet American Gut Project.

## Installation

### Prérequis

- Python 3.11+
- [Pixi](https://prefix.dev/docs/pixi/overview)
- GPU CUDA recommandé (optionnel)

### Installation

```bash
git clone https://github.com/rayanebelhadj/BiomeML.git
cd BiomeML
pixi install
pixi run check
```

### Données requises

Placer les fichiers suivants dans le dossier `data/`:
- `AGP.data.biom` - Données American Gut Project
- `AGP-metadata.tsv` - Métadonnées cliniques
- `2024.09.phylogeny.asv.nwk` - Arbre phylogénétique

## Utilisation

### Menu interactif (recommandé)

```bash
pixi run menu
```

Le menu permet de sélectionner les maladies, les types d'expériences, et le nombre de runs. Supporte les sélections multiples (ex: `1,3,5` ou `1-5`).

### Ligne de commande

```bash
# Lister les expériences disponibles
pixi run run --list

# Lancer des expériences spécifiques
pixi run run --experiments baseline ibd_gcn ibd_gat --num-runs 50

# Lancer toutes les expériences
pixi run run --all --num-runs 50

# Analyser les résultats
pixi run analyze

# Créer les visualisations
pixi run visualize
```

### Autres commandes

```bash
pixi run check      # Vérifier l'environnement (PyTorch, CUDA, PyG)
pixi run test       # Tester les imports
pixi run jupyter    # Lancer JupyterLab
```

## Structure du projet

```
BiomeML/
├── scripts/                      # Scripts exécutables
│   ├── run_interactive.py        # Menu interactif
│   ├── run_experiments.py        # Orchestrateur d'expériences
│   ├── analyze_results.py        # Analyse statistique
│   └── create_visualizations.py  # Génération de figures
│
├── src/                          # Code source
│   ├── models.py                 # Architectures GNN et fonctions de perte
│   ├── cross_validation.py       # Boucle d'entraînement et validation croisée
│   ├── edge_weights.py           # Stratégies de pondération
│   ├── graph_utils.py            # Construction de graphes (kNN, MST, seuil, arbre)
│   ├── gpu_graph_conversion.py   # Conversion NetworkX → PyTorch Geometric
│   ├── feature_loader.py         # Chargement des métadonnées cliniques
│   └── model_interpretation.py   # Interprétation des modèles (gradients, t-SNE)
│
├── notebooks/                    # Pipeline d'analyse
│   ├── 01_data_extraction.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
│
├── tests/                        # Tests unitaires et d'intégration
├── docs/                         # Documentation et rapports
├── config.yaml                   # Configuration de base
├── experiments.yaml              # Définition des 293 expériences
└── pixi.toml                     # Environnement et tâches
```

Les dossiers `data/`, `experiments/`, `analysis/`, et `figures/` sont générés localement mais non versionnés.

## Configuration

### config.yaml

Contient les paramètres par défaut pour l'extraction de données, la construction de graphes, et l'entraînement des modèles (learning rate, epochs, batch size, early stopping, etc.).

### experiments.yaml

Définit les 293 expériences organisées en catégories:
- **Baseline**: MLP, CNN
- **Distance-based**: GNN avec 4 pondérations de distance
- **Abundance-based**: GNN avec 5 pondérations incluant l'abondance
- **Control**: Graphes aléatoires, labels mélangés, graphe complet
- **Metadata ablation**: Avec/sans métadonnées cliniques
- **Multiclass**: Sous-types IBD, prédiction d'âge

Chaque expérience spécifie:
```yaml
experiment_name:
  model_type: "gcn"
  edge_weight_strategy: "inverse"
  hyperparameters:
    hidden_dim: 128
    num_layers: 3
    dropout: 0.3
    learning_rate: 0.001
  data_processing:
    use_metadata: true
```

## Architectures GNN

| Architecture | Arêtes | Description |
|--------------|--------|-------------|
| GCN | Non | Graph Convolutional Network - baseline simple |
| GINEConv | Oui | Graph Isomorphism Network - le plus expressif |
| GAT | Oui | Graph Attention Network - apprend l'importance des voisins |
| GraphSAGE | Non | Inductive learning sur graphes |
| EdgeCentricRGCN | Oui | Relational GCN centré sur les arêtes |
| MLP | N/A | Multi-Layer Perceptron - baseline sans graphe |
| CNN | N/A | Convolutional Neural Network - baseline 1D |

## Stratégies de pondération d'arêtes

| Nom | Formule | Catégorie |
|-----|---------|-----------|
| identity | w = d | Distance |
| inverse | w = 1/d | Distance |
| exponential | w = exp(-d) | Distance |
| binary | w = 1 | Distance |
| abundance_product | w = (1/d) × a₁ × a₂ | Abondance |
| abundance_geometric | w = (1/d) × √(a₁×a₂) | Abondance |
| abundance_log | w = (1/d) × log(1+a₁) × log(1+a₂) | Abondance |
| abundance_min | w = (1/d) × min(a₁,a₂) | Abondance |
| abundance_max | w = (1/d) × max(a₁,a₂) | Abondance |

Où `d` est la distance phylogénétique et `a₁`, `a₂` sont les abondances relatives des taxons.

## Maladies étudiées

10 conditions de santé issues de l'American Gut Project:

IBD, Diabetes, Cancer, Autoimmune, Depression, Mental Illness, PTSD, Arthritis, Asthma, Stomach/Bowel

Chaque maladie est analysée en cas vs contrôles.

## Résultats

Après exécution des expériences:

- `experiments/{exp_name}/`: Modèles entraînés et métriques par run
- `analysis/summary_table.csv`: Comparaison de toutes les expériences
- `analysis/statistical_comparison.csv`: Tests statistiques
- `figures/`: Graphiques de performance

Métriques reportées pour chaque expérience (50 runs):
- Accuracy, AUC-ROC, F1-score (mean ± std)
- Intervalles de confiance à 95%
- Tests statistiques (t-tests, Cohen's d)

## Exécution sur serveur

### Connexion

```bash
ssh user@server
cd ~/BiomeML
pixi run menu
```

### Exécution en arrière-plan

```bash
# Avec screen (recommandé)
screen -S experiments
pixi run run --all --num-runs 50 --parallel-runs 4
# Détacher: Ctrl+A puis D
# Réattacher: screen -r experiments
```

### Monitoring

```bash
# Voir les logs en temps réel
screen -r experiments

# Analyser les résultats partiels
pixi run analyze
```

## Workflow typique

1. Préparer les données avec `01_data_extraction.ipynb`
2. Construire les graphes avec `02_graph_construction.ipynb`
3. Lancer les expériences via `pixi run menu`
4. Analyser: `pixi run analyze && pixi run visualize`
5. Interpréter avec `04_model_interpretation.ipynb`

## Développement

### Ajouter une architecture

1. Définir le modèle dans `src/models.py`
2. Ajouter une entrée dans `experiments.yaml`
3. Lancer les expériences

### Ajouter une stratégie de pondération

1. Implémenter la fonction dans `src/edge_weights.py`
2. L'ajouter à `STRATEGY_FUNCTIONS`
3. Créer des expériences dans `experiments.yaml`

### Tests

```bash
pixi run test              # Vérifier les imports
pixi run check             # Vérifier l'environnement
pytest tests/              # Suite de tests complète
pixi run run --experiments baseline --num-runs 1  # Test end-to-end
```

## Dépendances principales

PyTorch, PyTorch Geometric, BioPython, scikit-learn, pandas, numpy, matplotlib, seaborn.

Voir `pixi.toml` pour la liste complète et les versions exactes.
