# BiomeML

Classification de maladies a partir du microbiome intestinal utilisant des reseaux de neurones sur graphes, avec support multi-dataset.

## Fonctionnalites

- **Datasets multiples** : AGP, curatedMetagenomicData (CMD), HMP, Custom
- **Loaders modulaires** : interface `BaseDataset` commune
- **Config-driven datasets** : ajout de nouveaux datasets via YAML uniquement
- **Comparaison cross-dataset** : memes experiences sur differentes donnees
- **curatedMetagenomicData** : donnees de meilleure qualite pour CRC, IBD, T2D, Cirrhosis, Obesity

## Datasets supportes

| Dataset | Description | Maladies |
|---------|-------------|----------|
| **AGP** | American Gut Project | IBD, Diabetes, Cancer, Depression, + 6 autres |
| **CMD** | curatedMetagenomicData | CRC, IBD, T2D, Cirrhosis, Obesity |
| **HMP** | Human Microbiome Project / IBDMDB | IBD, Crohn's, UC |
| **Custom** | Donnees utilisateur | Definies par l'utilisateur |

## Installation

```bash
git clone https://github.com/rayanebelhadj/BiomeML.git
cd BiomeML
pixi install
pixi run check
```

## Utilisation

### Menu interactif

```bash
pixi run menu
```

Selectionnez le dataset, la maladie, puis le type d'experience.

### Ligne de commande

```bash
pixi run run --list
pixi run run --experiments cmd_ibd_baseline --num-runs 50
pixi run run --all --num-runs 50
```

### Gestion des datasets

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --dataset cmd --check
```

## Configuration

### Selection du dataset

Dans `config.yaml` :

```yaml
dataset:
  name: "cmd"
  config_file: "datasets_config/cmd.yaml"
```

### Configuration par dataset

Chaque dataset a son fichier dans `datasets_config/` :
- `agp.yaml` : chemins BIOM, metadata, phylogenie
- `cmd.yaml` : etudes, niveaux taxonomiques
- `hmp.yaml` : source HMP, body sites
- `custom_template.yaml` : template pour donnees custom

## Structure du projet

```
BiomeML/
├── src/
│   ├── datasets/              # Loaders multi-dataset
│   │   ├── base.py            # Interface abstraite
│   │   ├── agp.py             # American Gut
│   │   ├── curated_metagenomic.py  # curatedMetagenomicData
│   │   ├── hmp.py             # HMP
│   │   ├── config_driven.py   # Loader generique (YAML-driven)
│   │   └── custom.py          # Donnees custom (legacy)
│   ├── models.py              # Architectures GNN
│   ├── cross_validation.py    # Entrainement et validation croisee
│   ├── edge_weights.py        # Strategies de ponderation
│   ├── graph_utils.py         # Construction de graphes
│   ├── gpu_graph_conversion.py # Conversion NetworkX -> PyG
│   ├── config_validation.py   # Validation stricte de config
│   ├── feature_loader.py      # Metadonnees cliniques
│   └── model_interpretation.py # Interpretation des modeles
│
├── datasets_config/           # Configs par dataset
│   ├── agp.yaml
│   ├── cmd.yaml
│   ├── hmp.yaml
│   └── custom_template.yaml
│
├── notebooks/
│   ├── 00_dataset_overview.ipynb  # Exploration des datasets
│   ├── 01_data_extraction.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
│
├── scripts/
│   ├── run_interactive.py     # Menu interactif
│   ├── run_experiments.py     # Orchestrateur
│   ├── download_datasets.py   # Outil de telechargement
│   ├── analyze_results.py
│   └── create_visualizations.py
│
├── tests/                     # Tests unitaires et d'integration
├── config.yaml                # Configuration globale + selection dataset
├── experiments.yaml           # Experiences AGP + CMD
└── pixi.toml                  # Environnement
```

## Architectures GNN

GCN, GINEConv, GAT, GraphSAGE, EdgeCentricRGCN, MLP, CNN, Ensemble, RandomForest, XGBoost.

## Strategies de ponderation

Distance : identity, inverse, exponential, binary.
Abondance : product, geometric, log, min, max.

## Ajouter un dataset

1. Creer `datasets_config/mon_dataset.yaml` avec les chemins, colonnes et conditions
2. Ajouter des experiences dans `experiments.yaml`

Ou, pour un loader specialise :
1. Creer `src/datasets/mon_dataset.py` heritant de `BaseDataset`
2. Enregistrer dans `src/datasets/__init__.py`
3. Creer `datasets_config/mon_dataset.yaml`
4. Ajouter des experiences dans `experiments.yaml`

## Tests

```bash
pytest tests/ -v
```
