# BiomeML

*[Version francaise](README.md)*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.6-3C2179)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![pixi](https://img.shields.io/badge/pixi-managed-brightgreen)](https://pixi.sh/)

Disease classification from the gut microbiome using graph neural networks (GNN), with multi-dataset support.

## Why this project

The human gut microbiome contains thousands of microbial taxa linked by phylogenetic relationships. Standard machine learning approaches (MLP, Random Forest) treat these taxa as independent variables and ignore the evolutionary structure connecting them. BiomeML encodes microbial relationships as graphs where each node represents a taxon and each edge represents phylogenetic proximity, then uses GNNs to leverage this structure for disease/healthy classification.

## Architecture

### Global pipeline

The project runs in 5 stages orchestrated by Jupyter notebooks. Each stage is driven by a centralized YAML configuration file.

![Global pipeline](assets/pipeline.png)

### Graph construction

The transformation from raw data to graphs is the step that sets BiomeML apart from standard approaches.

![Graph construction](assets/graph_construction.png)

### `src/` modules

![src/ modules](assets/modules.png)

## Supported datasets

| Dataset | Description | Diseases |
|---------|-------------|----------|
| **AGP** | American Gut Project | IBD, Diabetes, Cancer, Depression, + 6 others |
| **CMD** | curatedMetagenomicData | CRC, IBD, T2D, Cirrhosis, Obesity |
| **HMP** | Human Microbiome Project / IBDMDB | IBD, Crohn's, UC |
| **Custom** | User-provided data | Defined via YAML |

## Quick start

```bash
git clone https://github.com/rayanebelhadj/BiomeML.git
cd BiomeML
pixi install
pixi run check          # verify environment
pixi run menu           # interactive menu
pixi run app            # Streamlit dashboard
```

### Running experiments

```bash
pixi run run --list                                         # list experiments
pixi run run --experiments cmd_ibd_baseline --num-runs 50   # single experiment
pixi run run --all --num-runs 50                            # all experiments
```

## Technologies

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, PyTorch Geometric |
| Graphs | NetworkX, scikit-bio |
| Classical ML | scikit-learn, XGBoost |
| Visualization | Plotly, Matplotlib, Seaborn |
| Interface | Streamlit |
| Data | pandas, NumPy, BIOM-format |
| Environment | pixi |

## Project structure

```
BiomeML/
├── src/
│   ├── datasets/              # Multi-dataset loaders
│   │   ├── base.py            # Abstract BaseDataset interface
│   │   ├── agp.py             # American Gut Project
│   │   ├── curated_metagenomic.py
│   │   ├── hmp.py             # Human Microbiome Project
│   │   └── config_driven.py   # Generic YAML-driven loader
│   ├── models.py              # 10 architectures (GNN + baselines)
│   ├── cross_validation.py    # Stratified k-fold training
│   ├── graph_utils.py         # 5 graph construction methods
│   ├── edge_weights.py        # 9 weighting strategies
│   ├── gpu_graph_conversion.py # NetworkX to PyG conversion
│   ├── config_validation.py   # Strict config validation
│   ├── feature_loader.py      # Clinical metadata
│   └── model_interpretation.py # Gradients, embeddings, analysis
│
├── notebooks/                 # 5-stage pipeline
│   ├── 00_dataset_overview.ipynb
│   ├── 01_data_extraction.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
│
├── scripts/
│   ├── run_experiments.py     # Experiment orchestrator
│   ├── run_interactive.py     # Interactive CLI menu
│   ├── analyze_results.py     # Statistical analysis
│   └── create_visualizations.py
│
├── ui/                        # Streamlit dashboard
│   ├── app.py
│   ├── pages/
│   └── components/
│
├── datasets_config/           # Per-dataset YAML configs
├── tests/                     # Unit and integration tests
├── config.yaml                # Global configuration
├── experiments.yaml           # Experiment definitions
└── pixi.toml                  # Environment and tasks
```

## Architectures

| Model | Type | Description |
|-------|------|-------------|
| GCN | GNN | Graph Convolutional Network |
| GAT | GNN | Graph Attention Network |
| GINEConv | GNN | Graph Isomorphism Network with edge attributes |
| GraphSAGE | GNN | Neighborhood sampling and aggregation |
| EdgeCentricRGCN | GNN | Edge-centric relational GCN |
| MLP | Baseline | Multilayer perceptron (ignores graph structure) |
| CNN | Baseline | 1D convolutional network on abundances |
| Ensemble | Hybrid | Model combination |
| RandomForest | Classical ML | Via scikit-learn |
| XGBoost | Classical ML | Gradient boosting |

## Configuration

Dataset selection in `config.yaml`:

```yaml
dataset:
  name: "cmd"
  config_file: "datasets_config/cmd.yaml"
```

Each dataset has its own file in `datasets_config/` with paths, columns, and specific conditions.

Experiments are defined in `experiments.yaml` with configuration overrides that get merged with the base config.

## Tests

```bash
pytest tests/ -v
```

## License

[MIT](LICENSE)
