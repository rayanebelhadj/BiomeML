# Research Plan: Phylogenetic GNN for Disease Classification

**Status:**  COMPLETE (January 7, 2026)  
**Total Runs:** 8,300 (166 experiments × 50 runs)  
**Compute Time:** ~5 days

## Research Question

**How do evolutionary relationships between microbes enable disease classification?**

**Answer:** They don't significantly help. GNNs based on phylogenetic relationships do not outperform classical approaches (MLP, CNN).

---

## Summary: 166 Experiments (COMPLETED)

| Category | Count | Notes |
|----------|-------|-------|
| **IBD Experiments** | **48** | |
| - Controls | 6 | baseline, MLP, CNN, random, complete, shuffled |
| - Architectures | 5 | GCN, GINEConv, GAT, GraphSAGE, EdgeCentric |
| - Distance Matrices | 3 | Tree, Seq, Graph |
| - Hyperparameters | 8 | hidden_dim×2, dropout×2, lr×2, layers×2 |
| - Edge Weights | 9 | identity, inverse, exponential, binary, abundance×5 |
| - Graph Density | 4 | k=5,10,15,20 |
| - Metadata Ablation | 5 | age, sex, bmi, antibiotics, all |
| - Multi-class Subtypes | 7 | All 7 architectures |
| - Age Prediction | 1 | GINEConv |
| **Multi-Disease** | **118** | 9 other diseases |
| - Baselines (GINEConv) | 9 | One per disease |
| - MLP Control | 9 | One per disease |
| - CNN Baseline | 9 | One per disease |
| - Random Edges | 9 | One per disease |
| - Complete Graph | 9 | One per disease |
| - Shuffled Labels | 9 | One per disease |
| - All GNN Architectures | 36 | 4 archs × 9 diseases |
| - Age Prediction | 9 | One per disease |
| - Metadata Only | 10 | All 10 diseases |
| - GNN + Metadata | 9 | 9 diseases (IBD has meta_all) |
| **TOTAL** | **166** |  All completed with 50 runs each |

---

## Execution Phases

### Phase 1: IBD Core
```bash
pixi run python run_experiments.py --experiments \
  baseline no_graph_mlp cnn_baseline random_edges complete_graph \
  gnn_gcn gnn_gineconv gnn_gat gnn_graphsage gnn_edgecentric \
  --num-runs 50
```

### Phase 2: IBD Edge Weights
```bash
pixi run python run_experiments.py --experiments \
  edge_identity edge_inverse edge_exponential edge_binary \
  edge_abundance_product edge_abundance_geometric edge_abundance_log \
  edge_abundance_min edge_abundance_max \
  --num-runs 50
```

### Phase 3: IBD Graph Density & Metadata
```bash
pixi run python run_experiments.py --experiments \
  k_5 k_10 k_15 k_20 \
  meta_age meta_sex meta_bmi meta_antibiotics meta_all \
  --num-runs 50
```

### Phase 4: IBD Multi-class
```bash
pixi run python run_experiments.py --experiments \
  ibd_subtypes ibd_age_prediction \
  --num-runs 50
```

### Phase 5: Multi-Disease Baselines + Controls
```bash
pixi run python run_experiments.py --experiments \
  diabetes_baseline diabetes_mlp diabetes_cnn diabetes_random_edges diabetes_complete_graph \
  cancer_baseline cancer_mlp cancer_cnn cancer_random_edges cancer_complete_graph \
  autoimmune_baseline autoimmune_mlp autoimmune_cnn autoimmune_random_edges autoimmune_complete_graph \
  --num-runs 50
```

### Phase 6: Multi-Disease (continued)
```bash
pixi run python run_experiments.py --experiments \
  depression_baseline depression_mlp depression_cnn depression_random_edges depression_complete_graph \
  mental_illness_baseline mental_illness_mlp mental_illness_cnn mental_illness_random_edges mental_illness_complete_graph \
  ptsd_baseline ptsd_mlp ptsd_cnn ptsd_random_edges ptsd_complete_graph \
  --num-runs 50
```

### Phase 7: Multi-Disease (continued)
```bash
pixi run python run_experiments.py --experiments \
  arthritis_baseline arthritis_mlp arthritis_cnn arthritis_random_edges arthritis_complete_graph \
  asthma_baseline asthma_mlp asthma_cnn asthma_random_edges asthma_complete_graph \
  stomach_bowel_baseline stomach_bowel_mlp stomach_bowel_cnn stomach_bowel_random_edges stomach_bowel_complete_graph \
  --num-runs 50
```

### Phase 8: Multi-Disease Architectures
```bash
pixi run python run_experiments.py --experiments \
  diabetes_gcn diabetes_gat diabetes_graphsage diabetes_edgecentric \
  cancer_gcn cancer_gat cancer_graphsage cancer_edgecentric \
  autoimmune_gcn autoimmune_gat autoimmune_graphsage autoimmune_edgecentric \
  depression_gcn depression_gat depression_graphsage depression_edgecentric \
  --num-runs 50
```

### Phase 9: Multi-Disease Architectures (continued)
```bash
pixi run python run_experiments.py --experiments \
  mental_illness_gcn mental_illness_gat mental_illness_graphsage mental_illness_edgecentric \
  ptsd_gcn ptsd_gat ptsd_graphsage ptsd_edgecentric \
  arthritis_gcn arthritis_gat arthritis_graphsage arthritis_edgecentric \
  asthma_gcn asthma_gat asthma_graphsage asthma_edgecentric \
  stomach_bowel_gcn stomach_bowel_gat stomach_bowel_graphsage stomach_bowel_edgecentric \
  --num-runs 50
```

### Phase 10: Age Prediction (all diseases)
```bash
pixi run python run_experiments.py --experiments \
  diabetes_age cancer_age autoimmune_age depression_age mental_illness_age \
  ptsd_age arthritis_age asthma_age stomach_bowel_age \
  --num-runs 50
```

### Phase 11: Metadata Ablation (all diseases)
```bash
pixi run python run_experiments.py --experiments \
  ibd_metadata_only diabetes_metadata_only cancer_metadata_only autoimmune_metadata_only \
  depression_metadata_only mental_illness_metadata_only ptsd_metadata_only \
  arthritis_metadata_only asthma_metadata_only stomach_bowel_metadata_only \
  --num-runs 50
```

### Phase 12: GNN + Metadata (all diseases)
```bash
pixi run python run_experiments.py --experiments \
  diabetes_gnn_meta cancer_gnn_meta autoimmune_gnn_meta depression_gnn_meta \
  mental_illness_gnn_meta ptsd_gnn_meta arthritis_gnn_meta asthma_gnn_meta stomach_bowel_gnn_meta \
  --num-runs 50
```

---

## How to Run

```bash
# Single experiment
pixi run python run_experiments.py --experiments baseline --num-runs 50

# Background execution
nohup pixi run python run_experiments.py --experiments baseline --num-runs 50 > logs/exp.log 2>&1 &

# Check progress
tail -f logs/exp.log
```

---

## Research Questions & Results

| Question | Experiments | Result |
|----------|-------------|--------|
| **Does graph structure help?** | GNN vs MLP vs CNN (10 diseases) |  No significant difference |
| **Are graphs more flexible than CNNs?** | GNN vs CNN comparison |  CNNs perform as well or better |
| **Which architecture is best?** | 7 architectures compared |  GCN (simplest) at 61.8% |
| **Does edge weighting matter?** | 9 strategies tested |  Small impact (±1%) |
| **Does sparsity matter?** | k sweep + complete graph |  k=10-15 optimal |
| **Does metadata help?** | GNN vs Meta vs GNN+Meta |  Minimal improvement |
| **Binary vs multi-class same?** | Age prediction all diseases |  Age easier than disease |
| **Is it disease-specific?** | 10 diseases compared |  Consistent across diseases |

---

## Key Findings

### Main Result
**GNNs based on phylogenetic relationships do NOT significantly outperform classical approaches.**

- GNN accuracy: ~60%
- MLP accuracy: ~60%
- CNN accuracy: ~60%
- All close to random baseline (~50% for binary)

### Surprising Findings
1. **Age signal stronger than disease signal** (62% vs 60%)
2. **Simple GCN outperforms complex architectures** (GAT worst at 55.6%)
3. **Professor's edge weighting** (`abundance_product`) not in top 5
4. **CNNs competitive** despite fixed neighborhoods

### Why GNNs Don't Work Better
1. Weak microbiome-disease signal
2. Phylogenetic relationships not critical for prediction
3. k-NN graphs don't reflect real microbial interactions
4. Small sample sizes
5. Static graphs vs dynamic interactions

---

## Generated Deliverables

### French Reports
- **RAPPORT_RESULTATS.md** - Complete methodology
- **RESULTATS_PRINCIPAUX.md** - Detailed findings  
- **ANALYSE_COMPLETE.md** - Full analysis

### Analysis & Visualizations
- **analysis/** - CSV results, statistical tests
- **figures/** - 6 publication-quality figures

### Data
- **experiments/** - All 166 experiment results (on server)
