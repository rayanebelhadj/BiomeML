# Réseaux de Neurones en Graphe pour la Classification de Maladies à partir du Microbiome

**Date:** 6 janvier 2026  
**Statut:** Expérimentations complètes (166 expériences, 8,300 runs)

---

## Question de Recherche Principale

**Comment les relations évolutives entre les microbes, exprimées sous forme de graphes où les arêtes représentent la proximité relative (distance phylogénétique ou distance entre séquences), permettent de classifier les maladies dans les échantillons de microbiomes séquencés à partir de selles?**

### Sous-questions

1. Est-ce que le problème de classification binaire est "le même" que le problème version multi-classes (ex: catégories d'âge)?
2. Est-ce que les graphes offrent des représentations plus "subtiles"/flexibles que les CNNs qui représentent toutes les similarités sous forme de voisinage binaire dans un vecteur?

---

## Méthodologie

### Approche Expérimentale

Nous avons créé un "sandbox" pour tester systématiquement les effets des changements dans la représentation en graphe sur le pouvoir prédictif d'un modèle de réseau de neurones en graphe.

**166 expériences** ont été réalisées avec **50 runs chacune** (total: 8,300 runs) pour assurer la robustesse statistique.

### Architecture du Projet

```
IBD/
├── data/                          # Données brutes (AGP)
├── experiments/                   # Résultats des 166 expériences
├── notebooks/                     # Pipeline d'analyse
│   ├── 01_data_extraction.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
├── src/
│   ├── models.py                  # 6 architectures GNN + contrôles
│   ├── edge_weights.py            # 9 stratégies de pondération
│   └── graph_utils.py
├── experiments.yaml               # Définition des 166 expériences
└── run_experiments.py             # Orchestrateur
```

---

## Expériences Réalisées

### Phase 1: Maladies Étudiées (10 maladies)

- **IBD** (Maladie inflammatoire de l'intestin) - ~220 cas
- **Diabète** - ~657 cas
- **Cancer** - ~1,464 cas
- **Maladies auto-immunes** - ~248 cas
- **Dépression** - ~2,838 cas
- **Maladie mentale** - ~3,297 cas
- **PTSD** - ~529 cas
- **Arthrite** - ~286 cas
- **Asthme** - ~359 cas
- **Maladies gastro-intestinales** - ~175 cas

### Phase 2: Optimisation Séquentielle (IBD) - 33 expériences

#### 2.1 Densité du Graphe (k-NN)
- Testé: k = {5, 10, 15, 20}
- **Objectif:** Trouver la sparsité optimale

#### 2.2 Matrices de Distance (3 types)
- **Distance phylogénétique** (arbre évolutif)
- **Distance de séquence** (similarité ADN)
- **Distance de graphe** (neighbor-net)

#### 2.3 Stratégies de Pondération des Arêtes (9 stratégies)

| # | Stratégie | Formule | Objectif |
|---|-----------|---------|----------|
| 1 | identity | w = d | Distance brute |
| 2 | inverse | w = 1/d | Proximité |
| 3 | exponential | w = exp(-d) | Décroissance rapide |
| 4 | binary | w = 1 | Pas de pondération |
| 5 | **abundance_product** | w = (1/d) × a₁ × a₂ | **Suggestion du professeur** |
| 6 | abundance_geometric | w = (1/d) × √(a₁×a₂) | Moyenne géométrique |
| 7 | abundance_log | w = (1/d) × log(1+a₁) × log(1+a₂) | Stabilisation |
| 8 | abundance_min | w = (1/d) × min(a₁,a₂) | Limite inférieure |
| 9 | abundance_max | w = (1/d) × max(a₁,a₂) | Limite supérieure |

#### 2.4 Hyperparamètres (8 variations)
- `hidden_dim`: {64, 128, 256}
- `num_layers`: {2, 3, 4}
- `dropout`: {0.2, 0.3, 0.5}
- `learning_rate`: {0.0001, 0.001, 0.01}

### Phase 3: Comparaison d'Architectures - 70 expériences

**6 architectures GNN testées:**

| Architecture | Caractéristiques | Utilise Arêtes? |
|--------------|------------------|-----------------|
| **GCN** | Convolution de graphe simple | Non |
| **GINEConv** | Très expressif (Graph Isomorphism) | Oui |
| **GAT** | Mécanisme d'attention | Oui |
| **GraphSAGE** | Apprentissage inductif | Non |
| **EdgeCentricRGCN** | Référence du professeur | Oui |
| **MLP** | Pas de structure de graphe (contrôle) | N/A |

**1 architecture non-graphe:**
- **CNN** | Voisinage fixe dans un vecteur | N/A |

**Testé sur les 10 maladies** = 7 architectures × 10 maladies × 50 runs = **3,500 runs**

### Phase 4: Expériences de Contrôle - 30 expériences

| Contrôle | Question testée |
|----------|-----------------|
| **Arêtes aléatoires** | Est-ce que les relations SPÉCIFIQUES importent? |
| **Graphe complet** | Est-ce que la SPARSITÉ importe? |
| **Étiquettes mélangées** | Est-ce que le signal est RÉEL? |

Testé sur les 10 maladies = 3 contrôles × 10 maladies × 50 runs = **1,500 runs**

### Phase 5: Métadonnées Cliniques - 24 expériences

**Caractéristiques testées:**
- Âge
- Sexe
- IMC (Indice de Masse Corporelle)
- Historique d'antibiotiques

**Conditions:**
- GNN seul (baseline)
- Métadonnées seules
- GNN + Métadonnées (fusion tardive)

### Phase 6: Classification Multi-classes - 17 expériences

#### 6.1 Sous-types IBD (3 classes)
- Maladie de Crohn vs Colite ulcéreuse vs Contrôle
- Testé avec 7 architectures

#### 6.2 Prédiction d'Âge (5 catégories)
- Classes: <35 ans, 35-55 ans, >55 ans (simplifiées en 5 bins)
- Testé sur les 10 maladies avec 7 architectures

---

## Résultats Préliminaires

### Réponse à la Question Principale

**Les relations évolutives (graphes phylogénétiques) aident-elles à classifier les maladies?**

**Résultats attendus** (basés sur 8,300 runs complétés):

1. **GNN vs MLP:** Les résultats montreront si la structure de graphe améliore la prédiction
2. **GNN vs CNN:** Comparaison entre représentations flexibles (graphes) vs voisinage fixe
3. **Importance des arêtes:** Les expériences de contrôle (arêtes aléatoires) révéleront si les relations phylogénétiques spécifiques importent

### Réponse aux Sous-questions

#### 1. Classification Binaire vs Multi-classes

Les expériences de prédiction d'âge sur les 10 maladies permettront de comparer:
- Performance GNN en classification binaire (maladie vs contrôle)
- Performance GNN en classification multi-classes (catégories d'âge)
- Si les maladies où GNN aide en binaire bénéficient aussi en multi-classes

#### 2. Graphes vs CNNs (Flexibilité)

**Hypothèse testée:** Les graphes permettent des représentations plus subtiles que les CNNs qui forcent un voisinage binaire dans un vecteur.

Les 11 expériences CNN sur les 10 maladies comparées aux GNN révéleront si:
- Les graphes capturent mieux les relations évolutives
- La flexibilité de connexion des graphes surpasse le voisinage fixe des CNNs

---

## Configuration Finale Utilisée

### Graphes
- **Type:** k-NN phylogénétique
- **k:** 10 voisins
- **Distance:** Phylogénétique (arbre évolutif)
- **Pondération:** Inverse (w = 1/distance)

### Modèles
- **Architecture principale:** GINEConv (utilise les poids d'arêtes)
- **hidden_dim:** 64
- **num_layers:** 3
- **dropout:** 0.2
- **learning_rate:** 0.01
- **epochs:** 200

### Validation
- **Stratégie:** 50 runs indépendants par expérience
- **Split:** 70% train, 15% validation, 15% test
- **Métrique principale:** Accuracy avec intervalles de confiance à 95%

---

## Prochaines Étapes (Analyse)

### Phase 7: Interprétation des Modèles
- [ ] Importance des nœuds (quels microbes importent)
- [ ] Visualisation de l'attention (pour GAT)
- [ ] Visualisation des embeddings (t-SNE/UMAP)
- [ ] Analyse des erreurs

### Phase 8: Analyse Statistique
- [ ] Calcul des métriques agrégées (moyenne ± écart-type, IC 95%)
- [ ] Comparaisons par paires (t-tests, Cohen's d)
- [ ] Correction de tests multiples (Bonferroni)
- [ ] Classement final et rapport

---

## Fichiers Clés

- **`experiments.yaml`** - Définition des 166 expériences
- **`run_experiments.py`** - Script d'orchestration
- **`experiments/`** - Résultats des 8,300 runs
- **`PLAN.md`** - Plan détaillé en anglais
- **`general-research-question.txt`** - Question de recherche originale

---

## Résumé Exécutif

| Métrique | Valeur |
|----------|--------|
| **Expériences complétées** | 166/166 |
| **Runs totaux** | 8,300 |
| **Maladies étudiées** | 10 |
| **Architectures testées** | 7 (6 GNN + MLP + CNN) |
| **Stratégies d'arêtes** | 9 |
| **Temps de calcul** | ~5 jours |
| **Statut** |  Expérimentations terminées, prêt pour l'analyse |

---

**Ce projet répond systématiquement à la question de recherche en testant:**
1. Si les relations évolutives (graphes) améliorent la classification
2. Si la classification binaire ≈ multi-classes
3. Si les graphes sont plus flexibles que les CNNs
4. Quelle représentation de graphe (distance, pondération, densité) est optimale

