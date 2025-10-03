# T2D Reference - Fichiers de référence pour l'analyse du diabète de type 2

## 📋 Description
Ce dossier contient les fichiers de référence du projet d'analyse du microbiome dans le contexte du diabète de type 2 (T2D). Ces fichiers servent de référence pour comprendre les étapes similaires à implémenter pour le projet IBD.

## 🗂️ Structure des fichiers

### 📊 Données T2D/Âge
- `AGP_ages.metadata.txt` - Métadonnées pour l'analyse par âge
- `AGP_ages_seqs.fa` - Séquences pour l'analyse par âge
- `age_all_seqs.fasta` - Séquences d'âge
- `AGP.data.biom.filtered.ages.tsv` - Table filtrée par âge
- `MATRICES_AGES.pickle` - Matrices pour l'analyse par âge
- `BIG_GRAPH_ages_edge_weights_and_node_int_ids.pickle` - Graphe pour l'analyse par âge

### 🔬 Notebooks de référence
- `graphs-pytorch-AGES.ipynb` - Notebook PyTorch pour l'analyse par âge (contient du code T2D)
- `extract_age_data_from_AGP.ipynb` - Extraction des données par âge
- `extract_IBD_data_from_AGP-Copy1.ipynb` - Copie de référence du notebook IBD

### 📁 Fichiers de doublons et temporaires
- `AGP_IBDcontrol.metadata.txt` - Doublon de métadonnées
- `AGP_noIBDcontrol.metadata.txt` - Doublon de métadonnées
- `samples_ibd_vs_controls.tsv` - Doublon de listes d'échantillons
- `feature-table.biom` - Table de test
- `EXTRACTED_BIOM/` - Dossier temporaire
- `IBD/` - Dossier temporaire

## 🎯 Utilisation
Ces fichiers servent de référence pour :
- Comprendre les étapes du pipeline T2D
- Adapter le code pour le projet IBD
- Conserver l'historique des développements
- Référencer les approches méthodologiques similaires

## ⚠️ Note
Ces fichiers ne sont pas nécessaires pour l'exécution du pipeline IBD actuel, mais sont conservés pour référence et comparaison méthodologique.

