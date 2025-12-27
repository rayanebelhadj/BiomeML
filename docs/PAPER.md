# Les Reseaux de Neurones en Graphe Phylogenetiques n'Ameliorent pas la Classification de Maladies Basee sur le Microbiome Intestinal

**Rayane Belhadj**  
Universite du Quebec a Montreal (UQAM)  
Janvier 2026

---

## Resume

Les reseaux de neurones en graphe (GNN) sont de plus en plus utilises pour analyser des donnees biologiques structurees. Cette etude evalue si les relations evolutives entre microbes, representees sous forme de graphes phylogenetiques, ameliorent la classification de maladies a partir du microbiome intestinal. Nous avons realise 166 experiences totalisant 8,300 runs sur 10 maladies differentes, comparant 6 architectures GNN a des approches classiques (MLP, CNN). Nos resultats montrent qu'aucune architecture GNN ne surpasse significativement les methodes sans structure de graphe. La precision moyenne est d'environ 60% pour toutes les approches, proche du hasard pour une classification binaire. Ce resultat negatif suggere que les relations phylogenetiques ne sont pas critiques pour la prediction de maladies basee sur le microbiome, et que les abondances microbiennes seules contiennent l'essentiel de l'information predictive.

**Mots-cles:** Reseaux de neurones en graphe, microbiome, classification de maladies, phylogenetique, apprentissage automatique

---

## 1. Introduction

Le microbiome intestinal humain est compose de milliards de microorganismes dont la composition varie selon l'etat de sante de l'individu. De nombreuses etudes ont etabli des liens entre la dysbiose microbienne et diverses maladies, incluant les maladies inflammatoires de l'intestin (IBD), le diabete, et certains cancers.

Les reseaux de neurones en graphe (GNN) representent une avancee recente en apprentissage automatique, permettant de traiter des donnees structurees en graphe. Dans le contexte du microbiome, les microbes peuvent etre connectes selon leurs relations evolutives (phylogenie), offrant potentiellement une representation plus riche que les vecteurs d'abondance traditionnels.

**Question de recherche:** Les relations evolutives entre microbes, exprimees sous forme de graphes phylogenetiques, permettent-elles d'ameliorer la classification de maladies par rapport aux approches classiques?

**Hypothese:** Les graphes offrent des representations plus flexibles que les approches vectorielles (CNN, MLP), capturant des relations subtiles entre microbes evolutivement proches.

**Contributions:**

- Evaluation rigoureuse avec 8,300 runs experimentaux
- Comparaison systematique de 7 architectures sur 10 maladies
- Resultat negatif informatif pour la communaute scientifique

---

## 2. Methodologie

### 2.1 Donnees

Les donnees proviennent du American Gut Project (AGP), une des plus grandes bases de donnees publiques du microbiome humain. Nous avons extrait les echantillons pour 10 conditions medicales:

| Maladie | Cas | Controles | Total |
|---------|-----|-----------|-------|
| IBD | 110 | 110 | 220 |
| Diabete | 328 | 329 | 657 |
| Cancer | 732 | 732 | 1,464 |
| Autoimmune | 124 | 124 | 248 |
| Depression | 1,419 | 1,419 | 2,838 |
| Maladie mentale | 1,648 | 1,649 | 3,297 |
| PTSD | 264 | 265 | 529 |
| Arthrite | 143 | 143 | 286 |
| Asthme | 179 | 180 | 359 |
| Gastro-intestinal | 87 | 88 | 175 |

### 2.2 Construction des Graphes

Chaque echantillon est represente comme un graphe ou:

- **Noeuds:** Variants de sequences amplifiees (ASV), representant les microbes
- **Aretes:** Connexions k-NN basees sur la distance phylogenetique
- **Attributs de noeuds:** Abondance relative de chaque microbe

Nous avons teste 9 strategies de ponderation des aretes:
1. Identite (distance brute)
2. Inverse (1/distance)
3. Exponentielle (exp(-distance))
4. Binaire (1 si connecte)
5. Produit d'abondance
6. Moyenne geometrique d'abondance
7. Log d'abondance
8. Minimum d'abondance
9. Maximum d'abondance

### 2.3 Architectures Testees

**Reseaux de neurones en graphe (GNN):**

- GCN (Graph Convolutional Network)
- GINEConv (Graph Isomorphism Network avec aretes)
- GAT (Graph Attention Network)
- GraphSAGE (Sample and Aggregate)
- EdgeCentricRGCN (Relational GCN)

**Controles:**

- MLP (Multi-Layer Perceptron) - sans structure de graphe
- CNN (Convolutional Neural Network) - voisinage fixe

### 2.4 Protocole Experimental

Le protocole experimental est defini comme suit:

- **Runs:** 50 executions independantes par experience
- **Split:** 70% entrainement, 15% validation, 15% test
- **Hyperparametres:** hidden_dim=64, num_layers=3, dropout=0.2, lr=0.01
- **Metrique:** Accuracy avec intervalles de confiance a 95%
- **Total:** 166 experiences, 8,300 runs

---

## 3. Resultats

### 3.1 Comparaison GNN vs MLP

Le tableau suivant compare la meilleure architecture GNN a l'approche MLP (sans graphe) pour chaque maladie:

| Maladie | GNN | MLP | Difference |
|---------|-----|-----|------------|
| IBD | 59.4% | 59.7% | -0.4% |
| Diabete | 58.9% | 59.7% | -0.7% |
| Cancer | 59.4% | 60.5% | -1.1% |
| Autoimmune | 60.5% | 60.9% | -0.5% |
| Depression | 60.1% | 59.6% | +0.5% |
| Maladie mentale | 60.5% | 60.2% | +0.3% |
| PTSD | 60.3% | 60.1% | +0.3% |
| Arthrite | 61.1% | 60.1% | +1.0% |
| Asthme | 60.7% | 60.2% | +0.5% |
| Gastro-intestinal | 60.8% | 58.7% | +2.1% |

**Aucune difference n'est statistiquement significative** (test t, p > 0.05).

### 3.2 Comparaison GNN vs CNN

| Maladie | GNN | CNN | Difference | Significatif |
|---------|-----|-----|------------|--------------|
| IBD | 59.4% | 60.8% | -1.4% | Non |
| Diabete | 58.9% | 61.1% | -2.2% | Non |
| Cancer | 59.4% | 59.1% | +0.2% | Non |
| Autoimmune | 60.5% | 62.8% | -2.3% | **Oui (p<0.05)** |
| Depression | 60.1% | 60.4% | -0.4% | Non |
| Maladie mentale | 60.5% | 60.8% | -0.3% | Non |
| PTSD | 60.3% | 60.7% | -0.4% | Non |
| Arthrite | 61.1% | 61.3% | -0.2% | Non |
| Asthme | 60.7% | 61.3% | -0.6% | Non |
| Gastro-intestinal | 60.8% | 59.4% | +1.4% | Non |

Les CNN performent aussi bien ou mieux que les GNN. Pour la maladie autoimmune, le CNN est significativement meilleur.

### 3.3 Classification Binaire vs Multi-classes

Nous avons compare la classification de maladie (binaire) a la prediction d'age (5 categories):

| Maladie | Binaire (maladie) | Multi-classe (age) |
|---------|-------------------|-------------------|
| IBD | 59.4% | 60.5% |
| Diabete | 58.9% | 61.9% |
| Cancer | 59.4% | 61.4% |
| Autoimmune | 60.5% | 63.4% |
| PTSD | 60.3% | 63.5% |
| Moyenne | 60.0% | 62.2% |

La prediction d'age est plus facile que la classification de maladie, suggerant que le signal d'age dans le microbiome est plus fort que le signal de maladie.

### 3.4 Meilleure Architecture

Sur l'ensemble des experiences IBD:

| Architecture | Accuracy | Ecart-type |
|--------------|----------|------------|
| GCN | 61.8% | 5.9% |
| CNN | 60.8% | 7.8% |
| EdgeCentricRGCN | 60.0% | 4.7% |
| MLP | 59.7% | 4.3% |
| GINEConv | 59.4% | 6.5% |
| GAT | 55.6% | 6.1% |

L'architecture GCN, la plus simple des GNN, performe le mieux. Les architectures plus complexes n'apportent pas d'amelioration.

---

## 4. Discussion

### 4.1 Interpretation des Resultats

Nos resultats montrent clairement que les relations phylogenetiques, bien que biologiquement pertinentes, n'ameliorent pas la classification de maladies. Plusieurs facteurs peuvent expliquer ce resultat:

**Signal faible dans le microbiome:** Avec une precision moyenne d'environ 60%, le signal de maladie dans le microbiome est relativement faible. Ce resultat est coherent avec la litterature suggerant que le microbiome explique seulement une partie de la variabilite des maladies.

**Abondances suffisantes:** Les abondances microbiennes seules, traitees par un simple MLP, capturent deja l'essentiel de l'information predictive. La structure de graphe n'ajoute pas d'information discriminante supplementaire.

**Graphes k-NN non optimaux:** Les graphes k-NN bases sur la phylogenie ne refletent peut-etre pas les vraies interactions microbiennes, qui sont plutot metaboliques ou ecologiques.

### 4.2 Limites

Cette etude presente plusieurs limites:

- **Taille d'echantillons:** Certaines maladies ont peu de cas (ex: IBD ~220)
- **Donnees transversales:** Pas de suivi temporel
- **Graphes statiques:** Meme structure pour tous les echantillons
- **Une seule source:** Uniquement American Gut Project

### 4.3 Implications

Ce resultat negatif a des implications importantes pour la recherche:

1. **Ne pas utiliser de GNN phylogenetiques** pour la classification de maladies basee sur le microbiome sans justification supplementaire
2. **Explorer d'autres types de graphes:** interactions metaboliques, correlations d'abondance, reseaux de co-occurrence
3. **Considerer des approches plus simples:** MLP ou CNN comme baseline robuste

---

## 5. Conclusion

Cette etude repond negativement a la question de recherche: les relations evolutives entre microbes, representees sous forme de graphes phylogenetiques, n'ameliorent pas la classification de maladies par rapport aux approches classiques.

Avec 166 experiences et 8,300 runs sur 10 maladies, nous avons rigoureusement demontre que:

- Les GNN ne surpassent pas les MLP (sans graphe)
- Les CNN performent aussi bien ou mieux
- L'architecture la plus simple (GCN) est la meilleure
- Le signal de maladie dans le microbiome est faible (~60%)

Ce resultat negatif est neanmoins informatif: il evite a d'autres chercheurs d'investir du temps dans cette direction et oriente vers des representations alternatives potentiellement plus pertinentes.

---

## References

1. Human Microbiome Project Consortium. (2012). Structure, function and diversity of the healthy human microbiome. Nature.
2. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
3. McDonald, D., et al. (2018). American Gut: an Open Platform for Citizen Science Microbiome Research. mSystems.
4. Xu, K., et al. (2019). How Powerful are Graph Neural Networks? ICLR.
5. Velickovic, P., et al. (2018). Graph Attention Networks. ICLR.

---

## Annexe: Configuration Experimentale

**Environnement:**

- Python 3.11, PyTorch 2.4, PyTorch Geometric 2.6
- Serveur: frayere.cbe.uqam.ca
- Temps de calcul: ~5 jours

**Code disponible:** Le code source et les donnees sont disponibles sur demande.

