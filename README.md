# TP — Transfer Learning et Apprentissage Multitâche

**M2 SDIA — Advanced Artificial Intelligence · ENSET**

Ce dépôt contient le travail réalisé pour le TP « Transfer Learning et Apprentissage Multitâche » : un notebook exécutable qui traite les trois exercices et l'analyse transversale, et un compte-rendu PDF qui reprend les résultats demandés par l'énoncé.

---

## 1. Contenu


Dossiers créés à l'exécution du notebook :

| Dossier | Contenu |
|---|---|
| `~/.keras/datasets/flower_photos/` | Jeu de fleurs téléchargé automatiquement (3 670 images) |
| `backbone_outputs/` | Exercice 2 : modèles `.keras`, têtes logistiques `.joblib`, `comparaison_backbones.csv`, `predictions_imagenet_top5.csv` |

---

## 2. Environnement

- Python 3, **TensorFlow / Keras 2.21**, scikit-learn, pandas, matplotlib, joblib
- Optionnel : `pydot` + `graphviz` pour `keras.utils.plot_model` (le notebook l'ignore s'ils sont absents)
- Les résultats fournis ont été obtenus sur **CPU** (Windows, aucun GPU détecté). Google Colab avec GPU est recommandé pour aller plus vite.

```bash
pip install tensorflow scikit-learn pandas matplotlib joblib
# optionnel
pip install pydot
```

Une connexion Internet est nécessaire à la première exécution : téléchargement de `flower_photos`, des poids ImageNet des quatre réseaux et de Fashion-MNIST.

---

## 3. Exécution

1. Ouvrir le notebook (Jupyter, VS Code ou Colab) et **exécuter les cellules dans l'ordre**.
2. Les parties s'enchaînent : l'exercice 2 réutilise `data_dir` de l'exercice 1, et la partie 4 réutilise les modèles et résultats des exercices précédents.
3. Interrupteurs pour raccourcir l'exécution :

| Variable | Où | Effet |
|---|---|---|
| `RUN_ABLATION = False` | § 1.4b | Saute l'étude « nombre de couches dégelées × learning rate » (≈ 10–15 min sur CPU) |
| `RUN_LATENCY_BENCH = False` | § 2.12b | Saute le benchmark de latence propre |
| `N_SEEDS = 2` | § 3.1 | Passe de 3 à 2 graines pour l'exercice 3 (le temps total est de l'ordre de 25 à 50 min sur CPU avec 3 graines) |

Reproductibilité : graine 42 pour les exercices 1 et 2 ; graines 42, 59 et 76 pour l'exercice 3. Les valeurs peuvent varier légèrement d'une machine à l'autre.

---

## 4. Organisation du notebook et correspondance avec l'énoncé

| Partie | Contenu | Questions de l'énoncé couvertes |
|---|---|---|
| **1 — Exercice 1** (fleurs, 160×160) | Données et répartition ; CNN de zéro (`blocs4`, avec augmentation) ; extraction MobileNetV2 ; fine-tuning de 30 couches (lr 1e-5) ; étude couches × learning rate ; effet de la quantité de données ; oubli de `preprocess_input` | 1 à 6 |
| **2 — Exercice 2** (backbones, 224×224) | VGG16, ResNet50, MobileNetV2, EfficientNetB0 : extraction unique + régression logistique ; tableau comparatif ; benchmark de latence ; front de Pareto ; matrices de confusion ; t-SNE ; Top-1/Top-5 ImageNet ; recommandations | 1 à 6 |
| **3 — Exercice 3** (Fashion-MNIST) | Étiquettes de la tâche B ; schéma de l'architecture ; mono-tâche vs multitâche (paramètres, accuracy, temps d'inférence) ; balayage de λ<sub>B</sub> ∈ {0,1 ; 0,5 ; 1 ; 2 ; 5} ; régime à 3 000 images ; incohérences et solutions | 1 à 7 |
| **4 — Synthèse** | Expérience de robustesse (question 6) ; tableau récapitulatif ; réponses aux six questions ; recommandation de backbone ; apports et limites du multitâche ; limitations et pistes | Section 3 de l'énoncé |

---

## 5. Résultats principaux

| Exercice | Résultat |
|---|---|
| **1** | CNN de zéro **72,3 %** → MobileNetV2 extraction **90,2 %** (6 405 paramètres entraînés) → fine-tuning **90,6 %** (+0,4 pt, non significatif, sur-apprentissage). Avec 5 % des données, le transfert (75,3 %) dépasse le CNN de zéro à 100 % (73,7 %). |
| **1 (compléments)** | Dégeler 0 à 60 couches donne 89,8 à 91,1 % (écarts dans le bruit). Avec un learning rate de 1e-3, la validation s'effondre à 24,4 %. Oublier `preprocess_input` coûte 4,5 pts (90,6 % → 86,1 %). |
| **2** | EfficientNetB0 **93,9 %** · ResNet50 91,8 % · MobileNetV2 89,5 % · VGG16 86,7 %. MobileNetV2 est le plus compact (9,2 Mo) et le plus rapide (19,5 ms/image, batch = 1, CPU). Front de Pareto : EfficientNetB0 et MobileNetV2. |
| **2 (recommandation)** | (a) serveur GPU : **EfficientNetB0** ; (b) smartphone d'entrée de gamme : **MobileNetV2** (TFLite + quantification int8). |
| **3** | Multitâche : **−50 %** de paramètres, latence **−49,6 %** (batch = 1) et −38,0 % (batch = 256), accuracy équivalente (A : 91,48 % vs 91,58 % ; B : 98,10 % vs 98,09 %). Léger transfert négatif sur A pour λ<sub>B</sub> ≥ 1 (−0,8 pt à λ<sub>B</sub> = 5). Avec 3 000 images, pas de bénéfice démontré. Incohérences brutes : 1,43 % (2 mono) → 0,64 % (multi) → 0,23 % (tête B dérivée) ; 0 % avec un décodage conjoint ou hiérarchique. |
| **Robustesse** | Accuracy du modèle fine-tuné : sombre 79,7 % · bruit 70,4 % · flou 48,9 % · « nuit » 43,1 % (référence 90,6 %). |

---

## 6. Points d'attention

- **Précision statistique.** La validation compte 734 images : l'intervalle à 95 % est d'environ ± 2,2 pts. Les écarts inférieurs à ~2 pts mesurés sur une seule exécution (exercices 1 et 2) ne sont pas concluants. L'exercice 3 utilise 3 graines et rapporte moyenne ± écart-type.
- **Latences.** Elles sont mesurées sur CPU de PC, pas sur smartphone ni sur GPU. La latence « d'extraction » inclut le décodage des images ; la latence « propre » (§ 2.12b) mesure le modèle seul. La décision de déploiement finale doit être validée sur l'appareil cible.
- **Commentaires du notebook à remettre à jour.** Dans les cellules d'analyse des § 2.10 et 2.12, certaines valeurs de latence écrites en texte (46 / 91 / 56 / 182 ms ; « 29,8 puis 56,0 ms » pour MobileNetV2) ne correspondent pas aux sorties actuelles (EfficientNetB0 49,3 ; ResNet50 68,0 ; MobileNetV2 15,7 ; VGG16 164,8 ms). **Le rapport PDF utilise les valeurs des sorties réelles.** Les conclusions qualitatives restent vraies, mais il est préférable de corriger ces commentaires avant le dépôt.
- **Étiquettes de la tâche B.** Elles se déduisent de la tâche A : le multitâche n'apporte donc pas d'information nouvelle, et un résultat « multi ≈ mono » est le résultat attendu.
- **Noms des étudiants.** Le rapport affiche `[Noms des étudiants]` dans la page de titre et l'en-tête de chaque page : remplacer ce texte par vos noms avant le dépôt (l'énoncé l'exige).
- **Code dans le rapport.** Le PDF contient les extraits de code essentiels de chaque exercice ; le **code complet et commenté** est dans le notebook, à joindre au dépôt.

---

## 7. Livrables demandés par l'énoncé

- [x] Code source commenté (notebook)
- [x] Courbes d'apprentissage avant et après fine-tuning, tableaux comparatifs
- [x] Schéma de l'architecture multitâche (figure 10 du rapport)
- [x] Analyse des performances de chaque exercice
- [x] Tableau récapitulatif : de zéro vs extraction vs fine-tuning
- [x] Recommandation argumentée de backbone par scénario de déploiement
- [x] Apports et limites de l'apprentissage multitâche
- [x] Limitations observées et pistes d'amélioration
- [ ] Remplacer les noms en en-tête, puis déposer le PDF (10–15 pages) sur Classroom
