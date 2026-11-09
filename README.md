# Prédiction de la consommation de carburant avec des SVM (Auto MPG)

Devoir de Machine Learning : prédire la consommation `mpg` (miles par gallon) des voitures avec des SVM, en classification (SVC) puis en régression (SVR).

## Données

- Base **Auto MPG**, chargée avec `seaborn.load_dataset('mpg')`
- 398 lignes, 392 après suppression des 6 valeurs manquantes (`horsepower`)
- Variables continues : `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`
- Variables catégorielles : `cylinders`, `origin`

## Contenu du notebook

1. **Exploration** : valeurs manquantes, doublons, distribution de `mpg`, corrélations
2. **Prétraitement** : suppression des valeurs manquantes, vérification des outliers (IQR), One-Hot Encoding des variables catégorielles, normalisation (StandardScaler ajusté sur le train seulement)
3. **Classification (SVC)**
   - `mpg` découpé en 3 classes (faible, moyenne, élevée) : comparaison quantiles / seuils fixes
   - SVC **from scratch** (algorithme SMO, un-contre-tous), noyau linéaire et RBF
   - Comparaison avec `sklearn.svm.SVC` (accuracy, rapport de classification, matrices de confusion)
4. **Régression (SVR)**
   - SVR linéaire **from scratch** avec la perte ε-insensible (descente de sous-gradient)
   - Comparaison avec `sklearn.svm.SVR` (MSE, RMSE, MAE, R²)
   - Influence de ε sur le nombre de vecteurs de support
   - Visualisation du tube ε
   - Test du bruit : ajout de bruit gaussien et effet de ε

## Résultats principaux

- Les quantiles donnent des classes équilibrées (131 / 130 / 131). Avec les seuils fixes, la classe « élevée » est plus petite.
- SVC RBF : accuracy de **0.848**, identique entre notre SMO et scikit-learn.
- SVR (ε = 1) : RMSE de 2.90 (from scratch) contre 2.87 (scikit-learn), R² d'environ 0.84.
- Plus ε est grand, moins il y a de vecteurs de support. Si ε est trop grand, l'erreur augmente.
- Un ε proche de l'ordre du bruit rend le modèle moins sensible au bruit, mais l'erreur de base augmente.

## Lancer le projet

```bash
pip install numpy pandas seaborn matplotlib scikit-learn jupyter
jupyter notebook Devoir.ipynb
```

Puis : **Kernel → Restart & Run All**.

## Structure

```
.
├── Devoir.ipynb   # code, graphiques et explications
└── README.md
```

## Limites

- Un seul découpage train/test : les résultats peuvent varier un peu selon le tirage.
- Piste d'amélioration : validation croisée et recherche des meilleurs `C` et `gamma`.
