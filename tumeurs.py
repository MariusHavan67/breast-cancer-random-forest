"""
======================================================================
Random Forest — Analyse avancée sur Breast Cancer Wisconsin (Diagnostic)
Optimisation des hyperparamètres + interprétation approfondie
======================================================================

Pipeline :
  1. Chargement des données
  2. Séparation train / test
  3. Optimisation des hyperparamètres (RandomizedSearchCV)
  4. Comparaison modèle de base vs modèle optimisé
  5. Interprétation :
       - Importance par permutation (sur le jeu de test)
       - Valeurs SHAP (contribution de chaque variable)
       - Dépendance partielle (effet marginal des variables clés)
  6. Lecture médicale des résultats

Dépendances : scikit-learn, pandas, numpy, matplotlib, shap
    pip install scikit-learn pandas numpy matplotlib shap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)


# --------------------------------------------------------------------
# 1-2. Données et séparation train / test
# --------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# Modèle de base (référence)
rf_base = RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
)
rf_base.fit(X_train, y_train)
auc_base = roc_auc_score(y_test, rf_base.predict_proba(X_test)[:, 1])
acc_base = accuracy_score(y_test, rf_base.predict(X_test))


# --------------------------------------------------------------------
# 3. Optimisation des hyperparamètres
#    RandomizedSearchCV : plus rapide qu'une grille exhaustive, il tire
#    au hasard un nombre fixé de combinaisons dans les plages définies.
# --------------------------------------------------------------------
param_distributions = {
    "n_estimators": [200, 300, 400, 600, 800],
    "max_depth": [None, 5, 10, 15, 20],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=40,               # 40 combinaisons testées
    cv=5,                    # validation croisée à 5 plis
    scoring="roc_auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_rf = search.best_estimator_
auc_opt = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
acc_opt = accuracy_score(y_test, best_rf.predict(X_test))

print("=" * 62)
print("OPTIMISATION DES HYPERPARAMÈTRES")
print("=" * 62)
print("Meilleurs paramètres trouvés :")
for k, v in search.best_params_.items():
    print(f"   {k:<20} = {v}")
print(f"\nMeilleur ROC AUC en validation croisée : {search.best_score_:.4f}")
print("\nComparaison sur le jeu de test :")
print(f"   {'':<18}{'Accuracy':>10}{'ROC AUC':>10}")
print(f"   {'Modèle de base':<18}{acc_base:>10.4f}{auc_base:>10.4f}")
print(f"   {'Modèle optimisé':<18}{acc_opt:>10.4f}{auc_opt:>10.4f}")

print("\nRapport de classification (modèle optimisé) :")
print(classification_report(
    y_test, best_rf.predict(X_test), target_names=["malin", "bénin"]
))


# --------------------------------------------------------------------
# 4. Importance par permutation (sur le jeu de TEST)
#    Plus fiable que l'importance native : on mesure la chute de
#    performance quand on mélange aléatoirement une variable.
# --------------------------------------------------------------------
perm = permutation_importance(
    best_rf, X_test, y_test,
    n_repeats=30, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
)
perm_imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

print("=" * 62)
print("IMPORTANCE PAR PERMUTATION — TOP 10")
print("=" * 62)
print(perm_imp.head(10).round(4))

fig, ax = plt.subplots(figsize=(8, 5))
top10 = perm_imp.head(10).iloc[::-1]
ax.barh(top10.index, top10.values, color="#4C72B0",
        xerr=perm.importances_std[perm_imp.head(10).iloc[::-1].index.map(
            lambda n: list(feature_names).index(n))])
ax.set_title("Importance par permutation (chute du ROC AUC)")
ax.set_xlabel("Baisse moyenne du score")
plt.tight_layout()
plt.savefig("fig_permutation.png", dpi=130)
plt.close()


# --------------------------------------------------------------------
# 5. Valeurs SHAP
#    Décomposent chaque prédiction en contributions de chaque variable.
#    TreeExplainer est exact et rapide pour les modèles à base d'arbres.
# --------------------------------------------------------------------
explainer = shap.TreeExplainer(best_rf)
shap_raw = explainer.shap_values(X_test)

# Gère les différents formats de retour selon la version de SHAP
if isinstance(shap_raw, list):
    shap_vals = shap_raw[1]                 # classe "bénin"
elif np.asarray(shap_raw).ndim == 3:
    shap_vals = np.asarray(shap_raw)[:, :, 1]
else:
    shap_vals = shap_raw

# Beeswarm : vue globale de l'effet de chaque variable
plt.figure()
shap.summary_plot(shap_vals, X_test, show=False, max_display=12)
plt.title("SHAP — impact des variables sur la prédiction", fontsize=11)
plt.tight_layout()
plt.savefig("fig_shap_beeswarm.png", dpi=130, bbox_inches="tight")
plt.close()

# Bar : importance SHAP moyenne (|valeur| moyen)
plt.figure()
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False, max_display=12)
plt.title("SHAP — importance moyenne des variables", fontsize=11)
plt.tight_layout()
plt.savefig("fig_shap_bar.png", dpi=130, bbox_inches="tight")
plt.close()

shap_imp = pd.Series(
    np.abs(shap_vals).mean(axis=0), index=feature_names
).sort_values(ascending=False)
print("\n" + "=" * 62)
print("IMPORTANCE SHAP (|valeur| moyen) — TOP 10")
print("=" * 62)
print(shap_imp.head(10).round(4))


# --------------------------------------------------------------------
# 6. Dépendance partielle des 4 variables les plus importantes
#    Montre comment la probabilité prédite évolue avec chaque variable.
# --------------------------------------------------------------------
top4 = list(shap_imp.head(4).index)
fig, ax = plt.subplots(figsize=(10, 7))
PartialDependenceDisplay.from_estimator(
    best_rf, X_test, top4, ax=ax, n_cols=2
)
fig.suptitle("Dépendance partielle — 4 variables les plus influentes", fontsize=12)
plt.tight_layout()
plt.savefig("fig_pdp.png", dpi=130, bbox_inches="tight")
plt.close()

print("\nFigures enregistrées : fig_permutation.png, fig_shap_beeswarm.png,")
print("fig_shap_bar.png, fig_pdp.png")

# On sauvegarde les chiffres clés pour le rapport
resume = {
    "acc_base": acc_base, "auc_base": auc_base,
    "acc_opt": acc_opt, "auc_opt": auc_opt,
    "cv_auc": search.best_score_,
    "best_params": search.best_params_,
    "top_perm": perm_imp.head(6).round(4).to_dict(),
    "top_shap": shap_imp.head(6).round(4).to_dict(),
}
import json
with open("resume_resultats.json", "w") as f:
    json.dump({k: (v if not isinstance(v, dict) else v) for k, v in resume.items()},
              f, ensure_ascii=False, indent=2, default=str)
print("\nRésumé chiffré enregistré dans resume_resultats.json")