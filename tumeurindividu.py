"""
======================================================================
Interprétation INDIVIDUELLE — la démarche hospitalière
Breast Cancer Wisconsin (Diagnostic)
======================================================================

À l'hôpital, on n'interprète pas le modèle « en général » mais la
prédiction d'UNE patiente. Ce script illustre trois éléments clés :

  1. La CALIBRATION du modèle : transformer les scores en probabilités
     dignes de confiance, et le vérifier avec une courbe de fiabilité.
  2. L'explication LOCALE (SHAP waterfall) : ce qui, pour une patiente
     donnée, pousse la prédiction vers malin ou bénin.
  3. La sélection de cas représentatifs, dont un cas incertain proche
     du seuil de décision — celui qui mérite le plus d'attention.

Dépendances : scikit-learn, pandas, numpy, matplotlib, shap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

RANDOM_STATE = 42

# --------------------------------------------------------------------
# Données + modèle optimisé (mêmes réglages que l'analyse précédente)
# --------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
feature_names = list(X.columns)
class_names = {0: "malin", 1: "bénin"}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=800, max_depth=10, max_features="log2",
    min_samples_leaf=4, min_samples_split=5,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)


# --------------------------------------------------------------------
# 1. CALIBRATION
#    Une forêt aléatoire n'est pas naturellement bien calibrée : ses
#    probabilités peuvent être trop prudentes ou trop confiantes.
#    On les recalibre (isotonique) et on compare avant / après.
# --------------------------------------------------------------------
calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)

proba_brut = rf.predict_proba(X_test)[:, 1]
proba_cal = calibrated.predict_proba(X_test)[:, 1]

brier_brut = brier_score_loss(y_test, proba_brut)
brier_cal = brier_score_loss(y_test, proba_cal)

print("=" * 62)
print("CALIBRATION DU MODÈLE")
print("=" * 62)
print("Score de Brier (plus bas = mieux calibré) :")
print(f"   Modèle brut     : {brier_brut:.4f}")
print(f"   Modèle calibré  : {brier_cal:.4f}")

# Courbe de fiabilité
frac_brut, mean_brut = calibration_curve(y_test, proba_brut, n_bins=8)
frac_cal, mean_cal = calibration_curve(y_test, proba_cal, n_bins=8)

plt.figure(figsize=(6.5, 6))
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibration parfaite")
plt.plot(mean_brut, frac_brut, "o-", color="#C44E52", label=f"Brut (Brier {brier_brut:.3f})")
plt.plot(mean_cal, frac_cal, "s-", color="#4C72B0", label=f"Calibré (Brier {brier_cal:.3f})")
plt.xlabel("Probabilité prédite (bénin)")
plt.ylabel("Fréquence réelle observée")
plt.title("Courbe de fiabilité (calibration)")
plt.legend()
plt.tight_layout()
plt.savefig("fig_calibration.png", dpi=130)
plt.close()
print("Figure enregistrée : fig_calibration.png")


# --------------------------------------------------------------------
# 2. SÉLECTION DE CAS REPRÉSENTATIFS
#    On choisit 3 patientes du jeu de test :
#      - une clairement bénigne
#      - une clairement maligne
#      - une incertaine (proba proche de 0,5)
# --------------------------------------------------------------------
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
p = calibrated.predict_proba(X_test_reset)[:, 1]  # proba calibrée de "bénin"

idx_benin = int(np.argmax(p))                    # proba bénin la plus haute
idx_malin = int(np.argmin(p))                    # proba bénin la plus basse
idx_incertain = int(np.argmin(np.abs(p - 0.5)))  # le plus proche de 0,5

cas = {
    "Cas clairement BÉNIN": idx_benin,
    "Cas clairement MALIN": idx_malin,
    "Cas INCERTAIN (près du seuil)": idx_incertain,
}

print("\n" + "=" * 62)
print("TROIS PATIENTES REPRÉSENTATIVES")
print("=" * 62)
for label, i in cas.items():
    print(f"\n{label} (patiente #{i})")
    print(f"   Diagnostic réel        : {class_names[int(y_test_reset[i])]}")
    print(f"   Proba calibrée bénin   : {p[i]:.1%}")
    print(f"   Décision du modèle     : {class_names[int(p[i] >= 0.5)]}")


# --------------------------------------------------------------------
# 3. EXPLICATION LOCALE (SHAP waterfall) pour chaque patiente
#    SHAP explique le raisonnement interne du modèle : quelles variables
#    ont poussé SA prédiction, et dans quel sens.
# --------------------------------------------------------------------
explainer = shap.TreeExplainer(rf)
expl = explainer(X_test_reset)  # objet Explanation

# Sélectionne la classe "bénin" (indice 1) si sortie multi-classes
def explanation_classe1(e, i):
    vals = np.asarray(e.values)
    base = np.asarray(e.base_values)
    if vals.ndim == 3:            # (n, features, classes)
        v = vals[i, :, 1]
        b = base[i, 1] if base.ndim == 2 else base[1]
    else:                          # (n, features)
        v = vals[i]
        b = base[i] if base.ndim == 1 else base
    return shap.Explanation(
        values=v, base_values=b,
        data=X_test_reset.iloc[i].values, feature_names=feature_names,
    )

for label, i in cas.items():
    plt.figure()
    shap.plots.waterfall(explanation_classe1(expl, i), max_display=10, show=False)
    titre = f"{label} — patiente #{i}  (proba bénin calibrée : {p[i]:.0%})"
    plt.title(titre, fontsize=10)
    fname = f"fig_patient_{i}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Explication SHAP enregistrée : {fname}")

print("\nInterprétation : sur un waterfall, chaque barre montre la poussée")
print("d'une variable. Les barres qui augmentent la proba de bénignité")
print("s'opposent à celles qui la diminuent (vers malin). Le point de départ")
print("E[f(x)] est la prédiction moyenne ; la somme des poussées donne la")
print("prédiction finale pour cette patiente précise.")