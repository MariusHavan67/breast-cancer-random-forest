# 🩺 Breast Cancer — Random Forest & Explainable AI

<p align="center">
  <strong>Classification, calibration and interpretability of breast-tumor predictions with Random Forest</strong><br>
  <em>Machine Learning · Explainable AI · SHAP · Model Calibration · Medical Data</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Random%20Forest-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Explainability-SHAP-purple" alt="SHAP">
  <img src="https://img.shields.io/badge/Status-Academic%20Project-lightgrey" alt="Status">
</p>

> **Portfolio project — Machine Learning / Explainable AI**
>
> This project explores how a Random Forest can classify breast tumors as **benign** or **malignant**, while going beyond predictive performance to study **why** the model makes its predictions and **how reliable its probabilities are**.

> ⚠️ **Important:** this is an academic / demonstration project. It is **not a medical device** and must not be used to diagnose a patient or make clinical decisions.

---

## 🎯 Project overview

A high-performing machine-learning model is not necessarily an interpretable or trustworthy model. This project therefore follows a complete workflow:

```text
Breast Cancer Wisconsin dataset
            │
            ▼
     Train / test split
            │
            ▼
   Random Forest baseline
            │
            ▼
 RandomizedSearchCV (40 configs)
            │
            ▼
     Optimized Random Forest
            │
     ┌──────┼───────────────┐
     ▼      ▼               ▼
Permutation  SHAP          PDP
importance  analysis       analysis
     │      │               │
     └──────┼───────────────┘
            ▼
   Probability calibration
            │
            ▼
  Local SHAP explanations
            │
            ▼
  Clear / malignant / uncertain cases
```

The project deliberately combines **global explanations** (what matters across the dataset) with **local explanations** (what drives one individual prediction).

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset available through `scikit-learn`.

According to the accompanying report, the dataset contains:

| Property | Value |
|---|---:|
| Observations | **569** |
| Numerical features | **30** |
| Benign tumors | **357 (62.7%)** |
| Malignant tumors | **212 (37.3%)** |
| Task | Binary classification |

The variables describe morphological characteristics such as **radius, perimeter, area, texture, concavity and concave points**, calculated from cell images. fileciteturn0file0L36-L46

---

## 🧪 Experimental protocol

- **75 / 25** stratified train/test split
- Random Forest baseline
- **RandomizedSearchCV** with **40 configurations**
- **5-fold cross-validation**
- Optimization criterion: **ROC AUC**
- Final evaluation on a test set kept separate from training
- `random_state = 42` for reproducibility

The experimental protocol and search configuration are implemented in `tumeurs.py`. fileciteturn0file3L37-L45 fileciteturn0file3L57-L80

---

## 🏆 Results

### Baseline vs optimized model

| Model | Accuracy | ROC AUC |
|---|---:|---:|
| Random Forest — baseline | **0.958** | **0.995** |
| Random Forest — optimized | **0.951** | **0.994** |

Best cross-validation ROC AUC: **0.991**. fileciteturn0file1L2-L6

### Best hyperparameters

```text
n_estimators      = 800
max_depth         = 10
max_features      = log2
min_samples_leaf  = 4
min_samples_split = 5
class_weight      = balanced
```

These values are recorded in `resume_resultats.json`. fileciteturn0file1L7-L13

### A useful methodological result

The optimized model does **not** outperform the baseline on the held-out test set. The project therefore illustrates an important ML principle: hyperparameter optimization does not automatically translate into better generalization on a given dataset. The report explicitly discusses this comparison rather than presenting only the best-looking score. fileciteturn0file0L76-L85

---

## 🔎 Explainable AI

### 1. Permutation importance

Permutation importance measures the decrease in ROC AUC after randomly permuting a feature on the test set.

Top variables in the recorded results include:

| Feature | Mean ROC AUC decrease |
|---|---:|
| `worst concave points` | 0.0023 |
| `mean concave points` | 0.0015 |
| `worst concavity` | 0.0014 |
| `worst area` | 0.0011 |
| `worst texture` | 0.0010 |

fileciteturn0file1L15-L21

---

### 2. SHAP global importance

SHAP values quantify how individual variables contribute to model predictions. The mean absolute SHAP values highlight the features with the largest average impact.

Top features:

| Feature | Mean |SHAP| |
|---|---:|
| `worst area` | 0.0625 |
| `worst perimeter` | 0.0623 |
| `worst concave points` | 0.0541 |
| `worst radius` | 0.0512 |
| `mean concave points` | 0.0437 |
| `mean radius` | 0.0268 |

fileciteturn0file1L23-L29

### SHAP beeswarm

The beeswarm provides both **feature importance** and **direction of impact** across observations.

---

## 📈 Partial dependence

Partial dependence plots show how the model output changes as a feature varies, while averaging over the other variables.

The project focuses on the four most influential variables identified through SHAP and generates their partial-dependence curves. fileciteturn0file3L173-L184

---

## 🎯 Probability calibration

The second part of the project goes beyond classification labels and studies the reliability of predicted probabilities.

An **isotonic calibration** is compared with the raw Random Forest probabilities using:

- a reliability curve;
- the Brier score;
- calibrated probabilities for individual cases.

On this dataset, the recorded Brier scores are approximately **0.033 before and after calibration**, with both curves close to the calibration diagonal. fileciteturn0file0L142-L154

---

## 👤 Local explanations — three representative cases

The project also generates SHAP waterfall plots for three representative test-set observations:

### 🟦 Clearly benign case

The model assigns a calibrated benign probability of **100%** in the report's selected case.

### 🟥 Clearly malignant case

The selected malignant case receives a calibrated benign probability of **0%**.

### 🟪 Uncertain case

A third observation lies close to the decision threshold, with a calibrated benign probability of approximately **52%**. The waterfall shows competing contributions rather than a single coherent direction. fileciteturn0file0L183-L194

This is particularly useful for demonstrating why **probability + explanation + uncertainty** can be more informative than a simple binary label.

---

## 💡 Key takeaways

### From a Machine Learning perspective

- A strong baseline can already perform very well on a structured dataset.
- Hyperparameter optimization should always be compared against a reference model.
- Test-set performance and cross-validation performance should be kept conceptually separate.
- Different explainability methods can provide complementary information.

### From an Explainable AI perspective

- **Permutation importance** identifies features whose disruption affects predictive performance.
- **SHAP** provides both global importance and local explanations.
- **Partial dependence** helps visualize model behavior as a feature changes.
- **Waterfall explanations** make individual predictions easier to inspect.
- **Calibration** adds information about the reliability of predicted probabilities.

The report emphasizes the convergence of several interpretability methods around morphological variables such as concave points, concavity, area, perimeter and radius. fileciteturn0file0L124-L133

---

## 🏥 Medical / deployment perspective

This repository is **not a clinical validation study**.

The accompanying report identifies several limitations before any real-world deployment:

- relatively small dataset;
- single-source dataset;
- need for external validation;
- potential distribution shift on hospital data;
- need for appropriate regulatory and clinical oversight.

The report also recommends explicitly flagging predictions close to the decision threshold and keeping the clinician as the final decision-maker. fileciteturn0file0L203-L224

---

## 📁 Repository structure

```text
breast-cancer-random-forest/
│
├── README.md
├── requirements.txt
├── tumeurs.py                 # Global analysis and model interpretation
├── tumeurindividu.py          # Calibration + local SHAP explanations
├── resume_resultats.json      # Key numerical results
│
├── figures/
│   ├── fig_calibration.png
│   ├── fig_patient_1.png
│   ├── fig_patient_9.png
│   ├── fig_patient_88.png
│   ├── fig_pdp.png
│   ├── fig_permutation.png
│   ├── fig_shap_bar.png
│   └── fig_shap_beeswarm.png
│
└── report/
    └── Rapport_RandomForest_Tumeurs.pdf
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<USERNAME>/breast-cancer-random-forest.git
cd breast-cancer-random-forest
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Reproduce the analysis

### Global analysis

```bash
python tumeurs.py
```

This script trains the baseline and optimized Random Forest models and generates:

- permutation importance;
- SHAP beeswarm;
- SHAP global importance;
- partial dependence plots;
- `resume_resultats.json`.

The workflow is implemented directly in `tumeurs.py`. fileciteturn0file3L104-L128 fileciteturn0file3L132-L161

### Individual explanations

```bash
python tumeurindividu.py
```

This script performs probability calibration and generates SHAP waterfall plots for three representative observations. fileciteturn0file2L53-L65 fileciteturn0file2L93-L120 fileciteturn0file2L124-L155

---

## 📄 Full report

The complete project report is available here:

**[📘 Rapport_RandomForest_Tumeurs.pdf](report/Rapport_RandomForest_Tumeurs.pdf)**

It contains the methodology, model comparison, interpretability analysis, calibration study, individual cases and limitations. fileciteturn0file0L9-L24

---

## 🧰 Technologies

- **Python**
- **NumPy / Pandas**
- **scikit-learn**
- **Random Forest**
- **RandomizedSearchCV**
- **SHAP**
- **Matplotlib**
- Model calibration
- Explainable AI / interpretability

---

## 🚀 Possible extensions

The report identifies several natural next steps:

- external validation on an independent cohort;
- comparison with Gradient Boosting and regularized Logistic Regression;
- deeper analysis of misclassified cases;
- investigation of out-of-distribution / atypical observations;
- evaluation of calibration on data from a different source.

fileciteturn0file0L217-L227

---

## 👨‍💻 About this project

This repository was developed as a **Machine Learning / Explainable AI project** around the question:

> **Can a high-performing classifier also provide interpretable and reasonably calibrated predictions?**

The project intentionally combines **performance evaluation**, **model selection**, **global interpretability**, **local explanations** and **probability calibration** in one reproducible workflow.

---

## 📜 License

No open-source license is currently defined for this repository. Add a license such as **MIT** if you want to explicitly authorize reuse of the code.
