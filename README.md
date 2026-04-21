<div align="center">

# 🧬 PLGA Drug Release Predictor

### Machine Learning for Controlled Drug Release from PLGA Microspheres

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-brightgreen?style=for-the-badge)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Bridging pharmaceutical science and machine learning to predict long-acting injectable drug release profiles*

</div>

---

## 📖 Overview

PLGA (Poly(lactic-co-glycolic acid)) microspheres are the gold standard for **long-acting injectable (LAI) drug delivery** — used in everything from cancer therapy to contraception. Yet predicting their drug release behavior from formulation parameters alone remains a major challenge, often requiring months of lab experiments.

This project builds a **full ML pipeline** — from raw experimental data to interpretable predictions — to model cumulative drug release kinetics from PLGA microspheres, enabling *in silico* formulation design.

> 🔬 **Dataset**: 25 curated PLGA formulations with 40+ input features spanning drug properties, polymer characteristics, process parameters, and release conditions, mapped to 18-point release curves (6 hours → 112 days).

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Deep EDA** | Scientific exploratory analysis with distribution plots, correlation heatmaps, and endcapping effect visualization |
| ⚙️ **Feature Engineering** | Domain-informed transformations — log-scaled MW, LA:GA ratio encoding, polymer–drug interaction terms |
| 🤖 **Multi-output Modeling** | Predicts full 18-point release curves as well as summary metrics (Burst, T50, T90) |
| 🏆 **Model Benchmarking** | Compares Random Forest, Gradient Boosting, SVR, and ensemble strategies |
| 📊 **SHAP Interpretability** | Feature importance via SHAP beeswarm and bar plots for scientific insight |
| 🎛️ **Hyperparameter Tuning** | Automated cross-validated tuning for top-performing models |

---

## 🗂️ Repository Structure

```
PLGA/
│
├── 📊 Data
│   ├── PLGA dataset for ML.xlsx       ← Raw experimental dataset (n=25)
│   ├── X_features.csv                 ← Engineered feature matrix
│   ├── Y_release_curve.csv            ← 18-point release curve targets
│   └── Y_release_summary.csv          ← Summary targets (Burst, T50, T90, etc.)
│
├── 🐍 Pipeline Scripts
│   ├── 01_data_profiling.py           ← Data cleaning, distributions, correlation heatmap
│   ├── 02_scientific_eda.py           ← Scientific EDA (endcapping effects, polymer trends)
│   ├── 03_feature_engineering.py      ← Feature transformations and encoding
│   ├── 04_model_training.py           ← Model training (summary targets)
│   ├── 04b_multioutput_curve.py       ← Multi-output release curve modeling
│   ├── 05_evaluation.py               ← Cross-validation, learning curves
│   ├── 06_interpretability.py         ← SHAP feature importance analysis
│   ├── 07_hyperparameter_tuning.py    ← GridSearchCV / RandomizedSearchCV
│   └── 08_scientific_insights.py      ← Domain-driven insight extraction
│
├── 📈 Results & Plots
│   ├── 01_distributions.png
│   ├── 01_correlation_heatmap.png
│   ├── 01_release_curves.png
│   ├── 02_scientific_eda.png
│   ├── 02_endcapping_effect.png
│   ├── 04_multioutput_curves.png
│   ├── 05_learning_curves.png
│   ├── 06_shap_bar.png
│   ├── 06_shap_beeswarm.png
│   ├── 05_evaluation_results.csv
│   ├── 06_feature_importance.csv
│   ├── results_burst.csv
│   └── results_t50.csv
```

---

## 🧪 Feature Space

The model draws from **five categories** of formulation descriptors:

<details>
<summary><b>💊 Drug Properties</b></summary>

- Molecular Weight (Da)
- LogP (lipophilicity)
- pKa
- Aqueous Solubility (mg/mL)

</details>

<details>
<summary><b>🧫 Polymer Properties</b></summary>

- Lactide : Glycolide Ratio
- Polymer MW (kDa)
- Inherent Viscosity (dL/g)
- End-capping status (free acid vs. ester)
- Glass Transition Temperature Tg (°C)
- Crystallinity

</details>

<details>
<summary><b>⚗️ Process Parameters</b></summary>

- Encapsulation Method (S/O/W, W/O/W, etc.)
- Solvent System
- Stabilizer Type & Concentration (%)
- Drug Loading (%)
- Entrapment Efficiency (%)
- Polymer : Drug Ratio
- Drying Method (lyophilization, spray-drying)

</details>

<details>
<summary><b>🔬 Particle Characteristics</b></summary>

- Particle Size (µm)
- Polydispersity Index (PDI)
- Zeta Potential (mV)

</details>

<details>
<summary><b>🧲 Release Conditions</b></summary>

- Release Test Method (USP I/II/IV)
- Release Medium & pH
- Temperature (°C)
- Agitation (RPM)
- Dose (mg)

</details>

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy shap openpyxl
```

### Run the Pipeline

Execute scripts in order for a complete analysis:

```bash
# Step 1 — Data profiling & visualization
python 01_data_profiling.py

# Step 2 — Scientific exploratory analysis
python 02_scientific_eda.py

# Step 3 — Feature engineering
python 03_feature_engineering.py

# Step 4 — Model training (summary targets + release curves)
python 04_model_training.py
python 04b_multioutput_curve.py

# Step 5 — Evaluation & learning curves
python 05_evaluation.py

# Step 6 — SHAP interpretability
python 06_interpretability.py

# Step 7 — Hyperparameter tuning
python 07_hyperparameter_tuning.py

# Step 8 — Scientific insights
python 08_scientific_insights.py
```

---

## 🎯 Targets

The pipeline predicts both **summary pharmacokinetic descriptors** and full **release-time curves**:

| Target | Description |
|---|---|
| `Burst Release 24h (%)` | Drug released in first 24 hours |
| `T50 (days)` | Time to 50% cumulative release |
| `T90 (days)` | Time to 90% cumulative release |
| `Max Release (%)` | Plateau cumulative release |
| `Total Release Duration (days)` | End of measurable release |
| `Release curve` | 18 time points: 6h → 112 days |

---

## 📊 Sample Visualizations

| Plot | Description |
|---|---|
| `01_release_curves.png` | All 25 PLGA release profiles overlaid |
| `01_correlation_heatmap.png` | Pearson correlation across numerical features |
| `02_endcapping_effect.png` | Impact of polymer end-capping on burst release |
| `04_multioutput_curves.png` | Predicted vs. actual multi-output release curves |
| `06_shap_beeswarm.png` | SHAP beeswarm — per-sample feature contributions |
| `05_learning_curves.png` | Model learning curve diagnostics |

---

## 🏗️ ML Pipeline Architecture

```
Raw Excel Data
      │
      ▼
 Data Cleaning & Profiling  ──► Distributions, Heatmaps, Release Curves
      │
      ▼
 Scientific EDA  ──────────────► Endcapping effects, Polymer–drug trends
      │
      ▼
 Feature Engineering  ─────────► Log-transforms, Interaction terms, Encoding
      │
      ▼
 ┌────────────────────────────────────────────┐
 │            Model Training                  │
 │  ┌─────────────┐   ┌─────────────────────┐ │
 │  │  Summary    │   │   Full Curve        │ │
 │  │  Targets    │   │  (Multi-output)     │ │
 │  │ Burst, T50  │   │   18 time points    │ │
 │  └─────────────┘   └─────────────────────┘ │
 └────────────────────────────────────────────┘
      │
      ▼
 Cross-Validation & Evaluation  ──► R², RMSE, Learning Curves
      │
      ▼
 SHAP Interpretability  ────────► Feature Importance Bar & Beeswarm
      │
      ▼
 Hyperparameter Tuning  ────────► Optimized Final Models
      │
      ▼
 Scientific Insights  ──────────► Formulation design recommendations
```

---

## 🧠 Scientific Context

PLGA degrades via hydrolysis of ester bonds, with release kinetics governed by:

- **Polymer MW & LA:GA ratio** — Higher MW and more lactide → slower degradation
- **End-capping** — Free-acid end groups accelerate autocatalytic degradation
- **Particle size** — Larger particles prolong release duration
- **Drug physicochemical properties** — LogP and solubility dictate partitioning
- **Drug loading** — High loading can create osmotic pressure channels

This ML pipeline quantifies the relative importance of these mechanisms from real experimental data, providing **data-driven design guidance** for LAI formulation scientists.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Experimental data curated from peer-reviewed PLGA microsphere publications
- SHAP library for model interpretability: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- scikit-learn for ML infrastructure

---

<div align="center">

*Made with 🔬 + 🤖 to accelerate drug delivery science*

**[⭐ Star this repo](https://github.com/AyushChauhan910/PLGA)** if you find it useful!

</div>
