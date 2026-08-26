<div align="center">

# 🧬 Digital Twin Framework for Early Diabetes Prediction & Nutrition Extraction Using Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Domain-Digital%20Twins%20%7C%20Healthcare%20AI-blueviolet.svg)](#)

*An end-to-end multi-modal deep learning digital twin framework for physiological state tracking, continuous glucose forecasting, image-based nutrition extraction, and counterfactual clinical scenario simulation in Type-1 and Type-2 diabetes cohorts.*

---

</div>

## 📌 Table of Contents
- [1. Overview & Key Capabilities](#1-overview--key-capabilities)
- [2. System Architecture](#2-system-architecture)
- [3. Interactive Notebooks](#3-interactive-notebooks)
  - [3.1 Digital Twin Framework Notebook (`digital_twin.ipynb`)](#31-digital-twin-framework-notebook-digital_twinipynb)
  - [3.2 Multimodal Nutrition Analysis & Visual Extraction (`nutrition_analysis.ipynb`)](#32-multimodal-nutrition-analysis--visual-extraction-nutrition_analysisipynb)
  - [3.3 Food Nutrition Extractor Notebook (`food_nutrition_extractor.ipynb`)](#33-food-nutrition-extractor-notebook-food_nutrition_extractoripynb)
- [4. Multi-Modal Ingestion Pipeline](#4-multi-modal-ingestion-pipeline)
- [5. Deep Learning & Digital Twin Formulation](#5-deep-learning--digital-twin-formulation)
- [6. Experimental Results](#6-experimental-results)
- [7. Repository Structure](#7-repository-structure)
- [8. Installation & Quickstart](#8-installation--quickstart)
- [9. Research Literature & References](#9-research-literature--references)

---

## 1. Overview & Key Capabilities

Managing glucose homeostasis and predicting glycemic volatility requires synchronizing asynchronous, heterogeneous physiological signals. This repository provides a complete, self-contained **Deep Learning Digital Twin Framework** alongside a **CNN-based Visual Food Nutrition Extractor** to model patient-specific metabolic dynamics *in silico*.

### Key Capabilities:
- **5-Modality Hybrid LSTM-CNN Encoders**: Ingests continuous glucose monitoring (CGM), basal/bolus insulin delivery, macronutrient dietary logs with categorical embeddings, multi-metric physical activity, and polysomnographic sleep metrics. Each modality is processed through a sequential LSTM followed by a 1D Convolutional feature extractor.
- **Deep CNN Food Nutrition Extractor**: Employs hierarchical 2D Convolutional Neural Networks to extract continuous macronutrients (Calories, Carbohydrates, Protein, Fat, Fiber, Sugar, Sodium, Portion Weight) and classify meal types directly from food images.
- **Unified Causal State Encoding**: Projects variable-length asynchronous physiological histories into a compact, unified patient metabolic state vector $\mathcal{S}_t \in \mathbb{R}^{64}$ via non-linear MLP fusion.
- **Neural State-Transition Dynamics**: Models the temporal evolution operator $\mathcal{S}_{t+1} = \mathcal{S}_t + \Delta_{\theta}(\mathcal{S}_t)$ via residual deep dynamics.
- **Autoregressive Multi-Step Rollouts**: Forecasts patient state trajectories across short ($H=1, 5, 10$) and extended ($H=30, 60$) discrete time steps.
- **Counterfactual "What-If" In Silico Simulation**: Allows clinicians and researchers to apply hypothetical interventions (dietary adjustments, missed insulin doses, exercise variations) to the twin state ($\widetilde{\mathcal{S}}_t = \mathcal{S}_t + \mathbf{\delta}$) without affecting the real patient.

---

## 2. System Architecture

<div align="center">
  <img src="architecture.png" alt="Digital Twin Framework Architecture" width="90%">
  <p><em>Figure 1: Architectural schematic of 5-modality ingestion, LSTM-CNN state encoding, MLP fusion, and digital twin state-transition dynamics.</em></p>
</div>

---

## 3. Interactive Notebooks

The entire research codebase is organized into self-contained, interactive Jupyter Notebooks:

### 3.1 Digital Twin Framework Notebook (`digital_twin.ipynb`)
- **Multi-Modal Data Ingestion**: Parses and aligns Glucose, Insulin, Nutrition, Activity, and Sleep CSV records from the `t1d_uom_v1.0.3` cohort.
- **LSTM-CNN Encoders with Temporal Attention**: Encodes each input stream with an LSTM layer, 1D Convolutional blocks (`Conv1d`, `BatchNorm1d`, `ReLU`), and a learned `TemporalAttention` module ($\alpha_t = \text{softmax}(v^T \tanh(W h_t + b))$) to dynamically focus on critical glycemic excursions and interventions.
- **Cross-Modality Multi-Head Attention Fusion**: Employs `MultiModalAttentionFusion` (`nn.MultiheadAttention`) across all 5 modality tokens ($[z_G, z_I, z_N, z_A, z_S]$) to model inter-physiological coupling before projection into $\mathcal{S}_t \in \mathbb{R}^{64}$.
- **Attention-Gated Residual Dynamics**: Learns $\mathcal{S}_{t+1} = \mathcal{S}_t + \text{Gate}(\mathcal{S}_t) \odot \Delta_{\theta}(\mathcal{S}_t)$ and performs recursive rollouts up to horizon $H=60$.
- **What-If Scenario Simulation**: Counterfactual perturbation ($\widetilde{\mathcal{S}}_t = \mathcal{S}_t + \mathbf{\delta}$) and trajectory divergence analysis.
- **Comprehensive Visualizations**: Metric plots, rollout trajectories, state distributions, and counterfactual comparisons.

### 3.2 Multimodal Nutrition Analysis & Visual Extraction (`nutrition_analysis.ipynb`)
- **Dataset Characterization & Schema Alignment**: Ingests and cross-validates 5,006 dishes, 28,455 ingredient breakdown records, 555 reference ingredients, and 3,490 synchronized overhead dishes.
- **Nutritional EDA & Energy Decomposition**: Statistical distributions, Atwater caloric energy shares (Carbs vs Protein vs Fat), and multivariate Pearson correlation analysis.
- **Clinical Glycemic Segmentation**: Categorization of dishes into 4 diabetes-relevant carbohydrate tiers (Low Carb, Moderate, High Carb, Extreme Glycemic Load).
- **Multimodal Visual & Sensor Topography**: Decodes binary RGB and Depth image streams, renders 4x4 visual galleries, detailed dish decomposition breakdowns, and 3D surface height mesh profiles.
- **PyTorch Dataset Pipeline**: Production-ready `NutritionDataset` with data augmentation and multi-target continuous regression tensors ($[calories, mass, fat, carb, protein]$).
- **Digital Twin Ingestion Bridge**: Demonstrates translation of visual macronutrient estimates directly into the patient state vector $\mathcal{S}_t$.

---

## 4. Multi-Modal Ingestion Pipeline

The dataset processes continuous time-series records from the `t1d_uom_v1.0.3` cohort across 5 modalities:

| Modality | Features & Channels | Dimensionality / Representation |
| :--- | :--- | :--- |
| **1. Glucose (CGM)** | Interstitial glucose concentration ($mg/dL$) | 1D float time-series ($\Delta t \approx 5$ min) |
| **2. Insulin** | Dose units ($U$), delivery event type | 2D float vector (Basal $= 0$, Bolus $= 1$) |
| **3. Nutrition** | Carbs ($g$), Protein ($g$), Fat ($g$), Fibre ($g$), Meal Type, Meal Tag | 4D numeric + 10D Meal Type embedding + 10D Meal Tag embedding ($d_{nut} = 24$) |
| **4. Activity** | Active Kcal, Step count, Distance ($m$), Duration ($s$), Active time, MET, Motion intensity mean/max, Activity Type, Intensity | 10D numeric + 4D Activity Type embedding + 3D Intensity embedding ($d_{act} = 17$) |
| **5. Sleep & Biometrics** | Heart rate, Resting heart rate, Sleep level, Stress level, Intensity, Step count | 6D float metrics vector ($d_{slp} = 6$) |

---

## 5. Deep Learning & Digital Twin Formulation

### 5.1 Hybrid LSTM-CNN Modality Encoders
Each modality stream $\mathbf{x}^{(m)}$ is processed through a sequential LSTM followed by 1D Convolutional feature blocks:
$$\mathbf{h}^{(m)} = \text{LSTM}^{(m)}(\mathbf{x}^{(m)})$$
$$\mathbf{z}^{(m)} = \text{AdaptiveAvgPool1d}(\text{Conv1D}^{(m)}(\mathbf{h}^{(m)})), \quad m \in \{\text{glucose}, \text{insulin}, \text{nutrition}, \text{activity}, \text{sleep}\}$$

### 5.2 Nonlinear State Fusion
The 5 latent vectors are concatenated ($\mathbf{z}_{\text{concat}} \in \mathbb{R}^{320}$) and passed through a multi-layer perceptron:
$$\mathcal{S}_t = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{z}_{\text{concat}} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^{64}$$

### 5.3 Residual State-Transition Dynamics Operator
The Digital Twin advances patient state via residual updates:
$$\Delta(\mathcal{S}_t) = \mathbf{W}_d \cdot \tanh(\mathbf{W}_h \mathcal{S}_t + \mathbf{b}_h) + \mathbf{b}_d$$
$$\mathcal{S}_{t+1} = \mathcal{S}_t + \Delta(\mathcal{S}_t)$$

### 5.4 Twin Operator Interfaces
- **Initialize**: $\mathcal{S}_0^{\text{twin}} \leftarrow \text{StateEncoder}(\mathbf{x}_{1:5})$
- **Update**: $\mathcal{S}_t^{\text{twin}} \leftarrow \mathcal{S}_t^{\text{obs}}$ (assimilates new physical telemetry)
- **Scenario**: $\widetilde{\mathcal{S}}_t \leftarrow \mathcal{S}_t + \mathbf{\delta}$ (hypothetical clinical perturbation)
- **Rollout**: $\mathcal{S}_{t+1:t+H} \leftarrow \text{Rollout}(\mathcal{S}_t, H)$

---

## 6. Experimental Results

Evaluated on held-out test participants (`UoM2401` and `UoM2405`):

### 1-Step ML State Transition Performance
| Participant | Test Size | MSE | RMSE | MAE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UoM2401** | 1,500 states | $1.064 \times 10^{-7}$ | $3.262 \times 10^{-4}$ | $2.321 \times 10^{-4}$ | **1.0000** |
| **UoM2405** | 1,500 states | $3.491 \times 10^{-8}$ | $1.868 \times 10^{-4}$ | $1.255 \times 10^{-4}$ | **1.0000** |

### Digital Twin Recursive Multi-Step Rollout Horizons
| Rollout Horizon ($H$) | UoM2401 ($R^2$) | UoM2401 (RMSE) | UoM2405 ($R^2$) | UoM2405 (RMSE) |
| :---: | :---: | :---: | :---: | :---: |
| **$H = 1$ step** | **1.0000** | $3.262 \times 10^{-4}$ | **1.0000** | $1.868 \times 10^{-4}$ |
| **$H = 5$ steps** | **0.9998** | $9.401 \times 10^{-4}$ | **0.9999** | $7.882 \times 10^{-4}$ |
| **$H = 10$ steps** | **0.9995** | $1.579 \times 10^{-3}$ | **0.9996** | $1.485 \times 10^{-3}$ |
| **$H = 30$ steps** | **0.9975** | $3.655 \times 10^{-3}$ | **0.9972** | $3.864 \times 10^{-3}$ |
| **$H = 60$ steps** | **0.9904** | $7.157 \times 10^{-3}$ | **0.9899** | $7.315 \times 10^{-3}$ |

---

## 7. Repository Structure

```
├── configs/
│   └── config.yaml               # Hyperparameter, modality & cohort configs
├── data/
│   └── raw/t1d_uom_v1.0.3/       # Multimodal raw datasets (Glucose, Insulin, etc.)
├── docs/
│   └── architecture.md           # Detailed mathematical & architectural documentation
├── paper/                        # Reference literature & foundational papers
├── 1stReview_PPT.pdf             # Project review presentation slides
├── architecture.png              # System architecture block diagram
├── digital_twin.ipynb            # Complete Digital Twin pipeline notebook (LSTM-CNN)
├── food_nutrition_extractor.ipynb# CNN Food Nutrition Extractor notebook
├── requirements.txt              # Environment dependencies
└── README.md
```

---

## 8. Installation & Quickstart

### 8.1 Clone & Environment Setup
```bash
git clone https://github.com/Next-Gen-Coder-2007/Digital-Twin-Framework-for-Early-Diabetes-Prediction-Using-Deep-Learning.git
cd Digital-Twin-Framework-for-Early-Diabetes-Prediction-Using-Deep-Learning

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 8.2 Launch Jupyter Notebooks
```bash
# Launch Digital Twin Framework
jupyter notebook digital_twin.ipynb

# Launch Food Nutrition Extractor
jupyter notebook food_nutrition_extractor.ipynb
```

---

## 9. Research Literature & References

1. **GluNet**: *A Deep Learning Framework For Accurate Blood Glucose Forecasting*.
2. **ReplayBG**: *A Digital Twin-based Methodology to Identify Factors Affecting Glycemia in T1D*.
3. **Young et al.**: *Design and In Silico Evaluation of an Exercise Decision Support System Using Digital Twin Models*.
4. *Individualized Models for Glucose Prediction in Type 1 Diabetes*.

---

<div align="center">
  <sub>Developed for Advanced Deep Learning & Digital Twin Healthcare Applications.</sub>
</div>