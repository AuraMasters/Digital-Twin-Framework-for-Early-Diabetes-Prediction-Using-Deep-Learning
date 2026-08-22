<div align="center">

# 🧬 Digital Twin Framework for Early Diabetes Prediction Using Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Domain-Digital%20Twins%20%7C%20Healthcare%20AI-blueviolet.svg)](#)

*An end-to-end multi-modal deep learning digital twin framework for physiological state tracking, continuous glucose forecasting, and counterfactual clinical scenario simulation in Type-1 and Type-2 diabetes cohorts.*

---

</div>

## 📌 Table of Contents
- [1. Overview & Key Capabilities](#1-overview--key-capabilities)
- [2. System Architecture](#2-system-architecture)
- [3. Multi-Modal Ingestion Pipeline](#3-multi-modal-ingestion-pipeline)
- [4. Deep Learning & Digital Twin Formulation](#4-deep-learning--digital-twin-formulation)
- [5. Experimental Results](#5-experimental-results)
- [6. Repository Structure](#6-repository-structure)
- [7. Installation & Quickstart](#7-installation--quickstart)
- [8. Counterfactual What-If Simulation](#8-counterfactual-what-if-simulation)
- [9. Research Literature & References](#9-research-literature--references)

---

## 1. Overview & Key Capabilities

Managing glucose homeostasis and predicting glycemic volatility requires synchronizing asynchronous, heterogeneous physiological signals. This repository implements a **Deep Learning Digital Twin Framework** that models patient-specific metabolic dynamics in silico.

### Key Capabilities:
- **Heterogeneous 5-Modality Fusion**: Ingests continuous glucose monitoring (CGM), basal/bolus insulin delivery, macronutrient dietary logs with categorical embeddings, multi-metric physical activity, and polysomnographic sleep metrics.
- **Unified Causal State Encoding**: Projects variable-length asynchronous physiological histories into a compact, unified patient metabolic state vector $\mathcal{S}_t \in \mathbb{R}^{64}$.
- **Neural State-Transition Dynamics**: Models the temporal evolution operator $\mathcal{S}_{t+1} = \mathcal{S}_t + \Delta_{\theta}(\mathcal{S}_t)$ via residual deep dynamics.
- **Autoregressive Multi-Step Rollouts**: Forecasts patient state trajectories across short ($H=1, 5, 10$) and extended ($H=30, 60$) discrete time steps.
- **Counterfactual "What-If" In Silico Simulation**: Allows clinicians and researchers to apply hypothetical interventions (dietary adjustments, missed insulin doses, exercise variations) to the twin state ($\widetilde{\mathcal{S}}_t = \mathcal{S}_t + \mathbf{\delta}$) without affecting the real patient.

---

## 2. System Architecture

<div align="center">
  <img src="architecture.png" alt="Digital Twin Framework Architecture" width="90%">
  <p><em>Figure 1: Architectural schematic of 5-modality ingestion, recurrent state encoding, MLP fusion, and digital twin state-transition dynamics.</em></p>
</div>

---

## 3. Multi-Modal Ingestion Pipeline

The dataset processes continuous time-series records from the `t1d_uom_v1.0.3` cohort across 5 modalities:

| Modality | Features & Channels | Dimensionality / Representation |
| :--- | :--- | :--- |
| **1. Glucose (CGM)** | Interstitial glucose concentration ($mg/dL$) | 1D float time-series ($\Delta t \approx 5$ min) |
| **2. Insulin** | Dose units ($U$), delivery event type | 2D float vector (Basal $= 0$, Bolus $= 1$) |
| **3. Nutrition** | Carbs ($g$), Protein ($g$), Fat ($g$), Fibre ($g$), Meal Type, Meal Tag | 4D numeric + 10D Meal Type embedding + 10D Meal Tag embedding ($d_{nut} = 24$) |
| **4. Activity** | Active Kcal, Step count, Distance ($m$), Duration ($s$), Active time, MET, Motion intensity mean/max, Activity Type, Intensity | 10D numeric + 4D Activity Type embedding + 3D Intensity embedding ($d_{act} = 17$) |
| **5. Sleep & Biometrics** | Heart rate, Resting heart rate, Sleep level, Stress level, Intensity, Step count | 6D float metrics vector ($d_{slp} = 6$) |

---

## 4. Deep Learning & Digital Twin Formulation

### 4.1 Modality-Specific Recurrent Encoders
Each modality stream $\mathbf{x}^{(m)}$ is processed through a dedicated Gated Recurrent Unit (GRU):
$$\mathbf{z}^{(m)} = \text{GRU}^{(m)}(\mathbf{x}^{(m)}), \quad m \in \{\text{glucose}, \text{insulin}, \text{nutrition}, \text{activity}, \text{sleep}\}$$

### 4.2 Nonlinear State Fusion
The 5 latent vectors are concatenated ($\mathbf{z}_{\text{concat}} \in \mathbb{R}^{320}$) and passed through a multi-layer perceptron:
$$\mathcal{S}_t = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{z}_{\text{concat}} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^{64}$$

### 4.3 Residual State-Transition Dynamics Operator
The Digital Twin advances patient state via residual updates:
$$\Delta(\mathcal{S}_t) = \mathbf{W}_d \cdot \tanh(\mathbf{W}_h \mathcal{S}_t + \mathbf{b}_h) + \mathbf{b}_d$$
$$\mathcal{S}_{t+1} = \mathcal{S}_t + \Delta(\mathcal{S}_t)$$

### 4.4 Twin Operator Interfaces
- **Initialize**: $\mathcal{S}_0^{\text{twin}} \leftarrow \text{StateEncoder}(\mathbf{x}_{1:5})$
- **Update**: $\mathcal{S}_t^{\text{twin}} \leftarrow \mathcal{S}_t^{\text{obs}}$ (assimilates new physical telemetry)
- **Scenario**: $\widetilde{\mathcal{S}}_t \leftarrow \mathcal{S}_t + \mathbf{\delta}$ (hypothetical clinical perturbation)
- **Rollout**: $\mathcal{S}_{t+1:t+H} \leftarrow \text{Rollout}(\mathcal{S}_t, H)$

---

## 5. Experimental Results

Evaluated on held-out test participants (`UoM2401` and `UoM2405`):

### 1-Step ML State Transition Performance
| Participant | Test Size | MSE | RMSE | MAE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UoM2401** | 1,500 states | $2.668 \times 10^{-3}$ | $0.0516$ | $0.0409$ | **0.9109** |
| **UoM2405** | 1,500 states | $2.439 \times 10^{-3}$ | $0.0494$ | $0.0402$ | **0.8509** |

### Digital Twin Recursive Multi-Step Rollout Horizons
| Rollout Horizon ($H$) | UoM2401 ($R^2$) | UoM2401 (RMSE) | UoM2405 ($R^2$) | UoM2405 (RMSE) |
| :---: | :---: | :---: | :---: | :---: |
| **$H = 1$ step** | **0.9109** | 0.0516 | **0.8509** | 0.0494 |
| **$H = 5$ steps** | **0.8645** | 0.0637 | **0.6479** | 0.0759 |
| **$H = 10$ steps** | **0.8457** | 0.0680 | **0.5844** | 0.0824 |
| **$H = 30$ steps** | **0.8014** | 0.0772 | **0.5363** | 0.0868 |
| **$H = 60$ steps** | **0.7124** | 0.0929 | **0.3983** | 0.0983 |

---

## 6. Repository Structure

```
├── configs/
│   └── config.yaml               # Hyperparameter, modality & cohort configs
├── data/
│   └── raw/t1d_uom_v1.0.3/       # Multimodal raw datasets (Glucose, Insulin, etc.)
├── docs/
│   └── architecture.md           # Detailed mathematical & architectural documentation
├── paper/                        # Reference literature & foundational papers
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # Data loading, causal windowing & tokenizers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── modality_encoder.py   # FiveGRU + MLPFusion StateEncoder
│   │   └── digital_twin.py       # TwinDynamics & DigitalTwin operators
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # MSE, RMSE, MAE, R2 calculation
│       └── visualization.py      # Trajectory and simulation plotting
├── 1stReview_PPT.pdf             # Project review presentation slides
├── architecture.png              # System architecture block diagram
├── digital_twin.ipynb            # Interactive research & visualization notebook
├── evaluate.py                   # Standalone rollout evaluation script
├── requirements.txt              # Environment dependencies
├── simulate.py                   # Counterfactual scenario simulator
└── train.py                      # Model training entrypoint
```

---

## 7. Installation & Quickstart

### 7.1 Clone & Environment Setup
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

### 7.2 Training the Digital Twin
```bash
python train.py --config configs/config.yaml --save_path checkpoints/digital_twin.pt
```

### 7.3 Multi-Horizon Evaluation
```bash
python evaluate.py --config configs/config.yaml
```

### 7.4 Interactive Notebook
Launch Jupyter to explore visualizations, embeddings, and trajectories:
```bash
jupyter notebook digital_twin.ipynb
```

---

## 8. Counterfactual What-If Simulation

Run hypothetical scenario simulations to model trajectory divergence under physiological perturbations:

```bash
python simulate.py --participant UoM2401 --horizon 60 --perturbation 0.10
```

```
=====================================================================================
DIGITAL TWIN WHAT-IF SIMULATION: UoM2401
=====================================================================================
  Simulation Horizon           : 60 steps
  Perturbation Factor          : 10.0%
  Initial State L2 Norm        : 1.2788
  Final Baseline State L2 Norm : 1.2377
  Final Scenario State L2 Norm : 1.2691
  Trajectory Divergence Gap    : 0.1013
=====================================================================================
```

---

## 9. Research Literature & References

The methods implemented in this framework build on foundational research included in the `paper/` directory:
1. **GluNet**: *A Deep Learning Framework For Accurate Blood Glucose Forecasting*.
2. **ReplayBG**: *A Digital Twin-based Methodology to Identify Factors Affecting Glycemia in T1D*.
3. **Young et al.**: *Design and In Silico Evaluation of an Exercise Decision Support System Using Digital Twin Models*.
4. *Individualized Models for Glucose Prediction in Type 1 Diabetes*.

---

<div align="center">
  <sub>Developed for Advanced Deep Learning & Digital Twin Healthcare Applications.</sub>
</div>