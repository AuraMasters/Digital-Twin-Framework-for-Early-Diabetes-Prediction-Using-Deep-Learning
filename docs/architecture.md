# Digital Twin Architecture for Early Diabetes Prediction & Food Nutrition Extraction

## 1. Executive Summary
This framework establishes an end-to-end deep learning-powered **Digital Twin** architecture designed for early diabetes prediction, continuous glucose trajectory forecasting, image-based nutrition extraction, and counterfactual clinical scenario simulation in Type 1 and Type 2 Diabetes cohorts.

The system ingests 5 heterogeneous, asynchronous physiological and lifestyle modalities, encodes them into a unified patient state embedding via hybrid LSTM-CNN networks, and models the temporal state-transition operator using a deep dynamics model.

---

## 2. Multi-Modal Ingestion Pipeline

The architecture processes five distinct physiological time-series modalities sourced from wearable sensors and electronic logging:

| Modality | Features / Dimensions | Representations & Embeddings |
| :--- | :--- | :--- |
| **1. Continuous Glucose (CGM)** | Glucose concentration ($mg/dL$) | 1D float time-series ($\Delta t \approx 5$ min) |
| **2. Insulin Administration** | Dose units ($U$), Event type | 2D float vector (Basal $= 0$, Bolus $= 1$) |
| **3. Nutrition Logs** | Carbs ($g$), Protein ($g$), Fat ($g$), Fibre ($g$), Meal Type, Meal Tag | 4D numeric + 10D Meal Type embedding + 10D Meal Tag embedding ($d_{nut} = 24$) |
| **4. Physical Activity** | Active Kcal, Step count, Distance ($m$), Duration ($s$), Active time, Start times, MET, Motion intensity mean/max | 10D numeric + 4D Activity Type embedding + 3D Intensity embedding ($d_{act} = 17$) |
| **5. Sleep & Biometrics** | Heart rate, Resting heart rate, Sleep level, Stress level, Intensity, Step count | 6D float metrics vector ($d_{slp} = 6$) |

---

## 3. Deep Learning Architecture Pipeline

<div align="center">
  <img src="../architecture.png" alt="Digital Twin Framework Architecture" width="90%">
  <p><em>Figure 1: Architectural schematic of 5-modality ingestion, Hybrid LSTM-CNN state encoders, MLP fusion, and digital twin state-transition dynamics.</em></p>
</div>

```
  ┌─────────────────────────────────────────────────────────────┐
  │                   5-Modality Input Stream                   │
  │   [Glucose]   [Insulin]   [Nutrition]   [Activity]   [Sleep]│
  └───────┬───────────┬────────────┬─────────────┬──────────┬───┘
          │           │            │             │          │
          ▼           ▼            ▼             ▼          ▼
     ┌─────────┐ ┌─────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐
     │ Glucose │ │ Insulin │ │ Nutrition │ │ Activity  │ │  Sleep  │
     │LSTM+CNN │ │LSTM+CNN │ │ LSTM+CNN  │ │ LSTM+CNN  │ │LSTM+CNN │
     └────┬────┘ └────┬────┘ └─────┬─────┘ └─────┬─────┘ └────┬────┘
          │ (64d)     │ (64d)      │ (64d)       │ (64d)      │ (64d)
          └───────────┼────────────┼─────────────┼────────────┘
                      ▼            ▼             ▼
             ┌───────────────────────────────────────────┐
             │    Multimodal Concatenation [320-dim]     │
             └─────────────────────┬─────────────────────┘
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Nonlinear MLP Fusion Network        │
             │        [Linear(320->64) + ReLU +          │
             │             Linear(64->64)]               │
             └─────────────────────┬─────────────────────┘
                                   ▼
             ┌───────────────────────────────────────────┐
             │    Unified Patient State Embedding (S_t)  │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │      Digital Twin Dynamics Network        │
             │           S_{t+1} = S_t + Δ(S_t)          │
             └─────────────────────┬─────────────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
       ┌───────────────────────┐       ┌────────────────────────┐
       │ Multi-Step Rollout    │       │ What-If Counterfactual │
       │ Recursive Forecasting │       │ Scenario Perturbation  │
       │ (H=1, 5, 10, 30, 60)  │       │ S~_t = S_t + δ         │
       └───────────────────────┘       └────────────────────────┘
```

---

## 4. Visual Food Nutrition Extractor (CNN)

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    Input Food RGB Image                     │
  │                        [3, H, W]                            │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │          Hierarchical 2D CNN Feature Extractor              │
  │     [Conv2D + BatchNorm + ReLU + MaxPool + Dropout]         │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              Global Adaptive Average Pooling                │
  │                  Latent Feature (256d)                      │
  └──────────────┬───────────────┬────────────────┬─────────────┘
                 │               │                │
                 ▼               ▼                ▼
     ┌──────────────────────┐ ┌──────────────┐ ┌──────────────┐
     │ Nutrient Regression  │ │  Meal Type   │ │   Food Tag   │
     │      (8 targets)     │ │Classification│ │Classification│
     │ [Calories, Carbs...] │ │ [Breakfast..]│ │ [Food IDs]   │
     └──────────────────────┘ └──────────────┘ └──────────────┘
```

---

## 5. Digital Twin Operators

### 5.1 State Initialization
$$\mathcal{S}_0 = \text{StateEncoder}(\mathbf{x}_{1:5})$$

### 5.2 State Update (Observation Assimilation)
$$\mathcal{S}_t^{\text{twin}} \leftarrow \mathcal{S}_t^{\text{obs}}$$

### 5.3 Counterfactual Scenario Simulation
$$\widetilde{\mathcal{S}}_t = \mathcal{S}_t + \mathbf{\delta}$$

### 5.4 Autoregressive Multi-Step Rollout
$$\mathcal{S}_{t+k} = f_{\theta}(\mathcal{S}_{t+k-1}), \quad k = 1, \dots, H$$

---

## 6. References & Literature Included
1. **GluNet**: A Deep Learning Framework For Accurate Blood Glucose Forecasting.
2. **ReplayBG**: A Digital Twin-based Methodology for T1D In Silico Replay.
3. **Young et al.**: Design and In Silico Evaluation of an Exercise Decision Support System using Digital Twin Models.
4. Individualized Neural Time-Series Models for Blood Glucose Prediction.
