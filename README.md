# CardioXAI

**An Explainable AI System for No-Laboratory Coronary Heart Disease Risk Screening**

---

## Abstract

CardioXAI is an end-to-end explainable artificial intelligence (XAI) system designed to estimate an individual's risk of coronary heart disease (CHD) using exclusively self-reported lifestyle, demographic, and medical history data — requiring no laboratory tests, clinical appointments, or specialist access. The system trains three supervised classifiers (Logistic Regression, Random Forest, and XGBoost) on the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset comprising 253,680 adult observations, applies Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance, and constructs a weighted consensus ensemble from the two best-performing models. Every prediction is accompanied by dual-layer explainability through SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations), a confidence tier derived from inter-model disagreement and predictive entropy, an interactive CHD risk factor knowledge graph built with NetworkX and Pyvis, and personalised lifestyle recommendations grounded in the clinical evidence base.

---

## Table of Contents

1. [Motivation and Research Context](#1-motivation-and-research-context)
2. [System Architecture](#2-system-architecture)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Model Performance](#5-model-performance)
6. [Explainability Framework](#6-explainability-framework)
7. [Knowledge Graph](#7-knowledge-graph)
8. [Technology Stack](#8-technology-stack)
9. [Project Structure](#9-project-structure)
10. [Installation and Setup](#10-installation-and-setup)
11. [Usage](#11-usage)
12. [API Reference](#12-api-reference)
13. [Limitations and Ethical Considerations](#13-limitations-and-ethical-considerations)
14. [Future Work](#14-future-work)
15. [References](#15-references)
16. [Licence](#16-licence)

---

## 1. Motivation and Research Context

Cardiovascular disease (CVD) remains the leading cause of mortality globally, accounting for approximately 17.9 million deaths annually (WHO, 2021). Conventional risk assessment tools — including the Framingham Risk Score, QRISK3, and the Pooled Cohort Equations — depend on clinical and laboratory inputs such as total cholesterol, HDL cholesterol, systolic blood pressure readings, fasting glucose, and electrocardiogram (ECG) results. These requirements create a significant accessibility barrier for populations without regular healthcare access, particularly in low- and middle-income countries and underserved communities within high-income nations.

CardioXAI investigates the following research question:

> *Can self-reported behavioural and demographic data, processed through an explainable machine learning pipeline, produce clinically meaningful coronary heart disease risk estimates that are both accurate and interpretable to non-expert users?*

The system contributes to the growing body of literature on responsible AI in healthcare by demonstrating that:

- **Accessibility**: Risk screening can be conducted without laboratory inputs, reducing the barrier to early cardiovascular awareness.
- **Transparency**: Post-hoc explainability methods (SHAP and LIME) can render black-box ensemble predictions interpretable to lay users.
- **Uncertainty Quantification**: Inter-model disagreement and predictive entropy can be used to flag unreliable predictions and route users toward professional consultation rather than acting on uncertain scores.
- **Actionability**: Personalised, evidence-based recommendations can be generated from explainability outputs, converting risk awareness into concrete behavioural guidance.

---

## 2. System Architecture

CardioXAI follows a three-tier architecture:

```
+-------------------------------------------------------------+
|                    PRESENTATION LAYER                        |
|  Django Templates + Bootstrap 5 + Public Sans Typography     |
|  Landing | About | 3-Step Assessment | Results Dashboard    |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                    APPLICATION LAYER                         |
|  Django Views + REST API (DRF)                              |
|  Form Validation | Session Management | JSON Serialisation  |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                   MACHINE LEARNING LAYER                     |
|  Data Pipeline | SMOTE | Model Training | Ensemble          |
|  SHAP Explainer | LIME Explainer | Knowledge Graph Engine   |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                      DATA LAYER                              |
|  BRFSS Dataset (253,680 obs) | Serialised Models (joblib)   |
|  Feature Metadata | Evaluation Metrics (JSON)               |
+-------------------------------------------------------------+
```

---

## 3. Dataset

### 3.1 Source

The system uses the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset, a large-scale annual health survey conducted by the Centers for Disease Control and Prevention (CDC) across all 50 US states, the District of Columbia, and participating territories. The specific dataset variant used is the Heart Disease Health Indicators subset, which preprocesses the raw BRFSS survey into a binary classification format.

| Property | Value |
|---|---|
| **Records** | 253,680 |
| **Features (raw)** | 22 |
| **Features (used)** | 16 |
| **Target Variable** | `HeartDiseaseorAttack` (binary: 0 = No CHD, 1 = CHD) |
| **Class Distribution** | 87.8% negative, 12.2% positive |
| **Deduplication** | Applied (253,680 → 186,487 unique records) |
| **Missing Values** | Dropped (complete-case analysis) |

### 3.2 Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `HighBP` | Binary | Ever told by doctor you have high blood pressure |
| `HighChol` | Binary | Ever told by doctor you have high cholesterol |
| `BMI` | Continuous | Body Mass Index (kg/m²) |
| `Smoker` | Binary | Smoked at least 100 cigarettes in lifetime |
| `Stroke` | Binary | Ever told by doctor you had a stroke |
| `Diabetes` | Binary | Ever told by doctor you have diabetes |
| `PhysActivity` | Binary | Physical activity in past 30 days (outside work) |
| `Fruits` | Binary | Consume fruit one or more times per day |
| `Veggies` | Binary | Consume vegetables one or more times per day |
| `HvyAlcoholConsump` | Binary | Heavy drinking (men >14 drinks/wk, women >7 drinks/wk) |
| `GenHlth` | Ordinal (1–5) | Self-reported general health (1 = Excellent, 5 = Poor) |
| `MentHlth` | Integer (0–30) | Days of poor mental health in past 30 days |
| `PhysHlth` | Integer (0–30) | Days of poor physical health in past 30 days |
| `DiffWalk` | Binary | Serious difficulty walking or climbing stairs |
| `Sex` | Binary | Biological sex (0 = Female, 1 = Male) |
| `Age` | Ordinal (1–13) | Age category in 5-year bands (1 = 18–24 through 13 = 80+) |

### 3.3 Data Splitting Strategy

The dataset is partitioned using stratified random sampling to preserve the original class distribution across all splits:

| Split | Proportion | Records |
|---|---|---|
| Training | 70% | 130,540 |
| Validation | 15% | 27,973 |
| Test | 15% | 27,974 |

---

## 4. Methodology

### 4.1 Class Imbalance Handling

The target variable exhibits significant class imbalance (87.8% negative vs. 12.2% positive). The Synthetic Minority Oversampling Technique (SMOTE) (Chawla et al., 2002) is applied **exclusively to the training set** to generate synthetic positive-class observations, yielding a balanced training distribution (50/50). Validation and test sets remain untouched to ensure unbiased evaluation.

| Set | Before SMOTE | After SMOTE |
|---|---|---|
| Training | 130,540 (87.8% / 12.2%) | 229,188 (50% / 50%) |
| Validation | 27,973 (unchanged) | 27,973 (unchanged) |
| Test | 27,974 (unchanged) | 27,974 (unchanged) |

### 4.2 Model Selection

Three classifiers were selected to span the interpretability–performance spectrum:

#### 4.2.1 Logistic Regression (LR)

A linear model that estimates the log-odds of CHD as a weighted linear combination of input features. Selected for its inherent interpretability and coefficient-based feature attribution.

- **Solver**: L-BFGS
- **Max iterations**: 1,000
- **Regularisation**: L2 (default)

#### 4.2.2 Random Forest (RF)

A bagged ensemble of decision trees that reduces variance through bootstrap aggregation and random feature subspace selection.

- **Estimators**: 200
- **Max depth**: 15
- **Min samples split**: 5

#### 4.2.3 XGBoost (XGB)

A gradient-boosted tree ensemble that sequentially fits residual errors using regularised objective functions.

- **Estimators**: 200
- **Max depth**: 6
- **Learning rate**: 0.1
- **Evaluation metric**: Log-loss

### 4.3 Ensemble Construction

A weighted consensus ensemble is constructed from the two best-performing models as ranked by validation AUC-ROC. The ensemble probability is computed as:

```
P_ensemble(y=1|x) = w₁·P_model₁(y=1|x) + w₂·P_model₂(y=1|x)
```

where weights `w₁` and `w₂` are proportional to each model's validation AUC-ROC:

```
wᵢ = AUC_ROCᵢ / (AUC_ROC₁ + AUC_ROC₂)
```

**Selected ensemble**: XGBoost (weight: 0.502) + Logistic Regression (weight: 0.498)

### 4.4 Confidence Estimation

Each prediction is assigned a confidence tier based on two metrics:

1. **Inter-model disagreement**: Absolute difference between the ensemble members' predicted probabilities.
2. **Predictive entropy**: Shannon entropy of the ensemble's predicted probability distribution.

| Confidence Tier | Condition |
|---|---|
| **High** | Disagreement ≤ 0.15 AND Entropy ≤ 0.85 |
| **Medium** | Disagreement ≤ 0.25 AND Entropy ≤ 0.95 |
| **Low** | Disagreement > 0.25 OR Entropy > 0.95 |

Low-confidence predictions trigger a distinct UI pathway that routes users toward professional medical consultation rather than presenting a potentially misleading risk score.

---

## 5. Model Performance

### 5.1 Validation Set Results

| Model | Accuracy | F1 Score | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.7318 | 0.4063 | 0.8150 | 0.2784 | 0.7512 |
| Random Forest | 0.8575 | 0.3616 | 0.8131 | 0.3994 | 0.3304 |
| XGBoost | 0.8787 | 0.2438 | 0.8204 | 0.5112 | 0.1601 |
| **Ensemble (XGB + LR)** | **0.8388** | **0.4254** | **0.8203** | **0.3768** | **0.4884** |

### 5.2 Test Set Results

| Model | Accuracy | F1 Score | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.7297 | 0.4035 | 0.8112 | 0.2762 | 0.7486 |
| Random Forest | 0.8583 | 0.3558 | 0.8043 | 0.3999 | 0.3205 |
| XGBoost | 0.8792 | 0.2428 | 0.8117 | 0.5177 | 0.1586 |
| **Ensemble (XGB + LR)** | **0.8391** | **0.4126** | **0.8150** | **0.3724** | **0.4627** |

### 5.3 Analysis

- **AUC-ROC**: All models achieve AUC-ROC values above 0.80, indicating good discriminatory ability between CHD-positive and CHD-negative individuals. The ensemble achieves 0.815 on the held-out test set.
- **Precision–Recall Trade-off**: Logistic Regression prioritises recall (0.749) at the expense of precision (0.276), reflecting the SMOTE-augmented training distribution. XGBoost prioritises precision (0.518) with lower recall (0.159). The ensemble balances these trade-offs (precision 0.372, recall 0.463), achieving the highest F1 score (0.413).
- **Generalisation**: Minimal performance degradation between validation and test sets indicates that the models generalise well and are not overfitting.

---

## 6. Explainability Framework

CardioXAI implements a dual-layer explainability framework to maximise interpretability robustness. Each prediction is accompanied by both SHAP and LIME explanations, providing two independent perspectives on the factors driving the risk score.

### 6.1 SHAP (SHapley Additive exPlanations)

SHAP (Lundberg & Lee, 2017) computes the marginal contribution of each feature to the prediction, grounded in cooperative game theory (Shapley values). CardioXAI uses:

- **`LinearExplainer`** for the Logistic Regression component, which computes exact SHAP values analytically.
- **`TreeExplainer`** for the XGBoost component, which uses a polynomial-time algorithm for tree-based models.

The ensemble SHAP values are computed as a weighted average of the individual model SHAP values, using the same weights as the ensemble prediction:

```
SHAP_ensemble(feature_i) = w_xgb · SHAP_xgb(feature_i) + w_lr · SHAP_lr(feature_i)
```

SHAP values are displayed as a bidirectional bar chart, with red bars indicating risk-increasing factors and blue bars indicating protective factors. The top 8 features by absolute SHAP value are shown.

### 6.2 LIME (Local Interpretable Model-Agnostic Explanations)

LIME (Ribeiro et al., 2016) generates a local linear approximation of the ensemble's decision boundary by perturbing the input and observing how predictions change. This provides a second, independent explanation that does not rely on model internals.

- **Perturbation samples**: 500
- **Features shown**: 8
- **Prediction function**: The weighted ensemble predict_proba function

LIME explanations are displayed alongside SHAP values to allow users to cross-reference the two methods and increase confidence in the explanations.

### 6.3 Risk Score and Risk Bands

The ensemble probability is converted to a percentage risk score (0–100%) and categorised into three risk bands:

| Risk Band | Score Range | UI Treatment |
|---|---|---|
| **LOW** | 0–29% | Green gauge, reassuring tone |
| **MODERATE** | 30–60% | Amber gauge, advisory tone |
| **HIGH** | 61–100% | Red gauge, urgent consultation recommendation |

### 6.4 Personalised Recommendations

The system generates up to five personalised lifestyle recommendations by analysing the SHAP values and identifying modifiable risk factors that are driving the score upward. Each recommendation includes:

- A plain-language action title
- A detailed explanation with specific targets (e.g., "Aim for below 130/80 mmHg")
- A clinical context note explaining the evidence base

Recommendations are sourced from established cardiovascular guidelines and are not generated by a language model.

---

## 7. Knowledge Graph

CardioXAI includes a CHD risk factor knowledge graph constructed with NetworkX and visualised with Pyvis. The graph encodes the medical relationships between the 16 input features based on established cardiovascular epidemiology.

| Property | Value |
|---|---|
| **Nodes** | 16 (one per risk factor) |
| **Edges** | 28 (clinically established relationships) |
| **Node categories** | Cardiovascular, Metabolic, Lifestyle, Health Status, Demographic |
| **Node shape** | Circle (modifiable) / Diamond (non-modifiable) |
| **Edge weight** | Relationship strength (0.0–1.0) |
| **Layout** | Force-directed (ForceAtlas2) |

The graph provides two views:

1. **SVG inline graph**: A lightweight, SHAP-weighted network embedded in the results page showing the user's top contributing factors and their interconnections.
2. **Interactive Pyvis graph**: A full 16-node interactive visualisation with hover tooltips, drag-to-rearrange, and physics-based layout.

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, Django 5.2, Django REST Framework |
| **ML Pipeline** | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| **Explainability** | SHAP 0.51, LIME 0.2 |
| **Knowledge Graph** | NetworkX 3.6, Pyvis 0.3 |
| **Data Processing** | pandas, NumPy |
| **Model Serialisation** | joblib |
| **Frontend** | Django Templates, Bootstrap 5.3, Bootstrap Icons |
| **Typography** | Public Sans (Google Fonts) |
| **Visualisation** | SVG (inline), Chart.js-compatible data, Pyvis (interactive) |

---

## 9. Project Structure

```
CardioXAI/
├── cardioxai/                  # Django project configuration
│   ├── settings.py             # Django settings
│   ├── urls.py                 # URL routing
│   ├── wsgi.py                 # WSGI entry point
│   └── asgi.py                 # ASGI entry point
│
├── core/                       # Django application
│   ├── views.py                # View functions (landing, about, assess, results, API)
│   ├── templates/core/         # Django templates
│   │   ├── base.html           # Base template (navbar, footer, shared CSS)
│   │   ├── landing.html        # Landing page with hero and feature cards
│   │   ├── about.html          # Project information and methodology
│   │   ├── assess.html         # 3-step assessment form with validation
│   │   └── results.html        # Results dashboard (gauge, SHAP, LIME, graph)
│   └── static/core/            # Static assets
│       └── knowledge_graph.html # Pyvis interactive knowledge graph
│
├── ml_pipeline/                # Machine learning pipeline
│   ├── prepare_data.py         # Dataset download, preprocessing, SMOTE, splitting
│   ├── train_models.py         # Model training, evaluation, ensemble construction
│   ├── inference.py            # Real-time prediction, SHAP, LIME, recommendations
│   └── knowledge_graph.py      # NetworkX graph construction and Pyvis export
│
├── models_store/               # Serialised models and metadata
│   ├── lr_model.joblib         # Trained Logistic Regression model
│   ├── xgb_model.joblib        # Trained XGBoost model
│   ├── ensemble_info.joblib    # Ensemble weights and member list
│   ├── feature_names.joblib    # Ordered feature name list
│   └── metrics.json            # Evaluation metrics (validation + test)
│
├── data/                       # Dataset directory (gitignored)
│   └── brfss2022.csv           # BRFSS dataset (downloaded by pipeline)
│
├── manage.py                   # Django management script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 10. Installation and Setup

### 10.1 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### 10.2 Clone the Repository

```bash
git clone https://github.com/Abdul-Salam15/CardioXAI.git
cd CardioXAI
```

### 10.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 10.4 Download Dataset and Train Models

The dataset is not included in the repository due to its size (22 MB). Run the pipeline to download and preprocess it:

```bash
# Step 1: Download and preprocess the BRFSS dataset
python ml_pipeline/prepare_data.py

# Step 2: Train all models and build ensemble
python ml_pipeline/train_models.py

# Step 3 (optional): Generate the Pyvis knowledge graph
python ml_pipeline/knowledge_graph.py
```

> **Note**: The pre-trained Logistic Regression and XGBoost models are included in `models_store/`. If you wish to use the system without retraining, you only need to run `prepare_data.py` to generate the `splits.joblib` file required by the SHAP/LIME explainers.

### 10.5 Run Database Migrations

```bash
python manage.py migrate
```

### 10.6 Start the Development Server

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`.

---

## 11. Usage

### 11.1 Web Interface

1. **Landing Page** (`/`): Overview of the system, feature highlights, and a link to the sample report.
2. **About Page** (`/about/`): Detailed project information, methodology, and disclaimer.
3. **Assessment Form** (`/assess/`): A guided 3-step form collecting demographics, medical history, and lifestyle data. Client-side validation ensures completeness before submission.
4. **Results Dashboard** (`/results/`): Displays the risk score (animated gauge), SHAP and LIME explanations, the risk factor knowledge graph, personalised recommendations, and a downloadable PDF-ready report.
5. **Demo Report** (`/demo/`): Pre-fills the assessment with a sample profile and displays the full results.

### 11.2 Downloadable Report

The results page includes a "Download Report" button that generates a print-optimised HTML report containing:

- All submitted answers
- Risk score and risk band
- Confidence tier
- Top SHAP-ranked risk drivers
- Personalised recommendations
- Next steps and disclaimer

---

## 12. API Reference

### `POST /api/predict/`

Accepts a JSON body with patient data and returns the full prediction response.

**Request:**

```json
{
  "age": "55-59",
  "sex": "Male",
  "bmi": 28.4,
  "genHealth": "Fair",
  "highBP": true,
  "highChol": true,
  "diabetes": false,
  "stroke": false,
  "diffWalk": false,
  "smoker": false,
  "heavyAlcohol": false,
  "physAct": false,
  "fruits": false,
  "veggies": true,
  "mentalHealth": 4,
  "physHealth": 6
}
```

**Response:**

```json
{
  "risk_score": 42.7,
  "risk_band": "MODERATE",
  "confidence_tier": "low",
  "disagreement": 0.4124,
  "entropy": 0.9845,
  "model_probas": {"lr": 0.6338, "rf": 0.175, "xgb": 0.2214},
  "model_preds": {"lr": 1, "rf": 0, "xgb": 0},
  "ensemble_proba": 0.4269,
  "ensemble_pred": 0,
  "shap_values": ["..."],
  "lime_explanation": ["..."],
  "recommendations": ["..."]
}
```

---

## 13. Limitations and Ethical Considerations

### 13.1 Limitations

- **Self-reported data**: All inputs are self-reported and may be subject to recall bias, social desirability bias, and measurement error. The system does not incorporate objective clinical measurements.
- **Cross-sectional design**: The BRFSS dataset is cross-sectional, capturing a snapshot rather than longitudinal trajectories. The system estimates association-based risk, not prospective prediction.
- **Population specificity**: The training data is drawn from US adults. Risk estimates may not generalise to populations with substantially different demographic, dietary, or healthcare profiles.
- **Class imbalance**: Despite SMOTE augmentation, the 12.2% positive-class prevalence affects the precision–recall trade-off. The system may over-predict risk for borderline cases.
- **Feature granularity**: Several features are binary simplifications of continuous variables (e.g., diabetes status vs. HbA1c levels), which limits the model's discriminatory resolution.

### 13.2 Ethical Considerations

- **Not a diagnostic tool**: CardioXAI is a research prototype and does not constitute medical advice, diagnosis, or treatment. It has not been clinically validated or registered as a medical device.
- **Informed consent**: Users are clearly informed of the system's limitations before, during, and after the assessment through prominent disclaimers.
- **Uncertainty transparency**: The confidence tier system ensures that unreliable predictions are explicitly flagged rather than presented as authoritative.
- **No data retention**: The system does not store user health data beyond the browser session. No personal health information is persisted to a database or transmitted to third parties.
- **Algorithmic fairness**: The system has not been formally audited for demographic parity, equalised odds, or calibration across protected groups. Future work should include a comprehensive fairness evaluation.

---

## 14. Future Work

- **Longitudinal validation**: Evaluate the system against prospective cohort data (e.g., UK Biobank) to assess predictive validity over time.
- **Fairness audit**: Conduct a formal algorithmic fairness analysis across demographic subgroups (age, sex, race/ethnicity) using metrics such as equalised odds and calibration.
- **Feature expansion**: Incorporate additional BRFSS features (e.g., sleep quality, diet patterns, mental health screening scores) and explore feature engineering.
- **Deep learning baselines**: Compare against neural network architectures (e.g., TabNet, TabTransformer) with attention-based interpretability.
- **Clinical expert evaluation**: Conduct a user study with healthcare professionals to evaluate the clinical utility and interpretability of the explanations.
- **Mobile-first redesign**: Develop a progressive web application (PWA) for broader accessibility on mobile devices.
- **Multilingual support**: Extend the interface and recommendations to support multiple languages for wider global reach.

---

## 15. References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

2. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

5. World Health Organization. (2021). *Cardiovascular diseases (CVDs) fact sheet*. Retrieved from https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

6. Centers for Disease Control and Prevention. (2023). *Behavioral Risk Factor Surveillance System*. Retrieved from https://www.cdc.gov/brfss/

7. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

## 16. Licence

This project is developed for academic research purposes. All rights reserved.

---

**Disclaimer**: CardioXAI is a research prototype developed as an undergraduate academic project. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional regarding your cardiovascular health.
