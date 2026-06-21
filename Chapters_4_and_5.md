CHAPTER FOUR

RESULTS AND DISCUSSION

4.0 Introduction

This chapter presents the results obtained from the development, training, and evaluation of the CardioXAI framework. It is organized into eight sections corresponding to the methodological stages described in Chapter Three. Section 4.1 reports the outcomes of data preprocessing and exploratory data analysis. Section 4.2 presents the performance of the three individual classification models (Logistic Regression, Random Forest, and XGBoost) on both validation and test subsets. Section 4.3 describes the construction and evaluation of the consensus ensemble. Section 4.4 reports the outputs of the multi-layer explainability framework, including SHAP attributions, LIME cross-validation, and the CHD Risk Factor Knowledge Graph. Section 4.5 presents the uncertainty quantification and confidence tier classification results. Section 4.6 reports the calibration analysis of the ensemble model. Section 4.7 describes the deployed web application and its integration of all framework components. Section 4.8 discusses the findings in the context of the existing literature and the study objectives.


4.1 Data Preprocessing and Exploratory Data Analysis Results

4.1.1 Dataset Summary and Cleaning

The Pytlak CDC BRFSS 2022 Heart Disease Health Indicators dataset was loaded with a raw shape of 253,680 observations. From the original feature set, 16 self-reported behavioral, demographic, and health status features were retained for model training, consistent with the at-home assessment objective of excluding laboratory-dependent variables. The binary target variable HeartDiseaseorAttack indicates self-reported history of myocardial infarction or coronary heart disease. Following the removal of duplicate records and observations with missing values, the cleaned dataset comprised 186,487 unique records. No additional imputation was required, as the cleaned subset contained no missing values by construction. Table 4.1 summarises the 16 input features retained for modelling.

Table 4.1: Features Retained for Model Training (16 Features)

| Feature            | Type       | Description                                             |
|--------------------|------------|---------------------------------------------------------|
| HighBP             | Binary     | Self-reported high blood pressure diagnosis             |
| HighChol           | Binary     | Self-reported high cholesterol diagnosis                |
| BMI                | Continuous | Body Mass Index (kg/m²)                                 |
| Smoker             | Binary     | Smoked at least 100 cigarettes in lifetime              |
| Stroke             | Binary     | Self-reported history of stroke                         |
| Diabetes           | Binary     | Self-reported diabetes (Type 1 or Type 2)               |
| PhysActivity       | Binary     | Physical activity in past 30 days (outside work)        |
| Fruits             | Binary     | Consumes fruit daily                                    |
| Veggies            | Binary     | Consumes vegetables daily                               |
| HvyAlcoholConsump  | Binary     | Heavy alcohol consumption                               |
| GenHlth            | Ordinal    | Self-rated general health (1=Excellent to 5=Poor)       |
| MentHlth           | Continuous | Days of poor mental health in past 30 days (0-30)       |
| PhysHlth           | Continuous | Days of poor physical health in past 30 days (0-30)     |
| DiffWalk           | Binary     | Difficulty walking or climbing stairs                   |
| Sex                | Binary     | Biological sex (0=Female, 1=Male)                       |
| Age                | Ordinal    | Age group in 5-year bands (1=18-24 to 13=80+)          |


4.1.2 Class Distribution

The target variable exhibited a pronounced class imbalance. Of the 186,487 cleaned observations, 163,741 (87.8%) were classified as CHD-negative (HeartDiseaseorAttack = 0) and 22,746 (12.2%) as CHD-positive (HeartDiseaseorAttack = 1). This 87.8/12.2 distribution reflects the population-level prevalence of self-reported coronary heart disease among US adults captured by the BRFSS survey and is consistent with published epidemiological estimates (CDC, 2024; Martin et al., 2024). The degree of imbalance is less extreme than the approximately 5% positive rate reported in some BRFSS subsets, which may reflect differences in the specific Pytlak cleaning and variable selection criteria applied to this version of the dataset.


4.1.3 Data Splitting and SMOTE Application

The cleaned dataset was partitioned into training, validation, and test subsets using a stratified 70/15/15 split, preserving the 87.8/12.2 class ratio across all three subsets. The resulting subset sizes were:

- Training subset: 130,540 observations
- Validation subset: 27,973 observations
- Test subset: 27,974 observations

Synthetic Minority Over-sampling Technique (SMOTE) was applied exclusively to the training subset after splitting, in accordance with the methodology described in Section 3.3. SMOTE generated synthetic positive-class observations through interpolation between existing minority-class samples, producing a perfectly balanced training distribution:

- Before SMOTE: 130,540 observations (87.8% negative / 12.2% positive)
- After SMOTE: 229,188 observations (50.0% negative / 50.0% positive)

The validation and test subsets were left unmodified to ensure that all performance metrics reflect evaluation on the original, naturally imbalanced data distribution. This design prevents the optimistic bias that would result from evaluating on synthetically balanced data (Chawla et al., 2002).


4.2 Individual Model Performance

4.2.1 Model Training Configuration

Three supervised classification models were trained on the SMOTE-balanced training subset:

1. Logistic Regression: Implemented using scikit-learn's LogisticRegression class with L-BFGS solver, L2 regularisation (default), and a maximum of 1,000 iterations. Logistic Regression serves as the parametric baseline model whose linear coefficients provide a direct interpretability reference for the SHAP attributions generated by the more complex models.

2. Random Forest: Implemented using scikit-learn's RandomForestClassifier with 200 estimators, a maximum tree depth of 15, a minimum of 5 samples per split, and parallel computation (n_jobs = -1). Random state was set to 42 for reproducibility.

3. XGBoost: Implemented using the xgboost Python library's XGBClassifier with 200 estimators, a maximum tree depth of 6, a learning rate of 0.1, and log-loss as the evaluation metric. Random state was set to 42 for reproducibility.

All models were trained on the same SMOTE-balanced training subset and evaluated on both the unmodified validation and test subsets to ensure fair comparison.


4.2.2 Validation Set Results

Table 4.2 presents the performance of all three individual models on the held-out validation subset (27,973 observations, original class distribution preserved).

Table 4.2: Individual Model Performance on Validation Subset

| Model                | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.7318   | 0.2784    | 0.7512 | 0.4063   | 0.8150  |
| Random Forest        | 0.8575   | 0.3994    | 0.3304 | 0.3616   | 0.8131  |
| XGBoost              | 0.8787   | 0.5112    | 0.1601 | 0.2438   | 0.8204  |

Several key findings emerge from the validation results:

First, all three models achieved AUC-ROC values exceeding 0.81, demonstrating good discriminatory ability across all classification thresholds. XGBoost achieved the highest validation AUC-ROC of 0.8204, followed closely by Logistic Regression at 0.8150 and Random Forest at 0.8131. The narrow spread of AUC-ROC values (range: 0.0073) indicates that all three model architectures extract comparable discriminatory information from the self-reported behavioral features.

Second, the three models exhibit markedly different precision-recall trade-offs, reflecting their distinct inductive biases:

- Logistic Regression achieves the highest recall (0.7512), correctly identifying 75.1% of CHD-positive individuals, but at the cost of low precision (0.2784), meaning that 72.2% of its positive predictions are false positives. This high-sensitivity, low-specificity profile is characteristic of linear models operating on imbalanced data with overlapping class distributions.

- XGBoost achieves the highest precision (0.5112), meaning that 51.1% of its positive predictions are correct, but with the lowest recall (0.1601), capturing only 16.0% of actual CHD-positive cases. This conservative prediction behavior reflects XGBoost's gradient boosting architecture, which tends to optimize for majority-class accuracy on threshold-sensitive metrics.

- Random Forest occupies an intermediate position with moderate precision (0.3994) and recall (0.3304), achieving neither the sensitivity of Logistic Regression nor the precision of XGBoost.

Third, the complementary precision-recall profiles of XGBoost (high precision, low recall) and Logistic Regression (high recall, low precision) provide the theoretical basis for their combination into a consensus ensemble, as each model compensates for the other's primary weakness.


4.2.3 Test Set Results

Table 4.3 presents the performance of all three individual models on the held-out test subset (27,974 observations), which was not used during model training, hyperparameter selection, or ensemble construction.

Table 4.3: Individual Model Performance on Test Subset

| Model                | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.7297   | 0.2762    | 0.7486 | 0.4035   | 0.8112  |
| Random Forest        | 0.8583   | 0.3999    | 0.3205 | 0.3558   | 0.8043  |
| XGBoost              | 0.8792   | 0.5177    | 0.1586 | 0.2428   | 0.8117  |

The test set results confirm the patterns observed on the validation subset with minimal degradation. The maximum absolute difference between validation and test AUC-ROC across all models is 0.0088 (Random Forest: 0.8131 validation versus 0.8043 test), indicating stable generalisation and no evidence of overfitting. The consistency of precision-recall profiles between validation and test subsets further confirms the robustness of each model's learned decision boundaries.


4.3 Consensus Ensemble Construction and Evaluation

4.3.1 Ensemble Member Selection

Following the model selection procedure described in Section 3.7, the two models achieving the highest validation AUC-ROC were selected for the consensus ensemble:

1. XGBoost: Validation AUC-ROC = 0.8204
2. Logistic Regression: Validation AUC-ROC = 0.8150

Random Forest (validation AUC-ROC = 0.8131) was excluded from the ensemble as the third-ranked model. Although the difference in validation AUC-ROC between Logistic Regression and Random Forest is small (0.0019), the complementary precision-recall profiles of XGBoost and Logistic Regression provide a stronger theoretical basis for ensemble combination than XGBoost and Random Forest, whose recall profiles are more similar.


4.3.2 Ensemble Weight Computation

Ensemble weights were computed proportionally to each model's validation AUC-ROC, following the formula described in Section 3.7:

w_XGB = AUC_ROC_XGB / (AUC_ROC_XGB + AUC_ROC_LR) = 0.8204 / (0.8204 + 0.8150) = 0.5017

w_LR = AUC_ROC_LR / (AUC_ROC_XGB + AUC_ROC_LR) = 0.8150 / (0.8204 + 0.8150) = 0.4983

The near-equal weighting (50.17% XGBoost, 49.83% Logistic Regression) reflects the closely matched discriminatory performance of the two models. The consensus ensemble probability for any given input is computed as:

P_ensemble = 0.5017 * P_XGB + 0.4983 * P_LR

A classification threshold of 0.5 is applied to the ensemble probability to produce the binary prediction.


4.3.3 Ensemble Performance

Table 4.4 presents the consensus ensemble performance on both validation and test subsets.

Table 4.4: Consensus Ensemble Performance

| Subset     | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|------------|----------|-----------|--------|----------|---------|
| Validation | 0.8388   | 0.3768    | 0.4884 | 0.4254   | 0.8203  |
| Test       | 0.8391   | 0.3724    | 0.4627 | 0.4126   | 0.8150  |

The ensemble achieves the highest F1 score among all models on both validation (0.4254) and test (0.4126) subsets, confirming the value of combining complementary models. The ensemble F1 score exceeds the best individual model F1 score (Logistic Regression: 0.4063 validation, 0.4035 test) by approximately 5% on both evaluation subsets.

The ensemble successfully balances the precision-recall trade-off by combining the high-recall Logistic Regression component (recall = 0.7512) with the high-precision XGBoost component (precision = 0.5112), producing an intermediate recall of 0.4884 (validation) and precision of 0.3768 (validation). This trade-off is appropriate for a consumer-facing screening tool, where both missed cases (false negatives) and unnecessary alarm (false positives) carry potential harm.

The ensemble AUC-ROC on the test set (0.8150) is virtually identical to the individual XGBoost test AUC-ROC (0.8117) and Logistic Regression test AUC-ROC (0.8112), indicating that the ensemble preserves the discriminatory ability of its component models while improving threshold-dependent metrics.

Table 4.5 provides a comprehensive side-by-side comparison of all four models on the test subset.

Table 4.5: Complete Model Comparison on Test Subset

| Model                | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.7297   | 0.2762    | 0.7486 | 0.4035   | 0.8112  |
| Random Forest        | 0.8583   | 0.3999    | 0.3205 | 0.3558   | 0.8043  |
| XGBoost              | 0.8792   | 0.5177    | 0.1586 | 0.2428   | 0.8117  |
| Consensus Ensemble   | 0.8391   | 0.3724    | 0.4627 | 0.4126   | 0.8150  |


4.4 Explainability Framework Results

4.4.1 SHAP (SHapley Additive exPlanations) Results

SHAP was implemented using two model-specific explainers: shap.TreeExplainer for the XGBoost model, which computes exact Shapley values for tree-based models in polynomial time (Lundberg et al., 2020), and shap.LinearExplainer for the Logistic Regression model, which leverages the linear structure of the model to compute exact feature attributions. Ensemble SHAP values were computed as the weighted average of individual model SHAP values using the same AUC-ROC-proportional weights applied to the ensemble predictions (w_XGB = 0.5017, w_LR = 0.4983).

For each individual prediction, the framework computes SHAP values for all 16 features and presents the top 8 by absolute magnitude. Each SHAP value indicates the direction (risk-increasing if positive, protective if negative) and magnitude of that feature's contribution to the specific prediction, relative to the model's baseline output.

The SHAP outputs for each prediction are rendered in the CardioXAI results page as:

1. A bidirectional bar chart displaying the top 8 features ranked by absolute SHAP value, with red bars indicating risk-increasing contributions and blue bars indicating protective contributions (see Figure 4.4).

2. A plain-language summary identifying the top positive SHAP contributors as the user's primary behavioral CHD risk drivers, rendered as a bulleted list beneath the bar chart.

At the global level, SHAP analysis across the test set consistently identified the following features as the most influential risk drivers, consistent with established cardiovascular epidemiology:

- Age: The highest-impact feature, reflecting the well-documented non-modifiable risk amplification effect of advancing age on all cardiovascular outcomes (Visseren et al., 2021).
- General Health (GenHlth): Self-rated general health emerged as a strong predictor, consistent with prospective studies demonstrating that individuals who rate their health as poor or fair have significantly higher CHD event rates independent of clinical measurements (Pierannunzi et al., 2013).
- High Blood Pressure (HighBP): Hypertension is the leading modifiable risk factor for coronary heart disease, and its prominence in SHAP attributions aligns with the PURE study findings and the 2021 ESC prevention guidelines (Yusuf et al., 2020; Visseren et al., 2021).
- BMI: Higher BMI values consistently received positive SHAP values, reflecting the established association between obesity and cardiovascular risk through increased cardiac workload and metabolic strain.
- Difficulty Walking (DiffWalk): Mobility limitation appeared as a significant risk indicator, reflecting both functional impairment and the downstream effects of physical deconditioning on cardiovascular health.
- Diabetes: Self-reported diabetes status received consistently high SHAP attributions, consistent with the established finding that diabetes doubles coronary heart disease risk through vascular damage (Mensah et al., 2023).

Among protective factors, Physical Activity (PhysActivity) and daily Fruit and Vegetable consumption (Fruits, Veggies) consistently received negative SHAP values, indicating that these behavioral factors reduce the model's CHD risk estimate. This finding aligns with the PURE study evidence that modifiable behavioral factors explain a large proportion of CHD events across income settings (Yusuf et al., 2020).


4.4.2 LIME (Local Interpretable Model-Agnostic Explanations) Results

LIME was implemented using the lime Python library's LimeTabularExplainer class as a model-agnostic complement to SHAP. For each prediction, LIME generates 500 perturbed samples in the neighbourhood of the input instance, obtains the consensus ensemble's predictions on those samples, and fits a locally weighted linear model to produce feature attribution weights. The top 8 features by absolute LIME weight are displayed for each prediction (see Figure 4.5).

LIME serves as an internal cross-validation mechanism for the SHAP explanations, following the approach validated by Rezk et al. (2024). For each prediction, the framework generates both SHAP and LIME explanations independently, allowing users and system operators to assess whether the two explanation methods converge on the same primary risk drivers.

In testing across representative predictions, LIME attributions showed strong convergence with SHAP for the top-ranked features. Age, General Health, High Blood Pressure, BMI, and Diabetes consistently appeared among the top LIME features when they also appeared among the top SHAP features for the same prediction. This cross-method consistency provides additional confidence that the identified risk drivers reflect genuine patterns learned by the model rather than artifacts of a single explanation method (Rezk et al., 2024).

Minor divergences between SHAP and LIME attributions were observed in the ordering and magnitude of lower-ranked features (typically features ranked 5th through 8th). These differences are expected given the distinct theoretical foundations of the two methods: SHAP computes exact Shapley values based on marginal contributions across all feature coalitions, while LIME fits a local linear approximation using perturbed samples in the input neighbourhood. The stochastic perturbation process in LIME introduces sampling variability that primarily affects features with smaller absolute contributions (Garreau & von Luxburg, 2020). This observation supports the design decision described in Section 3.8.2 to position LIME as an internal validation mechanism rather than the primary user-facing explanation engine.


4.4.3 CHD Risk Factor Knowledge Graph Results

The CHD Risk Factor Knowledge Graph was constructed using NetworkX and visualised interactively using Pyvis, as described in Section 3.8.3. The graph comprises 16 nodes (one per input feature) and 29 clinically established edges representing known relationships between risk factors.

Graph Structure:

Nodes are categorised into five groups, each assigned a distinct colour for visual differentiation:

- Cardiovascular (red, #B71C1C): HighBP, HighChol, Stroke
- Metabolic (orange, #E65100): BMI, Diabetes
- Lifestyle (green, #2E7D32): Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump
- Health Status (blue, #1A73E8): GenHlth, MentHlth, PhysHlth, DiffWalk
- Demographic (grey, #6b7280): Sex, Age

Node shape encodes modifiability: circular nodes represent modifiable risk factors (13 of 16 features), while diamond-shaped nodes represent non-modifiable factors (Age, Sex, Stroke). Node size is scaled proportionally to degree centrality within the graph, ensuring that the most interconnected risk factors are visually prominent.

Edge Relationships:

The 29 edges encode clinically established relationships between CHD risk factors, each weighted between 0.4 and 0.9 to reflect the strength of the association. Key high-weight relationships include:

- HighBP to Stroke (weight = 0.9): Hypertension is the leading cause of stroke, and the co-presence of these factors indicates severe vascular compromise.
- HighBP to HighChol (weight = 0.8): Both factors contribute to atherosclerosis, the primary pathological mechanism underlying CHD.
- BMI to Diabetes (weight = 0.8): Obesity is the most significant modifiable risk factor for Type 2 diabetes.
- Fruits to Veggies (weight = 0.8): Both represent dietary protective factors that tend to co-occur and share cardiovascular benefits.
- GenHlth to PhysHlth (weight = 0.8): Physical health status and self-rated general health are strongly correlated indicators.

Personalised Graph Visualisation:

For each individual prediction, the Knowledge Graph is personalised by overlaying the user's SHAP-identified risk drivers onto the graph structure. Nodes corresponding to features with positive SHAP values (risk-increasing) are annotated with their SHAP contribution direction and magnitude, enabling users to see not only which individual behaviors drive their risk estimate but how those behaviors relate to and interact with each other within the broader risk factor network. This graph-level insight transforms isolated feature attribution scores into a connected risk narrative, going beyond what a ranked list of SHAP values or LIME coefficients can convey in isolation. The personalised knowledge graph as rendered in the CardioXAI application is shown in Figure 4.6.

The interactive Pyvis visualisation (Figure 4.7) supports hover-to-inspect functionality, displaying each node's full feature name, clinical description, and modifiability status. Users can drag nodes to rearrange the layout and zoom to explore subregions of the graph. The force-directed ForceAtlas2 layout algorithm produces a spatially coherent arrangement in which tightly connected risk factor clusters (e.g., the cardiovascular cluster of HighBP, HighChol, and Stroke; the metabolic cluster of BMI and Diabetes) are positioned closer together, visually communicating the systemic nature of CHD risk.


4.5 Uncertainty Quantification Results

4.5.1 Confidence Tier Classification

The uncertainty quantification engine computes two complementary metrics for each prediction:

1. Inter-model Disagreement: The absolute difference between the XGBoost and Logistic Regression probability estimates:

   Disagreement = |P_XGB - P_LR|

2. Predictive Entropy: The Shannon binary entropy of the consensus ensemble probability:

   H = -P_ensemble * log2(P_ensemble) - (1 - P_ensemble) * log2(1 - P_ensemble)

   Entropy ranges from 0 (complete certainty) to 1.0 (maximum uncertainty, at P = 0.5).

These two metrics are combined to classify each prediction into one of three confidence tiers:

- High Confidence: Disagreement <= 0.15 AND Entropy <= 0.85. Both models agree closely, and the ensemble probability is sufficiently distant from the decision boundary to warrant high confidence in the prediction.

- Medium Confidence: Disagreement <= 0.25 AND Entropy <= 0.95. Models show moderate agreement with some uncertainty, warranting caution but not blocking risk score display.

- Low Confidence: Disagreement > 0.25 OR Entropy > 0.95. Substantial disagreement between models or extreme proximity to the decision boundary indicates that the prediction is insufficiently reliable for unsupervised interpretation. Low-confidence predictions trigger a distinct user interface pathway that routes users to a physician consultation prompt rather than presenting a potentially misleading risk score (see Figure 4.9).

The three-tier system operationalises the ethical principle articulated by Kompa et al. (2021) that communicating prediction uncertainty is an ethical obligation in medical ML systems, preventing the false impression of model reliability that can lead to harmful decisions.

4.5.2 Confidence Tier Behaviour

Analysis of the confidence tier distribution reveals that predictions at the extremes of the probability spectrum (very low or very high risk) tend to achieve high confidence, as both XGBoost and Logistic Regression agree on the direction and approximate magnitude of the risk estimate. Predictions near the decision boundary (ensemble probability near 0.5) tend to fall into medium or low confidence tiers, as the two models' complementary biases produce the largest disagreement in this region.

As an illustrative example, a test case with the following profile — high blood pressure, high cholesterol, daily vegetable consumption, male, age category 6 (45-49) — produced:

- XGBoost probability: 0.221
- Logistic Regression probability: 0.634
- Disagreement: |0.221 - 0.634| = 0.413
- Ensemble probability: 0.5017(0.221) + 0.4983(0.634) = 0.427
- Entropy: -0.427 * log2(0.427) - 0.573 * log2(0.573) = 0.985
- Confidence tier: Low (disagreement > 0.25 and entropy > 0.95)

In this case, the large disagreement between the two models and the near-maximum entropy appropriately trigger the low-confidence pathway, routing the user to a physician consultation prompt rather than displaying a potentially misleading risk score. This example illustrates a scenario where the two models have learned genuinely different risk assessments from the same input profile, and presenting a single definitive risk score would be misleading.


4.6 Calibration Analysis

Calibration was assessed using a 10-bin calibration curve computed on the validation subset. Table 4.6 presents the calibration results.

Table 4.6: Ensemble Calibration Curve Data (10 Bins, Validation Subset)

| Bin | Mean Predicted Probability | Observed Fraction of Positives |
|-----|----------------------------|-------------------------------|
| 1   | 0.0475                     | 0.0137                        |
| 2   | 0.1474                     | 0.0421                        |
| 3   | 0.2481                     | 0.0949                        |
| 4   | 0.3478                     | 0.1406                        |
| 5   | 0.4470                     | 0.2355                        |
| 6   | 0.5480                     | 0.3069                        |
| 7   | 0.6447                     | 0.3907                        |
| 8   | 0.7409                     | 0.4956                        |
| 9   | 0.8313                     | 0.7153                        |
| 10  | 0.9067                     | 0.6667                        |

The calibration curve reveals that the ensemble model systematically overestimates CHD risk relative to the observed frequency of positive cases across most probability bins. In a perfectly calibrated model, the observed fraction of positives would equal the mean predicted probability in each bin (i.e., the calibration curve would follow the diagonal). The observed pattern shows that the ensemble predicts higher probabilities than are warranted by the actual outcome frequencies in the data.

This overestimation is most pronounced in the mid-to-high probability bins (bins 5-8), where, for example, a mean predicted probability of 0.5480 (bin 6) corresponds to an observed positive rate of only 0.3069. In bin 10, the mean predicted probability of 0.9067 corresponds to an observed positive rate of 0.6667, representing a significant overestimation.

The calibration pattern is attributable primarily to the SMOTE resampling applied during training. SMOTE creates a balanced training distribution (50/50) from an originally imbalanced dataset (87.8/12.2), which shifts the learned decision boundary toward more positive predictions. While this shift improves recall (sensitivity to positive cases), it inflates predicted probabilities relative to the true population prevalence of CHD. This is a known limitation of SMOTE-trained models and does not affect the model's discriminatory ability (AUC-ROC), which is invariant to monotonic probability transformations (Chawla et al., 2002).

For the CardioXAI self-assessment context, the overestimation is partially mitigated by the three-tier risk banding system (LOW: 0-29%, MODERATE: 30-60%, HIGH: 61-100%), which maps continuous probabilities to categorical risk labels. Users receive a risk category rather than a raw probability, and the system's recommendation engine operates on SHAP attributions rather than absolute probability values. Nevertheless, future iterations of CardioXAI should consider applying Platt scaling or isotonic regression as post-hoc calibration methods to improve the alignment between predicted probabilities and true positive rates (Platt, 1999).


4.7 Web Application Deployment

The complete CardioXAI framework was deployed as a Django web application, integrating all analytical components into a single consumer-facing interface accessible through any standard web browser. The application implements a four-page architecture:

4.7.1 Landing Page

The landing page introduces the CardioXAI system through a hero section communicating the framework's core value proposition: transparent, explainable CHD risk assessment using only self-reported behavioral data. A sample risk score card is embedded in the hero section, displaying a representative 47% risk score with conic-gradient gauge animation, providing visitors with an immediate visual preview of the results output format. Feature cards highlight the key differentiating capabilities of the system: SHAP-based explainability, uncertainty-aware predictions, and personalised lifestyle recommendations. A "How It Works" section describes the three-step process (enter health details, receive analysis, explore results) in a visually segmented layout. A sample report link allows visitors to preview the full results output before completing an assessment.

[INSERT FIGURE 4.1 HERE: Screenshot of the CardioXAI landing page showing the hero section with the sample 47% risk score card, feature cards (SHAP explainability, uncertainty-aware predictions, lifestyle recommendations), and the "How It Works" section.]

Figure 4.1: CardioXAI Landing Page

4.7.2 Assessment Page

The assessment page implements a three-step form collecting the 16 input features:

- Step 1 (Demographics): Age category, biological sex, and BMI.
- Step 2 (Medical History): Self-reported high blood pressure, high cholesterol, stroke, diabetes, and difficulty walking.
- Step 3 (Lifestyle): Smoking status, physical activity, daily fruit and vegetable consumption, heavy alcohol consumption, general health rating, and days of poor mental and physical health.

Client-side validation ensures that required fields are completed and that numeric inputs (BMI, health days) fall within physiologically plausible ranges. The form design follows a progressive disclosure pattern, preventing cognitive overload by presenting related questions in grouped steps rather than as a single long form.

[INSERT FIGURE 4.2 HERE: Screenshot of the CardioXAI assessment page showing Step 1 (Demographics) with age category selector, biological sex selector, and BMI input field. Include the step progress indicator showing Step 1 of 3.]

Figure 4.2: CardioXAI Assessment Page — Step 1 (Demographics)

[INSERT FIGURE 4.3 HERE: Screenshot of the CardioXAI assessment page showing Step 2 (Medical History) with toggle or checkbox inputs for high blood pressure, high cholesterol, stroke, diabetes, and difficulty walking.]

Figure 4.3: CardioXAI Assessment Page — Step 2 (Medical History)

[INSERT FIGURE 4.4 HERE: Screenshot of the CardioXAI assessment page showing Step 3 (Lifestyle) with inputs for smoking status, physical activity, fruit and vegetable consumption, heavy alcohol consumption, general health rating, and mental/physical health days.]

Figure 4.4: CardioXAI Assessment Page — Step 3 (Lifestyle)

4.7.3 Results Page

The results page presents the complete output of the CardioXAI inference engine, integrating all framework components into a vertically scrolling layout. Figure 4.5 shows the risk score gauge and confidence tier badge at the top of the results page.

1. Risk Score and Gauge: A conic-gradient animated gauge displays the ensemble CHD risk score as a percentage, colour-coded by risk band (green for LOW, amber for MODERATE, red for HIGH). The risk band label (e.g., "MODERATE RISK") is displayed prominently within the gauge alongside the numerical score. The gauge is implemented as a CSS conic-gradient arc with JavaScript-driven animation that sweeps from zero to the predicted risk score on page load.

2. Confidence Tier Badge: The prediction's confidence tier (High, Medium, or Low) is displayed as a badge beneath the gauge, using a checkmark icon for High confidence and a warning icon for Low confidence. Low-confidence predictions trigger a distinct display pathway described in Section 4.7.5.

[INSERT FIGURE 4.5 HERE: Screenshot of the CardioXAI results page showing the animated risk score gauge displaying the percentage score with the risk band label (e.g., "MODERATE RISK") inside the gauge, and the confidence tier badge (e.g., "High Confidence") beneath it.]

Figure 4.5: CardioXAI Results Page — Risk Score Gauge and Confidence Tier Badge

3. SHAP Explanation Chart: A bidirectional horizontal bar chart displays the top features ranked by absolute SHAP value. Red bars extending rightward indicate risk-increasing factors, and blue bars extending leftward indicate protective factors. Each bar is labelled with the human-readable feature name and its numerical SHAP contribution value. Below the chart, a bulleted plain-language summary identifies the user's primary risk drivers and protective factors.

[INSERT FIGURE 4.6 HERE: Screenshot of the SHAP explanation section from the CardioXAI results page, showing the bidirectional bar chart with red bars for risk-increasing factors (e.g., "High BP +0.31", "Cholesterol +0.22") and blue bars for protective factors (e.g., "Vegetables -0.09"), followed by the bulleted plain-language summary.]

Figure 4.6: CardioXAI Results Page — SHAP Feature Attribution Chart

4. LIME Explanation Panel: The top 8 LIME feature attributions are displayed as a secondary explanation layer, rendered as horizontal bars with feature condition labels (e.g., "Age > 9", "GenHlth > 3"). This panel provides independent confirmation of the SHAP-identified risk drivers using a distinct explanation methodology.

[INSERT FIGURE 4.7 HERE: Screenshot of the LIME explanation section from the CardioXAI results page, showing the horizontal bar chart with feature condition labels and LIME attribution weights.]

Figure 4.7: CardioXAI Results Page — LIME Feature Attribution Panel

5. Personalised Knowledge Graph (SVG): An inline SVG knowledge graph displays the user's SHAP-identified risk factors as colour-coded, interconnected nodes. Node size is proportional to the absolute SHAP value, and nodes are coloured by risk factor category (cardiovascular in red, metabolic in orange, lifestyle in green, health status in blue, demographic in grey). Risk-increasing nodes are annotated with their SHAP value (e.g., "SHAP +0.31"), while protective nodes display negative values (e.g., "SHAP -0.09"). Edges connect related risk factors, visually communicating the systemic, interconnected nature of the user's risk profile.

[INSERT FIGURE 4.8 HERE: Screenshot of the personalised SVG knowledge graph from the CardioXAI results page, showing colour-coded nodes of varying sizes connected by edges. Nodes should display labels such as "High BP SHAP +0.31", "Cholesterol SHAP +0.22", "BMI SHAP +0.16", "Age SHAP non-mod.", "Inactivity SHAP +0.11", and "Vegetables SHAP -0.09".]

Figure 4.8: CardioXAI Results Page — Personalised Risk Factor Knowledge Graph (SVG)

6. Interactive Pyvis Knowledge Graph: A link beneath the inline SVG opens the full interactive Pyvis knowledge graph in a new page. This graph displays all 16 risk factor nodes and 29 clinical edges with hover-to-inspect tooltips showing each node's full description, clinical significance, and modifiability status. The force-directed ForceAtlas2 layout algorithm spatially clusters related risk factors, providing an exploration-oriented view of the complete CHD risk factor network.

[INSERT FIGURE 4.9 HERE: Screenshot of the full interactive Pyvis knowledge graph page showing all 16 risk factor nodes and 29 edges. Nodes are colour-coded by category with the ForceAtlas2 force-directed layout. Include a tooltip popup showing node details if possible.]

Figure 4.9: CardioXAI Interactive Pyvis Knowledge Graph (Full View)

7. Personalised Recommendations: Up to 5 evidence-based lifestyle recommendations are generated based on the user's SHAP-identified modifiable risk factors. Each recommendation card includes an icon, an action title (e.g., "Increase Physical Activity", "Monitor Your Blood Pressure"), a detailed explanation with specific behavioral targets (e.g., "Aim for at least 150 minutes of moderate aerobic activity per week"), and a clinical evidence note linking the recommendation to established cardiovascular guidelines.

[INSERT FIGURE 4.10 HERE: Screenshot of the personalised recommendations section from the CardioXAI results page, showing the recommendation cards with icons, action titles, detailed explanations, and clinical evidence notes.]

Figure 4.10: CardioXAI Results Page — Personalised Lifestyle Recommendations

8. Medical Disclaimer: A prominent disclaimer is displayed on all results pages, explicitly stating that CardioXAI is an informational self-assessment tool and is not intended to replace clinical diagnosis or medical advice.

4.7.4 Downloadable Report

The results page includes a "Download Report" button that generates a printer-friendly version of the complete results output. The printable report includes the risk score, risk band, SHAP and LIME explanations, the knowledge graph, and personalised recommendations, formatted for PDF generation through the browser's native print-to-PDF functionality.

4.7.5 Low-Confidence Prediction Handling

When the uncertainty quantification engine classifies a prediction as low confidence (inter-model disagreement > 0.25 or predictive entropy > 0.95), the results page renders a distinct view. The risk score gauge is de-emphasised, the confidence tier badge displays a prominent "Low Confidence" warning, and a clearly worded advisory message directs the user to consult a physician rather than relying on the potentially unreliable risk estimate. The submitted assessment answers are displayed for the user's reference, but the full SHAP, LIME, and knowledge graph outputs are withheld to prevent interpretation of unreliable explanations.

[INSERT FIGURE 4.11 HERE: Screenshot of the CardioXAI results page for a low-confidence prediction, showing the de-emphasised or hidden risk score, the "Low Confidence" warning badge, and the physician consultation advisory message.]

Figure 4.11: CardioXAI Results Page — Low-Confidence Prediction with Physician Consultation Routing

4.7.6 API Endpoint

A JSON API endpoint (POST /api/predict/) is provided for programmatic access to the CardioXAI prediction engine. The API accepts the same input features as the web form and returns a comprehensive JSON response containing the risk score, risk band, confidence tier, all model probabilities, SHAP values, LIME attributions, and personalised recommendations. This endpoint enables integration with third-party health applications and supports future research use.


4.8 Discussion

4.8.1 Model Performance in Context

The consensus ensemble achieved a test AUC-ROC of 0.8150, which exceeds the 0.80 threshold generally considered clinically acceptable for cardiovascular risk prediction models (Weng et al., 2017; Krittanawong et al., 2020). This result is broadly consistent with the published benchmarks on BRFSS-derived CHD datasets. Hasnat et al. (2025) reported a test AUC of 0.8371 using a weighted gradient boosting ensemble on the BRFSS Heart Disease Indicators dataset, and Tompra et al. (2024) achieved comparable performance using XGBoost and Random Forest. The somewhat lower AUC-ROC achieved by CardioXAI relative to Hasnat et al. (2025) may reflect differences in feature set size, dataset version, preprocessing decisions, or the specific trade-off introduced by the consensus ensemble design.

Importantly, direct AUC-ROC comparison with Rezk et al. (2024), who achieved 0.92 using a SHAP and LIME-augmented voting ensemble, is not appropriate, as that study used a small clinical dataset (approximately 1,000 records) with laboratory features, which generally yield higher discriminatory performance than self-reported behavioral features. The CardioXAI results demonstrate that clinically acceptable discriminatory performance can be achieved using exclusively self-reported data, validating the feasibility of the laboratory-free design objective.


4.8.2 Explainability Findings in Clinical Context

The SHAP-identified top risk drivers, including Age, General Health, High Blood Pressure, BMI, Diabetes, and Difficulty Walking, are highly consistent with established cardiovascular risk factor hierarchies in the clinical literature. The PURE study identified smoking, poor diet, physical inactivity, and abdominal obesity as the dominant behavioral drivers of CHD across all income settings (Yusuf et al., 2020). The 2021 ESC guidelines identify hypertension, diabetes, dyslipidemia, and smoking as the primary modifiable risk factors for atherosclerotic cardiovascular disease (Visseren et al., 2021). The alignment between SHAP attributions and established clinical knowledge confirms that the CardioXAI models have learned epidemiologically coherent patterns rather than spurious correlations.

The convergence between SHAP and LIME attributions for top-ranked features replicates the cross-method consistency reported by Rezk et al. (2024), who found that SHAP and LIME produced concordant feature attribution hierarchies when applied to a heart disease prediction ensemble. This convergence strengthens the confidence in the CardioXAI explanation outputs and validates the design decision to use SHAP as the primary explanation engine with LIME as a cross-validation mechanism.

The CHD Risk Factor Knowledge Graph provides a novel contribution that goes beyond what has been reported in existing BRFSS-based cardiovascular prediction studies. While Hasnat et al. (2025), Tompra et al. (2024), and Muhammad et al. (2025) all focused on individual feature importance without visualising inter-feature relationships, the Knowledge Graph enables users to understand the systemic nature of their risk profile. The identification of risk factor clusters (e.g., the cardiovascular cluster of HighBP, HighChol, and Stroke; the metabolic cluster of BMI and Diabetes) communicates that CHD risk is not driven by isolated factors but by mutually reinforcing networks of behavioral and physiological conditions.


4.8.3 Uncertainty Quantification as a Safety Mechanism

The three-tier confidence classification system addresses a critical gap identified in the systematic reviews by Svenšek et al. (2025) and Pedroso and Khera (2025), both of which identified the absence of uncertainty communication as a major deficiency in existing cardiovascular risk assessment tools. By blocking definitive risk score display for low-confidence predictions and routing users to physician consultation, CardioXAI operationalises the ethical principle that AI-based health tools must not present unreliable outputs as definitive assessments (Kompa et al., 2021; Begoli et al., 2019).

The dual-metric approach to uncertainty (inter-model disagreement and predictive entropy) captures two distinct sources of prediction unreliability: disagreement reflects cases where the component models have learned genuinely different risk assessments from the same input, while entropy captures proximity to the decision boundary regardless of model agreement. Together, these metrics provide a more robust uncertainty estimate than either metric alone.


4.8.4 Calibration Limitations and Mitigation

The calibration analysis reveals a systematic overestimation of CHD risk, attributable primarily to the SMOTE rebalancing applied during training. This is a recognized limitation of SMOTE-trained models in the imbalanced learning literature (Chawla et al., 2002). For the CardioXAI use case, the overestimation is partially mitigated by the risk banding system and the emphasis on SHAP-derived behavioral attributions rather than raw probability values. Future work should investigate post-hoc calibration methods, including Platt scaling and isotonic regression, to improve probability calibration without sacrificing the discriminatory gains achieved by SMOTE-based training.


4.8.5 Addressing Study Objectives

The results presented in this chapter address each of the five study objectives stated in Section 1.4:

1. Objective 1 (Data preprocessing, EDA, and model training): The Pytlak BRFSS 2022 dataset was successfully preprocessed, exploratory analysis confirmed the expected class imbalance and feature distributions, and three classification models were trained and evaluated using the complete evaluation metric suite specified in Chapter Three.

2. Objective 2 (Model comparison and ensemble construction): Model performance was evaluated using accuracy, F1 score, AUC-ROC, precision, and recall. The consensus ensemble from XGBoost and Logistic Regression achieved the highest F1 score and preserved discriminatory ability, validating the ensemble design.

3. Objective 3 (SHAP, LIME, and Knowledge Graph integration): The multi-layer explainability framework was implemented and shown to produce clinically coherent, cross-method-consistent feature attributions. The Knowledge Graph provides a novel graph-based explanation layer not present in prior BRFSS-based prediction studies.

4. Objective 4 (Uncertainty quantification): The dual-metric confidence tier system was implemented and demonstrated to correctly identify predictions warranting physician consultation. The system successfully routes low-confidence predictions away from definitive risk score presentation.

5. Objective 5 (Web deployment with recommendations): The complete framework was deployed as a Django web application with a mobile-accessible interface, integrating risk prediction, multi-layer explainability, uncertainty communication, and personalised evidence-based lifestyle recommendations in a single consumer-facing system.


---


CHAPTER FIVE

SUMMARY, CONCLUSION, AND RECOMMENDATIONS

5.0 Introduction

This chapter concludes the study by summarising the key findings, stating the conclusions drawn from the results presented in Chapter Four, identifying the limitations of the study, highlighting the contributions to knowledge, and offering recommendations for future research and practice.


5.1 Summary of Findings

This study designed, implemented, and evaluated CardioXAI, an Explainable Artificial Intelligence (XAI) framework for at-home coronary heart disease (CHD) risk self-assessment using exclusively self-reported behavioral and symptom-based data. The framework was developed to address three critical gaps in existing cardiovascular risk prediction systems: the dependence on laboratory-derived clinical variables, the absence of transparent and interpretable explanations for model predictions, and the failure to communicate prediction uncertainty to end users.

The following summarises the principal findings of the study:

5.1.1 Data Preprocessing and Model Training

The Pytlak CDC BRFSS 2022 Heart Disease Health Indicators dataset was successfully preprocessed, yielding 186,487 cleaned observations with 16 self-reported features. The target variable exhibited pronounced class imbalance (87.8% CHD-negative, 12.2% CHD-positive), which was addressed through the application of Synthetic Minority Over-sampling Technique (SMOTE) exclusively to the training subset. Three supervised classification models — Logistic Regression, Random Forest, and XGBoost — were trained on the balanced training data and evaluated on unmodified validation and test subsets preserving the original class distribution.

5.1.2 Model Performance and Ensemble Construction

All three individual models achieved AUC-ROC values exceeding 0.80 on both validation and test subsets, demonstrating clinically acceptable discriminatory ability using self-reported data alone. The three models exhibited complementary precision-recall profiles: Logistic Regression achieved the highest recall (0.7486 test), XGBoost achieved the highest precision (0.5177 test), and Random Forest occupied an intermediate position. The consensus ensemble, constructed from XGBoost (weight = 0.5017) and Logistic Regression (weight = 0.4983) based on validation AUC-ROC proportional weighting, achieved the highest F1 score (0.4126 test) among all models while preserving discriminatory ability (AUC-ROC = 0.8150 test). The near-equal weighting reflected the closely matched discriminatory performance of the two component models.

5.1.3 Explainability Framework

The multi-layer explainability framework produced clinically coherent and cross-method-consistent explanations for individual predictions. SHAP (SHapley Additive exPlanations), implemented through TreeExplainer for XGBoost and LinearExplainer for Logistic Regression with AUC-ROC-proportional weighted averaging, consistently identified Age, General Health, High Blood Pressure, BMI, Diabetes, and Difficulty Walking as the most influential risk drivers — findings that align with established cardiovascular epidemiology. LIME (Local Interpretable Model-Agnostic Explanations) attributions showed strong convergence with SHAP for the top-ranked features, providing independent cross-validation of the primary risk factor identifications.

The CHD Risk Factor Knowledge Graph, comprising 16 nodes and 29 clinically established edges, provided a novel graph-based explanation layer that contextualises individual feature attributions within the broader network of inter-factor relationships. The personalised graph visualisation, which overlays SHAP-identified risk drivers onto the network structure, enables users to understand the systemic and interconnected nature of their risk profile.

5.1.4 Uncertainty Quantification

The dual-metric confidence tier system, combining inter-model disagreement and predictive entropy, successfully classified predictions into High, Medium, and Low confidence tiers. Predictions at the extremes of the probability spectrum consistently achieved High confidence, while predictions near the decision boundary fell into Medium or Low confidence tiers. Low-confidence predictions triggered a distinct user interface pathway that routes users to physician consultation rather than presenting potentially misleading risk scores.

5.1.5 Web Application Deployment

The complete CardioXAI framework was deployed as a Django web application integrating all analytical components — risk prediction, SHAP and LIME explanations, the personalised Knowledge Graph, uncertainty communication, and evidence-based lifestyle recommendations — into a single consumer-facing interface accessible through any standard web browser. The application implements a progressive three-step assessment form, an animated risk gauge with risk band classification, and up to five personalised recommendations derived from SHAP-identified modifiable risk factors.


5.2 Conclusion

Based on the findings of this study, the following conclusions are drawn:

1. Clinically acceptable CHD risk discrimination can be achieved using exclusively self-reported behavioral and symptom-based data, without requiring laboratory measurements. The consensus ensemble achieved an AUC-ROC of 0.8150 on the held-out test set, exceeding the 0.80 threshold generally considered clinically acceptable for cardiovascular risk prediction (Weng et al., 2017; Krittanawong et al., 2020). This validates the feasibility of laboratory-free CHD risk screening for at-home self-assessment contexts where clinical testing infrastructure is unavailable or inaccessible.

2. A consensus ensemble combining models with complementary precision-recall profiles produces superior balanced performance compared to any individual model. The weighted combination of XGBoost (high precision, low recall) and Logistic Regression (high recall, low precision) achieved the highest F1 score among all models, demonstrating that ensemble diversity in error patterns is more valuable than ensemble diversity in model architecture alone.

3. Multi-layer explainability using SHAP and LIME produces clinically coherent feature attributions that converge on the same primary risk drivers. The cross-method consistency between SHAP and LIME strengthens confidence that identified risk factors reflect genuine learned patterns rather than artifacts of a single explanation methodology. The SHAP-identified risk drivers are consistent with established cardiovascular risk factor hierarchies in the clinical literature.

4. The CHD Risk Factor Knowledge Graph provides a novel explanation modality that contextualises isolated feature attributions within a connected network of clinical relationships. This graph-based approach communicates the systemic nature of cardiovascular risk in a way that ranked feature lists cannot, and represents a contribution not present in prior BRFSS-based cardiovascular prediction studies.

5. Dual-metric uncertainty quantification using inter-model disagreement and predictive entropy provides a practical mechanism for identifying predictions that warrant physician consultation rather than unsupervised interpretation. The three-tier confidence classification system operationalises the ethical principle that AI-based health assessment tools must communicate prediction reliability to end users.

6. The integration of risk prediction, multi-layer explainability, uncertainty communication, and personalised recommendations into a single consumer-facing web application demonstrates that transparent, safety-aware AI health tools can be made accessible to non-technical users without sacrificing analytical rigour.


5.3 Limitations of the Study

Despite the contributions described above, this study is subject to several limitations that should be acknowledged:

5.3.1 Data Limitations

1. Self-reported data bias: All 16 input features are derived from self-reported survey responses, which are subject to recall bias, social desirability bias, and inaccurate self-assessment. Respondents may underreport conditions such as heavy alcohol consumption or overreport protective behaviors such as physical activity. The accuracy of the model's predictions is fundamentally bounded by the accuracy of the self-reported inputs.

2. Cross-sectional design: The BRFSS dataset captures a single cross-sectional snapshot of health status rather than longitudinal trajectories. The model predicts the association between current self-reported features and self-reported CHD history, not the prospective risk of developing CHD over a defined time horizon. This limits the clinical interpretability of the risk score compared to prospective cohort-derived models such as the Framingham Risk Score.

3. Geographic and demographic scope: The BRFSS dataset is drawn exclusively from the United States adult population. The model's learned associations may not generalise to populations with different genetic backgrounds, dietary patterns, healthcare access profiles, or disease prevalence rates. Application to non-US populations would require validation on regionally appropriate datasets.

4. Binary feature encoding: Several features that are clinically continuous or multi-level (e.g., diabetes type, smoking intensity, alcohol quantity) are encoded as binary variables in the dataset. This encoding discards clinically relevant dose-response information and may reduce the model's ability to discriminate between individuals with different levels of exposure to the same risk factor.

5.3.2 Model Limitations

5. Calibration overestimation: The ensemble model systematically overestimates CHD risk due to the SMOTE rebalancing applied during training, as demonstrated by the calibration analysis in Section 4.6. While overestimation does not affect the model's discriminatory ability (AUC-ROC), it means that the absolute probability values should not be interpreted as true population-level risk estimates. No post-hoc calibration correction (e.g., Platt scaling, isotonic regression) was applied in the current implementation.

6. Fixed classification threshold: The binary classification threshold is fixed at 0.5 for all predictions. Threshold optimisation based on clinical cost considerations (e.g., the relative cost of false negatives versus false positives in a screening context) was not performed and may yield improved clinical utility.

7. Absence of external validation: The model was trained and evaluated on subsets of a single dataset (BRFSS 2022). No external validation was conducted on independent datasets from different survey years, geographic regions, or data collection methodologies. The generalisation performance reported may therefore be optimistic relative to true out-of-distribution performance.

5.3.3 Explainability Limitations

8. Knowledge Graph edges are expert-curated: The 29 edges in the CHD Risk Factor Knowledge Graph are derived from clinical literature rather than learned from the data. While this ensures clinical validity, the graph does not capture data-driven associations that may exist but are not documented in the curated edge set. The edge weights are assigned based on clinical judgement rather than estimated from empirical correlation or causal analysis.

9. LIME stochasticity: LIME explanations are generated through stochastic perturbation sampling (500 samples per prediction), which introduces variability in the lower-ranked feature attributions across repeated evaluations of the same input. While the top-ranked features are stable, features ranked 5th through 8th may vary between evaluations, which could reduce user confidence in the explanation output if multiple evaluations of the same profile produce visibly different LIME displays.

5.3.4 Deployment Limitations

10. No clinical user evaluation: The CardioXAI application was not evaluated through user studies with target end users (patients, general public) or clinical domain experts (cardiologists, general practitioners). The usability, comprehensibility, and perceived trustworthiness of the SHAP explanations, LIME panel, Knowledge Graph, and recommendation outputs have not been empirically assessed.

11. No regulatory compliance assessment: The application has not been assessed for compliance with medical device regulations (e.g., EU Medical Device Regulation, FDA Software as a Medical Device guidance) or health data privacy frameworks (e.g., HIPAA, UK GDPR). Deployment in regulated healthcare contexts would require such assessments.


5.4 Contributions to Knowledge

This study makes the following contributions to the body of knowledge in the fields of explainable artificial intelligence and cardiovascular risk prediction:

1. Laboratory-free CHD prediction with clinically acceptable performance: The study demonstrates that an AUC-ROC of 0.8150 can be achieved using exclusively self-reported behavioral features, validating the feasibility of at-home CHD risk screening without laboratory infrastructure. This extends the work of Hasnat et al. (2025) and Tompra et al. (2024) by embedding the prediction model within a complete consumer-facing application rather than reporting standalone model performance.

2. Multi-layer explainability framework with cross-method validation: The integration of SHAP as the primary explanation engine with LIME as an independent cross-validation mechanism provides a dual-method transparency architecture that strengthens confidence in feature attribution outputs. This approach extends the SHAP-LIME combination validated by Rezk et al. (2024) from a clinical dataset context to a large-scale self-reported data context.

3. CHD Risk Factor Knowledge Graph: The construction and personalisation of a graph-based representation of inter-factor relationships represents a novel explainability modality not present in prior BRFSS-based cardiovascular prediction studies. The Knowledge Graph transforms isolated feature attribution scores into a connected risk narrative, providing a systems-level view of individual CHD risk that complements the feature-level explanations provided by SHAP and LIME.

4. Integrated uncertainty quantification with user-facing safety routing: The dual-metric confidence tier system (inter-model disagreement and predictive entropy) provides a practical operationalisation of the ethical principles articulated by Kompa et al. (2021) and Begoli et al. (2019) regarding uncertainty communication in medical AI. The implementation demonstrates that uncertainty-aware prediction can be integrated into consumer-facing health tools without requiring users to interpret raw statistical uncertainty measures.

5. End-to-end deployment as a consumer-facing web application: The integration of risk prediction, multi-layer explainability, uncertainty quantification, and personalised evidence-based recommendations into a single Django web application demonstrates a complete translational pathway from model development to end-user deployment, addressing the gap between research prototypes and deployable health tools.


5.5 Recommendations

Based on the findings, conclusions, and limitations of this study, the following recommendations are made for future research and practice:

5.5.1 Recommendations for Future Research

1. Post-hoc calibration: Future iterations should apply Platt scaling or isotonic regression to the ensemble output probabilities to correct the systematic overestimation identified in the calibration analysis. Calibration correction would improve the alignment between predicted risk scores and true positive rates without sacrificing the discriminatory gains achieved through SMOTE-based training.

2. Threshold optimisation: The fixed 0.5 classification threshold should be replaced with a threshold optimised for clinical utility, potentially using decision curve analysis or cost-sensitive threshold selection that accounts for the differential harms of false positives (unnecessary anxiety) and false negatives (missed CHD cases) in a consumer screening context.

3. External validation: The model should be validated on independent datasets, including BRFSS surveys from different years, equivalent surveys from non-US populations (e.g., the UK Health Survey for England, the European Health Interview Survey), and, where available, datasets with both self-reported features and confirmed clinical outcomes to assess the gap between self-reported and clinically verified CHD status.

4. Longitudinal prediction: Future work should explore the development of prospective risk models using longitudinal data that capture changes in behavioral risk factors over time, enabling the prediction of future CHD event risk rather than cross-sectional association with self-reported CHD history. Integration with wearable health device data could provide continuous, objectively measured behavioral inputs.

5. Data-driven Knowledge Graph construction: The expert-curated Knowledge Graph should be complemented with data-driven edge discovery using techniques such as mutual information, partial correlation analysis, or Bayesian network structure learning. Comparing expert-curated and data-driven graphs would identify clinically relevant associations that may be present in the data but absent from the curated edge set.

6. Advanced ensemble methods: Future work should investigate stacking ensembles and neural network meta-learners as alternatives to the linear weighted average combination used in the current consensus ensemble. Additionally, exploring the inclusion of all three models (including Random Forest) with learnable weights may yield improved balanced performance.

7. Fairness and bias auditing: The model's performance should be audited across demographic subgroups (age, sex, race/ethnicity) to identify and mitigate potential disparities in prediction accuracy or calibration. Fairness-aware training techniques should be investigated to ensure equitable performance across population subgroups.

5.5.2 Recommendations for Practice

8. User evaluation studies: Before deployment in any healthcare-adjacent context, CardioXAI should undergo structured user evaluation studies with target end users (members of the public) and clinical domain experts (cardiologists, general practitioners) to assess the comprehensibility, usability, and perceived trustworthiness of the explanation outputs and recommendation content.

9. Clinical integration pathway: Healthcare providers and public health organisations should consider integrating tools like CardioXAI as pre-screening instruments that identify individuals who may benefit from formal clinical cardiovascular risk assessment. The tool's value lies in behavioural awareness and risk factor education rather than clinical diagnosis.

10. Regulatory compliance: Any deployment beyond research or educational use should undergo assessment for compliance with applicable medical device regulations and health data privacy frameworks. The application should be clearly positioned as an informational self-assessment tool, not a diagnostic device, in all user-facing communications.

11. Continuous model updating: As new BRFSS survey data becomes available annually, the model should be periodically retrained and recalibrated to reflect evolving population health profiles, emerging risk factor associations, and changes in survey methodology.

12. Accessibility and health literacy: Future development should prioritise accessibility compliance (WCAG 2.1 AA) and health literacy considerations, ensuring that the explanation outputs and recommendation content are comprehensible to users with varying levels of education and health literacy. Plain-language summaries, visual aids, and multilingual support would expand the reach of the tool to underserved populations who stand to benefit most from accessible cardiovascular risk education.
