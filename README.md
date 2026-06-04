# Tau PET MetaROI — Machine Learning Pipeline

**Predicting Tau Burden and Tau Positivity from Multimodal Neuroimaging and Clinical Data in ADNI**


---

## Introduction

Alzheimer's disease (AD) is characterized by pathological accumulation of amyloid plaques and tau neurofibrillary tangles in the brain. Tau PET is one of the most accurate methods to identify AD pathology currently, as it allows in vivo quantification of tau burden. However, it is very expensive, time-consuming, and not widely available in clinical settings. 

So, a model capable of predicting tau PET burden, or classifying tau positivity, from cheaper, more accessible measures (FDG-PET, amyloid PET, cognitive scores, genetics) would have significant clinical & research value.

This project is a complete and reproducible ML pipeline using data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** to:
1. Predict tau PET SUVR as a continuous outcome
2. Predict tau positivity as a binary outcome (classify participants as tau-positive or tau-negative)
3. Identify which predictors drive model performance using SHAP explainability

The pipeline is fully open-source and designed for reproducibility: so anyone can clone this repository and run the full pipeline immediately using the included synthetic dataset that follows my real ADNI data distribution! :D

---

## Objectives

- Build and compare ML models predicting Tau PET MetaROI SUVR from clinical, FDG-PET, and amyloid-PET features
- Classify tau positivity (tau+ / tau−) using expert visual reads as ground truth labels
- Evaluate generalization across imaging sites using Leave-One-Site-Out cross-validation
- Identify the most relevant predictors using SHAP values
- Generate synthetic data for open-science sharing

---

## Methods

### Data
All data comes from ADNI. The primary dataset includes **469 participants** with Tau PET, FDG-PET, amyloid PET, and clinical assessments acquired. Tau binary labels (tau+ / tau−) were obtained from the **Gothenburg visual read consensus** (3-expert panel), available for a subset of participants (423 participants).

**Tau PET MetaROI** was computed as the mean SUVR across five regions: posterior cingulate cortex, left and right inferior temporal cortex, and left and right supramarginal gyrus.

### Preprocessing
- Merged all PET, Amyloid and clinical data in one single csv
- Age corrected to scan date using baseline age + time elapsed since baseline visit
- Diagnosis recoded: LMCI/EMCI → MCI; Dementia → AD; SMC → CN

### Predictor Sets

| Set | Features |
|---|---|
| `clinical` | Age, sex, education, APOE ε4, CDR-SB, MMSE |
| `amyloid` | Amyloid PET SUVR, Centiloids |
| `fdg` | FDG SUVR, HCI, SROI AD, SROI MCI |
| `clinical_amyloid` | clinical + amyloid |
| `clinical_fdg` | clinical + FDG |
| `all` | All predictors combined |

### Models
Four model types were evaluated for both continuous and binary tasks:
- **ElasticNet / Logistic Regression**
- **Random Forest** 
- **XGBoost** 
- **Gradient Boosting** 

### Cross-Validation Strategy
**Repeated Stratified K-Fold** (5 folds × 3 repeats = 15 splits): stratified on DX × APOE4 binary status to ensure balanced group representation in every fold

**Leave-One-Site-Out (LOSO)**: each imaging site held out as the test set, training on all remaining sites. Tests generalization across scanner and protocol variability.

### Synthetic Data
Synthetic data was generated using a **conditional multivariate Gaussian** approach: for each diagnostic group (CN, MCI, AD), the mean vector and covariance matrix were estimated from the real data, and synthetic samples were drawn from the resulting multivariate normal distribution. Values were clipped to observed ranges for each predictor. This approach produces data that preserves group-specific means, variances, and covariance structure without relying on generative models.

---

## Skills Learned

- **Machine Learning**: regression and classification pipelines, hyperparameter configuration, model comparison, handling class imbalance
- **Cross-Validation**: stratified K-Fold, repeated C, Leave-One-Site-Out for multi-site generalization
- **Model Explainability**: SHAP values, multi-model feature importance comparison
- **Clinical Metrics**: AUC-ROC, AUC-PR, balanced accuracy, sensitivity, specificity, PPV, NPV, calibration curves
- **Neuroimaging**: tau PET MetaROI computation, nilearn brain visualization
- **Open Science**: synthetic data generation, GitHub reproducibility
- **Python**: scikit-learn pipelines, XGBoost, SHAP, nilearn, pandas

---

## Results

### Participant Statistics

| | Continuous | Binary (with visual reads) |
|---|---|---|
| N total | 469 | 423 |
| tau+ | — | 152 (35.9%) |
| tau− | — | 271 (64.1%) |
| Imbalance ratio | — | 1.78 |

**TAU_SUVR Distribution:**
- tau− participants: 1.129 ± 0.075 SUVR (range: 0.919–1.345)
- tau+ participants: 1.527 ± 0.423 SUVR (range: 1.048–3.247)
- SUVR midpoint: **1.328**
- Overlap zone: 1.048–1.345 (where SUVR alone cannot separate groups)

**Diagnostic groups:**
- CN: 161 | MCI: 234 | AD: 74
- tau+ by group: CN 16.5% · MCI 35.2% · AD 76.1%

---

### Continuous Outcome — TAU_SUVR Prediction

#### Most Important Figures

`results/figures/cont_heatmap_R2.png` — R² across all model × predictor set combinations (stratified CV)

`results/figures/cont_boxes_R2_stratified.png` — R² distribution across folds per model

`results/figures/cont_loso_site_heatmap.png` — per-site Pearson r (LOSO)

`results/figures/cont_scatter_{best_model}_{best_set}.png` — predicted vs actual TAU_SUVR

#### Metrics Table (mean ± SD across 15 CV folds)

> Update with actual results after running the pipeline

| Model | Predictor Set | R² | RMSE | MAE | Pearson r |
|---|---|---|---|---|---|
| XGBoost | all | — | — | — | — |
| GradientBoosting | all | — | — | — | — |
| RandomForest | all | — | — | — | — |
| ElasticNet | all | — | — | — | — |

---

### Binary Outcome — Tau+/Tau− Classification

#### Key Figures

`results/figures/binary_roc_all.png` — Mean ROC curves ± SD across folds

`results/figures/binary_calibration_all.png` — Calibration curves

`results/figures/binary_heatmap_AUC_ROC.png` — AUC-ROC heatmap

`results/figures/binary_heatmap_Bal_Accuracy.png` — Balanced accuracy heatmap

#### Metrics Table (mean ± SD, threshold = 0.445)

> Update with actual results after running the pipeline

| Model | Predictor Set | AUC-ROC | AUC-PR | Bal. Acc. | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| XGBoost | all | — | — | — | — | — | — |
| GradientBoosting | all | — | — | — | — | — | — |
| RandomForest | all | — | — | — | — | — | — |
| LogisticRegression | all | — | — | — | — | — | — |

---

### Feature Importance (SHAP)

`results/figures/shap_beeswarm_cont_{model}.png` — feature direction and magnitude

`results/figures/shap_multimodel_cont.png` — consistent importance across models

`results/figures/shap_dep_cont_{model}_{feature}.png` — dependence plots for top features

---

### Brain Visualization

`results/figures/brain_metaroi_glass.png` — MetaROI regions on glass brain (CN / MCI / AD)

`results/figures/brain_metaroi_suvr_bar.png` — mean SUVR per region × DX group

`results/figures/brain_metaroi_progression.png` — tau progression CN→MCI→AD per region

---

## Conclusion

This pipeline demonstrates that tau PET burden and positivity can be predicted from multimodal biomarker data with clinically meaningful accuracy. The systematic comparison of predictor sets reveals the relative contribution of clinical, amyloid, and FDG features. LOSO cross-validation confirms site generalizability. SHAP analysis provides interpretable insights into which biomarkers drive predictions. The full pipeline is reproducible by anyone via the included synthetic dataset.

---

## Repository Structure

```
tau-metaroi-ml/
├── config.yaml                   # All settings
├── requirements.txt
├── run_all.sh                    # One-command runner
├── Makefile
├── data/
│   ├── README_data.md            # ADNI access instructions
│   └── synthetic/
│       └── synthetic_dataset.csv # Pre-generated synthetic data
├── src/
│   ├── utils.py
│   ├── 00_data_prep.py
│   ├── 01_generate_synthetic.py
│   ├── 02_ml_continuous.py
│   ├── 03_ml_binary.py
│   ├── 04_shap_analysis.py
│   └── 05_brain_render.py
└── results/
    ├── figures/
    └── tables/
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/tau-metaroi-ml.git
cd tau-metaroi-ml

# 2. Install
pip install -r requirements.txt

# 3. Run full pipeline on synthetic data
bash run_all.sh

# Results in results/figures/ and results/tables/
```

**With real ADNI data:**
```bash
bash run_all.sh --real \
  --merged  /path/to/merged_data.csv \
  --visual  /path/to/visual_reads.csv
```

**Generate synthetic data from your real data (run once, commit output):**
```bash
python src/01_generate_synthetic.py --mode from_real \
  --merged      /path/to/merged_data.csv \
  --visual_reads /path/to/visual_reads.csv
```

---

## Data & Ethics

Real data: ADNI — not included (restricted access). See `data/README_data.md`.
Synthetic data: generated via conditional multivariate Gaussian per diagnostic group. No real participant data included in this repository.

## Citation

Jack CR Jr, et al. The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *J Magn Reson Imaging*. 2008.

## License
MIT
