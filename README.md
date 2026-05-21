# 🧠 Predicting Tau PET SUVR in Preclinical Alzheimer's Disease
### A Machine Learning Approach Using Multi-Modal Neuroimaging

---

## About Me

Hi, I'm Sebastian Rodriguez and I'm currently a Master's student at Université de Montréal. This is my dog named Koko :D

<a href="https://github.com/sebrm2">
  <img src="https://avatars.githubusercontent.com/u/111035325?s=400&u=7e4efc0...&v=4" width="400px;" alt=""/>
  <br /><sub><b>Sebastian Rodriguez</b></sub>
</a>

---
 
## Topic of Interest
 
### Background
 
Alzheimer's disease (AD) progresses silently for decades before any clinical symptoms appear. This **preclinical phase** is now an important target for early intervention. Two main pathological processes occur during this window: the accumulation of **amyloid-beta (Aβ) plaques**, which appears first, followed by the spread of **tau neurofibrillary tangles**, which reflects much more closely neurodegeneration and cognitive decline.
 
**Tau PET imaging** allows us to measure tau burden using standardized uptake value ratios (SUVR), and it's very powerful in identifying preclinical AD, however it is also extremely expensive, unavailable in most places and complex. Here I'm focusing on the **meta-ROI** approach (pre-defined composite regions of interest that aggregate signal across areas particularly vulnerable in early AD (entorhinal cortex, inferior temporal, fusiform, parahippocampal gyri, and others).
 
The question I'm exploring: **can we predict an individual's tau PET meta-ROI SUVR from cheaper and more available data like FDG-PET metabolism, and demographics?** If so, which modalities matter most?

---
 
### Modalities & Tools I'm Working With
 
| Modality | What does it measure | Features used |
|---|---|---|
| **Tau PET** (flortaucipir) | Tau tangle burden | Meta-ROI SUVR ← *target variable* |
| **Amyloid PET** (florbetapir/florbetaben) | Aβ plaque load | Meta-ROI SUVR, Centiloid |
| **FDG-PET** | Regional glucose metabolism (neuronal activity) | Meta-ROI SUVR |
| **MRI (structural)** | Brain atrophy, cortical thickness | Hippocampal volume, entorhinal thickness, WMH |
| **Demographics / genetics** | Baseline risk factors | Age, sex, education, APOE ε4 status |
 
**Dataset:** [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/), a longitudinal multi-site study. I'm using cross-sectional baseline visits from cognitively unimpaired (CU) participants (preclinical AD population).
 
> ⚠️ **Note on data:** ADNI data requires a data use agreement and cannot be shared publicly. This repo will include a **synthetic dataset** that mirrors the real data structure and distributions so anyone can run the notebooks.
 
---
### Tools & Libraries
 
| Package | What I'm using it for |
|---|---|
| `scikit-learn` | ElasticNet, cross-validation (`KFold`, `GridSearchCV`), metrics (`r2_score`, `mean_absolute_error`) |
| `xgboost` / `lightgbm` | Gradient boosting regressors |
| `shap` | Feature importance, waterfall + beeswarm plots |
| `pandas` / `numpy` | Data wrangling, imputation |
| `scipy.stats` | Correlation analysis, distribution fitting for synthetic data |
| `matplotlib` / `seaborn` | Visualization |
| `sklearn.preprocessing` | StandardScaler, normalization |
| `sklearn.impute` | SimpleImputer, IterativeImputer (MICE) |
| `sklearn.model_selection` | StratifiedKFold, cross_val_score, nested CV |
| `sklearn.metrics` | R², MAE, RMSE |
 
---
 
## Skills I Want to Learn

### 1. Cross-validation for small neuroimaging datasets
I will use around 500 participants from ADNI. Learning to do **nested cross-validation** (outer loop for performance estimation, inner loop for hyperparameter tuning) correctly will be very useful in the future. This is something I've read about but never implemented from scratch, honestly.
 
### 2. Regression metrics
Start to use other stuff like **MAE, MCC, PR-AUC**, and understanding when each is appropriate. Tau SUVR values are in a narrow range, so I believe that metric choice affects how meaningful differences look. And also try to identify whether it's better to have a higher precision or recall.
 
### 3. Gradient boosting and Tree models
Specifically **XGBoost and/or LightGBM, RF**.
 
### 4. SHAP for model explainability
Learning to compute and visualize **SHAP values**, both global feature importance (beeswarm plots) and individual-level explanations (waterfall plots). Which biomarkers drive tau burden in which individuals?
 
### 5. Synthetic data generation
How to generate realistic synthetic tabular data that preserves the distributions and correlation structure of the real ADNI data, without exposing any real participant information. Open science :)

### 6. Reproducible workflows
Using **Jupyter notebooks** with clear structure and maybe implement a script at the end that requires no meddling with the code and one can just hit run and it runs!

---

## Resources
 
### Dataset
- **ADNI** (Alzheimer's Disease Neuroimaging Initiative): [adni.loni.usc.edu](https://adni.loni.usc.edu/)
  - Applying for access: requires a short application + data use agreement (free, I got a reply in less than a day)
  - I'm using ADNIMERGE, PET SUVR summary files, and plasma biomarker tables
- **Synthetic data**: generated to have ADNI distributions 

 
### Learning Resources
- [SHAP documentation](https://shap.readthedocs.io/)
- [scikit-learn User Guide — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [XGBoost docs](https://xgboost.readthedocs.io/)
- Chiotis et al. (2025) — The Role of Amyloid-β and Tau PET in the New Era of Alzheimer Disease Therapies. (in-depth explanation of Tau-PET)
- [ADNI Methods papers]([https://adni.loni.usc.edu/methods/](https://adni.loni.usc.edu/help-faqs/adni-documentation/))
---
 
## Objectives
 
By the end of this hackathon, I want to:
 
1. **Build and evaluate a multi-modal ML pipeline** that predicts tau PET meta-ROI SUVR using real ADNI data locally and synthetic data publicly.
2. **Run a modality ablation study** that compare models trained on each modality subset, different site data, and other variables, to quantify the value of each data type and test generalizability.
3. **Implement nested cross-validation correctly**, avoiding data leakage.
4. **Interpret results with SHAP** to identify which features are important for predictions globally and for individual participants.
5. **Make everything reproducible** so that anyone should be able to clone this repo, install the environment, and run every notebook using the synthetic data.
6. **Document the learning process** with stuff that worked and that didn't.
---
 
## Deliverables
 
### 1. This GitHub Repository
Structured and documented. My current tree skeleton looks like this:

```bash
.
├── LICENSE
├── README.md
├── data               # Raw data
├── docs               # Detailed project documentation
├── environment.yml
├── outputs            # Results
├── scripts            # Executable code
├── setup.py
└── src                # Core Python package
    ├── __init__.py
    ├── config.py      # Configuration & constants
    └── metadata.py    # Metadata utilities
```

Hopefully by the end it can look complete and organized :)

### 2. Fully Reproducible Jupyter Notebook
A single notebook that runs on the synthetic data.
 
### 3. Results Summary
A clean figures folder with all key plots and tables: feature importance, predicted vs. actual SUVR, modality ablation bar chart, SHAP beeswarm, model's metrics.

## Outcome
 
### Presentation: Short Paper and Slides (TBD)

A short paper with classic structure and some slides. I'm not sure yet if there is a more creative and useful way to present my results. 

