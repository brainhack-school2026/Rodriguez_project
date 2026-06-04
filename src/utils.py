"""
utils.py — helper functions for the ML pipeline.
"""

import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from pathlib import Path
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, confusion_matrix, brier_score_loss,
    balanced_accuracy_score,
)
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_dirs(cfg: dict) -> None:
    for key in ("figures_dir", "tables_dir", "data_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["synthetic_data"]).parent.mkdir(parents=True, exist_ok=True)


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Predictor sets ─────────────────────────────────────────────────────────────

def build_predictor_sets(cfg: dict, available_cols: list) -> dict:
    raw = cfg["predictor_sets"]
    base_sets = {k: v for k, v in raw.items() if isinstance(v, list)}

    resolved = {}
    for name, val in raw.items():
        if isinstance(val, list):
            cols = val
        else:
            cols = []
            for parent in val.get("extends", []):
                cols += base_sets.get(parent, [])
            cols = list(dict.fromkeys(cols))
        resolved[name] = [c for c in cols if c in available_cols]

    return {k: v for k, v in resolved.items() if v}


# ── Metrics ────────────────────────────────────────────────────────────────────

def continuous_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "Pearson_r", "Spearman_r"]}
    r, _   = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "R2":         r2_score(y_true, y_pred),
        "RMSE":       np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":        mean_absolute_error(y_true, y_pred),
        "Pearson_r":  r,
        "Spearman_r": rho,
    }


def binary_metrics(y_true, y_prob, threshold: float = 0.445) -> dict:
    """
    Threshold = 0.445 (Youden J optimal on this dataset, slightly below 0.5
    due to class imbalance: 64.1% tau-, 35.9% tau+, ratio 1.78).
    """
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in
                ["AUC_ROC", "AUC_PR", "Bal_Accuracy", "Accuracy",
                 "Sensitivity", "Specificity", "PPV", "NPV", "F1", "Brier"]}
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "AUC_ROC":     roc_auc_score(y_true, y_prob),
        "AUC_PR":      average_precision_score(y_true, y_prob),
        "Bal_Accuracy":balanced_accuracy_score(y_true, y_pred),
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV":         ppv,
        "NPV":         npv,
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Brier":       brier_score_loss(y_true, y_prob),
    }


# ── Model builders ─────────────────────────────────────────────────────────────

def build_continuous_models(cfg: dict) -> dict:
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor

    models = {}
    for name, params in cfg["models"]["continuous"].items():
        p = dict(params)
        p["random_state"] = cfg["seed"]
        if name == "ElasticNet":
            models[name] = ElasticNet(**p)
        elif name == "RandomForestRegressor":
            models[name] = RandomForestRegressor(**p)
        elif name == "XGBRegressor":
            models[name] = XGBRegressor(**p)
        elif name == "GradientBoostingRegressor":
            models[name] = GradientBoostingRegressor(**p)
    return models


def build_binary_models(cfg: dict, scale_pos_weight: float = 1.0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier

    models = {}
    for name, params in cfg["models"]["binary"].items():
        p = dict(params)
        p["random_state"] = cfg["seed"]
        if name == "LogisticRegression":
            models[name] = LogisticRegression(**p)
        elif name == "RandomForestClassifier":
            models[name] = RandomForestClassifier(**p)
        elif name == "XGBClassifier":
            p["scale_pos_weight"] = scale_pos_weight
            models[name] = XGBClassifier(**p)
        elif name == "GradientBoostingClassifier":
            models[name] = GradientBoostingClassifier(**p)
    return models


def build_pipeline(model, needs_scaling: bool = False):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


NEEDS_SCALING = {"ElasticNet", "LogisticRegression"}


# ── Plot style ─────────────────────────────────────────────────────────────────

def apply_style() -> None:
    mpl.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.linewidth":     0.8,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


PALETTE = {"CN": "#4C9BE8", "MCI": "#F4A261", "AD": "#E76F51"}

MODEL_COLORS = {
    "ElasticNet":                "#6BAED6",
    "LogisticRegression":        "#6BAED6",
    "RandomForestRegressor":     "#2CA25F",
    "RandomForestClassifier":    "#2CA25F",
    "XGBRegressor":              "#E05C2A",
    "XGBClassifier":             "#E05C2A",
    "GradientBoostingRegressor": "#8856A7",
    "GradientBoostingClassifier":"#8856A7",
}


# ── Table helpers ──────────────────────────────────────────────────────────────

def fmt_mean_sd(series: pd.Series, decimals: int = 3) -> str:
    s = series.dropna()
    return "—" if len(s) == 0 else f"{s.mean():.{decimals}f} ± {s.std():.{decimals}f}"
