"""
04_shap_analysis.py
-------------------
SHAP explainability analysis for the best continuous and binary models.

Plots
-----
  • Beeswarm summary        (global feature importance + direction)
  • Bar chart (mean |SHAP|) (simple ranking)
  • Dependence plots        (top N features)
  • Waterfall plots         (highest / lowest / median prediction)
  • SHAP heatmap            (participants × features)
  • Multi-model comparison  (mean |SHAP| across all models)
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline

from utils import (
    load_config, set_seed, apply_style,
    build_predictor_sets, build_continuous_models, build_binary_models,
    build_pipeline, MODEL_COLORS, NEEDS_SCALING,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


DISPLAY = {
    "AGE_CORRECTED":           "Age",
    "PTGENDER":                "Sex (0=F)",
    "PTEDUCAT":                "Education",
    "APOE4":                   "APOE ε4",
    "DX":                      "DX (follow-up)",
    "DX_bl":                   "DX (baseline)",
    "CDRSB":                   "CDR-SB",
    "MMSE":                    "MMSE",
    "SUMMARY_SUVR_AbPET":      "AB PET SUVR",
    "CENTILOIDS_AbPET":        "Centiloids",
    "MEAN_FDG":                "FDG SUVR",
    "HCI":                     "HCI",
    "SROI_AD":                 "SROI AD",
    "SROI_MCI":                "SROI MCI",
    "WM_HYPOINTENSITIES_SUVR": "WM Hypointensities",
    "AMYLOID_STATUS_AbPET":    "Amyloid Status",
}


def pretty(col):
    return DISPLAY.get(col, col)


def compute_shap(pipe, X, X_bg, model_name, n_bg=100):
    """Compute SHAP values, handling 2D and 3D array outputs from classifiers."""
    rng    = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_bg), size=min(n_bg, len(X_bg)), replace=False)
    bg     = X_bg.iloc[bg_idx]

    # Transform through pre-model steps
    transforms = [s for s in pipe.steps if s[0] != "model"]
    if transforms:
        tmp  = Pipeline(transforms)
        bg_t = pd.DataFrame(tmp.transform(bg), columns=bg.columns)
        X_t  = pd.DataFrame(tmp.transform(X), columns=X.columns)
    else:
        bg_t, X_t = bg.copy(), X.copy()

    model = pipe.named_steps["model"]
    if model_name in ("ElasticNet", "LogisticRegression"):
        explainer = shap.LinearExplainer(model, bg_t,
                                          feature_perturbation="correlation_dependent")
    else:
        explainer = shap.TreeExplainer(model, bg_t,
                                        feature_perturbation="tree_path_dependent")

    sv = explainer.shap_values(X_t)

    # Handle list output (some tree classifiers)
    if isinstance(sv, list):
        sv = sv[1]

    # Handle 3D output (n_samples, n_features, n_classes)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]

    return sv, X_t


def safe_importance(sv):
    """Return 1D mean |SHAP|, handling 2D arrays."""
    arr = np.abs(sv).mean(axis=0)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return arr


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_beeswarm(sv, X_t, feats, title, out_path):
    labels = [pretty(f) for f in feats]
    fig, _ = plt.subplots(figsize=(8, max(5, 0.4 * len(feats))))
    shap.summary_plot(sv, X_t, feature_names=labels, show=False,
                      plot_size=None, max_display=len(feats))
    plt.title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved beeswarm → {Path(out_path).name}")


def plot_bar(sv, feats, title, out_path):
    mean_abs = safe_importance(sv)
    order    = np.argsort(mean_abs)[::-1]
    labels   = [pretty(feats[i]) for i in order]
    vals     = mean_abs[order]
    fig, ax  = plt.subplots(figsize=(6, max(4, 0.4 * len(feats))))
    colors   = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(vals)))
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved bar → {Path(out_path).name}")


def plot_dependence(sv, X_t, feats, n_top, figs_dir, tag):
    mean_abs = safe_importance(sv)
    top_idx  = np.argsort(mean_abs)[::-1][:n_top]
    for i in top_idx:
        feat = feats[i]
        fig, ax = plt.subplots(figsize=(5, 4))
        shap.dependence_plot(i, sv, X_t,
                              feature_names=[pretty(f) for f in feats],
                              ax=ax, show=False)
        ax.set_title(f"SHAP Dependence — {pretty(feat)}",
                     fontsize=11, fontweight="bold")
        sns.despine(ax=ax)
        plt.tight_layout()
        out = Path(figs_dir) / f"shap_dep_{tag}_{feat}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()


def plot_waterfall(sv, X_t, feats, base_value, tag, figs_dir):
    total  = sv.sum(axis=1)
    idxs   = [np.argmax(total), np.argmin(total),
               np.argsort(np.abs(total - total.mean()))[len(total) // 2]]
    labels_wf = ["highest", "lowest", "median"]
    feat_labels = [pretty(f) for f in feats]
    for idx, label in zip(idxs, labels_wf):
        sv_exp = shap.Explanation(
            values=sv[idx], base_values=base_value,
            data=X_t.iloc[idx].values, feature_names=feat_labels)
        fig, _ = plt.subplots(figsize=(7, max(4, 0.4 * len(feats))))
        shap.waterfall_plot(sv_exp, show=False, max_display=12)
        plt.title(f"SHAP Waterfall — {label} ({tag})",
                  fontsize=10, fontweight="bold")
        plt.tight_layout()
        out = Path(figs_dir) / f"shap_waterfall_{tag}_{label}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"  Saved waterfall plots for {tag}")


def plot_heatmap(sv, X_t, feats, title, out_path, n_feat=12):
    mean_abs = safe_importance(sv)
    top_idx  = np.argsort(mean_abs)[::-1][:n_feat]
    labels   = [pretty(feats[i]) for i in top_idx]
    sv_sub   = sv[:, top_idx]
    order    = np.argsort(sv_sub.sum(axis=1))
    sv_sorted = sv_sub[order]
    vmax     = np.percentile(np.abs(sv_sorted), 95)
    fig, ax  = plt.subplots(figsize=(10, max(5, 0.015 * sv.shape[0])))
    im = ax.imshow(sv_sorted.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Participants (sorted by SHAP sum)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="SHAP value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap → {Path(out_path).name}")


def plot_multimodel_importance(importance_dict, feat_names, title, out_path):
    # Ensure all arrays are 1D
    importance_dict = {
        k: (v.mean(axis=-1) if isinstance(v, np.ndarray) and v.ndim > 1 else v)
        for k, v in importance_dict.items()
    }
    df_imp = pd.DataFrame(importance_dict,
                           index=[pretty(f) for f in feat_names])
    df_imp["_mean"] = df_imp.mean(axis=1)
    df_imp = df_imp.sort_values("_mean", ascending=True).drop(columns="_mean")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(feat_names))))
    x = np.arange(len(df_imp))
    width = 0.8 / max(len(df_imp.columns), 1)
    for i, col in enumerate(df_imp.columns):
        offset = (i - len(df_imp.columns) / 2 + 0.5) * width
        ax.barh(x + offset, df_imp[col], height=width,
                label=col, color=MODEL_COLORS.get(col, "#999"), alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(df_imp.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved multi-model importance → {Path(out_path).name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    set_seed(cfg["seed"])
    apply_style()
    n_bg     = cfg["shap"]["background_samples"]
    n_top    = cfg["shap"]["n_top_features"]
    figs_dir = cfg["paths"]["figures_dir"]

    print("\n── SHAP Analysis ────────────────────────────────────────────────")

    # ── Continuous ─────────────────────────────────────────────────────────────
    print("\n  [Continuous — TAU_SUVR]")
    df_cont      = pd.read_csv(cfg["paths"]["processed_data"])
    outcome_cont = cfg["outcome"]["continuous"]
    y_cont       = df_cont[outcome_cont]

    excl = [outcome_cont, cfg["site_col"], cfg["id_col"],
            cfg["outcome"]["binary"], "APOE4_binary"]
    avail     = [c for c in df_cont.columns if c not in excl]
    pred_sets = build_predictor_sets(cfg, avail)
    models_c  = build_continuous_models(cfg)

    pset_name = "all" if "all" in pred_sets else list(pred_sets.keys())[-1]
    feats     = [f for f in pred_sets[pset_name] if f in df_cont.columns]
    X_cont    = df_cont[feats].copy()

    importance_cont = {}
    for model_name, model in models_c.items():
        print(f"    {model_name}...")
        pipe = build_pipeline(model, model_name in NEEDS_SCALING)
        pipe.fit(X_cont, y_cont)
        try:
            sv, X_t = compute_shap(pipe, X_cont, X_cont, model_name, n_bg)
            importance_cont[model_name] = safe_importance(sv)
            if model_name == list(models_c.keys())[0]:
                plot_beeswarm(sv, X_t, feats,
                              f"SHAP Summary — {model_name} (continuous)",
                              f"{figs_dir}shap_beeswarm_cont_{model_name}.png")
                plot_bar(sv, feats, f"Feature Importance — {model_name}",
                         f"{figs_dir}shap_bar_cont_{model_name}.png")
                plot_dependence(sv, X_t, feats, min(n_top, 4), figs_dir,
                                f"cont_{model_name}")
                try:
                    exp = shap.TreeExplainer(pipe.named_steps["model"])
                    bv  = float(exp.expected_value) if not isinstance(
                        exp.expected_value, np.ndarray) else float(exp.expected_value[0])
                except Exception:
                    bv = float(y_cont.mean())
                plot_waterfall(sv, X_t, feats, bv,
                               f"cont_{model_name}", figs_dir)
                plot_heatmap(sv, X_t, feats,
                             f"SHAP Heatmap — {model_name} (continuous)",
                             f"{figs_dir}shap_heatmap_cont_{model_name}.png",
                             n_feat=min(n_top, len(feats)))
        except Exception as e:
            print(f"    WARNING: SHAP failed for {model_name}: {e}")

    if importance_cont:
        plot_multimodel_importance(
            importance_cont, feats,
            "Multi-Model Feature Importance — Continuous",
            f"{figs_dir}shap_multimodel_cont.png")

    # ── Binary ─────────────────────────────────────────────────────────────────
    print("\n  [Binary — tau+/tau-]")
    df_bin      = pd.read_csv(cfg["paths"]["binary_data"])
    outcome_bin = cfg["outcome"]["binary"]
    y_bin       = df_bin[outcome_bin].astype(int)

    excl_bin  = [outcome_bin, outcome_cont, cfg["site_col"],
                 cfg["id_col"], "APOE4_binary"]
    avail_bin = [c for c in df_bin.columns if c not in excl_bin]
    pred_sets_bin = build_predictor_sets(cfg, avail_bin)
    n_pos = (y_bin == 1).sum(); n_neg = (y_bin == 0).sum()
    models_b = build_binary_models(cfg, scale_pos_weight=n_neg / max(n_pos, 1))

    pset_name_bin = "all" if "all" in pred_sets_bin else list(pred_sets_bin.keys())[-1]
    feats_bin = [f for f in pred_sets_bin[pset_name_bin] if f in df_bin.columns]
    X_bin     = df_bin[feats_bin].copy()

    importance_bin = {}
    for model_name, model in models_b.items():
        print(f"    {model_name}...")
        pipe = build_pipeline(model, model_name in NEEDS_SCALING)
        pipe.fit(X_bin, y_bin)
        try:
            sv, X_t = compute_shap(pipe, X_bin, X_bin, model_name, n_bg)
            importance_bin[model_name] = safe_importance(sv)
            if model_name == list(models_b.keys())[0]:
                plot_beeswarm(sv, X_t, feats_bin,
                              f"SHAP Summary — {model_name} (binary)",
                              f"{figs_dir}shap_beeswarm_bin_{model_name}.png")
                plot_bar(sv, feats_bin,
                         f"Feature Importance — {model_name} (binary)",
                         f"{figs_dir}shap_bar_bin_{model_name}.png")
                plot_dependence(sv, X_t, feats_bin, min(n_top, 4),
                                figs_dir, f"bin_{model_name}")
                try:
                    exp = shap.TreeExplainer(pipe.named_steps["model"])
                    bv  = float(exp.expected_value[1]) if isinstance(
                        exp.expected_value, np.ndarray) else float(exp.expected_value)
                except Exception:
                    bv = float(y_bin.mean())
                plot_waterfall(sv, X_t, feats_bin, bv,
                               f"bin_{model_name}", figs_dir)
                plot_heatmap(sv, X_t, feats_bin,
                             f"SHAP Heatmap — {model_name} (binary)",
                             f"{figs_dir}shap_heatmap_bin_{model_name}.png",
                             n_feat=min(n_top, len(feats_bin)))
        except Exception as e:
            print(f"    WARNING: SHAP failed for {model_name}: {e}")

    if importance_bin:
        plot_multimodel_importance(
            importance_bin, feats_bin,
            "Multi-Model Feature Importance — Binary",
            f"{figs_dir}shap_multimodel_bin.png")

    print("\n── Done :D ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
