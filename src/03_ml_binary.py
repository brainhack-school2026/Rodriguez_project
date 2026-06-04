"""
03_ml_binary.py
---------------
Binary tau+/tau- classification using  visual reads
Restricted to participants with a valid VISUAL_READ_BINARY label.

Threshold = 0.445 (Youden J optimal because of class imbalance: 64.1% tau-, 35.9% tau+).
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from tqdm import tqdm

from utils import (
    load_config, set_seed, apply_style,
    build_predictor_sets, build_binary_models, build_pipeline,
    binary_metrics, fmt_mean_sd, PALETTE, MODEL_COLORS, NEEDS_SCALING,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


# ── CV ─────────────────────────────────────────────────────────────────────────

def run_cv_standard_binary(X, y, strat_key, model, model_name, cfg):
    records, all_preds = [], []
    thr = cfg.get("binary_threshold", 0.445)
    for repeat in range(cfg["cv"]["n_repeats"]):
        skf = StratifiedKFold(n_splits=cfg["cv"]["n_folds"],
                               shuffle=True, random_state=cfg["seed"] + repeat)
        for fold_i, (tr, te) in enumerate(skf.split(X, strat_key)):
            pipe = build_pipeline(model, model_name in NEEDS_SCALING)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            probs = pipe.predict_proba(X.iloc[te])[:, 1]
            m = binary_metrics(y.iloc[te], probs, threshold=thr)
            m.update({"repeat": repeat, "fold": fold_i,
                       "cv_type": "stratified",
                       "n_train": len(tr), "n_test": len(te)})
            records.append(m)
            all_preds.append({"y_true": y.iloc[te].values,
                               "y_prob": probs, "fold": fold_i, "repeat": repeat})
    return records, all_preds


def run_cv_loso_binary(X, y, sites, model, model_name, cfg):
    records, all_preds = [], []
    thr      = cfg.get("binary_threshold", 0.445)
    min_size = cfg["cv"]["min_site_size"]
    for tr, te in LeaveOneGroupOut().split(X, y, groups=sites):
        site = sites.iloc[te[0]]
        if len(te) < min_size or len(np.unique(y.iloc[tr])) < 2:
            continue
        pipe = build_pipeline(model, model_name in NEEDS_SCALING)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        probs = pipe.predict_proba(X.iloc[te])[:, 1]
        m = binary_metrics(y.iloc[te], probs, threshold=thr)
        m.update({"repeat": 0, "fold": site, "cv_type": "LOSO",
                  "n_train": len(tr), "n_test": len(te)})
        records.append(m)
        all_preds.append({"y_true": y.iloc[te].values,
                           "y_prob": probs, "fold": site, "repeat": 0})
    return records, all_preds


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_roc_curves(all_preds_by_model, pred_set, cfg):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model_name, preds_list in all_preds_by_model.items():
        tprs, aucs = [], []
        base_fpr = np.linspace(0, 1, 101)
        for p in preds_list:
            if len(np.unique(p["y_true"])) < 2:
                continue
            fpr, tpr, _ = roc_curve(p["y_true"], p["y_prob"])
            tprs.append(np.interp(base_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
        if not aucs:
            continue
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs,  axis=0)
        color = MODEL_COLORS.get(model_name, "#999")
        ax.plot(base_fpr, mean_tpr, color=color, linewidth=1.8,
                label=f"{model_name} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
        ax.fill_between(base_fpr,
                         np.clip(mean_tpr - std_tpr, 0, 1),
                         np.clip(mean_tpr + std_tpr, 0, 1),
                         color=color, alpha=0.12)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("1 – Specificity (FPR)", fontsize=10)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=10)
    ax.set_title(f"ROC Curves — {pred_set}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"binary_roc_{pred_set}.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"  Saved ROC → {out.name}")


def plot_pr_curves(all_preds_by_model, pred_set, cfg):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model_name, preds_list in all_preds_by_model.items():
        precs, aucs = [], []
        base_rec = np.linspace(0, 1, 101)
        for p in preds_list:
            if len(np.unique(p["y_true"])) < 2:
                continue
            prec, rec, _ = precision_recall_curve(p["y_true"], p["y_prob"])
            precs.append(np.interp(base_rec, rec[::-1], prec[::-1]))
            aucs.append(auc(rec, prec))
        if not aucs:
            continue
        mean_prec = np.mean(precs, axis=0)
        color = MODEL_COLORS.get(model_name, "#999")
        ax.plot(base_rec, mean_prec, color=color, linewidth=1.8,
                label=f"{model_name} (AP={np.mean(aucs):.3f})")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(f"Precision-Recall Curves — {pred_set}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"binary_pr_{pred_set}.png"
    plt.savefig(out, dpi=300); plt.close()


def plot_calibration(all_preds_by_model, pred_set, cfg):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
    for model_name, preds_list in all_preds_by_model.items():
        y_true_all = np.concatenate([p["y_true"] for p in preds_list])
        y_prob_all = np.concatenate([p["y_prob"] for p in preds_list])
        if len(np.unique(y_true_all)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y_true_all, y_prob_all, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", markersize=4,
                color=MODEL_COLORS.get(model_name, "#999"),
                linewidth=1.5, label=model_name)
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction Positives", fontsize=10)
    ax.set_title(f"Calibration Curves — {pred_set}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"binary_calibration_{pred_set}.png"
    plt.savefig(out, dpi=300); plt.close()


def plot_metric_boxes_binary(fold_df, metric, cfg):
    sub = fold_df[fold_df["cv_type"] == "stratified"]
    if sub.empty or metric not in sub.columns:
        return
    model_names = sub["model"].unique().tolist()
    fig, axes = plt.subplots(1, len(model_names),
                              figsize=(4 * len(model_names), 4.5), sharey=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, model in zip(axes, model_names):
        mdata = sub[sub["model"] == model]
        pred_order = sorted(mdata["predictor_set"].unique())
        sns.boxplot(data=mdata, x="predictor_set", y=metric, order=pred_order,
                    color=MODEL_COLORS.get(model, "#888"), width=0.5,
                    linewidth=1.0,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4), ax=ax)
        sns.stripplot(data=mdata, x="predictor_set", y=metric, order=pred_order,
                      color=MODEL_COLORS.get(model, "#888"),
                      size=3.5, alpha=0.5, jitter=True, ax=ax,
                      edgecolor="black", linewidth=0.5)
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel(metric, fontsize=10)
        sns.despine(ax=ax)
    fig.suptitle(f"{metric} — Binary CV", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"binary_boxes_{metric}.png"
    plt.savefig(out, dpi=300); plt.close()


def plot_summary_heatmap_binary(summary_df, metric, cfg):
    if f"{metric}_mean" not in summary_df.columns:
        return
    pivot = summary_df.pivot(index="model", columns="predictor_set",
                              values=f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(pivot.columns)),
                                    1.2 * len(pivot)))
    cmap = "RdYlGn_r" if metric == "Brier" else "RdYlGn"
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"{metric} — Binary CV (mean)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predictor Set"); ax.set_ylabel("Model")
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"binary_heatmap_{metric}.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"  Saved heatmap → {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg["seed"])
    apply_style()

    print("\n── Binary ML Pipeline ───────────────────────────────────────────")

    df = pd.read_csv(cfg["paths"]["binary_data"])
    outcome  = cfg["outcome"]["binary"]
    site_col = cfg["site_col"]

    y     = df[outcome].astype(int)
    sites = df[site_col] if site_col in df.columns else None

    n_pos = (y == 1).sum(); n_neg = (y == 0).sum()
    print(f"  tau+: {n_pos} ({100*n_pos/len(y):.1f}%)  "
          f"tau-: {n_neg} ({100*n_neg/len(y):.1f}%)")
    print(f"  Imbalance ratio: {n_neg/max(n_pos,1):.2f}")
    print(f"  Classification threshold: {cfg.get('binary_threshold', 0.445)}")

    scale_pos_weight = n_neg / max(n_pos, 1)

    dx_key   = df["DX"].fillna(-1).astype(int).astype(str) if "DX" in df.columns \
               else pd.Series(["0"] * len(df), index=df.index)
    apoe_key = df["APOE4_binary"].fillna(0).astype(int).astype(str) \
               if "APOE4_binary" in df.columns \
               else pd.Series(["0"] * len(df), index=df.index)
    strat_key = y.astype(str) + "_" + dx_key + "_" + apoe_key

    excl = [outcome, cfg["outcome"]["continuous"], site_col,
            cfg["id_col"], "APOE4_binary"]
    avail_cols  = [c for c in df.columns if c not in excl]
    pred_sets   = build_predictor_sets(cfg, avail_cols)
    models_dict = build_binary_models(cfg, scale_pos_weight)

    all_records = []
    preds_by_pset = {}

    pbar = tqdm(total=len(pred_sets) * len(models_dict), desc="Training")
    for pset_name, feats in pred_sets.items():
        feats = [f for f in feats if f in df.columns]
        if not feats:
            pbar.update(len(models_dict)); continue
        X = df[feats]
        preds_by_pset[pset_name] = {}

        for model_name, model in models_dict.items():
            recs_std, preds_std   = run_cv_standard_binary(
                X, y, strat_key, model, model_name, cfg)
            recs_loso, preds_loso = run_cv_loso_binary(
                X, y, sites, model, model_name, cfg) if sites is not None else ([], [])

            for r in recs_std + recs_loso:
                r.update({"model": model_name, "predictor_set": pset_name})
            all_records.extend(recs_std + recs_loso)
            preds_by_pset[pset_name][model_name] = preds_std + preds_loso
            pbar.update(1)
    pbar.close()

    fold_df = pd.DataFrame(all_records)
    fold_df.to_csv(f"{cfg['paths']['tables_dir']}binary_cv_metrics.csv", index=False)

    # ── Summary table ──────────────────────────────────────────────────────────
    metrics_list = ["AUC_ROC", "AUC_PR", "Bal_Accuracy", "Accuracy",
                    "Sensitivity", "Specificity", "F1", "Brier"]
    summary_rows = []
    for (model, pset), grp in fold_df[fold_df["cv_type"]=="stratified"].groupby(
            ["model", "predictor_set"]):
        row = {"model": model, "predictor_set": pset}
        for m in metrics_list:
            if m in grp.columns:
                row[f"{m}_mean"] = grp[m].mean()
                row[f"{m}_sd"]   = grp[m].std()
                row[f"{m}_fmt"]  = fmt_mean_sd(grp[m], decimals=3)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{cfg['paths']['tables_dir']}binary_summary_table.csv",
                      index=False)

    pivot = summary_df.pivot(index="model", columns="predictor_set",
                              values="AUC_ROC_fmt") \
            if "AUC_ROC_fmt" in summary_df.columns else pd.DataFrame()
    if not pivot.empty:
        print("\n  ── AUC-ROC Summary ─────────────────────────────────────────────")
        print(pivot.to_string())

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    for metric in metrics_list:
        plot_metric_boxes_binary(fold_df, metric, cfg)
        plot_summary_heatmap_binary(summary_df, metric, cfg)

    for pset_name in ["all", "clinical_amyloid_fdg"] + list(pred_sets.keys()):
        if pset_name in preds_by_pset:
            plot_roc_curves(preds_by_pset[pset_name], pset_name, cfg)
            plot_pr_curves(preds_by_pset[pset_name], pset_name, cfg)
            plot_calibration(preds_by_pset[pset_name], pset_name, cfg)
            break

    print("\n── Done :D ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
