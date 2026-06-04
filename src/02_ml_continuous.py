"""
02_ml_continuous.py
-------------------
ML pipeline for continuous TAU_SUVR prediction.

Cross-validation
----------------
1. Repeated Stratified K-Fold (5 × 3) — stratified on DX × APOE4_binary
2. Leave-One-Site-Out (LOSO) — pooled predictions across all sites for R²;
   per-site Pearson r for the heatmap (because R² is unstable for small per-site n)
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from scipy.stats import pearsonr
from tqdm import tqdm

from utils import (
    load_config, set_seed, apply_style,
    build_predictor_sets, build_continuous_models, build_pipeline,
    continuous_metrics, fmt_mean_sd, PALETTE, MODEL_COLORS, NEEDS_SCALING,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


# ── CV functions ───────────────────────────────────────────────────────────────

def run_cv_standard(X, y, strat_key, model, model_name, cfg):
    records = []
    for repeat in range(cfg["cv"]["n_repeats"]):
        skf = StratifiedKFold(n_splits=cfg["cv"]["n_folds"],
                               shuffle=True, random_state=cfg["seed"] + repeat)
        for fold_i, (tr, te) in enumerate(skf.split(X, strat_key)):
            pipe = build_pipeline(model, model_name in NEEDS_SCALING)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            preds = pipe.predict(X.iloc[te])
            m = continuous_metrics(y.iloc[te], preds)
            m.update({"repeat": repeat, "fold": fold_i,
                       "cv_type": "stratified",
                       "n_train": len(tr), "n_test": len(te)})
            records.append(m)
    return records


def run_cv_loso(X, y, sites, model, model_name, cfg):
    """
    LOSO CV
      • per-site records with Pearson r 
      • one pooled record with full metrics computed across all OOF predictions
    """
    records  = []
    all_true, all_pred = [], []
    min_size = cfg["cv"]["min_site_size"]

    for tr, te in LeaveOneGroupOut().split(X, y, groups=sites):
        site = sites.iloc[te[0]]
        if len(te) < min_size:
            continue
        pipe = build_pipeline(model, model_name in NEEDS_SCALING)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        preds = pipe.predict(X.iloc[te])

        all_true.extend(y.iloc[te].tolist())
        all_pred.extend(preds.tolist())

        # Per-site: Pearson r (robust to small n)
        if len(te) > 2:
            r, _ = pearsonr(y.iloc[te], preds)
        else:
            r = np.nan
        records.append({"Pearson_r": r, "n_test": len(te),
                         "fold": site, "cv_type": "LOSO",
                         "repeat": 0, "n_train": len(tr)})

    # Pooled LOSO — compute all metrics from aggregated OOF predictions
    if all_true:
        pooled = continuous_metrics(np.array(all_true), np.array(all_pred))
        pooled.update({"fold": "POOLED", "cv_type": "LOSO_pooled",
                        "repeat": 0, "n_train": -1, "n_test": len(all_true)})
        records.append(pooled)

    return records


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_metric_boxes(fold_df, metric, cv_type, cfg):
    sub = fold_df[fold_df["cv_type"] == cv_type].copy()
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
    fig.suptitle(f"{metric} across folds — {cv_type}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"cont_boxes_{metric}_{cv_type}.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"  Saved → {out.name}")


def plot_fold_lines(fold_df, metric, cv_type, pred_set, cfg):
    sub = fold_df[(fold_df["cv_type"] == cv_type) &
                  (fold_df["predictor_set"] == pred_set)].copy()
    if sub.empty or metric not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in sub["model"].unique():
        mdata = sub[sub["model"] == model].sort_values("fold")
        mn = mdata.groupby("fold")[metric].mean().reset_index()
        ax.plot(mn["fold"].astype(str), mn[metric], marker="o",
                color=MODEL_COLORS.get(model, "#999"),
                linewidth=1.5, markersize=5, label=model)
    ax.set_xlabel("Fold", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f"{metric} per fold — {cv_type} — {pred_set}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"cont_foldline_{metric}_{cv_type}_{pred_set}.png"
    plt.savefig(out, dpi=300); plt.close()


def plot_scatter_best(df, X, y, model, model_name, pred_set, cfg):
    pipe = build_pipeline(model, model_name in NEEDS_SCALING)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    inv = {0: "CN", 1: "MCI", 2: "AD"}
    fig, ax = plt.subplots(figsize=(5, 5))
    for dx_val, label in inv.items():
        mask = (df["DX"] == dx_val) if "DX" in df.columns else pd.Series(True, index=df.index)
        ax.scatter(y[mask], preds[mask], c=PALETTE.get(label, "#999"),
                   s=20, alpha=0.5, label=label, edgecolors="none")
    lims = [min(y.min(), preds.min()) - 0.05, max(y.max(), preds.max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual TAU_SUVR", fontsize=10)
    ax.set_ylabel("Predicted TAU_SUVR", fontsize=10)
    ax.set_title(f"{model_name} — {pred_set}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"cont_scatter_{model_name}_{pred_set}.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"  Saved scatter → {out.name}")


def plot_summary_heatmap(summary_df, metric, cfg):
    if f"{metric}_mean" not in summary_df.columns:
        return
    pivot = summary_df.pivot(index="model", columns="predictor_set",
                              values=f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(pivot.columns)),
                                    1.2 * len(pivot)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"{metric} — Repeated Stratified CV (mean)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predictor Set"); ax.set_ylabel("Model")
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / f"cont_heatmap_{metric}.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"  Saved heatmap → {out.name}")


def plot_loso_site_heatmap(loso_df, cfg):
    """
    Heatmap of per-site Pearson r across models.
    """
    site_df = loso_df[loso_df["fold"] != "POOLED"].copy()
    if site_df.empty or "Pearson_r" not in site_df.columns:
        return
    pivot = site_df.groupby(["fold", "model"])["Pearson_r"].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(pivot.columns)),
                                    0.4 * len(pivot) + 1.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.3, ax=ax,
                vmin=-0.5, vmax=1.0,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    ax.set_title("LOSO Pearson r per Site\n"
                 "(R² shown for pooled LOSO in summary table)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Model"); ax.set_ylabel("Held-out Site")
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / "cont_loso_site_heatmap.png"
    plt.savefig(out, dpi=300); plt.close()
    print(f"  Saved LOSO heatmap → {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg["seed"])
    apply_style()

    print("\n── Continuous ML Pipeline ───────────────────────────────────────")

    df = pd.read_csv(cfg["paths"]["processed_data"])
    outcome  = cfg["outcome"]["continuous"]
    site_col = cfg["site_col"]

    y     = df[outcome]
    sites = df[site_col] if site_col in df.columns else None

    dx_key   = df["DX"].fillna(-1).astype(int).astype(str) if "DX" in df.columns \
               else pd.Series(["0"] * len(df), index=df.index)
    apoe_key = df["APOE4_binary"].fillna(0).astype(int).astype(str) \
               if "APOE4_binary" in df.columns \
               else pd.Series(["0"] * len(df), index=df.index)
    strat_key = dx_key + "_" + apoe_key

    excl = [outcome, site_col, cfg["id_col"],
            cfg["outcome"]["binary"], "APOE4_binary"]
    avail_cols = [c for c in df.columns if c not in excl]
    pred_sets  = build_predictor_sets(cfg, avail_cols)
    models_d   = build_continuous_models(cfg)

    all_records = []
    best_r2 = -np.inf
    best_combo = (None, None, None)

    pbar = tqdm(total=len(pred_sets) * len(models_d), desc="Training")
    for pset_name, feats in pred_sets.items():
        feats = [f for f in feats if f in df.columns]
        if not feats:
            pbar.update(len(models_d)); continue
        X = df[feats]

        for model_name, model in models_d.items():
            recs_std  = run_cv_standard(X, y, strat_key, model, model_name, cfg)
            recs_loso = run_cv_loso(X, y, sites, model, model_name, cfg) \
                        if sites is not None else []

            for r in recs_std + recs_loso:
                r.update({"model": model_name, "predictor_set": pset_name})
            all_records.extend(recs_std + recs_loso)

            std_r2s = [r["R2"] for r in recs_std if not np.isnan(r.get("R2", np.nan))]
            if std_r2s and np.mean(std_r2s) > best_r2:
                best_r2 = np.mean(std_r2s)
                best_combo = (pset_name, model_name, model)
            pbar.update(1)
    pbar.close()

    fold_df = pd.DataFrame(all_records)
    fold_df.to_csv(f"{cfg['paths']['tables_dir']}continuous_cv_metrics.csv", index=False)

    # ── Summary table ──────────────────────────────────────────────────────────
    metrics_list = ["R2", "RMSE", "MAE", "Pearson_r", "Spearman_r"]
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
    summary_df.to_csv(f"{cfg['paths']['tables_dir']}continuous_summary_table.csv",
                      index=False)

    # Pooled LOSO summary
    loso_pooled = fold_df[fold_df["cv_type"] == "LOSO_pooled"]
    if not loso_pooled.empty:
        print("\n  ── Pooled LOSO Metrics ─────────────────────────────────────────")
        for (model, pset), grp in loso_pooled.groupby(["model", "predictor_set"]):
            r2  = grp["R2"].mean()
            mae = grp["MAE"].mean()
            print(f"  {model:<30} {pset:<20} R²={r2:.3f}  MAE={mae:.3f}")

    pivot = summary_df.pivot(index="model", columns="predictor_set", values="R2_fmt") \
            if "R2_fmt" in summary_df.columns else pd.DataFrame()
    if not pivot.empty:
        print("\n  ── R² Summary (Stratified CV) ──────────────────────────────────")
        print(pivot.to_string())
    print(f"\n  Best: {best_combo[1]} | {best_combo[0]} → R²={best_r2:.3f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    for metric in ["R2", "RMSE", "MAE"]:
        plot_metric_boxes(fold_df, metric, "stratified", cfg)
        plot_fold_lines(fold_df, metric, "stratified", "all", cfg)
        plot_summary_heatmap(summary_df, metric, cfg)

    loso_df = fold_df[fold_df["cv_type"] == "LOSO"]
    plot_loso_site_heatmap(loso_df, cfg)

    if best_combo[0]:
        pset_name, model_name, model = best_combo
        feats = [f for f in pred_sets[pset_name] if f in df.columns]
        plot_scatter_best(df, df[feats], y, model, model_name, pset_name, cfg)

    print("\n── Done :D ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
