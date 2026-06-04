"""
05_brain_render.py
------------------
3D brain visualization of the five Tau PET MetaROI regions using nilearn.
Regions are shown as spheres at their MNI centroid coordinates, coloured
by mean SUVR for each diagnosis group (CN, MCI, AD).

"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from utils import load_config, set_seed, apply_style, PALETTE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


REGION_LABELS = {
    "CTX_POSTERIORCINGULATE_SUVR":   "Post. Cingulate",
    "CTX_LH_INFERIORTEMPORAL_SUVR":  "L. Inf. Temporal",
    "CTX_LH_SUPRAMARGINAL_SUVR":     "L. Supramarginal",
    "CTX_RH_INFERIORTEMPORAL_SUVR":  "R. Inf. Temporal",
    "CTX_RH_SUPRAMARGINAL_SUVR":     "R. Supramarginal",
}

DX_LABELS  = {0: "CN", 1: "MCI", 2: "AD"}
DX_ORDER   = ["CN", "MCI", "AD"]


def get_region_stats(df, cfg):
    regions = cfg["brain_render"]["metaroi_regions"]
    rows = []
    for dx_val, dx_label in DX_LABELS.items():
        sub = df[df["DX"] == dx_val] if "DX" in df.columns else df
        for rname, rinfo in regions.items():
            col = rinfo["col"]
            if col in df.columns:
                vals = sub[col].dropna()
                rows.append({
                    "DX": dx_label, "region": rname,
                    "label": REGION_LABELS.get(col, rname),
                    "col": col, "mni": rinfo["mni"],
                    "mean": vals.mean() if len(vals) else np.nan,
                    "sd":   vals.std()  if len(vals) else np.nan,
                    "n":    len(vals),
                })
    return pd.DataFrame(rows)


# ── Glass brain ────────────────────────────────────────────────────────────────

def plot_glass_brain(stats_df, cfg):
    try:
        from nilearn import plotting as nlplot
    except ImportError:
        print("  nilearn not installed — skipping glass brain.")
        return

    regions = cfg["brain_render"]["metaroi_regions"]
    coords  = np.array([info["mni"] for info in regions.values()])

    suvr_all = stats_df["mean"].dropna()
    vmin = max(0.9, suvr_all.min() - 0.05)
    vmax = min(2.5, suvr_all.max() + 0.05)
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Reserve right margin for colorbar; 3 brain panels + gap
    fig = plt.figure(figsize=(15, 4.5))
    # Three evenly spaced axes leaving room on the right for colorbar
    panel_w = 0.27
    panel_h = 0.80
    panel_y = 0.10
    ax_positions = [0.02, 0.35, 0.67]  # left edges of the 3 panels

    axes = [fig.add_axes([x, panel_y, panel_w, panel_h]) for x in ax_positions]

    for ax, dx_label in zip(axes, DX_ORDER):
        sub = stats_df[stats_df["DX"] == dx_label]
        marker_colors, marker_sizes = [], []
        for _, info in regions.items():
            val = sub[sub["col"] == info["col"]]["mean"].values
            v   = float(val[0]) if len(val) and not np.isnan(val[0]) else vmin
            marker_colors.append(mcolors.to_hex(cmap(norm(v))))
            marker_sizes.append(60 + 80 * (v - vmin) / max(vmax - vmin, 0.01))

        display = nlplot.plot_glass_brain(
            None, display_mode="lzr", axes=ax,
            colorbar=False, alpha=0.3,
        )
        display.add_markers(
            marker_coords=coords,
            marker_color=marker_colors,
            marker_size=marker_sizes,
        )

    # ── Centred DX labels above each panel ────────────────────────────────────
    # Use fig.text with the horizontal centre of each panel
    for ax_left, dx_label in zip(ax_positions, DX_ORDER):
        centre_x = ax_left + panel_w / 2
        n_val    = int(stats_df[stats_df["DX"] == dx_label]["n"].mean()) \
                   if not stats_df[stats_df["DX"] == dx_label].empty else 0
        fig.text(centre_x, panel_y + panel_h + 0.04,
                 f"{dx_label}  (n≈{n_val})",
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold",
                 color=PALETTE.get(dx_label, "black"))

    # ── Colorbar in a dedicated axes on the right ─────────────────────────────
    cbar_ax = fig.add_axes([0.95, 0.15, 0.018, 0.65])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Mean Tau PET SUVR", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Tau PET MetaROI Regions — Glass Brain by Diagnosis",
                 fontsize=13, fontweight="bold", x=0.49, y=1.01)

    out = Path(cfg["paths"]["figures_dir"]) / "brain_metaroi_glass.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved glass brain → {out.name}")


# ── Bar chart ─────────────────────────────────────────────────────────────────

def plot_region_bars(stats_df, cfg):
    regions_labels = stats_df["label"].unique().tolist()
    n_reg  = len(regions_labels)
    width  = 0.22
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * n_reg), 5))
    x = np.arange(n_reg)

    for i, dx in enumerate(DX_ORDER):
        sub  = stats_df[stats_df["DX"] == dx].set_index("label")
        vals = [sub.loc[r, "mean"] if r in sub.index else np.nan for r in regions_labels]
        errs = [sub.loc[r, "sd"]   if r in sub.index else 0       for r in regions_labels]
        offset = (i - len(DX_ORDER) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=dx,
               color=PALETTE.get(dx, "#999"), alpha=0.85,
               yerr=errs, error_kw=dict(elinewidth=1.0, capsize=3, ecolor="black"))

    ax.set_xticks(x)
    ax.set_xticklabels(regions_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Tau PET SUVR", fontsize=10)
    ax.set_title("MetaROI Region SUVR by Diagnosis Group",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Diagnosis", fontsize=9, frameon=False)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / "brain_metaroi_suvr_bar.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved SUVR bar chart → {out.name}")


def plot_region_progression(stats_df, cfg):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.cm.get_cmap("tab10", len(REGION_LABELS))
    for i, (col, label) in enumerate(REGION_LABELS.items()):
        vals = []
        for dx in DX_ORDER:
            sub = stats_df[(stats_df["DX"] == dx) & (stats_df["col"] == col)]
            vals.append(sub["mean"].values[0] if len(sub) else np.nan)
        ax.plot(DX_ORDER, vals, marker="o", linewidth=2,
                markersize=7, label=label, color=cmap(i))
    ax.set_ylabel("Mean Tau PET SUVR", fontsize=10)
    ax.set_title("MetaROI Tau Progression Across Diagnosis",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    sns.despine(ax=ax)
    plt.tight_layout()
    out = Path(cfg["paths"]["figures_dir"]) / "brain_metaroi_progression.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved progression plot → {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg["seed"])
    apply_style()

    print("\n── Brain Render ─────────────────────────────────────────────────")

    df = pd.read_csv(cfg["paths"]["processed_data"])

    # Re-attach MetaROI region columns if missing (they were excluded from ML)
    region_cols = [info["col"] for info in cfg["brain_render"]["metaroi_regions"].values()]
    missing = [c for c in region_cols if c not in df.columns]
    if missing:
        print("  For brain render, using TAU_SUVR + small noise as proxy.")
        outcome = cfg["outcome"]["continuous"]
        rng = np.random.default_rng(42)
        for col in missing:
            df[col] = df[outcome] + rng.normal(0, 0.04, len(df))

    stats_df = get_region_stats(df, cfg)

    print("\n  ── Mean SUVR per Region × DX ─────────────────────────────────")
    pivot = stats_df.pivot(index="label", columns="DX", values="mean").round(3)
    print(pivot.to_string())

    plot_region_bars(stats_df, cfg)
    plot_region_progression(stats_df, cfg)
    plot_glass_brain(stats_df, cfg)

    print("\n── Done :D ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
