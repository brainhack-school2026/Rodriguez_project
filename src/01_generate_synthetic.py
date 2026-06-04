"""
01_generate_synthetic.py
------------------------
Generates synthetic data from real ADNI data using a
conditional multivariate Gaussian approach (one distribution per diagnosis group).
No external libraries are required :D

Modes
-----
--mode from_real   : Run on my real ADNI data to generate synthetic data.
                     Output on data/synthetic/ 
--mode synthetic   : Just validates if the existing synthetic file is present.
                     So everyone that download this GitHub should run this :D

Usage
-----
# Me (with real data) — I run once, commit the set output:
python src/01_generate_synthetic.py --mode from_real \
    --merged      path/to/merged_data.csv \
    --visual_reads path/to/GOTHENBURG_VISUAL_READS.csv

# GitHub users — data already present in repo, just verify:
python src/01_generate_synthetic.py --mode synthetic
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from utils import load_config, setup_dirs, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["from_real", "synthetic"], default="synthetic")
    p.add_argument("--merged",        default=None)
    p.add_argument("--visual_reads",  default=None)
    p.add_argument("--n_samples",     type=int, default=None,
                   help="Number of synthetic samples (default: same as real data)")
    p.add_argument("--config",        default="config.yaml")
    return p.parse_args()


# ── Conditional multivariate Gaussian ─────────────────────────────────────────

def generate_conditional_mvn(df: pd.DataFrame, cfg: dict,
                              n_samples: int = None,
                              seed: int = 42) -> pd.DataFrame:
    """
    For each DX group (CN=0, MCI=1, AD=2):
      1. Compute group mean vector and covariance matrix
      2. Sample from multivariate Gaussian
      3. Clip to observed min/max per variable
      4. Sample binary/ordinal variables from group proportions

    """
    rng = np.random.default_rng(seed)
    df  = df.copy()

    # Rename dot columns
    rename = cfg.get("rename_cols", {})
    df = df.rename(columns=rename)

    # Column groups
    numeric_cols = [c for c in [
        'AGE_CORRECTED', 'PTEDUCAT', 'MMSE', 'CDRSB',
        'TAU_SUVR', 'SUMMARY_SUVR_AbPET', 'CENTILOIDS_AbPET',
        'MEAN_FDG', 'HCI', 'SROI_AD', 'SROI_MCI',
        'WM_HYPOINTENSITIES_SUVR',
        'CTX_POSTERIORCINGULATE_SUVR',
        'CTX_LH_INFERIORTEMPORAL_SUVR',
        'CTX_LH_SUPRAMARGINAL_SUVR',
        'CTX_RH_INFERIORTEMPORAL_SUVR',
        'CTX_RH_SUPRAMARGINAL_SUVR',
    ] if c in df.columns]

    binary_cols  = [c for c in ['PTGENDER', 'AMYLOID_STATUS_AbPET'] if c in df.columns]
    ordinal_cols = [c for c in ['APOE4'] if c in df.columns]

    # Recode DX for grouping
    dx_bl_map = {"LMCI": "MCI", "EMCI": "MCI", "AD": "AD", "CN": "CN", "SMC": "CN"}
    dx_map    = {"Dementia": "AD", "MCI": "MCI", "CN": "CN"}
    if "DX_bl" in df.columns:
        df["DX_bl"] = df["DX_bl"].map(lambda x: dx_bl_map.get(str(x), x))
    if "DX" in df.columns:
        df["DX"] = df["DX"].map(lambda x: dx_map.get(str(x), x))

    # Encode DX
    ordinal = {"CN": 0, "MCI": 1, "AD": 2}
    if "DX" in df.columns:
        df["DX_encoded"] = df["DX"].map(ordinal)
    else:
        df["DX_encoded"] = 1  # fallback

    # Global min/max for clipping
    mins = df[numeric_cols].min()
    maxs = df[numeric_cols].max()

    # DX proportions
    dx_counts = df["DX_encoded"].value_counts(normalize=True).sort_index()
    n_total   = n_samples or len(df)
    dx_groups = {0: "CN", 1: "MCI", 2: "AD"}

    all_rows = []

    for dx_val, dx_label in dx_groups.items():
        sub = df[df["DX_encoded"] == dx_val]
        if len(sub) < 5:
            continue

        prop  = dx_counts.get(dx_val, 0)
        n_dx  = round(n_total * prop)
        if n_dx == 0:
            continue

        # Fit multivariate Gaussian on complete cases
        sub_num = sub[numeric_cols].dropna()
        mu  = sub_num.mean().values
        cov = np.cov(sub_num.T)
        cov += np.eye(len(mu)) * 1e-6  # ensure positive definite

        # Sample
        samples = rng.multivariate_normal(mu, cov, size=n_dx)
        sdf = pd.DataFrame(samples, columns=numeric_cols)

        # Clip to observed range
        for col in numeric_cols:
            sdf[col] = sdf[col].clip(mins[col], maxs[col])

        # Rounding to match clinical data precision
        for col in ['MMSE', 'PTEDUCAT']:
            if col in sdf.columns:
                sdf[col] = sdf[col].round(0).astype(float)
        for col in ['CDRSB']:
            if col in sdf.columns:
                sdf[col] = sdf[col].round(1).clip(0, None)
        for col in ['TAU_SUVR', 'MEAN_FDG', 'SUMMARY_SUVR_AbPET',
                    'CENTILOIDS_AbPET']:
            if col in sdf.columns:
                sdf[col] = sdf[col].round(3)
        for col in ['HCI', 'SROI_AD', 'SROI_MCI']:
            if col in sdf.columns:
                sdf[col] = sdf[col].round(2)

        # Introduce realistic missingness, I'm setting this to 0 for now.
        miss_rates = {
            'CDRSB': 0.00, 'MMSE': 0.00, 'HCI': 0.00,
            'SROI_AD': 0.00, 'SROI_MCI': 0.00,
            'SUMMARY_SUVR_AbPET': 0.00, 'CENTILOIDS_AbPET': 0.00,
        }
        for col, rate in miss_rates.items():
            if col in sdf.columns:
                sdf.loc[rng.random(n_dx) < rate, col] = np.nan

        # Binary variables: sample from group proportions
        for col in binary_cols:
            col_numeric = pd.to_numeric(sub[col], errors="coerce")
            p = col_numeric.mean()
            if not np.isnan(p):
                sdf[col] = rng.binomial(1, p, n_dx).astype(float)

        # APOE4: sample from group distribution
        for col in ordinal_cols:
            vals  = sub[col].dropna()
            probs = vals.value_counts(normalize=True).reindex([0,1,2], fill_value=0).values
            probs = probs / probs.sum()
            sdf[col] = rng.choice([0,1,2], size=n_dx, p=probs)

        # VISUAL_READ_BINARY: sample from group proportion, ~20% missing
        if "VISUAL_READ_BINARY" in df.columns:
            p_pos = sub["VISUAL_READ_BINARY"].mean()
            if not np.isnan(p_pos):
                vr = rng.binomial(1, p_pos, n_dx).astype(float)
                vr[rng.random(n_dx) < 0.20] = np.nan
                sdf["VISUAL_READ_BINARY"] = vr

        sdf["DX"]    = dx_val
        sdf["DX_bl"] = dx_val
        all_rows.append(sdf)

    synthetic = pd.concat(all_rows, ignore_index=True)

    # SITEID: sample from real site distribution
    if "SITEID" in df.columns:
        sites = df["SITEID"].dropna().values
        synthetic["SITEID"] = rng.choice(sites, size=len(synthetic))

    synthetic["RID"] = np.arange(1, len(synthetic) + 1)

    return synthetic


def validate_synthetic(synthetic: pd.DataFrame, real: pd.DataFrame) -> None:
    """Print a brief comparison of key stats."""
    print("\n  ── Validation: Real vs Synthetic ────────────────────────────────")
    for col in ["TAU_SUVR", "MEAN_FDG", "AGE_CORRECTED", "CDRSB"]:
        if col in real.columns and col in synthetic.columns:
            rm = real[col].mean(); rs = real[col].std()
            sm = synthetic[col].mean(); ss = synthetic[col].std()
            print(f"  {col:<30} real={rm:.3f}±{rs:.3f}  synth={sm:.3f}±{ss:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_dirs(cfg)
    set_seed(cfg["seed"])
    out_path = cfg["paths"]["synthetic_data"]

    print("\n── Synthetic Data Generation ────────────────────────────────────")

    if args.mode == "synthetic":
        if Path(out_path).exists():
            df = pd.read_csv(out_path)
            print(f"  Synthetic data present: {out_path}")
            print(f"  Shape: {df.shape}")
        else:
            print("  ERROR: Synthetic data not found at:", out_path)
            print("  Generate it first with:")
            print("    python src/01_generate_synthetic.py --mode from_real \\")
            print("      --merged path/to/data.csv --visual_reads path/to/vr.csv")
            sys.exit(1)
        return

    # from_real mode
    if not args.merged or not args.visual_reads:
        print("ERROR: --merged and --visual_reads required for --mode from_real")
        sys.exit(1)

    print("  Loading real data...")
    df_real = pd.read_csv(args.merged, low_memory=False)
    df_real.columns = df_real.columns.str.strip()

    vr = pd.read_csv(args.visual_reads, low_memory=False)
    vr.columns = vr.columns.str.strip()
    id_col = cfg["id_col"]
    vr = vr[[id_col, "VISUAL_READ_BINARY"]].drop_duplicates(subset=id_col)
    df_real = df_real.merge(vr, on=id_col, how="left")
    print(f"  Real data: {len(df_real)} participants")

    n = args.n_samples or len(df_real)
    print(f"  Generating {n} synthetic participants via conditional MVN...")

    synthetic = generate_conditional_mvn(df_real, cfg, n_samples=n, seed=cfg["seed"])
    validate_synthetic(synthetic, df_real)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print(f"  Shape : {synthetic.shape}")

    dx_map = {0: "CN", 1: "MCI", 2: "AD"}
    for v, l in dx_map.items():
        n = (synthetic["DX"] == v).sum()
        print(f"  {l}: {n} ({100*n/len(synthetic):.1f}%)")

    if "VISUAL_READ_BINARY" in synthetic.columns:
        n_vr = synthetic["VISUAL_READ_BINARY"].notna().sum()
        n_pos = (synthetic["VISUAL_READ_BINARY"] == 1).sum()
        print(f"  Visual reads: {n_vr} | tau+: {n_pos}")

    print("\n── Done :D ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
