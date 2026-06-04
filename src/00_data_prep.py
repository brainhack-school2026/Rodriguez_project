"""
00_data_prep.py
---------------
Loads, cleans, and preprocesses the data

how to use
-----
#my ADNI data:
python src/00_data_prep.py --mode real \
    --merged  path/to/merged_data.csv \
    --visual_reads path/to/GOTHENBURG_VISUAL_READS.csv

#synthetic data:
python src/00_data_prep.py --mode synthetic
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from utils import load_config, setup_dirs, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["real", "synthetic"], default="synthetic")
    p.add_argument("--merged",       default=None)
    p.add_argument("--visual_reads", default=None)
    p.add_argument("--config",       default="config.yaml")
    return p.parse_args()


# ── Loading ────────────────────────────────────────────────────────────────────

def load_real_data(merged_path, visual_path, cfg):
    print(f"  Loading merged data   : {merged_path}")
    df = pd.read_csv(merged_path, low_memory=False)
    df.columns = df.columns.str.strip()         
    print(f"  → {len(df)} participants, {df.shape[1]} columns")

    print(f"  Loading visual reads  : {visual_path}")
    vr = pd.read_csv(visual_path, low_memory=False)
    vr.columns = vr.columns.str.strip()
    id_col = cfg["id_col"]
    vr = vr[[id_col, "VISUAL_READ_BINARY"]].drop_duplicates(subset=id_col)
    print(f"  → {len(vr)} participants with visual reads")
    return df, vr


def load_synthetic_data(cfg):
    path = cfg["paths"]["synthetic_data"]
    if not Path(path).exists():
        print("  Synthetic data not found. Run:")
        print("    python src/01_generate_synthetic.py --mode from_real ...")
        sys.exit(1)
    print(f"  Loading synthetic data : {path}")
    df = pd.read_csv(path)
    id_col = cfg["id_col"]
    vr = df[[id_col, "VISUAL_READ_BINARY"]].drop_duplicates(subset=id_col).copy()
    print(f"  → {len(df)} participants")
    return df, vr


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # Rename dot columns (XGBoost incompatibility)
    df = df.rename(columns=cfg.get("rename_cols", {}))

    # Recode DX_bl & DX
    dx_bl_map = {"LMCI": "MCI", "EMCI": "MCI", "AD": "AD", "CN": "CN", "SMC": "CN"}
    dx_map    = {"Dementia": "AD", "MCI": "MCI", "CN": "CN"}
    if "DX_bl" in df.columns:
        df["DX_bl"] = df["DX_bl"].map(lambda x: dx_bl_map.get(str(x), x))
    if "DX" in df.columns:
        df["DX"] = df["DX"].map(lambda x: dx_map.get(str(x), x))

    # Sex to numeric
    gender_map = {"Female": 0, "Male": 1, "F": 0, "M": 1,
                  "1": 0, "2": 1, 1: 0, 2: 1}
    if "PTGENDER" in df.columns:
        df["PTGENDER"] = df["PTGENDER"].map(gender_map)

    # Ordinal-encode DX
    ordinal = cfg.get("ordinal_dx_map", {"CN": 0, "MCI": 1, "AD": 2})
    for col in ["DX", "DX_bl"]:
        if col in df.columns:
            df[col] = df[col].map(ordinal)

    # APOE4_binary for stratification
    if "APOE4" in df.columns:
        df["APOE4_binary"] = (df["APOE4"] > 0).astype(int)

    # Drop leakage + ID columns
    to_drop = (cfg.get("leakage_cols", []) + cfg.get("drop_cols", []))
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    return df


def make_binary_df(df: pd.DataFrame, vr: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Merge visual reads and restrict to participants with a valid label."""
    id_col     = cfg["id_col"]
    binary_col = cfg["outcome"]["binary"]

    if binary_col in df.columns:
        sub = df[df[binary_col].notna()].copy()
    else:
        sub = df.merge(vr, on=id_col, how="left")
        sub = sub[sub[binary_col].notna()].copy()

    sub[binary_col] = sub[binary_col].astype(int)
    n_pos = (sub[binary_col] == 1).sum()
    n_neg = (sub[binary_col] == 0).sum()
    print(f"  Binary subset : {len(sub)} participants "
          f"(tau+={n_pos} [{100*n_pos/len(sub):.1f}%], "
          f"tau-={n_neg} [{100*n_neg/len(sub):.1f}%])")
    return sub


def print_summary(df, label, cfg):
    outcome  = cfg["outcome"]["continuous"]
    site_col = cfg["site_col"]
    print(f"\n  ── {label} {'─'*(45-len(label))}")
    print(f"  N participants : {len(df)}")
    if site_col in df.columns:
        print(f"  N sites        : {df[site_col].nunique()}")
    if outcome in df.columns:
        t = df[outcome].dropna()
        print(f"  TAU_SUVR       : {t.mean():.3f} ± {t.std():.3f} "
              f"[{t.min():.3f} – {t.max():.3f}]")
    if "DX" in df.columns:
        inv = {0: "CN", 1: "MCI", 2: "AD"}
        for v, l in inv.items():
            n = (df["DX"] == v).sum()
            print(f"  {l:<4}           : {n} ({100*n/len(df):.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_dirs(cfg)
    set_seed(cfg["seed"])

    print("\n── Data Preparation ─────────────────────────────────────────────")

    if args.mode == "real":
        if not args.merged or not args.visual_reads:
            print("ERROR: --merged and --visual_reads required for --mode real")
            sys.exit(1)
        df, vr = load_real_data(args.merged, args.visual_reads, cfg)
    else:
        df, vr = load_synthetic_data(cfg)

    # ── Binary merge before preprocessing ─────────────────
    bin_df_raw = make_binary_df(df, vr, cfg)

    # ── Preprocess ───────────────────────────────────────────────────────────
    df         = preprocess(df, cfg)
    bin_df     = preprocess(bin_df_raw, cfg)

    # ── Continuous dataset ────────────────────────────────────────────────────
    outcome = cfg["outcome"]["continuous"]
    cont_df = df[df[outcome].notna()].copy()
    print_summary(cont_df, "Continuous outcome dataset", cfg)
    cont_df.to_csv(cfg["paths"]["processed_data"], index=False)
    print(f"\n  Saved → {cfg['paths']['processed_data']}")

    # ── Binary dataset ────────────────────────────────────────────────────────
    bin_df = bin_df[bin_df[outcome].notna()].copy()
    bin_df.to_csv(cfg["paths"]["binary_data"], index=False)
    print(f"  Saved → {cfg['paths']['binary_data']}")

    print("\n── Done :D ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
