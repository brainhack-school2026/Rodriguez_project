#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Tau MetaROI ML Pipeline
# =============================================================================
# Usage
# -----
#   bash run_all.sh                          # synthetic data (default, GitHub)
#   bash run_all.sh --real \
#       --merged  /path/to/merged.csv \
#       --visual  /path/to/visual_reads.csv
#
# Options
#   --real              Use real ADNI data
#   --merged  <path>    Path to merged CSV
#   --visual  <path>    Path to Gothenburg visual reads CSV
#   --skip-shap         Skip SHAP analysis
#   --skip-brain        Skip brain render
#   --help              Show this message
# =============================================================================

set -euo pipefail

MODE="synthetic"
MERGED=""
VISUAL=""
SKIP_SHAP=false
SKIP_BRAIN=false
CONFIG="config.yaml"

while [[ $# -gt 0 ]]; do
  case $1 in
    --real)       MODE="real";      shift ;;
    --merged)     MERGED="$2";      shift 2 ;;
    --visual)     VISUAL="$2";      shift 2 ;;
    --skip-shap)  SKIP_SHAP=true;   shift ;;
    --skip-brain) SKIP_BRAIN=true;  shift ;;
    --config)     CONFIG="$2";      shift 2 ;;
    --help)
      head -20 "$0" | grep "^#" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

GREEN="\033[0;32m"; BLUE="\033[0;34m"; YELLOW="\033[1;33m"
RED="\033[0;31m";   NC="\033[0m"

step() { echo -e "\n${BLUE}══════════════════════════════════════════${NC}"; \
         echo -e "${BLUE}  $1${NC}"; \
         echo -e "${BLUE}══════════════════════════════════════════${NC}"; }
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
err()  { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

step "Tau MetaROI ML Pipeline"
echo -e "  Mode   : ${MODE}"
echo -e "  Config : ${CONFIG}"

command -v python >/dev/null 2>&1 || err "Python not found on PATH"

mkdir -p results/figures results/tables data/synthetic
ok "Output directories ready"

# ── Step 1: Data preparation ──────────────────────────────────────────────────
step "Step 1 — Data Preparation"
if [[ "$MODE" == "real" ]]; then
  [[ -z "$MERGED" ]] && err "--merged required for --real mode"
  [[ -z "$VISUAL" ]] && err "--visual required for --real mode"
  python src/00_data_prep.py --mode real \
    --merged "$MERGED" --visual_reads "$VISUAL" --config "$CONFIG"
else
  python src/00_data_prep.py --mode synthetic --config "$CONFIG"
fi
ok "Data prepared"

# ── Step 2: Continuous ML ─────────────────────────────────────────────────────
step "Step 2 — Continuous ML (TAU_SUVR prediction)"
python src/02_ml_continuous.py --config "$CONFIG"
ok "Continuous ML complete"

# ── Step 3: Binary ML ─────────────────────────────────────────────────────────
step "Step 3 — Binary ML (tau+/tau- classification)"
python src/03_ml_binary.py --config "$CONFIG"
ok "Binary ML complete"

# ── Step 4: SHAP ─────────────────────────────────────────────────────────────
if [[ "$SKIP_SHAP" == false ]]; then
  step "Step 4 — SHAP Analysis"
  python src/04_shap_analysis.py --config "$CONFIG"
  ok "SHAP complete"
else
  warn "Skipping SHAP (--skip-shap)"
fi

# ── Step 5: Brain render ──────────────────────────────────────────────────────
if [[ "$SKIP_BRAIN" == false ]]; then
  step "Step 5 — Brain Render"
  python src/05_brain_render.py --config "$CONFIG"
  ok "Brain render complete"
else
  warn "Skipping brain render (--skip-brain)"
fi

step "Pipeline Complete"
echo -e "  Figures → results/figures/"
echo -e "  Tables  → results/tables/"
echo -e "${GREEN}  All done!${NC}"
