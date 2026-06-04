PYTHON = python
CONFIG = config.yaml

.PHONY: all synthetic real install clean help

help:
	@echo ""
	@echo "  make install     Install dependencies"
	@echo "  make synthetic   Run full pipeline on synthetic data (default)"
	@echo "  make real        Run on real data: make real MERGED=... VISUAL=..."
	@echo "  make prep        Data preparation only"
	@echo "  make ml-cont     Continuous ML only"
	@echo "  make ml-bin      Binary ML only"
	@echo "  make shap        SHAP analysis only"
	@echo "  make brain       Brain render only"
	@echo "  make clean       Remove generated results"
	@echo ""

install:
	pip install -r requirements.txt

synthetic:
	bash run_all.sh

real:
	bash run_all.sh --real --merged $(MERGED) --visual $(VISUAL)

prep:
	$(PYTHON) src/00_data_prep.py --mode synthetic --config $(CONFIG)

ml-cont:
	$(PYTHON) src/02_ml_continuous.py --config $(CONFIG)

ml-bin:
	$(PYTHON) src/03_ml_binary.py --config $(CONFIG)

shap:
	$(PYTHON) src/04_shap_analysis.py --config $(CONFIG)

brain:
	$(PYTHON) src/05_brain_render.py --config $(CONFIG)

clean:
	rm -rf results/figures/* results/tables/* data/processed_data.csv data/binary_data.csv
