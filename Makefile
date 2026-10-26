# Fake News & Misinformation Detector — Makefile (targets per master doc §24)
# Usage: make setup && make download && make data
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
PY      ?= .venv/bin/python
PIP_UV  := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
PYTHON_VERSION ?= 3.11
SPACY_MODEL_URL ?= https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
NOT_YET = @echo "make $@: not implemented until MSE2/ESE"

.PHONY: setup download data index train-baselines tune-baselines train-distilbert train-liar train models \
        eval-models eval-e2e eval run test paper docker-build docker-run nb-run lint clean-data help

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

setup: ## create .venv (Python 3.11 via uv, pip fallback), install requirements, spaCy en_core_web_sm (wheel URL), ipykernel
	@if [ ! -x $(PY) ]; then \
	  if [ -x "$(PIP_UV)" ]; then $(PIP_UV) venv .venv --python $(PYTHON_VERSION); \
	  else python3 -m venv .venv; fi; fi
	@if [ -x "$(PIP_UV)" ]; then $(PIP_UV) pip install --python $(PY) -r requirements.txt; \
	 else $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements.txt; fi
	@# `python -m spacy download` silently installs nowhere under a uv venv ("Audited 1 package"); install the wheel URL instead
	@if [ -x "$(PIP_UV)" ]; then $(PIP_UV) pip install --python $(PY) $(SPACY_MODEL_URL); \
	 else $(PY) -m pip install $(SPACY_MODEL_URL); fi
	$(PY) -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy en_core_web_sm OK')"
	$(PY) -m ipykernel install --user --name fakenews --display-name "Python (fakenews)"
	@echo "setup done: source .venv/bin/activate"

download: ## fetch raw datasets into data/raw/ (no Kaggle needed; see data/README.md)
	$(PY) scripts/download_data.py

data: ## unify + clean + dedup + split -> data/processed/*.parquet, data/splits/*.csv (log: docs/mse1_make_data.log)
	$(PY) -m src.preprocess.run_all 2>&1 | tee docs/mse1_make_data.log

nb-run: ## execute notebooks in place so outputs are saved (examiners need them)
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
	  --ExecutePreprocessor.kernel_name=fakenews notebooks/00_datasets.ipynb

test: ## run pytest
	$(PY) -m pytest tests

lint: ## ruff
	$(PY) -m ruff check src scripts tests

index: ## offline FAISS index: MiniLM over LIAR statements + FakeNewsNet PolitiFact titles -> data/index/ (~1.5 min CPU)
	$(PY) -m src.evidence.offline_index --build --query "Says the unemployment rate doubled under Barack Obama"
train-baselines: ## TF-IDF + NB/LR/SVM on WELFake + ISOT: MSE1 v0 config (tfidf_*_v0.joblib) + GridSearchCV winners from docs/results/gridsearch_*.csv (tfidf_{nb,lr,svm}.joblib, tfidf_*_isot.joblib) -> val logs + classifier_table.md (~8 min CPU)
	$(PY) -m src.models.baselines --dataset welfake --model all 2>&1 | tee docs/results/train_logs/train_baselines_welfake.log
	$(PY) -m src.models.baselines --dataset isot --model all 2>&1 | tee docs/results/train_logs/train_baselines_isot.log
	$(PY) -m src.models.baselines --dataset welfake --model all --tuned 2>&1 | tee docs/results/train_logs/train_baselines_tuned_welfake.log
	$(PY) -m src.models.baselines --dataset isot --model all --tuned 2>&1 | tee docs/results/train_logs/train_baselines_tuned_isot.log
	$(PY) -m src.models.baselines --render-table
tune-baselines: ## GridSearchCV (5-fold on WELFake train, macro-F1) NB 18 / LR 36 / SVM 36 configs -> docs/results/gridsearch_{nb,lr,svm}.csv (~2 h CPU, 1 core, peak ~3 GB; run before train-baselines)
	$(PY) -u -m src.models.baselines --grid-search --model all --dataset welfake 2>&1 | tee docs/results/train_logs/gridsearch_welfake.log
train-distilbert: ## DistilBERT on WELFake: 6-run sweep (20k subset) -> full retrain -> test eval -> sweep md/figures (GPU ~4.5 h; resumes)
	$(PY) -m src.models.distilbert --sweep --full --eval-test --render 2>&1 | tee docs/results/train_logs/train_distilbert.log
train-liar: ## DistilBERT LIAR 3-way head (lr x epochs sweep, class weights) -> data/models/distilbert-liar3-v1/, docs/results/liar_head.md (GPU ~20 min)
	$(PY) -m src.models.liar_head 2>&1 | tee docs/results/train_logs/train_liar.log
train: train-baselines train-distilbert train-liar ## umbrella: all three train-* targets
models: train index ## umbrella: train + index
eval-models: ## test metrics + CM/ROC/sweep figures (+ DistilBERT if data/models/distilbert-welfake-v1/ exists), classifier_table.md, error_analysis.md, cross_dataset.md (~15 min CPU; needs train-baselines)
	$(PY) -u -m src.eval.eval_models 2>&1 | tee docs/results/train_logs/eval_models.log
eval-e2e: ## end-to-end 5-class evaluation on the Live Claims Set (ESE)
	$(NOT_YET)
eval: eval-models eval-e2e ## everything
run: ## FastAPI on :8000 (MODE=offline make run = no network; CPU, < 4 GB RAM)
	$(PY) -m uvicorn src.api.main:app --host 127.0.0.1 --port $${PORT:-8000}
paper: ## build the IEEE paper -> paper/main.pdf (delegates to paper/Makefile: latexmk, tectonic fallback)
	$(MAKE) -C paper
docker-build: ## build the Docker image (ESE)
	$(NOT_YET)
docker-run: ## run the Docker image (ESE)
	$(NOT_YET)

clean-data: ## remove processed/splits (raw is kept)
	rm -f data/processed/*.parquet data/splits/*.csv

# --- baselines notebook (MSE1 Model Identification; appended by the baseline agent) ---
.PHONY: nb-run-baselines
nb-run-baselines: ## build notebooks/02_baselines.ipynb from scripts/build_notebook_02.py and execute it in place (~4 min CPU, outputs saved)
	$(PY) scripts/build_notebook_02.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
	  --ExecutePreprocessor.kernel_name=fakenews notebooks/02_baselines.ipynb

# --- MSE1 EDA (appended 2026-08-28): notebooks/01_eda.ipynb is generated by scripts/build_notebook_01.py; this target
#     executes it in place so the outputs are saved, writing docs/figures/eda_*.png and docs/eda_summary.md (≈ 8 min CPU)
.PHONY: eda
eda: ## execute notebooks/01_eda.ipynb in place -> docs/figures/eda_*.png + docs/eda_summary.md
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1500 \
	  --ExecutePreprocessor.kernel_name=fakenews notebooks/01_eda.ipynb

# --- MSE2 stance evaluation (appended by the evidence/API agent) ---
.PHONY: eval-stance nb-run-stance
eval-stance: ## nli-deberta-v3-small on 3k balanced FEVER dev pairs (gold text) -> docs/results/stance_fever.md (~CPU)
	$(PY) -m src.stance.fever_eval
nb-run-stance: ## build + execute notebooks/05_stance.ipynb in place
	$(PY) scripts/build_notebook_05.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 \
	  --ExecutePreprocessor.kernel_name=fakenews notebooks/05_stance.ipynb

# --- MSE2 transformers notebook (appended by the GPU-models agent): loads saved logs/checkpoints, never trains ---
.PHONY: nb-run-transformers
nb-run-transformers: ## build notebooks/03_transformers.ipynb from scripts/build_notebook_03.py and execute it in place (needs train-distilbert + train-liar)
	$(PY) scripts/build_notebook_03.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
	  --ExecutePreprocessor.kernel_name=fakenews notebooks/03_transformers.ipynb

# --- MSE2 tuning notebook (appended by the classical-models agent): reads gridsearch CSVs + eval logs, never trains ---
.PHONY: nb-run-tuning
nb-run-tuning: ## build notebooks/04_tuning.ipynb from scripts/build_notebook_04.py and execute it in place (needs tune-baselines + eval-models)
	$(PY) scripts/build_notebook_04.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
	  --ExecutePreprocessor.kernel_name=fakenews notebooks/04_tuning.ipynb
