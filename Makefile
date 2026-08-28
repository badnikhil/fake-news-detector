# Fake News & Misinformation Detector — Makefile (targets per master doc §24)
# Usage: make setup && make download && make data
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
PY      ?= .venv/bin/python
PIP_UV  := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
PYTHON_VERSION ?= 3.11
NOT_YET = @echo "make $@: not implemented until MSE2/ESE"

.PHONY: setup download data index train-baselines train-distilbert train-liar train models \
        eval-models eval-e2e eval run test paper docker-build docker-run nb-run lint clean-data help

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

setup: ## create .venv (Python 3.11 via uv, pip fallback), install requirements, spaCy model, ipykernel
	@if [ ! -x $(PY) ]; then \
	  if [ -x "$(PIP_UV)" ]; then $(PIP_UV) venv .venv --python $(PYTHON_VERSION); \
	  else python3 -m venv .venv; fi; fi
	@if [ -x "$(PIP_UV)" ]; then $(PIP_UV) pip install --python $(PY) -r requirements.txt; \
	 else $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements.txt; fi
	-$(PY) -m spacy download en_core_web_sm
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

index: ## build the offline FAISS fact-check index (MSE2)
	$(NOT_YET)
train-baselines: ## TF-IDF + NB/LR/SVM (MSE1 LR baseline lives in notebooks/02_baselines.ipynb; full grid at MSE2)
	$(NOT_YET)
train-distilbert: ## fine-tune DistilBERT on WELFake (MSE2)
	$(NOT_YET)
train-liar: ## LIAR 3-way head (MSE2)
	$(NOT_YET)
train: train-baselines train-distilbert train-liar ## umbrella: all three train-* targets
models: train index ## umbrella: train + index
eval-models: ## classifier metrics, cross-dataset, error analysis (MSE2)
	$(NOT_YET)
eval-e2e: ## end-to-end 5-class evaluation on the Live Claims Set (ESE)
	$(NOT_YET)
eval: eval-models eval-e2e ## everything
run: ## uvicorn src.api.main:app --port 8000 (ESE)
	$(NOT_YET)
paper: ## build the IEEE paper -> paper/main.pdf (delegates to paper/Makefile: latexmk, tectonic fallback)
	$(MAKE) -C paper
docker-build: ## build the Docker image (ESE)
	$(NOT_YET)
docker-run: ## run the Docker image (ESE)
	$(NOT_YET)

clean-data: ## remove processed/splits (raw is kept)
	rm -f data/processed/*.parquet data/splits/*.csv
