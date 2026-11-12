# Fake News & Misinformation Detector - one CPU-only image for local Docker, Hugging Face Spaces (Docker SDK) and
# Render. Build from the repo root:  docker build -f docker/Dockerfile -t fakenews:latest .   (or `make docker-build`)
# Models are BAKED IN (docs/deployment.md "Why baked"): the container then runs with the network disabled in
# MODE=offline. Requires data/index/ (`make index`) and data/models/ (classifier) in the build context.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf TOKENIZERS_PARALLELISM=false MODE=online PORT=8000
WORKDIR /app

COPY docker/requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements-serve.txt \
 && pip install --no-deps lime==0.2.0.1 \
 && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
 && python -c "import spacy; spacy.load('en_core_web_sm')"

# Hub models (stance NLI + embedder) are fetched once at build time into HF_HOME, so runtime never needs the network.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); \
from transformers import AutoTokenizer as T, AutoModelForSequenceClassification as M; \
T.from_pretrained('cross-encoder/nli-deberta-v3-small'); M.from_pretrained('cross-encoder/nli-deberta-v3-small')"

# Everything the runtime needs is now local: never contact the Hub at runtime (also keeps --network none from hanging).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Project artefacts (the .dockerignore allow-list keeps the build context to these).
COPY --chmod=755 data/index data/index
COPY --chmod=755 data/models data/models
COPY src src
COPY app app
COPY tests/test_api.py tests/test_explain.py tests/

# HF Spaces runs the container as uid 1000; only the evidence cache must be writable.
RUN useradd -m -u 1000 user && mkdir -p data/cache && chown user data/cache
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
