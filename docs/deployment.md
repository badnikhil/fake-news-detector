# Deployment (ESE §5)

One CPU-only Docker image (`docker/Dockerfile`) serves the API **and** the front-end (`app/`) on port 8000. It is used
for local Docker and is the image intended for the Hugging Face Space. All numbers below were **measured on the dev
laptop on 2026-11-12** while the CPU was shared with other jobs, so treat the latencies as upper bounds.

## Quick start

```bash
make index                      # data/index/ (offline fact-check index) - needed in the build context
make docker-build               # = nice -n 19 docker build -f docker/Dockerfile -t fakenews:latest .   (~6 min, needs network)
make docker-run                 # http://localhost:8000   (MODE=online)
MODE=offline make docker-run    # no network calls at all
docker compose -f docker/compose.yml up --build      # same, with a named volume for the online evidence cache
```

Network-disabled demo (what the examiner checks):

```bash
docker run -d --name fakenews --network none -e MODE=offline fakenews:latest
docker exec fakenews python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read())"
docker exec -w /app fakenews python -m pytest -q -p no:cacheprovider tests     # API + explanation tests inside the container
```

`--network none` leaves no port to publish, so these checks run inside the container. For a browser demo without
internet, run `MODE=offline make docker-run` and turn Wi-Fi off; `MODE=offline` makes no network calls.

## Why the models are baked into the image

| Option | Pros | Cons |
|---|---|---|
| **Baked in (chosen)** | `docker run` works with no network and no volume set-up; the HF Space and laptop run the same image. | Rebuild to change a model; image ≈ 3.1 GB. |
| Mounted volume | Smaller image, swap models without a rebuild. | Each machine has to fetch the models first, the HF Space has no host volume, and offline first-run fails. |

What gets baked in: `data/models/distilbert-welfake-v1/` (served classifier, if present at build time) and
`data/models/tfidf_lr_welfake_v0.joblib` (fallback), `data/index/` (FAISS + metadata), and the Hub models
`cross-encoder/nli-deberta-v3-small` + `sentence-transformers/all-MiniLM-L6-v2`, downloaded at build time into
`HF_HOME=/app/.hf`; after that the image sets `HF_HUB_OFFLINE=1` so the runtime never contacts the Hub (without it, a cold `SentenceTransformer` load hangs under `--network none`). `.dockerignore` is an allow-list, so the build context is only ≈ 294 MB (no raw data, venv or
notebooks).

## Measured footprint (2026-11-12)

| Item | Value |
|---|---|
| Image `fakenews:latest`, uncompressed layers | ≈ 3.1 GB (base 0.15 GB + Python deps 2.04 GB [torch CPU 0.77 GB] + HF models 0.67 GB + our models 0.27 GB + index 0.02 GB). `docker images` shows 4.63 GB "disk usage" because it counts the compressed blobs too; content size is 1.48 GB. |
| Model footprint on disk | DistilBERT-WELFake 259 MB, nli-deberta-v3-small ≈ 553 MB, MiniLM ≈ 88 MB, TF-IDF LR 2 MB, FAISS index 22 MB, spaCy sm ≈ 13 MB → **≈ 0.94 GB** |
| Build time | 6 min (`nice -n 19`, cold pip cache) |
| Peak RSS (uvicorn, `MODE=offline`, `--network none`, after the 5 gallery inputs ×2 + LIME) | **1.33 GB** (VmHWM 1,328,884 kB) — within the 4 GB budget (NFR-6) |
| `/analyze` latency, offline, in container | first request 6.8 s (loads the models), then median of the 5 gallery inputs **0.50 s** (first pass) / **0.22 s** (repeat); O7 target ≤ 5 s offline |
| `/explain` (LIME) | DistilBERT, 80 samples over the first 60 words: headlines 0.55–0.78 s, gallery articles 2.07–2.12 s (100 samples measured 2.7–3.4 s on the contended CPU, hence 80). TF-IDF LR fallback: 500 samples in < 0.05 s. Repeats are cached (0 ms). |
| Gallery check, container offline with no network | REAL R7 · FAKE R2 · PARTIALLY TRUE R5 · MISLEADING R4 · UNVERIFIABLE R1 — same as on the host (fusion tau 0.8 / min_evidence 3) |
| `pytest tests` inside the container, `--network none` | **23 passed** in 12.9 s (`tests/test_api.py` + `tests/test_explain.py`, incl. the gallery regression test) |

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MODE` | `online` | `online` = offline index + DuckDuckGo/Wikipedia (+ Google Fact Check if keyed); `offline` = index only, no network, URL input rejected (422) |
| `PORT` | `8000` | uvicorn port inside the container |
| `GOOGLE_FACTCHECK_API_KEY` | unset | optional; pass it as an env var or HF secret. Never put it in a file. `/config` only reports whether it is set. |

## Hugging Face Spaces (primary demo): steps for the team

Not deployed by the agent. To deploy:

1. On huggingface.co: **New Space** → SDK **Docker** → hardware **CPU basic (free, 2 vCPU / 16 GB)**.
2. `git clone https://huggingface.co/spaces/<team>/<space>` and copy into it: `src/`, `app/`, `tests/test_api.py`,
   `tests/test_explain.py`, `docker/requirements-serve.txt`, `.dockerignore`, `docker/Dockerfile` → **`Dockerfile` at
   the Space root**, and `docker/hf_space/README.md` → **`README.md` at the Space root** (its front-matter sets
   `sdk: docker`, `app_port: 8000`).
3. Models and index: `git lfs install && git lfs track "*.safetensors" "*.faiss" "*.joblib" "*.parquet"`, then copy
   `data/index/` and `data/models/distilbert-welfake-v1/` (without `trainer_state.json` / `training_args.json`)
   plus `data/models/tfidf_lr_welfake_v0.joblib`. Commit and push. The Space builds the same Dockerfile, which pulls
   DeBERTa and MiniLM from the Hub at build time.
4. Space **Settings → Variables and secrets**: add secret `GOOGLE_FACTCHECK_API_KEY` (optional) and variable `MODE`
   (`online`).
5. Check: `https://<team>-<space>.hf.space/health` returns `status: ok` with the model names. Open the Space and run
   an example. Record the Space URL and the measured latency here.
6. Wake-up: free Spaces sleep after inactivity, and the first request after waking takes about a minute. Open the URL
   ≈ 10 min before the viva and run one example to load the models.

## Render free tier (`MODE=lite` backup): read the gap first

Render free web services have **512 MB RAM**. Today `MODE=lite` only skips online retrieval; the pipeline still
loads DeBERTa-NLI, MiniLM and the classifier (≈ 1.3 GB RSS), so **it will not fit in 512 MB**. A real lite path would
need the pipeline to skip NLI (stance from fact-check ratings only) and serve TF-IDF instead of DistilBERT. That is a
change to `src/api/pipeline.py` and has not been made. Until then, there are two options:

- deploy the same image on a Render instance with ≥ 2 GB (paid), or
- leave Render out and use the fallback chain: HF Space → local Docker over a phone hotspot → `MODE=offline` on the
  laptop → recorded demo.

Steps, if a ≥ 2 GB instance is used: Render → **New Web Service** → connect the GitHub repo → runtime **Docker**,
Dockerfile path `docker/Dockerfile`, docker context `.` → env `MODE=offline` (or `online`) and `PORT=8000`. The repo
does not track `data/models/` or `data/index/`, so the build fails at `COPY data/index` unless those are provided,
for example from a release asset or the HF Hub via a build step.

## Clean-up after building

The agent kept `fakenews:latest` (4.63 GB) for the demo, removed its test container, and pruned only the BuildKit
cache entries its own builds created (`docker buildx prune -f --filter id=<id>`, repeated because cache entries are nested),
so the build cache went back from 68.5 GB to 63.7 GB. That cache is gone now, so a rebuild downloads the dependencies again (≈ 6 min). Do not run a bare
`docker builder prune` on this laptop, because it would clear other projects' caches too.
