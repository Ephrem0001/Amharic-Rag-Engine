# Amharic RAG Engine

A production-oriented **Retrieval-Augmented Generation (RAG)** API for Amharic-language documents. Upload PDFs, ask questions in Amharic, and receive answers grounded in your documents with source citations.

---

## Overview

This service provides:

- **Document ingestion:** PDF upload with configurable chunking and embedding (RoBERTa-based Amharic text embeddings).
- **Vector search:** FAISS indices per document and embedding model for fast similarity retrieval.
- **Answer generation:** Context-conditioned answers in Amharic via a LLaMA-based generator (e.g. `rasyosef/Llama-3.2-400M-Amharic`), with citations tied to retrieved chunks.

Typical flow: **Upload PDF → Chunk → Embed → Index (FAISS)**; then **Ask question → Embed query → Retrieve top‑k chunks → Generate answer with citations.**

---

## Architecture

| Layer | Technology |
|-------|------------|
| **API** | FastAPI (Python 3.10+) |
| **Database** | PostgreSQL (documents, pages, chunks, metadata) |
| **Embeddings** | RoBERTa Amharic text embedding models (Hugging Face); sentence-transformers |
| **Vector store** | FAISS (per-document, per–embedding-model indices on disk) |
| **Generator** | LLaMA-based causal LM (e.g. Llama-3.2-400M-Amharic) via Hugging Face Transformers |

- **Documents:** Stored on disk under `UPLOAD_DIR`; metadata and chunk text in PostgreSQL.
- **Chunks:** Configurable target length (default 1000 chars, range 700–1200 recommended) with 10–15% overlap; each chunk gets a vector and is written to the document’s FAISS index.
- **FAISS indices:** Stored as `{document_id}__{model_type}.faiss` and `{document_id}__{model_type}.mapping.json` under `INDEX_DIR` (e.g. `indexes/abc123__base.faiss`). Some specs use a single index per document (`{document_id}.faiss`); this project uses the `{document_id}__{model_type}.faiss` naming so the same document can have separate indices for base, medium, and finetuned embeddings, enabling direct comparison without reindexing.
- **RAG:** Query is embedded with the same model used at index time → FAISS returns top‑k chunk IDs → chunks are fetched from DB and passed to the generator with a structured Amharic prompt; response includes answer + list of citations (page, chunk id, snippet).

### Architecture diagram

```
                    ┌─────────────┐
                    │  PDF upload │
                    └──────┬──────┘
                           │
                           ▼
    ┌──────────────┐   extract text    ┌──────────────┐
    │   PyMuPDF    │ ───────────────►  │    pages     │
    └──────────────┘                    └──────┬───────┘
                                              │ chunk (700–1200 chars, 10–15% overlap)
                                              ▼
    ┌──────────────┐   embed (RoBERTa) ┌──────────────┐      ┌─────────────────┐
    │  FAISS index │ ◄────────────────│   chunks    │ ────► │  PostgreSQL     │
    │  (per doc +  │                   └─────────────┘       │  (metadata)     │
    │   model_type)│                                         └─────────────────┘
    └──────┬───────┘
           │
           │  query (Amharic)
           ▼
    ┌──────────────┐   top-k chunk IDs   ┌──────────────┐   context + prompt   ┌──────────────┐
    │  embed query │ ──────────────────► │ retrieve     │ ───────────────────► │  LLaMA 3.2   │
    │  + FAISS     │                     │ chunks from  │                       │  Amharic     │
    │  search      │                     │ DB           │                       │  generator   │
    └──────────────┘                     └──────────────┘                       └──────┬───────┘
                                                                                       │
                                                                                       ▼
                                                                              answer + citations
                                                                              (Amharic)
```

### Model choice justification

- **Embedding (RoBERTa Amharic):** The default base model (`rasyosef/roberta-amharic-text-embedding-base`) is a RoBERTa model pretrained and adapted for Amharic text, producing dense vectors suitable for semantic similarity and retrieval. The same family offers a larger “medium” variant for higher quality at higher cost. Fine-tuning on in-domain (query, positive, negative) triplets improves retrieval for your document domain while keeping the same interface and index layout.
- **Generator (LLaMA 3.2 400M Amharic):** The default `rasyosef/Llama-3.2-400M-Amharic` is a small causal LM fine-tuned for Amharic, which keeps inference fast and resource usage low while still producing fluent, context-conditioned answers. For more complex or longer answers, a larger generator can be swapped via `GENERATOR_MODEL`.

---

## Prerequisites

- **Python** 3.10 or 3.11 (recommended).
- **PostgreSQL** (e.g. 14+) with a dedicated database (e.g. `amharic_rag`).
- **OS:** Windows (PowerShell) or Linux/macOS (bash). Commands below use PowerShell where relevant.
- **Disk/RAM:** Allow several GB for Hugging Face model cache (embedding + generator). First request that loads a model may take minutes and use significant memory.

---

## Setup

### 1. Clone and enter project

```powershell
cd C:\path\to\amharic-rag-engine
```

### 2. Environment file

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/amharic_rag
```

See [Configuration](#configuration) for optional variables.

### 3. Virtual environment and dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Database

Create the database (once; uses `postgres` DB to run `CREATE DATABASE`):

```powershell
.\.venv\Scripts\python.exe scripts\ensure_db.py
```

Run Alembic migrations:

```powershell
.\.venv\Scripts\python.exe -m alembic -c alembic.ini upgrade head
```

### 5. Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- **API base:** http://127.0.0.1:8000  
- **Interactive docs:** http://127.0.0.1:8000/docs  

### 6. (Optional) Pre-download generator model

To avoid a long delay on the first `/rag/ask` call, pre-download the generator (progress in terminal):

```powershell
$env:HF_HUB_DISABLE_PROGRESS_BARS="0"
.\.venv\Scripts\python.exe scripts\download_generator_model.py
```

---

## Docker

**Prerequisites:** Docker and Docker Compose installed. On **Windows**, install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and **start it** (the tray icon must show Docker is running). The error `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` means the Docker daemon is not running—launch Docker Desktop and try again.

Run the API and PostgreSQL with Docker Compose:

```bash
# Optional: set Postgres password (default: postgres)
# PowerShell:
$env:POSTGRES_PASSWORD="your_secret"

# Bash / WSL:
# export POSTGRES_PASSWORD=your_secret

docker-compose up --build
```

- **API:** http://127.0.0.1:8000  
- **Docs:** http://127.0.0.1:8000/docs  

The app service runs migrations on startup and connects to the `db` service. Volumes persist `postgres_data`, `indexes`, `data/uploads`, and the Hugging Face cache (`hf_cache`). First upload or `/rag/ask` will download models from Hugging Face.

### Docker troubleshooting (e.g. "no such host" for auth.docker.io)

If the build fails with `failed to fetch oauth token` or `lookup auth.docker.io: no such host`, Docker cannot resolve Docker Hub (network/DNS). Try in order:

1. **Test DNS (PowerShell):** `ping auth.docker.io` and `nslookup auth.docker.io`. If they fail, DNS is the cause.
2. **Restart Docker:** Fully quit Docker Desktop. In `Win+R` → `services.msc`, find **Docker Desktop Service** → Right‑click → Restart. Open Docker Desktop again.
3. **Restart WSL (if you use WSL):** `wsl --shutdown`, then open Docker Desktop.
4. **Use Google DNS:** Network Connections → your adapter → Properties → IPv4 → Use: Preferred `8.8.8.8`, Alternate `8.8.4.4`.
5. **Test pull:** `docker pull python:3.11-slim`. If that works, run `docker-compose build --no-cache`.
6. **Flush DNS:** `ipconfig /flushdns`, then retry the pull/build.

Common causes: ISP DNS blocking, VPN, or Docker Desktop’s resolver; switching to 8.8.8.8 and restarting Docker often fixes it.

---

## Configuration

Settings are read from `.env` and `app/core/config.py` (Pydantic `BaseSettings`). Key variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | PostgreSQL URL, e.g. `postgresql+psycopg://user:pass@host:port/amharic_rag` |
| `INDEX_DIR` | No | `./indexes` | Directory for FAISS index files |
| `UPLOAD_DIR` | No | `./data/uploads` | Directory for uploaded PDFs |
| `EMBEDDING_MODEL_BASE` | No | `rasyosef/roberta-amharic-text-embedding-base` | Hugging Face model for `embedding_model_type=base` |
| `EMBEDDING_MODEL_MEDIUM` | No | `rasyosef/roberta-amharic-text-embedding-medium` | Model for `medium` |
| `EMBEDDING_MODEL_FINETUNED` | No | `models/embeddings_finetuned` | Local path for `finetuned` |
| `GENERATOR_MODEL` | No | `rasyosef/Llama-3.2-400M-Amharic` | Causal LM for `/rag/ask` |
| `DEVICE` | No | `cpu` | `cpu` or `cuda` for embedding/generator |
| `TORCH_DTYPE` | No | `float32` | `float32`, `float16`, or `bfloat16` for generator |
| `CHUNK_TARGET_CHARS` | No | `1000` | Target characters per chunk (recommended: 700–1200; configurable via `.env`) |
| `CHUNK_OVERLAP_RATIO` | No | `0.15` | Overlap ratio between chunks (recommended: 0.10–0.15, i.e. 10–15%) |
| `TOP_K_DEFAULT` | No | `5` | Default number of chunks retrieved for RAG |
| `TEXTGEN_MAX_NEW_TOKENS` | No | `256` | Max new tokens for generator |
| `TEXTGEN_TEMPERATURE` | No | `0.2` | Sampling temperature for generator |

---

## API Overview

### Health

- **`GET /health`** — Returns `{"status": "ok"}`. Use for liveness checks.

### Documents (`/documents`)

- **`POST /documents/upload`**  
  - **Body:** multipart form with `file` (PDF).  
  - **Query:** `embedding_model_type` = `base` \| `medium` \| `finetuned` (default: `base`).  
  - **Behavior:** Saves PDF under `UPLOAD_DIR`, extracts text (PyMuPDF), chunks, embeds with the chosen model, builds FAISS index for that document, stores metadata and chunks in DB.  
  - **Response:** `document_id`, `title`, `pages`, `chunks`, `embedding_model_type`, `embedding_model_name`.  
  - **Errors:** 400 if not PDF or no extractable text; 503 if Hugging Face unreachable (e.g. network); 500 on other failures (see `detail` in response).

### RAG (`/rag`)

- **`POST /rag/retrieve`**  
  - **Body:** `document_id`, `question`, optional `top_k` (0–50), optional `embedding_model_type`.  
  - **Response:** `document_id`, `embedding_model_type`, `top_k`, `results` (list of `chunk_id`, `page_number`, `score`, `chunk_text`).  
  - **Errors:** 404 if no FAISS index for that document/model (upload or reindex first).

- **`POST /rag/ask`**  
  - **Body:** `question` (required), optional `document_id` (if omitted, uses most recently uploaded document), optional `top_k`, optional `embedding_model_type`.  
  - **Behavior:** Retrieves top‑k chunks for the question, builds an Amharic prompt with context, runs the generator, returns answer and citations.  
  - **Response:** `answer`, `citations` (list of `page_number`, `chunk_id`, `snippet`), `document_id`.  
  - **Errors:** 400 if no documents in DB and `document_id` omitted; 404 if index missing; 500 if generation fails (see `detail`).

### Admin (`/admin`)

- **`GET /admin/documents`**  
  - **Response:** List of `document_id`, `title`, `created_at` (newest first).

- **`POST /admin/reindex/{document_id}`**  
  - **Query:** `embedding_model_type` = `base` \| `medium` \| `finetuned`.  
  - **Behavior:** Re-embeds all chunks for that document and rebuilds the FAISS index (e.g. after changing embedding model).  
  - **Response:** `document_id`, `embedding_model_type`, `embedding_model_name`, `chunks`, `status`.  
  - **Errors:** 404 if no chunks for that `document_id`.

- **`POST /admin/evaluate`**  
  - **Body (optional):** `eval_file` (path to JSON/JSONL, default `data/eval_questions.json`), `top_k` (default 10), `skip_finetuned` (default false), `store_results` (default true).  
  - **Behavior:** Loads eval questions, runs retrieval for base (and optionally finetuned) model, computes Recall@5/10, MRR, nDCG, Hit Rate, mean latency. If `store_results` is true, inserts one row per model into the `eval_runs` table (run_name, embedding_model_type, top_k, recall_at_k, mrr_at_k, ndcg_at_k, hit_rate_at_k, mean_latency_ms).  
  - **Response:** `run_name` (if stored), `eval_file`, `top_k`, `num_questions`, `models`: `{ base: {...}, finetuned: {...} }` with full metrics.  
  - **Errors:** 404 if eval file not found; 400 if no valid questions or load error.

---

## Usage

1. **Upload a PDF**  
   In Swagger (`/docs`): **POST /documents/upload** → choose file, set `embedding_model_type` if needed (default `base`), Execute. Note the returned `document_id` if you have multiple documents.

2. **Ask a question**  
   **POST /rag/ask** with JSON body, e.g.:  
   `{"question": "ይህ ሰነድ ስለ ምን ነው?"}`  
   Omit `document_id` to use the latest document; otherwise set `document_id` to the value from upload.

3. **List documents**  
   **GET /admin/documents** to see all documents and their IDs.

4. **Reindex a document**  
   **POST /admin/reindex/{document_id}?embedding_model_type=base** to rebuild the FAISS index (e.g. after changing embedding model or fixing index path).

---

## Operational Notes

- **First-time model load:** Embedding and generator models are downloaded from Hugging Face on first use. This can take several minutes and requires network. Pre-download the generator with `scripts/download_generator_model.py` to speed up the first `/rag/ask`.
- **CPU vs GPU:** Set `DEVICE=cuda` and ensure PyTorch sees CUDA if you want GPU. Default is `cpu`; generator and embedding both run on the same device.
- **Disk:** Hugging Face cache (e.g. `~/.cache/huggingface`), `INDEX_DIR`, and `UPLOAD_DIR` need sufficient space.
- **PDFs:** Only text-based PDFs are supported (no built-in OCR for scanned pages). If extraction yields no text, the API returns 400 with a message suggesting OCR.
- **Chunking:** Chunk target is configurable as 700–1200 characters via `CHUNK_TARGET_CHARS` in `.env` (default `1000`). Adjust as needed (e.g. `CHUNK_TARGET_CHARS=800` for shorter chunks, `CHUNK_TARGET_CHARS=1200` for longer). Overlap is `CHUNK_OVERLAP_RATIO=0.15` (15%).
- **FAISS index naming:** Indices are stored as `{document_id}__{model_type}.faiss` (e.g. `abc123__base.faiss`, `abc123__finetuned.faiss`), not the simpler `{document_id}.faiss`. This choice allows multiple embedding models per document (base vs finetuned) without conflicts; retrieval uses the index that matches the chosen `embedding_model_type`.
- **Generator:** The default `rasyosef/Llama-3.2-400M-Amharic` requires `tokenizers>=0.20` and `transformers>=4.46` (and `protobuf`) for the LLaMA tokenizer; do not downgrade to older tokenizers/transformers to avoid runtime errors.

---

## Known limitations

- **Text-only PDFs:** Only PDFs with extractable Unicode text are supported. Scanned documents or image-based PDFs require external OCR; the API returns 400 with a message suggesting OCR if no text is extracted.
- **No hybrid retrieval:** Retrieval is dense-only (FAISS). BM25 or hybrid (sparse + dense) is not implemented.
- **No reranking:** Top-k chunks are returned by vector similarity only; no cross-encoder or other reranker.
- **Amharic-focused generator:** The default generator is tuned for Amharic; mixed Amharic/English documents are supported for retrieval, but the answer is optimized for Amharic.
- **CPU default:** Default runs on CPU; GPU (CUDA) improves embedding and generation speed but is optional.
- **Single document per ask:** `/rag/ask` without `document_id` uses the most recently uploaded document; cross-document search is not supported.

---

## Evaluation

Retrieval is evaluated with **Recall@5**, **Recall@10**, **MRR@5**, **MRR@10**, **nDCG@5**, **nDCG@10**, **Hit Rate@5/10**, and **mean latency (ms)**. The script and the **POST /admin/evaluate** endpoint both run the same logic; results can be written to the **eval_runs** table (one row per model: run_name, embedding_model_type, top_k, recall_at_k, mrr_at_k, ndcg_at_k, hit_rate_at_k, mean_latency_ms) for history and comparison.

### Eval questions file

Create a JSON (or JSONL) file with one object per question, e.g.:

- **`document_id`** (required): UUID of the document (from upload or `GET /admin/documents`).
- **`question`** (required): Amharic question string.
- **`expected_chunk_ids`** (optional): List of chunk UUIDs that are relevant (ground truth).
- **`expected_pages`** (optional): If you don’t have chunk IDs, list page numbers; the script resolves chunk IDs from the DB for that document and pages.

Example: `data/eval_questions.example.json`. A **mock eval set** with **30 questions** for civil service–style documents is in `data/eval_questions_civil_service.json`; replace every `REPLACE_AFTER_UPLOAD` with your `document_id` (from upload or `GET /admin/documents`). Provided inputs may require 50+ questions; this mock set meets the minimum (30+) for evaluation.

### Run evaluation

**Via script** (from project root, venv activated, `.env` set):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --eval-file data\eval_questions.json
```

- **`--output-dir`** (default: `reports`): Directory for results JSON and markdown report.
- **`--top-k`** (default: 10): Retrieval depth (metrics are computed at 5 and 10).
- **`--skip-finetuned`**: Only run the base model (e.g. when the finetuned index is not built).
- **`--no-store`**: Do not write rows to the `eval_runs` table (default is to store).

**Via API:** **POST /admin/evaluate** with optional body `{ "eval_file": "data/eval_questions.json", "top_k": 10, "skip_finetuned": false, "store_results": true }` to get metrics and optionally persist to `eval_runs`.

Outputs:

- **`reports/eval_results_<timestamp>.json`**: Full metrics per model (base / finetuned), latency, and run info.
- **`reports/eval_report_<timestamp>.md`**: Markdown table of metrics, base-vs-finetuned comparison, and failure analysis (at least 5 cases).
- **`reports/eval.md`**: Same content as the latest report (deliverable filename); overwritten on each run.

If the finetuned FAISS index does not exist for a document, the finetuned model is skipped for that run; use `--skip-finetuned` to avoid attempting it.

---

## Fine-tuning pipeline

You can fine-tune the base Amharic embedding model on (query, positive chunk, negative chunk) triplets, then reindex documents so retrieval uses the finetuned model. The spec recommends **at least 800 triplets** (or 300 if documents are limited); the dataset builder default is 800.

### Dataset creation method & negative sampling

Training data are **retrieval pairs**: each row has a **query**, a **positive chunk** (ground-truth relevant passage), and **hard negative** chunk IDs. These are produced by **`scripts/build_dataset.py`**, which:

1. Iterates over chunks in the DB (optionally for one `--document-id`).
2. Builds a synthetic **query** from the chunk text (e.g. a short Amharic prompt plus a snippet) so the positive is the chunk itself.
3. Selects **hard negatives** by running the **base** embedding model: it embeds the query, runs FAISS search for that document, and takes top retrievals that are *not* the positive chunk. That yields in-domain, semantically plausible negatives (model thought they were relevant but they are not).
4. If the FAISS index is missing or too few hard negatives are found, it **falls back to random** negatives from the same document.

So negative sampling is **hard negatives from base-model retrieval first**, then **random same-document chunks** to fill. The resulting JSONL (and DB table `retrieval_pairs`) is used by **`scripts/train_embeddings.py`** to train with TripletLoss (query, positive, negative).

### 1. Build the dataset

**Option A – From DB (recommended):** Generate query–chunk pairs with hard negatives from the base-model FAISS index. Writes to both the `retrieval_pairs` table and a JSONL file.

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py --max-pairs 800 --out data\retrieval_pairs.jsonl
```

- **`--document-id`**: Limit to one document (UUID).
- **`--hard-negatives`**: Number of hard negatives per pair (default 3).
- **`--max-pairs`**: Max pairs to generate (default **800**; spec recommends 300–800).

**Option B – From existing JSONL:** If you already have a file with `document_id`, `query`, `positive_chunk_id`, and `hard_negative_chunk_ids` per line:

```powershell
.\.venv\Scripts\python.exe scripts\load_retrieval_pairs.py --input data\retrieval_pairs.jsonl --skip-invalid
```

Chunk text is always resolved from the DB, so chunks must exist for the given IDs.

### 2. Train the embedding model

Fine-tune the base model (TripletLoss) and save to the path used for `embedding_model_type=finetuned` (default `models/embeddings_finetuned`). Training uses **fixed random seeds** (Python, NumPy, PyTorch) for reproducibility.

```powershell
.\.venv\Scripts\python.exe scripts\train_embeddings.py --epochs 1 --batch-size 16 --lr 2e-5
```

- **`--output`**: Override output directory (default: `EMBEDDING_MODEL_FINETUNED` from `.env`).
- **`--data data\retrieval_pairs.jsonl`**: Use JSONL instead of DB; chunk text still loaded from DB.
- **`--limit`**: Limit number of triplets (0 = all). Aim for at least 300–800 triplets for the spec.

### 3. Reindex with the finetuned model

Build FAISS indices for the finetuned model so `/rag/retrieve` and `/rag/ask` can use `embedding_model_type=finetuned`:

**All documents:**

```powershell
.\.venv\Scripts\python.exe scripts\reindex_finetuned.py
```

**One document:**

```powershell
.\.venv\Scripts\python.exe scripts\reindex_finetuned.py --document-id <UUID>
```

**Or via API:** `POST /admin/reindex/{document_id}?embedding_model_type=finetuned` for each document.

After reindexing, run evaluation without `--skip-finetuned` to compare base vs finetuned retrieval.

---

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| **500 on upload** | Missing or incompatible `sentence-transformers` / tokenizers / transformers; slow-tokenizer fallback in `app/services/embeddings.py`; server logs for traceback. |
| **503 on upload** | Network/DNS to Hugging Face; firewall/proxy; try again when online or after pre-downloading the embedding model. |
| **500 on /rag/ask** | Generator load/generation failure: check `detail` in the 500 response; ensure `protobuf` installed and generator model ID in `.env` matches a full model (e.g. `rasyosef/Llama-3.2-400M-Amharic`), not an adapter-only repo. |
| **404 on /rag/retrieve or /rag/ask** | FAISS index missing for that `document_id` and `embedding_model_type`: upload the document or run **POST /admin/reindex/{document_id}**. |
| **400 “No documents in the database”** | Upload at least one PDF via **POST /documents/upload** before calling **POST /rag/ask** without `document_id`. |
| **DB connection errors** | PostgreSQL running; `DATABASE_URL` in `.env` correct; database created (`scripts/ensure_db.py`); migrations applied (`alembic upgrade head`). |

Server logs (e.g. loguru to stderr) and the JSON `detail` field on 4xx/5xx responses are the primary sources for debugging.

---

## Project structure (reference)

```
amharic-rag-engine/
├── app/
│   ├── main.py              # FastAPI app, routers, /health
│   ├── core/
│   │   └── config.py        # Pydantic settings from .env
│   ├── db/
│   │   ├── session.py       # SQLAlchemy session
│   │   ├── models.py        # Document, Page, Chunk
│   │   └── migrations/      # Alembic
│   ├── routers/
│   │   ├── documents.py     # POST /documents/upload
│   │   ├── rag.py          # POST /rag/retrieve, POST /rag/ask
│   │   └── admin.py        # GET /admin/documents, POST /admin/reindex, POST /admin/evaluate
│   └── services/
│       ├── pdf_extract.py   # PyMuPDF text extraction
│       ├── chunking.py     # Text chunking with overlap
│       ├── embeddings.py   # RoBERTa Amharic embeddings
│       ├── faiss_index.py  # Build/load FAISS index, search
│       ├── retrieval.py    # retrieve_chunks (embed query + FAISS + DB)
│       ├── generator.py    # LLaMA-based answer generation
│       ├── evaluation_metrics.py  # Recall@k, MRR@k, nDCG@k, Hit Rate
│       └── evaluation_runner.py    # run_evaluation, persist_eval_runs (script + API)
├── scripts/
│   ├── ensure_db.py           # Create DB if missing
│   ├── download_generator_model.py  # Pre-download generator
│   ├── build_dataset.py       # Build retrieval pairs (query + positive + hard negatives) from DB
│   ├── load_retrieval_pairs.py  # Load JSONL retrieval pairs into DB for training
│   ├── train_embeddings.py    # Fine-tune embedding model on triplets; save to EMBEDDING_MODEL_FINETUNED
│   ├── reindex_finetuned.py   # Reindex all (or one) document with finetuned model
│   ├── evaluate.py            # Eval script: load questions, run retrieval, metrics, report
│   ├── run_failure_analysis.py  # Compare expected vs retrieved, write failure report
│   └── build_eval_from_db.py  # Build eval_questions.json from latest document
├── data/
│   ├── retrieval_pairs.jsonl # Optional: query / positive_chunk_id / hard_negative_chunk_ids per line
│   └── eval_questions.example.json  # Example eval file
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── requirements.txt
└── README.md
```

---

## License and references

- Embedding models: e.g. [rasyosef/roberta-amharic-text-embedding-*](https://huggingface.co/rasyosef) (Hugging Face).
- Generator: e.g. [rasyosef/Llama-3.2-400M-Amharic](https://huggingface.co/rasyosef/Llama-3.2-400M-Amharic).

For license and contribution terms, see the repository or maintainers.
