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
- **Chunks:** Fixed target length + overlap (configurable); each chunk gets a vector and is written to the document’s FAISS index.
- **RAG:** Query is embedded with the same model used at index time → FAISS returns top‑k chunk IDs → chunks are fetched from DB and passed to the generator with a structured Amharic prompt; response includes answer + list of citations (page, chunk id, snippet).

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

Run the API and PostgreSQL with Docker Compose:

```bash
# Optional: set Postgres password (default: postgres)
export POSTGRES_PASSWORD=your_secret

docker-compose up --build
```

- **API:** http://127.0.0.1:8000  
- **Docs:** http://127.0.0.1:8000/docs  

The app service runs migrations on startup and connects to the `db` service. Volumes persist `postgres_data`, `indexes`, `data/uploads`, and the Hugging Face cache (`hf_cache`). First upload or `/rag/ask` will download models from Hugging Face.

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
| `CHUNK_TARGET_CHARS` | No | `1000` | Target characters per chunk |
| `CHUNK_OVERLAP_RATIO` | No | `0.15` | Overlap ratio between chunks |
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
- **Generator:** The default `rasyosef/Llama-3.2-400M-Amharic` requires `tokenizers>=0.20` and `transformers>=4.46` (and `protobuf`) for the LLaMA tokenizer; do not downgrade to older tokenizers/transformers to avoid runtime errors.

---

## Evaluation

Retrieval is evaluated with **Recall@5**, **Recall@10**, **MRR@5**, **MRR@10**, **nDCG@5**, **nDCG@10**, **Hit Rate@5/10**, and **mean latency (ms)**. The script compares the **base** embedding model with the **fine-tuned** model (when the finetuned FAISS index exists).

### Eval questions file

Create a JSON (or JSONL) file with one object per question, e.g.:

- **`document_id`** (required): UUID of the document (from upload or `GET /admin/documents`).
- **`question`** (required): Amharic question string.
- **`expected_chunk_ids`** (optional): List of chunk UUIDs that are relevant (ground truth).
- **`expected_pages`** (optional): If you don’t have chunk IDs, list page numbers; the script resolves chunk IDs from the DB for that document and pages.

Example: `data/eval_questions.example.json`. Copy to `data/eval_questions.json` and replace placeholders with real `document_id` and, if used, `expected_chunk_ids`.

### Run evaluation

From the project root (with venv activated and `.env` set):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --eval-file data/eval_questions.json
```

- **`--output-dir`** (default: `reports`): Directory for results JSON and markdown report.
- **`--top-k`** (default: 10): Retrieval depth (metrics are computed at 5 and 10).
- **`--skip-finetuned`**: Only run the base model (e.g. when the finetuned index is not built).

Outputs:

- **`reports/eval_results_<timestamp>.json`**: Full metrics per model (base / finetuned), latency, and run info.
- **`reports/eval_report_<timestamp>.md`**: Markdown table of metrics and a short base-vs-finetuned comparison.

If the finetuned FAISS index does not exist for a document, the finetuned model is skipped for that run; use `--skip-finetuned` to avoid attempting it.

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
│   │   └── admin.py        # GET /admin/documents, POST /admin/reindex
│   └── services/
│       ├── pdf_extract.py   # PyMuPDF text extraction
│       ├── chunking.py     # Text chunking with overlap
│       ├── embeddings.py   # RoBERTa Amharic embeddings
│       ├── faiss_index.py  # Build/load FAISS index, search
│       ├── retrieval.py    # retrieve_chunks (embed query + FAISS + DB)
│       ├── generator.py    # LLaMA-based answer generation
│       └── evaluation_metrics.py  # Recall@k, MRR@k, nDCG@k, Hit Rate
├── scripts/
│   ├── ensure_db.py        # Create DB if missing
│   ├── download_generator_model.py  # Pre-download generator
│   └── evaluate.py         # Eval script: load questions, run retrieval, metrics, report
├── data/
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
