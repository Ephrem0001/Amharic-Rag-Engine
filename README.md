# Amharic RAG (No Docker) – FastAPI + Postgres + FAISS + RoBERTa Embeddings + Walia-LLM

This project implements an **Amharic RAG** pipeline:

- Upload an Amharic PDF (Unicode text) → extract page text → chunk
- Embed chunks using **Amharic RoBERTa sentence-transformers**
- Build a **FAISS** index per document
- Retrieve top-K chunks for a question
- Generate an Amharic answer using **Walia-LLM** with **citations (page + snippet)**

## 1) Requirements (Windows)

- Windows 10/11
- Python 3.10+ recommended
- PostgreSQL installed locally (default port 5432)
- VS Code (optional)

## 2) Setup (PowerShell)

### 2.1 Create venv + install deps

```powershell
cd C:\path\to\amharic_rag_nodocker

python -m venv venv
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\Activate.ps1
```

### 2.2 Create `.env`

```powershell
Copy-Item .\.env.example .\.env
```

Edit `.env` and set your Postgres password:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/amharic_rag
```

### 2.3 Create the database in Postgres

Using `psql` (or pgAdmin), run:

```sql
CREATE DATABASE amharic_rag;
```

### 2.4 Run migrations

```powershell
alembic upgrade head
```

## 3) Run the API

```powershell
uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000/docs (Swagger UI)

Health:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 4) Upload a PDF

Put a PDF at `./data/docs/your_doc.pdf` then upload:

```powershell
$docPath = ".\data\docs\your_doc.pdf"
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" `
  -F "file=@$docPath"
```

Response includes `document_id`. Copy it.

## 5) Retrieve

```powershell
$DOC_ID = "PASTE_DOCUMENT_ID_HERE"

$body = @{
  document_id = $DOC_ID
  question    = "ይህ ሰነድ ስለ ምን ይናገራል?"
  top_k       = 5
  embedding_model_type = "base"   # base | medium | finetuned
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/rag/retrieve" `
  -ContentType "application/json" `
  -Body $body
```

## 6) Ask (RAG answer + citations)

```powershell
$body = @{
  document_id = $DOC_ID
  question    = "ዋና መስፈርቶቹ ምንድን ናቸው?"
  top_k       = 5
  embedding_model_type = "base"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/rag/ask" `
  -ContentType "application/json" `
  -Body $body
```

## 7) Build training pairs (optional)

Inside the same venv:

```powershell
python scripts\build_dataset.py --out .\data\retrieval_pairs.jsonl
```

## 8) Fine-tune embeddings (optional)

```powershell
python scripts\train_embeddings.py --epochs 1 --batch-size 16 --lr 2e-5 --output .\models\embeddings_finetuned
```

## 9) Reindex using fine-tuned embeddings

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/admin/reindex/$DOC_ID"
```

Now you can set `embedding_model_type="finetuned"` for retrieval/ask.

## Notes

- The first run will download models from Hugging Face. This can take time and disk space.
- If the PDF is scanned images (no selectable text), extraction will be empty (OCR is not included).

## Model references

- RoBERTa Amharic embedding model: rasyosef/roberta-amharic-text-embedding-base
- RoBERTa Amharic embedding (medium): rasyosef/roberta-amharic-text-embedding-medium
- Walia-LLM (Amharic-LLAMA-all-data): EthioNLP/Amharic-LLAMA-all-data
- FAISS Windows wheels are available via faiss-cpu on PyPI
