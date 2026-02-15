# Amharic RAG API — Python 3.11
FROM python:3.11-slim

WORKDIR /app

# No apt packages: PyMuPDF pip wheel includes bundled libs (PyMuPDFb). Saves ~3+ min and image size.

COPY requirements.txt .
# Install CPU-only PyTorch first (avoids ~2GB of CUDA/nvidia-* packages; app runs with DEVICE=cpu)
RUN pip install --no-cache-dir --default-timeout=600 "torch>=2.2,<2.3" --index-url https://download.pytorch.org/whl/cpu
# Then install the rest; pip will reuse the installed torch
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt

COPY alembic.ini .
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# Run migrations then start API (DATABASE_URL set by docker-compose)
CMD ["sh", "-c", "alembic -c alembic.ini upgrade head 2>/dev/null || true && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
