# Amharic RAG API — Python 3.11
FROM python:3.11-slim

WORKDIR /app

# No apt packages: PyMuPDF pip wheel includes bundled libs (PyMuPDFb). Saves ~3+ min and image size.

COPY requirements.txt .
# CPU-only PyTorch: ADD fetches the wheel (no apt/wget); pip install from file avoids hash mismatch on slow networks
ADD https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-linux_x86_64.whl /tmp/torch.whl
RUN pip install --no-cache-dir /tmp/torch.whl && rm /tmp/torch.whl
# Then install the rest; pip will reuse the installed torch
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt

COPY alembic.ini .
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# Run migrations then start API (DATABASE_URL set by docker-compose)
CMD ["sh", "-c", "alembic -c alembic.ini upgrade head 2>/dev/null || true && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
