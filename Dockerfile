# Amharic RAG API — Python 3.11
FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF (optional; remove if base image has them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini .
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# Run migrations then start API (DATABASE_URL set by docker-compose)
CMD ["sh", "-c", "alembic -c alembic.ini upgrade head 2>/dev/null || true && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
