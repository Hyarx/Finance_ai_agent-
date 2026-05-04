FROM mirror.gcr.io/library/python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv export --frozen --no-dev --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

ENV CHROMA_CACHE_DIR=/app/.chroma_cache
RUN python -c "import chromadb; chromadb.Client(); print('ChromaDB model ready')"

COPY src/ ./src/
COPY data/ ./data/

ENV PORT=8080
ENV CHROMA_CACHE_DIR=/app/.chroma_cache

CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "4", "--timeout", "300", "src.app:app"]