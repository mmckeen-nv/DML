FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dml

COPY pyproject.toml README.md setup.py ./
COPY dml_core ./dml_core
COPY dml_mcp ./dml_mcp

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[server,tokenizer,embeddings,faiss,mcp]"

EXPOSE 8001

CMD ["dml-mcp-server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8001"]
