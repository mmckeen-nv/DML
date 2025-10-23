FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DML_HOST=0.0.0.0 \
    DML_PORT=9000

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY daystrom_dml /app/daystrom_dml
COPY tests /app/tests

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[server]

EXPOSE 9000
CMD ["dml-server"]
