FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY daystrom_dml /app/daystrom_dml
COPY tests /app/tests

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[server]

EXPOSE 9000
CMD ["uvicorn", "daystrom_dml.server:app", "--host", "0.0.0.0", "--port", "9000"]
