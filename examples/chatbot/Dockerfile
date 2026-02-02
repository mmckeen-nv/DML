FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir gradio==4.* requests==2.*

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "/app/app.py"]
