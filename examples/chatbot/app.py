import os
import time

import gradio as gr
import requests

DML_BASE_URL = os.environ.get("DML_BASE_URL", "http://localhost:8000")
DML_SESSION_ID = os.environ.get("DML_SESSION_ID", "default")


def dml_query(message, history):
    if not message:
        return ""
    payload = {"prompt": message, "session_id": DML_SESSION_ID}
    start_time = time.perf_counter()
    response = requests.post(f"{DML_BASE_URL}/query", json=payload, timeout=60)
    latency_ms = (time.perf_counter() - start_time) * 1000
    response.raise_for_status()
    data = response.json()
    answer = data.get("response", "")
    return f"{answer}\n\n_({latency_ms:.0f} ms)_"


def fetch_stats():
    response = requests.get(f"{DML_BASE_URL}/stats", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_knowledge():
    response = requests.get(f"{DML_BASE_URL}/knowledge", timeout=30)
    response.raise_for_status()
    return response.json()


with gr.Blocks(title="DML Chatbot") as demo:
    gr.Markdown("# DML Chatbot + Telemetry")
    with gr.Tabs():
        with gr.TabItem("Chat"):
            gr.ChatInterface(fn=dml_query)
        with gr.TabItem("Telemetry"):
            gr.Markdown("## Deep Telemetry")
            with gr.Row():
                stats_button = gr.Button("Fetch /stats")
                knowledge_button = gr.Button("Fetch /knowledge")
            stats_output = gr.JSON(label="/stats")
            knowledge_output = gr.JSON(label="/knowledge")
            stats_button.click(fetch_stats, outputs=stats_output)
            knowledge_button.click(fetch_knowledge, outputs=knowledge_output)
            gr.Markdown(
                """
                ### Quick Links
                - [DML UI](/)
                - [/metrics](/metrics)
                - [/health](/health)
                """
            )


demo.launch(server_name="0.0.0.0", server_port=7860)
