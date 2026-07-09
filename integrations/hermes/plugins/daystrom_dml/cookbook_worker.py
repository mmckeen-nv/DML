"""Daystrom DML cookbook distillation worker.

Consumes compact ``memory_class=tool_cookbook_event`` records from a DML
``dml_state.jsonl`` store, asks a local Ollama llama model to distill them into
a reusable operational cookbook, then ingests that recipe back into DML as
``memory_class=tool_cookbook_recipe``.

This module is deliberately secret-safe and dependency-light: it uses only the
standard library plus the existing DML launcher for persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_SENSITIVE_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization|bearer)\b\s*[:=]\s*\S+"
)
_TOOL_LOG_NOISE_RE = re.compile(
    r"(?i)\b(?:embedding|wall time|chunk id|process exited|tool_calls|functions\.|multi_tool_use\.|\[truncated\])\b"
)
_DEFAULT_MODEL = "llama3:8b"
_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def redact_sensitive(text: str) -> str:
    return _SENSITIVE_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text or "")


def clean_text(value: Any, limit: int = 1200) -> str:
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value or "")
    text = " ".join(text.split())
    text = redact_sensitive(text)
    return text[:limit].rstrip()


def load_state_items(storage_dir: Path) -> List[Dict[str, Any]]:
    state = storage_dir / "dml_state.jsonl"
    if not state.exists():
        return []
    items: List[Dict[str, Any]] = []
    with state.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Header/checksum rows are not memory items.
            if line_no == 1 and obj.get("type") == "daystrom_dml.memory":
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def item_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("meta")
    return meta if isinstance(meta, dict) else {}


def event_id(item: Dict[str, Any]) -> str:
    raw = str(item.get("id") or "")
    if raw:
        return raw
    meta = item_meta(item)
    seed = "\n".join([
        str(item.get("timestamp") or ""),
        str(meta.get("tool_name") or ""),
        str(meta.get("tool_outcome") or ""),
        str(item.get("text") or meta.get("summary") or ""),
    ])
    return hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]


def covered_event_ids(items: Iterable[Dict[str, Any]]) -> set[str]:
    covered: set[str] = set()
    for item in items:
        meta = item_meta(item)
        if meta.get("memory_class") != "tool_cookbook_recipe":
            continue
        ids = meta.get("source_event_ids")
        if isinstance(ids, list):
            covered.update(str(x) for x in ids if x)
    return covered


def pending_tool_events(items: Iterable[Dict[str, Any]], *, limit: int = 12) -> List[Dict[str, Any]]:
    all_items = list(items)
    covered = covered_event_ids(all_items)
    events: List[Dict[str, Any]] = []
    for item in all_items:
        meta = item_meta(item)
        if meta.get("memory_class") != "tool_cookbook_event":
            continue
        if meta.get("memory_state") == "quarantined":
            continue
        eid = event_id(item)
        if eid in covered:
            continue
        text = clean_text(item.get("text") or meta.get("summary") or "", 900)
        if not text or _TOOL_LOG_NOISE_RE.search(text):
            # Tool-output boilerplate should not become cookbook material.
            continue
        events.append({
            "event_id": eid,
            "tool_name": clean_text(meta.get("tool_name"), 80),
            "outcome": clean_text(meta.get("tool_outcome"), 40),
            "duration": meta.get("tool_duration_seconds"),
            "text": text,
        })
    return events[-limit:]


def build_cookbook_prompt(events: List[Dict[str, Any]]) -> str:
    event_lines = []
    for idx, event in enumerate(events, 1):
        duration = event.get("duration")
        dur = f", duration={duration}s" if duration is not None else ""
        event_lines.append(
            f"{idx}. id={event['event_id']} tool={event.get('tool_name') or 'unknown'} "
            f"outcome={event.get('outcome') or 'unknown'}{dur}\n   {event['text']}"
        )
    return textwrap.dedent(f"""
    You are the local Daystrom DML cookbook worker. Distill these agent tool-call
    outcome events into a reusable cookbook for future related runs.

    Rules:
    - Do not include secrets, tokens, API keys, raw logs, embeddings, or transcript wrappers.
    - Preserve exact useful commands, file paths, tool names, outcomes, and failure causes.
    - Prefer concise operational guidance over narrative.
    - If a command worked, say what worked and when to reuse it.
    - If a command/tool failed or was blocked, say why and what to try next.
    - Output Markdown only with these sections:
      # Tool Cookbook
      ## Situation
      ## Worked
      ## Failed / Blocked
      ## Reuse Next Time
      ## Evidence Event IDs

    Events:
    {chr(10).join(event_lines)}
    """).strip()


def call_ollama(prompt: str, *, model: str = _DEFAULT_MODEL, base_url: str = _DEFAULT_OLLAMA_URL, timeout: float = 120.0) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama cookbook model call failed: {exc}") from exc
    text = clean_text(data.get("response") or "", 6000)
    if not text:
        raise RuntimeError("Ollama cookbook model returned an empty response")
    return text


def ingest_cookbook(
    cookbook: str,
    *,
    launcher: Path,
    storage_dir: Path,
    config_path: Optional[Path],
    session_id: str,
    tenant_id: str,
    client_id: str,
    source_event_ids: List[str],
    model: str,
    timeout: float = 60.0,
) -> None:
    meta = {
        "source": "hermes-tool-cookbook-worker",
        "phase": "cookbook-distillation",
        "memory_class": "tool_cookbook_recipe",
        "source_event_ids": source_event_ids,
        "summary_source": "ollama_cookbook_worker_v1",
        "cookbook_model": model,
        "created_at": int(time.time()),
    }
    cmd = [
        str(launcher),
        "--no-require-gpu",
        "--storage-dir", str(storage_dir),
    ]
    if config_path:
        cmd.extend(["--config-path", str(config_path)])
    cmd.extend([
        "ingest",
        "--kind", "action",
        "--session-id", session_id,
        "--tenant-id", tenant_id,
        "--client-id", client_id,
        "--summary-policy", "cheap",
        "--no-filter-noise",
        "--meta", json.dumps(meta, separators=(",", ":")),
        "--text", cookbook,
    ])
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout[:500] or f"ingest failed rc={proc.returncode}")


def run_worker(
    *,
    storage_dir: Path,
    launcher: Optional[Path] = None,
    config_path: Optional[Path] = None,
    session_id: str = "snips2-hermes-default",
    tenant_id: str = "openclaw",
    client_id: str = "snips2",
    model: str = _DEFAULT_MODEL,
    ollama_url: str = _DEFAULT_OLLAMA_URL,
    min_events: int = 1,
    limit: int = 12,
    dry_run: bool = False,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    items = load_state_items(storage_dir)
    events = pending_tool_events(items, limit=limit)
    if len(events) < min_events:
        return {"ok": True, "distilled": False, "reason": "not_enough_events", "pending_events": len(events)}
    prompt = build_cookbook_prompt(events)
    cookbook = call_ollama(prompt, model=model, base_url=ollama_url, timeout=timeout)
    source_ids = [event["event_id"] for event in events]
    if not dry_run:
        if launcher is None:
            raise ValueError("launcher is required unless dry_run=True")
        ingest_cookbook(
            cookbook,
            launcher=launcher,
            storage_dir=storage_dir,
            config_path=config_path,
            session_id=session_id,
            tenant_id=tenant_id,
            client_id=client_id,
            source_event_ids=source_ids,
            model=model,
            timeout=timeout,
        )
    return {
        "ok": True,
        "distilled": True,
        "dry_run": dry_run,
        "model": model,
        "event_count": len(events),
        "source_event_ids": source_ids,
        "cookbook_preview": cookbook[:1000],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Distill DML tool outcome events into cookbook recipes using Ollama.")
    parser.add_argument("--storage-dir", required=True, type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--session-id", default="snips2-hermes-default")
    parser.add_argument("--tenant-id", default="openclaw")
    parser.add_argument("--client-id", default="snips2")
    parser.add_argument("--model", default=os.environ.get("DAYSTROM_COOKBOOK_MODEL", _DEFAULT_MODEL))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", _DEFAULT_OLLAMA_URL))
    parser.add_argument("--min-events", type=int, default=1)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--watch", action="store_true", help="Run continuously instead of one-shot.")
    parser.add_argument("--interval", type=float, default=300.0, help="Seconds between watch ticks.")
    args = parser.parse_args(argv)
    watch = args.watch
    interval = args.interval
    delattr(args, "watch")
    delattr(args, "interval")
    if watch:
        while True:
            try:
                result = run_worker(**vars(args))
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
            time.sleep(max(1.0, interval))
    try:
        result = run_worker(**vars(args))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
