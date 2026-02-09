from __future__ import annotations

import json
import os
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional

from .prompts import get_active_prompt, get_context_payload
from llm_client import create_chat_completion


@dataclass
class AgentResult:
    name: str
    output: str
    tokens: int = 0


def _extract_completion_tokens(response: dict) -> int:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        return 0
    for key in ("completion_tokens", "output_tokens", "generated_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def build_messages(
    role_prompt: str,
    goal: str,
    scenario: str | None = None,
    extra_context: str = "",
    system_messages: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    user_content = f"Goal: {goal}\nScenario: {scenario or 'general'}"
    if extra_context:
        user_content += f"\nContext:\n{extra_context}"
    if not user_content.strip():
        user_content = "User goal not provided."
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": get_active_prompt("system")},
    ]
    context_payload = get_context_payload()
    if context_payload and context_payload.strip():
        messages.append(
            {
                "role": "system",
                "content": f"GB300 context:\n{context_payload}",
            }
        )
    if system_messages:
        for message in system_messages:
            if message and str(message).strip():
                messages.append({"role": "system", "content": str(message)})
    if role_prompt and role_prompt.strip():
        messages.append({"role": "system", "content": role_prompt})
    messages = [m for m in messages if m.get("content") and str(m.get("content")).strip()]
    messages.append({"role": "user", "content": user_content})
    return messages


def _resolve_continue_limit(role: str) -> Optional[int]:
    role_key = str(role or "").strip().upper()
    raw = os.getenv(f"ROLE_CONTINUE_MAX_{role_key}") if role_key else None
    if raw is None:
        raw = os.getenv("AGENT_CONTINUE_MAX", "0")
    raw_clean = str(raw).strip().lower()
    if raw_clean in {"", "none", "null", "no", "false", "0"}:
        return None
    try:
        value = int(raw_clean)
    except ValueError:
        return None
    return value if value > 0 else None


def call_agent(
    role: str,
    goal: str,
    scenario: str | None = None,
    max_tokens: Optional[int] = None,
    extra_context: str = "",
    system_messages: Optional[List[str]] = None,
) -> AgentResult:
    prompt = get_active_prompt(role)
    messages = build_messages(prompt, goal, scenario, extra_context, system_messages=system_messages)
    max_tokens_local = max_tokens
    attempt = 0
    while True:
        try:
            response = create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens_local,
                role=role,
            )
            break
        except requests.HTTPError as exc:
            text = str(exc).lower()
            token_issue = any(token in text for token in ("max_tokens", "context", "token", "length"))
            if not token_issue:
                raise
            if max_tokens_local is None or max_tokens_local <= 256 or attempt >= 3:
                raise
            max_tokens_local = max(256, max_tokens_local // 2)
            attempt += 1
    message = response.get("choices", [{}])[0].get("message", {}) or {}
    content = message.get("content", "") or ""
    if not content:
        tool_calls = message.get("tool_calls")
        if tool_calls:
            content = json.dumps({"tool_calls": tool_calls}, indent=2)
        elif message.get("function_call"):
            content = json.dumps({"function_call": message.get("function_call")}, indent=2)
        else:
            finish_reason = response.get("choices", [{}])[0].get("finish_reason")
            if finish_reason and finish_reason != "length":
                content = f"No content returned (finish_reason={finish_reason})."
    tokens = _extract_completion_tokens(response)
    finish_reason = response.get("choices", [{}])[0].get("finish_reason")
    if finish_reason == "length":
        parts = [content.strip()] if content else []
        total_tokens = tokens
        continue_limit = _resolve_continue_limit(role)
        continue_count = 0
        while True:
            if finish_reason != "length":
                break
            if continue_limit is not None and continue_count >= continue_limit:
                break
            if parts and parts[-1].strip():
                messages.append({"role": "assistant", "content": parts[-1]})
            messages.append({"role": "user", "content": "Continue."})
            follow = create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens_local,
                role=role,
            )
            follow_msg = follow.get("choices", [{}])[0].get("message", {}) or {}
            follow_content = (follow_msg.get("content", "") or "").strip()
            if not follow_content:
                tool_calls = follow_msg.get("tool_calls")
                if tool_calls:
                    follow_content = json.dumps({"tool_calls": tool_calls}, indent=2).strip()
                elif follow_msg.get("function_call"):
                    follow_content = json.dumps({"function_call": follow_msg.get("function_call")}, indent=2).strip()
            if not follow_content:
                break
            if parts and follow_content == parts[-1]:
                break
            parts.append(follow_content)
            total_tokens += _extract_completion_tokens(follow)
            finish_reason = follow.get("choices", [{}])[0].get("finish_reason")
            continue_count += 1
        content = "\n".join([p for p in parts if p]).strip()
        tokens = total_tokens
    return AgentResult(name=role, output=content.strip(), tokens=tokens)
