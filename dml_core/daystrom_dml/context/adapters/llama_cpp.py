"""llama.cpp server adapter for prompt-cache and KV-slot control."""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path, PurePath
from typing import Any, Dict, Optional, cast
from urllib import error, request

from daystrom_dml.context.execution import (
    RuntimeCacheOperation,
    RuntimeCacheOperationResult,
    RuntimeCheckpointDeleteResult,
    RuntimeCompletionTrace,
    RuntimeExecutionCapabilities,
    RuntimeExecutionError,
)


class LlamaCppExecutionAdapter:
    def __init__(
        self,
        endpoint_url: str,
        *,
        runtime_id: str,
        runtime_version: str = "unknown",
        timeout_seconds: float = 120.0,
        max_response_bytes: int = 4 * 1024 * 1024,
        checkpoint_directory: Optional[str | Path] = None,
        opener: Optional[Any] = None,
    ) -> None:
        if not endpoint_url or not runtime_id:
            raise RuntimeExecutionError("endpoint_url and runtime_id must be non-empty")
        self.endpoint_url = endpoint_url.rstrip("/")
        self.runtime_id = runtime_id
        self.runtime_version = runtime_version
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.checkpoint_directory: Optional[Path] = None
        if checkpoint_directory is not None:
            directory = Path(checkpoint_directory).expanduser().resolve()
            if not directory.is_dir():
                raise RuntimeExecutionError("checkpoint_directory must be an existing directory")
            self.checkpoint_directory = directory
        self.opener = opener or request.urlopen

    def capabilities(self) -> RuntimeExecutionCapabilities:
        return RuntimeExecutionCapabilities(
            runtime_id=self.runtime_id,
            adapter_id="llama_cpp_server",
            runtime_version=self.runtime_version,
            supports_prompt_cache=True,
            supports_kv_checkpoint=True,
            supports_kv_restore=True,
            supports_kv_erase=True,
            supports_kv_checkpoint_delete=self.checkpoint_directory is not None,
            supports_slot_affinity=True,
            supports_metrics=True,
        )

    def complete(
        self,
        prompt: str,
        *,
        slot_id: int,
        n_predict: int = 1,
        cache_prompt: bool = True,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> RuntimeCompletionTrace:
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeExecutionError("prompt must be a non-empty string")
        _slot(slot_id)
        payload = self._post(
            "/completion",
            {
                "prompt": prompt,
                "id_slot": slot_id,
                "cache_prompt": cache_prompt,
                "n_predict": n_predict,
                "temperature": temperature,
                "seed": seed,
                "return_tokens": True,
            },
        )
        content = payload.get("content")
        timings_raw = payload.get("timings")
        total_raw = payload.get("tokens_evaluated")
        if not isinstance(content, str) or not isinstance(timings_raw, dict):
            raise RuntimeExecutionError("malformed completion response")
        timings: Dict[str, Any] = timings_raw
        processed_raw = timings.get("prompt_n")
        if not _non_negative_int(total_raw) or not _non_negative_int(processed_raw):
            raise RuntimeExecutionError("invalid prompt token counters")
        total = cast(int, total_raw)
        processed = cast(int, processed_raw)
        if processed > total:
            raise RuntimeExecutionError("invalid prompt token counters")
        token_ids = payload.get("tokens") or []
        if not isinstance(token_ids, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in token_ids):
            raise RuntimeExecutionError("invalid output token ids")
        return RuntimeCompletionTrace(
            runtime_id=self.runtime_id,
            slot_id=slot_id,
            prompt_tokens_total=total,
            prompt_tokens_processed=processed,
            prompt_tokens_reused=total - processed,
            prompt_ms=float(timings.get("prompt_ms") or 0),
            predicted_tokens=int(timings.get("predicted_n") or 0),
            output_text=content,
            output_token_ids=token_ids,
            truncated=bool(payload.get("truncated")),
        )

    def save_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        return self._slot_operation(slot_id, RuntimeCacheOperation.SAVE, filename)

    def restore_slot(self, slot_id: int, filename: str) -> RuntimeCacheOperationResult:
        return self._slot_operation(slot_id, RuntimeCacheOperation.RESTORE, filename)

    def erase_slot(self, slot_id: int) -> RuntimeCacheOperationResult:
        _slot(slot_id)
        payload = self._post(f"/slots/{slot_id}?action=erase", {})
        return RuntimeCacheOperationResult(
            runtime_id=self.runtime_id,
            slot_id=slot_id,
            operation=RuntimeCacheOperation.ERASE,
            tokens_affected=_counter(payload, "n_erased", default=0),
        )

    def delete_checkpoint(self, filename: str) -> RuntimeCheckpointDeleteResult:
        """Delete a local runtime checkpoint without following links."""
        _filename(filename)
        if self.checkpoint_directory is None:
            raise RuntimeExecutionError("local checkpoint deletion is not configured")
        path = self.checkpoint_directory / filename
        started = time.monotonic()
        try:
            details = os.lstat(path)
        except FileNotFoundError:
            return RuntimeCheckpointDeleteResult(
                runtime_id=self.runtime_id,
                checkpoint_name=filename,
                existed=False,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        except OSError as exc:
            raise RuntimeExecutionError("checkpoint metadata could not be inspected") from exc
        if not stat.S_ISREG(details.st_mode):
            raise RuntimeExecutionError("checkpoint path must be a regular file")
        try:
            path.unlink()
        except OSError as exc:
            raise RuntimeExecutionError("checkpoint bytes could not be deleted") from exc
        return RuntimeCheckpointDeleteResult(
            runtime_id=self.runtime_id,
            checkpoint_name=filename,
            bytes_deleted=details.st_size,
            existed=True,
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

    def _slot_operation(
        self,
        slot_id: int,
        operation: RuntimeCacheOperation,
        filename: str,
    ) -> RuntimeCacheOperationResult:
        _slot(slot_id)
        _filename(filename)
        payload = self._post(f"/slots/{slot_id}?action={operation.value}", {"filename": filename})
        key = "n_saved" if operation is RuntimeCacheOperation.SAVE else "n_restored"
        bytes_key = "n_written" if operation is RuntimeCacheOperation.SAVE else "n_read"
        timings_raw = payload.get("timings")
        timings: Dict[str, Any] = timings_raw if isinstance(timings_raw, dict) else {}
        return RuntimeCacheOperationResult(
            runtime_id=self.runtime_id,
            slot_id=slot_id,
            operation=operation,
            tokens_affected=_counter(payload, key),
            bytes_affected=_counter(payload, bytes_key),
            elapsed_ms=float(timings.get(f"{operation.value}_ms") or 0),
            checkpoint_name=filename,
        )

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = request.Request(
            self.endpoint_url + path,
            data=body,
            headers={"content-type": "application/json", "accept": "application/json"},
            method="POST",
        )
        try:
            with self.opener(req, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
        except error.HTTPError as exc:
            raise RuntimeExecutionError(f"runtime returned HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeExecutionError("runtime transport failed") from exc
        if len(raw) > self.max_response_bytes:
            raise RuntimeExecutionError("runtime response exceeded byte limit")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeExecutionError("runtime returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeExecutionError("runtime response must be an object")
        return parsed


def _slot(slot_id: int) -> None:
    if not _non_negative_int(slot_id):
        raise RuntimeExecutionError("slot_id must be a non-negative integer")


def _filename(filename: str) -> None:
    if not isinstance(filename, str) or not filename or PurePath(filename).name != filename or filename in {".", ".."}:
        raise RuntimeExecutionError("checkpoint filename must be a basename")


def _counter(payload: Dict[str, Any], key: str, *, default: Optional[int] = None) -> int:
    value = payload.get(key, default)
    if not _non_negative_int(value):
        raise RuntimeExecutionError(f"invalid runtime counter: {key}")
    return cast(int, value)


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
