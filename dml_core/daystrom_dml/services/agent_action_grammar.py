"""Optional, request-bound syntactic constraints for public DML agent actions.

This module has no corpus or memory-store access. Values, references, tools and
the decision to finish remain model choices. The unchanged action parser still
owns semantic bounds, uniqueness and complete-action admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
import json

from ..contracts.agent_episode import ACTION_VERSION, canonical_json, episode_tool_definitions
from ..contracts.model_input import ModelInputError
from .model_input import ModelInputExecutionError


GRAMMAR_PROFILE = "dml-agent-action-grammar-v1"
RUNTIME_PINS = {"xgrammar": "0.2.7", "apache-tvm-ffi": "0.1.12"}
MAX_BOUND_REQUESTS = 64


def check_grammar_runtime():
    try:
        actual = {name: metadata.version(name) for name in RUNTIME_PINS}
    except metadata.PackageNotFoundError as exc:
        raise ModelInputError("Pinned optional action-grammar runtime is unavailable") from exc
    if actual != RUNTIME_PINS:
        raise ModelInputError("Action-grammar runtime differs from its pinned profile")
    return actual


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def action_schema(tools):
    """Derive only syntax from exact advertised public tool definitions.

    Field order follows this schema; any parser-valid action is expressible in
    that order. Semantic uniqueness and byte limits remain parser checks.
    """
    known = {tool["function"]["name"]: tool for tool in episode_tool_definitions()}
    if type(tools) is not list or len(tools) > len(known):
        raise ModelInputError("Action grammar requires a bounded public tool list")
    selected = set()
    for tool in tools:
        try:
            name = tool["function"]["name"]
            valid = type(name) is str and name in known and canonical_json(tool) == canonical_json(known[name])
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid or name in selected:
            raise ModelInputError("Action grammar requires distinct exact public tool schemas")
        selected.add(name)
    branches = []
    for name in sorted(selected):
        branches.append(_object({
            "schema_version": {"const": ACTION_VERSION}, "kind": {"const": "tool"},
            "name": {"const": name}, "arguments": deepcopy(known[name]["function"]["parameters"]),
        }))
    claim = _object({
        "key": {"type": "string"},
        "value": {"type": ["string", "number", "boolean", "null"]},
        "evidence_ids": {"type": "array", "items": {"type": "integer", "minimum": 0}, "maxItems": 128},
    })
    branches.append(_object({
        "schema_version": {"const": ACTION_VERSION}, "kind": {"const": "final"},
        "answer": _object({"claims": {"type": "array", "items": claim, "maxItems": 128}}),
    }))
    return {"anyOf": branches}


def schema_bytes(tools):
    # Preserve explicit property order. Sorted JSON would silently choose a
    # different generation order even though schema semantics look equivalent.
    return json.dumps(action_schema(tools), ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def policy_identity():
    return {"profile": GRAMMAR_PROFILE, "runtime_versions": check_grammar_runtime(),
            "public_schema_sha256": hashlib.sha256(schema_bytes(episode_tool_definitions())).hexdigest(),
            "public_tools_sha256": hashlib.sha256(canonical_json(episode_tool_definitions())).hexdigest(),
            "object_order": "declared", "strict_mode": True, "any_order": False,
            "any_whitespace": True, "max_whitespace_cnt": None,
            "cache_enabled": False, "compiler_threads": 1, "mask_backend": "cpu",
            "terminate_without_stop_token": False, "max_rollback_tokens": -1,
            "default_temperature": None, "special_tokens": "admitted-eos-only",
            "stop_tokens": "verified-tokenizer-eos-only",
            "max_bound_requests": MAX_BOUND_REQUESTS, "posthoc_repair": False}


def compile_action_grammar(tokenizer, vocab_size, tools):
    check_grammar_runtime()
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=[tokenizer.eos_token_id],
    )
    compiler = xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False)
    return compiler.compile_json_schema(
        schema_bytes(tools).decode(), strict_mode=True, any_order=False, any_whitespace=True,
        max_whitespace_cnt=None,
    )


class ActionLogitsProcessor:
    """One fresh matcher for one authenticated greedy CPU generation.

    Every chosen token is checked; no tokens are injected, decoded and repaired,
    or carried between calls. An incomplete bounded prefix is returned intact.
    """

    def __init__(self, compiled, tokenizer, input_ids, output_limit):
        import torch
        import xgrammar as xgr

        self._torch, self._xgr = torch, xgr
        self._prefix = tuple(input_ids)
        self._seen = self._prefix
        self._limit = output_limit
        self._vocab_size = compiled.tokenizer_info.vocab_size
        self._matcher = xgr.GrammarMatcher(
            compiled, override_stop_tokens=[tokenizer.eos_token_id], terminate_without_stop_token=False,
            max_rollback_tokens=-1, default_temperature=None,
        )
        with torch.device("cpu"):
            self._mask = xgr.allocate_token_bitmask(1, self._vocab_size)
        allowed = set(tokenizer.get_vocab().values())
        special = {index for index, token in tokenizer.added_tokens_decoder.items() if token.special}
        allowed -= special - {tokenizer.eos_token_id}
        self._allowed = torch.zeros(self._vocab_size, dtype=torch.bool, device="cpu")
        self._allowed[list(allowed)] = True
        self.calls = 0

    def _accept_next(self, complete_ids):
        ids = tuple(complete_ids)
        if (ids[:len(self._seen)] != self._seen or len(ids) != len(self._seen) + 1
                or len(ids) > len(self._prefix) + self._limit
                or type(ids[-1]) is not int or not 0 <= ids[-1] < self._vocab_size
                or not bool(self._allowed[ids[-1]])
                or not self._matcher.accept_token(ids[-1])):
            raise ModelInputExecutionError("Action generation violated its authenticated grammar prefix")
        self._seen = ids

    def __call__(self, input_ids, scores):
        torch = self._torch
        if (self.calls >= self._limit
                or type(self._mask) is not torch.Tensor or self._mask.dtype != torch.int32
                or self._mask.device.type != "cpu"
                or tuple(self._mask.shape) != (1, (self._vocab_size + 31) // 32)
                or not self._mask.is_contiguous()
                or type(input_ids) is not torch.Tensor or input_ids.dtype != torch.long
                or input_ids.device.type != "cpu" or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or type(scores) is not torch.Tensor or scores.dtype != torch.float32
                or scores.device.type != "cpu" or tuple(scores.shape) != (1, self._vocab_size)
                or not bool(torch.isfinite(scores).all())):
            raise ModelInputExecutionError("Action grammar requires finite single-request CPU logits")
        ids = tuple(input_ids[0].tolist())
        if not self.calls:
            if ids != self._prefix:
                raise ModelInputExecutionError("Action grammar initial prefix differs")
        else:
            self._accept_next(ids)
        if self._matcher.is_terminated():
            raise ModelInputExecutionError("Action grammar received tokens after termination")
        self._matcher.fill_next_token_bitmask(self._mask)
        self._xgr.apply_token_bitmask_inplace(scores, self._mask, vocab_size=self._vocab_size, backend="cpu")
        scores.masked_fill_(~self._allowed.unsqueeze(0), float("-inf"))
        if not bool(torch.isfinite(scores).any()):
            raise ModelInputExecutionError("Action grammar has no valid next token")
        self.calls += 1
        return scores

    def finish(self, complete_ids):
        ids = tuple(complete_ids)
        generated = len(ids) - len(self._prefix)
        if generated == 0 and self.calls == 0 and ids == self._prefix:
            return
        if self.calls != generated:
            raise ModelInputExecutionError("Action generation bypassed its token masks")
        self._accept_next(ids)
