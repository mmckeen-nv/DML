"""Neutral M8 live-generation binding; development-only, no M7 relabeling.

Reuses the qualified Llama3 generation mechanics (adapter loading, xgrammar
settings, sampler, stop/vocabulary controls) from llama3_sft_runtime without
its hardcoded DML system policy and tool schemas. The neutral M8 protocol
(m8_protocol) owns rendering, token evidence and action parsing; this module
owns grammar compilation and exact-token generation for one arm. The M7
runtime source, identity and evidence are unchanged.

No model is loaded by importing this module; heavy dependencies stay lazy.
"""

from __future__ import annotations

import hashlib

from ..contracts.agent_episode import canonical_json
from . import m8_protocol
from .llama3_sft_runtime import (
    STOPS,
    VOCAB,
    EnabledAdapterRuntime,
    cpu_rng,
    sample_next,
    special_ids,
)
from .llama3_sft_v3_action_input import sampling_policy_identity

RUNTIME_PREFIX = "dml-m8-neutral-action-runtime-v1:"
GRAMMAR_IDENTITY = "xgrammar0.1.33 strict any-whitespace; neutral retrieve/final branches"


def _arm(arm):
    if type(arm) is not str or arm not in m8_protocol.ARMS:
        raise ValueError("Unknown M8 arm")
    return arm


def neutral_identity(arm, *, base_identity: dict) -> dict:
    """Versioned neutral runtime identity; never an M7 runtime identity."""
    _arm(arm)
    policy = {
        "protocol": m8_protocol.policy_identity(arm),
        "base_identity": base_identity,
        "sampling": sampling_policy_identity(),
        "grammar": GRAMMAR_IDENTITY,
        "termination": "optional single terminal stop; complete-at-cap accepted by original parser; no repair",
    }
    return {
        "runtime_identity": RUNTIME_PREFIX + hashlib.sha256(canonical_json(policy)).hexdigest(),
        "policy": policy,
    }


def compile_grammar(tokenizer, arm):
    """Compile the NEUTRAL action schema for one arm with M7's pinned settings."""
    _arm(arm)
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=VOCAB, stop_token_ids=STOPS)
    return xgr.GrammarCompiler(info, max_threads=1, cache_enabled=False).compile_json_schema(
        m8_protocol.schema_bytes(arm).decode(),
        strict_mode=True,
        any_whitespace=True,
        max_whitespace_cnt=None,
    )


def project_output(output_ids, tokenizer, arm):
    """Project exact output IDs through the neutral parser; never repair."""
    _arm(arm)
    raw = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ended = bool(output_ids and output_ids[-1] in STOPS)
    body = output_ids[:-1] if ended else output_ids
    text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    result = dict(
        raw_text=raw,
        text=text,
        finish_reason="eos" if ended else "length",
        stop_token_id=output_ids[-1] if ended else None,
        derived_action=None,
        derived_action_text=None,
        action_error=None,
    )
    try:
        if any(type(t) is not int or not 0 <= t < VOCAB for t in output_ids):
            raise ValueError("Invalid token row")
        if any(t in special_ids(tokenizer) for t in body):
            raise ValueError("Embedded output control")
        # Neutral parser: rejects unadvertised tools, malformed actions and
        # context_only tool use; no DML tool schema is consulted.
        action = m8_protocol.parse_action(text, arm)
        result.update(derived_action=action, derived_action_text=text)
    except Exception as exc:
        result["action_error"] = type(exc).__name__ + ": " + str(exc)
    return result


class NeutralM8AdapterRuntime(EnabledAdapterRuntime):
    """Enabled-adapter generation bound to one neutral arm's grammar/parser."""

    def __init__(self, model_path, checkpoint, tokenizer, identity, validate, *, arm):
        _arm(arm)
        self._arm = arm
        super().__init__(model_path, checkpoint, tokenizer, identity, validate)

    def execute(self, compiled):
        if not isinstance(compiled, m8_protocol.CompiledM8Request):
            raise ValueError("Neutral generation requires an exact CompiledM8Request")
        if compiled.arm != self._arm:
            raise ValueError("Compiled request arm differs from this runtime")

        import torch
        import xgrammar as xgr

        result: dict = dict(
            arm=self._arm,
            protocol_profile=m8_protocol.PROFILE,
            input_ids=list(compiled.input_ids),
            output_ids=[],
            model_identity=self.identity,
            input_token_count=None,
            output_token_count=None,
            usage_unknown=True,
            execution_error=None,
            action_error=None,
        )
        ids = []
        past = None
        try:
            self.validate()
            model = self.load_model()
            grammar = compile_grammar(self.tokenizer, self._arm)
            matcher = xgr.GrammarMatcher(
                grammar,
                override_stop_tokens=STOPS,
                terminate_without_stop_token=False,
                max_rollback_tokens=-1,
            )
            mask = xgr.allocate_token_bitmask(1, VOCAB)
            allowed = torch.ones(VOCAB, dtype=torch.bool, device="cpu")
            allowed[list(special_ids(self.tokenizer) - set(STOPS))] = False
            all_ids = tuple(compiled.input_ids)
            input_tensor = torch.tensor([all_ids], dtype=torch.long, device="cuda:0")
            with torch.inference_mode(), cpu_rng(torch), torch.autocast("cuda", dtype=torch.bfloat16):
                for step in range(compiled.output_reserved_tokens):
                    output = model(input_ids=input_tensor, past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = output.past_key_values
                    scores = output.logits[:, -1, :].float().cpu()
                    if tuple(scores.shape) != (1, VOCAB) or not bool(torch.isfinite(scores).all()):
                        raise ValueError("Invalid logits")
                    matcher.fill_next_token_bitmask(mask)
                    xgr.apply_token_bitmask_inplace(scores, mask, vocab_size=VOCAB, backend="cpu")
                    scores.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
                    if not bool(torch.isfinite(scores).any()):
                        raise ValueError("Empty grammar distribution")
                    token = sample_next(torch, torch.tensor([all_ids], dtype=torch.long), scores)
                    if not bool(allowed[token]) or not matcher.accept_token(token):
                        raise ValueError("Sample violates token mask")
                    ids.append(token)
                    all_ids += (token,)
                    if token in STOPS:
                        break
                    input_tensor = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
            self.validate()
            result.update(
                output_ids=ids,
                input_token_count=len(compiled.input_ids),
                output_token_count=len(ids),
                usage_unknown=False,
            )
            result.update(project_output(ids, self.tokenizer, self._arm))
        except Exception as exc:
            result.update(output_ids=ids, execution_error=type(exc).__name__ + ": " + str(exc))
            try:
                result["raw_text"] = self.tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
            except Exception as decode_error:
                result["raw_text"] = None
                result["decode_error"] = type(decode_error).__name__ + ": " + str(decode_error)
        finally:
            past = None
        return result
