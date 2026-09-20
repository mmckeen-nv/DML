"""Real token-mask controls; no trained model, truth corpus or generated answers."""
from __future__ import annotations

from copy import deepcopy
import json
import os

import pytest

from daystrom_dml.contracts.agent_episode import (
    AGENT_POLICY, AgentEpisodeError, episode_tool_definitions, parse_agent_action,
)
from daystrom_dml.contracts.model_input import ModelInputError
from daystrom_dml.services.agent_action_grammar import (
    ActionLogitsProcessor, action_schema, check_grammar_runtime, compile_action_grammar,
)
from daystrom_dml.services.model_input import ModelInputExecutionError
from test_qwen_chat_template import tokenizer as tokenizer


@pytest.fixture(scope="module", autouse=True)
def grammar_runtime():
    try:
        return check_grammar_runtime()
    except ModelInputError as exc:
        if os.environ.get("REQUIRE_MODEL_INPUT_TESTS") == "1":
            pytest.fail(str(exc), pytrace=False)
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def grammar(tokenizer, grammar_runtime):
    return compile_action_grammar(tokenizer, len(tokenizer) + 8, episode_tool_definitions())


def _wire(action):
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def _masked_tokens(tokenizer, grammar, text, *, eos=True):
    import torch

    prefix = tuple(tokenizer.encode("Unrelated caller context.", add_special_tokens=False))
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if eos:
        tokens.append(tokenizer.eos_token_id)
    processor = ActionLogitsProcessor(grammar, tokenizer, prefix, max(len(tokens), 1))
    complete = list(prefix)
    for token in tokens:
        scores = torch.zeros((1, grammar.tokenizer_info.vocab_size), dtype=torch.float32)
        scores[0, token] = 10
        processor(torch.tensor([complete], dtype=torch.long), scores)
        if not torch.isfinite(scores[0, token]):
            return False, processor, tuple(complete)
        assert scores.argmax(-1).item() == token
        complete.append(token)
    processor.finish(complete)
    return True, processor, tuple(complete)


@pytest.mark.parametrize("name,arguments", [
    ("retrieve", {"query": "arbitrary topic", "top_k": 7}),
    ("ingest", {"text": "Arbitrary new memory"}),
    ("update", {"record_ref": "r91", "text": "An update", "reason": "User request"}),
    ("promote", {"record_refs": ["r4", "r19"], "text": "Combined memory", "reason": "Requested"}),
    ("supersede", {"record_ref": "r2", "replacement_ref": "r99", "reason": "Requested"}),
    ("retire", {"record_ref": "r1", "reason": "Requested"}),
])
def test_every_advertised_tool_and_model_selected_values_remain_available(tokenizer, grammar, name, arguments):
    action = {"schema_version": "dml-agent-action-v1", "kind": "tool", "name": name, "arguments": arguments}
    assert _masked_tokens(tokenizer, grammar, _wire(action))[0]
    assert parse_agent_action(_wire(action)) == action


@pytest.mark.parametrize("claims", [[]] + [
    [{"key": "free.choice", "value": value, "evidence_ids": []}]
    for value in ("自由 café & \\ \"", 42, 1.25, True, None)
])
def test_final_claims_can_be_empty_or_contain_any_scalar_without_forced_evidence(tokenizer, grammar, claims):
    action = {"schema_version": "dml-agent-action-v1", "kind": "final", "answer": {"claims": claims}}
    assert _masked_tokens(tokenizer, grammar, _wire(action))[0]
    assert parse_agent_action(_wire(action)) == action


@pytest.mark.parametrize("text", [
    '{"schema_version":"dml-qwen-chatml-fields-v2","kind":"final","answer":{"claims":[]}}',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]},"reason":"extra"}',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[],"confidence":1}}',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[{"key":"x","value":{},"evidence_ids":[]}]}}',
    '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[{"key":"x","value":1,"evidence_ids":[true]}]}}',
    '{"schema_version":"dml-agent-action-v1","answer":{"claims":[]}}',
    '{"content":"answer","role":"assistant"}',
    '```json\n{}\n```',
])
def test_invalid_structure_cannot_pass_token_masks(tokenizer, grammar, text):
    assert not _masked_tokens(tokenizer, grammar, text)[0]
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(text)


def test_semantic_duplicate_evidence_is_still_rejected_by_unchanged_parser(tokenizer, grammar):
    text = _wire({"schema_version": "dml-agent-action-v1", "kind": "final", "answer": {
        "claims": [{"key": "unverified", "value": "model choice", "evidence_ids": [5, 5]}],
    }})
    assert _masked_tokens(tokenizer, grammar, text)[0]
    with pytest.raises(AgentEpisodeError, match="Duplicate evidence"):
        parse_agent_action(text)


def test_token_limit_preserves_incomplete_prefix_without_closing_or_repair(tokenizer, grammar):
    text = '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":['
    accepted, processor, complete = _masked_tokens(tokenizer, grammar, text, eos=False)
    assert accepted and processor.calls == len(complete) - len(processor._prefix)
    decoded = tokenizer.decode(complete[len(processor._prefix):], skip_special_tokens=False)
    assert decoded == text
    with pytest.raises(AgentEpisodeError):
        parse_agent_action(decoded)


def test_complete_action_at_exact_token_limit_does_not_require_or_invent_eos(tokenizer, grammar):
    text = '{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]}}'
    accepted, processor, complete = _masked_tokens(tokenizer, grammar, text, eos=False)
    assert accepted and processor.calls == len(complete) - len(processor._prefix)
    assert parse_agent_action(text)["answer"] == {"claims": []}


def test_initial_mask_excludes_eos_special_tokens_and_unused_model_rows(tokenizer, grammar):
    import torch

    processor = ActionLogitsProcessor(grammar, tokenizer, (5,), 8)
    scores = processor(torch.tensor([[5]], dtype=torch.long),
                       torch.zeros((1, grammar.tokenizer_info.vocab_size), dtype=torch.float32))
    for token in (tokenizer.eos_token_id, tokenizer.pad_token_id,
                  tokenizer.convert_tokens_to_ids("<|im_start|>"), len(tokenizer)):
        assert torch.isneginf(scores[0, token])
    with pytest.raises(ModelInputExecutionError):
        processor.finish((5, tokenizer.convert_tokens_to_ids("<|im_start|>")))


def test_mask_storage_stays_on_cpu_when_process_default_device_differs(tokenizer, grammar):
    import torch

    with torch.device("meta"):
        processor = ActionLogitsProcessor(grammar, tokenizer, (5,), 8)
    assert processor._mask.device.type == processor._allowed.device.type == "cpu"


@pytest.mark.parametrize("mutation", ["batch", "prefix", "scores_dtype", "nan", "positive_inf", "mask_shape", "mask_dtype", "no_token"])
def test_mask_contract_violations_fail_closed(tokenizer, grammar, mutation):
    import torch

    processor = ActionLogitsProcessor(grammar, tokenizer, (5,), 8)
    ids = torch.tensor([[5]], dtype=torch.long)
    scores = torch.zeros((1, grammar.tokenizer_info.vocab_size), dtype=torch.float32)
    if mutation == "batch":
        ids = ids.repeat(2, 1)
    elif mutation == "prefix":
        ids[0, 0] = 6
    elif mutation == "scores_dtype":
        scores = scores.half()
    elif mutation == "nan":
        scores[0, 0] = float("nan")
    elif mutation == "positive_inf":
        scores[0, 0] = float("inf")
    elif mutation == "mask_shape":
        processor._mask = processor._mask[:, :1]
    elif mutation == "mask_dtype":
        processor._mask = processor._mask.long()
    else:
        processor._allowed.zero_()
    with pytest.raises(ModelInputExecutionError):
        processor(ids, scores)


def test_mutated_public_schema_and_duplicate_tools_are_not_silently_narrowed():
    tools = episode_tool_definitions()
    assert len(action_schema([])["anyOf"]) == 1
    changed = deepcopy(tools[:1])
    changed[0]["function"]["parameters"]["properties"]["top_k"]["maximum"] = 100
    for invalid in (changed, [tools[0], tools[0]], [{"type": "function", "function": {"name": "hidden", "parameters": {}}}]):
        with pytest.raises(ModelInputError):
            action_schema(invalid)


def test_policy_guidance_is_generic_and_does_not_relax_parser():
    assert "Repeating an identical retrieval against unchanged memory adds no evidence" in AGENT_POLICY
    assert "When the task explicitly requests a lifecycle operation" in AGENT_POLICY
    with pytest.raises(AgentEpisodeError):
        parse_agent_action('{"schema_version":"dml-agent-action-v1","kind":"final","answer":{"claims":[]},"extra":true}')
