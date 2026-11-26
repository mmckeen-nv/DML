"""V3 registration isolation and unchanged authenticated consumer mechanics."""

import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import struct
from pathlib import Path

import pytest

from daystrom_dml.contracts.model_input import ModelInputError
from daystrom_dml.services import llama3_sft_action_input as v2
from daystrom_dml.services import llama3_sft_v3_action_input as v3
from test_llama3_sft_action_input import consumer as legacy_fake, request


def normalized(node):
    text = ast.unparse(node).replace("LLAMA3_SFT_V3_CONSUMER_PROFILE", "LLAMA3_SFT_CONSUMER_PROFILE")
    return ast.dump(ast.parse(text), include_attributes=False)


def test_all_consumer_methods_remain_exact_v2_mechanics():
    left = ast.parse(inspect.getsource(v2.LocalLlama3SFTActionInputConsumer)).body[0]
    right = ast.parse(inspect.getsource(v3.LocalLlama3SFTV3ActionInputConsumer)).body[0]
    a = {n.name: normalized(n) for n in left.body if isinstance(n, ast.FunctionDef)}
    b = {n.name: normalized(n) for n in right.body if isinstance(n, ast.FunctionDef)}
    assert a == b
    assert v3.runtime is v2.runtime
    assert v3.sampling_policy_identity() == v2.sampling_policy_identity()
    assert inspect.getsource(v3.source_order_request) == inspect.getsource(v2.source_order_request)


def manifest(module):
    value = dict(
        schema_version=module.MANIFEST_VERSION,
        model_id=module.MODEL,
        revision=module.REVISION,
        adapter_enabled=True,
        model_directory="model",
        adapter_directory="adapter",
        files={"model/tokenizer.json": "a" * 64},
        chat_template_digest=module.VENDOR_TEMPLATE_DIGEST,
        runtime=dict(versions=module.VERSIONS, image=module.IMAGE, required_environment=module.REQUIRED_ENVIRONMENT),
    )
    if module is v3:
        value.update(
            consumer_profile=v3.LLAMA3_SFT_V3_CONSUMER_PROFILE, training_provenance=deepcopy(v3.TRAINING_PROVENANCE)
        )
    return value


def test_distinct_identity_binds_v3_checkpoint_and_training_lineage():
    m = manifest(v3)
    a = v3.identity_for_manifest(m)
    assert a.runtime_identity.startswith(v3.RUNTIME_PREFIX)
    assert a != v2.identity_for_manifest(manifest(v2))
    assert v3.CHECKPOINT != v2.CHECKPOINT
    changed = deepcopy(m)
    changed["training_provenance"]["optimizer_steps"] = 90
    assert v3.identity_for_manifest(changed) != a
    assert v3.OFFICIAL_INVENTORY_DIGEST == v2.OFFICIAL_INVENTORY_DIGEST
    assert v3.REQUIRED_ENVIRONMENT == v2.REQUIRED_ENVIRONMENT


@pytest.mark.parametrize("module,other", [(v3, v2), (v2, v3)])
def test_cross_version_manifest_refused_before_file_loading(tmp_path, module, other):
    (tmp_path / module.MANIFEST_NAME).write_text(json.dumps(manifest(other)))
    with pytest.raises(ModelInputError, match="enabled-only"):
        module.verify_manifest(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("optimizer_steps", 90), ("selection", "best_dev"), ("independent_training_review_sha256", "0" * 64)],
)
def test_v3_rejects_other_training_lineage(tmp_path, field, value):
    m = manifest(v3)
    m["training_provenance"][field] = value
    (tmp_path / v3.MANIFEST_NAME).write_text(json.dumps(m))
    with pytest.raises(ModelInputError, match="enabled-only"):
        v3.verify_manifest(tmp_path)


def test_v3_rejects_wrong_profile_before_manifest_access(tmp_path):
    with pytest.raises(ModelInputError, match="profile"):
        v3.LocalLlama3SFTV3ActionInputConsumer(tmp_path, consumer_profile=v2.LLAMA3_SFT_CONSUMER_PROFILE)


def test_v3_authenticated_artifacts_and_partial_failure_retention():
    c = legacy_fake()
    c.__class__ = v3.LocalLlama3SFTV3ActionInputConsumer
    c._identity = replace(c._identity, runtime_identity=v3.RUNTIME_PREFIX + "4" * 64)
    r = request()
    a = c.compile(r.messages, r.tools, output_reserved_tokens=8)
    with pytest.raises(ModelInputError):
        c.execute(replace(a, identity=replace(a.identity, runtime_identity=v2.RUNTIME_PREFIX + "4" * 64)))
    result = c.execute(a)
    assert result.output_ids == (1, 2, 128009)
    with pytest.raises(ModelInputError):
        c.execute(a)
    assert c.last_execution is None
    c.validate_output_tokens = lambda req, ids: c.decode_output(ids)
    envelope = dict(
        artifact_digest=a.artifact_digest,
        exception_type="RuntimeError",
        exception_message="failed",
        runtime_result=dict(
            model_identity=a.identity.to_payload(),
            input_ids=list(a.input_ids),
            output_ids=[1],
            raw_text="{",
            usage_unknown=True,
            execution_error="retained failure",
        ),
    )
    c.validate_failed_execution(a, r, envelope)
    envelope["runtime_result"]["raw_text"] = "invented"
    with pytest.raises(ModelInputError):
        c.validate_failed_execution(a, r, envelope)


@pytest.mark.parametrize("module,wrong", [(v3, v2), (v2, v3)])
def test_other_adapter_checkpoint_rejected_even_with_matching_caller_inventory(tmp_path, monkeypatch, module, wrong):
    # Synthetic file fixture exercises fixed-pin admission, not actual pretrained qualification.
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    model_files = {
        "model.safetensors.index.json": json.dumps({"weight_map": {"w": "shard.safetensors"}}),
        "config.json": "{}",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "generation_config.json": "{}",
        "special_tokens_map.json": "{}",
        "shard.safetensors": "x",
    }
    for name, value in model_files.items():
        (model / name).write_text(value)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
    m = manifest(module)
    actual_sha = module.sha
    m["files"] = {str(p.relative_to(tmp_path)): actual_sha(p) for p in model.iterdir()}
    m["files"].update({"adapter/" + name: h for name, h in wrong.CHECKPOINT.items()})
    monkeypatch.setattr(
        module, "OFFICIAL_INVENTORY_DIGEST", module._digest({p.name: actual_sha(p) for p in model.iterdir()})
    )
    monkeypatch.setattr(
        module, "sha", lambda p: wrong.CHECKPOINT[Path(p).name] if Path(p).parent == adapter else actual_sha(p)
    )
    (tmp_path / module.MANIFEST_NAME).write_text(json.dumps(m))
    with pytest.raises(ModelInputError, match="final scheduled"):
        module.verify_manifest(tmp_path)
