"""Admission guards do not load a model or contact Docker."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from durable_campaign import digest
from freeze_llama3_sft_v3_campaign import bound_json, validate_deployment


def test_evidence_role_requires_exact_reviewed_bytes(tmp_path):
    path = tmp_path/'receipt.json'
    path.write_text(json.dumps({'passed':True}))
    reference = {'path':str(path),'sha256':digest(path)}
    assert bound_json(reference,{str(path):digest(path)}) == {'passed':True}
    path.write_text(json.dumps({'passed':False}))
    with pytest.raises(ValueError):
        bound_json(reference,{str(path):reference['sha256']})


@pytest.mark.parametrize('field,value',[('command',['python','wrong.py']),('image','unversioned:latest'),('gpu',False)])
def test_config_must_match_reviewed_gpu_deployment(field,value):
    approved = {'command':['python3','-m','scripts.agent_episodes'],'image':'sha256:'+'a'*64,'gpu':True}
    candidate = {**approved,field:value}
    with pytest.raises(ValueError):
        validate_deployment(candidate,expected_command=approved['command'],image=approved['image'],reviewed=approved)


def test_reviewed_command_must_still_be_original_full_producer():
    approved = {'command':['fake-producer'],'image':'sha256:'+'a'*64}
    with pytest.raises(ValueError):
        validate_deployment(approved,expected_command=['python3','-m','scripts.agent_episodes'],image=approved['image'],reviewed=approved)


def test_active_freezer_must_be_tracked():
    from freeze_llama3_sft_v3_campaign import validate_source_inventory
    sources = dict.fromkeys([
        'scripts/freeze_llama3_sft_v3_campaign.py', 'scripts/durable_campaign.py',
        'dml_core/scripts/llama3_sft_v3_synthetic.py', 'dml_core/scripts/agent_campaign_evidence.py',
        'dml_core/daystrom_dml/services/llama3_sft_action_input.py',
        'dml_core/daystrom_dml/services/llama3_sft_v3_action_input.py',
        'dml_core/daystrom_dml/services/llama3_sft_runtime.py'], 'a'*64)
    validate_source_inventory(sources)
    del sources['scripts/freeze_llama3_sft_v3_campaign.py']
    sources['scripts/freeze_llama3_sft_campaign.py'] = 'a'*64
    with pytest.raises(ValueError, match='not tracked'):
        validate_source_inventory(sources)
