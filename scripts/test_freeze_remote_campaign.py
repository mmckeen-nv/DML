"""Freeze candidate selection and reviewed provenance without generation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def freeze(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import freeze_remote_campaign
    return freeze_remote_campaign


def test_profile_is_selected_from_manifest_with_legacy_default(freeze):
    from daystrom_dml.services.remote_vllm_action_input import CONSUMER_PROFILE, JSON_CONSUMER_PROFILE
    assert freeze.selected_profile({}) == CONSUMER_PROFILE
    assert freeze.selected_profile({'consumer_profile':JSON_CONSUMER_PROFILE}) == JSON_CONSUMER_PROFILE
    with pytest.raises(ValueError,match='differs'):
        freeze.selected_profile({'consumer_profile':JSON_CONSUMER_PROFILE},CONSUMER_PROFILE)


@pytest.mark.parametrize('profile',['nemotron-remote-vllm-action-v1','nemotron-remote-vllm-action-json-v2','nemotron-remote-vllm-native-tools-v1','nemotron-remote-vllm-native-tools-v2'])
def test_selected_profile_and_typed_limits_survive_real_cli(freeze,tmp_path,monkeypatch,profile):
    from dataclasses import asdict
    from scripts import agent_episodes
    captured={}
    def capture(**kwargs):
        captured.update(kwargs)
        return {'summary':{'attempted_tasks':0,'completed_tasks':0},'episodes':[]}
    monkeypatch.setattr(agent_episodes,'run_campaign',capture)
    command=freeze.producer_command(sys.executable,profile,tmp_path/'snapshot',tmp_path,freeze.declared_limits())
    assert agent_episodes.main(command[3:]) == 0
    assert captured['consumer_profile'] == profile
    assert json.dumps(asdict(captured['limits']),sort_keys=True) == json.dumps(freeze.declared_limits(),sort_keys=True)


def test_runtime_attestation_binds_manifest_and_evidence(freeze,tmp_path):
    manifest={'endpoint':'e','model':'m','model_revision':'r'}
    (tmp_path/'remote-vllm-manifest.json').write_text(json.dumps(manifest))
    evidence=tmp_path/'launch.json'
    evidence.write_text('{"synthetic":true}')
    receipt={'schema_version':'dml-remote-runtime-attestation-v1',**manifest,
        'loaded_model_revision_attested':True,'manifest_sha256':freeze.digest(tmp_path/'remote-vllm-manifest.json'),
        'evidence_files':{str(evidence):freeze.digest(evidence)}}
    path=tmp_path/'attestation.json'
    path.write_text(json.dumps(receipt))
    expected=freeze.digest(path)
    binding,files=freeze.runtime_attestation(path,expected,manifest,tmp_path)
    assert binding['sha256']==expected and str(evidence) in files and str(path) in files
    assert freeze.runtime_attestation(None,None,manifest,tmp_path)==(None,{})
    with pytest.raises(ValueError,match='both'):
        freeze.runtime_attestation(path,None,manifest,tmp_path)
    with pytest.raises(ValueError,match='digest'):
        freeze.runtime_attestation(path,'0'*64,manifest,tmp_path)
    evidence.write_text('tampered')
    with pytest.raises(ValueError,match='evidence differs'):
        freeze.runtime_attestation(path,expected,manifest,tmp_path)


@pytest.mark.parametrize("native",[None,"nemotron-remote-vllm-native-tools-v1","nemotron-remote-vllm-native-tools-v2"])
def test_full_freeze_uses_selected_identity_and_tracks_renderer_harness(freeze,tmp_path,monkeypatch,native):
    from daystrom_dml.services import remote_vllm_action_input as adapter
    candidate=tmp_path/'candidate'
    candidate.mkdir()
    bundle=candidate/'snapshot'
    bundle.mkdir()
    (bundle/'remote-vllm-manifest.json').write_text('{}')
    (candidate/'lifecycle').mkdir()
    (candidate/'lifecycle/qualification.json').write_text('{}')
    source=tmp_path/'source'
    source.mkdir()
    required=['dml_core/daystrom_dml/services/remote_vllm_action_input.py',
        'dml_core/daystrom_dml/services/qwen_model_snapshot.py',
        'dml_core/scripts/remote_vllm_synthetic.py','scripts/freeze_remote_campaign.py']
    if native:
        required += ['dml_core/daystrom_dml/services/native_remote_vllm_action_input.py',
                     'dml_core/scripts/native_dml_synthetic.py']
    for name in required:
        path=source/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('synthetic')
    sqlite=tmp_path/'sqlite.so'
    sqlite.write_bytes(b'synthetic')
    profile=native or adapter.JSON_CONSUMER_PROFILE
    monkeypatch.setattr(adapter,'verify_remote_manifest',lambda path:{'consumer_profile':profile})
    observed=[]
    class Consumer:
        def __init__(self,path,*,consumer_profile,offline):
            observed.append((consumer_profile,offline))
            self.identity=SimpleNamespace(to_payload=lambda:{'runtime_identity':'synthetic-v2'})
        def __enter__(self):return self
        def __exit__(self,*args):pass
    if native:
        from daystrom_dml.services import native_remote_vllm_action_input as native_adapter
        monkeypatch.setattr(native_adapter,'NativeRemoteVLLMActionInputConsumer',Consumer)
    else:
        monkeypatch.setattr(adapter,'RemoteVLLMActionInputConsumer',Consumer)
    monkeypatch.setattr(freeze.importlib.metadata,'distributions',lambda:[])
    monkeypatch.setattr(freeze.subprocess,'check_output',lambda args,**kwargs:
        '\0'.join(required).encode() if args[1]=='ls-files' else 'synthetic-commit\n')
    freeze.main(['--candidate-root',str(candidate),'--source-root',str(source),
                 '--sqlite-library',str(sqlite),'--consumer-profile',profile])
    spec=json.loads((candidate/'campaign-once/spec.json').read_text())
    frozen=json.loads((candidate/'campaign-once/freeze.json').read_text())
    assert observed==[(profile,True)]
    assert spec['consumer_profile']==profile and spec['model_identity']['runtime_identity']=='synthetic-v2'
    assert spec['loaded_model_revision_attested'] is False and spec['source_ci_qualified'] is False
    assert frozen['command'][frozen['command'].index('--consumer-profile')+1]==profile
    assert all(str(source/name) in frozen['files'] for name in required)
