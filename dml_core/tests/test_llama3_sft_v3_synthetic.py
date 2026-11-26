from copy import deepcopy
import pytest
from scripts import llama3_sft_v3_synthetic as qualification


def test_original_fixture_builders_and_limits(tmp_path):
    for stage, original in qualification.STAGES.items():
        suite = qualification.prepare_suite(tmp_path / (stage + '.json'), stage=stage, identity={'test': 'identity'})
        assert [case['id'] for case in suite['cases']] == list(original.CASE_IDS)
        assert qualification.validate_suite(suite).max_steps == 6
        assert suite['limits'] == qualification.shared.QUALIFICATION_LIMITS
        changed = deepcopy(suite)
        changed['limits']['max_steps'] = 7
        changed['suite_digest'] = qualification.shared._digest({k:v for k,v in changed.items() if k != 'suite_digest'})
        with pytest.raises(ValueError):
            qualification.validate_suite(changed)


def test_failed_planning_withholds_all_readiness(tmp_path, monkeypatch):
    paths = [tmp_path / (stage + '.json') for stage in ('planning','readiness')]
    for stage, path in zip(('planning','readiness'), paths):
        qualification.prepare_suite(path, stage=stage, identity={'test':'identity'})
    calls = []
    def run(path, **kwargs):
        calls.append(path)
        return {'all_synthetic_cases_pass': False}
    monkeypatch.setattr(qualification, 'run_suite', run)
    result = qualification.run_staged(*paths, snapshot=tmp_path, output=tmp_path/'results')
    assert calls == [paths[0]]
    assert result['readiness'] is None and not result['qualified']


def test_stages_cannot_mix_candidate_identities(tmp_path, monkeypatch):
    paths = [tmp_path / (stage + '.json') for stage in ('planning','readiness')]
    for i, (stage,path) in enumerate(zip(('planning','readiness'), paths)):
        qualification.prepare_suite(path, stage=stage, identity={'test':i})
    monkeypatch.setattr(qualification, 'run_suite', lambda *a, **k: pytest.fail('must not run'))
    with pytest.raises(ValueError, match='candidates'):
        qualification.run_staged(*paths, snapshot=tmp_path, output=tmp_path/'results')


def test_real_worker_authority_and_offline_replay(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    from queue import Queue
    from test_agent_episode_runtime import ValidationScriptedConsumer
    from daystrom_dml.services import agent_episode
    from daystrom_dml.contracts.model_input import ModelInputIdentity
    identity = ModelInputIdentity(model_digest='1'*64,tokenizer_digest='2'*64,
        chat_template_digest='3'*64,runtime_identity='dml-llama3-sft-action-runtime-v2:'+'a'*64,
        model_window_tokens=8192)
    path = tmp_path/'suite.json'
    suite = qualification.prepare_suite(path,stage='readiness',identity=identity.to_payload())
    case = suite['cases'][0]
    texts = {}
    def final(request):
        record = next(record for record in json.loads(request['messages'][-1]['content'])['records']
                      if record.get('claim_value') == case['expected']['value'])
        return {'schema_version':'dml-agent-action-v1','kind':'final','answer':{'claims':[
            {'key':case['expected']['key'],'value':case['expected']['value'],'evidence_ids':[record['id']]}]}}
    class Consumer(ValidationScriptedConsumer):
        _identity = identity
        def __init__(self,*args,**kwargs):
            super().__init__([{'schema_version':'dml-agent-action-v1','kind':'tool','name':'retrieve',
                              'arguments':{'query':'Synthetic archive access phrase','top_k':10}},final])
            self.identity = identity
        def compile(self,*args,**kwargs):
            artifact = replace(super().compile(*args,**kwargs),identity=identity,model_window_tokens=8192)
            self.current = artifact.request_digest
            return artifact
        def execute(self,artifact):
            result = super().execute(artifact)
            texts[artifact.request_digest] = result.text
            return result
        def release_compiled(self,artifact):
            pass
        def validate_output_tokens(self,request,ids):
            assert ids == [5,6]
        def decode_output(self,ids):
            return texts[self.current]
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
    monkeypatch.setattr(agent_episode,'_open_consumer',lambda *args:Consumer())
    monkeypatch.setattr(qualification,'LocalLlama3SFTV3ActionInputConsumer',Consumer)
    output = tmp_path/'results'
    directory = output/case['id']
    directory.mkdir(parents=True)
    channel = Queue()
    qualification.shared._worker(channel,snapshot='mock-only',directory=directory,suite=suite,case=case)
    trace = list(channel.queue)
    assert not [event for event in trace if event['kind']=='diagnostic_error'],trace
    previous = '0'*64
    retained = []
    for event in trace:
        event = {**event,**({'episode_sequence':event['sequence']} if 'sequence' in event else {}),
                 'sequence':len(retained),'received_ns':0,'previous_digest':previous}
        event['event_digest'] = qualification.shared._digest(event)
        previous = event['event_digest']
        retained.append(event)
    (directory/'events.jsonl').write_bytes(b''.join(qualification.canonical_json(event)+b'\n' for event in retained))
    assessed = qualification.shared.assess_case(case,retained)
    assessed.update(exitcode=0,evidence_digest=previous)
    summary = {'suite_digest':suite['suite_digest'],'cases':[assessed],
               'unrun_cases':[item['id'] for item in suite['cases'][1:]],
               'unrun_reason':'explicit test only','all_synthetic_cases_pass':False}
    (output/'summary.json').write_text(json.dumps(summary))
    result = qualification.replay_suite(path,snapshot='mock-only',output=output)
    assert result['passed'] and result['cases'][0]['quality_passed']
    assert not result['qualification_passed']
    summary['cases'][0]['id'] = 'another-case'
    (output/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='case identity'):
        qualification.replay_suite(path,snapshot='mock-only',output=output)
    summary['cases'][0]['id'] = case['id']
    (output/'summary.json').write_text(json.dumps(summary))
    changed = deepcopy(suite)
    changed['cases'][0]['task']['prompt'] += ' changed'
    changed['suite_digest'] = qualification.shared._digest({key:value for key,value in changed.items() if key != 'suite_digest'})
    path.write_text(json.dumps(changed))
    summary['suite_digest'] = changed['suite_digest']
    (output/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='start differs: prompt'):
        qualification.replay_suite(path,snapshot='mock-only',output=output)
    path.write_text(json.dumps(suite))
    summary['suite_digest'] = suite['suite_digest']
    summary['cases'].extend([assessed]*4)
    (output/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Extra undeclared'):
        qualification.replay_suite(path,snapshot='mock-only',output=output)
