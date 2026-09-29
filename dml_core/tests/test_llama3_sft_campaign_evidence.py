from types import SimpleNamespace
import pytest
from daystrom_dml.contracts.model_input import ModelInputRequest
from scripts.agent_campaign_evidence import _replay_llama3_sft_model


class Consumer:
    def compile(self, messages, tools, *, output_reserved_tokens):
        return SimpleNamespace(input_ids=(1,2),attention_mask=(1,1),output_reserved_tokens=output_reserved_tokens,model_window_tokens=8192)
    def release_compiled(self, artifact):
        self.released = True
    def validate_output_tokens(self, request, ids):
        if ids != [3,4]:
            raise ValueError('grammar mask violation')
    def decode_output(self, ids):
        return '{'


def events():
    request = {'messages':[{'role':'user','content':'test'}], 'tools':[], 'output_reserved_tokens':256}
    compiled = {'identity':{},'request_digest':ModelInputRequest.from_payload(request).request_digest,
                'input_ids':[1,2],'attention_mask':[1,1], 'output_reserved_tokens':256,'model_window_tokens':8192}
    return [{'kind':'model_requested','payload':{'request':request,'compiled':compiled,'artifact_digest':'a'*64}},
            {'kind':'model_completed','payload':{'output_ids':[3,4],'text':'{'}}]


def test_retains_grammar_valid_incomplete_failure_without_repair():
    consumer = Consumer()
    assert _replay_llama3_sft_model(events(),{},consumer) == 1
    assert consumer.released


@pytest.mark.parametrize('field,value',[('input_ids',[9]),('model_window_tokens',10000),('request_digest','0'*64)])
def test_rejects_compile_tampering(field,value):
    trace = events()
    trace[0]['payload']['compiled'][field] = value
    with pytest.raises(ValueError):
        _replay_llama3_sft_model(trace,{},Consumer())


def test_rejects_output_mask_tampering():
    trace = events()
    trace[1]['payload']['output_ids'] = [9]
    with pytest.raises(ValueError):
        _replay_llama3_sft_model(trace,{},Consumer())


def test_interrupted_null_prefix_remains_failed_evidence():
    class Failed(Consumer):
        def validate_failed_execution(self, artifact, request, evidence, *, recorded_digest):
            assert recorded_digest == 'a'*64
            assert evidence['runtime_result'] is None
            self.failure_seen = True
    trace = events()
    trace[1] = {'kind':'model_failed','payload':{'phase':'execute',
        'local_execution':{'artifact_digest':'a'*64,'exception_type':'Interrupted',
                           'exception_message':'stopped','runtime_result':None}}}
    consumer = Failed()
    assert _replay_llama3_sft_model(trace,{},consumer) == 0
    assert consumer.failure_seen
