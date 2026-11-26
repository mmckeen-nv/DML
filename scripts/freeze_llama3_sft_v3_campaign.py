#!/usr/bin/env python3
"""Freeze the qualified SFT candidate and its reviewed owned-container deployment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from durable_campaign import digest, publish
from freeze_remote_campaign import declared_limits, producer_command


def bound_json(reference, files):
    if (type(reference) is not dict or set(reference) != {'path', 'sha256'}
            or not Path(reference['path']).is_absolute()
            or files.get(reference['path']) != reference['sha256']
            or digest(Path(reference['path'])) != reference['sha256']):
        raise ValueError('Evidence role is not bound to reviewed bytes')
    return json.loads(Path(reference['path']).read_bytes())


def readiness_attestation(path, expected_sha256, *, commit, profile, identity, snapshots):
    path = Path(path).resolve()
    if digest(path) != expected_sha256:
        raise ValueError('Readiness hash differs')
    receipt = json.loads(path.read_bytes())
    expected = {'schema_version':'dml-llama3-sft-v3-readiness-v1', 'approved':True,
                'source_commit':commit, 'consumer_profile':profile, 'model_identity':identity,
                'snapshot_sha256':snapshots}
    if any(receipt.get(key) != value for key,value in expected.items()):
        raise ValueError('Readiness does not approve this exact candidate')
    files = receipt.get('evidence_files')
    if not isinstance(files,dict) or not files:
        raise ValueError('Missing complete reviewed evidence inventory')
    for name, value in files.items():
        if not Path(name).is_absolute() or digest(Path(name)) != value:
            raise ValueError('Readiness evidence changed')
    for stage,count in (('planning',2),('readiness',4)):
        qualified = bound_json(receipt.get(stage), files)
        if any(qualified.get(key) != value for key,value in {
                'passed':True, 'source_commit':commit,'consumer_profile':profile,
                'model_identity':identity,'cases_declared':count,'cases_passed':count}.items()):
            raise ValueError('Qualification does not match candidate and original gate')
        roles = qualified.get('evidence_roles',{})
        if set(roles) != {'suite','result','primary_replay','independent_replay'}:
            raise ValueError('Qualification requires both independent evidence replays')
        if len({item.get('path') for item in roles.values()}) != 4:
            raise ValueError('Qualification evidence roles must be distinct')
        documents = {role:bound_json(ref,files) for role,ref in roles.items()}
        suite_ref,result_ref = roles['suite'],roles['result']
        if (documents['suite'].get('stage') != stage or documents['suite'].get('consumer_profile') != profile
                or documents['suite'].get('model_identity') != identity
                or len(documents['suite'].get('cases',[])) != count
                or documents['result'].get('all_synthetic_cases_pass') is not True):
            raise ValueError('Bound qualification suite/result differs')
        for role in ('primary_replay','independent_replay'):
            replay = documents[role]
            if (replay.get('passed') is not True or replay.get('suite_sha256') != suite_ref['sha256']
                    or replay.get('result_sha256') != result_ref['sha256']
                    or replay.get('model_identity') != identity):
                raise ValueError('Replay does not bind exact qualification outcome')
    ci = bound_json(receipt.get('source_ci'),files)
    if ci.get('source_commit') != commit or ci.get('passed') is not True:
        raise ValueError('Exact-source CI is not qualified')
    ci_run = bound_json(ci.get('run_evidence'), files)
    jobs = ci_run.get('jobs')
    if (ci_run.get('headSha') != commit or ci_run.get('conclusion') != 'success'
            or not isinstance(jobs,list) or not jobs
            or any(job.get('conclusion') != 'success' for job in jobs)
            or not any(job.get('name') == 'llama3-sft-cpu' for job in jobs)
            or sorted(job.get('name') for job in jobs) != sorted(ci.get('expected_jobs',[]))):
        raise ValueError('Exact-head CI job evidence is incomplete')
    for role in ('gpu_admission','lifecycle'):
        evidence = bound_json(receipt.get(role),files)
        if evidence.get('passed') is not True:
            raise ValueError('Runtime or lifecycle admission failed')
        if role == 'gpu_admission' and (evidence.get('model_identity') != identity
                                        or evidence.get('source_commit') != commit):
            raise ValueError('GPU admission belongs to a different candidate')
    lifecycle = bound_json(receipt['lifecycle'], files)
    if (lifecycle.get('source_commit') != commit
            or lifecycle.get('launcher_sha256') != receipt.get('launcher',{}).get('sha256')
            or lifecycle.get('container_config_sha256') != receipt.get('container_config',{}).get('sha256')):
        raise ValueError('Lifecycle qualification does not bind the deployment')
    bound_json(receipt['container_config'],files)
    launcher = receipt.get('launcher',{})
    if set(launcher) != {'path','sha256'} or files.get(launcher['path']) != launcher['sha256']:
        raise ValueError('Launcher is not reviewed')
    return receipt, {**files,str(path):expected_sha256}


def validate_deployment(config, *, expected_command, image, reviewed):
    """The independently reviewed config is immutable; no host model invocation."""
    if (config.get('command') != expected_command or config.get('image') != image
            or not isinstance(image,str) or not image.startswith('sha256:')
            or config != reviewed):
        raise ValueError('Container command/image/config differs from reviewed deployment')


def validate_source_inventory(sources):
    required = {'scripts/freeze_llama3_sft_v3_campaign.py','scripts/durable_campaign.py',
                'dml_core/scripts/llama3_sft_v3_synthetic.py','dml_core/scripts/agent_campaign_evidence.py',
                'dml_core/daystrom_dml/services/llama3_sft_action_input.py',
                'dml_core/daystrom_dml/services/llama3_sft_v3_action_input.py',
                'dml_core/daystrom_dml/services/llama3_sft_runtime.py'}
    if not required <= sources.keys():
        raise ValueError('Qualification sources are not tracked')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-root','source-root','readiness-attestation','container-config','launcher'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--readiness-attestation-sha256',required=True)
    parser.add_argument('--container-python',default='python3')
    parser.add_argument('--host-python',type=Path,default=Path('/usr/bin/python3'))
    args = parser.parse_args(argv)
    root, source = args.candidate_root.resolve(),args.source_root.resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
        raise ValueError('Freeze requires clean exact source')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    sys.path.insert(0,str(source/'dml_core'))
    from daystrom_dml.contracts.agent_episode import LLAMA3_SFT_V3_CONSUMER_PROFILE as profile, execution_protocol_for_profile
    from daystrom_dml.services.agent_episode import EpisodeLimits
    from daystrom_dml.services.llama3_sft_v3_action_input import verify_manifest, identity_for_manifest
    from daystrom_dml.services.episode_verifiers import load_episode_corpus
    from scripts.agent_campaign_evidence import SPEC_VERSION_V2, GATES, _digest
    from scripts.agent_episodes import _source_digests
    bundle,run = root/'snapshot',root/'campaign-once'
    _, manifest, _, _, _ = verify_manifest(bundle)
    identity = identity_for_manifest(manifest).to_payload()
    snapshots = {name:digest(bundle/name) for name in (*manifest['files'],'llama3-sft-v3-manifest.json')}
    reviewed,files = readiness_attestation(args.readiness_attestation,args.readiness_attestation_sha256,
        commit=commit,profile=profile,identity=identity,snapshots=snapshots)
    ci = bound_json(reviewed['source_ci'],files)
    if ci.get('workflow_sha256') != digest(source/'.github/workflows/ci.yml'):
        raise ValueError('Reviewed CI workflow differs from frozen source')
    config_path,launcher = args.container_config.resolve(),args.launcher.resolve()
    config = json.loads(config_path.read_bytes())
    limits = asdict(EpisodeLimits(**declared_limits()))
    command = producer_command(args.container_python,profile,bundle,run,limits)
    reviewed_config = bound_json(reviewed.get('container_config'),files)
    if files.get(str(launcher)) != digest(launcher):
        raise ValueError('Owned-container launcher is not reviewed')
    validate_deployment(config,expected_command=command,image=manifest['runtime']['image'],reviewed=reviewed_config)
    if reviewed['container_config']['path'] != str(config_path):
        raise ValueError('Config path differs')
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=source).decode().split('\0')
    sources = {name:digest(source/name) for name in tracked if name}
    validate_source_inventory(sources)
    corpus = load_episode_corpus()
    spec = dict(schema_version=SPEC_VERSION_V2,consumer_profile=profile,
        execution_protocol=execution_protocol_for_profile(profile),acceptance=GATES,
        selection=[dict(scenario_id=scenario['id'],task_id=task['id']) for scenario in corpus['scenarios'] for task in scenario['tasks']],
        corpus_digest=_digest(corpus),limits=limits,source_sha256=sources,
        producer_source_sha256=_source_digests(consumer_profile=profile),snapshot_sha256=snapshots,
        model_identity=identity,source_commit=commit,source_ci_qualified=True,production_ready=False,
        frozen_at=time.time(),readiness_attestation={'path':str(args.readiness_attestation.resolve()),
                                                   'sha256':args.readiness_attestation_sha256})
    run.mkdir(mode=0o700, exist_ok=True)
    if any(path.name != 'container-config.json' for path in run.iterdir()):
        raise ValueError('Campaign run directory must be unstarted')
    if config_path != run/'container-config.json':
        publish(run/'container-config.json', config)
        files[str(run/'container-config.json')] = digest(run/'container-config.json')
    publish(run/'spec.json',spec)
    files.update({str(source/name):value for name,value in sources.items()})
    files.update({str(bundle/name):value for name,value in snapshots.items()})
    files[str(config_path)] = digest(config_path)
    files[str(launcher)] = digest(launcher)
    files[str(run/'spec.json')] = digest(run/'spec.json')
    host_python = args.host_python.resolve()
    if files.get(str(host_python)) != digest(host_python):
        raise ValueError('Host launcher interpreter is not hash-bound reviewed evidence')
    freeze = {'files':files,'command':[str(args.host_python.resolve()),str(launcher),'run',str(run)],
              'cwd':str(source),'max_seconds':3300,'environment':{},
              'container_config':str(config_path),'producer_command':command,
              'runtime_scope':'Reviewed container image/runtime; host launcher is orchestration only'}
    publish(run/'freeze.json',freeze)
    print(json.dumps({'run':str(run),'freeze_sha256':digest(run/'freeze.json'),'spec_sha256':digest(run/'spec.json')}))


if __name__ == '__main__':
    main()
