import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def run(cmd, env=None):
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=True)


@pytest.mark.parametrize(
    ('thread_key', 'thread_id', 'thread_safe_key', 'chat_id', 'topic_id', 'thread_label'),
    [
        (
            'discord:chan/alpha',
            'thread:test-contract',
            'discord_chan_alpha',
            'chan-alpha',
            'topic-42',
            'Alpha contract thread',
        ),
        (
            'self-architecture-portrait',
            'thread:self-architecture-portrait',
            'self-architecture-portrait',
            '1488668819655495782',
            'topic-self-2',
            'continuity substrate drill',
        ),
    ],
)
def test_writer_finder_registry_and_recall_align_on_canonical_checkpoint_contract(
    tmp_path, thread_key, thread_id, thread_safe_key, chat_id, topic_id, thread_label
):
    out_dir = tmp_path / 'out' / 'dml-checkpoints'
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = tmp_path / 'runtime' / 'continuity' / 'thread_registry.json'
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    expected_path = out_dir / f'{thread_safe_key}.md'

    if expected_path.exists():
        expected_path.unlink()

    original_registry = registry_path.read_text() if registry_path.exists() else None
    try:
        env = os.environ.copy()
        env['DML_CHECKPOINT_DIR'] = str(out_dir)
        env['DML_THREAD_REGISTRY'] = str(registry_path)
        env['DML_RUNTIME_ROOT'] = str(registry_path.parent)
        env['THREAD_ID'] = thread_id
        env['THREAD_TOPIC'] = 'Contract smoke topic'
        env['PROVIDER'] = 'discord'
        env['CHANNEL'] = 'discord'
        env['CHAT_ID'] = chat_id
        env['TOPIC_ID'] = topic_id
        env['THREAD_LABEL'] = thread_label
        env['SESSION_SCOPE'] = 'thread'

        writer = run([
            'bash', str(SCRIPTS / 'dml_checkpoint_thread.sh'), thread_key, 'Contract smoke summary'
        ], env=env)
        assert writer.stdout.strip() == str(expected_path)
        assert expected_path.exists()

        body = expected_path.read_text()
        assert f'- thread_key: {thread_key}' in body
        assert f'- thread_safe_key: {thread_safe_key}' in body
        assert f'- checkpoint_path: {expected_path}' in body
        assert '- provider: discord' in body
        assert '- channel: discord' in body
        assert f'- chat_id: {chat_id}' in body
        assert f'- topic_id: {topic_id}' in body
        assert f'- thread_label: {thread_label}' in body
        assert '- session_scope: thread' in body

        finder = run(['bash', str(SCRIPTS / 'find_thread_checkpoint.sh'), thread_key], env=env)
        assert finder.stdout.strip() == str(expected_path)

        empty_out_dir = tmp_path / 'empty-checkpoints'
        missing_dir = subprocess.run(
            ['bash', str(SCRIPTS / 'find_thread_checkpoint.sh'), thread_key, str(empty_out_dir)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        assert missing_dir.returncode == 1
        assert missing_dir.stdout == ''
        assert f'checkpoint directory not found: {empty_out_dir}' in missing_dir.stderr

        registry = json.loads(registry_path.read_text())
        entry = registry['threads'][thread_id]
        assert entry['thread_key'] == thread_key
        assert entry['thread_safe_key'] == thread_safe_key
        assert entry['latest_checkpoint'] == str(expected_path)
        assert entry['provider'] == 'discord'
        assert entry['channel'] == 'discord'
        assert entry['chat_id'] == chat_id
        assert entry['topic_id'] == topic_id
        assert entry['thread_label'] == thread_label
        assert entry['session_scope'] == 'thread'

        recall = run(['python3', str(SCRIPTS / 'continuity_recall.py'), '--json', thread_key], env=env)
        payload = json.loads(recall.stdout)
        assert payload['retrieval_status'] == 'hit'
        assert payload['thread']['latest_checkpoint'] == str(expected_path)
        assert payload['thread']['registry_latest_checkpoint'] == str(expected_path)
        assert payload['thread']['provider'] == 'discord'
        assert payload['thread']['channel'] == 'discord'
        assert payload['thread']['chat_id'] == chat_id
        assert payload['thread']['topic_id'] == topic_id
        assert payload['thread']['thread_label'] == thread_label
        assert payload['thread']['session_scope'] == 'thread'

        expected_path.unlink()
        recall_after_delete = run(['python3', str(SCRIPTS / 'continuity_recall.py'), '--json', thread_key], env=env)
        payload_after_delete = json.loads(recall_after_delete.stdout)
        assert payload_after_delete['retrieval_status'] == 'registry_only'
        assert payload_after_delete['thread']['registry_latest_checkpoint'] == str(expected_path)
        assert payload_after_delete['thread']['id'] == thread_id
    finally:
        if expected_path.exists():
            expected_path.unlink()
        if original_registry is None:
            if registry_path.exists():
                registry_path.unlink()
        else:
            registry_path.write_text(original_registry)
