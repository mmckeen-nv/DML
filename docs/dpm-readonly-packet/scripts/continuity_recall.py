#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def make_thread_safe_key(thread_key: str | None) -> str:
    raw = (thread_key or '').replace('/', '_').replace(':', '_')
    return ''.join(ch for ch in raw if ch.isalnum() or ch in '_.-')

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('DML_RUNTIME_ROOT', WORKSPACE / 'runtime' / 'continuity'))
THREADS = Path(os.environ.get('DML_THREAD_REGISTRY', ROOT / 'thread_registry.json'))
LOOPS = Path(os.environ.get('DML_OPEN_LOOPS', ROOT / 'open_loops.json'))
SELF = Path(os.environ.get('DML_SELF_STATE', ROOT / 'self_state.json'))
WORK_LOOP_HANDOFF = Path(os.environ.get('DML_WORK_LOOP_HANDOFF', WORKSPACE / 'out' / 'work-loop-handoff.md'))
CHECKPOINT_DIR = Path(os.environ.get('DML_CHECKPOINT_DIR', WORKSPACE / 'out' / 'dml-checkpoints'))
DPM_CONFIG = Path(os.environ.get('DML_CONFIG_PATH', WORKSPACE / 'examples' / 'dpm' / 'config.disabled.json'))
PROJECT_STATE = Path(os.environ.get('DML_PROJECT_STATE', ROOT / 'project_state.json'))


def load(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def parse_args(argv: list[str]) -> tuple[bool, str | None, str | None]:
    json_mode = False
    args = []
    for arg in argv:
        if arg == '--json':
            json_mode = True
        else:
            args.append(arg)
    thread_key = args[0] if len(args) > 0 else None
    thread_id = args[1] if len(args) > 1 else None
    return json_mode, thread_key, thread_id


def normalize(value: str | None) -> str:
    return (value or '').strip().lower()


def canonical_checkpoint_path(thread_key: str | None) -> Path | None:
    safe_key = make_thread_safe_key(thread_key)
    if not safe_key:
        return None
    return CHECKPOINT_DIR / f'{safe_key}.md'


def resolve_checkpoint_path(meta: dict[str, Any] | None) -> str | None:
    meta = meta or {}
    latest = meta.get('latest_checkpoint')
    if latest:
        path = Path(latest)
        if path.exists() and path.is_file():
            return str(path)

    thread_key = meta.get('thread_key') or meta.get('key')
    canonical = canonical_checkpoint_path(thread_key)
    if canonical and canonical.exists() and canonical.is_file():
        return str(canonical)

    safe_key = meta.get('thread_safe_key') or make_thread_safe_key(thread_key)
    if safe_key and CHECKPOINT_DIR.exists():
        matches = sorted(CHECKPOINT_DIR.glob(f'*_{safe_key}.md'))
        if matches:
            return str(matches[-1])
    return latest


def read_text(path_str: str | None, max_chars: int = 1200) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(errors='replace').strip()
    return text[:max_chars]


def extract_section_line(text: str | None, marker: str) -> str | None:
    if not text or marker not in text:
        return None
    after = text.split(marker, 1)[1].strip()
    if not after:
        return None
    return after.splitlines()[0].strip() or None


def extract_next_action_from_handoff(text: str | None) -> str | None:
    return extract_section_line(text, '## Next Action')


def extract_task_from_handoff(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith('- task:'):
            return stripped.split(':', 1)[1].strip() or None
    return None


def stable_thread_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    thread_key = meta.get('thread_key') or meta.get('key')
    topic = meta.get('topic')
    return {
        'provider': meta.get('provider'),
        'channel': meta.get('channel'),
        'chat_id': meta.get('chat_id'),
        'topic_id': meta.get('topic_id'),
        'thread_label': meta.get('thread_label') or topic or thread_key,
        'thread_key': thread_key,
        'session_scope': meta.get('session_scope') or ('thread' if thread_key else None),
    }



def match_thread(threads: dict[str, dict[str, Any]], thread_key: str | None, thread_id: str | None):
    query = normalize(thread_key)
    query_safe = make_thread_safe_key(thread_key).lower()
    if thread_id and thread_id in threads:
        return thread_id, threads[thread_id]
    if not query and not query_safe:
        return None, None

    matches: list[tuple[str, dict[str, Any]]] = []
    for tid, meta in threads.items():
        key = normalize(meta.get('thread_key') or meta.get('key'))
        topic = normalize(meta.get('topic'))
        safe_key = normalize(meta.get('thread_safe_key'))
        if query and (query == key or query == topic):
            matches.append((tid, meta))
            continue
        if query_safe and query_safe == safe_key:
            matches.append((tid, meta))

    if len(matches) == 1:
        return matches[0]
    return None, None


def filter_loops(loops: list[dict[str, Any]], matched_meta: dict[str, Any] | None, query: str | None):
    matched_key = normalize((matched_meta or {}).get('thread_key') or (matched_meta or {}).get('key'))
    matched_topic = normalize((matched_meta or {}).get('topic'))
    wanted = []
    q = normalize(query)
    for loop in loops:
        loop_thread = normalize(loop.get('thread'))
        title = normalize(loop.get('title'))
        goal = normalize(loop.get('goal'))
        loop_id = normalize(loop.get('id'))
        if matched_key and loop_thread == matched_key:
            wanted.append(loop)
            continue
        if q and (q == loop_thread or q == loop_id or q in title or q in goal):
            wanted.append(loop)
            continue
        if matched_topic and matched_topic in title:
            wanted.append(loop)
    return wanted


def handoff_matches_query(
    handoff_task: str | None,
    matched_meta: dict[str, Any] | None,
    related_loops: list[dict[str, Any]],
    query: str | None,
) -> bool:
    task = normalize(handoff_task)
    if not task:
        return False
    q = normalize(query)
    if q and (q == task or q in task):
        return True

    matched_key = normalize((matched_meta or {}).get('thread_key') or (matched_meta or {}).get('key'))
    matched_topic = normalize((matched_meta or {}).get('topic'))
    if matched_key and task == matched_key:
        return True
    if matched_topic and matched_topic in task:
        return True

    for loop in related_loops:
        if task in {
            normalize(loop.get('id')),
            normalize(loop.get('thread')),
            normalize(loop.get('title')),
        }:
            return True
    return False


def load_dpm_config() -> dict[str, Any]:
    config = load(DPM_CONFIG, {})
    return config if isinstance(config, dict) else {}


def load_project_state() -> dict[str, Any]:
    state = load(PROJECT_STATE, {})
    return state if isinstance(state, dict) else {}


def _thread_runtime_source(matched: dict[str, Any], checkpoint_text: str, allow_thread: bool) -> dict[str, Any] | None:
    thread_label = matched.get('thread_label') or matched.get('topic') or matched.get('thread_key') or matched.get('key')
    thread_id = matched.get('thread_key') or matched.get('key')
    if not (allow_thread and thread_label and thread_id):
        return None
    return {
        'source_id': thread_id,
        'scope': 'thread',
        'kind': 'thread_summary',
        'label': thread_label,
        'content': checkpoint_text,
        'priority': 1,
        'confidence': 1.0,
        'updated_at': matched.get('updated_at') or matched.get('last_summary_at'),
        'summary': f'Checkpoint-backed continuity for {thread_label}.',
    }


def _project_runtime_source(
    config: dict[str, Any],
    matched: dict[str, Any],
    project_state: dict[str, Any] | None,
    allow_project: bool,
) -> dict[str, Any] | None:
    state = project_state if isinstance(project_state, dict) else {}
    project_id = state.get('project_id') or config.get('project_id') or 'project:dpm'
    project_summary = state.get('summary')
    project_updated_at = state.get('updated_at')
    compatible_threads = state.get('compatible_thread_keys', [])
    if not isinstance(compatible_threads, list):
        compatible_threads = []

    thread_id = matched.get('thread_key') or matched.get('key')
    thread_safe = {normalize(thread_id), normalize(matched.get('thread_key')), normalize(matched.get('key'))}
    is_project_compatible = any(normalize(item) in thread_safe for item in compatible_threads if isinstance(item, str))
    if not (
        allow_project
        and isinstance(project_summary, str)
        and project_summary.strip()
        and isinstance(project_updated_at, str)
        and project_updated_at
        and is_project_compatible
    ):
        return None

    thread_label = matched.get('thread_label') or matched.get('topic') or matched.get('thread_key') or matched.get('key')
    project_label = state.get('label') or project_id
    return {
        'source_id': project_id,
        'scope': 'project',
        'kind': 'project_summary',
        'label': project_label,
        'content': project_summary.strip(),
        'priority': 2,
        'confidence': 0.8,
        'updated_at': project_updated_at,
        'summary': f'Project continuity compatible with {thread_label}.',
    }


def collect_runtime_read_sources(
    config: dict[str, Any],
    matched: dict[str, Any] | None,
    checkpoint_text: str | None,
    project_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if config.get('mode') != 'active-read':
        return []
    if not matched or not checkpoint_text:
        return []

    read = config.get('read', {})
    if not isinstance(read, dict):
        return []

    sources: list[dict[str, Any]] = []
    thread_source = _thread_runtime_source(matched, checkpoint_text, read.get('allow_thread', False))
    if thread_source:
        sources.append(thread_source)

    project_source = _project_runtime_source(config, matched, project_state, read.get('allow_project', False))
    if project_source:
        sources.append(project_source)

    return sources


def _render_overlay_text(sources: list[dict[str, Any]], max_chars: int) -> str:
    overlay_parts = []
    for source in sources:
        prefix = 'Thread continuity' if source['scope'] == 'thread' else 'Project continuity'
        overlay_parts.append(f"{prefix} for {source['label']}: {source['content']}")
    return ' '.join(overlay_parts)[:max_chars]


def _build_overlay_audit_notes(retrieval_order_applied: list[str]) -> list[str]:
    notes = ['Active-read overlay emitted via continuity recall seam.']
    if retrieval_order_applied == ['thread']:
        notes.append('Thread-only overlay emitted because no compatible project source was available.')
    else:
        notes.append('Project source included only after thread match and compatibility check passed.')
    return notes


def _shape_overlay_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'source_id': source['source_id'],
            'scope': source['scope'],
            'kind': source['kind'],
            'included': True,
            'priority': source['priority'],
            'confidence': source['confidence'],
            'updated_at': source['updated_at'],
            'summary': source['summary'],
        }
        for source in sources
    ]


def build_dpm_overlay(
    config: dict[str, Any],
    matched: dict[str, Any] | None,
    checkpoint_text: str | None,
    project_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    mode = config.get('mode')
    if mode != 'active-read':
        return None

    sources = collect_runtime_read_sources(config, matched, checkpoint_text, project_state)
    if not sources:
        return None

    audit = config.get('audit', {})
    max_chars = audit.get('max_overlay_chars', 0)
    if not isinstance(max_chars, int) or max_chars < 1:
        return None

    thread_id = matched.get('thread_key') or matched.get('key')
    if not isinstance(thread_id, str) or not thread_id:
        return None

    generated_candidates = [source.get('updated_at') for source in sources if isinstance(source.get('updated_at'), str) and source.get('updated_at')]
    generated_at = max(generated_candidates) if generated_candidates else (matched.get('updated_at') or matched.get('last_summary_at'))
    if not isinstance(generated_at, str) or not generated_at:
        return None

    project_id = config.get('project_id') or (project_state or {}).get('project_id') or 'project:dpm'
    relationship_id = config.get('relationship_id') or 'relationship:runtime'
    retrieval_order_applied = [source['scope'] for source in sources]
    rendered_text = _render_overlay_text(sources, max_chars)

    persona_summary = 'Provide concise continuity guidance without exposing hidden provenance.'
    style_directives = [
        'Prefer thread-specific context when it directly matches the current recall target.',
        'Allow compatible project context to refine, but never override, thread-local guidance.',
        'Keep continuity guidance compact and runtime-facing.',
    ]

    return {
        'schema_version': 'dpm.replay-overlay.v1',
        'overlay_id': f"overlay:thread:{thread_id}:{mode}",
        'mode': mode,
        'generated_at': generated_at,
        'scope': {
            'primary': 'thread',
            'thread_id': thread_id,
            'project_id': project_id,
            'relationship_id': relationship_id,
        },
        'retrieval_order_applied': retrieval_order_applied,
        'overlay': {
            'persona_summary': persona_summary,
            'style_directives': style_directives,
            'do_not_do': [
                'Do not expose broader-scope memory or write-oriented behavior in active-read mode.',
                'Do not let project context override explicit current-turn instructions or thread-local constraints.',
            ],
            'open_questions': [],
            'max_chars': max_chars,
            'rendered_text': rendered_text,
        },
        'effective_constraints': {
            'explicit_instruction_precedence': 'always_override',
            'narrowest_scope_wins': True,
            'cross_scope_fallback_requires_compatibility': True,
            'writes_allowed': False,
        },
        'sources': _shape_overlay_sources(sources),
        'audit': {
            'included_source_ids': [source['source_id'] for source in sources],
            'excluded_sources': [],
            'conflicts_detected': [],
            'notes': _build_overlay_audit_notes(retrieval_order_applied),
        },
        'override_state': {
            'has_explicit_instruction': False,
            'override_applied': False,
            'instruction_source_id': None,
            'suppressed_source_ids': [],
            'effective_for_turn': [source['source_id'] for source in sources],
        },
    }


def build_payload(thread_key: str | None, thread_id: str | None) -> dict[str, Any]:
    threads = load(THREADS, {}).get('threads', {})
    loops = load(LOOPS, {}).get('loops', [])
    self_state = load(SELF, {})
    matched_id, matched = match_thread(threads, thread_key, thread_id)
    resolved_checkpoint = resolve_checkpoint_path(matched)
    checkpoint_text = read_text(resolved_checkpoint)
    handoff_text = read_text(str(WORK_LOOP_HANDOFF) if WORK_LOOP_HANDOFF.exists() else None)
    related_loops = filter_loops(loops, matched, thread_key)
    handoff_task = extract_task_from_handoff(handoff_text)
    handoff_relevant = handoff_matches_query(handoff_task, matched, related_loops, thread_key)
    dpm_overlay = build_dpm_overlay(load_dpm_config(), matched, checkpoint_text, load_project_state())

    retrieval_status = 'miss'
    if matched and checkpoint_text:
        retrieval_status = 'hit'
    elif matched:
        retrieval_status = 'registry_only'

    stable = stable_thread_metadata(matched)

    return {
        'retrieval_status': retrieval_status,
        'thread': {
            'query': thread_key,
            'id': matched_id,
            'key': (matched or {}).get('key'),
            'thread_key': (matched or {}).get('thread_key') or (matched or {}).get('key'),
            'thread_safe_key': (matched or {}).get('thread_safe_key') or make_thread_safe_key((matched or {}).get('thread_key') or (matched or {}).get('key')),
            'topic': (matched or {}).get('topic'),
            'status': (matched or {}).get('status'),
            'latest_checkpoint': resolved_checkpoint,
            'registry_latest_checkpoint': (matched or {}).get('latest_checkpoint'),
            'last_summary_at': (matched or {}).get('last_summary_at'),
            'updated_at': (matched or {}).get('updated_at'),
            'provider': stable['provider'],
            'channel': stable['channel'],
            'chat_id': stable['chat_id'],
            'topic_id': stable['topic_id'],
            'thread_label': stable['thread_label'],
            'session_scope': stable['session_scope'],
        },
        'checkpoint_excerpt': checkpoint_text,
        'self_state': {
            'continuity_mode': self_state.get('continuity_mode'),
            'priorities': self_state.get('priorities', []),
            'known_weak_points': self_state.get('known_weak_points', []),
            'updated_at': self_state.get('updated_at'),
        },
        'open_loops': related_loops,
        'handoff': {
            'path': str(WORK_LOOP_HANDOFF) if WORK_LOOP_HANDOFF.exists() and handoff_relevant else None,
            'task': handoff_task,
            'is_relevant': handoff_relevant,
            'next_action': extract_next_action_from_handoff(handoff_text) if handoff_relevant else None,
            'excerpt': handoff_text if handoff_relevant else None,
        },
        'dpm_overlay': dpm_overlay,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    thread = payload['thread']
    handoff = payload['handoff']
    print('# Continuity Recall')
    print()
    print(f"- retrieval_status: {payload['retrieval_status']}")
    if thread.get('id'):
        print(f"- thread_id: {thread['id']}")
    if thread.get('thread_key'):
        print(f"- thread_key: {thread['thread_key']}")
    if thread.get('topic'):
        print(f"- topic: {thread['topic']}")
    if thread.get('latest_checkpoint'):
        print(f"- latest_checkpoint: {thread['latest_checkpoint']}")
    if thread.get('last_summary_at'):
        print(f"- last_summary_at: {thread['last_summary_at']}")
    if not thread.get('id') and not thread.get('thread_key'):
        print('- thread: no registry match')

    print()
    print('## Checkpoint Excerpt')
    print(payload['checkpoint_excerpt'] or 'No checkpoint content available.')

    overlay = payload.get('dpm_overlay')
    if overlay:
        print()
        print('## DPM Overlay')
        print(overlay.get('text') or 'No overlay content available.')

    print()
    print('## Self State')
    for p in payload['self_state'].get('priorities', []):
        print(f'- {p}')
    weak_points = payload['self_state'].get('known_weak_points', [])
    if weak_points:
        print()
        print('## Known Weak Points')
        for item in weak_points:
            print(f'- {item}')

    print()
    print('## Open Loops')
    loops = payload['open_loops']
    if not loops:
        print('- none')
    for loop in loops:
        print(f"- {loop.get('title')}: {loop.get('next_action')}")

    print()
    print('## Handoff')
    if handoff.get('task'):
        print(f"- task: {handoff['task']}")
    print(f"- relevant: {'yes' if handoff.get('is_relevant') else 'no'}")
    if handoff.get('next_action'):
        print(f"- next_action: {handoff['next_action']}")
    else:
        print('- next_action: none')
    if handoff.get('path'):
        print(f"- path: {handoff['path']}")


def main() -> int:
    json_mode, thread_key, thread_id = parse_args(sys.argv[1:])
    payload = build_payload(thread_key, thread_id)
    if json_mode:
        print(json.dumps(payload, indent=2))
    else:
        print_markdown(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
