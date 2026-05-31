#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

WS = Path('/Users/markmckeen/.openclaw/workspace')
DML_CLI = WS / 'skills' / 'daystrom-dml' / 'scripts' / 'dml_memory.py'
STORE = WS / 'data' / 'dml-gpu-prod'
OUT = Path('/opt/homebrew/lib/node_modules/openclaw/dist/control-ui/assets/dml-savings.json')


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def gather_files() -> list[Path]:
    pats = [
        'skills/**/*.md',
        'skills/**/*.json',
        'agentic-framework/**/*.md',
        'agentic-framework/**/*.py',
        'out/*.log',
        '*.log',
    ]
    files: list[Path] = []
    for pat in pats:
        files.extend([p for p in WS.glob(pat) if p.is_file()])
    uniq = []
    seen = set()
    for p in sorted(files):
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def run_query(query: str) -> tuple[int, float, float]:
    cmd = [
        'python3', str(DML_CLI), '--storage-dir', str(STORE), 'retrieve',
        '--query', query, '--top-k', '6', '--tenant-id', 'openclaw', '--no-with-ground-truth'
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-400:])
    obj = json.loads(proc.stdout)
    return int(obj.get('context_tokens', 0)), ms, float(obj.get('memory_confidence', 0.0) or 0.0)


def main() -> int:
    files = gather_files()
    baseline = 0
    for p in files:
        try:
            baseline += estimate_tokens(p.read_text(errors='ignore')[:50000])
        except Exception:
            pass

    queries = [
        'Summarize current project decisions and blockers.',
        'What constraints and implementation guardrails are active?',
        'What recent changes impact execution speed or token usage?',
    ]
    toks, lats, confs = [], [], []
    for q in queries:
        t, ms, c = run_query(q)
        toks.append(t); lats.append(ms); confs.append(c)

    avg_toks = sum(toks) / max(1, len(toks))
    savings = (1 - (avg_toks / max(1, baseline))) * 100.0
    payload = {
        'updatedAt': time.strftime('%Y-%m-%d %H:%M:%S'),
        'avgTokenSavingsPct': round(savings, 2),
        'avgLatencyMs': round(sum(lats) / len(lats), 2),
        'avgMemoryConfidence': round(sum(confs) / len(confs), 3),
        'baselineTokensEstimate': baseline,
        'avgContextTokens': round(avg_toks, 2),
        'sampleQueries': queries,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
