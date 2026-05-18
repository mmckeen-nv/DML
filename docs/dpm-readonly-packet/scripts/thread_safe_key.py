#!/usr/bin/env python3
from __future__ import annotations


def make_thread_safe_key(thread_key: str | None) -> str:
    raw = (thread_key or '').replace('/', '_').replace(':', '_')
    return ''.join(ch for ch in raw if ch.isalnum() or ch in '_.-')


if __name__ == '__main__':
    import sys

    print(make_thread_safe_key(sys.argv[1] if len(sys.argv) > 1 else None))
