#!/usr/bin/env python3
"""Simple test for DML."""

import sys
sys.path.insert(0, '/home/nvidia/.npm-global/lib/node_modules/openclaw/node_modules/daystrom-dml')

from dml_core.daystrom_dml.dml_adapter import DMLAdapter
from dml_core.daystrom_dml.agent_schema import MemoryKind
import shutil

print("=" * 60)
print("DML SIMPLE TEST")
print("=" * 60)

# Initialize
print("\n1. Initializing...")
adapter = DMLAdapter(
    config_overrides={
        "model_name": "gpt2",
        "embedding_model": "all-MiniLM-L6-v2",
        "storage_dir": "./data/simple_test"
    }
)
print("✅ OK")

# Ingest
print("\n2. Ingesting memory...")
adapter.ingest_agentic(
    text="Deployed app successfully",
    kind="action",
    meta={"phase": "execute"}
)
print("✅ OK")

# Retrieve
print("\n3. Retrieving...")
report = adapter.retrieve_context("deployment")
print("Context:")
print(report.get("raw_context", ""))

# Stats
print(f"\n4. Memories: {adapter.memory_count()}")

# Cleanup
adapter.close()
shutil.rmtree("./data/simple_test")

print("\n✅ TEST COMPLETE")