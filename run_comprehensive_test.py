#!/usr/bin/env python3
"""Comprehensive DML test with progress tracking."""

import sys
sys.path.insert(0, '/home/nvidia/.npm-global/lib/node_modules/openclaw/node_modules/daystrom-dml')

from dml_core.daystrom_dml.dml_adapter import DMLAdapter
import time
import shutil
from datetime import datetime

print("=" * 70)
print("COMPREHENSIVE DML VALIDATION TEST")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize DML
print("\n[0%] Initializing DML adapter...")
adapter = DMLAdapter(
    config_overrides={
        "model_name": "gpt2",
        "embedding_model": "all-MiniLM-L6-v2",
        "storage_dir": "./data/dml_test",
        "dml.agentic_mode.enabled": True
    }
)
print("✅ DML initialized")

# Track stats
stats = {
    "memories_ingested": 0,
    "retrievals": 0,
    "total_latency": 0,
    "total_tokens": 0
}

# TEST 1: Error Ingestion (25%)
print("\n[25%] Test 1: Error Ingestion")
print("-" * 70)

errors = [
    ("Container 'app-api' crashed with exit code 1", "container_crash", "execute"),
    ("Missing environment variable 'API_KEY'", "missing_env", "debug"),
    ("Port 8080 already in use", "port_conflict", "execute"),
    ("Database connection timeout", "db_timeout", "execute")
]

for i, (text, error_type, phase) in enumerate(errors, 1):
    adapter.ingest_agentic(
        text=text,
        kind="error",
        meta={"phase": phase, "error_type": error_type}
    )
    stats["memories_ingested"] += 1

print(f"✅ Ingested {len(errors)} error memories")
print(f"📊 Stats: {stats['memories_ingested']} memories, {stats['retrievals']} retrievals, {stats['total_latency']:.2f}s latency, {stats['total_tokens']} tokens")

# TEST 2: Context Retrieval (50%)
print("\n[50%] Test 2: Context Retrieval")
print("-" * 70)

queries = [
    "What errors occurred?",
    "What environment variables are missing?",
    "What caused the port conflict?",
    "Database connection issues"
]

for query in queries:
    start = time.time()
    report = adapter.retrieve_context(query)
    latency = time.time() - start

    stats["retrievals"] += 1
    stats["total_latency"] += latency
    stats["total_tokens"] += len(report.get("raw_context", ""))

    if (stats["retrievals"] // len(queries)) % 2 == 0:  # Every other query
        avg_latency = stats["total_latency"] / stats["retrievals"]
        avg_tokens = stats["total_tokens"] / stats["retrievals"]
        print(f"  Query '{query[:30]}...': {latency:.3f}s, {len(report['raw_context'])} tokens | Avg: {avg_latency:.3f}s, {avg_tokens:.0f} tokens")

print(f"✅ Completed {len(queries)} retrieval queries")
print(f"📊 Stats: {stats['memories_ingested']} memories, {stats['retrievals']} retrievals, {stats['total_latency']:.2f}s latency, {stats['total_tokens']} tokens")

# TEST 3: Long-Horizon Scenario (75%)
print("\n[75%] Test 3: Long-Horizon Scenario")
print("-" * 70)

scenario_steps = [
    ("Plan phase: Deploy microservices to production", "plan", "plan", "deployment"),
    ("Execute phase: Starting deployment process", "execute", "execute", "starting_deployment"),
    ("Execute phase: Container 'app-api' deployed successfully", "action", "execute", "app_api_deployed"),
    ("Execute phase: Container 'app-worker' deployed successfully", "action", "execute", "app_worker_deployed"),
    ("Execute phase: Health check passed for both services", "result", "execute", "health_check_passed"),
    ("Execute phase: Container 'app-api' crashed unexpectedly", "error", "execute", "app_api_crashed"),
    ("Debug phase: Checking logs for app-api", "observation", "debug", "checking_logs"),
    ("Debug phase: Found memory leak in app-api", "result", "debug", "memory_leak_found"),
    ("Debug phase: Applied patch to fix memory leak", "action", "debug", "patch_applied"),
    ("Execute phase: Restarted app-api with patch", "action", "execute", "app_api_restarted"),
    ("Execute phase: Health check passed after patch", "result", "execute", "health_check_passed"),
]

for text, kind, phase, metadata in scenario_steps:
    adapter.ingest_agentic(
        text=text,
        kind=kind,
        meta={"phase": phase, **metadata}
    )
    stats["memories_ingested"] += 1

print(f"✅ Simulated {len(scenario_steps)} steps of long-horizon task")
print(f"📊 Stats: {stats['memories_ingested']} memories, {stats['retrievals']} retrievals, {stats['total_latency']:.2f}s latency, {stats['total_tokens']} tokens")

# TEST 4: Advanced Retrieval (100%)
print("\n[100%] Test 4: Advanced Retrieval")
print("-" * 70)

advanced_queries = [
    "What was the final outcome?",
    "What caused the memory leak?",
    "How did we fix the issues?",
    "What services are deployed?",
    "What's the current status?"
]

for query in advanced_queries:
    start = time.time()
    report = adapter.retrieve_context(query, top_k=8, use_summary=True)
    latency = time.time() - start

    stats["retrievals"] += 1
    stats["total_latency"] += latency
    stats["total_tokens"] += len(report.get("raw_context", ""))

    if query == advanced_queries[-1]:  # Last query
        print(f"  Query '{query}': {latency:.3f}s, {len(report['raw_context'])} tokens")

# Final cleanup
adapter.close()
shutil.rmtree("./data/dml_test")

# Final Stats
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"✅ Total memories ingested: {stats['memories_ingested']}")
print(f"✅ Total retrievals: {stats['retrievals']}")
print(f"✅ Total latency: {stats['total_latency']:.2f}s")
print(f"✅ Average retrieval latency: {stats['total_latency']/stats['retrievals']:.3f}s")
print(f"✅ Total context tokens: {stats['total_tokens']}")
print(f"✅ Average tokens per retrieval: {stats['total_tokens']/stats['retrievals']:.0f}")
print(f"✅ Average tokens per memory: {stats['total_tokens']/stats['memories_ingested']:.0f}")
print(f"\n✅ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)