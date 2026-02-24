#!/bin/bash
# Agentic Evaluation Script
# Runs a suite of agentic tasks and prints a scoreboard

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "🦞 Running Agentic Evaluation Suite"
echo "===================================="
echo ""

# Create virtualenv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtualenv..."
    python -m venv venv
    source venv/bin/activate
    pip install -e .
else
    source venv/bin/activate
fi

# Run evaluation
echo "Running evaluation harness..."
python -m daystrom_dml.eval.harness

# Check results
RESULTS_FILE="eval/results/latest.json"
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Results saved to: $RESULTS_FILE"
    python -c "import json; data=json.load(open('$RESULTS_FILE')); print(f'Total tasks: {data[\"total_tasks\"]}, Success: {data[\"successful\"]} / {data[\"total_tasks\"]} ({data[\"successful\"]/data[\"total_tasks\"]*100:.1f}%)')"
else
    echo "Warning: No results file found"
fi

echo ""
echo "✓ Evaluation complete!"