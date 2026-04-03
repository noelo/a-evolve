#!/usr/bin/env bash
# MetaHarness experiment on MCP-Atlas benchmark.
#
# Usage:
#   bash examples/mcp_examples/run_metaharness.sh <run-name> [extra-args...]
#
# Examples:
#   bash examples/mcp_examples/run_metaharness.sh full_run --workers 5
#   bash examples/mcp_examples/run_metaharness.sh smoke --eval-sample-size 2 --max-cycles 1 --task-limit 5
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run-name> [extra-args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="examples/configs/metaharness_mcp.yaml"
LOG_DIR="logs/mcp_mh_${RUN_NAME}"
mkdir -p "$LOG_DIR"

# Prerequisites check
echo "=== Prerequisites ==="
command -v claude >/dev/null 2>&1 || { echo "ERROR: claude CLI not found"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found"; exit 1; }
echo "claude: $(claude --version 2>&1 | head -1)"
echo "docker: $(docker --version)"
echo "uv: $(uv --version)"

# Check API keys
echo ""
echo "=== API Key Status ==="
if [ -n "${EVAL_LLM_API_KEY:-}" ] || [ -n "${LLM_API_KEY:-}" ]; then
    echo "Eval judge API key: SET"
else
    echo "WARNING: No EVAL_LLM_API_KEY or LLM_API_KEY set."
    echo "  LLM-as-judge evaluation will fail without a key."
    echo "  Set one with: export EVAL_LLM_API_KEY=<your-gemini-key>"
fi

# Pull Docker image
echo ""
echo "=== Pulling MCP-Atlas Docker image ==="
docker pull ghcr.io/scaleapi/mcp-atlas:latest 2>&1 | tail -3

# Run experiment
echo ""
echo "=== Running MetaHarness MCP Experiment: $RUN_NAME ==="
echo "Config: $CONFIG"
echo "Log: $LOG_DIR/experiment.log"
echo ""

uv run python examples/mcp_examples/run_metaharness.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    "$@" \
    2>&1 | tee "$LOG_DIR/experiment.log"

echo ""
echo "=== Done ==="
echo "Log: $LOG_DIR/experiment.log"
echo "Workspace: evolution_workdir/mcp_mh_${RUN_NAME}/"
