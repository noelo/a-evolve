#!/usr/bin/env bash
set -euo pipefail

# OSWorld with A-EVOLVE propose+curator algorithm
#
# 369 tasks across 10 domains (chrome, gimp, libreoffice_calc, etc.)
# Each task runs on a dedicated AWS EC2 VM with a GUI desktop.
# Agent uses screenshot + accessibility tree to interact via computer_use tool.
# Evaluation: environment-specific checks return 0.0 or 1.0.
#
# Prerequisites:
#   1. Clone OSWorld: git clone https://github.com/xlang-ai/OSWorld
#   2. Set OSWORLD_PATH to the cloned directory
#   3. Configure AWS credentials and set the env vars below
#   4. Install: pip install -e . (from a-evolve root)
#   5. For Chrome 147+ AMIs, apply the --user-data-dir fix (see README)

# ── AWS Configuration ─────────────────────────────────────────────────
# Default: us-east-1 AMI (stable, Chrome < 147)
export OSWORLD_PATH="${OSWORLD_PATH:-/path/to/OSWorld}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_SUBNET_ID="${AWS_SUBNET_ID:?Set AWS_SUBNET_ID}"
export AWS_SECURITY_GROUP_ID="${AWS_SECURITY_GROUP_ID:?Set AWS_SECURITY_GROUP_ID}"

TASK_FILE="${OSWORLD_PATH}/evaluation_examples/test_all.json"

# ── Baseline: no evolution ────────────────────────────────────────────

# Pure baseline (no skills, no evolution, 30 steps)
# python examples/osworld_examples/evolve_osworld.py \
#     --task-file "$TASK_FILE" \
#     --provider aws \
#     --solver-model 1 \
#     --no-evolve --no-seed-skills \
#     --max-steps 30 \
#     --batch-size 10 --workers 10 \
#     --output-dir outputs/osworld_baseline_pure

# Baseline with seed skills (no evolution, 30 steps)
# python examples/osworld_examples/evolve_osworld.py \
#     --task-file "$TASK_FILE" \
#     --provider aws \
#     --solver-model 1 \
#     --no-evolve \
#     --lazy-load \
#     --max-steps 30 \
#     --batch-size 5 --workers 5 \
#     --output-dir outputs/osworld_baseline_seed

# ── Evolve: propose+curator ──────────────────────────────────────────

# V1: evolve with propose+curator, no seed skills
python examples/osworld_examples/evolve_osworld.py \
    --task-file "$TASK_FILE" \
    --provider aws \
    --no-seed-skills \
    --solver-model 1 \
    --curator-model 1 \
    --selector-model 1 \
    --batch-size 16 \
    --workers 16 \
    --max-skills-per-topic 5 \
    --max-general-skills 5 \
    --shuffle --shuffle-seed 42 \
    --max-steps 100 \
    --output-dir outputs/osworld_evolve_v1

# ── Quick test (subset) ──────────────────────────────────────────────

# Small test (39 tasks, single domain)
# python examples/osworld_examples/evolve_osworld.py \
#     --task-file "${OSWORLD_PATH}/evaluation_examples/test_small.json" \
#     --provider aws \
#     --solver-model 1 \
#     --no-evolve --no-seed-skills \
#     --max-steps 30 \
#     --batch-size 5 --workers 5 \
#     --output-dir outputs/osworld_test_small
