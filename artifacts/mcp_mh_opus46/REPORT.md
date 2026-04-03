# Meta-Harness on MCP-Atlas: Experiment Report

## Overview

This report presents results of applying the Meta-Harness algorithm
(Lee et al., 2026, [arXiv:2603.28052](https://arxiv.org/pdf/2603.28052)) to the MCP-Atlas benchmark, using
the A-Evolve framework. Meta-Harness evolves an AI agent's harness
(system prompt, skills, scaffolding code) over multiple cycles, with
Claude Code CLI as the proposer.

## Setup

| Parameter | Value |
|-----------|-------|
| Benchmark | MCP-Atlas (40 tasks, no API key required) |
| Solver | Claude Opus 4.6 (`us.anthropic.claude-opus-4-6-v1`) |
| Proposer | Claude Opus 4.6 (via Claude Code CLI) |
| Judge | Claude Opus 4.6 |
| Evolution cycles | 10 |
| Candidates per cycle | 2 |
| Eval workers | 20 |
| Final eval trials | 5 |
| Pass threshold | coverage_score >= 0.75 |

## Evolution Search Results

Score history across 10 cycles (avg score on 40 tasks):

| Cycle | Best Candidate | Avg Score |
|-------|---------------|-----------|
| 0 (Baseline) | -- | 0.814 |
| 1 | cand_0 | 0.860 |
| 2 | cand_1 | 0.872 |
| 3 | cand_1 | 0.874 |
| 4 | cand_0 | 0.849 |
| 5 | cand_0 | 0.871 |
| 6 | cand_1 | **0.894** |
| 7 | cand_1 | 0.883 |
| 8 | cand_1 | 0.864 |
| 9 | cand_1 | 0.869 |
| 10 | cand_0 | **0.896** |

Best candidate selected for final eval: `cycle_010_cand_0`
(score=0.896, cost=3,421,744 tokens) via Pareto frontier selection.

Total evolution wall time: 3.5 hours.

## Final Evaluation (5 trials x 40 tasks)

### Pass Rate

| Trial | Baseline | Evolved |
|-------|----------|---------|
| 1 | 70.0% (28/40) | 70.0% (28/40) |
| 2 | 67.5% (27/40) | 70.0% (28/40) |
| 3 | 62.5% (25/40) | 77.5% (31/40) |
| 4 | 70.0% (28/40) | 72.5% (29/40) |
| 5 | 75.0% (30/40) | 77.5% (31/40) |
| **Mean +/- Std** | **69.0% +/- 4.5%** | **73.5% +/- 3.8%** |

### Avg Score

| | Baseline | Evolved |
|---|---------|---------|
| **Mean +/- Std** | **0.789 +/- 0.007** | **0.834 +/- 0.010** |

### Summary

| Metric | Baseline | Evolved | Delta |
|--------|----------|---------|-------|
| Pass Rate (5-trial mean) | 69.0% | 73.5% | **+4.5%** |
| Avg Score (5-trial mean) | 0.789 | 0.834 | **+0.045** |
| Std Dev (pass rate) | 4.5% | 3.8% | -0.7% |
| Avg solve time per trial | 727s | 257s | **2.8x faster** |

## Note on Overfitting Prevention

As [Lee et al. (2026, §4.3)](https://arxiv.org/pdf/2603.28052) note,
overfitting is an inherent challenge in harness evolution: *"We additionally
check for overfitting by manual inspection and regex-based audits for
task-specific string leakage into evolved harnesses."*

Following the paper's recommendation, our engine includes a **regex-based
audit** (`_audit_leakage` in `engine.py`) that scans evolved workspace files
for hardcoded task IDs before accepting a candidate. Upon manual inspection
of the evolved artifact, we observed that the proposer occasionally embedded
task-derived factual hints (e.g., a specific date) into the harness routing
logic in `harness.py`. While the majority of the evolved harness consists
of genuinely general-purpose improvements (tool routing, budget management,
chain-task planning), these few instances highlight that **regex-based audits
alone are not sufficient to catch all forms of semantic leakage** — manual
review remains an important complementary step, consistent with the paper's
methodology. Automated semantic leakage detection (e.g., LLM-based auditing)
is a promising direction for future work.

## Artifact

The evolved harness is saved in this directory:

```
artifacts/mcp_mh_opus46/
├── CLAUDE.md                              # Claude Code entry point
├── .claude/skills/harness-optimizer/
│   └── SKILL.md                           # Agent skill
├── harness.py                             # Evolved scaffolding code
├── prompts/system.md                      # Evolved system prompt
├── memory/memories.jsonl                  # Agent memory
├── tools/registry.yaml                    # Tool configuration
├── results.json                           # Machine-readable results
├── final_eval_5trials.json                # Evolved 5-trial raw data
├── baseline_eval_5trials.json             # Baseline 5-trial raw data
└── REPORT.md                              # This report
```

## Task IDs (40 tasks, no API key required)

```
6863f438a500a4b36aab0f24  686bdf0ca7f3bf48518a1738  686d2c79e1db37ea23ddd9fa
686d2c79e1db37ea23ddda26  688706e2ca6b330849710721  688ba1b3e95696e72dd93e8a
688ba1b3e95696e72dd93e8e  688ba1b3e95696e72dd93eaf  688fb11183792b921381bd14
6890e7fd2516f66b0a628e68  6896416f7b30e5d8ccd7c8b4  6896416f7b30e5d8ccd7c8bf
6896416f7b30e5d8ccd7c8c6  6896416f7b30e5d8ccd7c909  6897a6002b64a8c831efa2fb
6897a6002b64a8c831efa30c  6897a6002b64a8c831efa317  6897a6002b64a8c831efa32f
68993ef3cf3e953b8ab83fa9  68993ef3cf3e953b8ab83fbb  68993ef3cf3e953b8ab83fc4
68993ef3cf3e953b8ab83fd3  68993ef3cf3e953b8ab83fd8  68993ef3cf3e953b8ab83fdf
68993ef3cf3e953b8ab83fe3  689cd6f8522029b7ad7b2017  689cd6f8522029b7ad7b2026
689e0b1d9c8e2ac413c1f202  689e0b1d9c8e2ac413c1f23b  689e0b1d9c8e2ac413c1f25c
689f4d693e212e8ef339071f  689f4d693e212e8ef3390740  68a398aa2f58036e8d45edd4
68a398aa2f58036e8d45eddc  68a398aa2f58036e8d45eddf  68a398aa2f58036e8d45ede0
68a398aa2f58036e8d45ede3  68a398aa2f58036e8d45ee01  68a398aa2f58036e8d45ee09
68a398aa2f58036e8d45ee0c
```

## Reproducing

```bash
# Run evolved harness (5 trials)
uv run python examples/mcp_examples/run_final_eval.py \
    --config examples/configs/metaharness_mcp.yaml \
    --workspace artifacts/mcp_mh_opus46 \
    --trials 5 --workers 20

# Run baseline (5 trials)
uv run python examples/mcp_examples/run_final_eval.py \
    --config examples/configs/metaharness_mcp.yaml \
    --workspace seed_workspaces/mcp_mh \
    --trials 5 --workers 20
```
