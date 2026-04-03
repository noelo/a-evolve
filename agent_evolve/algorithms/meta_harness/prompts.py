"""Proposer prompts for MetaHarness.

The key design principle (from the Meta-Harness paper, Appendix D):
  "The skill should constrain outputs and safety-relevant behavior,
   not the proposer's diagnosis procedure: it should specify what is
   forbidden, what artifacts to produce, and what objectives to optimize,
   while leaving the model free to inspect scores, traces, and prior
   code as needed."

The proposer receives a minimal prompt pointing to the growing
filesystem archive — it decides what to read, not the prompt.
"""

from __future__ import annotations

from ...contract.workspace import AgentWorkspace

# Minimal system prompt — role + constraints only.
# Diagnosis strategy is left entirely to the proposer.
# Matches the paper's Appendix D: "the skill should constrain outputs
# and safety-relevant behavior, not the proposer's diagnosis procedure."
PROPOSER_SYSTEM_PROMPT = """\
You are a harness optimizer.  Your job is to improve an AI agent's \
performance on a benchmark by writing new versions of its workspace \
files (prompts, skills, tools, and optional scaffolding code).

You have bash access to the workspace directory.  Use grep, cat, ls, \
and any standard CLI tools to inspect files.

The directory `evolution/candidates/` is your primary knowledge source.  \
It contains an archive of EVERY prior candidate — each with its source \
code snapshot, evaluation scores, and full execution traces.  Browse \
this archive to understand what has been tried, what worked, and what \
failed.

IMPORTANT: You are not limited to incremental edits of the current \
workspace.  You may completely rewrite any file from scratch.  You may \
base your changes on ANY prior candidate in the archive — copy its \
code from `evolution/candidates/<name>/snapshot/` and improve on it.  \
The current workspace files are just one starting point; the archive \
contains all alternatives.
"""


def build_proposer_prompt(
    workspace: AgentWorkspace,
    cycle: int,
    score_curve: list[float],
    *,
    harness_enabled: bool = False,
    candidate_index: int | None = None,
    num_candidates: int | None = None,
    num_archived: int = 0,
) -> str:
    """Build the user-message prompt for one Meta-Harness evolution step.

    Intentionally minimal.  Tells the proposer:
      1. The optimization objective + score history
      2. Directory layout (workspace + candidate archive)
      3. What it CAN modify
      4. What it MUST NOT do

    Everything else — what to read, how to diagnose, what to change —
    is left to the proposer.  The proposer browses the filesystem
    directly rather than receiving compressed summaries.
    """
    skills = workspace.list_skills()
    skill_names = [s.name for s in skills]

    # Score history
    if score_curve:
        scores_str = " -> ".join(f"{s:.3f}" for s in score_curve)
        latest = score_curve[-1]
    else:
        scores_str = "(no prior cycles)"
        latest = 0.0

    # Harness section
    harness_section = ""
    if harness_enabled:
        harness_path = workspace.root / "harness.py"
        harness_exists = harness_path.exists()
        harness_section = f"""
### Harness Code
- `harness.py` — agent scaffolding logic (prompt assembly, tool orchestration, etc.)
- Status: {"exists (" + str(harness_path.stat().st_size) + " bytes)" if harness_exists else "does not exist yet — you may create it"}
- The agent dynamically loads this file at runtime.  Changes take effect on next solve cycle.
- You may add functions, modify control flow, change prompt construction logic.
"""

    # Multi-candidate hint
    candidate_section = ""
    if candidate_index is not None and num_candidates is not None:
        candidate_section = f"""
### Candidate Selection
You are generating candidate {candidate_index + 1} of {num_candidates}.
Each candidate is evaluated independently; the best-scoring one is kept.
Be creative and try a distinct approach from what has been tried before.
"""

    # Archive stats
    archive_section = ""
    if num_archived > 0:
        archive_section = f"""
### Candidate Archive ({num_archived} prior candidates)
`evolution/candidates/` is your primary knowledge source.  Each subdirectory contains:
- `snapshot/` — the COMPLETE workspace files (prompts, skills, harness.py, etc.) at time of proposal
- `scores.json` — evaluation score, cost, and metadata (`"selected": true` = was deployed, `"pareto_optimal": true` = on efficiency frontier)
- `traces/` — symlink to the observation batch with full execution traces

Browse freely:
  ls evolution/candidates/
  cat evolution/candidates/*/scores.json | jq '.score'
  grep -r "pattern" evolution/candidates/*/snapshot/
  diff evolution/candidates/cycle_001_cand_0/snapshot/harness.py evolution/candidates/cycle_002_cand_0/snapshot/harness.py
  cat evolution/candidates/<name>/traces/*.jsonl | jq .

You are NOT limited to modifying the current workspace incrementally.
You may copy code from any prior candidate's snapshot and build on it:
  cp evolution/candidates/<best_candidate>/snapshot/harness.py ./harness.py
The archive contains all alternatives — use the best one as your starting point.
"""
    else:
        archive_section = """
### Candidate Archive
This is the first cycle — no prior candidates yet.
Browse `evolution/observations/` for execution traces from the current agent.
"""

    return f"""\
## Meta-Harness Evolution — Cycle {cycle}

### Objective
Improve the agent's benchmark score.  Current: {latest:.3f}.
Score history: {scores_str}
{candidate_section}\
{archive_section}\
### Workspace Layout
```
{workspace.root}/
├── prompts/system.md        — agent system prompt
├── skills/*/SKILL.md        — on-demand skill library ({len(skill_names)} skills)
├── memory/*.jsonl           — episodic memory
├── tools/                   — tool implementations
├── evolution/
│   ├── observations/        — batch_XXXX.jsonl with FULL execution traces
│   │   Each JSONL record: task_id, task_input, success, score,
│   │   feedback_detail, conversation (every message + tool call)
│   └── candidates/          — archive of all prior candidate harnesses
│       └── cycle_NNN_cand_M/
│           ├── snapshot/    — workspace files at time of proposal
│           ├── scores.json  — evaluation results
│           └── traces/      — symlink to observation batch
```
{harness_section}
### What You CAN Modify
- prompts/system.md
- skills/ (create, update, delete SKILL.md files)
- memory/*.jsonl (add insights, prune noise)
- tools/ (modify tool implementations)
{("- harness.py (scaffolding code)" if harness_enabled else "")}

### What You MUST NOT Do
- Do not modify anything under evolution/ (read-only archive).
- Do not hardcode task-specific answers or task IDs into any file.

### How to Work
1. **Diagnose**: Browse `evolution/candidates/` — compare high-scoring vs \
low-scoring candidates' code.  Read execution traces to understand failures.  \
You do NOT need to read everything; use `grep`, `jq`, `diff` selectively.
2. **Hypothesize**: Form specific hypotheses about what causes failures \
and what patterns lead to higher scores.
3. **Propose**: Write new workspace files.  You may:
   - Make incremental edits to the current workspace, OR
   - Copy a high-scoring prior candidate's code and improve on it, OR
   - Write completely new files from scratch.
   Choose whichever approach your diagnosis suggests will work best.

### Current Skills
{chr(10).join(f"- {s}" for s in skill_names) if skill_names else "None yet."}

When done, summarize what you changed and why in 2-3 sentences.
"""
