"""MetaHarnessEngine -- evolution via Claude Code as proposer.

Implements the Meta-Harness search framework (Lee et al., 2026):
  - Proposer is Claude Code CLI with Opus 4.6 via Bedrock
  - Growing filesystem archive stores every candidate's source code,
    evaluation scores, and execution traces
  - The proposer browses this archive with grep/cat/ls (~10M tokens)
    rather than receiving compressed summaries in the prompt
  - k candidates per iteration with Pareto-aware selection
  - Interface validation before expensive evaluation
  - Automatic rollback when score regresses
  - Candidate archive persists across runs for cross-run transfer
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ...config import EvolveConfig
from ...contract.workspace import AgentWorkspace
from ...engine.base import EvolutionEngine
from ...engine.history import EvolutionHistory
from ...engine.trial import TrialRunner
from ...types import Observation, StepResult
from .prompts import build_proposer_prompt

logger = logging.getLogger(__name__)

# Default model: Opus 4.6 via Claude Code CLI (same as the paper)
# Note: Claude Code CLI uses raw Bedrock model IDs without the "bedrock:" prefix
DEFAULT_MODEL = "us.anthropic.claude-opus-4-6-v1"

# Workspace files to snapshot into each candidate archive
_SNAPSHOT_DIRS = ("prompts", "skills", "memory", "tools", ".claude")
_SNAPSHOT_FILES = ("harness.py", "CLAUDE.md")


class MetaHarnessEngine(EvolutionEngine):
    """Evolution engine that uses Claude Code CLI as the proposer.

    Maintains a growing candidate archive in ``evolution/candidates/``
    that the proposer browses via filesystem access — matching the
    paper's design of full trace + source code access per candidate.

    Each evaluated candidate gets its own directory::

        evolution/candidates/cycle_003_cand_1/
        ├── snapshot/          # workspace files at time of proposal
        │   ├── prompts/
        │   ├── skills/
        │   ├── harness.py
        │   └── ...
        ├── scores.json        # evaluation results {score, cost, selected, valid, ...}
        └── traces/            # symlink or copy of observation batch

    Features matching the paper:
      - Full benchmark evaluation per candidate (eval_sample_size=0 → all tasks)
      - Interface validation before expensive evaluation (Algorithm 1 line 11)
      - Pareto frontier tracking across (score, cost) objectives
      - Candidate archive persists across runs (cross-run knowledge transfer)
      - Initial population evaluation handled by A-Evolve's loop (cycle 0)
    """

    def __init__(self, config: EvolveConfig):
        self.config = config
        self.harness_enabled: bool = config.extra.get("harness_enabled", False)
        self.model: str = config.extra.get("proposer_model", DEFAULT_MODEL)
        self.max_turns: int = config.extra.get("proposer_max_turns", 50)
        self.timeout_sec: int = config.extra.get("proposer_timeout_sec", 600)
        # Multi-candidate: generate k variants per cycle (paper: typically 2)
        self.num_candidates: int = config.extra.get("num_candidates", 2)
        # Evaluation sample size: 0 = all tasks (paper default), >0 = subsample
        self.eval_sample_size: int = config.extra.get("eval_sample_size", 0)
        # Rollback: revert if best candidate scores below current best.
        # Default False to match the paper — Meta-Harness stores all
        # candidates and allows temporary regressions for exploration.
        # The Pareto frontier tracks the best across all cycles.
        self.rollback_on_regression: bool = config.extra.get(
            "rollback_on_regression", False
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(
        self,
        workspace: AgentWorkspace,
        observations: list[Observation],
        history: EvolutionHistory,
        trial: TrialRunner,
        tasks: list | None = None,
        eval_factory: Callable[[Path], TrialRunner] | None = None,
    ) -> StepResult:
        """Run one Meta-Harness evolution step (Algorithm 1 inner loop).

        Phase A — Propose (serial): for each of k candidates, run the
        Claude Code proposer, capture the diff and snapshot, then reset.

        Phase B — Evaluate (parallel when eval_factory is provided):
        create temporary workspace copies, apply each candidate's diff,
        and evaluate all candidates concurrently.

        Phase C — Select: Pareto-aware selection, apply or rollback.

        Args:
            eval_factory: Optional callable ``(workspace_path) -> TrialRunner``.
                When provided and num_candidates > 1, candidates are
                evaluated in parallel using separate workspace copies.
                When None, falls back to serial evaluation on the main
                workspace (original behavior).

        Note: Initial population evaluation (Algorithm 1 lines 3-5) is
        handled by the A-Evolve loop's first SOLVE→OBSERVE cycle before
        engine.step() is called.
        """
        cycle_num = history.latest_cycle + 1
        score_curve = history.get_score_curve()
        current_best = max(score_curve) if score_curve else 0.0

        candidates_dir = workspace.root / "evolution" / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        # Count existing candidates (includes prior runs — cross-run archive)
        existing = len([
            d for d in candidates_dir.iterdir() if d.is_dir()
        ])

        parallel = eval_factory is not None and self.num_candidates > 1

        # ==============================================================
        # Phase A — Propose all candidates (serial)
        # ==============================================================
        proposed: list[dict[str, Any]] = []

        for i in range(self.num_candidates):
            if i > 0:
                self._git_reset(workspace.root)

            cand_label = f"cycle_{cycle_num:03d}_cand_{i}"

            prompt = build_proposer_prompt(
                workspace,
                cycle_num,
                score_curve,
                harness_enabled=self.harness_enabled,
                candidate_index=i,
                num_candidates=self.num_candidates,
                num_archived=existing + len(proposed),
            )

            result = self._run_claude_code(prompt, workspace.root)
            diff = self._git_diff(workspace.root)
            valid, validation_err = self._validate_candidate(workspace)

            # Regex audit for task-specific string leakage (paper §4.3)
            task_ids = [t.id for t in tasks] if tasks else []
            leakage = self._audit_leakage(workspace, task_ids)
            if leakage:
                logger.warning(
                    "Leakage audit for %s: %s", cand_label, "; ".join(leakage),
                )
                valid = False
                validation_err = (
                    (validation_err + "; " if validation_err else "")
                    + "leakage: " + "; ".join(leakage)
                )

            # Snapshot workspace state for archiving (before reset)
            snapshot_files = self._capture_snapshot(workspace)

            proposed.append({
                "index": i,
                "label": cand_label,
                "diff": diff,
                "valid": valid,
                "validation_err": validation_err,
                "proposer_result": result,
                "snapshot_files": snapshot_files,
            })
            logger.info(
                "Proposed %s: valid=%s (%d chars diff)",
                cand_label, valid, len(diff),
            )

        # Reset workspace after all proposals
        self._git_reset(workspace.root)

        # ==============================================================
        # Phase B — Evaluate candidates (parallel or serial)
        # ==============================================================
        candidates: list[dict[str, Any]] = []

        if parallel:
            candidates = self._evaluate_parallel(
                proposed, workspace, candidates_dir, cycle_num,
                eval_factory, tasks,
            )
        else:
            candidates = self._evaluate_serial(
                proposed, workspace, candidates_dir, cycle_num,
                trial, tasks,
            )

        # -- Selection --
        # Filter to valid candidates only
        valid_candidates = [c for c in candidates if c["valid"]]

        if not valid_candidates:
            logger.warning("All %d candidates failed validation", len(candidates))
            self._git_reset(workspace.root)
            return StepResult(
                mutated=False,
                summary=(
                    f"MetaHarness cycle {cycle_num}: "
                    f"all {self.num_candidates} candidates failed validation"
                ),
                metadata={
                    "cycle": cycle_num,
                    "num_candidates": self.num_candidates,
                    "all_invalid": True,
                    "validation_errors": [c["validation_err"] for c in candidates],
                    "proposer_model": self.model,
                },
            )

        # Compute Pareto frontier across (score↑, cost↓)
        frontier = _pareto_frontier(valid_candidates)
        # Select the highest-scoring candidate from the frontier
        best = max(frontier, key=lambda c: c["score"])

        logger.info(
            "Selected %s (score=%.3f, cost=%d) from %d candidates "
            "(%d on Pareto frontier)",
            best["label"], best["score"], best["cost"],
            len(candidates), len(frontier),
        )

        # Reset workspace to clean state
        self._git_reset(workspace.root)

        # Mark selected candidate + Pareto frontier in archive
        for c in candidates:
            scores_path = candidates_dir / c["label"] / "scores.json"
            if scores_path.exists():
                data = json.loads(scores_path.read_text())
                data["selected"] = (c["label"] == best["label"])
                data["pareto_optimal"] = c in frontier
                scores_path.write_text(json.dumps(data, indent=2))

        # Rollback check: if best candidate regresses, don't apply
        if self.rollback_on_regression and best["score"] < current_best:
            logger.info(
                "Best candidate %.3f < current best %.3f — rolling back",
                best["score"], current_best,
            )
            return StepResult(
                mutated=False,
                summary=(
                    f"MetaHarness cycle {cycle_num}: "
                    f"{len(valid_candidates)} valid candidates evaluated, "
                    f"rolled back (best={best['score']:.3f}, "
                    f"current={current_best:.3f})"
                ),
                metadata={
                    "cycle": cycle_num,
                    "rolled_back": True,
                    "num_candidates": self.num_candidates,
                    "num_valid": len(valid_candidates),
                    "candidate_scores": [c["score"] for c in candidates],
                    "pareto_frontier": [c["label"] for c in frontier],
                    "selected": best["label"],
                    "current_best": current_best,
                    "proposer_model": self.model,
                },
            )

        # Apply best candidate's diff to workspace
        if best["diff"]:
            self._apply_diff(workspace.root, best["diff"])

        changes_summary = (
            f"selected {best['label']} "
            f"(score={best['score']:.3f}, cost={best['cost']})"
            if best["diff"]
            else "no mutation"
        )

        return StepResult(
            mutated=bool(best["diff"]),
            summary=f"MetaHarness cycle {cycle_num}: {changes_summary}",
            metadata={
                "cycle": cycle_num,
                "num_candidates": self.num_candidates,
                "num_valid": len(valid_candidates),
                "selected": best["label"],
                "candidate_scores": [c["score"] for c in candidates],
                "candidate_costs": [c["cost"] for c in candidates],
                "pareto_frontier": [c["label"] for c in frontier],
                "best_score": best["score"],
                "best_cost": best["cost"],
                "harness_enabled": self.harness_enabled,
                "proposer_model": self.model,
                "total_archived": existing + len(candidates),
            },
        )

    # ------------------------------------------------------------------
    # Phase B helpers — serial and parallel evaluation
    # ------------------------------------------------------------------

    def _evaluate_serial(
        self,
        proposed: list[dict[str, Any]],
        workspace: AgentWorkspace,
        candidates_dir: Path,
        cycle_num: int,
        trial: TrialRunner,
        tasks: list | None,
    ) -> list[dict[str, Any]]:
        """Evaluate candidates one at a time on the main workspace (original behavior)."""
        candidates: list[dict[str, Any]] = []

        for p in proposed:
            # Apply this candidate's diff
            if p["diff"]:
                self._apply_diff(workspace.root, p["diff"])

            if p["valid"]:
                eval_result = self._evaluate_candidate(trial, tasks=tasks)
                score = eval_result["score"]
                cost = eval_result["cost"]
            else:
                logger.warning(
                    "Candidate %s failed validation: %s — skipping eval",
                    p["label"], p["validation_err"],
                )
                score = 0.0
                cost = 0.0

            # Archive
            cand_dir = candidates_dir / p["label"]
            self._archive_candidate_from_snapshot(
                workspace, cand_dir, p["snapshot_files"],
                score, cost, cycle_num, p["index"], p["proposer_result"],
                valid=p["valid"], validation_err=p["validation_err"],
            )

            candidates.append({
                "index": p["index"],
                "label": p["label"],
                "score": score,
                "cost": cost,
                "diff": p["diff"],
                "valid": p["valid"],
                "validation_err": p["validation_err"],
                "exit_code": p["proposer_result"].get("exit_code"),
                "output_chars": len(p["proposer_result"].get("output", "")),
            })
            logger.info(
                "Candidate %s: valid=%s, score=%.3f, cost=%d",
                p["label"], p["valid"], score, cost,
            )

            # Reset before next candidate
            self._git_reset(workspace.root)

        return candidates

    def _evaluate_parallel(
        self,
        proposed: list[dict[str, Any]],
        workspace: AgentWorkspace,
        candidates_dir: Path,
        cycle_num: int,
        eval_factory: Callable[[Path], TrialRunner],
        tasks: list | None,
    ) -> list[dict[str, Any]]:
        """Evaluate candidates in parallel using temporary workspace copies."""
        valid_proposals = [p for p in proposed if p["valid"]]
        invalid_proposals = [p for p in proposed if not p["valid"]]

        # Handle invalid candidates immediately
        candidates: list[dict[str, Any]] = []
        for p in invalid_proposals:
            logger.warning(
                "Candidate %s failed validation: %s — skipping eval",
                p["label"], p["validation_err"],
            )
            cand_dir = candidates_dir / p["label"]
            self._archive_candidate_from_snapshot(
                workspace, cand_dir, p["snapshot_files"],
                0.0, 0.0, cycle_num, p["index"], p["proposer_result"],
                valid=False, validation_err=p["validation_err"],
            )
            candidates.append({
                "index": p["index"],
                "label": p["label"],
                "score": 0.0,
                "cost": 0.0,
                "diff": p["diff"],
                "valid": False,
                "validation_err": p["validation_err"],
                "exit_code": p["proposer_result"].get("exit_code"),
                "output_chars": len(p["proposer_result"].get("output", "")),
            })

        if not valid_proposals:
            return candidates

        # Create temp workspace copies and evaluate in parallel
        tmp_dirs: list[Path] = []

        def _eval_one(p: dict[str, Any]) -> dict[str, Any]:
            tmp_root = Path(tempfile.mkdtemp(prefix=f"mh_{p['label']}_"))
            tmp_dirs.append(tmp_root)
            tmp_workspace = tmp_root / "workspace"
            shutil.copytree(workspace.root, tmp_workspace)

            if p["diff"]:
                self._apply_diff(tmp_workspace, p["diff"])

            eval_trial = eval_factory(tmp_workspace)
            eval_result = self._evaluate_candidate(eval_trial, tasks=tasks)

            return {
                "index": p["index"],
                "label": p["label"],
                "score": eval_result["score"],
                "cost": eval_result["cost"],
                "diff": p["diff"],
                "valid": True,
                "validation_err": "",
                "exit_code": p["proposer_result"].get("exit_code"),
                "output_chars": len(p["proposer_result"].get("output", "")),
                "snapshot_files": p["snapshot_files"],
                "proposer_result": p["proposer_result"],
            }

        logger.info(
            "Evaluating %d candidates in parallel",
            len(valid_proposals),
        )

        try:
            with ThreadPoolExecutor(max_workers=len(valid_proposals)) as pool:
                futures = {
                    pool.submit(_eval_one, p): p for p in valid_proposals
                }
                for future in as_completed(futures):
                    p = futures[future]
                    try:
                        result = future.result()
                        # Archive
                        cand_dir = candidates_dir / result["label"]
                        self._archive_candidate_from_snapshot(
                            workspace, cand_dir, result["snapshot_files"],
                            result["score"], result["cost"],
                            cycle_num, result["index"], result["proposer_result"],
                            valid=True, validation_err="",
                        )
                        candidates.append(result)
                        logger.info(
                            "Candidate %s: score=%.3f, cost=%d",
                            result["label"], result["score"], result["cost"],
                        )
                    except Exception as e:
                        logger.error("Parallel eval failed for %s: %s", p["label"], e)
                        candidates.append({
                            "index": p["index"],
                            "label": p["label"],
                            "score": 0.0,
                            "cost": 0.0,
                            "diff": p["diff"],
                            "valid": True,
                            "validation_err": f"eval_error: {e}",
                            "exit_code": p["proposer_result"].get("exit_code"),
                            "output_chars": len(p["proposer_result"].get("output", "")),
                        })
        finally:
            # Cleanup temp directories
            for tmp_dir in tmp_dirs:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

        return candidates

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _capture_snapshot(self, workspace: AgentWorkspace) -> dict[str, bytes]:
        """Capture mutable workspace files as in-memory bytes for later archiving."""
        snapshot: dict[str, bytes] = {}
        for dirname in _SNAPSHOT_DIRS:
            src = workspace.root / dirname
            if src.exists():
                for f in src.rglob("*"):
                    if f.is_file():
                        rel = str(f.relative_to(workspace.root))
                        snapshot[rel] = f.read_bytes()
        for fname in _SNAPSHOT_FILES:
            src = workspace.root / fname
            if src.exists():
                snapshot[fname] = src.read_bytes()
        return snapshot

    def _archive_candidate_from_snapshot(
        self,
        workspace: AgentWorkspace,
        cand_dir: Path,
        snapshot_files: dict[str, bytes],
        score: float,
        cost: int | float,
        cycle: int,
        cand_index: int,
        proposer_result: dict[str, Any],
        *,
        valid: bool = True,
        validation_err: str = "",
    ) -> None:
        """Archive a candidate using a pre-captured snapshot."""
        cand_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write snapshot files
        snapshot_dir = cand_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        for rel_path, content in snapshot_files.items():
            dest = snapshot_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        # 2. Write scores + metadata
        scores_data = {
            "cycle": cycle,
            "candidate_index": cand_index,
            "score": score,
            "cost": cost,
            "valid": valid,
            "validation_error": validation_err,
            "selected": False,
            "pareto_optimal": False,
            "proposer_model": self.model,
            "proposer_exit_code": proposer_result.get("exit_code"),
        }
        (cand_dir / "scores.json").write_text(
            json.dumps(scores_data, indent=2)
        )

        # 3. Link traces
        traces_dir = cand_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        obs_dir = workspace.root / "evolution" / "observations"
        if obs_dir.exists():
            batches = sorted(obs_dir.glob("batch_*.jsonl"))
            if batches:
                latest_batch = batches[-1]
                link_target = traces_dir / latest_batch.name
                try:
                    link_target.symlink_to(latest_batch.resolve())
                except OSError:
                    shutil.copy2(latest_batch, link_target)

        logger.debug("Archived candidate to %s", cand_dir)

    # ------------------------------------------------------------------
    # Interface validation (Algorithm 1 line 11)
    # ------------------------------------------------------------------

    def _validate_candidate(
        self, workspace: AgentWorkspace,
    ) -> tuple[bool, str]:
        """Validate candidate modifications before expensive evaluation.

        Checks that modified Python files compile and key files are intact.
        Returns (valid, error_message).
        """
        errors: list[str] = []

        # Validate harness.py if it exists
        harness_path = workspace.root / "harness.py"
        if harness_path.exists():
            try:
                source = harness_path.read_text()
                compile(source, str(harness_path), "exec")
            except SyntaxError as e:
                errors.append(f"harness.py: {e}")

        # Validate tool Python files
        tools_dir = workspace.root / "tools"
        if tools_dir.exists():
            for py_file in tools_dir.glob("*.py"):
                try:
                    source = py_file.read_text()
                    compile(source, str(py_file), "exec")
                except SyntaxError as e:
                    errors.append(f"{py_file.name}: {e}")

        # Validate system prompt is non-empty
        prompt = workspace.read_prompt()
        if prompt is not None and len(prompt.strip()) == 0:
            errors.append("prompts/system.md is empty")

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def _audit_leakage(
        self,
        workspace: AgentWorkspace,
        task_ids: list[str],
    ) -> list[str]:
        """Regex audit for task-specific string leakage (paper §4.3).

        Scans workspace files (prompts, skills, harness, tools) for
        hardcoded task IDs.  Returns list of warnings (empty = clean).
        """
        if not task_ids:
            return []

        import re

        # Collect text from all mutable workspace files
        texts: list[tuple[str, str]] = []  # (filename, content)
        for d in _SNAPSHOT_DIRS:
            d_path = workspace.root / d
            if d_path.exists():
                for f in d_path.rglob("*"):
                    if f.is_file():
                        try:
                            texts.append((str(f.relative_to(workspace.root)), f.read_text()))
                        except (UnicodeDecodeError, OSError):
                            pass
        for f_name in _SNAPSHOT_FILES:
            f_path = workspace.root / f_name
            if f_path.exists():
                try:
                    texts.append((f_name, f_path.read_text()))
                except (UnicodeDecodeError, OSError):
                    pass

        warnings: list[str] = []
        for task_id in task_ids:
            # Skip very short IDs that would cause false positives
            if len(task_id) < 8:
                continue
            pattern = re.escape(task_id)
            for filename, content in texts:
                if re.search(pattern, content):
                    warnings.append(f"task ID '{task_id}' found in {filename}")
                    break  # one warning per task ID

        return warnings

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self, trial: TrialRunner | None, tasks: list | None = None,
    ) -> dict[str, Any]:
        """Evaluate a candidate on benchmark tasks.

        Returns dict with 'score' and 'cost' (total tokens).
        If tasks are provided, uses them directly. Otherwise falls back
        to loading from trial runner (eval_sample_size controls limit).
        """
        if trial is None:
            return {"score": 0.0, "cost": 0}

        try:
            if tasks is None:
                if self.eval_sample_size > 0:
                    tasks = trial.get_tasks(limit=self.eval_sample_size)
                else:
                    tasks = trial.get_tasks(limit=10000)

            if not tasks:
                return {"score": 0.0, "cost": 0}

            obs = trial.run_tasks(tasks)
            if not obs:
                return {"score": 0.0, "cost": 0}

            score = sum(o.feedback.score for o in obs) / len(obs)

            # Estimate cost from trajectory token usage
            total_tokens = 0
            for o in obs:
                for step in o.trajectory.steps:
                    usage = step.get("usage", {})
                    total_tokens += usage.get("total_tokens", 0)

            return {"score": score, "cost": total_tokens}

        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return {"score": 0.0, "cost": 0}



    # ------------------------------------------------------------------
    # Claude Code CLI invocation
    # ------------------------------------------------------------------

    def _run_claude_code(self, prompt: str, workspace_root: Path) -> dict[str, Any]:
        """Invoke Claude Code CLI as the proposer.

        Runs in non-interactive mode (-p) with:
          - --model: Bedrock Opus 4.6
          - --max-turns: bounded exploration
          - --dangerously-skip-permissions: no interactive approval
          - cwd: workspace root (Claude Code discovers CLAUDE.md as skill)

        The proposer's skill is defined in CLAUDE.md at the workspace root,
        following the paper's approach (Appendix D): Claude Code's native
        skill discovery loads CLAUDE.md automatically, replacing the old
        --bare --system-prompt approach.
        """
        cmd = [
            "claude",
            "-p", prompt,
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--no-session-persistence",
        ]

        logger.info(
            "Running Claude Code proposer (model=%s, max_turns=%d, cwd=%s)",
            self.model, self.max_turns, workspace_root,
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=str(workspace_root),
            )

            output = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode != 0:
                logger.warning(
                    "Claude Code exited with code %d: %s",
                    proc.returncode,
                    stderr[:500],
                )

            # Try to parse JSON output for structured result
            result_text = output
            try:
                parsed = json.loads(output)
                result_text = parsed.get("result", output)
            except (json.JSONDecodeError, TypeError):
                pass

            logger.info(
                "Claude Code finished (exit=%d, output=%d chars)",
                proc.returncode,
                len(output),
            )

            return {
                "output": result_text,
                "stderr": stderr,
                "exit_code": proc.returncode,
            }

        except subprocess.TimeoutExpired:
            logger.error(
                "Claude Code timed out after %ds", self.timeout_sec
            )
            return {
                "output": "",
                "stderr": "TIMEOUT",
                "exit_code": -1,
            }
        except FileNotFoundError:
            logger.error(
                "Claude Code CLI not found. Install it: "
                "https://docs.anthropic.com/en/docs/claude-code"
            )
            return {
                "output": "",
                "stderr": "claude CLI not found",
                "exit_code": -1,
            }

    # ------------------------------------------------------------------
    # Git helpers for multi-candidate workflow
    # ------------------------------------------------------------------

    def _git_reset(self, root: Path) -> None:
        """Reset workspace to last committed state (discard uncommitted changes).

        Preserves evolution/ directory (observations + candidate archive).
        Uses pathspec to exclude evolution/ from checkout so that files like
        history.jsonl and metrics.json are not reverted.
        """
        subprocess.run(
            ["git", "checkout", "--", ":(exclude)evolution"],
            cwd=str(root),
            capture_output=True,
        )
        # Clean untracked files but preserve evolution/
        subprocess.run(
            ["git", "clean", "-fd", "--exclude=evolution/"],
            cwd=str(root),
            capture_output=True,
        )

    def _git_diff(self, root: Path) -> str:
        """Capture current uncommitted changes as a diff.

        Excludes evolution/ so only workspace mutations are captured.
        """
        # Stage everything so diff captures new files too
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(root),
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--", ".", ":(exclude)evolution/"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        # Unstage
        subprocess.run(
            ["git", "reset", "HEAD", "--quiet"],
            cwd=str(root),
            capture_output=True,
        )
        return result.stdout

    def _apply_diff(self, root: Path, diff: str) -> None:
        """Apply a previously captured diff to the workspace."""
        if not diff.strip():
            return
        proc = subprocess.run(
            ["git", "apply", "--allow-empty", "-"],
            input=diff,
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            logger.warning("git apply failed: %s", proc.stderr[:500])
            proc2 = subprocess.run(
                ["git", "apply", "--3way", "-"],
                input=diff,
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            if proc2.returncode != 0:
                logger.error(
                    "git apply --3way also failed: %s", proc2.stderr[:500]
                )


# ----------------------------------------------------------------------
# Pareto frontier computation
# ----------------------------------------------------------------------

def _pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return Pareto-optimal candidates (maximize score, minimize cost).

    A candidate is Pareto-optimal if no other candidate has both a
    higher (or equal) score AND a lower (or equal) cost, with at least
    one strict inequality.
    """
    frontier = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            # 'other' dominates 'c' if:
            #   other.score >= c.score AND other.cost <= c.cost
            #   with at least one strict inequality
            if (other["score"] >= c["score"]
                    and other["cost"] <= c["cost"]
                    and (other["score"] > c["score"]
                         or other["cost"] < c["cost"])):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return frontier
