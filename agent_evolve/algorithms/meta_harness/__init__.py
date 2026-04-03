"""MetaHarness -- evolution via Claude Code as proposer.

Faithful to Meta-Harness (Lee et al., 2026): use Claude Code CLI
(Bedrock Opus 4.6) as the proposer with full filesystem access to
the workspace, including raw execution traces and scores.  The
proposer decides what to inspect and how to mutate — including an
optional harness.py that contains agent scaffolding logic.
"""

from .engine import MetaHarnessEngine

__all__ = ["MetaHarnessEngine"]
