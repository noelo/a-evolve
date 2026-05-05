#!/usr/bin/env python3
"""Evolve an agent on OSWorld using the A-EVOLVE propose+curator loop.

Pipeline per batch:
  1. Parallel solve (ReAct agent with screenshot+a11y_tree on OSWorld VM)
  2. Parallel evaluate (env.evaluate() returns 0.0 or 1.0)
  3. Analyze feedback + propose skills (in solver conversation context)
  4. Curator per topic (self-assigned domain skills)
  5. General curator (cross-task patterns)
  6. Reload workspace, next batch

Usage:
    python examples/evolve_osworld.py \
        --task-file evaluation_examples/test_all.json \
        --domain libreoffice_calc \
        --batch-size 5 --workers 2 \
        --output-dir outputs/osworld_evolve_v1
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
import queue as queue_mod
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# OSWorld repo must be on path for desktop_env imports
OSWORLD_PATH = os.environ.get("OSWORLD_PATH")
if not OSWORLD_PATH:
    raise EnvironmentError("OSWORLD_PATH must be set to the OSWorld repo directory")
sys.path.insert(0, OSWORLD_PATH)
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# Generate a unique run ID so we can tag and find our instances
_RUN_ID = f"evolve-{os.getpid()}-{int(time.time())}"
os.environ["OSWORLD_RUN_ID"] = _RUN_ID

from agent_evolve.agents.osworld.react_solver import (
    react_solve, extract_conversation, SYSTEM_PROMPT as OSW_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Instance cleanup: track all live OSWorld VMs and terminate on exit
# ---------------------------------------------------------------------------
_live_envs: list = []        # list of DesktopEnv objects still open
_live_envs_lock = threading.Lock()
_cleanup_done = False


def _cleanup_all_envs():
    """Terminate every tracked DesktopEnv. Safe to call multiple times."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    with _live_envs_lock:
        envs = list(_live_envs)
        _live_envs.clear()
    if not envs:
        return
    logger.info("Cleanup: terminating %d OSWorld VM(s)...", len(envs))
    for env in envs:
        try:
            env.close()
        except Exception as e:
            logger.warning("Cleanup: env.close() failed: %s", e)
    logger.info("Cleanup: done.")


def _signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM: clean up VMs then exit."""
    sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
    logger.warning("Received %s — cleaning up OSWorld VMs before exit...", sig_name)
    _cleanup_all_envs()
    # Also try tag-based cleanup as a safety net
    try:
        _cleanup_tagged_instances()
    except Exception:
        pass
    sys.exit(1)


def _cleanup_tagged_instances(region: str = "us-west-2"):
    """Terminate any running instances tagged with our run ID (safety net)."""
    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_instances(Filters=[
            {"Name": "tag:osworld-managed", "Values": ["true"]},
            {"Name": "tag:osworld-run-id", "Values": [_RUN_ID]},
            {"Name": "instance-state-name", "Values": ["running", "pending"]},
        ])
        ids = [
            inst["InstanceId"]
            for res in resp.get("Reservations", [])
            for inst in res.get("Instances", [])
        ]
        if ids:
            ec2.terminate_instances(InstanceIds=ids)
            logger.info("Tag-based cleanup: terminated %d instance(s) for run %s", len(ids), _RUN_ID)
    except Exception as e:
        logger.warning("Tag-based cleanup failed: %s", e)


def cleanup_orphan_instances(region: str = "us-west-2", max_age_hours: int = 6):
    """Terminate osworld-managed instances older than max_age_hours (pre-run sweep)."""
    try:
        import boto3
        from datetime import timezone, timedelta
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_instances(Filters=[
            {"Name": "tag:osworld-managed", "Values": ["true"]},
            {"Name": "instance-state-name", "Values": ["running", "pending"]},
        ])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        stale_ids = []
        for res in resp.get("Reservations", []):
            for inst in res.get("Instances", []):
                launch = inst.get("LaunchTime")
                if launch and launch < cutoff:
                    stale_ids.append(inst["InstanceId"])
        if stale_ids:
            ec2.terminate_instances(InstanceIds=stale_ids)
            logger.warning("Pre-run cleanup: terminated %d orphan OSWorld instance(s) older than %dh",
                           len(stale_ids), max_age_hours)
        else:
            logger.info("Pre-run cleanup: no orphan OSWorld instances found.")
    except Exception as e:
        logger.warning("Pre-run orphan cleanup failed: %s", e)


# Register cleanup handlers
atexit.register(_cleanup_all_envs)
signal.signal(signal.SIGTERM, _signal_handler)
# SIGINT: only set in main thread (threading may call this at import time)
try:
    signal.signal(signal.SIGINT, _signal_handler)
except ValueError:
    pass  # not main thread

# ---------------------------------------------------------------------------
# Seed skills (from seed workspace)
# ---------------------------------------------------------------------------

SEED_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "seed_workspaces" / "osworld"


def _init_seed_skills(workspace_dir: Path):
    """Copy seed skills from repo into workspace/skills/seed/."""
    seed_skills_src = SEED_WORKSPACE / "skills"
    seed_skills_dst = workspace_dir / "skills" / "seed"
    if seed_skills_dst.exists():
        return  # already initialized
    if not seed_skills_src.exists():
        logger.warning("Seed skills not found at %s", seed_skills_src)
        return
    shutil.copytree(seed_skills_src, seed_skills_dst)
    logger.info("Copied %d seed skills from %s",
                len(list(seed_skills_dst.rglob("SKILL.md"))), seed_skills_src)


# ---------------------------------------------------------------------------
# Model map
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "1": "us.anthropic.claude-opus-4-6-v1",
    "2": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "3": "us.anthropic.claude-opus-4-5-20251101-v1:0",
}

# ---------------------------------------------------------------------------
# Bedrock helpers
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_client(region: str = "us-west-2"):
    """Get or create a thread-local Bedrock runtime client."""
    key = f"bedrock_{region}"
    if not hasattr(_thread_local, key):
        import boto3
        from botocore.config import Config as BotoConfig
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotoConfig(read_timeout=300, retries={"max_attempts": 0}),
        )
        setattr(_thread_local, key, client)
    return getattr(_thread_local, key)


def _init_worker(region: str):
    _get_client(region)


def _call_bedrock(client, model_id, system_prompt, user_message,
                  max_tokens=4096, temperature=0.0):
    """Simple Bedrock converse call. Returns (text, error)."""
    for attempt in range(5):
        try:
            resp = client.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            content = resp.get("output", {}).get("message", {}).get("content", [])
            text = "".join(b.get("text", "") for b in content)
            return text.strip(), None
        except Exception as e:
            err = str(e)
            base = 30 if "too many tokens" in err.lower() else (
                4 if "throttl" in err.lower() else 2
            )
            delay = base * (2 ** attempt)
            if attempt < 4:
                time.sleep(delay)
            else:
                return None, err
    return None, "exhausted retries"


def _truncate(s: str, n: int = 300) -> str:
    return s[:n] + "..." if len(s) > n else s


# ---------------------------------------------------------------------------
# Task loading (OSWorld JSON format)
# ---------------------------------------------------------------------------

def load_osworld_tasks(task_file: str, domain: str = None) -> list[dict]:
    """Load OSWorld task configs from a JSON file.

    OSWorld test_all.json is structured as {domain: [task_id, ...]}. Each task's
    full config lives at examples/{domain}/{task_id}.json relative to the
    directory containing the task file.

    Args:
        task_file: Path to test_all.json or similar.
        domain: Optional domain filter (e.g. "os", "chrome", "libreoffice_calc").

    Returns:
        List of task config dicts (each with added "domain" key).
    """
    task_file = Path(task_file)
    base_dir = task_file.parent  # e.g. .../evaluation_examples

    with open(task_file) as f:
        meta = json.load(f)

    # meta is {domain: [task_id_list]}
    if not isinstance(meta, dict):
        # Fallback: already a flat list of configs
        return [t for t in meta if not domain or t.get("domain") == domain]

    tasks = []
    for dom, task_ids in meta.items():
        if domain and dom != domain:
            continue
        for tid in task_ids:
            config_path = base_dir / "examples" / dom / f"{tid}.json"
            if not config_path.exists():
                logger.warning("Task config not found: %s", config_path)
                continue
            with open(config_path) as f:
                config = json.load(f)
            config.setdefault("id", tid)
            config.setdefault("domain", dom)
            tasks.append(config)

    logger.info("Loaded %d tasks from %s (domain=%s)", len(tasks), task_file, domain or "all")
    return tasks


def _task_id(task_config: dict) -> str:
    """Extract a unique task identifier from config."""
    return task_config.get("id", task_config.get("task_id", f"task-{hash(json.dumps(task_config, sort_keys=True)) % 100000}"))


def _task_domain(task_config: dict) -> str:
    """Extract domain from task config."""
    return task_config.get("domain", "unknown")


def _task_topic(task_config: dict) -> str:
    """Derive a kebab-case topic from the task config.

    Uses related_apps or domain to form the topic tag.
    """
    # Prefer related_apps list
    related_apps = task_config.get("related_apps", [])
    if related_apps:
        # Use first related app, kebab-cased
        app = related_apps[0].lower().replace(" ", "-").replace("_", "-")
        return re.sub(r"[^a-z0-9-]", "", app).strip("-") or "desktop"
    # Fall back to domain
    domain = task_config.get("domain", "desktop")
    return re.sub(r"[^a-z0-9-]", "-", domain.lower()).strip("-") or "desktop"


# ---------------------------------------------------------------------------
# Trajectory analysis helpers (adapted for GUI actions)
# ---------------------------------------------------------------------------

def _extract_trajectory_signals(conversation: list[dict]) -> dict:
    """Extract structured behavioral signals from a GUI conversation trajectory."""
    n_turns = 0
    n_actions = 0
    n_clicks = 0
    n_keystrokes = 0
    n_scrolls = 0
    n_errors = 0
    n_timeouts = 0
    actions_run: list[str] = []
    submitted = False
    submit_value = ""
    error_messages: list[str] = []

    for entry in conversation:
        role = entry.get("role", "")
        parts = entry.get("parts", [])

        if role == "assistant":
            n_turns += 1
            for part in parts:
                if part.get("type") == "tool_use":
                    fn = part.get("name", "")
                    inp = part.get("input", {})

                    if fn == "computer":
                        n_actions += 1
                        # Structured action format (computer_use native tool)
                        action_type = inp.get("action", "")
                        coord = inp.get("coordinate")
                        text_val = inp.get("text", "")
                        desc = str(action_type)
                        if coord:
                            desc += f"({coord[0]},{coord[1]})"
                        if text_val and len(str(text_val)) <= 50:
                            desc += f" '{text_val}'"
                        actions_run.append(desc[:120])
                        if action_type in ("left_click", "right_click", "double_click",
                                           "middle_click", "triple_click", "left_press"):
                            n_clicks += 1
                        if action_type in ("type", "key", "hold_key"):
                            n_keystrokes += 1
                        if action_type == "scroll":
                            n_scrolls += 1

                    elif fn in ("submit", "task_submit"):
                        submitted = True
                        submit_value = inp.get("answer", "")

        elif role == "user":
            for part in parts:
                if part.get("type") == "tool_result":
                    content = part.get("text", "")
                    if "Error:" in content or "ERROR:" in content:
                        n_errors += 1
                        error_messages.append(content[:150])
                    if "timed out" in content.lower() or "timeout" in content.lower():
                        n_timeouts += 1

    # Detect repeated actions (same code run 3+ times)
    action_counts: dict[str, int] = {}
    for a in actions_run:
        action_counts[a] = action_counts.get(a, 0) + 1
    repeated_actions = [a for a, cnt in action_counts.items() if cnt >= 3]

    return {
        "n_turns": n_turns,
        "n_actions": n_actions,
        "n_clicks": n_clicks,
        "n_keystrokes": n_keystrokes,
        "n_scrolls": n_scrolls,
        "n_errors": n_errors,
        "n_timeouts": n_timeouts,
        "submitted": submitted,
        "submit_value": submit_value,
        "repeated_actions": repeated_actions,
        "error_snippets": error_messages[:5],
    }


def _compress_trajectory(conversation: list[dict]) -> str:
    """Compress a GUI trajectory into a failure-focused summary."""
    events: list[dict] = []
    prev_code = ""

    for entry in conversation:
        role = entry.get("role", "")
        parts = entry.get("parts", [])

        if role == "assistant":
            for part in parts:
                if part.get("type") == "tool_use":
                    fn = part.get("name", "")
                    inp = part.get("input", {})
                    answer = inp.get("answer", "")

                    if fn in ("submit", "task_submit"):
                        events.append({"type": "submit", "value": answer})
                    elif fn == "computer":
                        # Structured action format (computer_use native tool)
                        action_type = inp.get("action", "")
                        coord = inp.get("coordinate")
                        text_val = inp.get("text", "")
                        desc = str(action_type)
                        if coord:
                            desc += f"({coord[0]},{coord[1]})"
                        if text_val:
                            desc += f" '{str(text_val)[:100]}'"
                        prev_code = desc[:250] if desc else ""
                        if prev_code:
                            events.append({"type": "action", "code": prev_code})

        elif role == "user":
            for part in parts:
                if part.get("type") == "tool_result":
                    content = part.get("text", "").strip()
                    is_error = (
                        "Error:" in content
                        or "ERROR:" in content
                        or "Traceback" in content[:200]
                        or "TIMEOUT" in content.upper()[:50]
                        or "timed out" in content.lower()[:80]
                    )
                    if is_error:
                        events.append({
                            "type": "error",
                            "code": prev_code,
                            "output": content[:300],
                        })

    # Build compressed summary
    parts: list[str] = []
    n_actions = sum(1 for e in events if e["type"] == "action")
    n_errors = sum(1 for e in events if e["type"] == "error")
    submitted = any(e["type"] == "submit" for e in events)

    parts.append(f"Actions: {n_actions}, Errors: {n_errors}, Submitted: {submitted}")

    # First 3 actions (approach)
    actions_seen = 0
    for e in events:
        if e["type"] == "action":
            actions_seen += 1
            if actions_seen <= 3:
                parts.append(f"[start] computer({e['code'][:150]})")

    # All errors with context
    if n_errors > 0:
        parts.append(f"\n--- Errors ({n_errors}) ---")
        for e in events:
            if e["type"] == "error":
                parts.append(f"  code: {e.get('code', '?')[:150]}")
                parts.append(f"  err: {e['output'][:200]}")

    # Detect loops (same action repeated 3+ times)
    action_list = [e["code"] for e in events if e["type"] == "action"]
    action_counts: dict[str, int] = {}
    for a in action_list:
        action_counts[a] = action_counts.get(a, 0) + 1
    loops = {a: n for a, n in action_counts.items() if n >= 3}
    if loops:
        parts.append(f"\n--- Repeated actions ---")
        for a, n in loops.items():
            parts.append(f"  {a[:120]} (x{n})")

    # Last 3 actions
    last_actions = [e for e in events if e["type"] == "action"][-3:]
    if last_actions:
        parts.append(f"\n--- Final actions ---")
        for e in last_actions:
            parts.append(f"  computer({e['code'][:150]})")

    if submitted:
        submit_events = [e for e in events if e["type"] == "submit"]
        if submit_events:
            parts.append(f"\n[submitted] {submit_events[-1].get('value', '')}")

    return "\n".join(parts)


def _format_signals(signals: dict) -> str:
    """Format trajectory signals as a concise text block."""
    lines = [
        f"Turns: {signals['n_turns']}, Actions: {signals['n_actions']}, "
        f"Clicks: {signals['n_clicks']}, Keystrokes: {signals['n_keystrokes']}, "
        f"Scrolls: {signals['n_scrolls']}",
        f"Errors: {signals['n_errors']}, Timeouts: {signals['n_timeouts']}",
        f"Submitted: {signals['submitted']}",
    ]
    if signals.get("repeated_actions"):
        lines.append(f"Repeated actions: {'; '.join(s[:80] for s in signals['repeated_actions'][:3])}")
    if signals.get("error_snippets"):
        lines.append(f"Error samples: {'; '.join(s[:60] for s in signals['error_snippets'][:3])}")
    if signals.get("bot_detection"):
        lines.append(f"Bot detection: {signals['bot_detection']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bot detection helpers
# ---------------------------------------------------------------------------

_BOT_PATTERNS = [
    (r"captcha|i['\u2019]m not a robot|recaptcha|hcaptcha", "captcha"),
    (r"cloudflare.*verify|verify you are human|challenge.*cloudflare", "cloudflare_challenge"),
    (r"access denied|403 forbidden|403 error|HTTP 403", "access_denied_403"),
    (r"unusual traffic|automated queries|bot.*detect|rate.?limit", "rate_limit_or_bot"),
    (r"please verify|security check|human verification", "verification_challenge"),
]


def _detect_bot_detection(conversation: list[dict], compressed_traj: str) -> str | None:
    """Scan trajectory for bot detection / access denial patterns.

    Returns a short label like 'captcha', 'cloudflare_challenge', 'access_denied_403',
    or None if no bot detection detected.
    """
    # Collect all text from assistant reasoning + tool results
    texts: list[str] = []
    for entry in conversation:
        for part in entry.get("parts", []):
            if part.get("type") == "text":
                texts.append(part.get("text", "")[:2000])
            elif part.get("type") == "tool_result":
                texts.append(part.get("text", "")[:2000])
    texts.append(compressed_traj)

    combined = " ".join(texts).lower()
    for pattern, label in _BOT_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return label
    return None


# ---------------------------------------------------------------------------
# Build rich evaluation text for propose prompt
# ---------------------------------------------------------------------------

def _build_eval_text(score: float, eval_detail: dict, bot_detection: str | None) -> str:
    """Build a detailed evaluation description for the propose prompt.

    Includes metric function, result_state (what agent produced), and failure reason.
    Does NOT include expected_state to avoid test leakage.
    """
    parts = [f"FAILED (score={score:.1f})"]

    metric_func = eval_detail.get("metric_func", "")
    if metric_func:
        parts.append(f"Evaluation metric: {metric_func}")

    failure_reason = eval_detail.get("failure_reason", "")
    if failure_reason and failure_reason != "none":
        parts.append(f"Failure reason: {failure_reason}")

    # Per-metric details (result_state only, no expected_state)
    details = eval_detail.get("details", [])
    if details:
        parts.append("Per-metric breakdown:")
        for d in details[:5]:  # cap at 5 metrics
            m_name = d.get("metric", "?")
            m_score = d.get("score", 0.0)
            result_repr = d.get("result_state", "")
            line = f"  - {m_name}: score={m_score:.1f}"
            if result_repr and result_repr not in ("None", "''", '""'):
                # Truncate large result_state to avoid prompt bloat
                result_repr = result_repr[:300]
                line += f", agent_result={result_repr}"
            fr = d.get("failure_reason", "")
            if fr:
                line += f" ({fr})"
            parts.append(line)

    # Top-level result_state if no per-metric details
    if not details:
        rs = eval_detail.get("result_state", "")
        if rs and rs not in ("None", "''", '""'):
            parts.append(f"Agent's result state: {rs[:400]}")

    if bot_detection:
        parts.append(f"\nNOTE: Bot detection/access denial was detected in the trajectory "
                     f"(type: {bot_detection}). This failure may be caused by anti-bot "
                     f"measures rather than agent error. Focus proposals on workarounds "
                     f"(alternative sites, cached pages, different approaches) rather than "
                     f"GUI technique improvements.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Build propose messages with last N screenshots kept
# ---------------------------------------------------------------------------

def _build_propose_messages(
    messages: list[dict],
    keep_last_n_images: int = 3,
) -> list[dict]:
    """Copy conversation messages for propose, keeping last N screenshots.

    - Strips thinking blocks (incompatible across models)
    - Strips all images EXCEPT the last N (from tool_result or top-level)
    - This gives the propose step visual context of the final state
    """
    # First pass: count total images to know which ones are the last N
    total_images = 0
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type", "")
            if bt == "image":
                total_images += 1
            elif bt == "tool_result" and isinstance(b.get("content"), list):
                for c in b["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        total_images += 1

    # Threshold: keep images with index >= this
    keep_from = max(0, total_images - keep_last_n_images)

    # Second pass: build messages, keeping only last N images
    img_idx = 0
    propose_messages = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            propose_messages.append(msg)
            continue
        if not isinstance(content, list):
            continue
        new_blocks = []
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type", "")
            # Strip thinking blocks
            if bt == "thinking":
                continue
            # Top-level image blocks: keep only last N
            if bt == "image":
                if img_idx >= keep_from:
                    new_blocks.append(b)
                img_idx += 1
                continue
            # For tool_result, selectively keep images
            if bt == "tool_result" and isinstance(b.get("content"), list):
                b = dict(b)
                new_content = []
                for c in b["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        if img_idx >= keep_from:
                            new_content.append(c)
                        img_idx += 1
                    else:
                        new_content.append(c)
                b["content"] = new_content
            new_blocks.append(b)
        if new_blocks:
            propose_messages.append({"role": msg["role"], "content": new_blocks})

    return propose_messages


# ---------------------------------------------------------------------------
# Propose system prompt (replaces the old minimal 22-word version)
# ---------------------------------------------------------------------------

PROPOSE_SYSTEM_PROMPT = """\
You are a skill extraction agent for an OSWorld GUI desktop agent. \
You analyze task attempts and distill reusable skills.

A good skill is:
- **Specific**: actual menu paths, keyboard shortcuts, a11y tree element names, CLI commands
- **Structured**: bullet points under "Key techniques" and "Gotchas", under 200 words
- **Actionable**: another agent can read it and immediately apply the technique
- **Transferable**: useful beyond this single task — focus on the application/domain pattern

A bad skill is:
- Generic advice ("be careful", "verify output", "check the screenshot")
- Task-specific (only applies to this exact task, not future similar ones)
- Redundant with an existing skill in the library

If the failure was caused by bot detection, CAPTCHA, or access denial, \
focus on workaround strategies (alternative sites, cached pages, different search engines) \
rather than GUI techniques. If nothing useful was learned, output ACTION: NONE."""

# ---------------------------------------------------------------------------
# Analyze + Propose prompt (merged, in solver conversation context)
# ---------------------------------------------------------------------------

ANALYZE_AND_PROPOSE_PROMPT = """\
The evaluation result for this task:

{eval_result}

## Trajectory signals
{trajectory_signals}

## Compressed trajectory
{compressed_trajectory}

{existing_skills_section}

## Step 1: Analyze the result
Consider the evaluation score, trajectory signals (errors, loops, timeouts), and compressed trajectory.
For EACH distinct issue or failure reason, output:
ISSUE: <one-line summary of what went wrong or what was needed>
DETAIL: <specific GUI actions, UI elements, or techniques that were missing>

## Step 2: Propose a skill
Based on your analysis, write a SHORT skill for future tasks of this type.

TOPIC: <broad application-level topic, e.g. "libreoffice-calc", "chrome", "gimp", "vlc", "vscode", "thunderbird", "os-desktop">
ACTION: NEW / ENHANCE / NONE
TARGET: existing_skill_name (only for ENHANCE)
NAME: short-kebab-name (only for NEW)
DESCRIPTION: one sentence saying WHEN this skill applies — the agent sees ONLY this line to decide whether to read the skill. Be specific: "For LibreOffice Calc tasks involving formula editing and cell formatting" not "For spreadsheet tasks"
CONTENT:
## Key techniques
- (specific GUI actions, keyboard shortcuts, menu paths, or a11y tree patterns)
## Gotchas
- (specific pitfalls based on what went wrong)

FORBIDDEN — do NOT include any of the following (the agent already knows these):
- Basic mouse/keyboard operations (how to click, type, press keys)
- How to take or interpret screenshots
- How to use the computer_use tool (coordinate clicking, typing, scrolling)
- Generic GUI advice ("look at the screen", "wait for the window to load", "check the result")
- Retry/timeout strategies

REQUIRED — only include application-specific knowledge the agent does NOT already have:
- Application-specific menu paths, keyboard shortcuts, and dialog sequences
- Application-specific a11y tree element names and patterns
- Non-obvious workflows (e.g., enabling a hidden feature, multi-step dialog navigation)

Rules:
- Bullet points, not paragraphs. CONTENT must be under 200 words.
- Be SPECIFIC: include actual menu paths, keyboard shortcuts, a11y tree element names
- Focus on application-specific knowledge, NOT generic advice (no "look carefully", "check errors")
- Prefer ENHANCE over NEW if an existing skill is related
- TOPIC should be BROAD (application-level), not task-specific. Use "libreoffice-calc" not "libreoffice-calc-vlookup"
- TOPIC should match an existing topic if applicable; create a new one only if no existing topic fits
- If the task passed easily or nothing useful was learned, output ACTION: NONE"""

ANALYZE_AND_PROPOSE_PASS_PROMPT = """\
The evaluation result for this task:

PASSED (score={score:.1f})

## Trajectory signals
{trajectory_signals}

## Compressed trajectory
{compressed_trajectory}

{existing_skills_section}

## Step 1: Analyze what worked
Review the trajectory and identify techniques that were effective — especially \
non-obvious ones (workarounds, specific menu paths, key combos, a11y patterns).

## Step 2: Propose a skill (if warranted)
If the approach contained reusable, non-obvious techniques worth preserving:

TOPIC: <broad application-level topic, e.g. "libreoffice-calc", "chrome", "gimp", "vlc", "vscode", "thunderbird", "os-desktop">
ACTION: NEW / ENHANCE / NONE
TARGET: existing_skill_name (only for ENHANCE)
NAME: short-kebab-name (only for NEW)
DESCRIPTION: one sentence saying WHEN this skill applies — the agent sees ONLY this line to decide whether to read the skill. Be specific: "For GIMP tasks involving layer manipulation and export settings" not "For image editing tasks"
CONTENT:
## Key techniques
- (specific steps that worked)
## Gotchas
- (obstacles encountered and how they were overcome)

FORBIDDEN — do NOT include any of the following (the agent already knows these):
- Basic mouse/keyboard operations (how to click, type, press keys)
- How to take or interpret screenshots
- How to use the computer_use tool (coordinate clicking, typing, scrolling)
- Generic GUI advice ("look at the screen", "wait for the window to load", "check the result")
- Retry/timeout strategies

REQUIRED — only include application-specific knowledge the agent does NOT already have:
- Application-specific menu paths, keyboard shortcuts, and dialog sequences
- Application-specific a11y tree element names and patterns
- Non-obvious workflows discovered during this task

Rules:
- Only propose if the technique is non-trivial and transferable
- Output ACTION: NONE if the task was straightforward or already covered by existing skills
- Prefer ENHANCE over NEW if an existing skill is related
- TOPIC should be BROAD (application-level), not task-specific. Use "libreoffice-calc" not "libreoffice-calc-vlookup"
- CONTENT under 200 words, bullet points only"""

# ---------------------------------------------------------------------------
# Curator prompt (reviews proposals per topic)
# ---------------------------------------------------------------------------

CURATOR_PROMPT = """\
You are a skill curator for a GUI task-solving agent on OSWorld (Ubuntu desktop). \
You review skill proposals and decide which to keep in the skill library for topic: {topic}.

## Current Skill Library ({n_skills}/{max_skills} slots used):
{existing_skills_list}

## Proposals from this batch:
{proposals_list}

For each proposal, output ONE of:

ACCEPT: <proposal_name>

MERGE: <proposal_name> INTO <existing_skill_name>
NEW_CONTENT:
(merged content, under 200 words, bullet points only)

SKIP: <proposal_name>
REASON: <brief reason>

Rules:
- MERGE is always preferred over ACCEPT — combine related techniques into fewer, broader skills
- Overlaps existing -> MERGE (append new techniques to existing skill)
- Multiple narrow proposals on the same app -> MERGE into one broad skill
- Budget full -> can only MERGE existing, or SKIP
- Check DESCRIPTION quality: it must clearly say WHEN the skill applies. The agent decides to read based on description alone. Vague descriptions like "for desktop tasks" → SKIP or rewrite in MERGE
- SKIP proposals containing FORBIDDEN content (basic mouse/keyboard ops, screenshot usage, computer_use tool usage, generic GUI advice, retry strategies) — the agent already knows these
- Keep skills SHORT and SPECIFIC -- actual menu paths, shortcuts, and GUI techniques, not advice
- Few broad skills > many narrow ones — one skill covering multiple techniques is better than many single-technique skills

If no proposals: NO_PROPOSALS"""

# ---------------------------------------------------------------------------
# General curator prompt
# ---------------------------------------------------------------------------

GENERAL_CURATOR_PROMPT = """\
You are a meta-learning curator. You analyze failure patterns ACROSS tasks \
to distill general skills that help the agent on ANY OSWorld desktop task.

## Failed Task Analysis ({n_failed} tasks):
{failed_summaries}

## Current General Skills ({n_general}/{max_general} slots):
{general_skills_list}

For REPEATED patterns across 2+ different tasks, output:

NEW_GENERAL: <kebab-name>
DESCRIPTION: <one line saying WHEN this skill applies — the agent sees ONLY this line to decide whether to read the skill>
CONTENT:
## Pattern
- (one line: what failure type)
## Strategy
- (3-5 bullet points: specific GUI techniques/shortcuts)
(Under 200 words, bullet points only)

UPDATE_GENERAL: <existing-name>
NEW_CONTENT:
(updated content, under 200 words)

DELETE_GENERAL: <existing-name>
REASON: <why>

If no cross-task patterns: NO_PATTERNS

FORBIDDEN — do NOT include any of the following in skills (the agent already knows these):
- Basic mouse/keyboard operations (how to click, type, press keys)
- How to take or interpret screenshots
- How to use the computer_use tool (coordinate clicking, typing, scrolling)
- Generic GUI advice ("look at the screen", "wait for the window to load", "check the result")
- Retry/timeout strategies

REQUIRED — only include application-specific knowledge the agent does NOT already have:
- Application-specific menu paths, keyboard shortcuts, and dialog sequences
- Application-specific a11y tree patterns
- Cross-application GUI interaction patterns (e.g., drag-and-drop conventions, file dialogs)

Rules:
- Max {max_general} general skills. Quality > quantity.
- Must appear in 2+ different tasks to be general.
- SPECIFIC and ACTIONABLE -- actual GUI actions, menu paths, shortcuts, not advice.
- DELETE skills that contain FORBIDDEN content or are too generic.
- Prefer UPDATE over NEW."""

# ---------------------------------------------------------------------------
# Agent curator (flat skills/evolved/ structure)
# ---------------------------------------------------------------------------

AGENT_BASH_TOOL = {
    "toolSpec": {
        "name": "bash",
        "description": "Run a bash command in the workspace directory. Use this to read, create, and update skill files.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute in the workspace directory",
                    }
                },
                "required": ["command"],
            }
        },
    }
}

AGENT_CURATOR_SYSTEM = """\
You are a skill curation agent for a GUI task-solving AI on OSWorld (Ubuntu desktop). \
You merge and refine skill proposals from solvers into a flat skill library.

## Workspace Layout
```
skills/
  seed/            # Read-only seed skills — do NOT modify
  evolved/<skill-name>/SKILL.md   # YOUR output — flat, no sub-directories
  SKILL_TREE.md    # Auto-generated — do NOT edit
```

## Skill File Format
```
---
name: <kebab-case-name>
description: <one sentence, under 100 chars>
---

## Key techniques
- (specific GUI actions, menu paths, keyboard shortcuts)
## Gotchas
- (pitfalls to avoid, with specifics)
```

## Your Job
You receive proposals from solvers who just completed GUI tasks. Each proposal contains \
domain knowledge extracted in-context. Your job is to MERGE these into a compact, \
high-quality skill library under skills/evolved/.

1. Read existing skills with bash to see what's already there
2. MERGE similar proposals into BROAD skills:
   - Multiple app-specific proposals (e.g., "calc-cell-formatting", "calc-conditional-formatting", \
"calc-chart-creation") should become ONE broad skill (e.g., "libreoffice-calc-techniques")
   - PRESERVE specific menu paths, shortcuts, and gotchas from every proposal — don't lose details
   - A single skill can cover multiple related techniques (e.g., "file-management" \
covers Nautilus, terminal file ops, archive handling, permissions)
3. UPDATE existing skills by appending new techniques rather than creating new ones
4. Write skills using bash:
   ```
   mkdir -p skills/evolved/<name>
   cat > skills/evolved/<name>/SKILL.md << 'SKILL'
   ...
   SKILL
   ```
5. Summarize all changes at the end

## Constraints
- HARD LIMIT: {max_skills} total skills under skills/evolved/ — count before writing
- If at limit, MERGE into existing skills instead of creating new ones
- Skills must be SPECIFIC: actual menu paths, keyboard shortcuts, GUI actions, tool quirks
- NO generic advice ("read carefully", "be thorough", "check errors")
- Bullet points only, under 300 words per skill body
- Prefer FEWER broader skills over MANY narrow ones
- Do NOT touch skills/seed/ — those are read-only
- Do NOT create topic/ or general/ subdirectories — everything goes in skills/evolved/
"""

# ---------------------------------------------------------------------------
# Skill data structures
# ---------------------------------------------------------------------------

class SkillMeta:
    """Lightweight skill metadata."""
    def __init__(self, name: str, description: str, path: str, body: str = ""):
        self.name = name
        self.description = description
        self.path = path
        self.body = body


def load_skills(workspace_dir: Path) -> list[SkillMeta]:
    """Load all skills from workspace."""
    skills = []
    skills_dir = workspace_dir / "skills"
    if not skills_dir.exists():
        return skills
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        content = skill_file.read_text().strip()
        name = skill_file.parent.name
        desc = ""
        for line in content.split("\n"):
            if line.strip().startswith("description:"):
                desc = line.split(":", 1)[1].strip()
                break
        body = content
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                body = content[end + 3:].strip()
        rel_path = str(skill_file.parent.relative_to(workspace_dir))
        skills.append(SkillMeta(name, desc, rel_path, body))
    return skills


def _select_relevant_topics(
    task_prompt: str,
    topics: dict[str, list[SkillMeta]],
    region: str,
    model_id: str,
) -> list[str]:
    """Quick LLM call to select relevant topics for a task."""
    topic_list = "\n".join(
        f"- {topic}: {', '.join(s.name + ' - ' + s.description for s in skills)}"
        for topic, skills in sorted(topics.items())
    )
    prompt = (
        f"Task:\n{task_prompt}\n\n"
        f"Available skill topics:\n{topic_list}\n\n"
        f"Which topics are relevant to this task? "
        f"You are NOT required to select any topic — only select topics that are genuinely relevant. "
        f"Output ONLY a JSON list of topic names, e.g. [\"libreoffice-calc\", \"chrome\"]. "
        f"If none are relevant, output []."
    )
    client = _get_client(region)
    resp, err = _call_bedrock(client, model_id, "You select relevant skill topics.", prompt,
                              max_tokens=256, temperature=0.0)
    if err or not resp:
        return list(topics.keys())  # fallback: all topics
    # Parse JSON list
    try:
        start = resp.find("[")
        end = resp.rfind("]") + 1
        if start >= 0 and end > start:
            selected = json.loads(resp[start:end])
            return [t for t in selected if t in topics]
    except (json.JSONDecodeError, ValueError):
        pass
    return list(topics.keys())  # fallback: all topics


def build_system_prompt(
    skills: list[SkillMeta],
    lazy_load: bool = False,
    selected_topics: list[str] | None = None,
) -> str:
    """Build system prompt with skills.

    lazy_load=False (default): full body injected into prompt.
    lazy_load=True: only name+description, solver uses read_skill() tool.
    selected_topics: if provided, only inject topic skills from these topics (full injection only).
    """
    parts = [OSW_SYSTEM_PROMPT]

    seed_skills = [s for s in skills if "seed/" in s.path]
    topic_skills = [s for s in skills if "topic/" in s.path]
    gen_skills = [s for s in skills if "general/" in s.path]
    evolved_skills = [s for s in skills if "evolved/" in s.path]

    all_skills = seed_skills + topic_skills + gen_skills + evolved_skills
    if not all_skills:
        return "\n".join(parts)

    if lazy_load:
        # Lazy load: show all skills by name+desc, solver picks via read_skill
        parts.append("\n\n## Available Skills\n")
        parts.append(
            "You have specialized skills available. "
            "Call `read_skill(name)` to load the full content "
            "before tackling a relevant challenge.\n"
        )
        for s in all_skills:
            parts.append(f"- **{s.name}**: {s.description}")
        parts.append(
            "\nAfter you think you have completed the task, "
            "read the self-verification skill to verify your solution."
        )
    else:
        # Seed skills: always inject full body
        if seed_skills:
            parts.append("\n\n## Core skills")
            for s in seed_skills:
                parts.append(f"\n### {s.name}\n{s.body}" if s.body else f"\n### {s.name}\n{s.description}")

        # Full injection: filter topic skills if selected_topics provided
        if topic_skills:
            by_topic: dict[str, list[SkillMeta]] = defaultdict(list)
            for s in topic_skills:
                path_parts = Path(s.path).parts
                topic = path_parts[2] if len(path_parts) > 3 else path_parts[1] if len(path_parts) > 2 else "other"
                by_topic[topic].append(s)

            if selected_topics is not None:
                by_topic = {t: ss for t, ss in by_topic.items() if t in selected_topics}

            if by_topic:
                parts.append("\n\n## Domain skills")
                for topic in sorted(by_topic):
                    parts.append(f"\n**[{topic}]**")
                    for s in by_topic[topic]:
                        parts.append(f"\n### {s.name}\n{s.body}" if s.body else f"\n### {s.name}\n{s.description}")

        if gen_skills:
            parts.append("\n\n## General strategies")
            for s in gen_skills:
                parts.append(f"\n### {s.name}\n{s.body}" if s.body else f"\n### {s.name}\n{s.description}")

        if evolved_skills:
            parts.append("\n\n## Evolved skills")
            for s in evolved_skills:
                parts.append(f"\n### {s.name}\n{s.body}" if s.body else f"\n### {s.name}\n{s.description}")

        parts.append(
            "\n\nAfter you think you have completed the task, "
            "read the self-verification skill above to verify your solution."
        )

    n_topic = len([s for s in topic_skills if selected_topics is None or
                   any(t in s.path for t in (selected_topics or []))])
    mode = "lazy" if lazy_load else "injected"
    logger.debug("%d seed + %d topic + %d general + %d evolved skills (%s)",
                 len(seed_skills), n_topic, len(gen_skills), len(evolved_skills), mode)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------

def _solve_one_task(
    task_config: dict,
    env,
    model_id: str,
    region: str,
    max_tokens: int,
    system_prompt: str,
    skills: list[SkillMeta],
    workspace_dir: Path,
    log_dir: Path,
    max_steps: int = 30,
    do_propose: bool = True,
    evolve_all: bool = False,
    lazy_load: bool = False,
    selector_model: str = "",
    curator_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    feedback_level: str = "standard",
) -> dict:
    """Full pipeline for one OSWorld task: solve -> evaluate -> analyze+propose.

    The caller owns the DesktopEnv lifetime; this function calls env.reset()
    to revert the VM snapshot for this task.

    Returns dict with task info, result, and optional proposal.
    """
    task_name = _task_id(task_config)
    domain = _task_domain(task_config)
    task_instruction = task_config.get("instruction", task_config.get("task", ""))
    t0 = time.time()

    # Setup per-task logger
    task_log_dir = log_dir / task_name
    task_log_dir.mkdir(parents=True, exist_ok=True)
    task_log = logging.getLogger(f"task.{task_name}")
    task_log.setLevel(logging.DEBUG)
    task_log.propagate = False
    task_log.handlers.clear()
    fh = logging.FileHandler(task_log_dir / "solve.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    task_log.addHandler(fh)

    try:
        # 1. Reset VM snapshot for this task (env is reused across tasks)
        setup_ok = False
        for _setup_attempt in range(3):
            try:
                env.reset(task_config=task_config)
                setup_ok = True
                break
            except Exception as setup_err:
                task_log.warning("Setup attempt %d/3 failed: %s", _setup_attempt + 1, str(setup_err)[:200])
                if _setup_attempt < 2:
                    time.sleep(10)
        if not setup_ok:
            raise RuntimeError(f"Environment setup failed after 3 attempts for {task_name}")
        task_log.info("Environment reset for %s (domain=%s), waiting 60s for setup...",
                       task_name, domain)
        time.sleep(60)  # Wait for VM environment to be ready (apps to load, etc.)
        obs = env._get_obs()  # Get fresh observation after setup

        # 2. Select relevant skills
        task_system_prompt = system_prompt
        skills_for_solver = None

        if lazy_load and skills:
            # Lazy load: all skills available via read_skill tool
            skills_for_solver = {s.name: s.body or s.description for s in skills}
        elif not lazy_load and skills and selector_model:
            # Full injection: select relevant topics first
            topic_skills = [s for s in skills if "topic/" in s.path]
            if topic_skills:
                by_topic: dict[str, list[SkillMeta]] = defaultdict(list)
                for s in topic_skills:
                    path_parts = Path(s.path).parts
                    topic = path_parts[2] if len(path_parts) > 3 else path_parts[1] if len(path_parts) > 2 else "other"
                    by_topic[topic].append(s)
                selected = _select_relevant_topics(
                    task_instruction, by_topic, region, selector_model,
                )
                task_system_prompt = build_system_prompt(skills, selected_topics=selected)
                task_log.info("Selected topics: %s (from %d available)", selected, len(by_topic))

        # 3. Solve via ReAct
        timeout_sec = task_config.get("agent_timeout_sec", 900)
        react_result = react_solve(
            task_prompt=task_instruction,
            env=env,
            model_id=model_id,
            region=region,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_turns=max_steps,
            log=task_log,
            system_prompt=task_system_prompt,
            skills=skills_for_solver,
            initial_obs=obs,
        )
        conversation = extract_conversation(react_result.messages)
        usage = {
            "input_tokens": react_result.total_input_tokens,
            "output_tokens": react_result.total_output_tokens,
        }
        solve_time = time.time() - t0

        # 4. Evaluate (env.evaluate_detailed() returns dict with score + details)
        time.sleep(20)  # Wait for environment to settle before evaluation
        eval_detail = {}
        try:
            eval_detail = env.evaluate_detailed()
            score = float(eval_detail.get("score", 0.0))
        except Exception as e:
            task_log.warning("Evaluation failed: %s", e)
            score = 0.0
            eval_detail = {"score": 0.0, "metric_func": "unknown",
                           "result_state": f"eval_exception: {str(e)[:200]}",
                           "failure_reason": "eval_exception", "details": []}

        passed = score >= 1.0
        eval_output = f"score={score:.1f}"

        task_log.info("RESULT: %s (score=%.1f, %.0fs, metric=%s, reason=%s)",
                      "PASS" if passed else "FAIL", score, time.time() - t0,
                      eval_detail.get("metric_func", "?"),
                      eval_detail.get("failure_reason", "?"))

        # 5. Extract trajectory signals
        traj_signals = _extract_trajectory_signals(conversation)
        compressed_traj = _compress_trajectory(conversation)

        # 6b. Detect bot detection / access denial in trajectory
        bot_detection = _detect_bot_detection(conversation, compressed_traj)
        if bot_detection:
            traj_signals["bot_detection"] = bot_detection
            task_log.info("Bot detection: %s", bot_detection)

        # 7. Analyze + Propose (in conversation context)
        #    - Failed tasks: always propose if do_propose
        #    - Passed tasks: propose only if evolve_all
        feedback_analysis = None
        proposal = None

        should_propose = (do_propose and react_result.messages
                          and (not passed or evolve_all))

        if should_propose:
            try:
                # Build existing skills section
                existing_all_skills = []
                skills_dir = workspace_dir / "skills"
                if skills_dir.exists():
                    for sf in sorted(skills_dir.rglob("SKILL.md")):
                        sn = sf.parent.name
                        rel = sf.parent.relative_to(skills_dir)
                        topic_tag = rel.parts[1] if len(rel.parts) > 2 else rel.parts[0]
                        sd = ""
                        for sline in sf.read_text().split("\n"):
                            if sline.strip().startswith("description:"):
                                sd = sline.split(":", 1)[1].strip()
                                break
                        existing_all_skills.append((sn, topic_tag, sd))

                if existing_all_skills:
                    skills_section = "Current skills:\n" + "\n".join(
                        f"- **{n}** [{t}]: {d}" for n, t, d in existing_all_skills
                    )
                else:
                    skills_section = "No existing skills yet."

                # Choose prompt template based on pass/fail
                if passed:
                    prompt_text = ANALYZE_AND_PROPOSE_PASS_PROMPT.format(
                        score=score,
                        trajectory_signals=_format_signals(traj_signals),
                        compressed_trajectory=_truncate(compressed_traj, 1500),
                        existing_skills_section=skills_section,
                    )
                else:
                    if feedback_level == "minimal":
                        eval_text = f"FAILED (score={score:.1f})"
                    elif feedback_level == "standard":
                        eval_text = _build_eval_text(score, eval_detail, bot_detection)
                    else:  # full
                        eval_text = _build_eval_text(score, eval_detail, bot_detection)
                        # Add full result_state and raw details for full level
                        if eval_detail.get("details"):
                            for d in eval_detail["details"]:
                                rs = d.get("result_state", "")
                                if rs:
                                    eval_text += f"\nFull result_state ({d.get('metric','?')}): {rs[:1000]}"
                        elif eval_detail.get("result_state"):
                            eval_text += f"\nFull result_state: {str(eval_detail['result_state'])[:1000]}"
                    prompt_text = ANALYZE_AND_PROPOSE_PROMPT.format(
                        eval_result=eval_text,
                        trajectory_signals=_format_signals(traj_signals),
                        compressed_trajectory=_truncate(compressed_traj, 1500),
                        existing_skills_section=skills_section,
                    )

                # Continue in solver conversation context via AnthropicBedrock
                from anthropic import AnthropicBedrock, APIError, APIStatusError

                # Copy messages, keeping last N screenshots for visual context
                propose_messages = _build_propose_messages(
                    react_result.messages, keep_last_n_images=3,
                )

                # Append propose prompt as new user message
                propose_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                })

                propose_client = AnthropicBedrock(
                    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    aws_region=region,
                )
                resp_text = ""
                try:
                    propose_resp = propose_client.messages.create(
                        model=curator_model,
                        max_tokens=1536,
                        messages=propose_messages,
                        system=PROPOSE_SYSTEM_PROMPT,
                        temperature=0.3,
                    )
                    for block in propose_resp.content:
                        if getattr(block, "type", None) == "text":
                            resp_text += block.text
                except (APIError, APIStatusError) as e:
                    task_log.warning("Propose call failed: %s", str(e)[:200])
                except Exception as e:
                    task_log.warning("Propose call error: %s", str(e)[:200])

                if resp_text:
                    # Extract analysis (before ACTION:)
                    action_idx = resp_text.upper().find("ACTION:")
                    if action_idx > 0:
                        feedback_analysis = resp_text[:action_idx].strip()
                    else:
                        feedback_analysis = resp_text.strip()

                    # Parse proposal
                    proposal = _parse_proposal(resp_text, task_name)

            except Exception as e:
                task_log.warning("Analyze+propose failed: %s", str(e)[:200])

        # Save artifacts
        (task_log_dir / "result.txt").write_text(f"passed={passed}\nscore={score}\n")
        (task_log_dir / "conversation.json").write_text(
            json.dumps(conversation, indent=2, ensure_ascii=False, default=str)
        )

        return {
            "task_name": task_name,
            "domain": domain,
            "passed": passed,
            "score": score,
            "eval_output": eval_output,
            "eval_detail": {
                "metric_func": eval_detail.get("metric_func", ""),
                "failure_reason": eval_detail.get("failure_reason", ""),
                "result_state": str(eval_detail.get("result_state", ""))[:500],
                "bot_detection": bot_detection,
            },
            "usage": usage,
            "solve_time": solve_time,
            "total_time": time.time() - t0,
            "feedback_analysis": feedback_analysis,
            "proposal": proposal,
            "trajectory_signals": traj_signals,
            "compressed_trajectory": compressed_traj,
            "task_config": {
                "id": task_name,
                "domain": domain,
                "instruction": task_instruction[:500],
                "related_apps": task_config.get("related_apps", []),
            },
        }

    except Exception as e:
        task_log.error("FATAL: %s", str(e)[:500])
        return {
            "task_name": task_name,
            "domain": domain,
            "passed": False,
            "score": 0.0,
            "eval_output": f"ERROR: {str(e)[:500]}",
            "usage": {},
            "solve_time": 0,
            "total_time": time.time() - t0,
            "feedback_analysis": None,
            "proposal": None,
            "task_config": {
                "id": task_name,
                "domain": domain,
                "instruction": task_config.get("instruction", "")[:500],
            },
            "error": str(e)[:500],
        }


def _parse_proposal(resp: str, task_name: str) -> dict | None:
    """Parse skill proposal from LLM response."""
    if "ACTION: NONE" in resp.upper():
        return None

    proposal = {
        "source_task": task_name,
        "topic": "general",
        "raw": resp,
        "action": "NEW",
        "target": "",
        "name": "",
        "description": "",
        "content": "",
    }

    for line in resp.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TOPIC:"):
            raw_topic = stripped.split(":", 1)[1].strip()
            proposal["topic"] = re.sub(r"[^a-z0-9-]", "-", raw_topic.lower()).strip("-") or "general"
        elif upper.startswith("ACTION:"):
            proposal["action"] = stripped.split(":", 1)[1].strip().upper()
        elif upper.startswith("TARGET:"):
            proposal["target"] = stripped.split(":", 1)[1].strip()
        elif upper.startswith("NAME:"):
            raw = stripped.split(":", 1)[1].strip()
            proposal["name"] = re.sub(r"[^a-z0-9-]", "-", raw.lower()).strip("-")
        elif upper.startswith("DESCRIPTION:"):
            proposal["description"] = stripped.split(":", 1)[1].strip()[:150]

    idx = resp.upper().find("CONTENT:")
    if idx >= 0:
        raw_content = resp[idx + len("CONTENT:"):].strip()
        # Strip trailing metadata lines that may leak after content
        _META_TAGS = ("TOPIC:", "ACTION:", "TARGET:", "NAME:", "DESCRIPTION:")
        lines = raw_content.split("\n")
        while lines:
            last = lines[-1].strip().upper()
            if any(last.startswith(tag) for tag in _META_TAGS):
                lines.pop()
            else:
                break
        proposal["content"] = "\n".join(lines).strip()

    if proposal["action"] == "ENHANCE" and proposal["target"] and not proposal["name"]:
        proposal["name"] = proposal["target"]
    if not proposal["name"] and proposal["action"] != "NONE":
        proposal["name"] = f"skill-{task_name[:20]}"
    if not proposal["content"]:
        return None

    return proposal


# ---------------------------------------------------------------------------
# Curation functions
# ---------------------------------------------------------------------------

def _curate_topic_proposals(
    topic: str,
    proposals: list[dict],
    workspace_dir: Path,
    region: str,
    model: str,
    max_skills: int = 5,
) -> dict:
    """Curator reviews proposals for one topic."""
    if not proposals:
        return {"added": 0, "merged": 0, "skipped": 0}

    topic_dir = workspace_dir / "skills" / "topic" / topic
    existing = []
    if topic_dir.exists():
        for sf in sorted(topic_dir.rglob("SKILL.md")):
            content = sf.read_text()
            sn = sf.parent.name
            sd = ""
            for sline in content.split("\n"):
                if sline.strip().startswith("description:"):
                    sd = sline.split(":", 1)[1].strip()
                    break
            existing.append((sn, sd))

    existing_list = "\n".join(f"- **{n}**: {d}" for n, d in existing) if existing else "(empty)"
    proposals_lines = []
    for p in proposals:
        proposals_lines.append(
            f"### [{p.get('action', 'NEW')}] {p.get('name', '?')}\n"
            f"  Source: {p['source_task']}\n"
            f"  Description: {p.get('description', '')[:150]}\n"
            f"  Content: {_truncate(p.get('content', ''), 300)}"
        )

    prompt = CURATOR_PROMPT.format(
        topic=topic,
        n_skills=len(existing),
        max_skills=max_skills,
        existing_skills_list=existing_list,
        proposals_list="\n\n".join(proposals_lines),
    )

    client = _get_client(region)
    resp, err = _call_bedrock(client, model, prompt, "Review and decide.", max_tokens=2048, temperature=0.0)
    if err or not resp:
        logger.warning("Curator failed for topic %s: %s", topic, err)
        return {"added": 0, "merged": 0, "skipped": 0}

    return _execute_topic_curation(resp, proposals, existing, workspace_dir, topic, max_skills)


def _execute_topic_curation(
    text: str, proposals: list[dict], existing: list[tuple[str, str]],
    workspace_dir: Path, topic: str, max_skills: int,
) -> dict:
    """Parse curator decisions, write skills."""
    proposal_map = {p["name"]: p for p in proposals}
    existing_names = {n for n, _ in existing}
    count = len(existing)
    stats = {"added": 0, "merged": 0, "skipped": 0}

    def _write(name, desc, content):
        d = workspace_dir / "skills" / "topic" / topic / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\n{content}")

    def _fuzzy(raw, names):
        clean = re.sub(r"[^a-z0-9-]", "-", raw.lower()).strip("-")
        if clean in names:
            return clean
        for n in names:
            if clean in n or n in clean:
                return n
        return None

    for line in text.split("\n"):
        s = line.strip()
        u = s.upper()
        if u.startswith("ACCEPT:"):
            pn = _fuzzy(s.split(":", 1)[1].strip(), set(proposal_map.keys()))
            if pn and pn not in existing_names and count < max_skills:
                p = proposal_map[pn]
                _write(pn, p.get("description", ""), p.get("content", ""))
                existing_names.add(pn)
                count += 1
                stats["added"] += 1
        elif u.startswith("MERGE:"):
            parts = s.split(":", 1)[1].strip()
            if " INTO " in parts.upper():
                sp = parts.split(" INTO " if " INTO " in parts else " into ")
                pn = _fuzzy(sp[0].strip(), set(proposal_map.keys()))
                tn = _fuzzy(sp[1].strip() if len(sp) > 1 else "", existing_names)
                if pn and tn:
                    merge_idx = text.find(s)
                    after = text[merge_idx + len(s):]
                    nc = ""
                    if "NEW_CONTENT:" in after:
                        nc = after.split("NEW_CONTENT:", 1)[1]
                        for m in ["ACCEPT:", "MERGE:", "SKIP:", "NO_PROPOSALS"]:
                            if m in nc:
                                nc = nc[:nc.index(m)]
                        nc = nc.strip()
                    if nc:
                        old_desc = next((d for n, d in existing if n == tn), "")
                        _write(tn, old_desc or proposal_map.get(pn, {}).get("description", ""), nc)
                        stats["merged"] += 1
        elif u.startswith("SKIP:"):
            stats["skipped"] += 1

    return stats


def _curate_general_skills(
    failed_summaries: list[dict],
    workspace_dir: Path,
    region: str,
    model: str,
    max_general: int = 10,
    feedback_level: str = "standard",
) -> dict:
    """General curator: cross-task patterns."""
    if not failed_summaries:
        return {"added": 0, "updated": 0, "deleted": 0}

    summary_lines = []
    for s in failed_summaries[:30]:
        if feedback_level == "minimal":
            # minimal: only task name, status, domain, trajectory signals
            parts = [f"### {s['task_name']} ({s.get('domain', '')})"]
            eval_info = []
            if s.get("eval_metric"):
                eval_info.append(f"metric={s['eval_metric']}")
            if s.get("failure_reason"):
                eval_info.append(f"reason={s['failure_reason']}")
            if s.get("bot_detection"):
                eval_info.append(f"bot_detection={s['bot_detection']}")
            if eval_info:
                parts.append(f"Eval: {', '.join(eval_info)}")
            if s.get("trajectory_signals"):
                parts.append(f"Signals: {_format_signals(s['trajectory_signals'])}")
        elif feedback_level == "full":
            # full: expand truncation limits
            parts = [f"### {s['task_name']} ({s.get('domain', '')})"]
            eval_info = []
            if s.get("eval_metric"):
                eval_info.append(f"metric={s['eval_metric']}")
            if s.get("failure_reason"):
                eval_info.append(f"reason={s['failure_reason']}")
            if s.get("bot_detection"):
                eval_info.append(f"bot_detection={s['bot_detection']}")
            if eval_info:
                parts.append(f"Eval: {', '.join(eval_info)}")
            if s.get("trajectory_signals"):
                parts.append(f"Signals: {_format_signals(s['trajectory_signals'])}")
            if s.get("compressed_trajectory"):
                parts.append(f"Trajectory:\n{_truncate(s['compressed_trajectory'], 1000)}")
            if s.get("feedback_analysis"):
                parts.append(f"Analysis:\n{_truncate(s['feedback_analysis'], 1000)}")
            if s.get("proposal_summary"):
                parts.append(f"Proposal: {_truncate(s['proposal_summary'], 500)}")
        else:
            # standard: current behavior
            parts = [f"### {s['task_name']} ({s.get('domain', '')})"]
            eval_info = []
            if s.get("eval_metric"):
                eval_info.append(f"metric={s['eval_metric']}")
            if s.get("failure_reason"):
                eval_info.append(f"reason={s['failure_reason']}")
            if s.get("bot_detection"):
                eval_info.append(f"bot_detection={s['bot_detection']}")
            if eval_info:
                parts.append(f"Eval: {', '.join(eval_info)}")
            if s.get("trajectory_signals"):
                parts.append(f"Signals: {_format_signals(s['trajectory_signals'])}")
            if s.get("compressed_trajectory"):
                parts.append(f"Trajectory:\n{_truncate(s['compressed_trajectory'], 400)}")
            if s.get("feedback_analysis"):
                parts.append(f"Analysis:\n{_truncate(s['feedback_analysis'], 400)}")
            if s.get("proposal_summary"):
                parts.append(f"Proposal: {_truncate(s['proposal_summary'], 200)}")
        summary_lines.append("\n".join(parts))

    gen_dir = workspace_dir / "skills" / "general"
    existing = []
    if gen_dir.exists():
        for sf in sorted(gen_dir.rglob("SKILL.md")):
            content = sf.read_text()
            sn = sf.parent.name
            sd = ""
            for sline in content.split("\n"):
                if sline.strip().startswith("description:"):
                    sd = sline.split(":", 1)[1].strip()
                    break
            body = content
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    body = content[end + 3:].strip()
            existing.append((sn, sd, body[:300]))

    gen_list = "\n".join(
        f"- **{n}**: {d}" for n, d, _ in existing
    ) if existing else "(empty)"

    prompt = GENERAL_CURATOR_PROMPT.format(
        n_failed=len(failed_summaries),
        failed_summaries="\n\n".join(summary_lines),
        n_general=len(existing),
        max_general=max_general,
        general_skills_list=gen_list,
    )

    client = _get_client(region)
    resp, err = _call_bedrock(client, model, prompt, "Analyze and decide.", max_tokens=4096, temperature=0.0)
    if err or not resp:
        return {"added": 0, "updated": 0, "deleted": 0}

    return _execute_general_curation(resp, workspace_dir, existing, max_general)


def _execute_general_curation(text, workspace_dir, existing, max_general):
    """Parse general curator decisions."""
    existing_names = {n for n, _, _ in existing}
    count = len(existing)
    stats = {"added": 0, "updated": 0, "deleted": 0}

    def _write_gen(name, desc, content):
        d = workspace_dir / "skills" / "general" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\n{content}")

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        u = s.upper()

        if u.startswith("NEW_GENERAL:"):
            name = re.sub(r"[^a-z0-9-]", "-", s.split(":", 1)[1].strip().lower()).strip("-")
            desc, content = "", ""
            i += 1
            while i < len(lines):
                sl = lines[i].strip()
                if sl.upper().startswith("DESCRIPTION:"):
                    desc = sl.split(":", 1)[1].strip()[:150]
                elif sl.upper().startswith("CONTENT:"):
                    cl = []
                    i += 1
                    while i < len(lines):
                        su = lines[i].strip().upper()
                        if any(su.startswith(m) for m in ["NEW_GENERAL:", "UPDATE_GENERAL:", "DELETE_GENERAL:", "NO_PATTERNS"]):
                            break
                        cl.append(lines[i])
                        i += 1
                    content = "\n".join(cl).strip()
                    break
                i += 1
            if name and content and count < max_general:
                _write_gen(name, desc, content)
                count += 1
                stats["added"] += 1
            continue

        elif u.startswith("UPDATE_GENERAL:"):
            raw = s.split(":", 1)[1].strip()
            name = re.sub(r"[^a-z0-9-]", "-", raw.lower()).strip("-")
            matched = next((n for n in existing_names if name == n or name in n or n in name), None)
            content = ""
            i += 1
            while i < len(lines):
                sl = lines[i].strip()
                if sl.upper().startswith("NEW_CONTENT:") or sl.upper().startswith("CONTENT:"):
                    cl = []
                    i += 1
                    while i < len(lines):
                        su = lines[i].strip().upper()
                        if any(su.startswith(m) for m in ["NEW_GENERAL:", "UPDATE_GENERAL:", "DELETE_GENERAL:", "NO_PATTERNS"]):
                            break
                        cl.append(lines[i])
                        i += 1
                    content = "\n".join(cl).strip()
                    break
                i += 1
            if matched and content:
                old_desc = next((d for n, d, _ in existing if n == matched), "")
                _write_gen(matched, old_desc, content)
                stats["updated"] += 1
            continue

        elif u.startswith("DELETE_GENERAL:"):
            raw = s.split(":", 1)[1].strip()
            name = re.sub(r"[^a-z0-9-]", "-", raw.lower()).strip("-")
            matched = next((n for n in existing_names if name == n or name in n or n in name), None)
            if matched:
                d = workspace_dir / "skills" / "general" / matched
                if d.exists():
                    shutil.rmtree(d)
                    existing_names.discard(matched)
                    count -= 1
                    stats["deleted"] += 1

        i += 1
    return stats


# ---------------------------------------------------------------------------
# Agent curator functions
# ---------------------------------------------------------------------------

def _workspace_bash(command: str, workspace_dir: Path, timeout: int = 30) -> str:
    """Execute a bash command scoped to the workspace directory."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=str(workspace_dir),
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output[:4000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s"
    except Exception as e:
        return f"ERROR: {str(e)[:200]}"


def _run_agent_loop(
    system_prompt: str,
    user_prompt: str,
    workspace_dir: Path,
    region: str,
    model_id: str,
    max_rounds: int = 15,
    label: str = "agent",
) -> dict:
    """Shared agentic loop: LLM + bash tool, multi-turn conversation."""
    messages = [{"role": "user", "content": [{"text": user_prompt}]}]

    client = _get_client(region)
    total_input = 0
    total_output = 0
    tool_calls = 0
    final_text = ""
    round_i = 0

    for round_i in range(max_rounds):
        try:
            resp = client.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=messages,
                toolConfig={"tools": [AGENT_BASH_TOOL]},
                inferenceConfig={"maxTokens": 4096, "temperature": 0.3},
            )
        except Exception as e:
            err_str = str(e)
            logger.warning("%s converse failed (round %d): %s", label, round_i, err_str[:200])
            if "throttl" in err_str.lower():
                time.sleep(5 * (round_i + 1))
                continue
            break

        usage = resp.get("usage", {})
        total_input += usage.get("inputTokens", 0)
        total_output += usage.get("outputTokens", 0)

        assistant_content = resp.get("output", {}).get("message", {}).get("content", [])
        stop_reason = resp.get("stopReason", "")

        messages.append({"role": "assistant", "content": assistant_content})

        text_parts = []
        tool_uses = []
        for block in assistant_content:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tool_uses.append(block["toolUse"])
        if text_parts:
            final_text = "\n".join(text_parts)

        if not tool_uses or stop_reason == "end_turn":
            break

        tool_results = []
        for tu in tool_uses:
            tool_calls += 1
            cmd = tu.get("input", {}).get("command", "")
            logger.debug("%s bash [%d]: %s", label, tool_calls, cmd[:120])
            output = _workspace_bash(cmd, workspace_dir)
            tool_results.append({
                "toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": [{"text": output}],
                }
            })
        messages.append({"role": "user", "content": tool_results})

    logger.info("%s done: %d rounds, %d bash calls, %d/%d tokens",
                label, round_i + 1, tool_calls, total_input, total_output)
    return {
        "text": final_text,
        "tool_calls": tool_calls,
        "rounds": round_i + 1,
        "input_tokens": total_input,
        "output_tokens": total_output,
    }


def _build_curator_agent_prompt(
    proposals: list[dict],
    batch_results: list[dict],
    workspace_dir: Path,
    evolve_all: bool = False,
) -> str:
    """Build user prompt for the agent curator with all proposals."""
    passed = sum(1 for r in batch_results if r.get("passed"))
    failed = len(batch_results) - passed

    parts = [f"## Skill Proposals ({len(proposals)} proposals from "
             f"{len(batch_results)} tasks, {passed} passed, {failed} failed)\n"]

    for i, p in enumerate(proposals, 1):
        status = "PASS" if p.get("source_passed") else "FAIL"
        parts.append(
            f"### Proposal {i}: {p.get('name', '?')}\n"
            f"Source: {p.get('source_task', '?')} [{status}]\n"
            f"Solver topic: {p.get('topic', '?')}\n"
            f"Action: {p.get('action', 'NEW')}\n"
            f"Description: {p.get('description', '')[:150]}\n"
            f"Content:\n{_truncate(p.get('content', ''), 400)}\n"
        )

    # Current skill library summary
    skills = load_skills(workspace_dir)
    seed_skills = [s for s in skills if "seed/" in s.path]
    evolved_skills = [s for s in skills if "evolved/" in s.path]

    parts.append("\n## Current Skill Library")
    if seed_skills:
        parts.append(f"\n**Seed skills** ({len(seed_skills)}, read-only):")
        for s in seed_skills:
            parts.append(f"- {s.name}: {s.description}")
    if evolved_skills:
        parts.append(f"\n**Evolved skills** ({len(evolved_skills)}):")
        for s in evolved_skills:
            parts.append(f"- {s.name}: {s.description}")
    else:
        parts.append("\n**Evolved skills**: (none yet)")

    parts.append(
        "\n## Instructions\n"
        "1. `ls skills/evolved/` to see current evolved skills\n"
        "2. Read any existing skills you want to update\n"
        "3. Merge similar proposals into broad domain skills\n"
        "4. Write skills: `mkdir -p skills/evolved/<name> && "
        "cat > skills/evolved/<name>/SKILL.md << 'SKILL'`\n"
        "5. Summarize all changes\n"
    )

    return "\n".join(parts)


def _run_agent_curator(
    proposals: list[dict],
    batch_results: list[dict],
    workspace_dir: Path,
    region: str,
    model_id: str,
    max_skills: int = 10,
    evolve_all: bool = False,
    max_rounds: int = 10,
) -> dict:
    """Run the agent curator: merge proposals + write skills via bash tool."""
    system = AGENT_CURATOR_SYSTEM.format(max_skills=max_skills)
    user = _build_curator_agent_prompt(proposals, batch_results, workspace_dir, evolve_all)
    return _run_agent_loop(system, user, workspace_dir, region, model_id,
                           max_rounds=max_rounds, label="curator")


# ---------------------------------------------------------------------------
# Skill tree
# ---------------------------------------------------------------------------

def update_skill_tree(workspace_dir: Path):
    """Regenerate SKILL_TREE.md."""
    skills = load_skills(workspace_dir)
    if not skills:
        content = "# Skill Tree\n\nNo skills yet.\n"
    else:
        lines = ["# Skill Tree", "", f"Total: {len(skills)}", ""]
        by_prefix = defaultdict(list)
        for s in skills:
            parts = Path(s.path).parts
            prefix = parts[1] if len(parts) > 2 else "root"
            by_prefix[prefix].append(s)
        for prefix in sorted(by_prefix):
            lines.append(f"## {prefix}/")
            for s in sorted(by_prefix[prefix], key=lambda x: x.name):
                lines.append(f"- **{s.name}**: {s.description}")
            lines.append("")
        content = "\n".join(lines)
    (workspace_dir / "skills" / "SKILL_TREE.md").write_text(content)


# ---------------------------------------------------------------------------
# Batch evolve loop
# ---------------------------------------------------------------------------

def evolve_batch(
    tasks: list[dict],
    workspace_dir: Path,
    model_id: str,
    curator_model: str,
    region: str,
    max_tokens: int,
    batch_workers: int,
    max_skills_per_topic: int,
    max_general_skills: int,
    log_dir: Path,
    results_dir: Path,
    provider: str = "aws",
    max_steps: int = 30,
    batch_label: str = "",
    no_evolve: bool = False,
    evolve_all: bool = False,
    lazy_load: bool = False,
    selector_model: str = "",
    enable_proxy: bool = False,
    agent_curate: bool = False,
    feedback_level: str = "standard",
) -> list[dict]:
    """Run one batch: parallel solve -> curate -> update skills."""

    # Load current skills (always load -- seed skills should be available even in no_evolve)
    skills = load_skills(workspace_dir)

    # Base system prompt (used as-is for lazy_load, or as fallback for full injection)
    system_prompt = build_system_prompt(skills, lazy_load=lazy_load)

    # Parallel solve — one DesktopEnv per worker, reused across tasks via env.reset()
    logger.info("[%s] Solving %d tasks with %d workers...", batch_label, len(tasks), batch_workers)

    task_queue: queue_mod.Queue = queue_mod.Queue()
    for t in tasks:
        task_queue.put(t)

    task_outputs: dict[str, dict] = {}
    _outputs_lock = threading.Lock()

    def _worker(worker_idx: int):
        """Worker thread: create one DesktopEnv, pull tasks from queue."""
        from desktop_env.desktop_env import DesktopEnv

        env = None
        try:
            screen_size = (1920, 1080)
            env_kwargs = dict(
                provider_name=provider,
                region=region,
                os_type="Ubuntu",
                action_space="claude_computer_use",
                screen_size=screen_size,
                require_a11y_tree=False,
                enable_proxy=enable_proxy,
            )
            if provider == "aws":
                from desktop_env.providers.aws.manager import IMAGE_ID_MAP
                ami_id = IMAGE_ID_MAP[region].get(
                    screen_size, IMAGE_ID_MAP[region][(1920, 1080)]
                )
                env_kwargs["snapshot_name"] = ami_id
            env = DesktopEnv(**env_kwargs)
            with _live_envs_lock:
                _live_envs.append(env)
            logger.info("[%s][worker-%d] DesktopEnv created", batch_label, worker_idx)

            while True:
                try:
                    t = task_queue.get_nowait()
                except queue_mod.Empty:
                    break
                tid = _task_id(t)
                try:
                    out = _solve_one_task(
                        task_config=t, env=env,
                        model_id=model_id, region=region,
                        max_tokens=max_tokens, system_prompt=system_prompt,
                        skills=skills, workspace_dir=workspace_dir,
                        log_dir=log_dir, max_steps=max_steps,
                        do_propose=not no_evolve,
                        evolve_all=evolve_all,
                        lazy_load=lazy_load,
                        selector_model=selector_model,
                        curator_model=curator_model,
                        feedback_level=feedback_level,
                    )
                except Exception as e:
                    logger.error("[worker-%d] Task %s failed: %s", worker_idx, tid, e)
                    out = {
                        "task_name": tid,
                        "domain": "unknown",
                        "passed": False, "score": 0.0, "error": str(e),
                        "proposal": None, "feedback_analysis": None,
                    }
                with _outputs_lock:
                    task_outputs[tid] = out
        except Exception as e:
            logger.error("[worker-%d] Worker-level error: %s", worker_idx, e)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
                with _live_envs_lock:
                    try:
                        _live_envs.remove(env)
                    except ValueError:
                        pass
                logger.info("[%s][worker-%d] DesktopEnv closed", batch_label, worker_idx)

    threads = []
    for i in range(min(batch_workers, len(tasks))):
        t = threading.Thread(target=_worker, args=(i,), name=f"osw-worker-{i}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Collect results
    results = []
    proposals = []
    failed_summaries = []

    for t in tasks:
        tid = _task_id(t)
        out = task_outputs.get(tid, {})
        passed = out.get("passed", False)
        results.append(out)

        # Save per-task result
        (results_dir / f"{tid}.json").write_text(json.dumps(out, indent=2, default=str))

        if out.get("proposal"):
            out["proposal"]["source_passed"] = passed
            proposals.append(out["proposal"])

        if not passed:
            proposal_summary = ""
            if out.get("proposal"):
                p = out["proposal"]
                proposal_summary = f"[{p.get('action', 'NEW')}] {p.get('name', '')}: {p.get('description', '')}"
            ed = out.get("eval_detail", {})
            failed_summaries.append({
                "task_name": tid,
                "domain": out.get("domain", "unknown"),
                "feedback_analysis": out.get("feedback_analysis", ""),
                "proposal_summary": proposal_summary,
                "trajectory_signals": out.get("trajectory_signals"),
                "compressed_trajectory": out.get("compressed_trajectory", ""),
                "eval_metric": ed.get("metric_func", ""),
                "failure_reason": ed.get("failure_reason", ""),
                "bot_detection": ed.get("bot_detection"),
            })

    passed_count = sum(1 for r in results if r.get("passed"))
    logger.info("[%s] %d/%d passed, %d proposals", batch_label, passed_count, len(tasks), len(proposals))

    if no_evolve:
        return results

    if agent_curate:
        # Agent curator: per-task proposals + single agent merges & writes via bash
        if proposals:
            curator_result = _run_agent_curator(
                proposals=proposals,
                batch_results=results,
                workspace_dir=workspace_dir,
                region=region,
                model_id=curator_model,
                max_skills=max_skills_per_topic,
                evolve_all=evolve_all,
            )
            logger.info("[%s] Agent curator: %d bash calls in %d rounds, %d proposals",
                         batch_label,
                         curator_result.get("tool_calls", 0),
                         curator_result.get("rounds", 0),
                         len(proposals))
        else:
            logger.info("[%s] No proposals to curate", batch_label)
        update_skill_tree(workspace_dir)
        new_skills = load_skills(workspace_dir)
        logger.info("[%s] Skills: %d total", batch_label, len(new_skills))
        return results

    # Curate per topic (self-assigned by proposer)
    if proposals:
        topic_proposals = defaultdict(list)
        for p in proposals:
            topic_proposals[p.get("topic", "general")].append(p)

        total_stats = {"added": 0, "merged": 0, "skipped": 0}
        for topic, topic_props in topic_proposals.items():
            stats = _curate_topic_proposals(
                topic, topic_props, workspace_dir, region, curator_model, max_skills_per_topic,
            )
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

        logger.info("[%s] Topic curation: +%d added, %d merged, %d skipped",
                     batch_label, total_stats["added"], total_stats["merged"], total_stats["skipped"])

    # General curator
    if max_general_skills > 0 and len(failed_summaries) >= 2:
        gen_stats = _curate_general_skills(
            failed_summaries, workspace_dir, region, curator_model, max_general_skills,
            feedback_level=feedback_level,
        )
        logger.info("[%s] General curator: +%d added, %d updated, %d deleted",
                     batch_label, gen_stats["added"], gen_stats["updated"], gen_stats["deleted"])

    # Update skill tree
    update_skill_tree(workspace_dir)
    new_skills = load_skills(workspace_dir)
    logger.info("[%s] Skills: %d total", batch_label, len(new_skills))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OSWorld with A-EVOLVE propose+curator")
    p.add_argument("--task-file", type=str, required=True,
                   help="Path to test_all.json or similar OSWorld task file")
    p.add_argument("--provider", type=str, default="aws",
                   choices=["aws", "vmware", "docker"],
                   help="OSWorld VM provider (default: aws)")
    p.add_argument("--domain", type=str, default=None,
                   help="Filter by domain (e.g. os, chrome, libreoffice_calc, gimp, vlc)")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task IDs to run")
    p.add_argument("--exclude", type=str, default=None,
                   help="Comma-separated task IDs to exclude")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="Random seed for shuffle (default: 42)")
    p.add_argument("--solver-model", type=str, default="1")
    p.add_argument("--curator-model", type=str, default="2")
    p.add_argument("--selector-model", type=str, default="2",
                   help="Model for topic selection (default: 2=Sonnet). Set empty to disable.")
    p.add_argument("--region", type=str, default="us-west-2")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=30,
                   help="Max turns per task (default: 30)")
    p.add_argument("--max-skills-per-topic", type=int, default=5)
    p.add_argument("--max-general-skills", type=int, default=10)
    p.add_argument("--no-evolve", action="store_true",
                   help="Baseline mode: skip propose + curation, just solve + evaluate")
    p.add_argument("--evolve-all", action="store_true",
                   help="Propose skills for ALL tasks (passed + failed), not just failures")
    p.add_argument("--lazy-load", action="store_true",
                   help="Lazy-load skills: name+desc in prompt, full body via read_skill tool")
    p.add_argument("--no-seed-skills", action="store_true",
                   help="Skip copying seed skills (for pure baseline without any skills)")
    p.add_argument("--seed-workspace", type=str, default=None,
                   help="Seed workspace to copy (default: built-in osworld seed)")
    p.add_argument("--enable-proxy", action="store_true",
                   help="Enable proxy for VM (requires DataImpulse proxy config in OSWorld)")
    p.add_argument("--agent-curate", action="store_true",
                   help="Use agent curator: flat skills/evolved/ structure, "
                        "max-skills-per-topic = total evolved skill limit")
    p.add_argument("--feedback-level", type=str, default="standard",
                   choices=["minimal", "standard", "full"],
                   help="How much eval detail the evolver sees")
    p.add_argument("--output-dir", type=str, default="outputs/osworld_evolve")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for n in ("botocore", "urllib3", "httpcore", "httpx"):
        logging.getLogger(n).setLevel(logging.WARNING)

    # Resolve models
    model_id = MODEL_MAP.get(args.solver_model, args.solver_model)
    curator_model_id = MODEL_MAP.get(args.curator_model, args.curator_model)
    selector_model_id = MODEL_MAP.get(args.selector_model, args.selector_model) if args.selector_model else ""

    # Load tasks
    all_tasks = load_osworld_tasks(args.task_file, domain=args.domain)
    logger.info("Loaded %d tasks from %s (domain=%s)", len(all_tasks), args.task_file, args.domain)

    if args.tasks:
        ids = set(n.strip() for n in args.tasks.split(","))
        all_tasks = [t for t in all_tasks if _task_id(t) in ids]
    if args.exclude:
        excl = set(n.strip() for n in args.exclude.split(","))
        all_tasks = [t for t in all_tasks if _task_id(t) not in excl]
    if args.shuffle:
        import random
        random.seed(args.shuffle_seed)
        random.shuffle(all_tasks)
    if args.limit:
        all_tasks = all_tasks[:args.limit]

    if not all_tasks:
        print("No tasks to run.")
        return

    # Setup workspace
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = output_dir / "workspace"

    if not workspace_dir.exists():
        workspace_dir.mkdir(parents=True, exist_ok=True)
        if args.seed_workspace and Path(args.seed_workspace).exists():
            shutil.copytree(args.seed_workspace, workspace_dir, dirs_exist_ok=True)
            logger.info("Copied seed workspace from %s", args.seed_workspace)
        else:
            # Minimal workspace
            (workspace_dir / "skills").mkdir(exist_ok=True)
            (workspace_dir / "skills" / "topic").mkdir(exist_ok=True)
            (workspace_dir / "skills" / "general").mkdir(exist_ok=True)
            (workspace_dir / "skills" / "evolved").mkdir(exist_ok=True)
        # Always init seed skills
        if not args.no_seed_skills:
            _init_seed_skills(workspace_dir)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    logger.info(
        "Running %d tasks | solver=%s curator=%s | provider=%s | batch=%d workers=%d | "
        "max_steps=%d max_topic_skills=%d max_general=%d | run_id=%s",
        len(all_tasks), model_id, curator_model_id, args.provider,
        args.batch_size, args.workers, args.max_steps,
        args.max_skills_per_topic, args.max_general_skills, _RUN_ID,
    )

    # Pre-run: clean up orphan instances from previous crashed runs
    if args.provider == "aws":
        cleanup_orphan_instances(region=args.region, max_age_hours=6)

    # Batch loop
    all_results = []
    t0 = time.time()
    batches = [all_tasks[i:i+args.batch_size] for i in range(0, len(all_tasks), args.batch_size)]

    for bi, batch in enumerate(batches):
        logger.info("=== Batch %d/%d (%d tasks) ===", bi+1, len(batches), len(batch))
        batch_results = evolve_batch(
            tasks=batch, workspace_dir=workspace_dir,
            model_id=model_id, curator_model=curator_model_id,
            region=args.region, max_tokens=args.max_tokens,
            batch_workers=args.workers,
            max_skills_per_topic=args.max_skills_per_topic,
            max_general_skills=args.max_general_skills,
            log_dir=log_dir, results_dir=results_dir,
            provider=args.provider, max_steps=args.max_steps,
            batch_label=f"B{bi+1}/{len(batches)}",
            no_evolve=args.no_evolve,
            evolve_all=args.evolve_all,
            lazy_load=args.lazy_load,
            selector_model=selector_model_id,
            enable_proxy=args.enable_proxy,
            agent_curate=args.agent_curate,
            feedback_level=args.feedback_level,
        )
        all_results.extend(batch_results)

        passed = sum(1 for r in all_results if r.get("passed"))
        total = len(all_results)
        logger.info("Cumulative: %d/%d (%.1f%%)", passed, total, 100 * passed / max(total, 1))

    # Final summary
    elapsed = time.time() - t0
    total_passed = sum(1 for r in all_results if r.get("passed"))
    total = len(all_results)

    by_domain = defaultdict(lambda: {"p": 0, "t": 0})
    for r in all_results:
        dom = r.get("domain", "unknown")
        by_domain[dom]["t"] += 1
        if r.get("passed"):
            by_domain[dom]["p"] += 1

    logger.info("=" * 70)
    logger.info("FINAL: %d/%d (%.1f%%) in %.0fs", total_passed, total,
                100 * total_passed / max(total, 1), elapsed)
    for dom, d in sorted(by_domain.items()):
        logger.info("  %s: %d/%d (%.1f%%)", dom, d["p"], d["t"], 100*d["p"]/max(d["t"],1))

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "solver_model": model_id,
        "curator_model": curator_model_id,
        "provider": args.provider,
        "total": total,
        "passed": total_passed,
        "rate": total_passed / max(total, 1),
        "elapsed": elapsed,
        "by_domain": {k: dict(v) for k, v in by_domain.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Save all results
    with open(output_dir / "all_results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    # Copy final skills
    skills_out = output_dir / "final_skills"
    if (workspace_dir / "skills").exists():
        if skills_out.exists():
            shutil.rmtree(skills_out)
        shutil.copytree(workspace_dir / "skills", skills_out)

    # Final safety-net cleanup: terminate any instances from this run still alive
    _cleanup_all_envs()
    if args.provider == "aws":
        _cleanup_tagged_instances(region=args.region)


if __name__ == "__main__":
    main()
