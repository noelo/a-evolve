"""OSWorld solver using Claude's native computer_use tool via AnthropicBedrock.

Matches the official OSWorld Anthropic agent (mm_agents/anthropic/) but uses
AnthropicBedrock for API calls. Key features:
- Native computer_20251124 tool (structured actions: click, type, key, scroll)
- Screenshot scaling: 1920x1080 -> 1280x720
- Coordinate upscaling: model coords x resize_factor -> real screen coords
- Image truncation: keep only N most recent screenshots
- Extended thinking: enabled for non-Opus-4.6 models (budget_tokens=2048)
- Optional read_skill tool for lazy-loaded skill library
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# Fixed computer_use version — matches official OSWorld agent (all models use the same)
COMPUTER_USE_TYPE = "computer_20251124"
COMPUTER_USE_BETA_FLAG = "computer-use-2025-11-24"

# Model sends coordinates in this space; screenshots are resized to this.
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Actual VM screen size.
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

API_RETRY_TIMES = 500
API_RETRY_INTERVAL = 5
MAX_FORMAT_RETRIES = 3  # retry when LLM returns no tool_use (format error)

# ── System prompt (from official OSWorld Anthropic agent) ─────────────

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using available tools.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* TASK FEASIBILITY: You can declare a task infeasible at any point during execution - whether at the beginning after taking a screenshot, or later after attempting some actions and discovering barriers. Carefully evaluate whether the task is feasible given the current system state, available applications, and task requirements. If you determine that a task cannot be completed due to:
  - Missing required applications or dependencies that cannot be installed
  - Insufficient permissions or system limitations
  - Contradictory or impossible requirements
  - Any other fundamental barriers that make completion impossible
  Then you MUST output exactly "[INFEASIBLE]" (including the square brackets) anywhere in your response to trigger the fail action. The system will automatically detect this pattern and terminate the task appropriately.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is 'osworld-public-evaluation'.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
</IMPORTANT>"""

CONTINUE_PROMPT = (
    "Please continue with the task. If you have completed the task, "
    "use the computer tool with action='done'. If the task is infeasible, "
    "include [INFEASIBLE] in your response."
)

# ── Tool definitions ──────────────────────────────────────────────────

READ_SKILL_TOOL = {
    "name": "read_skill",
    "description": (
        "Read the full content of a skill by name. "
        "Use this to load detailed guidance for a skill listed in your system prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name to read.",
            }
        },
        "required": ["name"],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────

def _is_opus_4_6(model_id: str) -> bool:
    """Check if model is Opus 4.6 (no extended thinking support)."""
    return "opus-4-6" in model_id or "opus-4.6" in model_id


def _to_png_bytes(screenshot, log: logging.Logger) -> bytes | None:
    """Convert a screenshot (raw bytes, numpy, or PIL) to PNG bytes."""
    if screenshot is None:
        return None
    if isinstance(screenshot, (bytes, bytearray)):
        return bytes(screenshot)
    try:
        from PIL import Image
        import numpy as np
        if isinstance(screenshot, np.ndarray):
            img = Image.fromarray(screenshot)
        elif isinstance(screenshot, Image.Image):
            img = screenshot
        else:
            return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        log.debug("Screenshot conversion failed: %s", e)
        return None


def _resize_screenshot(png_bytes: bytes) -> bytes:
    """Resize screenshot from screen size (1920x1080) to display size (1280x720)."""
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes))
    if img.size == (DISPLAY_WIDTH, DISPLAY_HEIGHT):
        return png_bytes
    resized = img.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def _get_obs_dict(env, initial_obs, log: logging.Logger) -> dict:
    """Get observation as dict."""
    if initial_obs is not None:
        return initial_obs if isinstance(initial_obs, dict) else {"screenshot": initial_obs}
    try:
        obs = env._get_obs()
        return obs if isinstance(obs, dict) else {}
    except Exception as e:
        log.warning("Failed to get observation: %s", e)
        return {}


# ── Response parsing ──────────────────────────────────────────────────

def _response_to_params(response) -> list[dict]:
    """Convert BetaMessage response content blocks to serializable dicts."""
    params = []
    if not response.content:
        return params
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            if block.text:
                params.append({"type": "text", "text": block.text})
        elif block_type == "thinking":
            tp = {"type": "thinking", "thinking": getattr(block, "thinking", "")}
            if hasattr(block, "signature") and block.signature:
                tp["signature"] = block.signature
            params.append(tp)
        elif block_type == "tool_use":
            # Only keep standard fields — model_dump() may include beta-specific
            # fields like 'caller' that cause 400 errors when sent back.
            tu = {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
            params.append(tu)
        else:
            # For unknown block types, dump but strip None values
            if hasattr(block, "model_dump"):
                dumped = block.model_dump()
                dumped = {k: v for k, v in dumped.items() if v is not None}
                params.append(dumped)
    return params


# ── Image management ──────────────────────────────────────────────────

def _filter_to_n_most_recent_images(
    messages: list[dict], images_to_keep: int, min_removal_threshold: int = 10,
):
    """Remove old screenshots from tool_result blocks, keeping N most recent."""
    if images_to_keep is None:
        return

    tool_result_blocks = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    tool_result_blocks.append(item)

    total_images = sum(
        1 for tr in tool_result_blocks
        for c in tr.get("content", [])
        if isinstance(c, dict) and c.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    images_to_remove -= images_to_remove % min_removal_threshold

    if images_to_remove <= 0:
        return

    for tr in tool_result_blocks:
        if isinstance(tr.get("content"), list):
            new_content = []
            for c in tr["content"]:
                if isinstance(c, dict) and c.get("type") == "image" and images_to_remove > 0:
                    images_to_remove -= 1
                    continue
                new_content.append(c)
            tr["content"] = new_content


# ── Action parsing (from official OSWorld Anthropic agent) ────────────

def parse_actions_from_tool_call(
    tool_input: dict, resize_factor: tuple[float, float],
) -> str:
    """Convert computer_use structured action to pyautogui code string.

    Args:
        tool_input: The "input" field from a computer tool_use block.
                    Contains action, coordinate, text, scroll_direction, etc.
        resize_factor: (width_factor, height_factor) for scaling coordinates
                       from display space (1280x720) to real screen (1920x1080).

    Returns:
        PyAutoGUI code string, or "DONE"/"FAIL"/"CALL_USER" for terminal actions.
    """
    result = ""

    action = tool_input.get("action", "")
    action_conversion = {"left click": "click", "right click": "right_click"}
    action = action_conversion.get(action, action)

    text = tool_input.get("text")
    coordinate = tool_input.get("coordinate")
    start_coordinate = tool_input.get("start_coordinate")
    scroll_direction = tool_input.get("scroll_direction")
    scroll_amount = tool_input.get("scroll_amount")
    duration = tool_input.get("duration")

    # Scale coordinates from display space to real screen space
    if coordinate and resize_factor:
        coordinate = (
            int(coordinate[0] * resize_factor[0]),
            int(coordinate[1] * resize_factor[1]),
        )
    if start_coordinate and resize_factor:
        start_coordinate = (
            int(start_coordinate[0] * resize_factor[0]),
            int(start_coordinate[1] * resize_factor[1]),
        )

    # ── Mouse button state ────────────────────────────────────────
    if action == "left_mouse_down":
        result += "pyautogui.mouseDown()\n"
    elif action == "left_mouse_up":
        result += "pyautogui.mouseUp()\n"

    # ── Hold key ──────────────────────────────────────────────────
    elif action == "hold_key":
        if not isinstance(text, str):
            raise ValueError(f"text must be a string for hold_key, got {text}")
        for key in text.split("+"):
            result += f"pyautogui.keyDown('{key.strip().lower()}')\n"

    # ── Mouse move / drag ─────────────────────────────────────────
    elif action in ("mouse_move", "left_click_drag"):
        if coordinate is None:
            raise ValueError(f"coordinate is required for {action}")
        x, y = coordinate
        if action == "mouse_move":
            result += f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"
        elif action == "left_click_drag":
            if start_coordinate:
                sx, sy = start_coordinate
                result += f"pyautogui.moveTo({sx}, {sy}, duration={duration or 0.5})\n"
            result += f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"

    # ── Keyboard ──────────────────────────────────────────────────
    elif action in ("key", "type"):
        if text is None:
            raise ValueError(f"text is required for {action}")
        if action == "key":
            key_conversion = {
                "page_down": "pagedown", "page_up": "pageup",
                "super_l": "win", "super": "command", "escape": "esc",
            }
            keys = text.split("+")
            for key in keys:
                k = key_conversion.get(key.strip().lower(), key.strip().lower())
                result += f"pyautogui.keyDown('{k}')\n"
            for key in reversed(keys):
                k = key_conversion.get(key.strip().lower(), key.strip().lower())
                result += f"pyautogui.keyUp('{k}')\n"
        elif action == "type":
            for char in text:
                if char == "\n":
                    result += "pyautogui.press('enter')\n"
                elif char == "'":
                    result += 'pyautogui.press("\'")\n'
                elif char == "\\":
                    result += "pyautogui.press('\\\\')\n"
                elif char == '"':
                    result += 'pyautogui.press(\'"\')\n'
                else:
                    result += f"pyautogui.press('{char}')\n"

    # ── Scroll ────────────────────────────────────────────────────
    elif action == "scroll":
        if text is not None:
            result += f"pyautogui.keyDown('{text.lower()}')\n"
        if coordinate is None:
            if scroll_direction in ("up", "down"):
                amt = scroll_amount if scroll_direction == "up" else -scroll_amount
                result += f"pyautogui.scroll({amt})\n"
            elif scroll_direction in ("left", "right"):
                amt = scroll_amount if scroll_direction == "right" else -scroll_amount
                result += f"pyautogui.hscroll({amt})\n"
        else:
            x, y = coordinate
            if scroll_direction in ("up", "down"):
                amt = scroll_amount if scroll_direction == "up" else -scroll_amount
                result += f"pyautogui.scroll({amt}, {x}, {y})\n"
            elif scroll_direction in ("left", "right"):
                amt = scroll_amount if scroll_direction == "right" else -scroll_amount
                result += f"pyautogui.hscroll({amt}, {x}, {y})\n"
        if text is not None:
            result += f"pyautogui.keyUp('{text.lower()}')\n"

    # ── Click actions ─────────────────────────────────────────────
    elif action in ("left_click", "right_click", "double_click", "middle_click",
                     "left_press", "triple_click"):
        # Modifier keys during click
        if text:
            for key in text.split("+"):
                result += f"pyautogui.keyDown('{key.strip().lower()}')\n"
        if coordinate is not None:
            x, y = coordinate
            click_map = {
                "left_click": f"pyautogui.click({x}, {y})\n",
                "right_click": f"pyautogui.rightClick({x}, {y})\n",
                "double_click": f"pyautogui.doubleClick({x}, {y})\n",
                "middle_click": f"pyautogui.middleClick({x}, {y})\n",
                "triple_click": f"pyautogui.tripleClick({x}, {y})\n",
                "left_press": (
                    f"pyautogui.mouseDown({x}, {y})\n"
                    f"time.sleep(1)\n"
                    f"pyautogui.mouseUp({x}, {y})\n"
                ),
            }
            result += click_map.get(action, f"pyautogui.click({x}, {y})\n")
        else:
            no_coord_map = {
                "left_click": "pyautogui.click()\n",
                "right_click": "pyautogui.rightClick()\n",
                "double_click": "pyautogui.doubleClick()\n",
                "middle_click": "pyautogui.middleClick()\n",
                "triple_click": "pyautogui.tripleClick()\n",
                "left_press": (
                    "pyautogui.mouseDown()\n"
                    "time.sleep(1)\n"
                    "pyautogui.mouseUp()\n"
                ),
            }
            result += no_coord_map.get(action, "pyautogui.click()\n")
        # Release modifier keys
        if text:
            for key in reversed(text.split("+")):
                result += f"pyautogui.keyUp('{key.strip().lower()}')\n"

    # ── Special actions ───────────────────────────────────────────
    elif action == "wait":
        result += "pyautogui.sleep(0.5)\n"
    elif action == "fail":
        return "FAIL"
    elif action == "done":
        return "DONE"
    elif action == "call_user":
        return "CALL_USER"
    elif action == "screenshot":
        result += "pyautogui.sleep(0.1)\n"
    else:
        raise ValueError(f"Invalid action: {action}")

    return result


# ── Conversation extraction ──────────────────────────────────────────

def extract_conversation(messages: list[dict]) -> list[dict]:
    """Convert Anthropic Messages format to standardized format for analysis.

    Strips images and thinking blocks, keeps text, tool_use, and tool_result.
    """
    conversation = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content_blocks = msg.get("content", [])
        if isinstance(content_blocks, str):
            content_blocks = [{"type": "text", "text": content_blocks}]

        entry = {"role": role, "parts": []}
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            bt = block.get("type", "")

            if bt == "text":
                entry["parts"].append({"type": "text", "text": block.get("text", "")})
            elif bt == "tool_use":
                entry["parts"].append({
                    "type": "tool_use",
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                    "id": block.get("id", ""),
                })
            elif bt == "tool_result":
                result_text = ""
                for c in block.get("content", []):
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            t = c.get("text", "")
                            if len(t) > 3000:
                                t = t[:1500] + "\n...[truncated]...\n" + t[-1500:]
                            result_text += t + "\n"
                        elif c.get("type") == "image":
                            result_text += "[screenshot]\n"
                entry["parts"].append({
                    "type": "tool_result",
                    "text": result_text.strip(),
                    "id": block.get("tool_use_id", ""),
                })
            # Skip thinking and image blocks

        conversation.append(entry)
    return conversation


# ── Result class ─────────────────────────────────────────────────────

class ReactSolverResult:
    """Result from the solver."""

    def __init__(self):
        self.messages: list[dict] = []
        self.submitted: bool = False
        self.submit_answer: str = ""
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.tool_call_count: int = 0
        self.timed_out: bool = False
        self.final_reward: float = 0.0


# ── Main solver ──────────────────────────────────────────────────────

def react_solve(
    task_prompt: str,
    env,
    model_id: str = "us.anthropic.claude-opus-4-6-v1",
    region: str = "us-west-2",
    max_tokens: int = 4096,
    timeout_sec: int = 900,
    max_turns: int = 100,
    log: logging.Logger | None = None,
    system_prompt: str | None = None,
    skills: dict[str, str] | None = None,
    initial_obs: dict | None = None,
) -> ReactSolverResult:
    """Run the solver loop for one OSWorld task.

    Uses AnthropicBedrock client with native computer_use tool, matching
    the official OSWorld Anthropic agent implementation.
    """
    from anthropic import AnthropicBedrock, APIError, APIStatusError, APIResponseValidationError

    if log is None:
        log = logger

    result = ReactSolverResult()
    resize_factor = (SCREEN_WIDTH / DISPLAY_WIDTH, SCREEN_HEIGHT / DISPLAY_HEIGHT)

    # ── Client ────────────────────────────────────────────────────
    client = AnthropicBedrock(
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region=region,
    )

    # ── Computer use version (fixed, matches official) ─────────────
    computer_type = COMPUTER_USE_TYPE
    computer_beta = COMPUTER_USE_BETA_FLAG
    log.info("Computer use: type=%s beta=%s", computer_type, computer_beta)

    # ── Thinking mode ─────────────────────────────────────────────
    use_thinking = not _is_opus_4_6(model_id)
    betas = [computer_beta]
    if use_thinking:
        budget_tokens = 2048
        extra_body: dict = {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
        actual_max_tokens = max(max_tokens, budget_tokens + 500)
        log.info("Thinking: ENABLED (budget=%d, max_tokens=%d)", budget_tokens, actual_max_tokens)
    else:
        extra_body = {}
        actual_max_tokens = max_tokens
        log.info("Thinking: ADAPTIVE (Opus 4.6 default, no explicit param needed)")

    # ── System prompt ─────────────────────────────────────────────
    sys_prompt = [{"type": "text", "text": system_prompt or SYSTEM_PROMPT}]

    # ── Tools ─────────────────────────────────────────────────────
    tools: list[dict] = [
        {
            "name": "computer",
            "type": computer_type,
            "display_width_px": DISPLAY_WIDTH,
            "display_height_px": DISPLAY_HEIGHT,
            "display_number": 1,
        }
    ]
    if skills:
        tools.append(READ_SKILL_TOOL)

    # ── Image management ──────────────────────────────────────────
    only_n_most_recent_images = 10
    min_removal_threshold = 10

    # ── Initial observation ───────────────────────────────────────
    obs = _get_obs_dict(env, initial_obs, log)
    screenshot_bytes = _to_png_bytes(obs.get("screenshot"), log)

    messages: list[dict] = []
    first_content: list[dict] = []
    if screenshot_bytes:
        resized = _resize_screenshot(screenshot_bytes)
        first_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(resized).decode("utf-8"),
            },
        })
    first_content.append({"type": "text", "text": task_prompt})
    messages.append({"role": "user", "content": first_content})
    result.messages = messages

    t0 = time.time()
    step = 0
    consecutive_no_tool = 0  # track consecutive responses with no tool_use

    for step in range(max_turns):
        # ── Timeout check ─────────────────────────────────────────
        if time.time() - t0 >= timeout_sec:
            result.timed_out = True
            log.warning("Timeout (%.0fs >= %ds)", time.time() - t0, timeout_sec)
            break

        # ── Image truncation ──────────────────────────────────────
        _filter_to_n_most_recent_images(
            messages, only_n_most_recent_images, min_removal_threshold,
        )

        # ── API call with retries ─────────────────────────────────
        response = None
        for attempt in range(API_RETRY_TIMES):
            try:
                response = client.beta.messages.create(
                    max_tokens=actual_max_tokens,
                    messages=messages,
                    model=model_id,
                    system=sys_prompt,
                    tools=tools,
                    betas=betas,
                    extra_body=extra_body,
                )
                break
            except (APIError, APIStatusError, APIResponseValidationError) as e:
                error_msg = str(e)
                log.warning(
                    "API error (attempt %d/%d): %s",
                    attempt + 1, API_RETRY_TIMES, error_msg[:200],
                )
                # Auto-reduce images on 413 / payload too large
                if any(kw in error_msg for kw in [
                    "25000000", "request_too_large", "413",
                    "Member must have length less than or equal to",
                    "maximum size",
                ]):
                    only_n_most_recent_images = max(1, only_n_most_recent_images // 2)
                    _filter_to_n_most_recent_images(
                        messages, only_n_most_recent_images, min_removal_threshold,
                    )
                    log.info("Reduced images to %d", only_n_most_recent_images)
                if attempt < API_RETRY_TIMES - 1:
                    time.sleep(API_RETRY_INTERVAL)
                else:
                    log.error("All %d API attempts failed", API_RETRY_TIMES)
            except Exception as e:
                log.error("Unexpected API error: %s", str(e)[:300])
                break

        if response is None:
            log.error("No response — aborting")
            break

        # ── Track usage ───────────────────────────────────────────
        if hasattr(response, "usage") and response.usage:
            result.total_input_tokens += getattr(response.usage, "input_tokens", 0) or 0
            result.total_output_tokens += getattr(response.usage, "output_tokens", 0) or 0

        # ── Parse response ────────────────────────────────────────
        response_params = _response_to_params(response)
        messages.append({"role": "assistant", "content": response_params})

        tool_use_blocks = [
            b for b in response_params
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        text_blocks = [
            b for b in response_params
            if isinstance(b, dict) and b.get("type") == "text"
        ]

        if text_blocks:
            log.debug("[step %d] %s", step + 1, text_blocks[0].get("text", "")[:200])

        # ── Check for [INFEASIBLE] ────────────────────────────────
        full_text = " ".join(
            b.get("text", "") or b.get("thinking", "")
            for b in response_params
            if isinstance(b, dict) and b.get("type") in ("text", "thinking")
        )
        if "[INFEASIBLE]" in full_text:
            log.info("Detected [INFEASIBLE]")
            env.step("FAIL", pause=0.5)
            result.submitted = True
            result.submit_answer = "FAIL"
            break

        # ── No tool calls → retry or done ────────────────────────
        if not tool_use_blocks:
            stop = getattr(response, "stop_reason", "end_turn")
            if stop == "max_tokens":
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": CONTINUE_PROMPT}],
                })
                continue

            consecutive_no_tool += 1

            if consecutive_no_tool < MAX_FORMAT_RETRIES:
                # Format retry: LLM didn't return tool_use, nudge it
                log.warning(
                    "[step %d] No tool calls (stop=%s), retrying %d/%d",
                    step + 1, stop, consecutive_no_tool, MAX_FORMAT_RETRIES,
                )
                # Remove the assistant message we just appended (the bad response)
                messages.pop()
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text":
                        "You must respond with a tool_use block (computer tool). "
                        "Do not respond with only text. "
                        "What is your next action?"}],
                })
                continue

            log.info("[step %d] No tool calls (stop=%s) after %d retries — treating as done",
                     step + 1, stop, consecutive_no_tool)
            result.submitted = True
            result.submit_answer = "DONE"
            break

        consecutive_no_tool = 0  # reset on successful tool_use

        # ── Execute tool calls ────────────────────────────────────
        tool_results: list[dict] = []
        task_done = False
        last_obs = None

        for i, block in enumerate(tool_use_blocks):
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            tool_id = block.get("id", "")
            is_last = i == len(tool_use_blocks) - 1
            result.tool_call_count += 1

            if tool_name == "computer":
                # Parse structured action → pyautogui code
                try:
                    code = parse_actions_from_tool_call(tool_input, resize_factor)
                except Exception as exc:
                    log.warning("Action parse error: %s", exc)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "is_error": True,
                    })
                    continue

                # Terminal actions
                if code == "DONE":
                    log.info("[step %d] Agent signaled DONE", step + 1)
                    result.submitted = True
                    result.submit_answer = "DONE"
                    task_done = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [{"type": "text", "text": "Task marked as done."}],
                    })
                    continue
                elif code == "FAIL":
                    log.info("[step %d] Agent signaled FAIL", step + 1)
                    env.step("FAIL", pause=0.5)
                    result.submitted = True
                    result.submit_answer = "FAIL"
                    task_done = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [{"type": "text", "text": "Task marked as failed."}],
                    })
                    continue
                elif code == "CALL_USER":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [{"type": "text", "text": "User not available. Continue on your own."}],
                    })
                    continue

                # Execute pyautogui action
                action_label = tool_input.get("action", "?")
                log.info("[step %d] %s → %s", step + 1, action_label, code.strip()[:120])
                try:
                    obs_new, reward, done, info = env.step(code, pause=3.0)
                    last_obs = obs_new
                    result.final_reward = reward
                except Exception as exc:
                    log.error("env.step error: %s", exc)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [{"type": "text", "text": f"Error executing action: {exc}"}],
                        "is_error": True,
                    })
                    continue

                # Build tool result (screenshot only on last tool_use)
                tr_content: list[dict] = [{"type": "text", "text": "Success"}]
                if is_last and last_obs is not None:
                    ss = _to_png_bytes(
                        last_obs.get("screenshot") if isinstance(last_obs, dict) else None,
                        log,
                    )
                    if ss:
                        resized = _resize_screenshot(ss)
                        tr_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(resized).decode("utf-8"),
                            },
                        })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tr_content,
                })

            elif tool_name == "read_skill":
                sn = tool_input.get("name", "")
                if skills and sn in skills:
                    log.info("[read_skill] %s (%d chars)", sn, len(skills[sn]))
                    tr_text = skills[sn]
                else:
                    avail = ", ".join(skills.keys()) if skills else "none"
                    tr_text = f"Skill '{sn}' not found. Available: {avail}"
                    log.warning("[read_skill] not found: %s", sn)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [{"type": "text", "text": tr_text}],
                })

            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "is_error": True,
                })

        # Append all tool results as one user message
        messages.append({"role": "user", "content": tool_results})

        if task_done:
            log.info("Task done after %d steps (%.0fs)", step + 1, time.time() - t0)
            break

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    result.messages = messages
    log.info(
        "Solver done: %d steps, %d tool calls, %.0fs, "
        "tokens=%d in + %d out, submitted=%s, reward=%.1f",
        step + 1, result.tool_call_count, elapsed,
        result.total_input_tokens, result.total_output_tokens,
        result.submitted, result.final_reward,
    )
    return result
