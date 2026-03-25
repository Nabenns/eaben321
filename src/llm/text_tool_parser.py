"""
Text-based tool call parser for browser LLM providers.

Since browser-based LLMs (ChatGPT / Claude via web UI) don't support structured
tool calling, we include tool definitions in the prompt as plain text and ask the
LLM to respond with a structured pattern:

    TOOL_CALL: {"tool": "<tool_name>", "params": {...}}

This module parses those patterns out of raw LLM response text.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Pattern: TOOL_CALL: {...}
# Allows optional whitespace and newlines inside the JSON block.
_TOOL_CALL_PATTERN = re.compile(
    r"TOOL_CALL\s*:\s*(\{.*?\})",
    re.DOTALL | re.IGNORECASE,
)

# Fallback pattern: ```json\nTOOL_CALL: {...}\n```
_TOOL_CALL_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*\n?TOOL_CALL\s*:\s*(\{.*?\})\s*\n?```",
    re.DOTALL | re.IGNORECASE,
)


class ToolCall:
    """Represents a parsed tool call from LLM text output."""

    def __init__(self, tool_id: str, tool_name: str, params: dict):
        self.id = tool_id          # synthetic id (e.g. "tc_0")
        self.name = tool_name      # e.g. "get_chart"
        self.input = params        # e.g. {"pair": "XAUUSD", "timeframe": "M15"}

    def __repr__(self):
        return f"ToolCall(id={self.id!r}, name={self.name!r}, input={self.input!r})"


def parse_tool_calls(text: str) -> list[ToolCall]:
    """
    Parse all TOOL_CALL patterns from an LLM response string.

    Returns a list of ToolCall objects. Returns empty list if none found
    or if JSON is malformed (with a warning log).

    The LLM is expected to produce output like:

        TOOL_CALL: {"tool": "get_chart", "params": {"pair": "XAUUSD", "timeframe": "M15"}}

    Or optionally inside a code block:

        ```json
        TOOL_CALL: {"tool": "get_chart", "params": {"pair": "XAUUSD"}}
        ```
    """
    results: list[ToolCall] = []

    # First try code-block pattern (more specific)
    matches = list(_TOOL_CALL_CODE_BLOCK_PATTERN.finditer(text))
    if not matches:
        matches = list(_TOOL_CALL_PATTERN.finditer(text))

    for idx, match in enumerate(matches):
        raw_json = match.group(1).strip()
        parsed = _safe_parse_json(raw_json)
        if parsed is None:
            # Try to recover by extracting the JSON more aggressively
            parsed = _try_recover_json(raw_json)
        if parsed is None:
            logger.warning(
                "[TextToolParser] Skipping malformed TOOL_CALL JSON at match %d: %r",
                idx, raw_json[:200]
            )
            continue

        tool_name = parsed.get("tool") or parsed.get("name") or parsed.get("function")
        params = parsed.get("params") or parsed.get("input") or parsed.get("arguments") or {}

        if not tool_name:
            logger.warning(
                "[TextToolParser] TOOL_CALL missing 'tool' key at match %d: %r",
                idx, parsed
            )
            continue

        tool_id = f"tc_{idx}"
        results.append(ToolCall(tool_id=tool_id, tool_name=str(tool_name), params=params))
        logger.debug("[TextToolParser] Parsed tool call: %s(%s)", tool_name, params)

    return results


def has_tool_calls(text: str) -> bool:
    """Quick check: does the text contain any TOOL_CALL pattern?"""
    return bool(
        _TOOL_CALL_CODE_BLOCK_PATTERN.search(text) or
        _TOOL_CALL_PATTERN.search(text)
    )


def strip_tool_calls(text: str) -> str:
    """Remove all TOOL_CALL blocks from text, returning only the narrative portions."""
    text = _TOOL_CALL_CODE_BLOCK_PATTERN.sub("", text)
    text = _TOOL_CALL_PATTERN.sub("", text)
    return text.strip()


def format_tool_result_for_prompt(tool_name: str, result: str) -> str:
    """
    Format a tool result to be injected back into the conversation as a user message.
    This keeps context so the LLM knows what the tool returned.
    """
    return f"TOOL_RESULT [{tool_name}]:\n{result}"


def build_tools_prompt_section(tools: list[dict]) -> str:
    """
    Convert tool definitions (Anthropic-format) into a text block to inject into the system prompt.

    The LLM is instructed to emit TOOL_CALL patterns instead of structured calls.
    """
    lines = [
        "## Available Tools",
        "You can call tools by outputting EXACTLY this format (one call at a time):",
        "",
        '    TOOL_CALL: {"tool": "<tool_name>", "params": {<parameters>}}',
        "",
        "After each TOOL_CALL, wait for the TOOL_RESULT before continuing.",
        "Do not call multiple tools in a single message.",
        "",
        "### Tool Definitions:",
        "",
    ]

    for t in tools:
        name = t.get("name", "unknown")
        desc = t.get("description", "")
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        lines.append(f"**{name}**: {desc}")
        if props:
            lines.append("  Parameters:")
            for param_name, param_info in props.items():
                req_marker = " (required)" if param_name in required else ""
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                lines.append(f"    - `{param_name}` ({param_type}){req_marker}: {param_desc}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_parse_json(text: str) -> Optional[dict]:
    """Attempt to parse JSON; return None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _try_recover_json(text: str) -> Optional[dict]:
    """
    Try to recover from slightly malformed JSON by:
    1. Fixing trailing commas
    2. Fixing single-quoted strings
    3. Truncating at last valid closing brace
    """
    # Fix trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", text)

    # Replace single quotes with double quotes (naive, may break strings with apostrophes)
    fixed = re.sub(r"(?<![\\])'", '"', fixed)

    result = _safe_parse_json(fixed)
    if result:
        return result

    # Try to find the outermost valid JSON object
    brace_depth = 0
    start = text.find("{")
    if start == -1:
        return None
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                candidate = text[start:i + 1]
                result = _safe_parse_json(candidate)
                if result:
                    return result
    return None
