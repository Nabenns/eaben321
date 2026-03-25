"""LLM Provider abstraction — support OpenAI dan Anthropic dengan interface seragam."""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMResponse:
    """Response terstandarisasi dari provider manapun."""
    def __init__(self, text: str, tool_calls: list[dict], finished: bool):
        self.text = text                  # teks reasoning dari LLM
        self.tool_calls = tool_calls      # list of {"id", "name", "input"}
        self.finished = finished          # True = tidak ada tool call lagi


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> LLMResponse:
        """Kirim request ke LLM dan return LLMResponse."""

    @abstractmethod
    def build_tool_result_message(self, response_raw: Any, tool_results: list[dict]) -> tuple[dict, dict]:
        """
        Return (assistant_message, user_message) untuk dilanjutkan ke loop berikutnya.
        tool_results: list of {"id": tool_call_id, "content": result_text}
        """


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self._last_raw = None

    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        self._last_raw = response

        text = " ".join(b.text for b in response.content if hasattr(b, "text"))
        tool_calls = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in response.content if b.type == "tool_use"
        ]
        finished = response.stop_reason == "end_turn"
        return LLMResponse(text=text, tool_calls=tool_calls, finished=finished)

    def build_tool_result_message(self, response_raw: Any, tool_results: list[dict]) -> tuple[dict, dict]:
        assistant_msg = {"role": "assistant", "content": self._last_raw.content}
        user_msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
                for r in tool_results
            ],
        }
        return assistant_msg, user_msg


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self._last_assistant_msg = None

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Konversi format tools Anthropic → OpenAI."""
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return openai_tools

    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> LLMResponse:
        oai_messages = [{"role": "system", "content": system}] + messages
        oai_tools = self._convert_tools(tools)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=oai_messages,
            tools=oai_tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        self._last_assistant_msg = msg

        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})

        finished = not bool(msg.tool_calls)
        return LLMResponse(text=text, tool_calls=tool_calls, finished=finished)

    def build_tool_result_message(self, response_raw: Any, tool_results: list[dict]) -> tuple[dict, dict]:
        # OpenAI: assistant message dengan tool_calls, lalu tool results sebagai role "tool"
        msg = self._last_assistant_msg
        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ],
        }
        # OpenAI tidak bisa batching tool results dalam satu user message
        # Kita return sebagai list — engine harus handle ini
        # Workaround: return dummy user_msg, engine akan insert tool messages sendiri
        tool_messages = [
            {"role": "tool", "tool_call_id": r["id"], "content": r["content"]}
            for r in tool_results
        ]
        return assistant_msg, tool_messages  # type: ignore


class GroqProvider(LLMProvider):
    """Groq — gratis, cepat, OpenAI-compatible API dengan tool calling."""
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self._last_assistant_msg = None

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> LLMResponse:
        oai_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=oai_messages,
            tools=self._convert_tools(tools),
            tool_choice="auto",
        )
        msg = response.choices[0].message
        self._last_assistant_msg = msg
        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})
        finished = not bool(msg.tool_calls)
        return LLMResponse(text=text, tool_calls=tool_calls, finished=finished)

    def build_tool_result_message(self, response_raw: Any, tool_results: list[dict]) -> tuple[dict, dict]:
        msg = self._last_assistant_msg
        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ],
        }
        tool_messages = [
            {"role": "tool", "tool_call_id": r["id"], "content": r["content"]}
            for r in tool_results
        ]
        return assistant_msg, tool_messages  # type: ignore


class BrowserLLMProvider(LLMProvider):
    """
    Browser automation provider that drives ChatGPT or Claude via Playwright.

    Unlike API providers, this one:
    - Does NOT support native tool calling (handled by LLMEngine via text parsing)
    - Uses browser_provider_name = "chatgpt" | "claude"
    - Manages its own async event loop internally for sync compatibility
    """

    def __init__(self, browser_provider_name: str = "chatgpt"):
        from src.llm.browser_provider import BrowserProvider
        self.browser_provider_name = browser_provider_name
        self._browser = BrowserProvider(provider=browser_provider_name)
        self._loop = None
        self._initialized = False
        self.supports_tool_calling = False   # Flag for LLMEngine to detect

    def _get_loop(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop
        except RuntimeError:
            pass
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        """Run an async coroutine from sync context."""
        import asyncio
        loop = self._get_loop()
        if loop.is_running():
            # If already in async context, schedule it
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=150)
        return loop.run_until_complete(coro)

    def _ensure_initialized(self):
        if not self._initialized:
            self._run(self._browser.initialize())
            self._initialized = True

    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> "LLMResponse":
        """
        Send a message to the browser-based LLM.

        NOTE: Tool calls are handled as text patterns by LLMEngine when using this provider.
        The system prompt and tools are expected to already be formatted as text by LLMEngine.
        """
        self._ensure_initialized()

        # Reconstruct prompt from messages (last user message)
        # The engine is responsible for injecting tool definitions into system/prompt
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_msg = content
                elif isinstance(content, list):
                    # Handle multi-part content (text + tool results)
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif part.get("type") == "tool_result":
                                parts.append(f"Tool result: {part.get('content', '')}")
                        else:
                            parts.append(str(part))
                    last_user_msg = "\n".join(parts)
                break

        # Prepend system prompt to first message if this is the start of conversation
        full_prompt = last_user_msg
        if len(messages) == 1:
            # First turn: include system prompt
            full_prompt = f"{system}\n\n---\n\n{last_user_msg}"

        response_text = self._run(self._browser.send_message(full_prompt))

        from src.llm.text_tool_parser import parse_tool_calls, has_tool_calls
        if has_tool_calls(response_text):
            parsed_calls = parse_tool_calls(response_text)
            tool_calls = [
                {"id": tc.id, "name": tc.name, "input": tc.input}
                for tc in parsed_calls
            ]
            # Not finished — engine should handle tool calls and continue
            return LLMResponse(text=response_text, tool_calls=tool_calls, finished=False)

        return LLMResponse(text=response_text, tool_calls=[], finished=True)

    def build_tool_result_message(self, response_raw: Any, tool_results: list[dict]) -> tuple[dict, dict]:
        """
        For browser provider: format tool results as plain text messages.
        The engine will send them back to the LLM in the next turn.
        """
        from src.llm.text_tool_parser import format_tool_result_for_prompt

        result_parts = [
            format_tool_result_for_prompt(r["id"], r["content"])
            for r in tool_results
        ]
        result_text = "\n\n".join(result_parts)

        # Return as a simple user message with tool results
        assistant_msg = {"role": "assistant", "content": response_raw or ""}
        user_msg = {"role": "user", "content": result_text}
        return assistant_msg, user_msg

    def start_new_conversation(self):
        """Start a fresh conversation in the browser."""
        self._ensure_initialized()
        self._run(self._browser.new_conversation())

    def close(self):
        """Close browser and cleanup."""
        if self._initialized:
            try:
                self._run(self._browser.close())
            except Exception as e:
                logger.warning("[BrowserLLMProvider] Error during close: %s", e)
            self._initialized = False


def create_provider(provider: str = None, model: str = None) -> LLMProvider:
    """Factory — buat provider berdasarkan env var atau parameter."""
    provider = provider or os.environ.get("LLM_PROVIDER", "anthropic").lower()

    # Browser-based providers
    if provider == "browser_chatgpt":
        logger.info("LLM Provider: Browser (ChatGPT)")
        return BrowserLLMProvider(browser_provider_name="chatgpt")

    if provider == "browser_claude":
        logger.info("LLM Provider: Browser (Claude)")
        return BrowserLLMProvider(browser_provider_name="claude")

    if provider == "openai" or provider == "api_openai":
        model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        logger.info("LLM Provider: OpenAI | Model: %s", model)
        return OpenAIProvider(model=model)

    if provider == "groq":
        model = model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        logger.info("LLM Provider: Groq (FREE) | Model: %s", model)
        return GroqProvider(model=model)

    if provider in ("anthropic", "api_anthropic"):
        model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        logger.info("LLM Provider: Anthropic | Model: %s", model)
        return AnthropicProvider(model=model)

    # Default: Anthropic
    model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
    logger.info("LLM Provider: Anthropic (default) | Model: %s", model)
    return AnthropicProvider(model=model)
