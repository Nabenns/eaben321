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
        self.text = text
        self.tool_calls = tool_calls
        self.finished = finished


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096) -> LLMResponse:
        pass

    @abstractmethod
    def build_tool_result_message(self, response_raw: Any, tool_results: list[dict]) -> tuple[dict, Any]:
        pass


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
    def __init__(self, model: str = "gpt-4o", base_url: str = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
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

    def build_tool_result_message(self, response_raw: Any, tool_results: list[dict]) -> tuple[dict, list]:
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
        return assistant_msg, tool_messages


class GroqProvider(OpenAIProvider):
    """Groq — free tier, OpenAI-compatible API dengan tool calling."""
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", "dummy"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self._last_assistant_msg = None


def create_provider(provider: str = None, model: str = None) -> LLMProvider:
    """Factory — buat provider berdasarkan env var LLM_PROVIDER.
    Pilihan: openai | anthropic | groq
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        logger.info("LLM Provider: Anthropic | Model: %s", model)
        return AnthropicProvider(model=model)

    if provider == "groq":
        model = model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        logger.info("LLM Provider: Groq | Model: %s", model)
        return GroqProvider(model=model)

    # Default: OpenAI
    model = model or os.environ.get("LLM_MODEL", "gpt-4o")
    logger.info("LLM Provider: OpenAI | Model: %s", model)
    return OpenAIProvider(model=model)
