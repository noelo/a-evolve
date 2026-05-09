"""OpenAI-compatible LLM provider that supports arbitrary base URLs."""

from __future__ import annotations

from typing import Any

from .base import LLMMessage, LLMProvider, LLMResponse


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider using an OpenAI-compatible API.

    Supports any server that implements the OpenAI chat completions endpoint,
    e.g. vLLM, Ollama, LM Studio, Together.ai, Groq, etc.

    Args:
        model: Model name to use (e.g. "meta-llama/Llama-3-70b").
        base_url: API base URL (e.g. "http://localhost:8000/v1" or "https://api.groq.com/openai/v1").
        api_key: API key (pass when connecting to non-OpenAI servers, or for custom auth).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai  (or: pip install agent-evolve[openai])")

        self.model = model
        self.base_url = base_url
        self.client = (
            openai.OpenAI(api_key=api_key, base_url=base_url)
            if base_url or api_key
            else openai.OpenAI()
        )

    def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
            raw=response,
        )

    def complete_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            max_tokens=max_tokens,
            tools=tools,
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
            raw=response,
        )
