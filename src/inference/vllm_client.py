"""Thin wrapper around the OpenAI-compatible vLLM API."""

import openai
import time


class VLLMClient:
    """Client for the vLLM OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "qwen3-coder",
        api_key: str = "dummy",
    ):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        tools: list[dict] | None = None,
    ) -> openai.types.chat.ChatCompletion:
        """Send a chat completion request."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs
        if tools:
            kwargs["tools"] = tools

        start = time.time()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        # Track token usage
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens

        return response

    def get_token_usage(self) -> dict:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def check_health(self) -> bool:
        """Check if the vLLM server is responding."""
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False
