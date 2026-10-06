"""Model client for NVIDIA Nemotron models on Nebius Token Factory.

Token Factory exposes an OpenAI-compatible API, so this wraps the `openai` SDK.
Callers pick a *role* ("fast", "mid", "reasoning") rather than a model ID; the
role -> model mapping comes from .env (see fixloop.config).

Nemotron models are reasoning models: they emit a reasoning trace before the
answer, and that trace counts against max_tokens. If the budget runs out
mid-trace, `message.content` comes back empty. `LLMClient.chat` handles this by
retrying, doubling max_tokens (up to a cap) when the finish reason is "length".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import Settings

log = logging.getLogger(__name__)


class EmptyResponseError(RuntimeError):
    """The model returned no answer content after all retries."""


@dataclass
class Completion:
    role: str
    model: str
    content: str
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    attempts: int


@dataclass
class UsageLog:
    """Running record of every successful call, for cost/latency reporting."""

    calls: list[Completion] = field(default_factory=list)

    def add(self, completion: Completion) -> None:
        self.calls.append(completion)

    def totals_by_model(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = {}
        for c in self.calls:
            t = totals.setdefault(
                c.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
            )
            t["calls"] += 1
            t["prompt_tokens"] += c.prompt_tokens
            t["completion_tokens"] += c.completion_tokens
            t["latency_s"] += c.latency_s
        return totals


def _reasoning_text(message: Any) -> str | None:
    # Providers expose the trace under different non-standard field names.
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None)
        if isinstance(value, str) and value:
            return value
    return None


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
        max_empty_retries: int = 3,
        retry_delay_s: float = 2.0,
    ) -> None:
        self.settings = settings
        # The SDK already retries connection errors, 429s and 5xx with backoff.
        self.client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=4,
        )
        self.max_empty_retries = max_empty_retries
        self.retry_delay_s = retry_delay_s
        self.usage = UsageLog()

    def chat(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        **params: Any,
    ) -> Completion:
        """Send a chat completion to the model for `role`; retry on empty content."""
        model = self.settings.model_for(role)
        budget = max_tokens or self.settings.max_tokens
        attempts = 1 + self.max_empty_retries
        last_finish: str | None = None

        for attempt in range(1, attempts + 1):
            start = time.monotonic()
            response = self.client.chat.completions.create(
                model=model, messages=messages, max_tokens=budget, **params
            )
            latency = time.monotonic() - start

            choice = response.choices[0] if response.choices else None
            message = choice.message if choice else None
            content = ((message.content if message else None) or "").strip()
            last_finish = choice.finish_reason if choice else None

            if content:
                usage = response.usage
                completion = Completion(
                    role=role,
                    model=model,
                    content=content,
                    reasoning=_reasoning_text(message),
                    finish_reason=last_finish,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_s=latency,
                    attempts=attempt,
                )
                self.usage.add(completion)
                return completion

            log.warning(
                "empty content from %s (attempt %d/%d, finish_reason=%s, max_tokens=%d)",
                model, attempt, attempts, last_finish, budget,
            )
            if attempt == attempts:
                break
            if last_finish == "length":
                # Reasoning trace ate the whole budget: give it more room.
                budget = min(budget * 2, self.settings.max_tokens_cap)
            if self.retry_delay_s:
                time.sleep(self.retry_delay_s * attempt)

        raise EmptyResponseError(
            f"{model} returned empty content {attempts} times "
            f"(last finish_reason={last_finish}, last max_tokens={budget})"
        )

    def ask(self, role: str, prompt: str, *, system: str | None = None, **params: Any) -> str:
        """Convenience wrapper: single user prompt in, answer text out."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return self.chat(role, messages, **params).content
