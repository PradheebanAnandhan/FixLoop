"""Scripted stand-in for the Token Factory API, used through the real LLMClient."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from fixloop.config import Settings
from fixloop.llm import LLMClient


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        api_key="test", base_url="http://unused/v1/",
        models={"fast": "nano", "mid": "super", "reasoning": "ultra"},
        max_fix_attempts=5, max_tokens=1000, max_tokens_cap=4000, request_timeout=10,
        runs_dir=tmp_path / "runs", prices={"fast": (0.1, 0.4), "mid": (0.5, 1.0), "reasoning": (1.0, 3.0)},
    )
    base.update(overrides)
    return Settings(**base)


class ScriptedOpenAI:
    """`responder(model, messages) -> str`; every call is recorded."""

    def __init__(self, responder: Callable[[str, list[dict]], str]):
        self.responder = responder
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, max_tokens, **params):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages]})
        content = self.responder(model, messages)
        message = SimpleNamespace(content=content, reasoning_content="(trace)")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


def scripted_llm(settings: Settings, responder) -> tuple[LLMClient, ScriptedOpenAI]:
    fake = ScriptedOpenAI(responder)
    return LLMClient(settings, client=fake, retry_delay_s=0), fake


def last_user(messages: list[dict]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def first_user(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")
