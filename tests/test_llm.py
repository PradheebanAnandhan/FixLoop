from types import SimpleNamespace

import pytest

from fixloop.config import Settings
from fixloop.llm import EmptyResponseError, LLMClient

SETTINGS = Settings(
    api_key="test",
    base_url="http://unused/v1/",
    models={"fast": "nano", "mid": "super", "reasoning": "ultra"},
    max_fix_attempts=5,
    max_tokens=1000,
    max_tokens_cap=3000,
    request_timeout=10,
)


def response(content, finish_reason="stop", reasoning=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


class FakeOpenAI:
    """Stands in for openai.OpenAI; replays canned responses and records calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def make_client(responses, retries=3):
    fake = FakeOpenAI(responses)
    return LLMClient(SETTINGS, client=fake, max_empty_retries=retries, retry_delay_s=0), fake


def test_routes_role_to_configured_model_and_records_usage():
    llm, fake = make_client([response("hello", reasoning="thinking...")])
    c = llm.chat("reasoning", [{"role": "user", "content": "hi"}])
    assert (c.content, c.model, c.attempts, c.reasoning) == ("hello", "ultra", 1, "thinking...")
    assert fake.calls[0]["model"] == "ultra"
    assert fake.calls[0]["max_tokens"] == 1000
    assert llm.usage.totals_by_model()["ultra"]["completion_tokens"] == 20


def test_retries_empty_content_and_grows_budget_on_length():
    llm, fake = make_client([
        response(None, finish_reason="length"),
        response("   ", finish_reason="length"),
        response("answer"),
    ])
    c = llm.chat("fast", [{"role": "user", "content": "hi"}])
    assert c.content == "answer"
    assert c.attempts == 3
    assert [call["max_tokens"] for call in fake.calls] == [1000, 2000, 3000]


def test_budget_unchanged_when_empty_for_other_reasons():
    llm, fake = make_client([response("", finish_reason="stop"), response("ok")])
    assert llm.ask("mid", "hi") == "ok"
    assert [call["max_tokens"] for call in fake.calls] == [1000, 1000]


def test_gives_up_after_max_retries():
    llm, fake = make_client([response(None, finish_reason="length")] * 3, retries=2)
    with pytest.raises(EmptyResponseError, match="3 times"):
        llm.chat("fast", [{"role": "user", "content": "hi"}])
    assert len(fake.calls) == 3
    assert llm.usage.calls == []


def test_ask_includes_system_prompt():
    llm, fake = make_client([response("ok")])
    llm.ask("fast", "question", system="be brief", temperature=0.6)
    assert fake.calls[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "question"},
    ]
    assert fake.calls[0]["temperature"] == 0.6
