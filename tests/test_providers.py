"""Provider presets, free-tier pacing and token-limit handling."""

import httpx2 as httpx
import openai
import pytest

from fakes import make_settings
from fixloop.agent import _compact
from fixloop.config import ConfigError, load_settings
from fixloop.llm import LLMClient
from test_llm import FakeOpenAI, response

PROVIDER_VARS = ("LLM_", "NEBIUS_", "GROQ_", "NVIDIA_", "OPENROUTER_", "MODEL_")
MODELS = "MODEL_FAST=a\nMODEL_MID=b\nMODEL_REASONING=c\n"


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith(PROVIDER_VARS):
            monkeypatch.delenv(k)

    def write(text):
        path = tmp_path / ".env"
        path.write_text(text)
        return path
    return write


def test_nebius_is_the_default(env_file):
    s = load_settings(env_file("NEBIUS_API_KEY=k\n" + MODELS))
    assert (s.provider, s.provider_label, s.api_key) == ("nebius", "Nebius Token Factory", "k")
    assert s.base_url == "https://api.tokenfactory.nebius.com/v1/"
    assert (s.max_tokens, s.context_chars, s.min_interval_s) == (16384, 60_000, 0.0)


def test_nebius_base_url_from_dashboard_still_wins(env_file):
    s = load_settings(env_file("NEBIUS_API_KEY=k\nNEBIUS_BASE_URL=https://custom/v1/\n" + MODELS))
    assert s.base_url == "https://custom/v1/"


def test_groq_preset_uses_free_tier_defaults(env_file):
    s = load_settings(env_file("LLM_PROVIDER=groq\nGROQ_API_KEY=g\n" + MODELS))
    assert (s.provider_label, s.api_key, s.base_url) == ("Groq", "g", "https://api.groq.com/openai/v1")
    assert (s.max_tokens, s.context_chars, s.min_interval_s, s.max_retries) == (8192, 20_000, 2.0, 6)


def test_overrides_and_custom_provider(env_file):
    s = load_settings(env_file(
        "LLM_PROVIDER=custom\nLLM_API_KEY=x\nLLM_BASE_URL=http://localhost:8000/v1\n"
        "LLM_CONTEXT_CHARS=5000\nLLM_MIN_INTERVAL_SECONDS=0.5\nLLM_MAX_TOKENS=2048\n" + MODELS))
    assert (s.base_url, s.context_chars, s.min_interval_s, s.max_tokens) == ("http://localhost:8000/v1", 5000, 0.5, 2048)


@pytest.mark.parametrize("text,needle", [
    ("LLM_PROVIDER=groq\n" + MODELS, "GROQ_API_KEY"),
    ("LLM_PROVIDER=custom\nLLM_API_KEY=x\n" + MODELS, "LLM_BASE_URL"),
    ("LLM_PROVIDER=nope\n", "LLM_PROVIDER must be one of"),
])
def test_config_errors(env_file, text, needle):
    with pytest.raises(ConfigError, match=needle):
        load_settings(env_file(text))


def bad_request(message):
    return openai.BadRequestError(
        message, response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)


class RaisingThenOK(FakeOpenAI):
    def __init__(self, errors, responses):
        super().__init__(responses)
        self.errors = list(errors)

    def _create(self, **kwargs):
        if self.errors:
            self.calls.append(kwargs)
            raise self.errors.pop(0)
        return super()._create(**kwargs)


def test_token_budget_shrinks_when_provider_rejects_it(tmp_path):
    settings = make_settings(tmp_path, max_tokens=8192)
    fake = RaisingThenOK([bad_request("max_tokens must be <= 4096"), bad_request("max_completion_tokens too big")],
                         [response("ok")])
    llm = LLMClient(settings, client=fake, retry_delay_s=0)
    assert llm.ask("fast", "hi") == "ok"
    assert [c["max_tokens"] for c in fake.calls] == [8192, 4096, 2048]


def test_other_bad_requests_are_not_swallowed(tmp_path):
    fake = RaisingThenOK([bad_request("model not found")], [response("ok")])
    llm = LLMClient(make_settings(tmp_path), client=fake, retry_delay_s=0)
    with pytest.raises(openai.BadRequestError):
        llm.ask("fast", "hi")


def test_requests_are_paced(tmp_path, monkeypatch):
    sleeps = []
    clock = [100.0]
    monkeypatch.setattr("fixloop.llm.time.sleep", lambda s: (sleeps.append(round(s, 2)), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr("fixloop.llm.time.monotonic", lambda: clock[0])
    llm = LLMClient(make_settings(tmp_path, min_interval_s=2.0), client=FakeOpenAI([response("a"), response("b")]),
                    retry_delay_s=0)
    llm.ask("fast", "1")
    llm.ask("fast", "2")
    assert sleeps == [2.0]


def test_compact_keeps_task_and_latest_exchange():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(4):
        msgs += [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": f"f{i}"}]
    out = _compact(msgs)
    assert [m["content"] for m in out[2:]] == ["a2", "f2", "a3", "f3"]
    assert "4 earlier messages omitted" in out[1]["content"]
    assert msgs[1]["content"] == "task"  # original not mutated
    roles = [m["role"] for m in out[1:]]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    assert _compact(msgs[:4]) == msgs[:4]
