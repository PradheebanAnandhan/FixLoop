"""Settings loaded from the environment and the repo's .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Role -> env var holding the model ID for that role.
#   fast:      cheap calls (localize, summarize, held-out tests)  e.g. Nemotron Nano
#   mid:       mid-weight work (judge, PR text)                    e.g. Nemotron Super
#   reasoning: reproduce and fix                                   e.g. Nemotron Ultra
ROLE_ENV_VARS = {
    "fast": "MODEL_FAST",
    "mid": "MODEL_MID",
    "reasoning": "MODEL_REASONING",
}
ROLES = tuple(ROLE_ENV_VARS)


@dataclass(frozen=True)
class Provider:
    label: str
    key_env: str
    base_url: str | None
    max_tokens: int        # default output budget per call
    context_chars: int     # default budget for source files included in prompts
    min_interval_s: float  # pause between requests (free tiers have per-minute limits)


# Any OpenAI-compatible endpoint works. Nebius Token Factory is the default (and
# what the hackathon submission must use); the others are for free development.
PROVIDERS = {
    "nebius": Provider("Nebius Token Factory", "NEBIUS_API_KEY",
                       "https://api.tokenfactory.nebius.com/v1/", 16384, 60_000, 0.0),
    "nvidia": Provider("NVIDIA API Catalog", "NVIDIA_API_KEY",
                       "https://integrate.api.nvidia.com/v1", 16384, 40_000, 2.0),
    "groq": Provider("Groq", "GROQ_API_KEY",
                     "https://api.groq.com/openai/v1", 8192, 20_000, 2.0),
    "openrouter": Provider("OpenRouter", "OPENROUTER_API_KEY",
                           "https://openrouter.ai/api/v1", 16384, 40_000, 3.0),
    "custom": Provider("OpenAI-compatible endpoint", "LLM_API_KEY", None, 16384, 60_000, 0.0),
}


class ConfigError(RuntimeError):
    """Raised when required settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    models: dict[str, str]
    max_fix_attempts: int
    # Nemotron models spend tokens on a reasoning trace before the answer,
    # so the default budget is generous and can be raised further on retry.
    max_tokens: int
    max_tokens_cap: int
    request_timeout: float
    provider: str = "nebius"
    context_chars: int = 60_000
    min_interval_s: float = 0.0
    max_retries: int = 4
    max_repro_attempts: int = 3
    heldout_mode: str = "report"           # "gate", "report" or "off"
    sandbox_base_image: str = "python:3.12-slim"
    runs_dir: Path = REPO_ROOT / "runs"
    github_token: str | None = None
    # USD per 1M (input, output) tokens per role, if configured; used for cost reports.
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)

    def model_for(self, role: str) -> str:
        try:
            return self.models[role]
        except KeyError:
            raise ConfigError(f"unknown model role {role!r}; expected one of {ROLES}") from None

    @property
    def provider_label(self) -> str:
        return PROVIDERS[self.provider].label if self.provider in PROVIDERS else self.provider


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _float_env(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    """Load settings from `env_file` (default: <repo>/.env) and the process env.

    Variables already set in the process environment take precedence over .env.
    """
    load_dotenv(env_file or REPO_ROOT / ".env", override=False)

    provider_name = (_env("LLM_PROVIDER") or "nebius").lower()
    if provider_name not in PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER must be one of: {', '.join(PROVIDERS)}")
    provider = PROVIDERS[provider_name]

    api_key = _env("LLM_API_KEY") or _env(provider.key_env)
    base_url = _env("LLM_BASE_URL") or (_env("NEBIUS_BASE_URL") if provider_name == "nebius" else "") \
        or provider.base_url
    missing = [] if api_key else [f"{provider.key_env} (or LLM_API_KEY)"]
    if not base_url:
        missing.append("LLM_BASE_URL")
    missing += [var for var in ROLE_ENV_VARS.values() if not _env(var)]
    if missing:
        raise ConfigError(
            f"missing required settings for provider {provider_name!r}: " + ", ".join(missing)
            + "\nCopy .env.example to .env and fill it in."
        )

    max_tokens = _int_env("LLM_MAX_TOKENS", provider.max_tokens)
    settings = Settings(
        api_key=api_key,
        base_url=base_url,
        models={role: _env(var) for role, var in ROLE_ENV_VARS.items()},
        max_fix_attempts=_int_env("MAX_FIX_ATTEMPTS", 5),
        max_tokens=max_tokens,
        max_tokens_cap=max(_int_env("LLM_MAX_TOKENS_CAP", max(65536, max_tokens)), max_tokens),
        request_timeout=float(_int_env("LLM_TIMEOUT_SECONDS", 600)),
        provider=provider_name,
        context_chars=_int_env("LLM_CONTEXT_CHARS", provider.context_chars),
        min_interval_s=_float_env("LLM_MIN_INTERVAL_SECONDS", provider.min_interval_s),
        max_retries=_int_env("LLM_MAX_RETRIES", 6 if provider.min_interval_s else 4),
        max_repro_attempts=_int_env("MAX_REPRO_ATTEMPTS", 3),
        heldout_mode=_env("HELDOUT_MODE") or "report",
        sandbox_base_image=_env("SANDBOX_BASE_IMAGE") or "python:3.12-slim",
        runs_dir=Path(_env("RUNS_DIR") or REPO_ROOT / "runs"),
        github_token=_env("GITHUB_TOKEN") or None,
        prices=_prices(),
    )
    if settings.max_fix_attempts < 1:
        raise ConfigError("MAX_FIX_ATTEMPTS must be at least 1")
    if settings.heldout_mode not in ("gate", "report", "off"):
        raise ConfigError("HELDOUT_MODE must be one of: gate, report, off")
    return settings


def _prices() -> dict[str, tuple[float, float]]:
    prices = {}
    for role, var in ROLE_ENV_VARS.items():
        raw = os.environ.get(f"{var}_PRICE", "").strip()
        if not raw:
            continue
        try:
            inp, out = (float(x) for x in raw.split(","))
        except ValueError:
            raise ConfigError(f"{var}_PRICE must look like '0.10,0.40' (USD per 1M input,output tokens)") from None
        prices[role] = (inp, out)
    return prices
