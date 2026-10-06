"""Settings loaded from the environment and the repo's .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Role -> env var holding the model ID for that role.
#   fast:      Nemotron Nano  (localize, summarize, triage)
#   mid:       Nemotron Super (mid-weight work)
#   reasoning: Nemotron Ultra (reproduce and fix)
ROLE_ENV_VARS = {
    "fast": "MODEL_FAST",
    "mid": "MODEL_MID",
    "reasoning": "MODEL_REASONING",
}
ROLES = tuple(ROLE_ENV_VARS)


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

    def model_for(self, role: str) -> str:
        try:
            return self.models[role]
        except KeyError:
            raise ConfigError(f"unknown model role {role!r}; expected one of {ROLES}") from None


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

    required = ["NEBIUS_API_KEY", "NEBIUS_BASE_URL", *ROLE_ENV_VARS.values()]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise ConfigError(
            "missing required settings: " + ", ".join(missing)
            + "\nCopy .env.example to .env and fill it in."
        )

    max_tokens = _int_env("LLM_MAX_TOKENS", 16384)
    settings = Settings(
        api_key=os.environ["NEBIUS_API_KEY"].strip(),
        base_url=os.environ["NEBIUS_BASE_URL"].strip(),
        models={role: os.environ[var].strip() for role, var in ROLE_ENV_VARS.items()},
        max_fix_attempts=_int_env("MAX_FIX_ATTEMPTS", 5),
        max_tokens=max_tokens,
        max_tokens_cap=max(_int_env("LLM_MAX_TOKENS_CAP", 65536), max_tokens),
        request_timeout=float(_int_env("LLM_TIMEOUT_SECONDS", 600)),
    )
    if settings.max_fix_attempts < 1:
        raise ConfigError("MAX_FIX_ATTEMPTS must be at least 1")
    return settings
