#!/usr/bin/env python3
"""Confirm API access and the configured model IDs (Nebius Token Factory by default).

1. Lists the models available to your key via GET /v1/models.
2. Checks that MODEL_FAST / MODEL_MID / MODEL_REASONING from .env are in that list.
3. Sends one short test prompt to each configured model (or just one, via --role).

Usage:
    python scripts/check_models.py                 # list + prompt all three roles
    python scripts/check_models.py --role fast     # prompt only the Nano model
    python scripts/check_models.py --list-only     # skip the test prompts

Exit code is 0 only if listing worked, every configured ID is listed, and every
prompted model answered.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402

from fixloop.config import ROLE_ENV_VARS, ROLES, ConfigError, load_settings  # noqa: E402
from fixloop.llm import EmptyResponseError, LLMClient  # noqa: E402

TEST_PROMPT = "Reply with exactly this text and nothing else: FixLoop OK"


def mask(secret: str) -> str:
    return secret[:4] + "…" + secret[-4:] if len(secret) > 12 else "…"


def list_models(llm: LLMClient) -> list[str] | None:
    print("\n== GET /v1/models ==")
    try:
        ids = sorted(m.id for m in llm.client.models.list())
    except openai.AuthenticationError as e:
        print(f"  FAIL: authentication rejected ({e.status_code}). Check NEBIUS_API_KEY.")
        return None
    except openai.NotFoundError:
        print("  FAIL: 404 from /models. Check NEBIUS_BASE_URL (it should end in /v1/).")
        return None
    except openai.APIConnectionError as e:
        print(f"  FAIL: could not connect to {llm.settings.base_url}: {e}")
        return None
    except openai.APIError as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return None

    nemotron = [i for i in ids if "nemotron" in i.lower()]
    print(f"  {len(ids)} models available, {len(nemotron)} Nemotron:")
    for model_id in nemotron:
        print(f"    {model_id}")
    if not nemotron:
        print("  (no Nemotron models listed; all model IDs follow)")
        for model_id in ids:
            print(f"    {model_id}")
    return ids


def check_configured(llm: LLMClient, available: list[str]) -> bool:
    print("\n== Configured models ==")
    ok = True
    for role in ROLES:
        model_id = llm.settings.model_for(role)
        if model_id in available:
            print(f"  ok    {ROLE_ENV_VARS[role]:<16} {model_id}")
            continue
        ok = False
        print(f"  MISS  {ROLE_ENV_VARS[role]:<16} {model_id}  (not in /v1/models)")
        lowered = {m.lower(): m for m in available}
        close = difflib.get_close_matches(model_id.lower(), lowered, n=3, cutoff=0.5)
        if close:
            print("        did you mean: " + ", ".join(lowered[c] for c in close))
    return ok


def test_prompt(llm: LLMClient, role: str) -> bool:
    model_id = llm.settings.model_for(role)
    print(f"\n== Test prompt: {role} ({model_id}) ==")
    try:
        c = llm.chat(role, [{"role": "user", "content": TEST_PROMPT}])
    except EmptyResponseError as e:
        print(f"  FAIL: {e}")
        return False
    except openai.APIError as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False

    print(f"  answer:        {c.content!r}")
    print(f"  finish_reason: {c.finish_reason}")
    print(f"  tokens:        {c.prompt_tokens} prompt / {c.completion_tokens} completion")
    print(f"  latency:       {c.latency_s:.1f}s  (attempts: {c.attempts})")
    if c.reasoning:
        print(f"  reasoning:     {len(c.reasoning)} chars of trace returned")
    if "FixLoop OK" not in c.content:
        print("  note: answer did not match the expected text exactly (access still works)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--role", choices=[*ROLES, "all"], default="all",
                        help="which configured model to send the test prompt to (default: all)")
    parser.add_argument("--list-only", action="store_true", help="only list models; skip test prompts")
    parser.add_argument("--env-file", help="path to a .env file (default: <repo>/.env)")
    parser.add_argument("-v", "--verbose", action="store_true", help="also show HTTP request logs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="  [%(levelname)s] %(name)s: %(message)s")

    try:
        settings = load_settings(args.env_file)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    print(f"Provider:   {settings.provider_label} (LLM_PROVIDER={settings.provider})")
    print(f"Base URL:   {settings.base_url}")
    print(f"API key:    {mask(settings.api_key)}")
    print(f"max_tokens: {settings.max_tokens} (cap on retry: {settings.max_tokens_cap})")

    llm = LLMClient(settings)
    available = list_models(llm)
    if available is None:
        return 1
    configured_ok = check_configured(llm, available)

    prompts_ok = True
    if not args.list_only:
        roles = ROLES if args.role == "all" else (args.role,)
        for role in roles:
            prompts_ok = test_prompt(llm, role) and prompts_ok

    print("\n== Summary ==")
    print("  model listing:     ok")
    print(f"  configured IDs:    {'ok' if configured_ok else 'some IDs not listed (see above)'}")
    if not args.list_only:
        print(f"  test prompts:      {'ok' if prompts_ok else 'FAILED (see above)'}")
    return 0 if configured_ok and prompts_ok else 1


if __name__ == "__main__":
    sys.exit(main())
