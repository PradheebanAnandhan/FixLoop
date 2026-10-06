"""End-to-end test of scripts/check_models.py against a local OpenAI-compatible stub."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_models.py"
LISTED = ["nvidia/nano", "nvidia/super", "nvidia/ultra", "meta/other-model"]


class StubHandler(BaseHTTPRequestHandler):
    chat_calls: list = []

    def log_message(self, *args):
        pass

    def _send(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer good-key-123456":
            return self._send(401, {"error": {"message": "bad key"}})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "x"} for m in LISTED
            ]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubHandler.chat_calls.append(body)
        # Simulate a reasoning model: the first call per model runs out of budget.
        first = sum(c["model"] == body["model"] for c in StubHandler.chat_calls) == 1
        content, finish = (None, "length") if first else ("FixLoop OK", "stop")
        self._send(200, {
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": finish, "message": {
                "role": "assistant", "content": content, "reasoning_content": "hmm",
            }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        })


@pytest.fixture
def stub_url():
    StubHandler.chat_calls = []
    server = HTTPServer(("127.0.0.1", 0), StubHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/"
    server.shutdown()


def run(tmp_path, base_url, *args, key="good-key-123456", fast="nvidia/nano"):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"NEBIUS_API_KEY={key}\nNEBIUS_BASE_URL={base_url}\n"
        f"MODEL_FAST={fast}\nMODEL_MID=nvidia/super\nMODEL_REASONING=nvidia/ultra\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEBIUS_", "MODEL_", "LLM_", "GROQ_", "NVIDIA_", "OPENROUTER_"))}
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--env-file", str(env_file), *args],
        capture_output=True, text=True, env=env, timeout=60,
    )


def test_all_good(tmp_path, stub_url):
    result = run(tmp_path, stub_url)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "attempts: 2" in result.stdout  # empty-content retry kicked in
    assert result.stdout.count("answer:        'FixLoop OK'") == 3
    assert "good…3456" in result.stdout and "good-key-123456" not in result.stdout


def test_single_role(tmp_path, stub_url):
    result = run(tmp_path, stub_url, "--role", "fast")
    assert result.returncode == 0, result.stdout + result.stderr
    assert {c["model"] for c in StubHandler.chat_calls} == {"nvidia/nano"}


def test_missing_model_suggests_close_match(tmp_path, stub_url):
    result = run(tmp_path, stub_url, "--list-only", fast="nvidia/nan0")
    assert result.returncode == 1
    assert "MISS" in result.stdout and "did you mean: nvidia/nano" in result.stdout
    assert StubHandler.chat_calls == []


def test_bad_key(tmp_path, stub_url):
    result = run(tmp_path, stub_url, key="wrong-key-000000")
    assert result.returncode == 1
    assert "authentication rejected" in result.stdout


def test_missing_config(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("NEBIUS_API_KEY=x\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEBIUS_", "MODEL_", "LLM_", "GROQ_", "NVIDIA_", "OPENROUTER_"))}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--env-file", str(env_file)],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode == 2
    assert "MODEL_FAST" in result.stderr and "NEBIUS_API_KEY" not in result.stderr
