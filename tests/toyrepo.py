"""A tiny git repo with a planted bug, plus a local (non-Docker) sandbox for fast tests."""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from verifier.sandbox import PLUGIN_DIR, RunResult, base_env, read_report

CALC = '''\
def add(a, b):
    return a + b


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return lo
    return x
'''

EXISTING_TESTS = '''\
import os

from calc import add, clamp


def test_add():
    assert add(2, 3) == 5


def test_clamp_low():
    assert clamp(-1, 0, 3) == 0


def test_clamp_mid():
    assert clamp(2, 0, 3) == 2


def test_known_broken():
    assert add(0.1, 0.2) == 0.3


def test_env_dependent():
    assert os.environ.get("TOY_FLAKY", "pass") == "pass"
'''

REPRO = '''\
from calc import clamp


def test_clamp_above_range_returns_hi():
    assert clamp(5, 0, 3) == 3
'''

HELDOUT = '''\
from calc import clamp


def test_clamp_far_above_range():
    assert clamp(100, -5, 7) == 7


def test_clamp_at_upper_bound():
    assert clamp(7, -5, 7) == 7
'''


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout


def make_repo(root: Path) -> tuple[Path, str]:
    repo = root / "toy"
    (repo / "tests").mkdir(parents=True)
    (repo / "calc.py").write_text(CALC)
    (repo / "tests" / "test_calc.py").write_text(EXISTING_TESTS)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo, git(repo, "rev-parse", "HEAD").strip()


def make_patch(repo: Path, edits: dict[str, str]) -> str:
    """Apply {path: new_content} to the working tree, return the diff, then reset."""
    for path, content in edits.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git(repo, "add", "-A")
    diff = git(repo, "diff", "--cached")
    git(repo, "reset", "-q", "--hard")
    return diff


FIXED_CALC = CALC.replace("    if x > hi:\n        return lo", "    if x > hi:\n        return hi")


class LocalSandbox:
    """Runs pytest on the host with the verifier's plugin. No isolation: tests only."""

    def __init__(self) -> None:
        self.extra_env: dict[str, str] = {}
        self.calls: list[list[str]] = []

    def run_pytest(self, workdir: Path, args: list[str], timeout: float) -> RunResult:
        self.calls.append(args)
        report_path = Path(workdir).parent / f"report-{secrets.token_hex(4)}.json"
        env = {
            **base_env(),
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(PLUGIN_DIR),
            "FIXLOOP_REPORT_PATH": str(report_path),
            **self.extra_env,
        }
        argv = [sys.executable, "-m", "pytest", "-p", "fixloop_report", "-p", "no:cacheprovider", *args]
        start = time.monotonic()
        try:
            proc = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
            code, out, timed_out = proc.returncode, proc.stdout + proc.stderr, False
        except subprocess.TimeoutExpired as e:
            code, out, timed_out = None, str(e.stdout or ""), True
        report = read_report(report_path)
        report_path.unlink(missing_ok=True)
        return RunResult(argv, code, timed_out, out, time.monotonic() - start, report)
