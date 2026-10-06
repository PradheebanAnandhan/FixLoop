"""Docker sandbox: build a per-repo image once, then run code in throwaway containers.

Image builds are the only step with network access (to install the repo's
dependencies). Every run afterwards gets a fresh container with:
  - no network (--network none)
  - a read-only root filesystem, and the checkout mounted read-only by default
  - a cleared environment (`env -i` plus a short allowlist)
  - no capabilities, no privilege escalation, pid/memory/cpu limits
  - a hard wall-clock timeout, after which the container is killed
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PLUGIN_DIR = Path(__file__).resolve().parent / "plugin"
PLUGIN_MOUNT = "/opt/fixloop-verifier"
WORKDIR = "/work"
MAX_OUTPUT_CHARS = 200_000


class SandboxError(RuntimeError):
    """The sandbox itself failed (Docker missing, image build failed, ...)."""


@dataclass
class RunResult:
    argv: list[str]
    exit_code: int | None
    timed_out: bool
    output: str
    duration_s: float
    report: dict | None = None  # parsed fixloop_report JSON, for pytest runs


class Sandbox(Protocol):
    def run_pytest(self, workdir: Path, args: list[str], timeout: float) -> RunResult: ...


@dataclass(frozen=True)
class SandboxConfig:
    image: str
    memory: str = "2g"
    cpus: str = "2"
    pids_limit: int = 512
    tmpfs_size: str = "512m"
    read_only_repo: bool = True
    docker: str = "docker"


def base_env() -> dict[str, str]:
    """The only environment variables a sandboxed process sees."""
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "COVERAGE_FILE": "/tmp/.coverage",
    }


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n\n... [output truncated] ...\n\n" + text[-half:]


def read_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class DockerSandbox:
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    def _docker(self, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [self.config.docker, *args], capture_output=True, text=True, timeout=timeout
            )
        except FileNotFoundError:
            raise SandboxError(f"{self.config.docker!r} not found; install Docker") from None

    def run(
        self,
        workdir: Path,
        command: list[str],
        timeout: float,
        *,
        env: dict[str, str] | None = None,
        mounts: list[tuple[Path, str, str]] = (),
        writable: bool | None = None,
    ) -> RunResult:
        """Run `command` in a fresh container with `workdir` mounted at /work.

        `mounts` are extra (host_path, container_path, "ro"|"rw") bind mounts.
        """
        cfg = self.config
        name = f"fixloop-{secrets.token_hex(6)}"
        ro = cfg.read_only_repo if writable is None else not writable
        run_env = {**base_env(), **(env or {})}
        argv = [
            cfg.docker, "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only",
            "--tmpfs", f"/tmp:rw,exec,size={cfg.tmpfs_size}",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(cfg.pids_limit),
            "--memory", cfg.memory,
            "--cpus", cfg.cpus,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{Path(workdir).resolve()}:{WORKDIR}:{'ro' if ro else 'rw'}",
        ]
        for host, target, mode in mounts:
            argv += ["-v", f"{Path(host).resolve()}:{target}:{mode}"]
        argv += ["-w", WORKDIR, cfg.image, "env", "-i"]
        argv += [f"{k}={v}" for k, v in run_env.items()]
        argv += command

        start = time.monotonic()
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            raise SandboxError(f"{cfg.docker!r} not found; install Docker") from None
        try:
            out, _ = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            self._docker("kill", name, timeout=30)
            out, _ = proc.communicate()
            timed_out = True
        duration = time.monotonic() - start
        output = _clip(out.decode("utf-8", errors="replace"))
        if not timed_out and proc.returncode == 125:
            raise SandboxError(f"docker could not start the container:\n{output[-2000:]}")
        return RunResult(
            argv=argv,
            exit_code=None if timed_out else proc.returncode,
            timed_out=timed_out,
            output=output,
            duration_s=duration,
        )

    def run_pytest(self, workdir: Path, args: list[str], timeout: float) -> RunResult:
        """Run pytest with the verifier's report plugin and parse its JSON report."""
        with tempfile.TemporaryDirectory(prefix="fixloop-out-") as out_dir:
            report_name = f"{secrets.token_hex(8)}.json"
            result = self.run(
                workdir,
                ["python", "-m", "pytest", "-p", "fixloop_report", "-p", "no:cacheprovider", *args],
                timeout,
                env={
                    "PYTHONPATH": PLUGIN_MOUNT,
                    "FIXLOOP_REPORT_PATH": f"/out/{report_name}",
                },
                mounts=[(PLUGIN_DIR, PLUGIN_MOUNT, "ro"), (Path(out_dir), "/out", "rw")],
            )
            result.report = read_report(Path(out_dir) / report_name)
        return result

    def image_exists(self) -> bool:
        return self._docker("image", "inspect", self.config.image).returncode == 0


# --- image build --------------------------------------------------------------

_DOCKERFILE = """\
FROM {base_image}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore \\
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
COPY . {workdir}
WORKDIR {workdir}
RUN --mount=type=secret,id=ca,required=false set -e; \\
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca SSL_CERT_FILE=/run/secrets/ca; fi; \\
    python -c "import pytest" 2>/dev/null || python -m pip install pytest; \\
    if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi; \\
    if [ -f pyproject.toml ] || [ -f setup.py ]; then \\
        python -m pip install -e ".[test,tests,testing,dev]" || python -m pip install -e .; \\
    fi; \\
    for f in requirements-dev.txt requirements-test.txt requirements-tests.txt \\
             requirements_dev.txt requirements_test.txt test-requirements.txt \\
             requirements/dev.txt requirements/test.txt requirements/tests.txt \\
             requirements/testing.txt; do \\
        if [ -f "$f" ]; then python -m pip install -r "$f" || echo "warning: could not install $f"; fi; \\
    done; \\
    for g in test tests dev; do python -m pip install --group "$g" >/dev/null 2>&1 || true; done; \\
    python -m pytest --version
"""

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


def build_image(
    source_dir: Path,
    tag: str,
    *,
    base_image: str = "python:3.12-slim",
    timeout: float = 1800,
    docker: str = "docker",
    ca_bundle: str | Path | None = None,
) -> RunResult:
    """Build an image with the repo's dependencies installed (network allowed here only).

    `source_dir` must be a clean checkout of the base commit. `ca_bundle` (default:
    $FIXLOOP_BUILD_CA_BUNDLE) is an extra CA file for pip behind a TLS-intercepting
    proxy; it is mounted as a build secret and not stored in the image.
    Raises SandboxError if the build fails.
    """
    ca_bundle = ca_bundle or os.environ.get("FIXLOOP_BUILD_CA_BUNDLE") or None
    with tempfile.TemporaryDirectory(prefix="fixloop-build-") as tmp:
        dockerfile = Path(tmp) / "Dockerfile"
        dockerfile.write_text(_DOCKERFILE.format(base_image=base_image, workdir=WORKDIR))
        argv = [docker, "build", "-f", str(dockerfile), "-t", tag, "--label", "fixloop=1"]
        for var in _PROXY_VARS:
            if os.environ.get(var):
                argv += ["--build-arg", f"{var}={os.environ[var]}"]
        if ca_bundle:
            argv += ["--secret", f"id=ca,src={Path(ca_bundle).resolve()}"]
        argv.append(str(Path(source_dir).resolve()))

        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout
            )
        except FileNotFoundError:
            raise SandboxError(f"{docker!r} not found; install Docker") from None
        except subprocess.TimeoutExpired:
            raise SandboxError(f"image build timed out after {timeout:.0f}s") from None
        output = _clip(proc.stdout.decode("utf-8", errors="replace"))
        if proc.returncode != 0:
            raise SandboxError(f"image build failed:\n{output[-4000:]}")
        return RunResult(argv, proc.returncode, False, output, time.monotonic() - start)
