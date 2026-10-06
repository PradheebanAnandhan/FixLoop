"""Shared fixtures: a Docker base image with pytest preinstalled (skips without Docker)."""

import shutil
import subprocess
import sys

import pytest

BASE_TAG = "fixloop-test-base:py312"


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture(scope="session")
def base_image(tmp_path_factory):
    if not _docker_ok():
        pytest.skip("Docker daemon not available")
    if subprocess.run(["docker", "image", "inspect", BASE_TAG], capture_output=True).returncode == 0:
        return BASE_TAG
    ctx = tmp_path_factory.mktemp("base")
    dl = subprocess.run(
        [sys.executable, "-m", "pip", "download", "-q", "-d", str(ctx / "wheels"), "--only-binary=:all:",
         "--python-version", "3.12", "--platform", "manylinux2014_x86_64", "pytest"],
        capture_output=True, text=True,
    )
    if dl.returncode != 0:
        pytest.skip(f"could not download pytest wheels: {dl.stderr[-300:]}")
    (ctx / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY wheels /wheels\n"
        "RUN pip install -q --no-index --find-links /wheels pytest && rm -rf /wheels\n"
    )
    subprocess.run(["docker", "build", "-q", "-t", BASE_TAG, str(ctx)], check=True, capture_output=True)
    return BASE_TAG
