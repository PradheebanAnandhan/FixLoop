"""Docker integration tests: real containers, real isolation. Skipped without Docker.

The test image gets pytest from wheels downloaded on the host, so the image
build itself needs no network inside Docker.
"""

import subprocess

import pytest

from toyrepo import FIXED_CALC, REPRO, make_patch, make_repo
from verifier import (
    DockerSandbox, RepoSource, SandboxConfig, TestFile, Verifier, build_image,
)

pytestmark = pytest.mark.docker

@pytest.fixture(scope="session")
def toy_image(base_image, tmp_path_factory):
    repo, _ = make_repo(tmp_path_factory.mktemp("img"))
    tag = "fixloop-test-toy:latest"
    build_image(repo, tag, base_image=base_image, timeout=600)
    return tag


@pytest.fixture
def sandbox(toy_image):
    return DockerSandbox(SandboxConfig(image=toy_image))


def test_no_network(sandbox, tmp_path):
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('BLOCKED', e)\n"
    )
    result = sandbox.run(tmp_path, ["python", "-c", code], timeout=30)
    assert "BLOCKED" in result.output and "CONNECTED" not in result.output


def test_environment_is_cleared(sandbox, tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "secret-should-not-leak")
    result = sandbox.run(tmp_path, ["python", "-c", "import os; print(sorted(os.environ))"], timeout=30)
    assert "NEBIUS_API_KEY" not in result.output
    assert "GPG_KEY" not in result.output  # image ENV is dropped too
    assert "'HOME'" in result.output


def test_repo_mount_and_root_fs_are_read_only(sandbox, tmp_path):
    (tmp_path / "f.txt").write_text("x")
    code = (
        "for p in ['/work/f.txt', '/work/new.txt', '/usr/local/lib/x']:\n"
        "    try:\n"
        "        open(p, 'w').write('y'); print('WROTE', p)\n"
        "    except OSError:\n"
        "        print('DENIED', p)\n"
        "open('/tmp/ok', 'w').write('fine'); print('TMP OK')\n"
    )
    result = sandbox.run(tmp_path, ["python", "-c", code], timeout=30)
    assert "WROTE" not in result.output
    assert result.output.count("DENIED") == 3 and "TMP OK" in result.output
    assert (tmp_path / "f.txt").read_text() == "x"


def test_timeout_kills_container(sandbox, tmp_path):
    result = sandbox.run(tmp_path, ["sleep", "60"], timeout=3)
    assert result.timed_out and result.exit_code is None
    assert result.duration_s < 30
    ps = subprocess.run(["docker", "ps", "-q", "--filter", "name=fixloop-"], capture_output=True, text=True)
    assert ps.stdout.strip() == ""


def test_full_verification_in_docker(sandbox, tmp_path):
    repo, commit = make_repo(tmp_path)
    with Verifier(RepoSource(repo, commit), sandbox, tmp_path / "evidence", work_root=tmp_path) as v:
        baseline = v.establish_baseline(TestFile("tests/test_fixloop_repro.py", REPRO))
        assert baseline.ok, baseline.feedback()
        assert baseline.suite.failing() == {"tests/test_calc.py::test_known_broken"}

        regression = v.verify(make_patch(repo, {"calc.py": FIXED_CALC.replace("a + b", "a * b")}))
        assert [r.code for r in regression.reasons] == ["regression"]

        verdict = v.verify(make_patch(repo, {"calc.py": FIXED_CALC}))
        assert verdict.accepted, verdict.feedback()
        assert "1 passed" in (verdict.evidence_dir / "targeted.log").read_text()


def test_patched_code_cannot_rewrite_injected_test(sandbox, tmp_path):
    repo, commit = make_repo(tmp_path)
    tamper = (
        "import pathlib\n"
        "try:\n"
        "    pathlib.Path(__file__).with_name('tests').joinpath('test_fixloop_repro.py')"
        ".write_text('def test_x():\\n    pass\\n')\n"
        "except OSError:\n"
        "    pass\n"
    )
    with Verifier(RepoSource(repo, commit), sandbox, tmp_path / "evidence", work_root=tmp_path) as v:
        assert v.establish_baseline(TestFile("tests/test_fixloop_repro.py", REPRO)).ok
        verdict = v.verify(make_patch(repo, {"calc.py": tamper + open(repo / "calc.py").read()}))
        assert not verdict.accepted
        assert [r.code for r in verdict.reasons] == ["repro_still_failing"]
