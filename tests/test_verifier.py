"""Verifier decision logic, end to end on a toy repo with a local (non-Docker) sandbox."""

import json

import pytest

from toyrepo import (
    CALC, EXISTING_TESTS, FIXED_CALC, HELDOUT, REPRO, LocalSandbox, git, make_patch, make_repo,
)
from verifier import RepoSource, TestFile, Verifier, VerifierConfig, VerifierError
from verifier.policy import PatchPolicy

REPRO_TEST = TestFile("tests/test_fixloop_repro.py", REPRO)
HELDOUT_TEST = TestFile("tests/test_fixloop_heldout.py", HELDOUT)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture
def sandbox():
    return LocalSandbox()


@pytest.fixture
def verifier(tmp_path, repo, sandbox):
    path, commit = repo
    v = Verifier(RepoSource(path, commit), sandbox, tmp_path / "evidence", work_root=tmp_path)
    yield v
    v.close()


@pytest.fixture
def ready(verifier):
    baseline = verifier.establish_baseline(REPRO_TEST)
    assert baseline.ok, baseline.feedback()
    return verifier


def codes(result):
    return [r.code for r in result.reasons]


# --- baseline ------------------------------------------------------------------

def test_baseline_requires_failing_repro_and_records_existing_failures(verifier):
    baseline = verifier.establish_baseline(REPRO_TEST)
    assert baseline.ok
    assert baseline.targeted.outcome("tests/test_fixloop_repro.py::test_clamp_above_range_returns_hi") == "failed"
    assert baseline.suite.failing() == {"tests/test_calc.py::test_known_broken"}
    assert "tests/test_fixloop_repro.py::test_clamp_above_range_returns_hi" not in baseline.suite.tests
    saved = json.loads((baseline.evidence_dir / "baseline.json").read_text())
    assert saved["existing_failures"] == ["tests/test_calc.py::test_known_broken"]
    assert (baseline.evidence_dir / "frozen/tests/test_fixloop_repro.py").read_text() == REPRO


@pytest.mark.parametrize("content,code", [
    ("from calc import clamp\n\ndef test_ok():\n    assert clamp(2, 0, 3) == 2\n", "repro_passes_on_baseline"),
    ("from calc import clampp\n\ndef test_x():\n    pass\n", "repro_collection_error"),
    ("def test_x():\n    assert undefined_name == 1\n", "repro_fails_for_wrong_reason"),
    ("def helper():\n    pass\n", "repro_no_tests"),
    ("import pytest\n\n@pytest.mark.xfail\ndef test_x():\n    assert False\n", "repro_uses_skip"),
    ("import pytest\n\n@pytest.fixture\ndef boom():\n    raise RuntimeError\n\n"
     "def test_x(boom):\n    pass\n", "repro_error"),
])
def test_baseline_rejects_bad_repro_tests(verifier, content, code):
    baseline = verifier.establish_baseline(TestFile("tests/test_fixloop_repro.py", content))
    assert not baseline.ok
    assert codes(baseline) == [code]


def test_baseline_rejects_existing_or_bad_paths(verifier):
    assert codes(verifier.establish_baseline(TestFile("tests/test_calc.py", REPRO))) == ["repro_path_exists"]
    assert codes(verifier.establish_baseline(TestFile("../test_x.py", REPRO))) == ["bad_test_path"]
    assert codes(verifier.establish_baseline(TestFile("tests/repro.py", REPRO))) == ["bad_test_path"]


def test_repro_is_frozen_once_accepted(ready):
    with pytest.raises(VerifierError, match="frozen"):
        ready.establish_baseline(TestFile("tests/test_other.py", REPRO))


def test_verify_requires_baseline(verifier):
    with pytest.raises(VerifierError):
        verifier.verify(make_patch(verifier.source.path, {"calc.py": FIXED_CALC}))


# --- verify ------------------------------------------------------------------------

def test_accepts_real_fix_and_saves_evidence(ready, repo):
    verdict = ready.verify(make_patch(repo[0], {"calc.py": FIXED_CALC}))
    assert verdict.accepted, verdict.feedback()
    assert verdict.feedback() == "accepted"
    out = verdict.evidence_dir
    assert json.loads((out / "verdict.json").read_text())["accepted"] is True
    assert "1 passed" in (out / "targeted.log").read_text()
    assert (out / "suite.log").exists() and (out / "patch.diff").exists()


def test_rejects_patch_touching_tests_without_running_anything(ready, repo, sandbox):
    calls_before = len(sandbox.calls)
    patch = make_patch(repo[0], {
        "calc.py": FIXED_CALC,
        "tests/test_calc.py": EXISTING_TESTS.replace("== 0.3", "!= 0.3"),
    })
    verdict = ready.verify(patch)
    assert codes(verdict) == ["touches_test_file"]
    assert "tests/test_calc.py" in verdict.feedback()
    assert len(sandbox.calls) == calls_before


@pytest.mark.parametrize("edits,code", [
    ({"conftest.py": "def pytest_collection_modifyitems(items):\n    items.clear()\n"}, "touches_test_config"),
    ({".github/workflows/ci.yml": "on: push\n"}, "touches_ci"),
    ({"calc.py": "import pytest\n" + CALC.replace("return lo\n    return x", "pytest.skip()\n    return x")},
     "adds_skip_marker"),
])
def test_policy_rejections(ready, repo, edits, code):
    assert code in codes(ready.verify(make_patch(repo[0], edits)))


def test_rejects_non_fix_with_failure_detail(ready, repo):
    patch = make_patch(repo[0], {"calc.py": CALC.replace("return x", "return x  # no-op")})
    verdict = ready.verify(patch)
    assert codes(verdict) == ["repro_still_failing"]
    assert "assert 0 == 3" in verdict.feedback()


def test_rejects_regression_against_baseline(ready, repo):
    broken = FIXED_CALC.replace("return a + b", "return a - b")
    verdict = ready.verify(make_patch(repo[0], {"calc.py": broken}))
    assert codes(verdict) == ["regression"]
    assert verdict.reasons[0].items == ["tests/test_calc.py::test_add"]
    assert "test_known_broken" not in verdict.feedback()  # already failing at baseline


def test_rejects_new_collection_error_in_suite(ready, repo):
    patch = make_patch(repo[0], {"calc.py": FIXED_CALC.replace("def add", "def add_renamed")})
    verdict = ready.verify(patch)
    assert codes(verdict) == ["regression"]
    assert verdict.reasons[0].items == ["<collect> tests/test_calc.py"]  # reported once, not per test


def test_rejects_large_and_unappliable_patches(ready, repo):
    big = make_patch(repo[0], {"calc.py": FIXED_CALC + "\n".join(f"X{i} = {i}" for i in range(300))})
    assert codes(ready.verify(big)) == ["patch_too_large"]
    stale = make_patch(repo[0], {"calc.py": FIXED_CALC}).replace("return lo\n", "return LO\n")
    assert codes(ready.verify(stale)) == ["patch_does_not_apply"]


def test_workspace_changes_after_setup_do_not_affect_verifier(verifier, repo):
    path, _ = repo
    # The agent "fixes" the bug by committing directly in its workspace.
    (path / "calc.py").write_text(FIXED_CALC)
    git(path, "commit", "-qam", "sneaky")
    assert verifier.establish_baseline(REPRO_TEST).ok  # still checks the pinned base commit


def test_heldout_tests_catch_special_cased_fix(verifier, repo):
    baseline = verifier.establish_baseline(REPRO_TEST, heldout=[HELDOUT_TEST])
    assert baseline.ok
    special_cased = CALC.replace("    if x > hi:", "    if (x, lo, hi) == (5, 0, 3):\n        return 3\n    if x > hi:")
    verdict = verifier.verify(make_patch(repo[0], {"calc.py": special_cased}))
    assert codes(verdict) == ["heldout_failed"]
    assert "test_clamp_far_above_range" not in verdict.feedback()  # fixer never sees held-out tests

    verdict = verifier.verify(make_patch(repo[0], {"calc.py": FIXED_CALC}))
    assert verdict.accepted, verdict.feedback()
    assert set(verdict.heldout.values()) == {"passed"}


def test_heldout_report_only_mode(tmp_path, repo, sandbox):
    v = Verifier(RepoSource(*repo), sandbox, tmp_path / "ev", VerifierConfig(heldout_required=False),
                 work_root=tmp_path)
    assert v.establish_baseline(REPRO_TEST, heldout=[HELDOUT_TEST]).ok
    special_cased = CALC.replace("    if x > hi:", "    if (x, lo, hi) == (5, 0, 3):\n        return 3\n    if x > hi:")
    verdict = v.verify(make_patch(repo[0], {"calc.py": special_cased}))
    assert verdict.accepted
    assert "failed" in verdict.heldout.values()
    v.close()


def test_flaky_test_is_confirmed_on_baseline_and_ignored(ready, repo, sandbox):
    sandbox.extra_env["TOY_FLAKY"] = "fail"  # environment drifts after the baseline run
    verdict = ready.verify(make_patch(repo[0], {"calc.py": FIXED_CALC}))
    assert verdict.accepted, verdict.feedback()
    assert verdict.flaky == ["tests/test_calc.py::test_env_dependent"]


def test_size_policy_is_configurable(tmp_path, repo, sandbox):
    v = Verifier(RepoSource(*repo), sandbox, tmp_path / "ev",
                 VerifierConfig(policy=PatchPolicy(max_changed_lines=1)), work_root=tmp_path)
    assert v.establish_baseline(REPRO_TEST).ok
    assert codes(v.verify(make_patch(repo[0], {"calc.py": FIXED_CALC}))) == ["patch_too_large"]
    v.close()
