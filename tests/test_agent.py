"""End-to-end agent runs on the toy repo with a scripted model.

The fast tests use the local sandbox; the Docker test runs the same flow in
real containers with the default image-building sandbox factory.
"""

import json

import pytest

from fakes import first_user, last_user, make_settings, scripted_llm
from fixloop.agent import Agent, docker_sandbox_factory
from fixloop.github import Issue
from toyrepo import CALC, HELDOUT, REPRO, LocalSandbox, make_repo

ISSUE = Issue(
    owner="toy", repo="toy", number=7, url="https://github.com/toy/toy/issues/7",
    title="clamp() returns the lower bound for values above the range",
    body="`clamp(5, 0, 3)` returns `0` but should return `3`. Values above `hi` should clamp to `hi`.",
)

PASSING_TEST = "from calc import clamp\n\n\ndef test_mid():\n    assert clamp(2, 0, 3) == 2\n"


def edit(search, replace, path="calc.py"):
    return f'<edit file="{path}">\n<search>\n{search}</search>\n<replace>\n{replace}</replace>\n</edit>'


GOOD_EDIT = "Values above hi return lo instead of hi.\n" + edit(
    "    if x > hi:\n        return lo\n", "    if x > hi:\n        return hi\n")
STALE_EDIT = "Root cause.\n" + edit("    if x >= hi:\n        return lo\n", "    if x >= hi:\n        return hi\n")
REGRESSING_EDIT = "Root cause.\n" + edit(
    "def add(a, b):\n    return a + b\n", "def add(a, b):\n    return a - b\n") + "\n" + edit(
    "    if x > hi:\n        return lo\n", "    if x > hi:\n        return hi\n")


def happy_responder(model, messages):
    first, last = first_user(messages), last_user(messages)
    if "Summarize this GitHub issue" in first:
        return json.dumps({"summary": "clamp returns lo above the range", "expected": "returns hi",
                           "actual": "returns lo", "keywords": ["clamp"], "is_bug": True})
    if "Write a pytest test that reproduces" in first:
        if "That test was rejected" in last:
            return f"```python\n{REPRO}```"
        return f"Here it is:\n```python\n{PASSING_TEST}```"
    if "Decide whether this failing test" in first:
        return '{"reproduces": true, "explanation": "asserts clamp(5,0,3)==3"}'
    if "extra pytest tests" in first:
        return f"```python\n{HELDOUT}```"
    if "Fix this bug." in first:
        replies = [STALE_EDIT, REGRESSING_EDIT, '<read_file path="calc.py"/>', GOOD_EDIT]
        n = sum(1 for m in messages if m["role"] == "assistant")
        return replies[min(n, len(replies) - 1)]
    if "summary section of a pull request" in first:
        return "`clamp` returned `lo` for values above the range.\n\nIt now returns `hi`."
    raise AssertionError(f"unexpected prompt: {first[:200]}")


@pytest.fixture
def toy(tmp_path):
    return make_repo(tmp_path)


def run_agent(tmp_path, toy, responder, factory=None, **settings):
    settings = make_settings(tmp_path, **settings)
    llm, fake = scripted_llm(settings, responder)
    events = []
    agent = Agent(settings, llm, sandbox_factory=factory or (lambda *a: LocalSandbox()), on_event=events.append)
    result = agent.run(ISSUE, repo_source=str(toy[0]), commit=toy[1], run_dir=tmp_path / "run")
    return result, fake, events


def test_happy_path_fixes_issue_with_verifier_feedback_loop(tmp_path, toy):
    result, fake, events = run_agent(tmp_path, toy, happy_responder)
    assert result.status == "fixed", result.message
    assert result.base_commit == toy[1]
    assert result.repro_attempts == 2  # first test passed on the original code and was rejected
    codes = [a.reasons[0]["code"] if a.reasons else "accepted" for a in result.fix_attempts]
    assert codes == ["edit_failed", "regression", "accepted"]

    run = tmp_path / "run"
    pr = (run / "PR.md").read_text()
    assert "Fixes https://github.com/toy/toy/issues/7" in pr
    assert "It now returns `hi`." in pr
    assert "| Reproducing test | 1 failed | 1 passed |" in pr
    assert "Held-out edge-case tests (informational) | 1 passed, 1 failed | 2/2 passed" in pr
    assert "test_clamp_far_above_range" not in pr.split("Reproducing test output before")[1].split("</details>")[0]
    assert "already failing" in pr  # test_known_broken is a pre-existing failure
    assert "return hi" in (run / "fix.diff").read_text()
    pr_diff = (run / "pr.diff").read_text()
    assert "tests/test_fixloop_repro.py" in pr_diff and "calc.py" in pr_diff
    assert "Attempt 2: rejected" in (run / "REPORT.md").read_text()

    saved = json.loads((run / "result.json").read_text())
    assert saved["status"] == "fixed"
    assert set(saved["usage"]["by_role"]) == {"fast", "mid", "reasoning"}
    assert saved["usage"]["cost_usd"] > 0
    steps = {json.loads(l)["step"] for l in (run / "trace.jsonl").read_text().splitlines()}
    assert {"intake", "localize", "reproduce", "heldout", "baseline", "fix", "deliver", "done"} <= steps
    assert events[-1].data["status"] == "fixed"


def test_fixer_never_sees_heldout_tests(tmp_path, toy):
    _, fake, _ = run_agent(tmp_path, toy, happy_responder)
    fix_calls = [c for c in fake.calls if "Fix this bug." in first_user(c["messages"])]
    assert fix_calls
    for call in fix_calls:
        text = json.dumps(call["messages"])
        assert "test_clamp_far_above_range" not in text and "test_fixloop_heldout" not in text
    assert {c["model"] for c in fix_calls} == {"ultra"}


def test_models_are_used_for_their_roles(tmp_path, toy):
    _, fake, _ = run_agent(tmp_path, toy, happy_responder)
    steps = {"Summarize this GitHub issue": "nano", "extra pytest tests": "nano",
             "Write a pytest test that reproduces": "ultra", "Fix this bug.": "ultra",
             "Decide whether this failing test": "super", "summary section of a pull request": "super"}
    for marker, model in steps.items():
        used = {c["model"] for c in fake.calls if marker in first_user(c["messages"])}
        assert used == {model}, marker


def test_judge_can_reject_a_test(tmp_path, toy):
    judged = []

    def responder(model, messages):
        first = first_user(messages)
        if "Decide whether this failing test" in first:
            judged.append(1)
            if len(judged) == 1:
                return '{"reproduces": false, "explanation": "fails for an unrelated reason"}'
        if "Write a pytest test that reproduces" in first:
            return f"```python\n{REPRO}```"
        return happy_responder(model, messages)

    result, fake, events = run_agent(tmp_path, toy, responder)
    assert result.status == "fixed"
    assert result.repro_attempts == 2
    assert any("unrelated reason" in e.message for e in events)


def test_not_reproduced(tmp_path, toy):
    def responder(model, messages):
        if "Write a pytest test that reproduces" in first_user(messages):
            return f"```python\n{PASSING_TEST}```"
        return happy_responder(model, messages)

    result, _, _ = run_agent(tmp_path, toy, responder, max_repro_attempts=2)
    assert result.status == "not_reproduced"
    assert "repro_passes_on_baseline" in result.message
    assert json.loads((tmp_path / "run/result.json").read_text())["status"] == "not_reproduced"


def test_not_fixed_stops_at_max_attempts(tmp_path, toy):
    def responder(model, messages):
        if "Fix this bug." in first_user(messages):
            return "Root cause.\n" + edit("def add(a, b):\n", "def add(a, b):  # touched\n")
        return happy_responder(model, messages)

    result, fake, _ = run_agent(tmp_path, toy, responder, max_fix_attempts=2, heldout_mode="off")
    assert result.status == "not_fixed"
    assert [a.reasons[0]["code"] for a in result.fix_attempts] == ["repro_still_failing"] * 2
    assert "Attempt 2: rejected" in (tmp_path / "run/REPORT.md").read_text()
    assert not (tmp_path / "run/PR.md").exists()


def test_heldout_gate_rejects_special_cased_fix(tmp_path, toy):
    special = "Special case.\n" + edit("    if x > hi:\n", "    if (x, lo, hi) == (5, 0, 3):\n        return 3\n    if x > hi:\n")

    def responder(model, messages):
        if "Fix this bug." in first_user(messages):
            n = sum(1 for m in messages if m["role"] == "assistant")
            return special if n == 0 else GOOD_EDIT
        return happy_responder(model, messages)

    result, _, _ = run_agent(tmp_path, toy, responder, heldout_mode="gate")
    assert result.status == "fixed"
    assert [a.reasons[0]["code"] if a.reasons else "ok" for a in result.fix_attempts] == ["heldout_failed", "ok"]


def test_setup_failure_is_reported(tmp_path, toy):
    settings = make_settings(tmp_path)
    llm, _ = scripted_llm(settings, happy_responder)
    agent = Agent(settings, llm, sandbox_factory=lambda *a: LocalSandbox())
    result = agent.run(ISSUE, repo_source=str(tmp_path / "missing"), run_dir=tmp_path / "run2")
    assert result.status == "setup_failed"
    assert "git clone failed" in result.message


@pytest.mark.docker
def test_happy_path_in_docker(tmp_path, toy, base_image):
    result, _, _ = run_agent(tmp_path, toy, happy_responder, factory=docker_sandbox_factory,
                             sandbox_base_image=base_image)
    assert result.status == "fixed", result.message
    assert "| Reproducing test | 1 failed | 1 passed |" in (tmp_path / "run/PR.md").read_text()


def test_cli_offline_run(tmp_path, toy, monkeypatch, capsys):
    import fixloop.__main__ as cli

    settings = make_settings(tmp_path)
    llm, _ = scripted_llm(settings, happy_responder)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "Agent", lambda s, on_event: Agent(
        s, llm, sandbox_factory=lambda *a: LocalSandbox(), on_event=on_event))
    issue_file = tmp_path / "issue.json"
    issue_file.write_text(json.dumps(ISSUE.to_dict()))

    code = cli.main(["--issue-file", str(issue_file), "--repo", str(toy[0]), "--commit", toy[1],
                     "--run-dir", str(tmp_path / "cli-run"), "--heldout", "off", "--no-color"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "[fix]" in out and "ACCEPTED by the verifier" in out
    assert "status:   fixed" in out and "total cost: $" in out
    assert (tmp_path / "cli-run/PR.md").exists()
    assert "heldout" not in json.loads((tmp_path / "cli-run/result.json").read_text())["step_seconds"]


def test_usage_is_per_run_and_failed_steps_emit_fail(tmp_path, toy):
    settings = make_settings(tmp_path, heldout_mode="off")
    llm, _ = scripted_llm(settings, happy_responder)
    events = []
    agent = Agent(settings, llm, sandbox_factory=lambda *a: LocalSandbox(), on_event=events.append)
    first = agent.run(ISSUE, repo_source=str(toy[0]), commit=toy[1], run_dir=tmp_path / "r1")
    second = agent.run(ISSUE, repo_source=str(toy[0]), commit=toy[1], run_dir=tmp_path / "r2")
    assert first.usage == second.usage

    bad = agent.run(ISSUE, repo_source=str(tmp_path / "missing"), run_dir=tmp_path / "r3")
    assert bad.status == "setup_failed"
    assert any(e.step == "intake" and e.kind == "fail" for e in events)
