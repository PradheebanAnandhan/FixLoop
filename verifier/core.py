"""The independent verifier: deterministic accept/reject for a candidate patch.

No LLM calls happen here. The flow for one issue:

    v = Verifier(RepoSource(repo_path, base_commit), sandbox, evidence_dir)
    baseline = v.establish_baseline(repro_test)   # repro test must FAIL on clean code
    verdict = v.verify(patch_text)                # repeat per fix attempt

Every run starts from a fresh clone of the pinned base commit in a directory
the verifier owns. The reproducing test (and any held-out tests) is frozen in
memory when the baseline is accepted and injected by the verifier itself, so
the agent cannot alter it. Evidence (logs, reports, verdicts) is written under
`evidence_dir`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Sequence

from .policy import PatchPolicy, check_paths, check_patch
from .reasons import Reason
from .sandbox import RunResult, Sandbox

FAILING = ("failed", "error")
WRONG_REASON_ERRORS = ("NameError", "SyntaxError", "IndentationError", "ImportError",
                       "ModuleNotFoundError")
SKIP_IN_TEST = re.compile(r"mark\s*\.\s*(skip|skipif|xfail)\b|pytest\s*\.\s*(skip|xfail|importorskip)\b")
DETAIL_CHARS = 1500


class VerifierError(RuntimeError):
    """Misuse or infrastructure failure (not a rejection of the patch)."""


@dataclass(frozen=True)
class TestFile:
    """A test file the verifier injects into the checkout (never part of the patch)."""

    path: str  # repo-relative POSIX path, e.g. "tests/test_fixloop_repro.py"
    content: str

    __test__ = False  # not a pytest test class

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


@dataclass(frozen=True)
class RepoSource:
    path: str | Path  # git repository (local path or URL) to clone from
    commit: str       # base commit (any rev reachable from the source's refs or HEAD)


@dataclass(frozen=True)
class VerifierConfig:
    policy: PatchPolicy = PatchPolicy()
    targeted_timeout: float = 300
    suite_timeout: float = 1200
    suite_args: tuple[str, ...] = ()        # extra args for the existing-suite run
    confirm_regressions: bool = True       # re-run new failures on clean code to spot flakes
    heldout_required: bool = True          # held-out tests must pass (else report only)
    fail_fast: bool = True                 # skip the suite run if the repro test still fails

    @classmethod
    def from_env(cls) -> "VerifierConfig":
        def num(name, default, cast=int):
            raw = os.environ.get(name, "").strip()
            return cast(raw) if raw else default
        return cls(
            policy=PatchPolicy(
                max_changed_lines=num("VERIFIER_MAX_PATCH_LINES", 200),
                max_files=num("VERIFIER_MAX_PATCH_FILES", 6),
            ),
            targeted_timeout=num("VERIFIER_TARGETED_TIMEOUT", 300, float),
            suite_timeout=num("VERIFIER_SUITE_TIMEOUT", 1200, float),
        )


# --- run summaries --------------------------------------------------------------

@dataclass
class RunSummary:
    """Per-test outcomes from one sandboxed pytest run."""

    name: str
    exit_code: int | None
    timed_out: bool
    duration_s: float
    tests: dict[str, dict]
    collect_errors: dict[str, str]
    has_report: bool

    @classmethod
    def from_result(cls, name: str, result: RunResult) -> "RunSummary":
        report = result.report or {}
        return cls(
            name=name,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_s=round(result.duration_s, 2),
            tests=report.get("tests", {}),
            collect_errors=report.get("collect_errors", {}),
            has_report=result.report is not None,
        )

    def outcome(self, nodeid: str) -> str | None:
        entry = self.tests.get(nodeid)
        return entry["outcome"] if entry else None

    def in_file(self, path: str) -> dict[str, dict]:
        return {n: t for n, t in self.tests.items() if n.split("::", 1)[0] == path}

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tests.values():
            counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
        if self.collect_errors:
            counts["collection_errors"] = len(self.collect_errors)
        return counts

    def failing(self) -> set[str]:
        return {n for n, t in self.tests.items() if t["outcome"] in FAILING} | {
            f"<collect> {n}" for n in self.collect_errors
        }

    def passed(self) -> set[str]:
        return {n for n, t in self.tests.items() if t["outcome"] == "passed"}

    def consistent(self) -> bool:
        """The report must agree with pytest's own exit status."""
        if not self.has_report or self.timed_out:
            return False
        clean = not self.failing()
        return (self.exit_code == 0) == clean or self.exit_code == 5

    def to_dict(self) -> dict:
        return {
            "name": self.name, "exit_code": self.exit_code, "timed_out": self.timed_out,
            "duration_s": self.duration_s, "counts": self.counts(),
        }


def _failure_detail(summary: RunSummary, nodeids: Sequence[str], limit: int = DETAIL_CHARS) -> str:
    parts = []
    for n in nodeids:
        if n.startswith("<collect> "):
            text = summary.collect_errors.get(n[len("<collect> "):], "")
        else:
            entry = summary.tests.get(n, {})
            text = entry.get("excerpt") or entry.get("crash") or ""
        if text:
            parts.append(f"--- {n}\n{text.strip()[-limit:]}")
    return "\n".join(parts)[-limit:]


# --- results ------------------------------------------------------------------------

@dataclass
class Baseline:
    ok: bool
    reasons: list[Reason]
    commit: str
    repro: TestFile
    heldout: list[TestFile]
    targeted: RunSummary | None
    suite: RunSummary | None
    evidence_dir: Path

    def feedback(self) -> str:
        return _feedback(self.reasons)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": [r.to_dict() for r in self.reasons],
            "commit": self.commit,
            "repro_test": {"path": self.repro.path, "sha256": self.repro.sha256},
            "heldout_tests": [{"path": t.path, "sha256": t.sha256} for t in self.heldout],
            "targeted_run": self.targeted.to_dict() if self.targeted else None,
            "suite_run": self.suite.to_dict() if self.suite else None,
            "existing_failures": sorted(self.suite.failing()) if self.suite else [],
        }


@dataclass
class Verdict:
    accepted: bool
    attempt: int
    reasons: list[Reason]
    evidence_dir: Path
    runs: list[RunSummary] = field(default_factory=list)
    flaky: list[str] = field(default_factory=list)
    heldout: dict[str, str] = field(default_factory=dict)

    def feedback(self) -> str:
        """Short text for the agent's next attempt."""
        return "accepted" if self.accepted else _feedback(self.reasons)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "attempt": self.attempt,
            "reasons": [r.to_dict() for r in self.reasons],
            "runs": [r.to_dict() for r in self.runs],
            "flaky_ignored": self.flaky,
            "heldout": self.heldout,
        }


def _feedback(reasons: list[Reason]) -> str:
    lines = []
    for r in reasons:
        lines.append(f"- [{r.code}] {r.message}")
        if r.detail:
            lines.append("  " + r.detail.replace("\n", "\n  "))
    return "\n".join(lines)


# --- the verifier -------------------------------------------------------------------

def _git(cwd: Path, *args: str, check: bool = True, input: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"}
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, env=env, input=input
    )
    if check and proc.returncode != 0:
        raise VerifierError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def _validate_test_path(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or ".git" in p.parts:
        return "test path must be a relative path inside the repository"
    if p.suffix != ".py" or not p.name.startswith("test_"):
        return "test file name must look like test_*.py"
    return None


class Verifier:
    def __init__(
        self,
        source: RepoSource,
        sandbox: Sandbox,
        evidence_dir: Path,
        config: VerifierConfig = VerifierConfig(),
        work_root: Path | None = None,
    ) -> None:
        self.source = source
        self.sandbox = sandbox
        self.config = config
        self.evidence_dir = Path(evidence_dir)
        self.work_root = Path(work_root) if work_root else None
        self.baseline: Baseline | None = None
        self._baseline_attempts = 0
        self.attempts = 0

        # Private bare mirror: later changes to the source repo (e.g. the agent's
        # workspace) cannot affect what the verifier checks out.
        src = source.path
        if isinstance(src, Path) or Path(str(src)).exists():
            src = str(Path(src).resolve())
        self._mirror = Path(tempfile.mkdtemp(prefix="fixloop-mirror-", dir=self.work_root))
        _git(self._mirror, "clone", "-q", "--bare", "--no-hardlinks", src, ".")
        self.commit = _git(self._mirror, "rev-parse", "--verify", f"{source.commit}^{{commit}}").stdout.strip()

    def close(self) -> None:
        shutil.rmtree(self._mirror, ignore_errors=True)

    def __enter__(self) -> "Verifier":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- checkouts -----------------------------------------------------------------

    def _fresh_checkout(self) -> Path:
        dest = Path(tempfile.mkdtemp(prefix="fixloop-verify-", dir=self.work_root))
        _git(dest, "clone", "-q", "--no-checkout", "--no-hardlinks", str(self._mirror), ".")
        _git(dest, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", self.commit)
        return dest

    @staticmethod
    def _inject(checkout: Path, tests: Sequence[TestFile]) -> None:
        for t in tests:
            target = checkout / t.path
            if target.exists() or target.is_symlink():
                raise VerifierError(f"refusing to overwrite existing file {t.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(t.content)

    def _run(self, checkout: Path, name: str, args: list[str], timeout: float, out: Path) -> RunSummary:
        result = self.sandbox.run_pytest(checkout, args, timeout)
        (out / f"{name}.log").write_text(
            "$ pytest " + " ".join(args) + "\n\n" + result.output
            + ("\n\n[verifier] TIMED OUT\n" if result.timed_out else "")
        )
        if result.report is not None:
            (out / f"{name}.json").write_text(json.dumps(result.report, indent=1))
        return RunSummary.from_result(name, result)

    def _suite_args(self, injected: Sequence[TestFile]) -> list[str]:
        return ["-rfE", "--continue-on-collection-errors", *self.config.suite_args,
                *[f"--ignore={t.path}" for t in injected]]

    @staticmethod
    def _targeted_args(tests: Sequence[TestFile]) -> list[str]:
        return ["-rfE", *[t.path for t in tests]]

    # -- step 1: baseline -----------------------------------------------------------------

    def establish_baseline(self, repro: TestFile, heldout: Sequence[TestFile] = ()) -> Baseline:
        """Run the reproducing test on clean code; it must fail. Records suite failures.

        May be called again with a new test until one is accepted; after that the
        reproducing test is frozen.
        """
        if self.baseline and self.baseline.ok:
            raise VerifierError("baseline already established; the reproducing test is frozen")
        self._baseline_attempts += 1
        out = self.evidence_dir / f"baseline-{self._baseline_attempts:02d}"
        out.mkdir(parents=True, exist_ok=True)
        heldout = list(heldout)
        injected = [repro, *heldout]

        reasons = self._check_test_files(repro, heldout)
        targeted = suite = None
        if not reasons:
            checkout = self._fresh_checkout()
            try:
                existing = [t.path for t in injected if (checkout / t.path).exists()]
                if existing:
                    reasons.append(Reason(
                        "repro_path_exists",
                        f"test path already exists in the repo; use a new file: {', '.join(existing)}",
                        items=existing,
                    ))
                else:
                    self._inject(checkout, injected)
                    targeted = self._run(checkout, "targeted", self._targeted_args(injected),
                                         self.config.targeted_timeout, out)
                    reasons += self._judge_baseline_repro(targeted, repro)
                    heldout = self._valid_heldout(targeted, heldout)
                    if not reasons:
                        suite = self._run(checkout, "suite", self._suite_args(injected),
                                          self.config.suite_timeout, out)
                        reasons += self._judge_baseline_suite(suite)
            finally:
                shutil.rmtree(checkout, ignore_errors=True)

        baseline = Baseline(
            ok=not reasons, reasons=reasons, commit=self.commit, repro=repro, heldout=heldout,
            targeted=targeted, suite=suite, evidence_dir=out,
        )
        frozen = out / "frozen"
        for t in (repro, *heldout):
            (frozen / t.path).parent.mkdir(parents=True, exist_ok=True)
            (frozen / t.path).write_text(t.content)
        (out / "baseline.json").write_text(json.dumps(baseline.to_dict(), indent=2))
        self.baseline = baseline
        return baseline

    @staticmethod
    def _check_test_files(repro: TestFile, heldout: Sequence[TestFile]) -> list[Reason]:
        reasons = []
        paths = [t.path for t in (repro, *heldout)]
        if len(set(paths)) != len(paths):
            reasons.append(Reason("bad_test_path", "test file paths must be unique"))
        for t in (repro, *heldout):
            problem = _validate_test_path(t.path)
            if problem:
                reasons.append(Reason("bad_test_path", f"{t.path}: {problem}", items=[t.path]))
        if SKIP_IN_TEST.search(repro.content):
            reasons.append(Reason("repro_uses_skip", "reproducing test must not use skip/xfail"))
        return reasons

    def _judge_baseline_repro(self, run: RunSummary, repro: TestFile) -> list[Reason]:
        if run.timed_out:
            return [Reason("repro_timeout", f"reproducing test timed out after {self.config.targeted_timeout:.0f}s on clean code")]
        if not run.has_report:
            return [Reason("no_report", "test run produced no report", detail=_tail_note(run))]
        if repro.path in run.collect_errors:
            return [Reason(
                "repro_collection_error", "reproducing test fails to import/collect, not a genuine failure",
                detail=run.collect_errors[repro.path][-DETAIL_CHARS:],
            )]
        tests = run.in_file(repro.path)
        if not tests:
            return [Reason("repro_no_tests", "no tests were collected from the reproducing test file")]
        outcomes = {t["outcome"] for t in tests.values()}
        if outcomes & {"skipped", "xfailed", "xpassed"}:
            return [Reason("repro_uses_skip", "reproducing test was skipped or xfailed")]
        errors = [n for n, t in tests.items() if t["outcome"] == "error"]
        if errors:
            return [Reason("repro_error", "reproducing test errors in setup/teardown instead of failing",
                           items=errors, detail=_failure_detail(run, errors))]
        failed = [n for n, t in tests.items() if t["outcome"] == "failed"]
        if not failed:
            return [Reason("repro_passes_on_baseline",
                           "reproducing test passes on the original code, so it does not reproduce the bug")]
        wrong = [n for n in failed if any(e in tests[n].get("crash", "") for e in WRONG_REASON_ERRORS)]
        if wrong:
            return [Reason("repro_fails_for_wrong_reason",
                           "reproducing test fails with a name/import/syntax error rather than the bug",
                           items=wrong, detail=_failure_detail(run, wrong))]
        return []

    @staticmethod
    def _valid_heldout(run: RunSummary, heldout: list[TestFile]) -> list[TestFile]:
        # Held-out tests that cannot even be collected on clean code are discarded.
        return [t for t in heldout if t.path not in run.collect_errors and run.in_file(t.path)]

    def _judge_baseline_suite(self, run: RunSummary) -> list[Reason]:
        if run.timed_out:
            return [Reason("baseline_suite_timeout",
                           f"existing test suite timed out after {self.config.suite_timeout:.0f}s on clean code")]
        if not run.has_report or run.exit_code not in (0, 1, 5):
            return [Reason("baseline_suite_unusable",
                           f"existing test suite could not run on clean code (exit {run.exit_code})",
                           detail=_tail_note(run))]
        return []

    # -- steps 2-5: verify a patch --------------------------------------------------------

    def verify(self, patch: str) -> Verdict:
        if not (self.baseline and self.baseline.ok):
            raise VerifierError("establish an accepted baseline before verifying patches")
        self.attempts += 1
        out = self.evidence_dir / f"attempt-{self.attempts:02d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "patch.diff").write_text(patch)

        verdict = self._verify(patch, out)
        (out / "verdict.json").write_text(json.dumps(verdict.to_dict(), indent=2))
        (out / "feedback.txt").write_text(verdict.feedback() + "\n")
        return verdict

    def _verify(self, patch: str, out: Path) -> Verdict:
        base = self.baseline
        verdict = Verdict(accepted=False, attempt=self.attempts, reasons=[], evidence_dir=out)

        # (2) static policy, before anything runs
        verdict.reasons = check_patch(patch, self.config.policy)
        if verdict.reasons:
            return verdict

        checkout = self._fresh_checkout()
        try:
            # (3) the verifier applies the patch itself
            patch_file = out / "patch.diff"
            applied = _git(checkout, "apply", "--whitespace=nowarn", str(patch_file.resolve()), check=False)
            if applied.returncode != 0:
                verdict.reasons = [Reason("patch_does_not_apply", "patch does not apply to the base commit",
                                          detail=applied.stderr.strip()[-DETAIL_CHARS:])]
                return verdict
            changed, added = self._changed_paths(checkout)
            verdict.reasons = check_paths(changed, added)
            if not changed:
                verdict.reasons.append(Reason("empty_patch", "patch applies but changes nothing"))
            if verdict.reasons:
                return verdict

            injected = [base.repro, *base.heldout]
            self._inject(checkout, injected)

            # (3) reproducing test (and held-out tests) must now pass
            targeted = self._run(checkout, "targeted", self._targeted_args(injected),
                                 self.config.targeted_timeout, out)
            verdict.runs.append(targeted)
            verdict.reasons += self._judge_patched_repro(targeted)
            verdict.heldout, heldout_reasons = self._judge_heldout(targeted)
            if self.config.heldout_required:
                verdict.reasons += heldout_reasons
            if verdict.reasons and self.config.fail_fast:
                return verdict

            # (4) existing suite: no new failures versus baseline
            suite = self._run(checkout, "suite", self._suite_args(injected), self.config.suite_timeout, out)
            verdict.runs.append(suite)
            verdict.reasons += self._judge_suite(suite, verdict, out)
        finally:
            shutil.rmtree(checkout, ignore_errors=True)

        verdict.accepted = not verdict.reasons
        return verdict

    @staticmethod
    def _changed_paths(checkout: Path) -> tuple[set[str], set[str]]:
        """(all changed paths, newly added paths), as git sees the applied patch."""
        _git(checkout, "add", "-A")
        fields = _git(checkout, "diff", "--cached", "--name-status", "--no-renames", "-z").stdout.split("\0")
        pairs = list(zip(fields[0::2], fields[1::2]))
        return {path for _, path in pairs if path}, {path for status, path in pairs if status == "A"}

    def _judge_patched_repro(self, run: RunSummary) -> list[Reason]:
        repro = self.baseline.repro
        if run.timed_out:
            return [Reason("repro_timeout", f"tests timed out after {self.config.targeted_timeout:.0f}s with the patch applied")]
        if not run.has_report:
            return [Reason("no_report", "test run produced no report", detail=_tail_note(run))]
        if repro.path in run.collect_errors:
            return [Reason("repro_collection_error", "reproducing test no longer imports with the patch applied",
                           detail=run.collect_errors[repro.path][-DETAIL_CHARS:])]
        expected = set(self.baseline.targeted.in_file(repro.path))
        now = run.in_file(repro.path)
        missing = sorted(expected - set(now))
        not_passing = sorted(n for n, t in now.items() if t["outcome"] != "passed")
        if missing:
            return [Reason("repro_not_run", "reproducing test cases did not run with the patch applied",
                           items=missing)]
        if not_passing:
            return [Reason("repro_still_failing", "reproducing test still fails with the patch applied",
                           items=not_passing, detail=_failure_detail(run, not_passing))]
        if not run.consistent():
            return [Reason("inconsistent_run", "pytest exit status disagrees with the test report",
                           detail=_tail_note(run))]
        return []

    def _judge_heldout(self, run: RunSummary) -> tuple[dict[str, str], list[Reason]]:
        results: dict[str, str] = {}
        for t in self.baseline.heldout:
            for nodeid, entry in run.in_file(t.path).items():
                results[nodeid] = entry["outcome"]
            if t.path in run.collect_errors:
                results[t.path] = "collection_error"
        failing = [n for n, o in results.items() if o != "passed"]
        if not failing:
            return results, []
        # Deliberately no test names or output: the fixer must not see held-out tests.
        return results, [Reason("heldout_failed",
                                f"{len(failing)} held-out edge-case test(s) fail; the fix may only cover the reproducing case")]

    def _judge_suite(self, run: RunSummary, verdict: Verdict, out: Path) -> list[Reason]:
        if run.timed_out:
            return [Reason("suite_timeout", f"existing test suite timed out after {self.config.suite_timeout:.0f}s with the patch applied")]
        if not run.has_report or run.exit_code not in (0, 1, 5):
            return [Reason("suite_crashed", f"existing test suite crashed with the patch applied (exit {run.exit_code})",
                           detail=_tail_note(run))]

        before = self.baseline.suite
        new_failures = run.failing() - before.failing()
        # A file that newly fails to collect is reported once, not test by test.
        broken_files = {n[len("<collect> "):] for n in new_failures if n.startswith("<collect> ")}
        stopped_passing = {
            n for n in before.passed()
            if run.outcome(n) != "passed" and n.split("::", 1)[0] not in broken_files
        }
        regressions = sorted(new_failures | stopped_passing)
        if regressions and self.config.confirm_regressions:
            flaky = self._flaky(regressions, out)
            verdict.flaky = sorted(flaky)
            regressions = [n for n in regressions if n not in flaky]
        if not regressions:
            if not run.consistent():
                return [Reason("inconsistent_run", "pytest exit status disagrees with the test report",
                               detail=_tail_note(run))]
            return []
        shown = regressions[:5]
        more = f" (+{len(regressions) - 5} more)" if len(regressions) > 5 else ""
        return [Reason(
            "regression",
            f"{len(regressions)} previously passing test(s) now fail: {', '.join(shown)}{more}",
            items=regressions,
            detail=_failure_detail(run, [n for n in shown if run.outcome(n) in FAILING or n.startswith("<collect>")]),
        )]

    def _flaky(self, regressions: list[str], out: Path) -> set[str]:
        """Re-run regressed tests on clean code; any that don't pass there are flaky."""
        rerun = [n for n in regressions if not n.startswith("<collect>") and "::" in n][:50]
        if not rerun:
            return set()
        checkout = self._fresh_checkout()
        try:
            run = self._run(checkout, "confirm-on-baseline", ["-rfE", *rerun], self.config.targeted_timeout, out)
        finally:
            shutil.rmtree(checkout, ignore_errors=True)
        if run.timed_out or not run.has_report:
            return set()
        return {n for n in rerun if run.outcome(n) != "passed"}


def _tail_note(run: RunSummary) -> str:
    return f"see {run.name}.log in the evidence directory (exit {run.exit_code})"
