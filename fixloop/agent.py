"""The FixLoop agent: issue -> reproducing test -> verified fix -> PR description.

Steps and models:
    intake     clone the repo, build the sandbox image             (no LLM)
    localize   summarize the issue, pick relevant files            (Nano)
    reproduce  write a failing test; verifier checks it; judged    (Ultra; judge: Super)
    heldout    optional edge-case tests the fixer never sees       (Nano)
    baseline   verifier freezes the test and records the suite     (no LLM)
    fix        propose edits -> verifier accepts/rejects -> retry  (Ultra)
    deliver    diff + PR description with verifier evidence        (Super)

The agent never decides whether a fix is correct; only `verifier.verify` does.
Every step emits events (for the terminal and the web UI) and everything is
written under the run directory.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from verifier import (
    Baseline, DockerSandbox, RepoSource, SandboxConfig, SandboxError, TestFile, Verdict, Verifier,
    VerifierConfig, VerifierError, build_image,
)
from verifier.sandbox import Sandbox

from . import edits as edits_mod
from . import prompts
from . import repo as repo_mod
from .config import Settings
from .github import Issue
from .llm import LLMClient
from .localize import rank_files, traceback_files

SAMPLING = {"temperature": 0.6, "top_p": 0.95}  # recommended for Nemotron reasoning models
MAX_FILES = 4
MAX_READ_REQUESTS = 3
FAILURE_CHARS = 3000


@dataclass
class Event:
    step: str
    kind: str  # "start" | "info" | "done" | "fail"
    message: str
    data: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)


@dataclass
class Attempt:
    number: int
    accepted: bool
    reasons: list[dict]
    feedback: str
    patch: str
    explanation: str
    evidence_dir: str | None = None


@dataclass
class AgentResult:
    status: str  # fixed | not_fixed | not_reproduced | baseline_failed | setup_failed | error
    issue_url: str
    run_dir: str
    base_commit: str | None = None
    message: str = ""
    relevant_files: list[str] = field(default_factory=list)
    repro_test_path: str | None = None
    repro_attempts: int = 0
    fix_attempts: list[Attempt] = field(default_factory=list)
    patch: str = ""
    step_seconds: dict[str, float] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


SandboxFactory = Callable[["Agent", Issue, Path, str], Sandbox]


class StepFailed(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def docker_sandbox_factory(agent: "Agent", issue: Issue, workspace: Path, commit: str) -> Sandbox:
    """Build (or reuse) the per-repo image at the base commit, return a Docker sandbox."""
    tag = re.sub(r"[^a-z0-9._/-]", "-", f"fixloop/{issue.owner}-{issue.repo}".lower()) + f":{commit[:12]}"
    sandbox = DockerSandbox(SandboxConfig(image=tag))
    if sandbox.image_exists():
        agent.emit("intake", "info", f"reusing sandbox image {tag}")
        return sandbox
    agent.emit("intake", "info", f"building sandbox image {tag} (installs dependencies; can take a few minutes)")
    context = agent.run_dir / "build-context"
    repo_mod.export_tree(workspace, commit, context)
    try:
        result = build_image(context, tag, base_image=agent.settings.sandbox_base_image)
        (agent.run_dir / "build.log").write_text(result.output)
    finally:
        shutil.rmtree(context, ignore_errors=True)
    return sandbox


class Agent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        *,
        sandbox_factory: SandboxFactory = docker_sandbox_factory,
        on_event: Callable[[Event], None] | None = None,
        verifier_config: VerifierConfig | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or LLMClient(settings)
        self.sandbox_factory = sandbox_factory
        self.on_event = on_event
        self.verifier_config = verifier_config or VerifierConfig.from_env()
        self.run_dir = Path()
        self._trace = None
        self._verifier: Verifier | None = None

    # -- events --------------------------------------------------------------------

    def emit(self, step: str, kind: str, message: str, **data) -> None:
        event = Event(step, kind, message, data)
        if self._trace:
            self._trace.write(json.dumps(asdict(event)) + "\n")
            self._trace.flush()
        if self.on_event:
            self.on_event(event)

    @contextmanager
    def _step(self, name: str, result: AgentResult, message: str):
        self.emit(name, "start", message)
        start = time.monotonic()
        try:
            yield
        except Exception as e:
            self.emit(name, "fail", str(e).splitlines()[0][:300] if str(e) else type(e).__name__)
            raise
        finally:
            result.step_seconds[name] = round(result.step_seconds.get(name, 0) + time.monotonic() - start, 2)

    # -- model helpers ---------------------------------------------------------------

    def _ask(self, role: str, messages: list[dict]) -> str:
        return self.llm.chat(role, _compact(messages), **SAMPLING).content

    def _ask_json(self, role: str, prompt: str, system: str = prompts.SYSTEM) -> dict | None:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        for _ in range(2):
            text = self._ask(role, messages)
            data = prompts.extract_json(text)
            if data is not None:
                return data
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Respond with only the JSON object, nothing else."}]
        return None

    # -- main entry point ---------------------------------------------------------------

    def run(self, issue: Issue, *, repo_source: str | None = None, commit: str | None = None,
            run_dir: Path | None = None) -> AgentResult:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir or self.settings.runs_dir / f"{issue.slug}-{stamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "issue.json").write_text(json.dumps(issue.to_dict(), indent=2))
        result = AgentResult(status="error", issue_url=issue.url, run_dir=str(self.run_dir))
        start = time.monotonic()
        self._usage_start = len(self.llm.usage.calls)
        self._trace = open(self.run_dir / "trace.jsonl", "a")
        try:
            self._run(issue, repo_source, commit, result)
        except StepFailed as e:
            result.status, result.message = e.status, str(e)
        except Exception as e:  # keep a record of anything unexpected
            result.status, result.message = "error", f"{type(e).__name__}: {e}"
            (self.run_dir / "error.txt").write_text(traceback.format_exc())
        finally:
            if self._verifier:
                self._verifier.close()
                self._verifier = None
            result.total_seconds = round(time.monotonic() - start, 2)
            result.usage = self.usage_report()
            (self.run_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
            self.emit("done", "done" if result.status == "fixed" else "fail",
                      f"{result.status}: {result.message}", status=result.status, run_dir=str(self.run_dir))
            self._trace.close()
            self._trace = None
        return result

    def _run(self, issue: Issue, repo_source: str | None, commit: str | None, result: AgentResult) -> None:
        workspace = self.run_dir / "repo"

        # 1. intake ------------------------------------------------------------------
        with self._step("intake", result, f"cloning {issue.owner}/{issue.repo}"):
            try:
                base = repo_mod.clone(repo_source or issue.clone_url, workspace, commit)
            except repo_mod.RepoError as e:
                raise StepFailed("setup_failed", str(e))
            result.base_commit = base
            self.emit("intake", "info", f"base commit {base[:12]}", commit=base)
            try:
                sandbox = self.sandbox_factory(self, issue, workspace, base)
            except SandboxError as e:
                raise StepFailed("setup_failed", f"sandbox setup failed: {e}")
            verifier = self._verifier = Verifier(RepoSource(workspace, base), sandbox,
                                                 self.run_dir / "evidence", self._verifier_config())
            self.emit("intake", "done", "repository and sandbox ready")

        issue_text = issue.as_text()

        # 2. localize --------------------------------------------------------------------
        with self._step("localize", result, "summarizing the issue and finding relevant files"):
            summary = self._ask_json("fast", prompts.summarize(issue_text, f"{issue.owner}/{issue.repo}")) or {}
            summary.setdefault("summary", issue.title)
            self.emit("localize", "info", f"summary: {summary.get('summary', '')}", summary=summary)
            if summary.get("is_bug") is False:
                self.emit("localize", "info", "warning: the issue may not describe a bug; continuing anyway")
            files = self._localize(workspace, issue_text, summary)
            result.relevant_files = files
            self.emit("localize", "done", f"relevant files: {', '.join(files)}", files=files)

        # 3. reproduce -------------------------------------------------------------------
        test_dir = repo_mod.test_dir(workspace)
        repro_path = self._free_path(workspace, test_dir, "test_fixloop_repro.py")
        with self._step("reproduce", result, f"writing a failing test at {repro_path}"):
            repro, failure = self._reproduce(issue_text, summary, workspace, files, repro_path, verifier, result)
            (self.run_dir / "repro_test.py").write_text(repro.content)
            result.repro_test_path = repro.path
            extra = [f for f in traceback_files(failure, repo_mod.source_files(workspace)) if f not in files]
            if extra:
                files = (extra + files)[:MAX_FILES + 2]
                result.relevant_files = files
                self.emit("reproduce", "info", f"failure traceback adds: {', '.join(extra)}")
            self.emit("reproduce", "done", "reproducing test fails for the reported reason", test=repro.content)

        # 4. held-out edge cases (never shown to the fixer) ----------------------------------
        heldout: list[TestFile] = []
        if self.settings.heldout_mode != "off":
            with self._step("heldout", result, "writing held-out edge-case tests (Nano)"):
                heldout = self._heldout(summary, repro, workspace, files, test_dir)

        # 5. baseline -----------------------------------------------------------------------
        with self._step("baseline", result, "verifier: freezing the test and recording the existing suite"):
            baseline = verifier.establish_baseline(repro, heldout)
            if not baseline.ok:
                raise StepFailed("baseline_failed", "baseline rejected: " + baseline.feedback())
            failing = sorted(baseline.suite.failing())
            self.emit("baseline", "done",
                      f"baseline recorded: {_counts(baseline.suite.counts())}; "
                      f"{len(failing)} existing failure(s) will be ignored; "
                      f"{len(baseline.heldout)} held-out test file(s)",
                      existing_failures=failing)

        # 6. fix ------------------------------------------------------------------------------
        with self._step("fix", result, "proposing fixes (Ultra) and submitting each to the verifier"):
            accepted = self._fix(issue_text, summary, workspace, base, files, repro, failure, verifier, result)

        # 7. deliver -----------------------------------------------------------------------------
        with self._step("deliver", result, "writing the diff and PR description"):
            from .report import write_failure_report, write_pr
            if accepted:
                result.status, result.patch = "fixed", accepted.patch
                path = write_pr(self, issue, summary, result, baseline, accepted, workspace, base)
                result.message = f"verified fix after {len(result.fix_attempts)} attempt(s); see {path.name}"
            else:
                result.status = "not_fixed"
                write_failure_report(self, issue, result)
                result.message = f"no fix accepted after {len(result.fix_attempts)} attempt(s)"
            self.emit("deliver", "done", result.message)

    # -- steps ----------------------------------------------------------------------------

    def _verifier_config(self) -> VerifierConfig:
        return dataclasses.replace(self.verifier_config, heldout_required=self.settings.heldout_mode == "gate")

    def _localize(self, workspace: Path, issue_text: str, summary: dict) -> list[str]:
        keywords = [k for k in summary.get("keywords", []) if isinstance(k, str)]
        candidates = rank_files(workspace, issue_text, keywords)
        if not candidates:
            raise StepFailed("setup_failed", "no Python source files found in the repository")
        self.emit("localize", "info", "candidates: " + ", ".join(f"{c.path} ({c.score:g})" for c in candidates[:8]))
        top = [c.path for c in candidates]
        if len(top) <= MAX_FILES - 1:
            return top
        listing = [(c.path, repo_mod.outline(repo_mod.read(workspace, c.path))) for c in candidates[:12]]
        picked = self._ask_json("fast", prompts.localize(summary, listing, MAX_FILES)) or {}
        chosen = [p for p in picked.get("files", []) if isinstance(p, str) and p in top]
        return chosen[:MAX_FILES] or top[:3]

    def _free_path(self, workspace: Path, test_dir: str, name: str) -> str:
        stem, n = name[:-3], 1
        path = f"{test_dir}/{name}"
        while (workspace / path).exists():
            n += 1
            path = f"{test_dir}/{stem}_{n}.py"
        return path

    def _reproduce(self, issue_text, summary, workspace, files, repro_path, verifier, result) -> tuple[TestFile, str]:
        example = repo_mod.example_test(workspace, [repo_mod.module_name(f) for f in files])
        context = prompts.files_context(workspace, files, self.settings.context_chars)
        messages = [{"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": prompts.reproduce(issue_text, summary, context, example, repro_path)}]
        last_feedback = ""
        for attempt in range(1, self.settings.max_repro_attempts + 1):
            result.repro_attempts = attempt
            text = self._ask("reasoning", messages)
            messages.append({"role": "assistant", "content": text})
            code = prompts.extract_code(text)
            if not code:
                last_feedback = "- [no_code] no ```python code block found in the reply"
            else:
                test = TestFile(repro_path, code)
                check = verifier.try_reproduction(test)
                if not check.ok:
                    last_feedback = check.feedback()
                else:
                    failure = _failure_text(check, repro_path)
                    self.emit("reproduce", "info", f"attempt {attempt}: test fails on the original code; asking Super to judge")
                    verdict = self._ask_json("mid", prompts.judge(summary, code, failure)) or {}
                    if verdict.get("reproduces") is not False:
                        return test, failure
                    last_feedback = ("- [judge] a reviewer found that this failure does not demonstrate the reported bug: "
                                     + str(verdict.get("explanation", "")))
            self.emit("reproduce", "info", f"attempt {attempt} rejected:\n{last_feedback}")
            messages.append({"role": "user", "content": prompts.reproduce_retry(last_feedback)})
        raise StepFailed("not_reproduced",
                         f"could not write a valid reproducing test in {self.settings.max_repro_attempts} attempts; "
                         f"last reason:\n{last_feedback}")

    def _heldout(self, summary, repro, workspace, files, test_dir) -> list[TestFile]:
        path = self._free_path(workspace, test_dir, "test_fixloop_heldout.py")
        messages = [{"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": prompts.heldout(
                        summary, repro.content, prompts.files_context(workspace, files, self.settings.context_chars), path)}]
        code = prompts.extract_code(self._ask("fast", messages))
        if not code:
            self.emit("heldout", "info", "no held-out tests produced; continuing without them")
            return []
        (self.run_dir / "heldout_test.py").write_text(code)
        mode = "must pass" if self.settings.heldout_mode == "gate" else "reported only"
        self.emit("heldout", "done", f"held-out tests written ({mode}); hidden from the fixer")
        return [TestFile(path, code)]

    def _fix(self, issue_text, summary, workspace, base, files, repro, failure, verifier, result) -> Attempt | None:
        known = set(repo_mod.source_files(workspace))
        messages = [
            {"role": "system", "content": prompts.FIX_SYSTEM},
            {"role": "user", "content": prompts.fix(issue_text, summary, repro.path, repro.content, failure,
                                                    prompts.files_context(workspace, files, self.settings.context_chars))},
        ]
        reads_left = MAX_READ_REQUESTS
        while len(result.fix_attempts) < self.settings.max_fix_attempts:
            text = self._ask("reasoning", messages)
            messages.append({"role": "assistant", "content": text})
            edits = edits_mod.parse_edits(text)
            reads = edits_mod.parse_read_requests(text)
            if not edits and reads and reads_left > 0:
                reads_left -= 1
                self.emit("fix", "info", f"model asked to read: {', '.join(reads[:3])}")
                messages.append({"role": "user", "content": prompts.read_files_reply(workspace, reads, known)})
                continue

            number = len(result.fix_attempts) + 1
            explanation = prompts.explanation_before_edits(text)
            repo_mod.reset(workspace, base)
            try:
                edits_mod.apply_edits(workspace, edits)
                patch = repo_mod.diff(workspace)
                edit_error = None
            except edits_mod.EditError as e:
                patch, edit_error = "", str(e)
            finally:
                repo_mod.reset(workspace, base)

            if edit_error:
                attempt = Attempt(number, False, [{"code": "edit_failed", "message": edit_error}],
                                  f"- [edit_failed] {edit_error}", "", explanation)
            else:
                self.emit("fix", "info", f"attempt {number}: submitting patch to the verifier", patch=patch)
                verdict: Verdict = verifier.verify(patch)
                attempt = Attempt(number, verdict.accepted, [r.to_dict() for r in verdict.reasons],
                                  verdict.feedback(), patch, explanation, str(verdict.evidence_dir))
            result.fix_attempts.append(attempt)
            if attempt.accepted:
                self.emit("fix", "done", f"attempt {number}: ACCEPTED by the verifier", attempt=number)
                return attempt
            self.emit("fix", "info", f"attempt {number}: rejected\n{attempt.feedback}",
                      attempt=number, reasons=attempt.reasons)
            messages.append({"role": "user", "content": prompts.fix_retry(number, attempt.feedback)})
        self.emit("fix", "fail", f"no accepted fix after {self.settings.max_fix_attempts} attempts")
        return None

    # -- reporting helpers ---------------------------------------------------------------------

    def usage_report(self) -> dict:
        by_role: dict[str, dict] = {}
        for c in self.llm.usage.calls[getattr(self, "_usage_start", 0):]:
            r = by_role.setdefault(c.role, {"model": c.model, "calls": 0, "prompt_tokens": 0,
                                            "completion_tokens": 0, "seconds": 0.0})
            r["calls"] += 1
            r["prompt_tokens"] += c.prompt_tokens
            r["completion_tokens"] += c.completion_tokens
            r["seconds"] = round(r["seconds"] + c.latency_s, 2)
        total_cost = 0.0
        priced = bool(self.settings.prices)
        for role, r in by_role.items():
            if role in self.settings.prices:
                pin, pout = self.settings.prices[role]
                r["cost_usd"] = round((r["prompt_tokens"] * pin + r["completion_tokens"] * pout) / 1e6, 6)
                total_cost += r["cost_usd"]
        return {"by_role": by_role, "cost_usd": round(total_cost, 6) if priced else None}


def _compact(messages: list[dict], keep_last: int = 4) -> list[dict]:
    """System prompt + task + only the latest exchanges, so retries don't grow the prompt
    without bound (matters on free tiers with tokens-per-minute limits)."""
    if len(messages) <= 2 + keep_last:
        return messages
    tail = messages[-keep_last:]
    if tail[0]["role"] == "user":  # keep strict user/assistant alternation
        tail = tail[1:]
    dropped = len(messages) - 2 - len(tail)
    task = dict(messages[1])
    task["content"] += f"\n\n[{dropped} earlier messages omitted; your latest attempt and its feedback follow.]"
    return [messages[0], task] + tail


def _failure_text(check: Baseline, repro_path: str) -> str:
    parts = []
    for nodeid, entry in check.targeted.in_file(repro_path).items():
        if entry["outcome"] == "failed":
            parts.append(f"{nodeid}\n{entry.get('excerpt') or entry.get('crash', '')}".strip())
    return "\n\n".join(parts)[-FAILURE_CHARS:]


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no tests"
