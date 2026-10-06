"""Static patch policy: what a fix is allowed to change.

Everything here is pure and deterministic. `check_patch` inspects the unified
diff text; `check_paths` is re-run by the verifier on the file list git reports
after actually applying the patch, so a diff the parser misreads still cannot
sneak a protected file past the policy.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .reasons import Reason


@dataclass(frozen=True)
class PatchPolicy:
    max_changed_lines: int = 200
    max_files: int = 6


@dataclass
class FilePatch:
    old_path: str | None = None
    new_path: str | None = None
    added: list[str] = field(default_factory=list)
    removed: int = 0
    binary: bool = False
    symlink: bool = False

    @property
    def paths(self) -> set[str]:
        return {p for p in (self.old_path, self.new_path) if p}


# --- path classification ----------------------------------------------------

TEST_DIRS = {"test", "tests", "testing"}
TEST_CONFIG_FILES = {
    "conftest.py", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
    "noxfile.py", ".coveragerc",
}
INTERPRETER_HOOKS = {"sitecustomize.py", "usercustomize.py"}
CI_FILES = {
    ".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml", "appveyor.yml", ".appveyor.yml",
    "bitbucket-pipelines.yml", ".drone.yml", "Jenkinsfile", ".pre-commit-config.yaml",
    "codecov.yml", ".codecov.yml",
}
CI_DIRS = {".github", ".gitlab", ".circleci", ".azure-pipelines", ".buildkite"}
# Tests run as `python -m pytest` from the repo root, so a top-level module with
# one of these names would replace the test runner or the verifier's report plugin.
RUNNER_MODULES = {"pytest", "_pytest", "py", "pluggy", "iniconfig", "fixloop_report"}
IMPORT_ROOTS = {(), ("src",)}  # directories whose children are importable top-level names


def _top_level_modules(parts: tuple[str, ...]) -> set[str]:
    """Importable top-level names this path would define (at the repo root or under src/)."""
    names = set()
    for root in IMPORT_ROOTS:
        if parts[:len(root)] == root and len(parts) > len(root):
            child = parts[len(root)]
            if len(parts) > len(root) + 1:
                names.add(child)
            elif child.endswith(".py"):
                names.add(child[:-3])
    return names


def classify_path(path: str, *, added: bool = False) -> str | None:
    """Return a rejection code for a protected path, or None if it may change.

    `added` marks a file the patch creates (new top-level stdlib names are blocked).
    """
    p = PurePosixPath(path)
    parts = p.parts
    if not parts or p.is_absolute() or ".." in parts or ".git" in parts:
        return "unsafe_path"
    modules = _top_level_modules(parts)
    if modules & RUNNER_MODULES:
        return "shadows_test_runner"
    if added and modules & set(sys.stdlib_module_names):
        return "shadows_stdlib"
    name = p.name
    if name in TEST_CONFIG_FILES:
        return "touches_test_config"
    if name in INTERPRETER_HOOKS or name.endswith(".pth"):
        return "touches_interpreter_hook"
    if name in CI_FILES or parts[0] in CI_DIRS:
        return "touches_ci"
    if (TEST_DIRS & set(parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "tests.py"):
        return "touches_test_file"
    return None


_PATH_MESSAGES = {
    "unsafe_path": "patch touches an unsafe path",
    "touches_test_config": "patch modifies test configuration",
    "touches_interpreter_hook": "patch adds or modifies an interpreter startup hook",
    "touches_ci": "patch modifies CI configuration",
    "touches_test_file": "patch modifies a test file; only source files may change",
    "shadows_test_runner": "patch adds a top-level module that would shadow the test runner",
    "shadows_stdlib": "patch adds a top-level module that would shadow the standard library",
}


def check_paths(paths: list[str] | set[str], added: set[str] = frozenset()) -> list[Reason]:
    by_code: dict[str, list[str]] = {}
    for path in sorted(paths):
        code = classify_path(path, added=path in added)
        if code:
            by_code.setdefault(code, []).append(path)
    return [
        Reason(code, f"{_PATH_MESSAGES[code]}: {', '.join(found)}", items=found)
        for code, found in by_code.items()
    ]


# --- added-line checks --------------------------------------------------------

# Ways a source patch could skip, xfail or otherwise game the test run.
FORBIDDEN_ADDED = [
    ("adds_skip_marker", re.compile(
        r"mark\s*\.\s*(skip|skipif|xfail)\b|pytest\s*\.\s*(skip|xfail|importorskip|exit)\b"
        r"|unittest\s*\.\s*(skip|skipIf|skipUnless|expectedFailure)\b|\bSkipTest\b|\bskipTest\s*\("
    )),
    ("references_test_runner", re.compile(r"\bpytest\b|\b_pytest\b|PYTEST_|\bunittest\b")),
    ("suspicious_runtime_hook", re.compile(
        r"\bos\s*\.\s*_exit\b|\bsys\s*\.\s*(settrace|setprofile)\b|\batexit\b"
    )),
]
_ADDED_MESSAGES = {
    "adds_skip_marker": "patch adds a skip/xfail marker",
    "references_test_runner": "source patch references the test runner",
    "suspicious_runtime_hook": "patch adds a process-exit or tracing hook",
}


def check_added_lines(files: list[FilePatch]) -> list[Reason]:
    reasons = []
    for code, pattern in FORBIDDEN_ADDED:
        hits = sorted({
            f.new_path or f.old_path or "?"
            for f in files
            if any(pattern.search(line) for line in f.added)
        })
        if hits:
            reasons.append(Reason(code, f"{_ADDED_MESSAGES[code]} in {', '.join(hits)}", items=hits))
    return reasons


# --- unified diff parsing -------------------------------------------------------

_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def _diff_path(raw: str) -> str | None:
    raw = raw.split("\t", 1)[0].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    if raw == "/dev/null":
        return None
    if raw[:2] in ("a/", "b/"):
        raw = raw[2:]
    return raw


def parse_patch(text: str) -> list[FilePatch]:
    """Parse a unified (optionally git-extended) diff into per-file changes."""
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    in_hunk_body = False
    old_left = new_left = 0

    for line in text.splitlines():
        if old_left > 0 or new_left > 0:
            if line.startswith("+"):
                cur.added.append(line[1:])
                new_left -= 1
                continue
            if line.startswith("-"):
                cur.removed += 1
                old_left -= 1
                continue
            if line.startswith(" ") or line == "":
                old_left -= 1
                new_left -= 1
                continue
            if line.startswith("\\"):
                continue
            old_left = new_left = 0  # malformed hunk; fall through to header parsing

        if line.startswith("diff --git "):
            cur = FilePatch()
            files.append(cur)
            in_hunk_body = False
            m = re.match(r'diff --git ("?a/.+?"?) ("?b/.+"?)$', line)
            if m:
                cur.old_path, cur.new_path = _diff_path(m.group(1)), _diff_path(m.group(2))
        elif line.startswith("--- "):
            if cur is None or in_hunk_body:
                cur = FilePatch()
                files.append(cur)
                in_hunk_body = False
            cur.old_path = _diff_path(line[4:])
        elif line.startswith("+++ ") and cur is not None:
            cur.new_path = _diff_path(line[4:])
        elif line.startswith("@@") and cur is not None:
            m = _HUNK.match(line)
            if m:
                old_left = int(m.group(1) or 1)
                new_left = int(m.group(2) or 1)
                in_hunk_body = True
        elif cur is not None:
            if line.startswith(("rename from ", "copy from ")):
                cur.old_path = line.split(" ", 2)[2]
            elif line.startswith(("rename to ", "copy to ")):
                cur.new_path = line.split(" ", 2)[2]
            elif line.startswith("deleted file mode"):
                cur.new_path = None
            elif line.startswith("new file mode"):
                cur.old_path = None
                cur.symlink = cur.symlink or line.endswith("120000")
            elif line.startswith("new mode"):
                cur.symlink = cur.symlink or line.endswith("120000")
            elif line.startswith("GIT binary patch") or line.startswith("Binary files "):
                cur.binary = True
    return files


def check_patch(text: str, policy: PatchPolicy) -> list[Reason]:
    """All static policy violations for a patch; empty list means it may proceed."""
    files = parse_patch(text)
    if not any(f.added or f.removed or f.binary or f.old_path != f.new_path for f in files):
        return [Reason("empty_patch", "patch is empty or could not be parsed as a unified diff")]

    reasons: list[Reason] = []
    if any(f.binary for f in files):
        reasons.append(Reason("binary_patch", "patch contains binary changes"))
    if any(f.symlink for f in files):
        reasons.append(Reason("adds_symlink", "patch creates a symlink"))
    reasons += check_paths({p for f in files for p in f.paths},
                           added={f.new_path for f in files if f.old_path is None and f.new_path})
    reasons += check_added_lines(files)

    changed = sum(len(f.added) + f.removed for f in files)
    if changed > policy.max_changed_lines:
        reasons.append(Reason(
            "patch_too_large",
            f"patch changes {changed} lines (limit {policy.max_changed_lines}); keep the fix minimal",
        ))
    if len(files) > policy.max_files:
        reasons.append(Reason(
            "patch_too_large", f"patch touches {len(files)} files (limit {policy.max_files})"
        ))
    return reasons
