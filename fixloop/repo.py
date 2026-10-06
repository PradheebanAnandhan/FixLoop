"""Working with the target repository: cloning, file listing, test layout."""

from __future__ import annotations

import ast
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

from verifier.policy import classify_path

SKIP_DIRS = {".git", ".venv", "venv", "env", "build", "dist", "node_modules", "__pycache__",
             ".tox", ".nox", ".eggs", "site-packages"}
NON_SOURCE_DIRS = {"docs", "doc", "examples", "example", "benchmarks", "bench", "scripts", "tools"}


class RepoError(RuntimeError):
    pass


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RepoError(f"git {' '.join(args)} failed: {proc.stderr.strip()[-500:]}")
    return proc.stdout


def clone(url_or_path: str, dest: Path, commit: str | None = None) -> str:
    """Clone into `dest` and check out `commit` (default: the default branch). Returns the SHA."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["git", "clone", "-q", str(url_or_path), str(dest)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RepoError(f"git clone failed: {proc.stderr.strip()[-500:]}")
    if commit:
        git(dest, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", commit)
    return git(dest, "rev-parse", "HEAD").strip()


def export_tree(repo: Path, commit: str, dest: Path) -> None:
    """Write the files of `commit` (no .git) into `dest`, e.g. as an image build context."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True)
    if archive.returncode != 0:
        raise RepoError(f"git archive failed: {archive.stderr.decode()[-500:]}")
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)


def reset(repo: Path, commit: str) -> None:
    git(repo, "reset", "-q", "--hard", commit)
    git(repo, "clean", "-q", "-fdx")


def diff(repo: Path) -> str:
    git(repo, "add", "-A")
    out = git(repo, "diff", "--cached")
    git(repo, "reset", "-q")
    return out


def python_files(repo: Path) -> list[str]:
    files = []
    for path in repo.rglob("*.py"):
        rel = path.relative_to(repo)
        if SKIP_DIRS & set(rel.parts[:-1]):
            continue
        files.append(rel.as_posix())
    return sorted(files)


def source_files(repo: Path) -> list[str]:
    """Python files a fix may touch: not tests, test config, docs or examples."""
    return [
        f for f in python_files(repo)
        if classify_path(f) is None and not NON_SOURCE_DIRS & set(PurePosixPath(f).parts[:-1])
        and PurePosixPath(f).name not in ("setup.py", "noxfile.py")
    ]


def test_files(repo: Path) -> list[str]:
    return [f for f in python_files(repo) if classify_path(f) == "touches_test_file"
            and PurePosixPath(f).name.startswith("test_")]


def test_dir(repo: Path) -> str:
    """Directory where the repo keeps most of its test files ("tests" if there are none)."""
    counts = Counter(str(PurePosixPath(f).parent) for f in test_files(repo))
    if not counts:
        return "tests"
    return sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]


def example_test(repo: Path, modules: list[str], max_lines: int = 120) -> tuple[str, str] | None:
    """An existing test file to show the model the repo's test style, preferring one
    that imports one of `modules` (dotted names)."""
    tests = test_files(repo)
    if not tests:
        return None
    names = {m.split(".")[-1] for m in modules} | set(modules)

    def score(f: str) -> tuple[int, int]:
        text = read(repo, f)
        return (sum(1 for n in names if re.search(rf"\b{re.escape(n)}\b", text)), -len(text))

    best = max(tests, key=score)
    lines = read(repo, best).splitlines()
    return best, "\n".join(lines[:max_lines]) + ("\n# ..." if len(lines) > max_lines else "")


def module_name(path: str) -> str:
    """Dotted import name for a source path (handles src/ layouts)."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def read(repo: Path, path: str) -> str:
    try:
        return (repo / path).read_text(errors="replace")
    except OSError:
        return ""


def outline(source: str) -> list[str]:
    """Top-level and class-level definitions with line numbers."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    items = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(f"L{node.lineno} def {node.name}")
        elif isinstance(node, ast.ClassDef):
            items.append(f"L{node.lineno} class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    items.append(f"L{sub.lineno}   def {node.name}.{sub.name}")
    return items
