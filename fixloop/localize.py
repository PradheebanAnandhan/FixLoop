"""Deterministic first pass at finding the files an issue is about.

The agent then asks Nano to choose from these candidates; when it can't (or
the repo is tiny), the top-scored files are used directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import repo as repo_mod

TRACEBACK_FILE = re.compile(r'File "([^"]+\.py)", line \d+')
INLINE_CODE = re.compile(r"`([^`\n]{1,200})`")
CODE_BLOCK = re.compile(r"```[\w+-]*\n(.*?)```", re.S)
IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "import", "return", "def", "class", "true",
    "false", "none", "self", "print", "assert", "raise", "error", "python", "version", "when", "should",
    "expected", "actual", "issue", "bug", "test", "tests", "value", "values", "result", "results",
    "file", "line", "lambda", "not", "is", "in", "if", "else", "elif", "try", "except", "pass", "str",
    "int", "float", "list", "dict", "set", "tuple", "bool", "len", "range", "type", "object",
}


@dataclass
class Candidate:
    path: str
    score: float
    hits: list[str]


def keywords(issue_text: str, extra: list[str] = ()) -> tuple[set[str], set[str]]:
    """(identifier-like keywords, file paths mentioned in tracebacks)."""
    paths = {m.replace("\\", "/") for m in TRACEBACK_FILE.findall(issue_text)}
    code = " ".join(INLINE_CODE.findall(issue_text) + CODE_BLOCK.findall(issue_text))
    words = set()
    for token in IDENTIFIER.findall(code) + list(extra):
        for part in [token, *token.split(".")]:
            if len(part) >= 3 and part.lower() not in STOPWORDS:
                words.add(part)
    # Identifiers in prose are only kept when they look like code (snake_case, CamelCase, dotted).
    for token in IDENTIFIER.findall(issue_text):
        if ("_" in token or "." in token or re.search(r"[a-z][A-Z]", token)) and len(token) >= 4:
            for part in [token, *token.split(".")]:
                if len(part) >= 3 and part.lower() not in STOPWORDS:
                    words.add(part)
    return words, paths


def rank_files(repo: Path, issue_text: str, extra_keywords: list[str] = (), limit: int = 15) -> list[Candidate]:
    words, tb_paths = keywords(issue_text, extra_keywords)
    candidates = []
    for path in repo_mod.source_files(repo):
        text = repo_mod.read(repo, path)
        score, hits = 0.0, []
        if any(tb.endswith(path) or tb.endswith(path.removeprefix("src/")) for tb in tb_paths):
            score += 10
            hits.append("traceback")
        mod = repo_mod.module_name(path)
        stem = PurePosixPath(path).stem
        for w in words:
            if w == mod or w == stem or mod.endswith("." + w):
                score += 4
                hits.append(f"module:{w}")
            if re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(w)}\b", text, re.M):
                score += 3
                hits.append(f"defines:{w}")
            elif re.search(rf"\b{re.escape(w)}\b", text):
                score += 1
        if score > 0:
            candidates.append(Candidate(path, score, hits))
    candidates.sort(key=lambda c: (-c.score, c.path))
    if not candidates:  # nothing matched: fall back to the largest source files
        files = sorted(repo_mod.source_files(repo), key=lambda f: -len(repo_mod.read(repo, f)))
        candidates = [Candidate(f, 0.0, []) for f in files]
    return candidates[:limit]


def traceback_files(text: str, known: list[str]) -> list[str]:
    """Repo source files that appear in a pytest failure excerpt."""
    found = []
    for path in known:
        if re.search(rf"(^|[\s/\"']){re.escape(path)}[:\"]", text, re.M) and path not in found:
            found.append(path)
    return found
