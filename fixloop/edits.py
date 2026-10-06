"""Search/replace edit blocks: the format the fixer model writes patches in.

Models are unreliable at producing unified diffs with correct line numbers, so
the fixer emits blocks like

    <edit file="pkg/core.py">
    <search>
        if x > hi:
            return lo
    </search>
    <replace>
        if x > hi:
            return hi
    </replace>
    </edit>

which are applied to a clean checkout; `git diff` then produces the patch that
goes to the verifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

EDIT_BLOCK = re.compile(
    r'<edit\s+file\s*=\s*"([^"]+)"\s*>\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>\s*</edit>',
    re.S,
)
READ_REQUEST = re.compile(r'<read_file\s+path\s*=\s*"([^"]+)"\s*/?>')


class EditError(ValueError):
    pass


@dataclass
class Edit:
    path: str
    search: str
    replace: str


def _block_text(raw: str) -> str:
    # Drop the newline right after the opening tag; keep the text otherwise verbatim.
    return raw[1:] if raw.startswith("\n") else raw


def parse_edits(text: str) -> list[Edit]:
    return [Edit(p.strip(), _block_text(s), _block_text(r)) for p, s, r in EDIT_BLOCK.findall(text)]


def parse_read_requests(text: str) -> list[str]:
    return [p.strip() for p in READ_REQUEST.findall(text)]


def _safe_path(repo: Path, path: str) -> Path:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or ".git" in p.parts:
        raise EditError(f"{path}: path must be relative and inside the repository")
    return repo / p


def _apply_one(content: str, edit: Edit) -> str:
    count = content.count(edit.search)
    if count == 1:
        return content.replace(edit.search, edit.replace, 1)
    if count > 1:
        raise EditError(f"{edit.path}: SEARCH text matches {count} places; include more surrounding lines")

    # Fallback: match line by line ignoring trailing whitespace.
    lines = content.splitlines(keepends=True)
    want = [l.rstrip() for l in edit.search.splitlines()]
    while want and not want[-1]:
        want.pop()
    if not want:
        raise EditError(f"{edit.path}: SEARCH block is empty")
    have = [l.rstrip() for l in lines]
    starts = [i for i in range(len(have) - len(want) + 1) if have[i:i + len(want)] == want]
    if len(starts) != 1:
        problem = "not found" if not starts else f"matches {len(starts)} places"
        raise EditError(f"{edit.path}: SEARCH text {problem}; copy the original lines exactly")
    i = starts[0]
    replacement = edit.replace if edit.replace.endswith("\n") or not edit.replace else edit.replace + "\n"
    return "".join(lines[:i]) + replacement + "".join(lines[i + len(want):])


def apply_edits(repo: Path, edits: list[Edit]) -> list[str]:
    """Apply edits in order; returns changed paths. Raises EditError on the first failure."""
    if not edits:
        raise EditError("no <edit> blocks found; use the exact format from the instructions")
    changed = []
    for edit in edits:
        target = _safe_path(repo, edit.path)
        if not target.exists():
            if edit.search.strip():
                raise EditError(f"{edit.path}: file does not exist")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edit.replace)
        else:
            target.write_text(_apply_one(target.read_text(), edit))
        if edit.path not in changed:
            changed.append(edit.path)
    return changed
