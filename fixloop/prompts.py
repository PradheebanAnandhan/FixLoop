"""Prompt templates for each agent step, and helpers to parse model responses."""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import repo as repo_mod

SYSTEM = (
    "You are FixLoop, an automated software engineer that fixes bugs in Python repositories. "
    "Be precise, minimal and follow the requested output format exactly."
)

FILE_BUDGET_CHARS = 60_000
PER_FILE_CHARS = 25_000


# --- formatting -------------------------------------------------------------------

def file_block(path: str, content: str) -> str:
    return f'<file path="{path}">\n{content.rstrip()}\n</file>'


def files_context(repo: Path, paths: list[str], budget: int = FILE_BUDGET_CHARS) -> str:
    blocks, used = [], 0
    per_file = min(PER_FILE_CHARS, budget)
    for path in paths:
        content = repo_mod.read(repo, path)
        if len(content) > per_file:
            content = content[:per_file] + f"\n# [... truncated: file is {len(content)} chars ...]"
        if used + len(content) > budget:
            blocks.append(f'<file path="{path}">[omitted: context budget reached; ask with read_file]</file>')
            continue
        blocks.append(file_block(path, content))
        used += len(content)
    return "\n\n".join(blocks)


def summary_block(summary: dict) -> str:
    return (
        f"<summary>\n{summary.get('summary', '').strip()}\n"
        f"Expected: {summary.get('expected', '').strip()}\n"
        f"Actual: {summary.get('actual', '').strip()}\n</summary>"
    )


# --- step prompts -------------------------------------------------------------------

def summarize(issue_text: str, repo_name: str) -> str:
    return f"""Summarize this GitHub issue from {repo_name} for an engineer who will reproduce and fix it.

<issue>
{issue_text}
</issue>

Respond with only a JSON object:
{{"summary": "<2-3 sentences: what is wrong>", "expected": "<correct behavior>", "actual": "<observed behavior>", "keywords": ["<function, class, method, module or parameter names involved>"], "is_bug": <true if this describes a bug in existing behavior, false for feature requests or questions>}}"""


def localize(summary: dict, candidates: list[tuple[str, list[str]]], k: int) -> str:
    listing = "\n\n".join(
        f"### {path}\n" + ("\n".join(defs[:40]) or "(no top-level definitions)") for path, defs in candidates
    )
    return f"""Pick the source files most likely to need changes to fix this bug.

{summary_block(summary)}

Candidate files and their definitions:

{listing}

Respond with only a JSON object: {{"files": ["<path>", ...]}} listing at most {k} paths from the candidates above, most likely first."""


def reproduce(issue_text: str, summary: dict, files: str, example: tuple[str, str] | None, test_path: str) -> str:
    example_text = ""
    if example:
        example_text = f"\nAn existing test in this repository, showing how tests import the code:\n\n{file_block(*example)}\n"
    return f"""Write a pytest test that reproduces this bug.

<issue>
{issue_text}
</issue>

{summary_block(summary)}

Relevant source files:

{files}
{example_text}
Requirements:
- The test file will be saved as `{test_path}` (a new file).
- It must FAIL on the current code because of the bug, and PASS once the bug is fixed.
- Assert the correct behavior described in the issue. Prefer one focused test function (at most three).
- Import the code under test the same way the existing tests do. Use only the standard library, pytest and packages the project already uses.
- Do not use skip, xfail or network access, and do not mock the code under test.

Respond with the complete test file in a single ```python code block and nothing else."""


def reproduce_retry(feedback: str) -> str:
    return f"""That test was rejected:
{feedback}

Write a corrected, complete test file in a single ```python code block."""


def judge(summary: dict, test_code: str, failure: str) -> str:
    return f"""Decide whether this failing test demonstrates the bug described in the issue.

{summary_block(summary)}

<test>
{test_code}
</test>

Output when run against the current (unfixed) code:
<failure>
{failure}
</failure>

The test fails. Does it fail *because of the reported bug* (not an unrelated error, wrong API usage or a mistake in the test), and do its assertions describe the correct behavior from the issue?

Respond with only a JSON object: {{"reproduces": true or false, "explanation": "<one or two sentences>"}}"""


def heldout(summary: dict, repro_code: str, files: str, test_path: str) -> str:
    return f"""Write 2 to 4 extra pytest tests for this bug that cover edge cases beyond the reproducing test below: other inputs, boundary values, or related code paths that a correct, general fix must also handle.

{summary_block(summary)}

Reproducing test (do not duplicate it):
<test>
{repro_code}
</test>

Relevant source files:

{files}

Requirements:
- The file will be saved as `{test_path}`. Import the code the same way the reproducing test does.
- Every test must PASS once the bug is fixed correctly in general (not just for the reproducing input).
- Only assert behavior that the issue or the existing code clearly implies. No skip/xfail, no network, no mocks of the code under test.

Respond with the complete test file in a single ```python code block and nothing else."""


FIX_SYSTEM = SYSTEM + """

You fix bugs by editing source files. Output format, strictly:

1. One or two sentences explaining the root cause.
2. One or more edit blocks:

<edit file="path/to/file.py">
<search>
exact lines copied from the current file
</search>
<replace>
the new lines
</replace>
</edit>

The SEARCH text must match the current file exactly, including indentation, and be unique in that file; include a few surrounding lines for context. To create a new file, use an empty <search></search>.
If you must see another file before editing, reply with only <read_file path="path/to/file.py"/> lines (at most 3 per reply)."""


def fix(issue_text: str, summary: dict, repro_path: str, repro_code: str, failure: str, files: str) -> str:
    return f"""Fix this bug.

<issue>
{issue_text}
</issue>

{summary_block(summary)}

This reproducing test fails on the current code. It must pass after your fix, and you cannot change it:

{file_block(repro_path, repro_code)}

Current failure:
<failure>
{failure}
</failure>

Relevant source files:

{files}

Rules:
- Change only source files. Changes to tests, conftest.py, pytest/tox config or CI files are rejected automatically.
- Make the smallest change that fixes the bug in general, not just for the reproducing input. Hidden edge-case tests may also be run.
- All existing tests must keep passing.
- Never add skip/xfail markers, reference pytest from source code, or special-case the test's inputs."""


def fix_retry(attempt: int, feedback: str) -> str:
    return f"""An independent verifier rejected attempt {attempt}:
{feedback}

The repository has been reset to the original code. Reply with a root-cause sentence and a complete new set of edit blocks against the ORIGINAL files."""


def read_files_reply(repo: Path, paths: list[str], known: set[str]) -> str:
    blocks = []
    for p in paths[:3]:
        if p in known or (repo / p).is_file():
            blocks.append(file_block(p, repo_mod.read(repo, p)[:PER_FILE_CHARS]))
        else:
            blocks.append(f'<file path="{p}">[no such file]</file>')
    return "\n\n".join(blocks) + "\n\nNow reply with the root cause and the edit blocks."


def pr_summary(issue_title: str, summary: dict, diff: str, explanation: str) -> str:
    return f"""Write the summary section of a pull request description for this bug fix.

Issue: {issue_title}
{summary_block(summary)}

Root cause, as noted by the fixer: {explanation}

<diff>
{diff}
</diff>

Respond with two short Markdown paragraphs: what was wrong (the root cause), and what the change does. No headings, no test results, no mention of AI or automation; those parts are added separately."""


# --- parsing --------------------------------------------------------------------------

CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    stripped = text.strip()
    if re.match(r"(import |from |def |class |@)", stripped):
        return stripped + "\n"
    return None


def extract_json(text: str) -> dict | None:
    candidates = [text.strip()]
    candidates += re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            value = json.loads(c)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def explanation_before_edits(text: str) -> str:
    head = text.split("<edit", 1)[0].strip()
    return re.sub(r"\s+", " ", head)[:600]
