"""The verifier must stay independent of the agent: stdlib only, no LLM client."""

import ast
import sys
from pathlib import Path

VERIFIER = Path(__file__).resolve().parent.parent / "verifier"


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_verifier_imports_only_stdlib():
    files = sorted(VERIFIER.rglob("*.py"))
    assert files
    for f in files:
        non_stdlib = imported_modules(f) - set(sys.stdlib_module_names)
        assert not non_stdlib, f"{f.relative_to(VERIFIER.parent)} imports {non_stdlib}"
