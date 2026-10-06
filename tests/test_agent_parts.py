import pytest

from fixloop import prompts
from fixloop import repo as repo_mod
from fixloop.edits import EditError, apply_edits, parse_edits, parse_read_requests
from fixloop.github import IssueError, fetch_issue, parse_issue_url
from fixloop.localize import keywords, rank_files, traceback_files
from toyrepo import make_repo

# --- edits -----------------------------------------------------------------------

SOURCE = "def f(x):\n    if x > 1:\n        return 1\n    return x\n\n\ndef g():\n    return 1\n"


def write(tmp_path, content=SOURCE, name="m.py"):
    (tmp_path / name).write_text(content)
    return tmp_path


def block(search, replace, path="m.py"):
    return f'<edit file="{path}">\n<search>\n{search}</search>\n<replace>\n{replace}</replace>\n</edit>'


def test_exact_edit(tmp_path):
    repo = write(tmp_path)
    apply_edits(repo, parse_edits(block("    if x > 1:\n        return 1\n", "    if x > 2:\n        return 2\n")))
    assert "if x > 2:\n        return 2\n    return x" in (repo / "m.py").read_text()


def test_trailing_whitespace_fallback(tmp_path):
    repo = write(tmp_path, SOURCE.replace("if x > 1:", "if x > 1:   "))
    apply_edits(repo, parse_edits(block("    if x > 1:\n        return 1\n", "    if x > 1:\n        return 0\n")))
    assert "return 0" in (repo / "m.py").read_text()


@pytest.mark.parametrize("search,message", [
    ("    return 1\n", "matches 2 places"),
    ("    return 42\n", "not found"),
])
def test_ambiguous_or_missing_search(tmp_path, search, message):
    repo = write(tmp_path)
    with pytest.raises(EditError, match=message):
        apply_edits(repo, parse_edits(block(search, "x\n")))


def test_new_file_unsafe_path_and_no_edits(tmp_path):
    repo = write(tmp_path)
    apply_edits(repo, parse_edits('<edit file="pkg/new.py">\n<search>\n</search>\n<replace>\nX = 1\n</replace>\n</edit>'))
    assert (repo / "pkg/new.py").read_text() == "X = 1\n"
    with pytest.raises(EditError, match="inside the repository"):
        apply_edits(repo, parse_edits(block("a\n", "b\n", path="../evil.py")))
    with pytest.raises(EditError, match="no <edit> blocks"):
        apply_edits(repo, parse_edits("I think the fix is to change line 3."))


def test_multiple_edits_and_read_requests():
    text = "cause\n" + block("a\n", "b\n") + "\n" + block("c\n", "d\n", path="n.py")
    assert [(e.path, e.search, e.replace) for e in parse_edits(text)] == [("m.py", "a\n", "b\n"), ("n.py", "c\n", "d\n")]
    assert parse_read_requests('<read_file path="a/b.py"/>\n<read_file path="c.py">') == ["a/b.py", "c.py"]


# --- github ------------------------------------------------------------------------

def test_parse_issue_url():
    assert parse_issue_url("https://github.com/pallets/click/issues/123") == ("pallets", "click", 123)
    assert parse_issue_url("https://github.com/a/b.py/issues/9#issuecomment-1") == ("a", "b.py", 9)
    with pytest.raises(IssueError):
        parse_issue_url("https://github.com/a/b/pull/9")


def test_fetch_issue_with_comments_and_pr_rejection():
    def get(url, token):
        if url.endswith("/comments?per_page=5"):
            return [{"body": "same here"}, {"body": None}]
        return {"title": "Bug", "body": "x" * 10000, "comments": 2}

    issue = fetch_issue("https://github.com/o/r/issues/1", get=get)
    assert issue.title == "Bug" and issue.comments == ["same here", ""]
    assert issue.body.endswith("[... truncated ...]")
    assert issue.clone_url == "https://github.com/o/r.git"
    with pytest.raises(IssueError, match="pull request"):
        fetch_issue("https://github.com/o/r/issues/2", get=lambda u, t: {"pull_request": {}, "title": "x"})


# --- localization and repo helpers ----------------------------------------------------------

def test_keywords_from_code_and_tracebacks():
    text = (
        "Calling `parse_date('2020')` breaks.\n```\nTraceback:\n  File \"/x/site-packages/pkg/dates.py\", line 3\n"
        "ValueError\n```\nAlso DateParser.parse is wrong; the result is fine otherwise."
    )
    words, paths = keywords(text)
    assert {"parse_date", "DateParser", "parse"} <= words
    assert "result" not in words and "the" not in words
    assert paths == {"/x/site-packages/pkg/dates.py"}


def test_rank_files_and_repo_helpers(tmp_path):
    repo, _ = make_repo(tmp_path)
    (repo / "other.py").write_text("def unrelated():\n    pass\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "conf.py").write_text("clamp = 1\n")
    ranked = rank_files(repo, "`clamp(5, 0, 3)` is wrong")
    assert ranked[0].path == "calc.py" and "defines:clamp" in ranked[0].hits
    assert "docs/conf.py" not in [c.path for c in ranked]
    assert repo_mod.test_dir(repo) == "tests"
    path, text = repo_mod.example_test(repo, ["calc"])
    assert path == "tests/test_calc.py" and "from calc import" in text
    assert repo_mod.module_name("src/pkg/__init__.py") == "pkg"
    assert repo_mod.module_name("src/pkg/a/b.py") == "pkg.a.b"
    assert "L5 def clamp" in repo_mod.outline((repo / "calc.py").read_text())


def test_traceback_files():
    excerpt = 'calc.py:9: in clamp\n    return lo\nE assert 0 == 3\n  File "/work/pkg/util.py", line 3'
    assert traceback_files(excerpt, ["calc.py", "pkg/util.py", "other.py"]) == ["calc.py", "pkg/util.py"]


# --- response parsing ------------------------------------------------------------------------

def test_extract_code_and_json():
    assert prompts.extract_code("blah\n```python\nimport x\n```\n") == "import x\n"
    assert prompts.extract_code("from a import b\n\ndef test(): pass") == "from a import b\n\ndef test(): pass\n"
    assert prompts.extract_code("no code here") is None
    assert prompts.extract_json('{"a": 1}') == {"a": 1}
    assert prompts.extract_json('Sure!\n```json\n{"files": ["x.py"]}\n```') == {"files": ["x.py"]}
    assert prompts.extract_json('The answer is {"reproduces": false} ok') == {"reproduces": False}
    assert prompts.extract_json("nothing") is None
    assert prompts.explanation_before_edits("Root cause: off by one.\n<edit file=...") == "Root cause: off by one."
