import pytest

from verifier.policy import PatchPolicy, check_patch, classify_path, parse_patch


def codes(patch, **policy):
    return [r.code for r in check_patch(patch, PatchPolicy(**policy))]


def diff(path, added=("x = 1",), removed=(), old=None):
    old = old or path
    body = "".join(f"-{l}\n" for l in removed) + "".join(f"+{l}\n" for l in added)
    return (
        f"diff --git a/{old} b/{path}\n--- a/{old}\n+++ b/{path}\n"
        f"@@ -1,{len(removed)} +1,{len(added)} @@\n{body}"
    )


@pytest.mark.parametrize("path,expected", [
    ("src/pkg/core.py", None),
    ("pkg/utils.py", None),
    ("docs/index.rst", None),
    ("tests/test_core.py", "touches_test_file"),
    ("pkg/tests/helpers.py", "touches_test_file"),
    ("test_thing.py", "touches_test_file"),
    ("pkg/thing_test.py", "touches_test_file"),
    ("conftest.py", "touches_test_config"),
    ("src/conftest.py", "touches_test_config"),
    ("pytest.ini", "touches_test_config"),
    ("tox.ini", "touches_test_config"),
    ("setup.cfg", "touches_test_config"),
    ("pyproject.toml", "touches_test_config"),
    (".github/workflows/ci.yml", "touches_ci"),
    (".gitlab-ci.yml", "touches_ci"),
    ("sitecustomize.py", "touches_interpreter_hook"),
    ("src/evil.pth", "touches_interpreter_hook"),
    ("../outside.py", "unsafe_path"),
    ("/etc/passwd", "unsafe_path"),
    (".git/hooks/pre-commit", "unsafe_path"),
])
def test_classify_path(path, expected):
    assert classify_path(path) == expected


@pytest.mark.parametrize("path,added,expected", [
    ("pytest.py", False, "shadows_test_runner"),
    ("pytest/__init__.py", True, "shadows_test_runner"),
    ("src/_pytest/main.py", True, "shadows_test_runner"),
    ("fixloop_report.py", True, "shadows_test_runner"),
    ("json.py", True, "shadows_stdlib"),
    ("src/collections/__init__.py", True, "shadows_stdlib"),
    ("json.py", False, None),           # an existing top-level module may be edited
    ("pkg/json.py", True, None),        # nested modules don't shadow anything
    ("pkg/pytest.py", True, None),
])
def test_shadowing_top_level_modules(path, added, expected):
    assert classify_path(path, added=added) == expected


def test_new_runner_shadow_in_patch_is_caught():
    patch = (
        "diff --git a/pytest.py b/pytest.py\nnew file mode 100644\n--- /dev/null\n+++ b/pytest.py\n"
        "@@ -0,0 +1 @@\n+import sys\n"
    )
    assert codes(patch) == ["shadows_test_runner"]


def test_clean_source_patch_passes():
    assert codes(diff("pkg/core.py", added=["return hi"], removed=["return lo"])) == []


def test_rename_into_tests_dir_is_caught():
    patch = (
        "diff --git a/pkg/a.py b/tests/a.py\nsimilarity index 100%\n"
        "rename from pkg/a.py\nrename to tests/a.py\n"
    )
    assert codes(patch) == ["touches_test_file"]


def test_deleting_test_file_is_caught():
    patch = (
        "diff --git a/tests/test_x.py b/tests/test_x.py\ndeleted file mode 100644\n"
        "--- a/tests/test_x.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-def test_x(): assert False\n"
    )
    assert codes(patch) == ["touches_test_file"]


@pytest.mark.parametrize("line", [
    "@pytest.mark.skip(reason='later')",
    "@pytest.mark.xfail",
    "pytest.skip('nope')",
    "    unittest.skip('x')",
    "raise SkipTest('x')",
])
def test_skip_markers_in_source_are_caught(line):
    assert "adds_skip_marker" in codes(diff("pkg/core.py", added=[line]))


def test_runtime_hooks_and_runner_references_are_caught():
    assert "suspicious_runtime_hook" in codes(diff("pkg/core.py", added=["os._exit(0)"]))
    assert "suspicious_runtime_hook" in codes(diff("pkg/core.py", added=["import atexit"]))
    assert "references_test_runner" in codes(diff("pkg/core.py", added=["if 'PYTEST_CURRENT_TEST' in env:"]))


def test_size_limits():
    assert codes(diff("pkg/core.py", added=["x"] * 11), max_changed_lines=10) == ["patch_too_large"]
    many = "".join(diff(f"pkg/m{i}.py") for i in range(3))
    assert codes(many, max_files=2) == ["patch_too_large"]


def test_binary_symlink_and_empty():
    assert "binary_patch" in codes(
        "diff --git a/pkg/x.bin b/pkg/x.bin\nBinary files a/pkg/x.bin and b/pkg/x.bin differ\n"
    )
    assert "adds_symlink" in codes(
        "diff --git a/pkg/l b/pkg/l\nnew file mode 120000\n--- /dev/null\n+++ b/pkg/l\n"
        "@@ -0,0 +1 @@\n+../tests/test_x.py\n"
    )
    assert codes("") == ["empty_patch"]
    assert codes("not a diff at all") == ["empty_patch"]


def test_parser_counts_hunk_lines_that_look_like_headers():
    # A removed line "-- x" renders as "--- x" and an added "++ y" as "+++ y".
    patch = (
        "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n"
        "@@ -1,3 +1,3 @@\n keep\n--- x\n+++ y\n keep\n"
    )
    [f] = parse_patch(patch)
    assert (f.old_path, f.new_path, f.added, f.removed) == ("pkg/a.py", "pkg/a.py", ["++ y"], 1)


def test_plain_unified_diff_without_git_header():
    patch = "--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-a\n+b\n--- a/tests/t.py\n+++ b/tests/t.py\n@@ -1 +1 @@\n-a\n+b\n"
    files = parse_patch(patch)
    assert [f.new_path for f in files] == ["pkg/a.py", "tests/t.py"]
    assert codes(patch) == ["touches_test_file"]
