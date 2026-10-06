"""pytest plugin the verifier loads with `-p fixloop_report` inside the sandbox.

Records every test's final outcome by node ID, plus collection errors, and
writes them as JSON at session end. The output path comes from an environment
variable that is removed as soon as the plugin loads, before any project code
is imported.
"""

import json
import os

_REPORT_PATH = os.environ.pop("FIXLOOP_REPORT_PATH", None)
_tests = {}
_collect_errors = {}
_FAILING = ("failed", "error")


def _excerpt(report, limit=3000):
    text = getattr(report, "longreprtext", "") or ""
    return text[-limit:]


def _crash(report):
    crash = getattr(getattr(report, "longrepr", None), "reprcrash", None)
    message = getattr(crash, "message", None) or ""
    return message.splitlines()[0][:300] if message else ""


def pytest_collectreport(report):
    if report.failed:
        _collect_errors[report.nodeid] = _excerpt(report)


def pytest_runtest_logreport(report):
    xfail = hasattr(report, "wasxfail")
    if report.when == "call":
        if report.passed:
            outcome = "xpassed" if xfail else "passed"
        elif report.skipped:
            outcome = "xfailed" if xfail else "skipped"
        else:
            outcome = "failed"
    elif report.failed:  # error in setup or teardown
        outcome = "error"
    elif report.skipped:  # skipped during setup (skip/skipif markers)
        outcome = "xfailed" if xfail else "skipped"
    else:
        return

    previous = _tests.get(report.nodeid)
    if previous and previous["outcome"] not in ("passed", "xpassed"):
        return  # keep the first non-passing outcome (e.g. call failure over teardown error)
    entry = {"outcome": outcome}
    if outcome in _FAILING:
        entry["crash"] = _crash(report)
        entry["excerpt"] = _excerpt(report)
    _tests[report.nodeid] = entry


def pytest_sessionfinish(session, exitstatus):
    if not _REPORT_PATH:
        return
    data = {"exitstatus": int(exitstatus), "tests": _tests, "collect_errors": _collect_errors}
    tmp = _REPORT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, _REPORT_PATH)
