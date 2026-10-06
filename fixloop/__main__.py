"""Terminal entry point.

    python -m fixloop https://github.com/<owner>/<repo>/issues/<n>
    python -m fixloop --issue-file issue.json --repo path/or/url --commit <sha>

Exit code: 0 fixed, 1 not fixed / not reproduced, 2 config or setup error.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

from .agent import Agent, Event
from .config import ConfigError, load_settings
from .github import Issue, IssueError, fetch_issue

COLORS = {"start": "\033[1;36m", "done": "\033[1;32m", "fail": "\033[1;31m", "info": "\033[0m"}
RESET = "\033[0m"


class TerminalPrinter:
    def __init__(self, color: bool) -> None:
        self.color = color
        self.t0 = time.time()

    def __call__(self, event: Event) -> None:
        stamp = f"{event.t - self.t0:6.1f}s"
        tag = f"[{event.step}]"
        lines = event.message.splitlines() or [""]
        if self.color:
            print(f"{stamp} {COLORS.get(event.kind, '')}{tag:<12}{RESET} {lines[0]}")
        else:
            print(f"{stamp} {tag:<12} {lines[0]}")
        for line in lines[1:]:
            print(" " * 20 + line)
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fixloop", description="Issue-to-fix agent with an independent verifier")
    parser.add_argument("issue_url", nargs="?", help="GitHub issue URL")
    parser.add_argument("--issue-file", help="issue JSON instead of fetching from GitHub (offline runs, evals)")
    parser.add_argument("--repo", help="clone from this path or URL instead of github.com")
    parser.add_argument("--commit", help="base commit to fix against (default: default branch HEAD)")
    parser.add_argument("--run-dir", help="output directory (default: runs/<issue>-<timestamp>)")
    parser.add_argument("--max-attempts", type=int, help="override MAX_FIX_ATTEMPTS")
    parser.add_argument("--heldout", choices=["gate", "report", "off"], help="override HELDOUT_MODE")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    if not args.issue_url and not args.issue_file:
        parser.error("give an issue URL or --issue-file")

    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    overrides = {}
    if args.max_attempts:
        overrides["max_fix_attempts"] = args.max_attempts
    if args.heldout:
        overrides["heldout_mode"] = args.heldout
    settings = dataclasses.replace(settings, **overrides)

    try:
        issue = Issue.from_file(args.issue_file) if args.issue_file else fetch_issue(args.issue_url, settings.github_token)
    except (IssueError, OSError, ValueError, KeyError) as e:
        print(f"Could not load the issue: {e}", file=sys.stderr)
        return 2

    agent = Agent(settings, on_event=TerminalPrinter(color=sys.stdout.isatty() and not args.no_color))
    result = agent.run(issue, repo_source=args.repo, commit=args.commit, run_dir=args.run_dir)

    usage = result.usage
    print()
    print(f"status:   {result.status}")
    print(f"run dir:  {result.run_dir}")
    for role, u in usage.get("by_role", {}).items():
        cost = f", ${u['cost_usd']:.4f}" if "cost_usd" in u else ""
        print(f"  {role:<10} {u['model']}: {u['calls']} calls, "
              f"{u['prompt_tokens']}+{u['completion_tokens']} tokens, {u['seconds']:.0f}s{cost}")
    if usage.get("cost_usd") is not None:
        print(f"total cost: ${usage['cost_usd']:.4f}")
    print(f"total time: {result.total_seconds:.0f}s")
    if result.status == "fixed":
        return 0
    return 2 if result.status in ("setup_failed", "error") else 1


if __name__ == "__main__":
    sys.exit(main())
