"""Command-line entry point for the verifier.

    # build the sandbox image for a repo at its base commit (network allowed here only)
    python -m verifier build-image --repo PATH --commit REV --tag fixloop/myrepo:base

    # check a reproducing test on clean code, and optionally a patch
    python -m verifier check --repo PATH --commit REV --image fixloop/myrepo:base \\
        --repro tests/test_fixloop_repro.py=./repro_test.py [--patch fix.diff] \\
        [--heldout tests/test_fixloop_heldout.py=./heldout.py] --evidence runs/demo

Exit codes for `check`: 0 accepted (or baseline ok when no --patch), 1 patch
rejected, 2 baseline rejected, 3 infrastructure error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .core import RepoSource, TestFile, Verifier, VerifierConfig, VerifierError
from .sandbox import DockerSandbox, SandboxConfig, SandboxError, build_image


def _test_file(spec: str) -> TestFile:
    repo_path, sep, local = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("expected REPO_PATH=LOCAL_FILE")
    return TestFile(repo_path, Path(local).read_text())


def cmd_build_image(args) -> int:
    with tempfile.TemporaryDirectory(prefix="fixloop-build-src-") as tmp:
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(args.repo), tmp], check=True)
        subprocess.run(["git", "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", args.commit],
                       cwd=tmp, check=True)
        shutil.rmtree(Path(tmp) / ".git")
        result = build_image(Path(tmp), args.tag, base_image=args.base_image, timeout=args.timeout)
    print(f"built {args.tag} in {result.duration_s:.0f}s")
    return 0


def cmd_check(args) -> int:
    sandbox = DockerSandbox(SandboxConfig(image=args.image, read_only_repo=not args.writable_repo))
    with Verifier(RepoSource(args.repo, args.commit), sandbox, Path(args.evidence),
                  VerifierConfig.from_env()) as verifier:
        baseline = verifier.establish_baseline(args.repro, heldout=args.heldout or [])
        print(json.dumps({"baseline": baseline.to_dict()}, indent=2))
        if not baseline.ok:
            print("\nBASELINE REJECTED\n" + baseline.feedback(), file=sys.stderr)
            return 2
        if not args.patch:
            return 0
        verdict = verifier.verify(Path(args.patch).read_text())
        print(json.dumps({"verdict": verdict.to_dict()}, indent=2))
        print(("\nACCEPTED" if verdict.accepted else "\nREJECTED\n" + verdict.feedback())
              + f"\nevidence: {verdict.evidence_dir}", file=sys.stderr)
        return 0 if verdict.accepted else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m verifier", description="FixLoop independent verifier")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-image", help="build the sandbox image for a repo at a commit")
    b.add_argument("--repo", required=True, help="git repo path or URL")
    b.add_argument("--commit", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--base-image", default="python:3.12-slim")
    b.add_argument("--timeout", type=float, default=1800)
    b.set_defaults(func=cmd_build_image)

    c = sub.add_parser("check", help="run the baseline and optionally verify a patch")
    c.add_argument("--repo", required=True, help="git repo path or URL")
    c.add_argument("--commit", required=True)
    c.add_argument("--image", required=True)
    c.add_argument("--repro", required=True, type=_test_file, help="REPO_PATH=LOCAL_FILE")
    c.add_argument("--heldout", action="append", type=_test_file, help="REPO_PATH=LOCAL_FILE (repeatable)")
    c.add_argument("--patch", help="unified diff to verify")
    c.add_argument("--evidence", required=True, help="directory for logs and verdicts")
    c.add_argument("--writable-repo", action="store_true", help="mount the checkout read-write")
    c.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SandboxError, VerifierError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
