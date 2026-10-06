"""Independent, deterministic verifier for FixLoop patches (no LLM calls).

Deliberately imports nothing from the `fixloop` agent package and uses only the
standard library, so the agent cannot influence how a verdict is reached.
"""

from .core import (
    Baseline, RepoSource, TestFile, Verdict, Verifier, VerifierConfig, VerifierError,
)
from .policy import PatchPolicy, check_patch
from .reasons import Reason
from .sandbox import DockerSandbox, SandboxConfig, SandboxError, build_image

__all__ = [
    "Baseline", "DockerSandbox", "PatchPolicy", "Reason", "RepoSource", "SandboxConfig",
    "SandboxError", "TestFile", "Verdict", "Verifier", "VerifierConfig", "VerifierError",
    "build_image", "check_patch",
]
