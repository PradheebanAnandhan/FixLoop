"""Structured accept/reject reasons shared by the policy and the verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Reason:
    code: str      # stable machine-readable code, e.g. "touches_test_file"
    message: str   # one short line the agent sees on retry
    items: list[str] = field(default_factory=list)  # paths or test IDs involved
    detail: str = ""  # optional failure excerpt (truncated) to help the next attempt

    def to_dict(self) -> dict:
        return asdict(self)
