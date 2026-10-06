"""Fetch a GitHub issue (title, body, first comments) over the REST API."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

ISSUE_URL = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+)/issues/(\d+)/?(?:[#?].*)?$")
MAX_BODY_CHARS = 8000
MAX_COMMENTS = 5
MAX_COMMENT_CHARS = 1500


class IssueError(RuntimeError):
    pass


@dataclass
class Issue:
    owner: str
    repo: str
    number: int
    title: str
    body: str
    comments: list[str] = field(default_factory=list)
    url: str = ""

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"

    @property
    def slug(self) -> str:
        return f"{self.owner}-{self.repo}-{self.number}".lower()

    def as_text(self) -> str:
        text = f"# {self.title}\n\n{self.body.strip()}"
        for i, c in enumerate(self.comments, 1):
            text += f"\n\n## Comment {i}\n{c.strip()}"
        return text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> "Issue":
        """Load an issue saved as JSON (fields as in to_dict), for offline runs and evals."""
        data = json.loads(Path(path).read_text())
        return cls(**{k: data[k] for k in ("owner", "repo", "number", "title", "body")},
                   comments=data.get("comments", []), url=data.get("url", ""))


def parse_issue_url(url: str) -> tuple[str, str, int]:
    m = ISSUE_URL.match(url.strip())
    if not m:
        raise IssueError(f"not a GitHub issue URL: {url!r} (expected https://github.com/<owner>/<repo>/issues/<n>)")
    owner, repo, number = m.groups()
    return owner, repo.removesuffix(".git"), int(number)


def _get(url: str, token: str | None) -> object:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "fixloop",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        hint = " (rate limited? set GITHUB_TOKEN)" if e.code in (403, 429) else ""
        raise IssueError(f"GitHub API returned {e.code} for {url}{hint}") from None
    except urllib.error.URLError as e:
        raise IssueError(f"could not reach GitHub API: {e.reason}") from None


def _clip(text: str | None, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n[... truncated ...]"


def fetch_issue(url: str, token: str | None = None, *, get=_get) -> Issue:
    owner, repo, number = parse_issue_url(url)
    api = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    data = get(api, token)
    if "pull_request" in data:
        raise IssueError(f"{url} is a pull request, not an issue")
    comments = []
    if data.get("comments"):
        raw = get(f"{api}/comments?per_page={MAX_COMMENTS}", token)
        comments = [_clip(c.get("body"), MAX_COMMENT_CHARS) for c in raw[:MAX_COMMENTS]]
    return Issue(owner, repo, number, data.get("title", ""), _clip(data.get("body"), MAX_BODY_CHARS),
                 comments, url=url)
