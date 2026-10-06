# FixLoop

**An issue-to-fix coding agent whose patches are accepted only by an independent verifier, never on the model's say-so.**

Give FixLoop a GitHub issue URL for a Python project. It clones the repo, writes a failing test that reproduces the bug, and proposes a fix. A separate, deterministic verifier then runs the patch in a fresh sandbox and makes the final accept/reject call. The agent iterates on the verifier's rejection reasons, and a passing run ends with a diff and a PR description that includes the verifier's evidence.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) — **Coding and Agentic Engineering Track**.

> **Project name is a placeholder.** Rename it anywhere you like.

[Demo video](#) · [Live demo](#) · [Devpost submission](#)

<!-- Replace the three links above before submitting. -->

---

## Why this exists

Chat assistants can *guess* at a bug fix, but they can't verify it. FixLoop closes that gap: every claim the agent makes is backed by code it actually executed. A fix only counts if a reproducing test goes from red to green.

## How it works

```
GitHub issue URL
      │
      ▼
1. Intake        clone repo, install dependencies in a sandbox
2. Localize      search the codebase for the files likely involved
3. Reproduce     write a failing test, confirm it fails for the right reason
4. Fix           propose a patch (source files only) → submit to the verifier
5. Verify        independent verifier accepts or rejects → on reject, retry with its reasons (max N)
6. Deliver       diff + PR description with the verifier's before/after evidence
```

### The independent verifier

The model never gets to declare its own fix correct. The final verdict comes from deterministic code (no LLM involved) that runs in its own fresh container, against a clean checkout of the repo:

1. **Baseline run.** The original code is tested first. The reproducing test must **fail**, and the failures of the existing suite are recorded.
2. **Patch policy.** The patch is rejected outright if it touches test files, `conftest.py`, pytest or tox config, or CI files, adds skip/xfail markers, or exceeds a size limit.
3. **Patched run.** The verifier applies the patch itself. The reproducing test is stored outside the agent's reach and injected by the verifier. It must now **pass**.
4. **Regression check.** The existing suite is rerun and compared with the baseline. Any newly failing test rejects the patch.
5. **Isolation.** Every run uses no network, a read-only mount where possible, a cleared environment, and a hard timeout.

A rejection returns a short reason (for example "modified a test file" or "regression in `tests/test_x.py`") that the agent uses on its next attempt. The verdict, reasons, and test output are saved as evidence and included in the PR description.

**Stretch goal:** a held-out check, where a separate Nano call writes a few extra edge-case tests the fixer never saw, and the patch must pass those as well. This catches fixes that only special-case the reproducing test.

### Where each Nemotron model is used

| Step | Model | Why |
|---|---|---|
| Summarize the issue, localize files, write held-out edge-case tests | `NVIDIA-Nemotron-3-Nano-30B-A3B` | Fast, cheap calls that run many times per issue |
| Judge whether the failing test reproduces the reported bug, write the PR summary | `nemotron-3-super-120b-a12b` | Balance of speed and capability |
| Reproduce and Fix | `Nemotron-3-Ultra-550b-a55b` | Serious reasoning for the hard steps |

The verifier makes no model calls at all.

All models are served through **Nebius Token Factory** via its OpenAI-compatible API. Model IDs are configurable in `.env` (see below). Confirm exact IDs against your account's `/v1/models` listing.

### Nebius services used

- **Nebius Token Factory** — all LLM inference (NVIDIA Nemotron models)
- <!-- Add: Nebius AI Cloud / Serverless Endpoints / Serverless Jobs, only if you actually use them -->

### Where Token Factory helped

<!-- Fill in honestly after building. Examples: latency per step, cost per fix, ease of switching between Nano/Super/Ultra with a one-line model change. -->

---

## Results

Evaluated on a fixed set of real issues from public Python repositories.

| Metric | Result |
|---|---|
| Issues attempted | TBD |
| Issues fixed (reproducing test passes, existing tests still pass) | TBD |
| Median cost per fix | TBD |
| Median time per fix | TBD |

The full list of issues used and per-issue outcomes (including failures) is in [`eval/results.md`](eval/results.md).

<!-- Do not fill this table until you have run the evaluation. Report failures honestly. -->

## Scope and limitations

- Python repositories that use `pytest` only.
- Targets small, well-defined bugs with a clear reproduction path, not large refactors or feature requests.
- The verifier checks the repo's own tests plus the generated reproducing test, and blocks common ways of gaming them. Passing tests still do not guarantee a correct fix, so review every PR before merging.
- Code runs in a network-isolated container with a timeout.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (used for the sandbox)
- A Nebius Token Factory API key

### Install

```bash
git clone https://github.com/PradheebanAnandhan/FixLoop.git
cd FixLoop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy the example environment file and fill it in:

```bash
cp .env.example .env
```

```ini
NEBIUS_API_KEY=your-key-here
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/   # use the URL shown in your dashboard
MODEL_FAST=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
MODEL_MID=nvidia/nemotron-3-super-120b-a12b
MODEL_REASONING=nvidia/Nemotron-3-Ultra-550b-a55b
MAX_FIX_ATTEMPTS=5
```

Verify your connection and model IDs:

```bash
python scripts/check_models.py
```

### Run

```bash
python -m fixloop https://github.com/<owner>/<repo>/issues/<number>
```

Useful options: `--commit <sha>` (fix against a specific commit), `--repo <path-or-url>` (clone from somewhere other than github.com), `--issue-file issue.json` (offline, no GitHub API call), `--max-attempts N`, `--heldout gate|report|off`.

Each run writes to `runs/<owner>-<repo>-<n>-<timestamp>/`:

| File | Contents |
|---|---|
| `PR.md` | PR description with the verifier's before/after evidence |
| `fix.diff` | The verified source-only patch |
| `pr.diff` | The fix plus the new reproducing test, ready to apply |
| `REPORT.md` | Every fix attempt with the verifier's feedback |
| `result.json` | Status, attempts, per-step timings, tokens and cost per model |
| `trace.jsonl` | Every agent event (what the web UI streams) |
| `evidence/` | Verifier logs, per-test reports and verdicts for each run |

To launch the web UI that shows the agent's steps live:

```bash
python -m fixloop.web
```

The verifier can also be run on its own, for example to check a hand-written patch:

```bash
python -m verifier build-image --repo path/to/repo --commit <sha> --tag fixloop/repo:base
python -m verifier check --repo path/to/repo --commit <sha> --image fixloop/repo:base \
    --repro tests/test_fixloop_repro.py=./repro_test.py --patch fix.diff --evidence runs/demo
```

### Run the evaluation

```bash
python eval/run_eval.py
```

### Run FixLoop's own tests

```bash
pip install -r requirements-dev.txt
pytest            # Docker integration tests are skipped if no Docker daemon is running
```

---

## Implementation notes

- **Reasoning models need headroom.** The Nemotron models on Token Factory produce a reasoning trace before the answer. FixLoop sets a generous `max_tokens` and retries when a response comes back with empty content.
- **Sandboxing.** Each run executes in a fresh Docker container with networking disabled and a hard timeout.
- **Separation of roles.** The fixer agent and the verifier share no code path at decision time. The agent can read the verifier's rejection reasons but cannot modify the verifier, the stored reproducing test, or the evidence it writes.
- **Bounded retries.** The fix loop stops after `MAX_FIX_ATTEMPTS` and reports what it tried instead of looping forever.

## Project structure

```
FixLoop/
├── fixloop/          # agent loop and Nemotron model client
├── verifier/         # independent, deterministic accept/reject and Docker sandbox (no LLM calls, stdlib only)
├── eval/             # issue set, runner, results
├── scripts/          # utilities such as check_models.py
├── tests/            # FixLoop's own test suite
├── .env.example
├── LICENSE
└── README.md
```

<!-- Update this tree to match your real layout before submitting. -->

---

## Feedback on the tools

<!-- The hackathon requires feedback on Nebius Token Factory / AI Cloud and any NVIDIA tools used. Keep notes while you build and summarize them here and in the Devpost form. -->

## License

Released under the [MIT License](LICENSE). The hackathon requires an open-source license that is visible at the top of the repository page, so make sure the `LICENSE` file is committed to the root.

## Acknowledgments

Built with [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) models on [Nebius Token Factory](https://nebius.com/services/token-factory).
