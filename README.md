# FixLoop

**An issue-to-fix coding agent that proves its patches by running the tests.**

Give FixLoop a GitHub issue URL for a Python project. It clones the repo, writes a failing test that reproduces the bug, then iterates on a fix in a sandbox until the test passes. It finishes with a diff and a PR description that includes before/after test output as evidence.

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
4. Fix           propose a patch → run tests → read failures → retry (max N)
5. Deliver       diff + PR description with before/after test evidence
```

### Where each Nemotron model is used

| Step | Model | Why |
|---|---|---|
| Localize, summarize, triage | `NVIDIA-Nemotron-3-Nano-30B-A3B` | Fast, cheap calls that run many times per issue |
| Mid-weight tasks | `nemotron-3-super-120b-a12b` | Balance of speed and capability |
| Reproduce and Fix | `Nemotron-3-Ultra-550b-a55b` | Serious reasoning for the hard steps |

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
- Patches are verified against the repo's own tests plus the generated reproducing test. Passing tests do not guarantee a correct fix, so review every PR before merging.
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

To launch the web UI that shows the agent's steps live:

```bash
python -m fixloop.web
```

### Run the evaluation

```bash
python eval/run_eval.py
```

---

## Implementation notes

- **Reasoning models need headroom.** The Nemotron models on Token Factory produce a reasoning trace before the answer. FixLoop sets a generous `max_tokens` and retries when a response comes back with empty content.
- **Sandboxing.** Each run executes in a fresh Docker container with networking disabled and a hard timeout.
- **Bounded retries.** The fix loop stops after `MAX_FIX_ATTEMPTS` and reports what it tried instead of looping forever.

## Project structure

```
fixloop/
├── fixloop/          # agent loop, model client, sandbox runner
├── eval/             # issue set, runner, results
├── scripts/          # utilities such as check_models.py
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
