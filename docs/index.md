# GenAI-Companion-with-ACE

[![Release](https://img.shields.io/github/v/release/EgeCankaya/GenAI-Companion-with-ACE)](https://img.shields.io/github/v/release/EgeCankaya/GenAI-Companion-with-ACE)
[![Build status](https://img.shields.io/github/actions/workflow/status/EgeCankaya/GenAI-Companion-with-ACE/main.yml?branch=main)](https://github.com/EgeCankaya/GenAI-Companion-with-ACE/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/EgeCankaya/GenAI-Companion-with-ACE)](https://img.shields.io/github/commit-activity/m/EgeCankaya/GenAI-Companion-with-ACE)
[![License](https://img.shields.io/github/license/EgeCankaya/GenAI-Companion-with-ACE)](https://img.shields.io/github/license/EgeCankaya/GenAI-Companion-with-ACE)

Gen AI Companion that utilizes ACE

## Highlights

- Retrieval-Augmented Generation (RAG) with hybrid dense + keyword retrieval.
- ACE-powered playbook loops (generator → reflector → curator).
- Rich CLI with multi-mode chat and on-demand document attachments.
- Synthetic evaluation dataset with rubric scoring and metrics reporting.

## Quick Start

```bash
uv run genai-companion chat --mode study
```

Commands inside the CLI:

- `:attach <file>` – ingest an ad-hoc PDF/Markdown/Text file
- `:mode <study|quiz|quick>` – adapt the response style
- `:history` – show recent turns
- `:exit` – end the session

## Evaluation

```bash
uv run pytest tests/evaluation
```

Evaluation reports are saved to `outputs/metrics/`.