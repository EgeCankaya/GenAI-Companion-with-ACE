# Operations & Troubleshooting

## Core CLI commands

| Command                               | Description |
|---------------------------------------|-------------|
| `genai-companion chat`                | Interactive assistant with `:help`, `:history`, `:debug-retrieval`, `:ace-status`. |
| `genai-companion evaluate`            | Runs the rubric-based benchmark using the current runtime stack and writes metrics to `outputs/metrics/`. |
| `genai-companion reset-vector-store`  | Backs up and resets the Chroma collection referenced in `companion_config.yaml`. |
| `genai-companion trigger-ace`         | Forces an ACE improvement cycle using the logged conversation turns. |

Use `genai-companion COMMAND --help` for per-command options (dataset paths, iteration counts, etc.).

## Running evaluations

1. Ensure `data/eval/eval_questions_100.json` exists (the CLI will auto-generate the default dataset if it does not).
2. Execute `genai-companion evaluate --limit 25` to sample a subset or omit `--limit` for the full set.
3. Metrics are written to `outputs/metrics/evaluation.json` and `evaluation.md`. These integrate with MkDocs so you can publish the markdown report directly.

The evaluator reuses the normal retrieval + answer generation stack, so any configuration tweaks (new embeddings, provider switches, playbook updates) are captured automatically.

## Health checks

- Run the Python helper:

  ```python
  from pathlib import Path
  from genai_companion_with_ace.config import CompanionConfig
  from genai_companion_with_ace.diagnostics import run_diagnostics

  config = CompanionConfig.from_file(Path("configs/companion_config.yaml"))
  print(run_diagnostics(config))
  ```

- Inside the chat CLI:
  - `:ace-status` prints the most recent playbook metadata and ACE repository status.
  - `:debug-retrieval` prints the last dense / keyword / attachment hits with scores for quick tuning.

## Troubleshooting tips

- **Vector store corruption** – run `genai-companion reset-vector-store`. The command now validates health, creates a timestamped backup, and reminds you to re-ingest content.
- **ACE auto-trigger cadence** – conversation turn counts are cached on disk so automatic ACE runs resume exactly at the configured threshold even after restarts. Inspect the state file at `outputs/ace_playbooks/.ace_trigger_state.json` if needed.
- **Large conversation logs** – JSONL files rotate once they exceed ~5 MB (suffix `session-id.<timestamp>.jsonl`). This keeps ACE exports fast without changing the log format.
- **Attachments rejected** – ingestion now enforces an extension whitelist and a configurable max file size (`ingestion.max_file_size_mb`). Adjust it if you routinely feed large PDFs, otherwise the CLI will explain why a file was skipped.

