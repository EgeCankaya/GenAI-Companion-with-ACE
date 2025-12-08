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

## ACE Validation & Reporting

### Playbook Validation

ACE playbooks are automatically validated for structural issues:

- **Run validator manually:**
  ```bash
  uv run python scripts/validate_playbook.py --all
  ```

- **Validate specific playbook:**
  ```bash
  uv run python scripts/validate_playbook.py outputs/ace_playbooks/playbook_auto_YYYYMMDD_HHMMSS.yaml
  ```

The validator checks for:
- Duplicate heuristic IDs
- Repetitive system instructions
- Near-duplicate heuristics

Validation runs automatically:
- In pre-commit hooks (when committing playbook files)
- In `make check` (validates all playbooks)

### Evaluation Reports

After each ACE run, a comprehensive evaluation report is automatically generated:

- **Location:** `outputs/ace_runs/ACE_EVALUATION_REPORT_<timestamp>.md`
- **Generated automatically:** After `make ace-improve` completes
- **Generate manually:**
  ```bash
  uv run python scripts/generate_ace_report.py
  ```

The report includes:
- Performance metrics analysis (emphasizes semantic similarity over BLEU/ROUGE)
- Trend analysis comparing with previous runs
- Convergence status and recommendations
- Playbook structure issue flags

**Note:** BLEU and ROUGE scores are typically low for instructional/creative content where exact matches are rare. The report emphasizes semantic similarity, token counts, and inference time as more relevant metrics for educational content.

### ACE Iteration Tuning

If convergence status is consistently "degraded":

1. **Review evaluation metrics** – BLEU/ROUGE may not be appropriate for your content type
2. **Adjust iteration count** – Try fewer iterations with higher-quality datasets:
   ```bash
   make ace-improve ACE_ITERATIONS=3 ACE_LIMIT=50
   ```
3. **Increase dataset diversity** – Ensure your dataset covers diverse question types
4. **Check playbook structure** – Run the validator to ensure no structural issues
5. **Review ACE framework config** – The ACE framework's evaluation criteria may need adjustment

## Troubleshooting tips

- **Vector store corruption** – run `genai-companion reset-vector-store`. The command now validates health, creates a timestamped backup, and reminds you to re-ingest content.
- **ACE auto-trigger cadence** – conversation turn counts are cached on disk so automatic ACE runs resume exactly at the configured threshold even after restarts. Inspect the state file at `outputs/ace_playbooks/.ace_trigger_state.json` if needed.
- **Large conversation logs** – JSONL files rotate once they exceed ~5 MB (suffix `session-id.<timestamp>.jsonl`). This keeps ACE exports fast without changing the log format.
- **Attachments rejected** – ingestion now enforces an extension whitelist and a configurable max file size (`ingestion.max_file_size_mb`). Adjust it if you routinely feed large PDFs, otherwise the CLI will explain why a file was skipped.
- **Playbook validation failures** – If validation fails, check for duplicate heuristic IDs or repetitive instructions. The validator output will indicate the specific issues.
