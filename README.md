<<<<<<< HEAD
# GenAI-Companion-with-ACE

[![Release](https://img.shields.io/github/v/release/EgeCankaya/GenAI-Companion-with-ACE)](https://img.shields.io/github/v/release/EgeCankaya/GenAI-Companion-with-ACE)
[![Build status](https://img.shields.io/github/actions/workflow/status/EgeCankaya/GenAI-Companion-with-ACE/main.yml?branch=main)](https://github.com/EgeCankaya/GenAI-Companion-with-ACE/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/EgeCankaya/GenAI-Companion-with-ACE/branch/main/graph/badge.svg)](https://codecov.io/gh/EgeCankaya/GenAI-Companion-with-ACE)
[![Commit activity](https://img.shields.io/github/commit-activity/m/EgeCankaya/GenAI-Companion-with-ACE)](https://img.shields.io/github/commit-activity/m/EgeCankaya/GenAI-Companion-with-ACE)
[![License](https://img.shields.io/github/license/EgeCankaya/GenAI-Companion-with-ACE)](https://img.shields.io/github/license/EgeCankaya/GenAI-Companion-with-ACE)

Gen AI Companion that utilizes ACE.

- **Github repository**: <https://github.com/EgeCankaya/GenAI-Companion-with-ACE/>
- **Documentation**: <https://EgeCankaya.github.io/GenAI-Companion-with-ACE/>

## Overview

The IBM Gen AI Companion is an agentic study assistant designed for the IBM Generative AI Professional Certificate.
It combines a Retrieval-Augmented Generation (RAG) pipeline with the Agentic Context Engineering (ACE) framework to
deliver grounded answers with continuous, playbook-driven improvements.

Key capabilities:

- Local-first RAG pipeline with hybrid retrieval and dynamic document ingestion.
- ACE-powered generator, reflector, and curator loops that refine the companion playbook.
- Multi-mode conversations (Study, Quiz, Quick Reference) with persistent history.
- Evaluation suite with rubric-driven scoring and analytics to track improvement over time.

## Prerequisites

### System Requirements

- Python 3.9 – 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management (`pip install uv`)
- [Ollama](https://ollama.com/) installed locally
- 16 GB RAM (minimum recommended) for running `llama3.1:8b` inference locally

### Local Models

Once Ollama is installed, pull the required base models:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b      # optional fallback
```

Embeddings are generated with `sentence-transformers/all-MiniLM-L6-v2`. Models are downloaded on first use.

### Optional Hosted Services

- OpenAI API key (if you want to test hosted embedding or completion endpoints)
- Qdrant (remote vector store) if you outgrow local Chroma

## Project Setup

Clone the repository and install dependencies with uv:

```bash
git clone https://github.com/EgeCankaya/GenAI-Companion-with-ACE.git
cd GenAI-Companion-with-ACE
make install
```

This command creates a uv-managed virtual environment, installs all dependencies, and registers pre-commit hooks.

If you need to regenerate the lock file explicitly:

```bash
uv lock
```

## Development Workflow

Common tasks are wrapped in the `Makefile`:

```bash
make run      # start the interactive companion
make check    # lint, type-check, dependency audit
make test     # run pytest with coverage
make test-offline  # run the offline-safe subset
make docs     # build documentation site locally
```

Offline-friendly tests cover unit suites that rely on the new `@pytest.mark.offline` marker. You can also invoke them directly:

```bash
uv run pytest -m offline
uv run pytest tests/rag/test_answer_generation.py  # run a focused module
```

## Interactive CLI

Launch the study companion from your terminal:

```bash
make run
```

Or directly with uv:

```bash
uv run genai-companion chat --mode study
```

During a session you can:

- `:attach path/to/file.pdf` – inject ad-hoc documents into the next answer
- `:mode quiz` – switch between *study*, *quiz*, and *quick* response styles
- `:detail on` – force deep-dive answers (use `:detail off` to revert)
- `:long 800` – set a target length (words) for deep-dive answers
- `:box off` – fall back to unboxed answers if your terminal is narrow
- `:reflow off` – stop automatic re-printing when you resize the terminal
- `:source` – print the sources backing the last answer (nothing is appended inline by default)
- `:history` – recap the last conversational turns
- `:exit` – end the session (history is stored under `outputs/conversations/`)

The default blue box automatically stretches to the current terminal width. If you resize the window, the companion reprints the latest answer with the new width (best-effort—existing lines in the terminal cannot be retroactively reflowed). Use `:box off` or set `ui.cli.boxed_answers: false` if you prefer minimalist output everywhere. Sources are no longer appended to the bottom of answers; run `:source` whenever you want to see them.
### Detail-aware answers

When learners type phrases such as “in detail”, “deep dive”, or “comprehensive explanation”, the companion automatically switches to a deep-dive mode: it widens retrieval context, synthesizes the retrieved chunks into key points, generates a JSON outline, and then expands every outline section into a long-form response (with automatic continuation if the model stops early). You can force the behavior with `:detail on`, set a length target via `:long <words>`, and fetch the sources at any time with `:source`. If you need even more depth, configure `llm.deep_dive_provider` to use a hosted model for this two-pass workflow while normal answers stay local.

## Evaluation & Reporting

- Synthetic evaluation questions are generated automatically at `data/eval/eval_questions_100.json`.
- Run `uv run pytest tests/evaluation` to execute rubric-based scoring (RAGAS is used when available).
- Metrics reports are written to `outputs/metrics/` via `save_metrics_report`.

## Preview UI (Optional)

A lightweight Streamlit prototype lives in `ui/streamlit_app.py`. It lets you preview attached documents
and visualize evaluation coverage:

```bash
uv run streamlit run ui/streamlit_app.py
```

## Documentation

Detailed architecture, ACE integration notes, and API reference are maintained under `docs/` and published via MkDocs.
To preview docs locally:

```bash
uv run mkdocs serve
```

## Performance Evaluation & Troubleshooting

### Running Performance Evaluation

Evaluate the overall system health, including databases, playbooks, and agent success metrics:

```bash
python evaluate_performance.py
```

This generates a comprehensive report at `outputs/metrics/performance_evaluation.md` with:
- SQLite conversation history database status
- ChromaDB vector store health
- ACE playbook performance metrics
- Agent success metrics from conversation logs
- Actionable recommendations

Use the `--fix` flag for interactive issue detection and automated fixes:

```bash
python evaluate_performance.py --fix
```

### Common Issues and Solutions

#### ChromaDB Vector Store Corruption

**Symptoms:**
- Errors like "range start index X out of range" or "PanicException"
- RAG retrieval fails or returns no results
- Evaluation shows "corrupted_or_incompatible" status

**Solution:**
Reset the vector store (with automatic backup):

```bash
genai-companion reset-vector-store
```

This will:
1. Create a timestamped backup of the existing store
2. Clear the corrupted database
3. Allow re-initialization on next use

**After reset:** Re-ingest your documents using the ingestion pipeline.

#### Playbook Convergence Degraded

**Symptoms:**
- Evaluation shows convergence status as "degraded"
- Low BLEU/ROUGE scores in playbook metrics
- Poor answer quality

**Solution:**
Manually trigger ACE improvement cycles:

```bash
genai-companion trigger-ace
```

This runs the ACE framework's generator, reflector, and curator loops to refine the playbook. You need at least one conversation turn logged.

**Note:** ACE cycles also trigger automatically every 50 conversations (configurable in `configs/companion_config.yaml`).

#### No Conversation Logs

**Symptoms:**
- Evaluation shows "no_logs" status
- Cannot run ACE cycles
- No agent success metrics available

**Solution:**
Start using the companion to generate conversation logs:

```bash
genai-companion chat
```

Conversation logs are automatically saved to `outputs/conversations/` as JSONL files.

#### Heuristic Usage Tracking

**Symptoms:**
- All heuristics show 0 usage count
- Cannot determine which heuristics are effective

**Solution:**
Heuristic usage is now automatically tracked. New conversation turns will include heuristic IDs in metadata. To see usage statistics:

```bash
python evaluate_performance.py
```

Check the "Heuristic Usage Analysis" section in the generated report.

### CLI Commands Reference

- `genai-companion chat` - Start interactive chat session
- `genai-companion reset-vector-store` - Reset corrupted ChromaDB vector store
- `genai-companion trigger-ace` - Manually trigger ACE improvement cycles
- `python evaluate_performance.py` - Run comprehensive performance evaluation
- `python evaluate_performance.py --fix` - Interactive issue detection and fixes

### Checking System Status

Within a chat session, use the `:ace-status` command to check:
- Current playbook version and convergence status
- ACE framework availability
- Next automatic trigger point
- Recommendations for improvement

## Automated ACE Improvement

You can generate synthetic GenAI question/answer turns and run ACE improvement cycles locally.

### 1. Build an ACE Dataset

```bash
python scripts/build_ace_dataset.py \
  --questions data/ace/questions_genai.json \
  --output outputs/ace_datasets/ace_dataset_manual.json \
  --limit 25 \
  --shuffle-seed 42
```

The script loads curated IBM GenAI course questions, produces stub answers grounded in the key points, and saves an ACE-ready dataset. Use `scripts/validate_questions.py` to verify custom question files.

### 2. Run ACE Iterations (requires ACE framework)

```bash
python scripts/run_ace_iterations.py \
  --dataset outputs/ace_datasets/ace_dataset_manual.json \
  --iterations 2
```

This triggers ACE cycles (generator/reflector/curator). Resulting playbooks are saved to `outputs/ace_playbooks` with the naming pattern `playbook_auto_<timestamp>.yaml`. Each run also creates a Markdown report under `outputs/ace_runs/` summarizing the dataset hash, iterations, and playbook metadata. Ensure the ACE repository path in `configs/companion_config.yaml` points to a valid checkout. If the ACE framework uses a non-default configuration file, set `ace.config_path` (for example `../Agentic-Context-Engineering/configs/default.yaml`) so the runner can load it.

### 3. One-Command Workflow

```bash
make ace-improve ACE_ITERATIONS=2 ACE_LIMIT=20 ACE_SHUFFLE=7
```

`make ace-improve` chains the dataset builder and runner with sensible defaults:

1. Builds `outputs/ace_datasets/ace_dataset_auto.json`
2. Runs the ACE iterations count specified via `ACE_ITERATIONS` (default `1`)

Tune the optional variables:

- `ACE_LIMIT` – limit number of questions consumed from the curated list
- `ACE_SHUFFLE` – seed to shuffle question order
- `ACE_DATASET` – override output dataset path

Use these tools to keep the GenAI companion’s playbook current without touching production conversation logs.

## Configuration

Runtime behavior is controlled through `configs/companion_config.yaml`. The file now groups settings by concern:

- `llm.providers` and `embedding.providers` define multiple backends (e.g., Ollama locally, OpenAI in the cloud). Switching providers simply requires changing the `provider` key—provider-specific options (model, base URL, API key env vars) stay in one place. To allow longer answers, increase `llm.providers.<name>.max_tokens` (and `num_ctx`) in this section—the runtime now scales the per-call limit up to 1.5× whenever deep detail is requested.
- `ui.cli` controls presentation details for the Rich CLI. Example:

```yaml
ui:
  cli:
    boxed_answers: true        # set false to default to plain text
    reflow_on_resize: true     # re-render last answer when the terminal width changes
    box_style: simple          # simple or rounded
llm:
  provider: ollama             # primary model (local)
  deep_dive_provider: openai   # optional secondary provider for deep-dive two-pass pipeline
```
- `outputs` declares every directory the companion writes to (conversation logs, metrics, ACE playbooks, diagnostics), so deployments can relocate storage without digging through the codebase.
- `retrieval`, `ingestion`, `vector_store`, and `conversation_history` hold tunable knobs for chunking, hybrid retrieval, and memory retention.
- Paths are validated on startup; missing directories are created automatically for a smoother first-run experience.

## Ingestion Notes

- Supported formats: PDF, Markdown, plain text, JSON, YAML, SRT/VTT, IPYNB, and DOCX. Unsupported extensions now raise a clear error in both the CLI (`:attach`) and the ingestion scripts.
- Files larger than 10 MB are rejected by default (`ingestion.max_file_size_mb`). Adjust the limit per environment or keep it low to prevent accidental multi-hundred-megabyte uploads from blocking the pipeline.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
=======
# GenAI-Companion-with-ACE
>>>>>>> e3730d6e60502c8ed201d90fa229ce188f58a2f6
