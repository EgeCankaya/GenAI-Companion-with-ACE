# Architecture Overview

The IBM Gen AI Companion is intentionally modular so that the CLI, Streamlit UI, and automated evaluation flows can reuse the same services. Each layer owns a clear set of responsibilities:

## Layers

1. **Interface layer**  
   The Rich-based CLI (`genai_companion_with_ace.cli`) and the experimental Streamlit UI talk to the runtime through the bootstrap helpers in `genai_companion_with_ace.app.bootstrap`. They never instantiate LangChain or ACE primitives directly.

2. **Chat & session management**  
   `ConversationManager` wraps a SQLite-backed `SessionHistoryStore` so every interface gets the same truncation and persistence logic. Modes (study / quiz / quick) live under `genai_companion_with_ace.chat`.

3. **RAG pipeline**  
   The `rag` package encapsulates ingestion, vector store access, retrieval orchestration, and answer generation. `AnswerGenerator` stitches together retrieved evidence, ACE playbooks, and the active conversation context before invoking the configured LLM provider.

4. **ACE integration**  
   `genai_companion_with_ace.ace_integration` owns conversation logging, trigger configuration, and playbook loading. The logger now maintains a turn counter cache and rotates JSONL files when they grow beyond ~5 MB to keep long-running agents stable.

5. **Evaluation**  
   The evaluation package (`genai_companion_with_ace.evaluation`) provides datasets, rubrics, and reporting helpers. The new CLI command `genai-companion evaluate` reuses the runtime stack to answer benchmark questions and writes metrics to `outputs/metrics/`.

## Configuration & Paths

- All runtime settings live in `configs/companion_config.yaml`. The `llm` and `embedding` sections support multiple providers at once, while the new `outputs` block lists every directory the app writes to (conversations, metrics, ACE playbooks, diagnostic logs).
- `CompanionConfig` validates and creates directories at startup so deployments fail fast when misconfigured.

## Diagnostics & Health

- `genai_companion_with_ace.diagnostics.run_diagnostics` performs a lightweight health check (Python version, Ollama model availability, vector store access, ACE repository presence, optional libraries). The CLI will expose this soon, but you can already call it programmatically.
- `:debug-retrieval` in the CLI prints the latest dense/keyword/attachment matches along with their scores so tunable parameters in `retrieval` can be inspected live.

## Storage overview

| Component              | Location (defaults)            | Notes                                                                          |
|------------------------|--------------------------------|--------------------------------------------------------------------------------|
| Vector store (Chroma)  | `data/chroma`                  | Health-checked before resets; backups created automatically on reset.          |
| Processed documents    | `data/processed`               | Persistent chunk manifests for ingestion builds.                               |
| Conversation history   | `outputs/conversations`        | JSONL logs rotate automatically; SQLite history DB used for session context.   |
| Playbooks              | `outputs/ace_playbooks`        | New playbooks + trigger state stored here.                                     |
| Metrics & evaluation   | `outputs/metrics`              | `save_metrics_report` writes JSON + Markdown summaries here.                   |


