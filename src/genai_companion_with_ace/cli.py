"""Command-line interface for the IBM Gen AI Companion."""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from genai_companion_with_ace.ace_integration import (
    ACETriggerConfig,
    ConversationLogger,
    PlaybookLoader,
    run_ace_cycles,
)
from genai_companion_with_ace.app.bootstrap import RuntimeComponents, load_runtime_from_path
from genai_companion_with_ace.chat import MODE_REGISTRY, ConversationManager, ConversationMode
from genai_companion_with_ace.chat.query_processor import AttachmentInput
from genai_companion_with_ace.config import CompanionConfig
from genai_companion_with_ace.evaluation import (
    EvaluationDataset,
    EvaluationEngine,
    EvaluationRubric,
    ensure_default_dataset,
    save_metrics_report,
)
from genai_companion_with_ace.llm import check_ollama_connection, check_ollama_model
from genai_companion_with_ace.rag import (
    DocumentIngestionPipeline,
    EmbeddingFactory,
    GenerationError,
    VectorStoreConfig,
    VectorStoreManager,
)

console = Console()

_BOX_CHARSETS = {
    "simple": {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"},
    "rounded": {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"},
}


def _is_ace_framework_available() -> bool:
    return importlib.util.find_spec("agentic_context_engineering.runners.ace_runner") is not None


def perform_vector_store_reset(companion_config: CompanionConfig, *, force: bool) -> None:
    vector_cfg = companion_config.vector_store_config()
    persist_dir = vector_cfg.persist_directory
    collection_name = vector_cfg.collection_name

    console.print("[bold]Vector Store Reset[/bold]")
    console.print(f"Collection: [cyan]{collection_name}[/cyan]")
    console.print(f"Directory: [cyan]{persist_dir}[/cyan]")
    console.print()

    if not persist_dir.exists():
        console.print("[yellow]Vector store directory does not exist. Nothing to reset.[/yellow]")
        return

    vector_store = _build_vector_store_manager(companion_config, vector_cfg)
    _report_vector_store_health(vector_store)
    proceed, backup_dir = _maybe_backup_existing_store(persist_dir, force)
    if not proceed:
        return
    _perform_store_reset(vector_store, backup_dir)


def _build_vector_store_manager(companion_config: CompanionConfig, vector_cfg: VectorStoreConfig) -> VectorStoreManager:
    embeddings = EmbeddingFactory(companion_config.embedding_settings()).build()
    return VectorStoreManager(embeddings, vector_cfg)


def _report_vector_store_health(vector_store: VectorStoreManager) -> None:
    is_healthy, error_msg = vector_store.check_health()
    if is_healthy:
        console.print("[yellow]Warning: Vector store appears healthy. Reset may not be necessary.[/yellow]")
    else:
        console.print(f"[red]Vector store health check failed:[/red] {error_msg}")
    console.print()


def _maybe_backup_existing_store(persist_dir: Path, force: bool) -> tuple[bool, Path | None]:
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        return True, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = persist_dir.parent / f"{persist_dir.name}.backup.{timestamp}"
    if not force:
        console.print("[yellow]This will backup the existing store to:[/yellow]")
        console.print(f"  [cyan]{backup_dir}[/cyan]")
        console.print()
        console.print("[yellow]After reset, you will need to re-ingest your documents.[/yellow]")
        console.print()
        if not click.confirm("Do you want to continue?"):
            console.print("[yellow]Reset cancelled.[/yellow]")
            return False, None
    try:
        console.print("[dim]Creating backup...[/dim]")
        shutil.copytree(persist_dir, backup_dir, dirs_exist_ok=True)
        console.print(f"[green]Backup created:[/green] {backup_dir}")
    except Exception as exc:  # pragma: no cover - filesystem errors vary
        console.print(f"[red]Warning: Failed to create backup:[/red] {exc}")
        if force or click.confirm("Continue without backup?"):
            return True, None
        console.print("[yellow]Reset cancelled.[/yellow]")
        return False, None
    else:
        return True, backup_dir


def _perform_store_reset(vector_store: VectorStoreManager, backup_dir: Path | None) -> None:
    try:
        console.print("[dim]Resetting vector store...[/dim]")
        vector_store.reset()
        console.print("[green]Vector store reset successfully![/green]\n")
        console.print("[bold]Next steps:[/bold]")
        console.print("1. Re-ingest your documents using the ingestion pipeline")
        console.print("2. The vector store will be recreated automatically on first use")
        if backup_dir:
            console.print(f"3. Backup is available at: [cyan]{backup_dir}[/cyan]")
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]Error resetting vector store:[/red] {exc}")
        if backup_dir and backup_dir.exists():
            console.print(f"[yellow]Backup is still available at:[/yellow] {backup_dir}")


def perform_trigger_ace(companion_config: CompanionConfig, *, iterations: int | None) -> None:
    ace_cfg = companion_config.ace_config()
    conversation_logs_dir = companion_config.outputs.conversations

    logger = ConversationLogger(conversation_logs_dir)
    playbook_loader = PlaybookLoader.from_companion_config(companion_config)

    turn_count = logger.count_logged_turns()
    if turn_count == 0:
        console.print("[red]Error: No conversation logs found.[/red]")
        console.print("You need to have at least one conversation turn logged to run ACE cycles.")
        console.print(f"Conversation logs directory: [cyan]{conversation_logs_dir}[/cyan]")
        return

    console.print("[bold]ACE Improvement Cycles[/bold]")
    console.print(f"Conversation turns available: [cyan]{turn_count}[/cyan]")
    console.print()

    current_playbook = playbook_loader.load_latest()
    console.print(f"Current playbook version: [cyan]{current_playbook.version}[/cyan]")
    console.print()

    num_iterations = iterations if iterations is not None else ace_cfg.get("iterations", 1)
    trigger_config = ACETriggerConfig(
        repo_path=Path(ace_cfg.get("repository_path", "../Agentic-Context-Engineering")),
        playbook_output_dir=companion_config.outputs.ace_playbooks,
        iterations=num_iterations,
        trigger_threshold=ace_cfg.get("trigger_threshold", 50),
        config_path=Path(str(ace_cfg.get("config_path"))) if ace_cfg.get("config_path") else None,
    )

    console.print(f"Running [cyan]{num_iterations}[/cyan] ACE iteration(s)...")
    console.print()

    if not _is_ace_framework_available():
        console.print("[red]Error: ACE framework not available.[/red]")
        console.print()
        console.print("Make sure 'agentic-context-engineering' is installed:")
        console.print("  [cyan]pip install agentic-context-engineering[/cyan]")
        console.print()
        console.print("Or ensure the repository path is correct:")
        console.print(f"  [cyan]{trigger_config.repo_path}[/cyan]")
        return

    console.print("[green]ACE Framework:[/green] Available")

    try:
        console.print("[dim]Running ACE cycles...[/dim]")
        new_playbook = run_ace_cycles(
            conversation_logger=logger,
            playbook_loader=playbook_loader,
            trigger_config=trigger_config,
        )

        console.print()
        console.print("[green]ACE cycles completed successfully![/green]")
        console.print()
        console.print(f"New playbook version: [cyan]{new_playbook.version}[/cyan]")

        playbooks = playbook_loader.list_available()
        if playbooks:
            latest_playbook_path = playbooks[-1]
            try:
                playbook_data = yaml.safe_load(latest_playbook_path.read_text(encoding="utf-8"))
                metadata = playbook_data.get("metadata", {})
                perf_metrics = metadata.get("performance_metrics", {})
                convergence = metadata.get("convergence_status", "unknown")

                console.print()
                console.print("[bold]Playbook Metrics:[/bold]")
                console.print(f"  Convergence Status: [cyan]{convergence}[/cyan]")
                if perf_metrics:
                    console.print(f"  Accuracy: {perf_metrics.get('accuracy', 0.0):.3f}")
                    console.print(f"  Semantic Similarity: {perf_metrics.get('semantic_similarity', 0.0):.3f}")
                    console.print(f"  BLEU Score: {perf_metrics.get('bleu_score', 0.0):.4f}")
                    console.print(f"  ROUGE Score: {perf_metrics.get('rouge_score', 0.0):.4f}")
            except Exception as exc:  # pragma: no cover - best effort diagnostics
                console.print(f"[yellow]Warning: Unable to read playbook metrics ({exc}).[/yellow]")

        console.print()
        console.print("[bold]Next steps:[/bold]")
        console.print("1. The new playbook will be used automatically in future conversations")
        console.print("2. Run the evaluation script to see performance improvements")
        console.print(
            f"3. View playbook: [cyan]cat {playbooks[-1]}[/cyan]"
            if playbooks
            else "3. View playbook in outputs/ace_playbooks"
        )

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
    except Exception as e:  # pragma: no cover
        console.print(f"[red]Error running ACE cycles:[/red] {e}")
        console.print()
        console.print("[yellow]Troubleshooting:[/yellow]")
        console.print("1. Ensure ACE framework repository is accessible")
        console.print(f"   Path: [cyan]{trigger_config.repo_path}[/cyan]")
        console.print("2. Check that conversation logs are valid")
        console.print(f"   Directory: [cyan]{conversation_logs_dir}[/cyan]")
        console.print("3. Verify ACE framework dependencies are installed")


def render_assistant_message(content: str, *, boxed: bool = True, box_style: str = "simple") -> None:
    """Render assistant message with optional boxing that adapts to terminal width."""
    markdown = Markdown(content)
    if boxed:
        if not console.is_terminal:
            _render_boxed_fallback(content, box_style)
            return
        box_styles = {
            "simple": box.SIMPLE,
            "rounded": box.ROUNDED,
        }
        chosen_box = box_styles.get(box_style.lower(), box.SIMPLE)
        console.print(
            Panel(
                markdown,
                title="Companion",
                border_style="blue",
                box=chosen_box,
                expand=True,
                padding=(0, 1),
            )
        )
    else:
        console.rule("Companion")
        console.print(markdown)


def maybe_reflow_last_answer(
    last_render: dict[str, Any] | None,
    *,
    auto_reflow: bool,
) -> dict[str, Any] | None:
    """Re-render the last assistant answer when the terminal width changes."""
    if not auto_reflow or not last_render:
        return last_render

    current_width = console.size.width
    if current_width != last_render.get("width"):
        last_render["width"] = current_width
        render_assistant_message(
            last_render["content"],
            boxed=last_render["boxed"],
            box_style=last_render["box_style"],
        )
    return last_render


def handle_attachment(ingestion: DocumentIngestionPipeline, path_str: str) -> AttachmentInput | None:
    # Strip quotes from the path if present (handles both single and double quotes)
    cleaned_path = path_str.strip().strip('"').strip("'")
    path = Path(cleaned_path).expanduser()
    if not path.exists():
        console.print(f"[red]Attachment not found:[/] {path}")
        return None
    try:
        docs = ingestion.ingest_path(path, persist=False)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return None
    combined = "\n\n".join(doc.page_content for doc in docs)
    if not combined.strip():
        console.print(f"[yellow]Attachment {path} produced no extractable text.[/]")
        return None
    return AttachmentInput(name=path.name, content=combined, metadata={"source_path": str(path)})


def list_modes() -> str:
    descriptions = {
        ConversationMode.STUDY: "Detailed, pedagogy-focused responses",
        ConversationMode.QUIZ: "Hint-style answers to encourage recall",
        ConversationMode.QUICK: "Concise reference answers",
    }
    lines = [f"- {mode.value}: {desc}" for mode, desc in descriptions.items()]
    return "\n".join(lines)


def _render_boxed_fallback(content: str, box_style: str) -> None:
    """Render a unicode box manually when Rich falls back to plain text output."""
    glyphs = _BOX_CHARSETS.get(box_style.lower(), _BOX_CHARSETS["simple"])
    lines = content.splitlines() or [""]
    size = getattr(console, "size", None)
    available_width = getattr(console, "width", None) or (size.width if size else None) or 80
    available_width = max(20, available_width)
    title = " Companion "
    max_line = max(len(line) for line in [*lines, title.strip()])
    inner_width = min(available_width - 2, max_line + 4)
    inner_width = max(inner_width, len(title), 6)
    content_width = max(1, inner_width - 2)

    def wrap_line(text: str) -> list[str]:
        if len(text) <= content_width:
            return [text]
        chunks: list[str] = []
        remaining = text
        while remaining:
            chunks.append(remaining[:content_width])
            remaining = remaining[content_width:]
        return chunks

    wrapped_lines: list[str] = []
    for raw_line in lines:
        wrapped_lines.extend(wrap_line(raw_line.rstrip()))
    if not wrapped_lines:
        wrapped_lines.append("")

    top = f"{glyphs['tl']}{title.center(inner_width, glyphs['h'])}{glyphs['tr']}"
    bottom = f"{glyphs['bl']}{glyphs['h'] * inner_width}{glyphs['br']}"

    render_lines = [top]
    for wrapped in wrapped_lines:
        padded = wrapped.ljust(content_width)
        render_lines.append(f"{glyphs['v']} {padded[:content_width]} {glyphs['v']}")
    render_lines.append(bottom)

    console.print(Text("\n".join(render_lines)))


def handle_cli_command(  # noqa: C901
    command: str,
    *,
    ingestion: DocumentIngestionPipeline,
    pending_attachments: list[AttachmentInput],
    mode_ref: dict[str, Any],
    runtime: RuntimeComponents | None = None,
    conversation_manager: ConversationManager | None = None,
    session_id: str | None = None,
) -> bool:
    lowered = command.strip().lower()
    if lowered == ":help":
        _print_help_menu()
        return True
    if lowered == ":playbook":
        _render_playbook(runtime)
        return True
    if lowered == ":ace-status":
        _render_ace_status(runtime)
        return True
    if lowered.startswith(":attach"):
        _attach_document(command, ingestion, pending_attachments)
        return True
    if lowered.startswith(":mode"):
        _set_conversation_mode(command, mode_ref)
        return True
    if lowered.startswith(":detail"):
        _toggle_detail_mode(command, mode_ref)
        return True
    if lowered.startswith(":long"):
        _set_target_words(command, mode_ref)
        return True
    if lowered.startswith(":box"):
        _toggle_boolean_setting(command, mode_ref, "boxed", "Boxed answers")
        return True
    if lowered.startswith(":reflow"):
        _toggle_boolean_setting(command, mode_ref, "auto_reflow", "Auto reflow")
        return True
    if lowered == ":history" and conversation_manager and session_id:
        _render_history(conversation_manager, session_id)
        return True
    if lowered == ":sessions" and conversation_manager:
        _render_sessions(conversation_manager)
        return True
    if lowered.startswith(":view") and conversation_manager:
        _render_session_view(command, conversation_manager)
        return True
    if lowered == ":debug-retrieval":
        _render_retrieval_debug(runtime)
        return True
    return False


def _print_help_menu() -> None:
    console.print(
        "Commands:\n"
        "  :attach <path>  - Attach a document for the next question\n"
        "  :mode <name>    - Switch mode (study, quiz, quick)\n"
        "  :detail on/off  - Toggle detailed answers for the session\n"
        "  :long <words>   - Set target word count for deep-dive answers\n"
        "  :box on/off     - Toggle boxed rendering for answers\n"
        "  :reflow on/off  - Toggle automatic re-render on terminal resize\n"
        "  :history        - Show recent conversation turns (current session)\n"
        "  :sessions       - List all conversation sessions\n"
        "  :view <id>      - View full history of a specific session\n"
        "  :playbook       - Show current playbook location and version\n"
        "  :ace-status     - Check ACE integration status and agents\n"
        "  :debug-retrieval- Inspect the last retrieval context\n"
        "  :exit           - End the session"
    )


def _render_playbook(runtime: RuntimeComponents | None) -> None:  # noqa: C901
    playbook_loader = runtime.playbook_loader if runtime else None
    if not playbook_loader:
        console.print("[red]Playbook loader not available[/red]")
        return
    playbook = playbook_loader.load_latest()
    available = playbook_loader.list_available()
    console.print("[bold]Current Playbook:[/bold]")
    console.print(f"  Version: [cyan]{playbook.version}[/cyan]")
    console.print(f"  Directory: [cyan]{playbook_loader._playbook_dir}[/cyan]")
    if available:
        latest = available[-1]
        console.print(f"  Latest file: [cyan]{latest}[/cyan]")
        console.print(f"  File size: [dim]{latest.stat().st_size} bytes[/dim]")
        console.print(f"  Modified: [dim]{latest.stat().st_mtime}[/dim]")
    else:
        console.print("[yellow]No playbook files found. Using default.[/yellow]")
        if playbook_loader._default_path and playbook_loader._default_path.exists():
            console.print(f"  Default playbook: [cyan]{playbook_loader._default_path}[/cyan]")

    console.print("\n[bold]System Instructions:[/bold]")
    console.print(f"  {playbook.system_instructions}")
    if playbook.heuristics:
        console.print(f"\n[bold]Heuristics ({len(playbook.heuristics)}):[/bold]")
        for idx, heuristic in enumerate(playbook.heuristics, 1):
            console.print(f"  {idx}. {heuristic}")
    if playbook.examples:
        console.print(f"\n[bold]Examples ({len(playbook.examples)}):[/bold]")
        for idx, example in enumerate(playbook.examples[:3], 1):
            if "user" in example:
                console.print(f"  {idx}. User: {example['user'][:60]}...")
            if "assistant" in example:
                console.print(f"     Assistant: {example['assistant'][:60]}...")

    console.print("\n[bold]How to view/edit:[/bold]")
    if available:
        console.print(f"  View file: [cyan]cat {available[-1]}[/cyan]")
        console.print(f"  Or open in editor: [cyan]code {available[-1]}[/cyan]")
    console.print(f"  Directory: [cyan]dir {playbook_loader._playbook_dir}[/cyan]")


def _attach_document(
    command: str,
    ingestion: DocumentIngestionPipeline,
    pending_attachments: list[AttachmentInput],
) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) != 2:
        console.print("Usage: :attach <file_path>")
        return
    attachment = handle_attachment(ingestion, parts[1])
    if attachment:
        pending_attachments.append(attachment)
        console.print(f"[cyan]Attached:[/] {attachment.name}")


def _set_conversation_mode(command: str, mode_ref: dict[str, Any]) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) == 2 and parts[1] in MODE_REGISTRY:
        mode_ref["current"] = ConversationMode(parts[1])
        console.print(f"[cyan]Switched to {mode_ref['current'].value} mode.[/]")
    else:
        console.print("Available modes: study, quiz, quick")


def _toggle_detail_mode(command: str, mode_ref: dict[str, Any]) -> None:
    parts = command.split(maxsplit=1)
    setting = parts[1].strip().lower() if len(parts) == 2 else ""
    if setting in {"on", "off", "deep"}:
        mode_ref["detail"] = setting in {"on", "deep"}
        state = "on" if mode_ref["detail"] else "off"
        console.print(f"[cyan]Detail mode: {state}[/]")
    else:
        console.print("Usage: :detail on|off")


def _set_target_words(command: str, mode_ref: dict[str, Any]) -> None:
    parts = command.split(maxsplit=1)
    value = parts[1].strip() if len(parts) == 2 else ""
    if value.isdigit():
        mode_ref["target_words"] = int(value)
        console.print(f"[cyan]Target length set to {mode_ref['target_words']} words.[/]")
    else:
        console.print("Usage: :long <word_count>")


def _toggle_boolean_setting(command: str, mode_ref: dict[str, Any], key: str, label: str) -> None:
    parts = command.split(maxsplit=1)
    setting = parts[1].strip().lower() if len(parts) == 2 else ""
    if setting in {"on", "off"}:
        mode_ref[key] = setting == "on"
        console.print(f"[cyan]{label}: {'on' if mode_ref[key] else 'off'}[/]")
    else:
        console.print(f"Usage: :{key.replace('_', '')} on|off")


def _render_history(conversation_manager: ConversationManager, session_id: str) -> None:
    history = conversation_manager.get_recent_context(session_id)
    if not history:
        console.print("[yellow]No conversation history found for current session.[/yellow]")
        console.print("[dim]Tip: use :sessions to see other session IDs.[/dim]")
        console.print()
        return
    console.print(f"[bold]Conversation History (Session: {session_id})[/bold]\n")
    for message in history:
        speaker = "You" if message.role == "user" else "Companion"
        role_color = "green" if message.role == "user" else "blue"
        console.print(f"[{role_color}]{speaker}:[/] {message.content}")
        if message.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in message.metadata.items() if k != "content")
            if meta_str:
                console.print(f"  [dim]{meta_str}[/dim]")
    console.print("[dim]Use :view <session_id> to see the complete conversation.[/dim]\n")


def _render_sessions(conversation_manager: ConversationManager) -> None:
    sessions = conversation_manager.list_sessions(limit=20)
    if not sessions:
        console.print("[yellow]No conversation sessions found.[/yellow]\n")
        return
    console.print(f"[bold]Conversation Sessions ({len(sessions)} found)[/bold]\n")
    for idx, sess in enumerate(sessions, 1):
        console.print(f"{idx}. [cyan]{sess.session_id}[/cyan]")
        console.print(f"   Created: [dim]{sess.created_at}[/dim]")
        if sess.course:
            console.print(f"   Course: [yellow]{sess.course}[/yellow]")
        if sess.mode:
            console.print(f"   Mode: [yellow]{sess.mode}[/yellow]")
        if sess.summary:
            console.print(f"   Summary: {sess.summary[:100]}...")
        console.print()
    console.print("[dim]Use :view <session_id> to inspect a session or resume it later.[/dim]\n")


def _render_session_view(command: str, conversation_manager: ConversationManager) -> None:
    parts = command.split(maxsplit=1)
    if len(parts) != 2:
        console.print("[yellow]Usage: :view <session_id>[/yellow]")
        console.print("[yellow]Use :sessions to list available session IDs[/yellow]\n")
        return
    target_session_id = parts[1].strip()
    all_messages = conversation_manager._store.get_messages(target_session_id)
    session_info = conversation_manager.get_session(target_session_id)
    if not all_messages:
        console.print(f"[red]Session '{target_session_id}' not found or has no messages.[/red]\n")
        return
    if session_info:
        console.print(f"[bold]Session: {target_session_id}[/bold]")
        console.print(f"Created: [dim]{session_info.created_at}[/dim]")
        if session_info.course:
            console.print(f"Course: [yellow]{session_info.course}[/yellow]")
        if session_info.mode:
            console.print(f"Mode: [yellow]{session_info.mode}[/yellow]")
        console.print()
    console.print(f"[bold]Full Conversation History ({len(all_messages)} messages)[/bold]\n")
    for message in all_messages:
        speaker = "You" if message.role == "user" else "Companion"
        role_color = "green" if message.role == "user" else "blue"
        console.print(f"[{role_color}]{speaker}:[/] {message.content}")
        if message.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in message.metadata.items() if k != "content")
            if meta_str:
                console.print(f"  [dim]{meta_str}[/dim]")
        console.print()


def _render_ace_status(runtime: RuntimeComponents | None) -> None:  # noqa: C901
    if runtime is None:
        console.print("[yellow]Runtime not available; ACE status is unknown.[/yellow]")
        return
    logger = runtime.conversation_logger
    trigger_config = runtime.trigger_config
    playbook_loader = runtime.playbook_loader
    console.print("[bold]ACE Integration Status[/bold]\n")
    if logger:
        turn_count = logger.count_logged_turns()
        console.print("✅ [green]Conversation Logger:[/green] Active")
        console.print(f"   Logged turns: [cyan]{turn_count}[/cyan]")
        console.print(f"   Log directory: [cyan]{logger._output_dir}[/cyan]")
    else:
        console.print("❌ [red]Conversation Logger:[/red] Not configured")
    if trigger_config:
        console.print("\n✅ [green]ACE Trigger Config:[/green] Configured")
        console.print(f"   Repository path: [cyan]{trigger_config.repo_path}[/cyan]")
        console.print(f"   Trigger threshold: [cyan]{trigger_config.trigger_threshold}[/cyan] conversations")
        console.print(f"   Iterations: [cyan]{trigger_config.iterations}[/cyan]")
        if trigger_config.repo_path.exists():
            console.print("   ✅ ACE repository found")
        else:
            console.print(f"   ⚠️  [yellow]ACE repository not found at: {trigger_config.repo_path}[/yellow]")
    else:
        console.print("\n❌ [red]ACE Trigger Config:[/red] Not configured")
    if playbook_loader:
        console.print("\n✅ [green]Playbook Loader:[/green] Active")
        console.print(f"   Playbook directory: [cyan]{playbook_loader._playbook_dir}[/cyan]")
        available = playbook_loader.list_available()
        console.print(f"   Available playbooks: [cyan]{len(available)}[/cyan]")
        if available:
            latest_path = available[-1]
            console.print(f"   Latest: [cyan]{latest_path.name}[/cyan]")
            try:
                playbook_data = yaml.safe_load(latest_path.read_text(encoding="utf-8"))
                metadata = playbook_data.get("metadata", {})
                convergence = metadata.get("convergence_status", "unknown")
                console.print("\n[bold]Current Playbook Metrics:[/bold]")
                console.print(f"   Version: [cyan]{playbook_data.get('version', 'unknown')}[/cyan]")
                console.print(f"   Iteration: [cyan]{metadata.get('iteration', 0)}[/cyan]")
                console.print(f"   Convergence: [yellow]{convergence}[/yellow]")
            except Exception as exc:
                console.print(f"   [yellow]Could not load playbook metrics: {exc}[/yellow]")
        console.print("")
        if _is_ace_framework_available():
            console.print("✅ [green]ACE Framework:[/green] Available")
            console.print("   Generator, Reflector, and Curator agents are available")
        else:
            console.print("❌ [red]ACE Framework:[/red] Not installed")
            console.print("   Make sure 'agentic-context-engineering' is available on PYTHONPATH")
    else:
        console.print("\n❌ [red]Playbook Loader:[/red] Not configured")
    if logger and trigger_config:
        turn_count = logger.count_logged_turns()
        threshold = trigger_config.trigger_threshold
        next_trigger = ((turn_count // threshold) + 1) * threshold
        remaining = next_trigger - turn_count
        console.print("\n[bold]Next ACE Cycle:[/bold]")
        if remaining <= threshold:
            console.print(f"   Will trigger automatically after [cyan]{remaining}[/cyan] more conversations")
        else:
            console.print(f"   [yellow]Not scheduled (need {remaining} more conversations)[/yellow]")
            console.print("   [dim]Run: [cyan]genai-companion trigger-ace[/cyan][/dim]")
    console.print()


def _render_retrieval_debug(runtime: RuntimeComponents | None) -> None:
    if runtime is None:
        console.print("[yellow]Runtime not available; retrieval debug is skipped.[/yellow]\n")
        return
    debug_lines = runtime.answer_generator.retrieval_debug_lines()
    if not debug_lines:
        console.print("[yellow]No retrieval context captured yet. Ask a question first.[/yellow]\n")
        return
    config = runtime.retrieval.config
    console.print("[bold]Retrieval Debug[/bold]")
    console.print(
        f"[dim]dense_top_k={config.dense_top_k}, keyword_top_k={config.keyword_top_k}, "
        f"hybrid_top_k={config.hybrid_top_k}, min_score={config.min_score_threshold}[/dim]"
    )
    for line in debug_lines:
        console.print(line)
    console.print()


def _ensure_ollama_ready(runtime: RuntimeComponents, skip_check: bool) -> None:
    llm_settings = runtime.llm_settings
    if llm_settings.get("provider") != "ollama" or skip_check:
        return
    base_url = llm_settings.get("base_url", "http://localhost:11434")
    model_name = llm_settings.get("model", "llama3.1:8b")
    if not check_ollama_connection(base_url):
        console.print(
            Panel(
                "[bold red]Ollama is not running![/bold red]\n\n"
                "The companion requires Ollama to be running to generate answers.\n\n"
                "[bold]To start Ollama:[/bold]\n"
                "  1. Open a new terminal window\n"
                "  2. Run: [cyan]ollama serve[/cyan]\n"
                "  3. Or start Ollama from your applications\n\n"
                "[bold]To verify Ollama is running:[/bold]\n"
                "  Run: [cyan]ollama list[/cyan]\n\n"
                "[bold]For more information:[/bold]\n"
                "  Visit: [link]https://ollama.com/[/link]\n\n"
                "You can skip this check with [cyan]--skip-ollama-check[/cyan], "
                "but the companion will fail when trying to generate answers.",
                title="⚠️  Ollama Connection Error",
                border_style="red",
            )
        )
        raise click.Abort()
    model_installed, available_models = check_ollama_model(base_url, model_name)
    if model_installed:
        return
    available_list = "\n  - ".join(available_models) if available_models else "  (none installed)"
    console.print(
        Panel(
            f"[bold red]Model '{model_name}' is not installed![/bold red]\n\n"
            f"The companion requires the model [cyan]{model_name}[/cyan] to generate answers.\n\n"
            "[bold]To install the model:[/bold]\n"
            f"  Run: [cyan]ollama pull {model_name}[/cyan]\n\n"
            "[bold]Currently installed models:[/bold]\n"
            f"  {available_list}\n\n"
            "[bold]After installing, restart the companion.[/bold]",
            title="⚠️  Model Not Found",
            border_style="yellow",
        )
    )
    raise click.Abort()


@click.group()
def cli() -> None:
    """Gen AI Companion CLI."""


@cli.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=Path("configs/companion_config.yaml"))
@click.option("--session-id", type=str, default=None, help="Resume an existing session ID.")
@click.option("--mode", type=click.Choice([mode.value for mode in ConversationMode]), default="study")
@click.option("--skip-ollama-check", is_flag=True, help="Skip checking if Ollama is running (not recommended)")
def chat(config_path: Path, session_id: str | None, mode: str, skip_ollama_check: bool) -> None:  # noqa: C901
    """Start an interactive chat session."""
    runtime = load_runtime_from_path(config_path)
    _ensure_ollama_ready(runtime, skip_ollama_check)
    conversation_manager = runtime.conversation_manager
    answer_generator = runtime.answer_generator
    query_processor = runtime.query_processor
    ingestion = runtime.ingestion
    ui_settings = getattr(runtime, "ui_settings", {}) or {}
    boxed_default = bool(ui_settings.get("boxed_answers", True))
    auto_reflow_default = bool(ui_settings.get("reflow_on_resize", True))
    box_style = str(ui_settings.get("box_style", "simple"))

    current_mode = ConversationMode(mode)
    session_id = session_id or str(uuid.uuid4())
    conversation_manager.start_session(session_id=session_id, mode=current_mode.value)

    console.print("[bold magenta]IBM Gen AI Companion[/]")
    console.print(f"Type ':help' for available commands. Session ID: [cyan]{session_id}[/]")
    console.print(list_modes())

    pending_attachments: list[AttachmentInput] = []
    mode_ref: dict[str, Any] = {
        "current": current_mode,
        "detail": False,
        "boxed": boxed_default,
        "auto_reflow": auto_reflow_default,
        "target_words": None,
    }
    last_render_payload: dict[str, Any] | None = None
    last_citations: list[dict[str, str | None]] = []

    while True:
        last_render_payload = maybe_reflow_last_answer(
            last_render_payload,
            auto_reflow=bool(mode_ref.get("auto_reflow", True)),
        )
        user_input = Prompt.ask("[bold green]You[/]")
        stripped = user_input.strip()

        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered in {":exit", ":quit"}:
            console.print("[cyan]Ending session.[/]")
            break
        if lowered in {":source", ":sources"}:
            if last_citations:
                console.print("[bold]Sources:[/bold]")
                for idx, citation in enumerate(last_citations, start=1):
                    entry = f"[{idx}] {citation['source']}"
                    snippet = citation.get("snippet")
                    if snippet:
                        entry += f" — {snippet.strip()}"
                    console.print(entry)
            else:
                console.print("[yellow]No sources available for the last answer.[/yellow]")
            continue
        if handle_cli_command(
            stripped,
            ingestion=ingestion,
            pending_attachments=pending_attachments,
            mode_ref=mode_ref,
            runtime=runtime,
            conversation_manager=conversation_manager,
            session_id=session_id,
        ):
            current_mode = mode_ref["current"]
            continue

        try:
            request_metadata = {}
            if mode_ref.get("detail"):
                request_metadata["detail_level"] = "deep"
            if mode_ref.get("target_words"):
                request_metadata["target_words"] = str(mode_ref["target_words"])
            processed = query_processor.process(
                session_id=session_id,
                user_input=user_input,
                attachments=pending_attachments,
                mode=current_mode,
                metadata=request_metadata or None,
            )
            formatted = answer_generator.generate(processed)
            rendered_answer = formatted.render()
            boxed = bool(mode_ref.get("boxed", True))
            render_assistant_message(
                rendered_answer,
                boxed=boxed,
                box_style=box_style,
            )
            last_render_payload = {
                "content": rendered_answer,
                "boxed": boxed,
                "box_style": box_style,
                "width": console.size.width,
            }
            last_citations = [
                {"source": citation.source, "snippet": citation.snippet} for citation in formatted.citations
            ]
            pending_attachments = []
        except GenerationError as exc:
            console.print(Panel(str(exc), title="[bold red]Error[/bold red]", border_style="red"))
            console.print(
                "\n[yellow]Tip:[/yellow] Make sure Ollama is running. Run [cyan]ollama serve[/cyan] in another terminal.\n"
            )


@cli.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=Path("configs/companion_config.yaml"))
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def reset_vector_store(config_path: Path, force: bool) -> None:
    """Reset the ChromaDB vector store, backing up existing data if present.

    This command is useful when the vector store is corrupted or needs to be reinitialized.
    After reset, you will need to re-ingest your documents.
    """

    companion_config = CompanionConfig.from_file(config_path)
    perform_vector_store_reset(companion_config, force=force)


@cli.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=Path("configs/companion_config.yaml"))
@click.option("--iterations", type=int, default=None, help="Number of ACE iterations to run (defaults to config value)")
def trigger_ace(config_path: Path, iterations: int | None) -> None:
    """Manually trigger ACE improvement cycles to refine the playbook.

    This command runs the ACE framework's generator, reflector, and curator loops
    using logged conversation data to improve the playbook.
    """
    companion_config = CompanionConfig.from_file(config_path)
    perform_trigger_ace(companion_config, iterations=iterations)


@cli.command(name="evaluate")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=Path("configs/companion_config.yaml"))
@click.option("--dataset", type=click.Path(path_type=Path), default=Path("data/eval/eval_questions_100.json"))
@click.option("--limit", type=int, default=None, help="Optional limit on number of questions to evaluate")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("outputs/metrics"))
def evaluate_command(
    config_path: Path,
    dataset: Path,
    limit: int | None,
    output_dir: Path,
) -> None:
    """Run the rubric-based evaluation suite and save metrics."""
    runtime = load_runtime_from_path(config_path)
    evaluation_dataset = ensure_default_dataset(dataset)
    if limit is not None:
        evaluation_dataset = EvaluationDataset(evaluation_dataset.examples[:limit])
    engine = EvaluationEngine(EvaluationRubric())

    def answer_provider(example: Any) -> str:
        session_id = f"eval-{example.id}"
        processed = runtime.query_processor.process(
            session_id=session_id,
            user_input=example.question,
            attachments=None,
            mode=ConversationMode.STUDY,
            metadata={"course": example.course, "module": example.module},
        )
        formatted = runtime.answer_generator.generate(processed)
        contexts = runtime.answer_generator.last_retrieval_contexts()
        return {"answer": formatted.answer, "contexts": contexts}

    console.print(f"[bold]Running evaluation on {len(evaluation_dataset.examples)} question(s)...[/bold]")
    results = engine.evaluate(evaluation_dataset, answer_provider)
    report_path = save_metrics_report(engine, results, output_dir, report_name="evaluation")
    console.print(
        f"[green]Evaluation complete.[/green] JSON: [cyan]{report_path}[/cyan], "
        f"Markdown: [cyan]{report_path.with_suffix('.md')}[/cyan]"
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
