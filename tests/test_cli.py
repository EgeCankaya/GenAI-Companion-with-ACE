from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner
from rich.console import Console

from genai_companion_with_ace import cli as cli_module
from genai_companion_with_ace.chat import ConversationMode
from genai_companion_with_ace.cli import cli, handle_cli_command


def test_cli_help_displays_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output


def test_render_assistant_message_boxed(monkeypatch) -> None:
    test_console = Console(record=True, width=120)
    monkeypatch.setattr(cli_module, "console", test_console)
    cli_module.render_assistant_message("Hello", boxed=True, box_style="simple")
    output = test_console.export_text()
    assert "Hello" in output
    assert any(char in output for char in ("╭", "┌", "┏"))


def test_render_assistant_message_unboxed(monkeypatch) -> None:
    test_console = Console(record=True, width=80)
    monkeypatch.setattr(cli_module, "console", test_console)
    cli_module.render_assistant_message("Hello", boxed=False)
    output = test_console.export_text()
    assert "Hello" in output
    assert not any(char in output for char in ("╭", "┌", "┏"))
    assert "Companion" in output


def test_maybe_reflow_last_answer(monkeypatch) -> None:
    test_console = Console(record=True, width=100)
    monkeypatch.setattr(cli_module, "console", test_console)
    payload = {"content": "Hello", "boxed": True, "box_style": "simple", "width": 70}

    cli_module.maybe_reflow_last_answer(payload, auto_reflow=True)

    output = test_console.export_text()
    assert "Hello" in output
    assert payload["width"] == 100


def test_maybe_reflow_respects_flag(monkeypatch) -> None:
    test_console = Console(record=True, width=100)
    monkeypatch.setattr(cli_module, "console", test_console)
    payload = {"content": "Hello", "boxed": True, "box_style": "simple", "width": 70}

    cli_module.maybe_reflow_last_answer(payload, auto_reflow=False)

    assert test_console.export_text().strip() == ""
    assert payload["width"] == 70


def test_long_command_sets_target_words(monkeypatch) -> None:
    mode_ref = {"current": ConversationMode.STUDY, "target_words": None}
    ingestion = MagicMock()
    assert handle_cli_command(
        ":long 900",
        ingestion=ingestion,
        pending_attachments=[],
        mode_ref=mode_ref,
        runtime=None,
    )
    assert mode_ref["target_words"] == 900
