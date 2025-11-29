from __future__ import annotations

from pathlib import Path

import yaml

from genai_companion_with_ace.ace_integration.playbook_loader import PlaybookLoader


def write_playbook(path: Path, version: str, instructions: str) -> None:
    payload = {
        "version": version,
        "context": {
            "system_instructions": instructions,
            "heuristics": ["Always cite sources."],
            "examples": [],
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_loader_prefers_local_playbooks(tmp_path: Path) -> None:
    local_dir = tmp_path / "local"
    local_dir.mkdir()

    write_playbook(local_dir / "playbook_v1.yaml", "1.0.0", "Older instructions")
    write_playbook(local_dir / "playbook_v2.yaml", "2.0.0", "Local instructions")

    loader = PlaybookLoader(playbook_dir=local_dir)
    context = loader.load_latest()

    assert context.version == "2.0.0"
    assert "Local instructions" in context.system_instructions


def test_loader_uses_default_when_empty(tmp_path: Path) -> None:
    local_dir = tmp_path / "local"
    local_dir.mkdir()

    loader = PlaybookLoader(playbook_dir=local_dir)
    context = loader.load_latest()

    # Should use default playbook tailored for GenAI Companion
    assert context.version == "default"
    assert "IBM Gen AI Study Companion" in context.system_instructions
    assert len(context.heuristics) > 0


def test_loader_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    playbook_dir = tmp_path / "local_playbooks"
    playbook_dir.mkdir(parents=True)

    write_playbook(playbook_dir / "playbook_v4.yaml", "4.0.0", "Local instructions")

    yaml.safe_dump(
        {
            "ace": {
                "repository_path": str(tmp_path / "ace_repo"),  # Not used for playbook loading anymore
                "playbook_output_dir": str(playbook_dir),
            }
        },
        config_path.open("w", encoding="utf-8"),
    )

    loader = PlaybookLoader.from_config(config_path)
    context = loader.load_latest()

    assert context.version == "4.0.0"
    assert "Local instructions" in context.system_instructions
