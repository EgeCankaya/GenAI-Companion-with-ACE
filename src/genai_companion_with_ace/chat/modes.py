"""Conversation mode definitions for the AI companion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConversationMode(str, Enum):
    STUDY = "study"
    QUIZ = "quiz"
    QUICK = "quick"


@dataclass(slots=True, frozen=True)
class ModeSettings:
    """Configuration options that tailor responses for a mode."""

    name: str
    include_objectives: bool
    hint_only: bool
    concise: bool


MODE_REGISTRY: dict[ConversationMode, ModeSettings] = {
    ConversationMode.STUDY: ModeSettings(
        name="Study Mode",
        include_objectives=False,
        hint_only=False,
        concise=False,
    ),
    ConversationMode.QUIZ: ModeSettings(
        name="Quiz Mode",
        include_objectives=False,
        hint_only=True,
        concise=True,
    ),
    ConversationMode.QUICK: ModeSettings(
        name="Quick Reference",
        include_objectives=False,
        hint_only=False,
        concise=True,
    ),
}


def get_mode_settings(mode: ConversationMode) -> ModeSettings:
    """Return the settings for the requested conversation mode."""
    return MODE_REGISTRY[mode]
