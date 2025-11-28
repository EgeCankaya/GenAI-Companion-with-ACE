"""Time-related helper utilities."""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow_isoformat() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()

