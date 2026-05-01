"""Time parsing helpers for the vNext extraction pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Module-level default timezone name; overridden by config at startup.
_DEFAULT_TZ_NAME: str = "UTC"


def set_default_timezone(tz_name: str) -> None:
    """Set the module-level default timezone used by render_time_text."""
    global _DEFAULT_TZ_NAME
    _DEFAULT_TZ_NAME = tz_name


def _resolve_tz(tz_name: str | None) -> datetime.tzinfo:
    name = tz_name or _DEFAULT_TZ_NAME
    if name.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError):
        return timezone.utc


def parse_timestamp_to_unix(value: Any) -> float:
    """Best-effort conversion of many timestamp formats into unix seconds."""

    if isinstance(value, (int, float)):
        return float(value)

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("timestamp is empty")

    if raw.isdigit():
        return float(int(raw))

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    for candidate in (
        raw,
        raw.replace("/", "-"),
    ):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_resolve_tz(None))
            return dt.timestamp()
        except ValueError:
            continue

    for fmt in (
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d (%A) %H:%M",
        "%Y-%m-%d (%a) %H:%M",
        "%Y-%m-%d (%A) %H:%M",
        "%B %d, %Y(%A) at %I:%M %p",
        "%B %d, %Y (%A) at %I:%M %p UTC",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=_resolve_tz(None)).timestamp()
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp format: {value!r}")


def render_time_text(timestamp_unix: float, tz_name: str | None = None) -> str:
    """Render unix seconds to human-readable text including day of week.

    Uses *tz_name* if provided, otherwise falls back to the module-level
    default set via :func:`set_default_timezone` (defaults to UTC).
    """
    tz = _resolve_tz(tz_name)
    dt = datetime.fromtimestamp(float(timestamp_unix), tz=tz)
    tz_label = _render_utc_offset_label(dt)
    return dt.strftime(f"%B %d, %Y (%A) at %I:%M %p {tz_label}")


def _render_utc_offset_label(dt: datetime) -> str:
    offset = dt.utcoffset()
    if offset is None:
        return "UTC"
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"UTC{sign}{hours:02d}:{minutes:02d}"
