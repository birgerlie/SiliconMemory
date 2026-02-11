"""Core utility functions for Silicon Memory."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    This function replaces deprecated datetime.utcnow() with the recommended
    timezone-aware approach using datetime.now(timezone.utc).

    Returns:
        datetime: Current UTC time with timezone info.
    """
    return datetime.now(timezone.utc)
