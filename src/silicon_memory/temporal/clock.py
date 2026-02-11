"""Time abstraction for testability."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from silicon_memory.core.utils import utc_now


class Clock(ABC):
    """Abstract clock interface."""

    @abstractmethod
    def now(self) -> datetime:
        """Get current time."""
        ...


class SystemClock(Clock):
    """Real system time."""

    def now(self) -> datetime:
        return utc_now()


class FakeClock(Clock):
    """Controllable clock for testing."""

    def __init__(self, initial: datetime | None = None) -> None:
        self._now = initial or utc_now()

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: int | float = 0, **kwargs: int) -> None:
        """Advance time by specified duration.

        Args:
            seconds: Number of seconds to advance
            **kwargs: Additional timedelta arguments (minutes, hours, days, etc.)
        """
        self._now = self._now + timedelta(seconds=seconds, **kwargs)

    def set(self, time: datetime) -> None:
        """Set clock to specific time."""
        self._now = time

    def advance_to(self, time: datetime) -> None:
        """Advance clock to specific time (must be in future)."""
        if time < self._now:
            raise ValueError("Cannot advance to a time in the past")
        self._now = time
