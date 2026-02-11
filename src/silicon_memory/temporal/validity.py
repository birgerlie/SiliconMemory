"""Temporal validity checking for beliefs and knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from silicon_memory.temporal.clock import Clock, SystemClock
from silicon_memory.temporal.decay import DecayConfig, apply_decay

if TYPE_CHECKING:
    from silicon_memory.core.types import Belief, TemporalContext


@dataclass
class ValidityResult:
    """Result of validity check."""

    is_valid: bool
    is_expired: bool
    is_stale: bool
    effective_confidence: float
    reason: str


class TemporalValidator:
    """Validates temporal aspects of beliefs."""

    def __init__(
        self,
        clock: Clock | None = None,
        decay_config: DecayConfig | None = None,
        staleness_threshold_seconds: int = 86400 * 30,  # 30 days
    ) -> None:
        self._clock = clock or SystemClock()
        self._decay = decay_config or DecayConfig()
        self._staleness_threshold = staleness_threshold_seconds

    def check_validity(
        self,
        belief: Belief,
        as_of: datetime | None = None,
    ) -> ValidityResult:
        """Check if a belief is valid at a given time.

        Args:
            belief: The belief to check
            as_of: Time to check validity at (default: now)

        Returns:
            ValidityResult with details
        """
        check_time = as_of or self._clock.now()

        # Check temporal bounds
        if belief.temporal:
            if not belief.temporal.is_valid_at(check_time):
                return ValidityResult(
                    is_valid=False,
                    is_expired=False,
                    is_stale=False,
                    effective_confidence=0.0,
                    reason="Outside temporal validity bounds",
                )

            if belief.temporal.is_expired(check_time):
                return ValidityResult(
                    is_valid=False,
                    is_expired=True,
                    is_stale=False,
                    effective_confidence=0.0,
                    reason="TTL expired",
                )

        # Compute age and staleness
        reference_time = self._get_reference_time(belief)
        age_seconds = (check_time - reference_time).total_seconds()
        is_stale = age_seconds > self._staleness_threshold

        # Apply decay to confidence
        effective_confidence = apply_decay(
            belief.confidence,
            age_seconds,
            self._decay,
        )

        return ValidityResult(
            is_valid=True,
            is_expired=False,
            is_stale=is_stale,
            effective_confidence=effective_confidence,
            reason="Valid" if not is_stale else "Valid but stale",
        )

    def is_valid(
        self,
        belief: Belief,
        as_of: datetime | None = None,
    ) -> bool:
        """Quick check if belief is valid."""
        return self.check_validity(belief, as_of).is_valid

    def get_effective_confidence(
        self,
        belief: Belief,
        as_of: datetime | None = None,
    ) -> float:
        """Get effective confidence after decay."""
        return self.check_validity(belief, as_of).effective_confidence

    def _get_reference_time(self, belief: Belief) -> datetime:
        """Get the reference time for age calculation."""
        if belief.temporal:
            if belief.temporal.last_verified:
                return belief.temporal.last_verified
            return belief.temporal.observed_at
        # Fallback to now (assumes just created)
        return self._clock.now()

    def needs_verification(
        self,
        belief: Belief,
        threshold_seconds: int | None = None,
    ) -> bool:
        """Check if a belief needs re-verification.

        Args:
            belief: The belief to check
            threshold_seconds: Custom threshold (default: half of staleness threshold)

        Returns:
            True if belief should be verified
        """
        if threshold_seconds is None:
            threshold_seconds = self._staleness_threshold // 2

        reference_time = self._get_reference_time(belief)
        age_seconds = (self._clock.now() - reference_time).total_seconds()

        return age_seconds > threshold_seconds
