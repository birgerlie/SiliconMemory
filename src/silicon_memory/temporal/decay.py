"""Temporal decay functions for confidence degradation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class DecayFunction(Enum):
    """Available decay function types."""

    NONE = "none"  # No decay
    LINEAR = "linear"  # Linear decay to floor
    EXPONENTIAL = "exponential"  # Exponential decay (half-life based)
    STEP = "step"  # Step function at half-life


@dataclass(frozen=True)
class DecayConfig:
    """Configuration for temporal decay."""

    function: DecayFunction = DecayFunction.EXPONENTIAL
    half_life_seconds: int = 86400 * 7  # 1 week default
    floor: float = 0.1  # Minimum decay factor
    ceiling: float = 1.0  # Maximum decay factor

    def __post_init__(self) -> None:
        if self.floor < 0 or self.floor > 1:
            raise ValueError("Floor must be between 0 and 1")
        if self.ceiling < 0 or self.ceiling > 1:
            raise ValueError("Ceiling must be between 0 and 1")
        if self.floor > self.ceiling:
            raise ValueError("Floor cannot exceed ceiling")
        if self.half_life_seconds <= 0:
            raise ValueError("Half-life must be positive")


def compute_decay(age_seconds: float, config: DecayConfig) -> float:
    """Compute decay factor (0.0 - 1.0) based on age.

    Args:
        age_seconds: Age of the item in seconds
        config: Decay configuration

    Returns:
        Decay factor between floor and ceiling
    """
    if age_seconds < 0:
        raise ValueError("Age cannot be negative")

    if config.function == DecayFunction.NONE:
        return config.ceiling

    if config.function == DecayFunction.LINEAR:
        # Linear decay over 2x half-life
        max_age = config.half_life_seconds * 2
        factor = max(0.0, 1.0 - (age_seconds / max_age))

    elif config.function == DecayFunction.EXPONENTIAL:
        # Exponential decay: f(t) = 0.5 ^ (t / half_life)
        exponent = age_seconds / config.half_life_seconds
        factor = math.pow(0.5, exponent)

    elif config.function == DecayFunction.STEP:
        # Step function: 1.0 before half-life, 0.5 after
        factor = 1.0 if age_seconds < config.half_life_seconds else 0.5

    else:
        factor = 1.0

    # Clamp to [floor, ceiling]
    return max(config.floor, min(config.ceiling, factor))


def apply_decay(
    base_value: float,
    age_seconds: float,
    config: DecayConfig,
) -> float:
    """Apply decay to a base value.

    Args:
        base_value: Original value (e.g., confidence)
        age_seconds: Age in seconds
        config: Decay configuration

    Returns:
        Decayed value
    """
    decay_factor = compute_decay(age_seconds, config)
    return base_value * decay_factor


def compute_effective_confidence(
    base_confidence: float,
    age_seconds: float,
    verification_boost: float = 0.0,
    config: DecayConfig | None = None,
) -> float:
    """Compute effective confidence with decay and verification boost.

    Args:
        base_confidence: Original confidence
        age_seconds: Time since last verification
        verification_boost: Additional confidence from recent verification
        config: Decay configuration

    Returns:
        Effective confidence (clamped to 0.0-1.0)
    """
    if config is None:
        config = DecayConfig()

    decayed = apply_decay(base_confidence, age_seconds, config)
    boosted = decayed + verification_boost

    return max(0.0, min(1.0, boosted))
