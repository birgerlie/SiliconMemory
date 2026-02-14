"""Entity resolution package — self-learning entity detection and normalization."""

from silicon_memory.entities.cache import EntityCache
from silicon_memory.entities.learner import RuleLearner
from silicon_memory.entities.resolver import EntityResolver
from silicon_memory.entities.rules import RuleEngine
from silicon_memory.entities.types import (
    Candidate,
    DetectorRule,
    EntityEntry,
    EntityReference,
    ExtractorRule,
    ResolveResult,
)

__all__ = [
    "Candidate",
    "DetectorRule",
    "EntityCache",
    "EntityEntry",
    "EntityReference",
    "EntityResolver",
    "ExtractorRule",
    "ResolveResult",
    "RuleEngine",
    "RuleLearner",
]
