"""Reflection Engine - Extract beliefs from experiences.

The reflection engine processes episodic memories (experiences) and extracts
semantic knowledge (beliefs). This mimics human memory consolidation.

Example:
    >>> from silicon_memory import SiliconMemory
    >>> from silicon_memory.reflection import ReflectionEngine
    >>>
    >>> with SiliconMemory("/path/to/db") as memory:
    ...     engine = ReflectionEngine(memory)
    ...     result = await engine.reflect()
    ...     print(f"Extracted {len(result.new_beliefs)} new beliefs")
"""

from silicon_memory.reflection.types import (
    Pattern,
    PatternType,
    BeliefCandidate,
    ReflectionResult,
    ReflectionConfig,
)
from silicon_memory.reflection.processor import ExperienceProcessor
from silicon_memory.reflection.patterns import PatternExtractor
from silicon_memory.reflection.generator import BeliefGenerator
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.reflection.tool import ReflectionTool

__all__ = [
    # Types
    "Pattern",
    "PatternType",
    "BeliefCandidate",
    "ReflectionResult",
    "ReflectionConfig",
    # Components
    "ExperienceProcessor",
    "PatternExtractor",
    "BeliefGenerator",
    "ReflectionEngine",
    # Tool
    "ReflectionTool",
]
