"""DSPy Integration - Declarative LLM programming with memory.

Provides DSPy modules for memory operations that can be composed
into declarative LLM programs.

Example:
    >>> import dspy
    >>> from silicon_memory import SiliconMemory
    >>> from silicon_memory.dspy import MemoryRecall, MemoryStore, MemoryAugmentedChain
    >>>
    >>> with SiliconMemory("/path/to/db") as memory:
    ...     # Use individual modules
    ...     recall = MemoryRecall(memory)
    ...     context = recall(query="Python programming")
    ...
    ...     # Or use the augmented chain
    ...     chain = MemoryAugmentedChain(memory)
    ...     response = chain(question="What is Python?")
"""

from silicon_memory.dspy.modules import (
    MemoryRecall,
    MemoryStore,
    MemoryAugmentedChain,
    MemorySignature,
    RecallSignature,
    StoreSignature,
)

__all__ = [
    "MemoryRecall",
    "MemoryStore",
    "MemoryAugmentedChain",
    "MemorySignature",
    "RecallSignature",
    "StoreSignature",
]
