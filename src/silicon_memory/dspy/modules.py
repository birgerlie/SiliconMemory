"""DSPy modules for memory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

# Check if dspy is available
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


def _require_dspy():
    """Raise error if dspy is not installed."""
    if not DSPY_AVAILABLE:
        raise ImportError(
            "DSPy is required for these modules. "
            "Install with: pip install dspy"
        )


# Define signatures (these work even without dspy for documentation)
if DSPY_AVAILABLE:

    class RecallSignature(dspy.Signature):
        """Recall relevant memories for a query."""

        query: str = dspy.InputField(desc="The query to search memories for")
        context: str = dspy.OutputField(desc="Retrieved context from memory")
        confidence: float = dspy.OutputField(desc="Overall confidence in the context")

    class StoreSignature(dspy.Signature):
        """Store a fact in memory."""

        fact: str = dspy.InputField(desc="The fact to store")
        confidence: float = dspy.InputField(desc="Confidence in the fact (0-1)", default=0.7)
        stored: bool = dspy.OutputField(desc="Whether the fact was stored successfully")
        belief_id: str = dspy.OutputField(desc="ID of the stored belief")

    class MemorySignature(dspy.Signature):
        """Answer a question using memory."""

        question: str = dspy.InputField(desc="The question to answer")
        context: str = dspy.InputField(desc="Context from memory", default="")
        answer: str = dspy.OutputField(desc="The answer to the question")
        reasoning: str = dspy.OutputField(desc="Reasoning behind the answer")

    class MemoryRecall(dspy.Module):
        """DSPy module for recalling memories.

        This module retrieves relevant context from Silicon Memory
        based on a query.

        Example:
            >>> recall = MemoryRecall(memory)
            >>> result = recall(query="Python programming")
            >>> print(result.context)
        """

        def __init__(
            self,
            memory: "SiliconMemory",
            max_facts: int = 10,
            max_experiences: int = 5,
            min_confidence: float = 0.3,
        ) -> None:
            """Initialize the recall module.

            Args:
                memory: SiliconMemory instance
                max_facts: Maximum facts to retrieve
                max_experiences: Maximum experiences to retrieve
                min_confidence: Minimum confidence threshold
            """
            super().__init__()
            _require_dspy()

            self._memory = memory
            self._max_facts = max_facts
            self._max_experiences = max_experiences
            self._min_confidence = min_confidence

        def forward(self, query: str) -> dspy.Prediction:
            """Recall relevant memories for a query.

            Args:
                query: The query to search for

            Returns:
                Prediction with context and confidence
            """
            import asyncio

            # Run async recall in sync context
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._async_recall(query))
            return result

        async def _async_recall(self, query: str) -> dspy.Prediction:
            """Async implementation of recall."""
            from silicon_memory.memory.silicondb_router import RecallContext

            ctx = RecallContext(
                query=query,
                max_facts=self._max_facts,
                max_experiences=self._max_experiences,
                min_confidence=self._min_confidence,
            )

            response = await self._memory.recall(ctx)

            # Build context string
            parts = []

            if response.facts:
                parts.append("Facts:")
                for fact in response.facts:
                    parts.append(f"  - {fact.content} ({fact.confidence:.0%})")

            if response.experiences:
                parts.append("Previous interactions:")
                for exp in response.experiences:
                    parts.append(f"  - {exp.content}")

            context = "\n".join(parts) if parts else "No relevant context found."

            # Calculate overall confidence
            if response.facts:
                avg_confidence = sum(f.confidence for f in response.facts) / len(response.facts)
            else:
                avg_confidence = 0.0

            return dspy.Prediction(
                context=context,
                confidence=avg_confidence,
            )

    class MemoryStore(dspy.Module):
        """DSPy module for storing facts in memory.

        This module stores facts as beliefs in Silicon Memory.

        Example:
            >>> store = MemoryStore(memory)
            >>> result = store(fact="Python was created by Guido van Rossum", confidence=0.95)
            >>> print(f"Stored: {result.stored}")
        """

        def __init__(self, memory: "SiliconMemory") -> None:
            """Initialize the store module.

            Args:
                memory: SiliconMemory instance
            """
            super().__init__()
            _require_dspy()

            self._memory = memory

        def forward(self, fact: str, confidence: float = 0.7) -> dspy.Prediction:
            """Store a fact in memory.

            Args:
                fact: The fact to store
                confidence: Confidence level (0-1)

            Returns:
                Prediction with stored status and belief_id
            """
            import asyncio

            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._async_store(fact, confidence))
            return result

        async def _async_store(self, fact: str, confidence: float) -> dspy.Prediction:
            """Async implementation of store."""
            from silicon_memory.core.types import Belief, Source, SourceType

            belief = Belief(
                id=uuid4(),
                content=fact,
                confidence=min(1.0, max(0.0, confidence)),
                source=Source(
                    id="dspy_module",
                    type=SourceType.EXTERNAL,
                    reliability=0.7,
                ),
            )

            try:
                await self._memory.commit_belief(belief)
                return dspy.Prediction(
                    stored=True,
                    belief_id=str(belief.id),
                )
            except Exception as e:
                return dspy.Prediction(
                    stored=False,
                    belief_id="",
                    error=str(e),
                )

    class MemoryAugmentedChain(dspy.Module):
        """DSPy module for memory-augmented question answering.

        Automatically retrieves relevant context from memory
        before answering questions.

        Example:
            >>> chain = MemoryAugmentedChain(memory)
            >>> result = chain(question="What is Python?")
            >>> print(result.answer)
        """

        def __init__(
            self,
            memory: "SiliconMemory",
            max_facts: int = 10,
            max_experiences: int = 5,
            min_confidence: float = 0.3,
            store_interactions: bool = True,
        ) -> None:
            """Initialize the chain.

            Args:
                memory: SiliconMemory instance
                max_facts: Maximum facts to retrieve
                max_experiences: Maximum experiences to retrieve
                min_confidence: Minimum confidence threshold
                store_interactions: Whether to store interactions as experiences
            """
            super().__init__()
            _require_dspy()

            self._memory = memory
            self._store_interactions = store_interactions

            # Sub-modules
            self._recall = MemoryRecall(
                memory,
                max_facts=max_facts,
                max_experiences=max_experiences,
                min_confidence=min_confidence,
            )
            self._store = MemoryStore(memory)

            # Chain of thought module for answering
            self._answer = dspy.ChainOfThought(MemorySignature)

        def forward(self, question: str) -> dspy.Prediction:
            """Answer a question using memory.

            Args:
                question: The question to answer

            Returns:
                Prediction with answer and reasoning
            """
            import asyncio

            # Recall relevant context
            recall_result = self._recall(query=question)

            # Answer with context
            answer_result = self._answer(
                question=question,
                context=recall_result.context,
            )

            # Store interaction if enabled
            if self._store_interactions:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(
                    self._store_interaction(question, answer_result.answer)
                )

            return dspy.Prediction(
                answer=answer_result.answer,
                reasoning=answer_result.reasoning,
                context=recall_result.context,
                context_confidence=recall_result.confidence,
            )

        async def _store_interaction(self, question: str, answer: str) -> None:
            """Store the interaction as an experience."""
            from silicon_memory.core.types import Experience

            experience = Experience(
                id=uuid4(),
                content=f"Question: {question[:500]}",
                outcome=f"Answer: {answer[:500]}",
            )

            await self._memory.record_experience(experience)

else:
    # Provide stub classes when dspy is not available
    class RecallSignature:
        """Stub for RecallSignature when dspy is not installed."""
        def __init__(self):
            _require_dspy()

    class StoreSignature:
        """Stub for StoreSignature when dspy is not installed."""
        def __init__(self):
            _require_dspy()

    class MemorySignature:
        """Stub for MemorySignature when dspy is not installed."""
        def __init__(self):
            _require_dspy()

    class MemoryRecall:
        """Stub for MemoryRecall when dspy is not installed."""
        def __init__(self, *args, **kwargs):
            _require_dspy()

    class MemoryStore:
        """Stub for MemoryStore when dspy is not installed."""
        def __init__(self, *args, **kwargs):
            _require_dspy()

    class MemoryAugmentedChain:
        """Stub for MemoryAugmentedChain when dspy is not installed."""
        def __init__(self, *args, **kwargs):
            _require_dspy()
