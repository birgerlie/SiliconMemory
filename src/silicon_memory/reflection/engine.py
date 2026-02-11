"""Main reflection engine orchestrating the reflection process."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from silicon_memory.core.types import Belief
from silicon_memory.core.decision import DecisionStatus
from silicon_memory.reflection.types import (
    BeliefCandidate,
    Pattern,
    ReflectionConfig,
    ReflectionResult,
)
from silicon_memory.reflection.processor import ExperienceProcessor
from silicon_memory.reflection.patterns import PatternExtractor
from silicon_memory.reflection.generator import BeliefGenerator

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class ReflectionEngine:
    """Main reflection engine that orchestrates memory consolidation.

    The reflection engine processes episodic memories (experiences) and
    extracts semantic knowledge (beliefs), mimicking human memory
    consolidation during sleep.

    The process:
    1. Fetch unprocessed experiences
    2. Group related experiences
    3. Extract patterns from groups
    4. Generate belief candidates
    5. Validate against existing knowledge
    6. Commit valid beliefs (if auto_commit enabled)
    7. Mark experiences as processed

    Example:
        >>> from silicon_memory import SiliconMemory
        >>> from silicon_memory.reflection import ReflectionEngine
        >>>
        >>> with SiliconMemory("/path/to/db") as memory:
        ...     engine = ReflectionEngine(memory)
        ...
        ...     # Run a reflection cycle
        ...     result = await engine.reflect()
        ...     print(result.summary())
        ...
        ...     # Commit approved beliefs
        ...     for candidate in result.new_beliefs:
        ...         if user_approves(candidate):
        ...             await engine.commit_belief(candidate)
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()

        # Initialize components
        self._processor = ExperienceProcessor(memory, config)
        self._extractor = PatternExtractor(memory, config)
        self._generator = BeliefGenerator(memory, config)

    @property
    def config(self) -> ReflectionConfig:
        """Get the current configuration."""
        return self._config

    async def reflect(
        self,
        max_experiences: int | None = None,
        auto_commit: bool | None = None,
    ) -> ReflectionResult:
        """Run a reflection cycle.

        Args:
            max_experiences: Override max experiences to process
            auto_commit: Override auto-commit setting

        Returns:
            ReflectionResult with new beliefs and patterns
        """
        result = ReflectionResult()

        # Step 1: Fetch unprocessed experiences
        limit = max_experiences or self._config.max_experiences_per_batch
        experiences = await self._processor.fetch_unprocessed(limit)

        if not experiences:
            return result

        result.experiences_processed = len(experiences)

        # Step 2: Group related experiences
        groups = await self._processor.process_batch(experiences)

        if not groups:
            # Mark as processed even if no groups formed
            await self._processor.mark_processed([e.id for e in experiences])
            return result

        # Step 3: Extract patterns
        patterns = await self._extractor.extract_patterns(groups)
        result.patterns_found = patterns

        if not patterns:
            await self._processor.mark_processed([e.id for e in experiences])
            return result

        # Step 4: Generate belief candidates
        candidates = await self._generator.generate_beliefs(patterns)
        result.new_beliefs = candidates

        # Step 5: Track contradictions
        for candidate in candidates:
            for contra_id in candidate.contradicts:
                result.contradictions.append((candidate.id, contra_id))

        # Step 5b: Review active decisions for assumption drift
        await self._review_active_decisions()

        # Step 6: Auto-commit if enabled
        should_commit = auto_commit if auto_commit is not None else self._config.auto_commit_beliefs
        if should_commit:
            for candidate in candidates:
                if not candidate.is_contested:
                    belief = await self._generator.commit_belief(candidate)
                    if belief:
                        result.updated_beliefs.append((belief.id, belief.confidence))

        # Step 7: Mark experiences as processed
        await self._processor.mark_processed([e.id for e in experiences])

        return result

    async def _review_active_decisions(self) -> None:
        """Review active decisions for assumption confidence drift.

        For each active decision, checks if any critical assumption's
        current confidence has changed significantly (>0.2) from the
        confidence at decision time. If so, flags the decision as
        REVISIT_SUGGESTED.
        """
        try:
            decisions = await self._memory.recall_decisions(
                query="*", k=50, min_confidence=0.0
            )
        except Exception:
            return

        for decision in decisions:
            if decision.status != DecisionStatus.ACTIVE:
                continue

            for assumption in decision.assumptions:
                if not assumption.is_critical:
                    continue

                # Check current belief confidence
                try:
                    belief = await self._memory.get_belief(assumption.belief_id)
                except Exception:
                    continue

                if belief is None:
                    continue

                drift = abs(belief.confidence - assumption.confidence_at_decision)
                if drift > 0.2:
                    decision.status = DecisionStatus.REVISIT_SUGGESTED
                    try:
                        await self._memory._backend.record_decision_outcome(
                            decision.id,
                            f"Auto-flagged: assumption '{assumption.description}' "
                            f"drifted {drift:.2f} from {assumption.confidence_at_decision:.2f} "
                            f"to {belief.confidence:.2f}",
                        )
                    except Exception:
                        pass
                    break  # One flagged assumption is enough

    async def reflect_incremental(
        self,
        batch_size: int = 20,
    ) -> ReflectionResult:
        """Run reflection in smaller incremental batches.

        Useful for processing large backlogs without blocking.

        Args:
            batch_size: Number of experiences per batch

        Returns:
            Combined ReflectionResult from all batches
        """
        combined = ReflectionResult()

        while True:
            result = await self.reflect(max_experiences=batch_size)

            if result.experiences_processed == 0:
                break

            # Combine results
            combined.experiences_processed += result.experiences_processed
            combined.patterns_found.extend(result.patterns_found)
            combined.new_beliefs.extend(result.new_beliefs)
            combined.updated_beliefs.extend(result.updated_beliefs)
            combined.contradictions.extend(result.contradictions)

        return combined

    async def commit_belief(
        self,
        candidate: BeliefCandidate,
    ) -> Belief | None:
        """Commit a specific belief candidate.

        Args:
            candidate: The candidate to commit

        Returns:
            The committed Belief, or None if failed
        """
        return await self._generator.commit_belief(candidate, require_approval=False)

    async def commit_all_valid(
        self,
        result: ReflectionResult,
    ) -> list[Belief]:
        """Commit all non-contested beliefs from a reflection result.

        Args:
            result: The reflection result containing candidates

        Returns:
            List of committed beliefs
        """
        return await self._generator.commit_all_valid(result.new_beliefs)

    async def get_pending_candidates(
        self,
        min_confidence: float = 0.5,
    ) -> list[BeliefCandidate]:
        """Get belief candidates waiting for approval.

        This runs a reflection cycle without committing, returning
        candidates that meet the confidence threshold.

        Args:
            min_confidence: Minimum confidence for candidates

        Returns:
            List of belief candidates for review
        """
        # Temporarily disable auto-commit
        original = self._config.auto_commit_beliefs
        self._config.auto_commit_beliefs = False

        try:
            result = await self.reflect()
            return [c for c in result.new_beliefs if c.confidence >= min_confidence]
        finally:
            self._config.auto_commit_beliefs = original

    async def analyze_patterns(
        self,
        max_experiences: int = 100,
    ) -> list[Pattern]:
        """Analyze experiences for patterns without generating beliefs.

        Useful for understanding what patterns exist before committing.

        Args:
            max_experiences: Maximum experiences to analyze

        Returns:
            List of patterns found
        """
        experiences = await self._processor.fetch_unprocessed(max_experiences)
        if not experiences:
            return []

        groups = await self._processor.process_batch(experiences)
        if not groups:
            return []

        return await self._extractor.extract_patterns(groups)

    def update_config(self, **kwargs) -> None:
        """Update configuration settings.

        Args:
            **kwargs: Configuration fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        # Propagate to components
        self._processor._config = self._config
        self._extractor._config = self._config
        self._generator._config = self._config
