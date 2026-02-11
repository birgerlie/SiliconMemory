"""Belief generator for the reflection engine."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.types import Belief, BeliefStatus, Source, SourceType, Triplet
from silicon_memory.reflection.types import (
    BeliefCandidate,
    Pattern,
    PatternType,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class BeliefGenerator:
    """Generates belief candidates from patterns.

    The generator:
    1. Converts patterns into belief candidates
    2. Validates against existing knowledge
    3. Detects contradictions
    4. Assigns confidence based on evidence

    Example:
        >>> generator = BeliefGenerator(memory)
        >>> candidates = await generator.generate_beliefs(patterns)
        >>> for c in candidates:
        ...     if not c.is_contested:
        ...         await generator.commit_belief(c)
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()

    async def generate_beliefs(
        self,
        patterns: list[Pattern],
        existing_beliefs: list[Belief] | None = None,
    ) -> list[BeliefCandidate]:
        """Generate belief candidates from patterns.

        Args:
            patterns: Patterns to convert to beliefs
            existing_beliefs: Optional list of existing beliefs for validation

        Returns:
            List of belief candidates
        """
        candidates: list[BeliefCandidate] = []

        for pattern in patterns:
            # Skip weak patterns
            if pattern.confidence < self._config.min_confidence_threshold:
                continue

            # Check if pattern has enough evidence
            if (self._config.require_multiple_sources and
                len(pattern.evidence) < 2):
                continue

            # Generate candidate from pattern
            candidate = self._pattern_to_candidate(pattern)
            if candidate:
                candidates.append(candidate)

        # Validate candidates against existing knowledge
        candidates = await self._validate_candidates(candidates, existing_beliefs)

        # Limit number of candidates
        candidates = sorted(
            candidates,
            key=lambda c: c.confidence,
            reverse=True,
        )[:self._config.max_beliefs_per_cycle]

        return candidates

    def _pattern_to_candidate(self, pattern: Pattern) -> BeliefCandidate | None:
        """Convert a pattern to a belief candidate."""
        # Generate content based on pattern type
        if pattern.type == PatternType.CAUSAL:
            content = f"{pattern.subject} causes {pattern.object}"
            predicate = "causes"
        elif pattern.type == PatternType.TEMPORAL:
            content = pattern.description
            predicate = "is followed by"
        elif pattern.type == PatternType.CORRELATION:
            content = f"{pattern.subject} is associated with {pattern.object}"
            predicate = "is associated with"
        elif pattern.type == PatternType.GENERALIZATION:
            content = pattern.description
            predicate = pattern.predicate or "relates to"
        elif pattern.type == PatternType.PREFERENCE:
            content = f"Preference: {pattern.subject} over {pattern.object}"
            predicate = "is preferred over"
        elif pattern.type == PatternType.FACT:
            content = pattern.description
            predicate = pattern.predicate or "is"
        else:
            content = pattern.description
            predicate = "relates to"

        # Calculate confidence
        base_confidence = pattern.confidence
        # Boost for more evidence
        evidence_boost = min(0.2, 0.02 * len(pattern.evidence))
        # Boost for more occurrences
        occurrence_boost = min(0.2, 0.02 * pattern.occurrences)
        confidence = min(0.95, base_confidence + evidence_boost + occurrence_boost)

        return BeliefCandidate(
            id=uuid4(),
            content=content,
            subject=pattern.subject,
            predicate=predicate,
            object=pattern.object,
            confidence=confidence,
            source_patterns=[pattern.id],
            source_experiences=pattern.evidence,
            reasoning=f"Extracted from {pattern.type.value} pattern with {len(pattern.evidence)} evidence items",
        )

    async def _validate_candidates(
        self,
        candidates: list[BeliefCandidate],
        existing_beliefs: list[Belief] | None = None,
    ) -> list[BeliefCandidate]:
        """Validate candidates against existing knowledge."""
        if existing_beliefs is None:
            # Fetch existing beliefs for validation
            existing_beliefs = []
            for candidate in candidates:
                if candidate.subject:
                    beliefs = await self._memory._backend.get_beliefs_by_entity(
                        candidate.subject
                    )
                    existing_beliefs.extend(beliefs)

        # Create a lookup for existing beliefs
        existing_by_subject: dict[str, list[Belief]] = {}
        for belief in existing_beliefs:
            if belief.triplet:
                key = belief.triplet.subject.lower()
                if key not in existing_by_subject:
                    existing_by_subject[key] = []
                existing_by_subject[key].append(belief)

        # Validate each candidate
        for candidate in candidates:
            if candidate.subject:
                key = candidate.subject.lower()
                related = existing_by_subject.get(key, [])

                for belief in related:
                    # Check for support
                    if self._beliefs_support(candidate, belief):
                        candidate.supports.append(belief.id)
                        candidate.is_novel = False

                    # Check for contradiction
                    if self._beliefs_contradict(candidate, belief):
                        candidate.contradicts.append(belief.id)

        return candidates

    def _beliefs_support(
        self,
        candidate: BeliefCandidate,
        existing: Belief,
    ) -> bool:
        """Check if an existing belief supports the candidate."""
        if not existing.triplet:
            return False

        # Same subject and predicate with similar object = support
        if (candidate.subject and candidate.predicate and
            existing.triplet.subject.lower() == candidate.subject.lower() and
            existing.triplet.predicate.lower() == candidate.predicate.lower()):
            # Check object similarity
            if candidate.object:
                return (
                    existing.triplet.object.lower() == candidate.object.lower() or
                    candidate.object.lower() in existing.triplet.object.lower() or
                    existing.triplet.object.lower() in candidate.object.lower()
                )
        return False

    def _beliefs_contradict(
        self,
        candidate: BeliefCandidate,
        existing: Belief,
    ) -> bool:
        """Check if an existing belief contradicts the candidate."""
        if not existing.triplet:
            return False

        # Same subject and predicate but different object = potential contradiction
        if (candidate.subject and candidate.predicate and candidate.object and
            existing.triplet.subject.lower() == candidate.subject.lower() and
            existing.triplet.predicate.lower() == candidate.predicate.lower()):
            # Different object values that are mutually exclusive
            if existing.triplet.object.lower() != candidate.object.lower():
                # Check for mutual exclusivity indicators
                if self._are_mutually_exclusive(candidate.object, existing.triplet.object):
                    return True

        return False

    def _are_mutually_exclusive(self, obj1: str, obj2: str) -> bool:
        """Check if two object values are mutually exclusive."""
        o1 = obj1.lower()
        o2 = obj2.lower()

        # Obvious contradictions
        if o1 == "true" and o2 == "false":
            return True
        if o1 == "false" and o2 == "true":
            return True
        if o1 == "yes" and o2 == "no":
            return True
        if o1 == "no" and o2 == "yes":
            return True

        # Negation check
        if o1.startswith("not ") and o1[4:] == o2:
            return True
        if o2.startswith("not ") and o2[4:] == o1:
            return True

        # Different numeric values for same property
        try:
            n1 = float(o1)
            n2 = float(o2)
            return n1 != n2
        except ValueError:
            pass

        return False

    async def commit_belief(
        self,
        candidate: BeliefCandidate,
        require_approval: bool | None = None,
    ) -> Belief | None:
        """Commit a belief candidate to memory.

        Args:
            candidate: The candidate to commit
            require_approval: Override config's auto_commit_beliefs

        Returns:
            The committed Belief, or None if not committed
        """
        should_auto = not (require_approval if require_approval is not None
                          else not self._config.auto_commit_beliefs)

        if not should_auto and candidate.is_contested:
            # Don't auto-commit contested beliefs
            return None

        # Create the belief
        triplet = None
        if candidate.has_triplet:
            triplet = Triplet(
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
            )

        belief = Belief(
            id=candidate.id,
            content=candidate.content if not triplet else "",
            triplet=triplet,
            confidence=candidate.confidence,
            source=Source(
                id="reflection_engine",
                type=SourceType.REFLECTION,
                reliability=0.7,
                metadata={
                    "patterns": [str(p) for p in candidate.source_patterns],
                    "experiences": [str(e) for e in candidate.source_experiences],
                    "reasoning": candidate.reasoning,
                },
            ),
            status=BeliefStatus.PROVISIONAL,
            evidence_for=candidate.supports,
            evidence_against=candidate.contradicts,
        )

        await self._memory.commit_belief(belief)
        return belief

    async def commit_all_valid(
        self,
        candidates: list[BeliefCandidate],
    ) -> list[Belief]:
        """Commit all non-contested candidates.

        Args:
            candidates: List of candidates to commit

        Returns:
            List of committed beliefs
        """
        committed = []
        for candidate in candidates:
            if not candidate.is_contested:
                belief = await self.commit_belief(candidate, require_approval=False)
                if belief:
                    committed.append(belief)
        return committed
