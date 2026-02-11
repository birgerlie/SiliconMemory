"""Pattern extractor for the reflection engine."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID
import re

from silicon_memory.core.types import Experience
from silicon_memory.reflection.types import (
    Pattern,
    PatternType,
    ExperienceGroup,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class PatternExtractor:
    """Extracts patterns from experience groups.

    Patterns include:
    - Causal relationships (A causes B)
    - Temporal sequences (A followed by B)
    - Correlations (A often occurs with B)
    - Generalizations (All X have property Y)
    - Preferences (User prefers X over Y)
    - Facts (Simple observations)

    Example:
        >>> extractor = PatternExtractor(memory)
        >>> patterns = await extractor.extract_patterns(groups)
        >>> for p in patterns:
        ...     print(f"{p.type.value}: {p.description}")
    """

    # Causal indicators in text
    CAUSAL_INDICATORS = [
        r'\b(because|caused|resulted in|led to|due to|therefore|hence)\b',
        r'\b(made|triggered|enabled|prevented|blocked)\b',
        r'->\s*',  # Arrow notation
    ]

    # Temporal indicators
    TEMPORAL_INDICATORS = [
        r'\b(then|after|before|following|next|subsequently)\b',
        r'\b(first|second|finally|lastly)\b',
    ]

    # Outcome indicators
    OUTCOME_POSITIVE = [
        r'\b(success|worked|completed|solved|fixed|resolved)\b',
        r'\b(good|great|excellent|perfect)\b',
    ]

    OUTCOME_NEGATIVE = [
        r'\b(fail|error|bug|issue|problem|broken)\b',
        r'\b(bad|wrong|incorrect)\b',
    ]

    def __init__(
        self,
        memory: "SiliconMemory",
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()

    async def extract_patterns(
        self,
        groups: list[ExperienceGroup],
        min_occurrences: int | None = None,
    ) -> list[Pattern]:
        """Extract patterns from experience groups.

        Args:
            groups: Experience groups to analyze
            min_occurrences: Minimum occurrences for a pattern to be valid

        Returns:
            List of extracted patterns
        """
        min_occ = min_occurrences or self._config.min_pattern_occurrences
        all_patterns: list[Pattern] = []

        for group in groups:
            # Fetch full experiences for the group
            experiences = await self._fetch_experiences(group.experiences)
            if len(experiences) < 2:
                continue

            # Extract different pattern types
            if self._config.enable_causal_patterns:
                causal = self._extract_causal_patterns(experiences, group)
                all_patterns.extend(causal)

            if self._config.enable_temporal_patterns:
                temporal = self._extract_temporal_patterns(experiences, group)
                all_patterns.extend(temporal)

            if self._config.enable_correlation_patterns:
                correlations = self._extract_correlations(experiences, group)
                all_patterns.extend(correlations)

            if self._config.enable_generalizations:
                generalizations = self._extract_generalizations(experiences, group)
                all_patterns.extend(generalizations)

            # Always extract fact patterns
            facts = self._extract_fact_patterns(experiences, group)
            all_patterns.extend(facts)

        # Filter by minimum occurrences and deduplicate
        patterns = self._filter_and_deduplicate(all_patterns, min_occ)

        return patterns

    async def _fetch_experiences(
        self,
        experience_ids: list[UUID],
    ) -> list[Experience]:
        """Fetch full experience objects."""
        experiences = []
        for eid in experience_ids:
            exp = await self._memory.get_experience(eid)
            if exp:
                experiences.append(exp)
        return experiences

    def _extract_causal_patterns(
        self,
        experiences: list[Experience],
        group: ExperienceGroup,
    ) -> list[Pattern]:
        """Extract causal patterns (A causes B)."""
        patterns = []

        for exp in experiences:
            text = exp.content + " " + (exp.outcome or "")

            # Look for causal language
            for indicator in self.CAUSAL_INDICATORS:
                if re.search(indicator, text, re.IGNORECASE):
                    # Try to extract cause and effect
                    parts = re.split(indicator, text, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) >= 2:
                        cause = parts[0].strip()[-100:]  # Last 100 chars before
                        effect = parts[1].strip()[:100]  # First 100 chars after

                        if cause and effect:
                            pattern = Pattern(
                                type=PatternType.CAUSAL,
                                description=f"{cause} -> {effect}",
                                evidence=[exp.id],
                                confidence=0.6,
                                subject=self._extract_subject(cause),
                                predicate="causes",
                                object=self._extract_subject(effect),
                            )
                            patterns.append(pattern)
                    break

        return patterns

    def _extract_temporal_patterns(
        self,
        experiences: list[Experience],
        group: ExperienceGroup,
    ) -> list[Pattern]:
        """Extract temporal sequence patterns."""
        patterns = []

        # Sort by time
        sorted_exps = sorted(experiences, key=lambda e: e.occurred_at)

        # Look for sequential patterns
        for i in range(len(sorted_exps) - 1):
            exp1 = sorted_exps[i]
            exp2 = sorted_exps[i + 1]

            # If they share a session, they may be sequential steps
            if exp1.session_id and exp1.session_id == exp2.session_id:
                pattern = Pattern(
                    type=PatternType.TEMPORAL,
                    description=f"'{self._summarize(exp1.content)}' followed by '{self._summarize(exp2.content)}'",
                    evidence=[exp1.id, exp2.id],
                    confidence=0.5,
                    first_observed=exp1.occurred_at,
                    last_observed=exp2.occurred_at,
                    context={
                        "step1": exp1.content[:200],
                        "step2": exp2.content[:200],
                        "session": exp1.session_id,
                    },
                )
                patterns.append(pattern)

        return patterns

    def _extract_correlations(
        self,
        experiences: list[Experience],
        group: ExperienceGroup,
    ) -> list[Pattern]:
        """Extract correlation patterns (A often occurs with B)."""
        patterns = []

        # Use common entities from the group
        if len(group.common_entities) >= 2:
            entities = list(group.common_entities)
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    pattern = Pattern(
                        type=PatternType.CORRELATION,
                        description=f"'{entities[i]}' often co-occurs with '{entities[j]}'",
                        evidence=group.experiences[:5],  # First 5 as evidence
                        confidence=min(0.9, 0.4 + 0.1 * group.size),
                        occurrences=group.size,
                        subject=entities[i],
                        predicate="co-occurs with",
                        object=entities[j],
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_generalizations(
        self,
        experiences: list[Experience],
        group: ExperienceGroup,
    ) -> list[Pattern]:
        """Extract generalization patterns."""
        patterns = []

        # Look for common outcomes
        positive_count = 0
        negative_count = 0

        for exp in experiences:
            text = (exp.outcome or "") + " " + exp.content
            if any(re.search(p, text, re.IGNORECASE) for p in self.OUTCOME_POSITIVE):
                positive_count += 1
            if any(re.search(p, text, re.IGNORECASE) for p in self.OUTCOME_NEGATIVE):
                negative_count += 1

        # If there's a strong trend, create a generalization
        total = len(experiences)
        if total >= 3:
            if positive_count >= total * 0.7:
                for entity in group.common_entities:
                    pattern = Pattern(
                        type=PatternType.GENERALIZATION,
                        description=f"Actions involving '{entity}' tend to succeed",
                        evidence=[e.id for e in experiences[:5]],
                        confidence=positive_count / total,
                        occurrences=positive_count,
                        subject=entity,
                        predicate="tends to",
                        object="succeed",
                    )
                    patterns.append(pattern)

            elif negative_count >= total * 0.7:
                for entity in group.common_entities:
                    pattern = Pattern(
                        type=PatternType.GENERALIZATION,
                        description=f"Actions involving '{entity}' tend to fail",
                        evidence=[e.id for e in experiences[:5]],
                        confidence=negative_count / total,
                        occurrences=negative_count,
                        subject=entity,
                        predicate="tends to",
                        object="fail",
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_fact_patterns(
        self,
        experiences: list[Experience],
        group: ExperienceGroup,
    ) -> list[Pattern]:
        """Extract simple fact patterns from experiences."""
        patterns = []

        # Create fact patterns from common entities and tags
        for entity in group.common_entities:
            pattern = Pattern(
                type=PatternType.FACT,
                description=f"'{entity}' appears in related experiences",
                evidence=group.experiences[:5],
                confidence=0.5 + 0.05 * min(10, group.size),
                occurrences=group.size,
                subject=entity,
                predicate="is mentioned in",
                object=f"{group.size} related experiences",
            )
            patterns.append(pattern)

        return patterns

    def _filter_and_deduplicate(
        self,
        patterns: list[Pattern],
        min_occurrences: int,
    ) -> list[Pattern]:
        """Filter patterns by occurrence count and deduplicate."""
        # Group similar patterns
        pattern_groups: dict[str, list[Pattern]] = defaultdict(list)

        for pattern in patterns:
            # Create a key based on type and core content
            key = f"{pattern.type.value}:{pattern.subject}:{pattern.predicate}:{pattern.object}"
            pattern_groups[key].append(pattern)

        # Merge similar patterns
        merged: list[Pattern] = []
        for key, group in pattern_groups.items():
            if len(group) >= min_occurrences:
                # Merge into single pattern with combined evidence
                best = max(group, key=lambda p: p.confidence)
                all_evidence: set[UUID] = set()
                for p in group:
                    all_evidence.update(p.evidence)

                best.evidence = list(all_evidence)[:10]
                best.occurrences = len(group)
                best.confidence = min(0.95, best.confidence + 0.05 * len(group))
                merged.append(best)

        return merged

    def _extract_subject(self, text: str) -> str:
        """Extract a subject/noun from text."""
        # Simple extraction: first capitalized word or first noun-like word
        words = text.split()
        for word in words:
            cleaned = word.strip('.,!?()[]"\'')
            if cleaned and cleaned[0].isupper():
                return cleaned
        # Fall back to first significant word
        for word in words:
            cleaned = word.strip('.,!?()[]"\'')
            if len(cleaned) > 3:
                return cleaned
        return text[:30]

    def _summarize(self, text: str, max_len: int = 50) -> str:
        """Create a short summary of text."""
        text = text.strip()
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
