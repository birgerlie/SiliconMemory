"""Experience processor for the reflection engine."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID
import re

from silicon_memory.core.types import Experience
from silicon_memory.reflection.types import (
    ExperienceGroup,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class ExperienceProcessor:
    """Processes experiences to prepare them for pattern extraction.

    The processor:
    1. Fetches unprocessed experiences
    2. Groups related experiences (by session, entity, time)
    3. Extracts entities and keywords
    4. Prepares data for pattern extraction

    Example:
        >>> processor = ExperienceProcessor(memory)
        >>> groups = await processor.process_batch()
        >>> for group in groups:
        ...     print(f"Group with {group.size} experiences")
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()

    async def fetch_unprocessed(
        self,
        limit: int | None = None,
    ) -> list[Experience]:
        """Fetch unprocessed experiences from memory."""
        max_exp = limit or self._config.max_experiences_per_batch
        return await self._memory._backend.get_unprocessed_experiences(max_exp)

    async def process_batch(
        self,
        experiences: list[Experience] | None = None,
    ) -> list[ExperienceGroup]:
        """Process a batch of experiences and return grouped results.

        Args:
            experiences: Experiences to process. If None, fetches unprocessed.

        Returns:
            List of experience groups for pattern extraction.
        """
        if experiences is None:
            experiences = await self.fetch_unprocessed()

        if not experiences:
            return []

        # Group by multiple strategies
        groups: list[ExperienceGroup] = []

        # 1. Group by session
        session_groups = self._group_by_session(experiences)
        groups.extend(session_groups)

        # 2. Group by common entities
        entity_groups = self._group_by_entities(experiences)
        groups.extend(entity_groups)

        # 3. Group by time proximity
        time_groups = self._group_by_time(experiences)
        groups.extend(time_groups)

        # Deduplicate groups that are too similar
        groups = self._deduplicate_groups(groups)

        return groups

    def _group_by_session(
        self,
        experiences: list[Experience],
    ) -> list[ExperienceGroup]:
        """Group experiences by session ID."""
        session_map: dict[str, list[Experience]] = defaultdict(list)

        for exp in experiences:
            if exp.session_id:
                session_map[exp.session_id].append(exp)

        groups = []
        for session_id, exps in session_map.items():
            if len(exps) >= 2:  # Need at least 2 for patterns
                group = ExperienceGroup(
                    experiences=[e.id for e in exps],
                    session_id=session_id,
                    common_tags=self._find_common_tags(exps),
                    common_entities=self._extract_common_entities(exps),
                )
                if exps:
                    times = [e.occurred_at for e in exps]
                    group.time_span = (min(times), max(times))
                groups.append(group)

        return groups

    def _group_by_entities(
        self,
        experiences: list[Experience],
    ) -> list[ExperienceGroup]:
        """Group experiences by common entities mentioned."""
        # Extract entities from each experience
        entity_to_exps: dict[str, list[Experience]] = defaultdict(list)

        for exp in experiences:
            entities = self._extract_entities(exp)
            for entity in entities:
                entity_to_exps[entity.lower()].append(exp)

        groups = []
        for entity, exps in entity_to_exps.items():
            if len(exps) >= 2:
                group = ExperienceGroup(
                    experiences=[e.id for e in exps],
                    common_entities={entity},
                    common_tags=self._find_common_tags(exps),
                )
                groups.append(group)

        return groups

    def _group_by_time(
        self,
        experiences: list[Experience],
        window_minutes: int = 30,
    ) -> list[ExperienceGroup]:
        """Group experiences that occurred within a time window."""
        if not experiences:
            return []

        # Sort by time
        sorted_exps = sorted(experiences, key=lambda e: e.occurred_at)
        groups = []
        current_group: list[Experience] = [sorted_exps[0]]

        for exp in sorted_exps[1:]:
            last_time = current_group[-1].occurred_at
            delta = (exp.occurred_at - last_time).total_seconds() / 60

            if delta <= window_minutes:
                current_group.append(exp)
            else:
                if len(current_group) >= 2:
                    groups.append(self._create_time_group(current_group))
                current_group = [exp]

        # Don't forget the last group
        if len(current_group) >= 2:
            groups.append(self._create_time_group(current_group))

        return groups

    def _create_time_group(self, exps: list[Experience]) -> ExperienceGroup:
        """Create an experience group from a list of experiences."""
        times = [e.occurred_at for e in exps]
        return ExperienceGroup(
            experiences=[e.id for e in exps],
            common_tags=self._find_common_tags(exps),
            common_entities=self._extract_common_entities(exps),
            time_span=(min(times), max(times)),
        )

    def _extract_entities(self, exp: Experience) -> set[str]:
        """Extract entity mentions from an experience."""
        entities: set[str] = set()
        text = exp.content

        # Simple entity extraction: capitalized words, quoted strings
        # Capitalized words (potential proper nouns)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(words)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.update(quoted)

        # From context if available
        if exp.context:
            for key in ['entity', 'subject', 'topic', 'user', 'name']:
                if key in exp.context:
                    val = exp.context[key]
                    if isinstance(val, str):
                        entities.add(val)
                    elif isinstance(val, list):
                        entities.update(str(v) for v in val)

        # From tags
        entities.update(exp.tags)

        return entities

    def _extract_common_entities(
        self,
        experiences: list[Experience],
    ) -> set[str]:
        """Find entities mentioned in multiple experiences."""
        if not experiences:
            return set()

        entity_counts: dict[str, int] = defaultdict(int)
        for exp in experiences:
            for entity in self._extract_entities(exp):
                entity_counts[entity.lower()] += 1

        # Return entities mentioned in at least 2 experiences
        return {
            entity for entity, count in entity_counts.items()
            if count >= 2
        }

    def _find_common_tags(self, experiences: list[Experience]) -> set[str]:
        """Find tags common to multiple experiences."""
        if not experiences:
            return set()

        tag_counts: dict[str, int] = defaultdict(int)
        for exp in experiences:
            for tag in exp.tags:
                tag_counts[tag] += 1

        # Return tags present in at least half the experiences
        threshold = max(2, len(experiences) // 2)
        return {
            tag for tag, count in tag_counts.items()
            if count >= threshold
        }

    def _deduplicate_groups(
        self,
        groups: list[ExperienceGroup],
    ) -> list[ExperienceGroup]:
        """Remove groups that are too similar (>80% overlap)."""
        if not groups:
            return []

        unique_groups = []
        seen_experience_sets: list[set[UUID]] = []

        for group in groups:
            exp_set = set(group.experiences)

            # Check overlap with existing groups
            is_duplicate = False
            for seen_set in seen_experience_sets:
                overlap = len(exp_set & seen_set)
                total = len(exp_set | seen_set)
                if total > 0 and overlap / total > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_groups.append(group)
                seen_experience_sets.append(exp_set)

        return unique_groups

    async def mark_processed(self, experience_ids: list[UUID]) -> None:
        """Mark experiences as processed."""
        await self._memory.mark_experiences_processed(experience_ids)
