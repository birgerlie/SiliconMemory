"""EntityResolver — orchestrates detect → extract → disambiguate → cache."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from silicon_memory.entities.cache import EntityCache
from silicon_memory.entities.rules import RuleEngine
from silicon_memory.entities.types import EntityReference, ResolveResult

if TYPE_CHECKING:
    from silicon_memory.entities.learner import RuleLearner

logger = logging.getLogger(__name__)


class EntityResolver:
    """Self-learning entity resolver with three-pass architecture.

    Pass 1: Detect candidates (broad regex, full text)
    Pass 2: Extract typed references (precise regex, per-candidate)
    Pass 3: Disambiguate via context embedding (only if ambiguous)
    Cache:  Resolve canonical ID via in-memory dict
    """

    def __init__(
        self,
        cache: EntityCache,
        rules: RuleEngine,
        learner: "RuleLearner | None" = None,
    ) -> None:
        self.cache = cache
        self.rules = rules
        self._learner = learner
        self._unresolved_queue: list[dict] = []

    async def resolve(self, text: str) -> ResolveResult:
        """Resolve entity references in text. Called by all ingestion adapters."""
        if not text:
            return ResolveResult()

        # Pass 1: Detect candidates (broad regex)
        candidates = self.rules.detect(text)
        if not candidates:
            return ResolveResult()

        # Pass 2: Extract typed references (precise regex per candidate)
        extractions = self.rules.extract(candidates)

        # Pass 3: Disambiguate (only where multiple extractors matched)
        references = self._disambiguate(extractions)

        # Enhance canonical_ids via cache (cache enriches, doesn't gate)
        for ref in references:
            better = self.cache.lookup(ref.canonical_id) or self.cache.lookup(ref.text)
            if better:
                ref.canonical_id = better

        # Candidates with no extractor match → unresolved
        extracted_indices = set(extractions.keys())
        unresolved: list[str] = []
        for i, candidate in enumerate(candidates):
            if i not in extracted_indices:
                unresolved.append(candidate.text)
                self._unresolved_queue.append({
                    "text": candidate.text,
                    "context": candidate.context_text,
                })

        return ResolveResult(resolved=references, unresolved=unresolved)

    def _disambiguate(
        self, extractions: dict[int, list[EntityReference]]
    ) -> list[EntityReference]:
        """Pass 3: For candidates with multiple extractor matches, pick best.

        Currently picks highest confidence. Context embedding disambiguation
        can be added when SiliconDB embedding support is wired in.
        """
        results: list[EntityReference] = []
        for matches in extractions.values():
            if len(matches) == 1:
                results.append(matches[0])
            else:
                best = max(matches, key=lambda r: r.confidence)
                results.append(best)
        return results

    async def resolve_single(self, name: str) -> EntityReference | None:
        """Resolve a single entity name via cache lookup."""
        canonical = self.cache.lookup(name)
        if canonical:
            entity_type = self.cache.get_type(canonical) or "unknown"
            return EntityReference(
                text=name,
                canonical_id=canonical,
                entity_type=entity_type,
                confidence=1.0,
                span=(0, len(name)),
                context_text=name,
            )
        return None

    async def register_alias(
        self, alias: str, canonical_id: str, entity_type: str
    ) -> None:
        """Manually register an alias → canonical mapping."""
        self.cache.store(alias, canonical_id, entity_type)

    async def learn_rules(self) -> int:
        """Trigger offline rule learning from unresolved queue."""
        if not self._learner or not self._unresolved_queue:
            return 0
        detectors, extractors = await self._learner.generate_rules(
            self._unresolved_queue
        )
        for d in detectors:
            self.rules.add_detector(d)
        for e in extractors:
            self.rules.add_extractor(e)
        count = len(detectors) + len(extractors)
        self._unresolved_queue.clear()
        return count

    @property
    def unresolved_count(self) -> int:
        return len(self._unresolved_queue)
