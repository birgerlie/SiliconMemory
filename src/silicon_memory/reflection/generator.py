"""Belief generator for the reflection engine."""

from __future__ import annotations

import calendar
from datetime import datetime
from datetime import timezone
import hashlib
import re
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.claims import (
    AntiTrivialityPolicy,
    ClaimDecision,
    ClaimDecisionLabel,
    ClaimPrecisionJudge,
)
from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Source,
    SourceType,
    TemporalContext,
    Triplet,
)
from silicon_memory.core.utils import utc_now
from silicon_memory.entities.date_normalizer import normalize_date
from silicon_memory.reflection.types import (
    BeliefCandidate,
    Pattern,
    PatternType,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


def _parse_iso_date(canonical: str) -> datetime | None:
    parts = canonical.split("-")
    try:
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            return datetime(y, m, d, tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _triplet_key(subject: str, predicate: str, obj: str) -> str:
    return f"{_norm_text(subject)}|{_norm_text(predicate)}|{_norm_text(obj)}"


def _doc_id_from_source_context(source_context: dict[str, Any]) -> str:
    source_doc = source_context.get("source_document")
    if isinstance(source_doc, dict):
        for key in ("document_id", "title", "experience_id"):
            value = source_doc.get(key)
            if value:
                return str(value)
    if isinstance(source_doc, str) and source_doc.strip():
        return source_doc.strip()
    return "unknown-doc"


def _build_embedding_text(candidate: BeliefCandidate) -> str:
    subject = (candidate.subject or "").strip()
    predicate = (candidate.predicate or "").strip()
    obj = (candidate.object or "").strip()
    source_context = candidate.source_context or {}
    doc_id = _doc_id_from_source_context(source_context)

    date_value = ""
    date_meta = source_context.get("date_normalized")
    if isinstance(date_meta, dict):
        date_value = str(date_meta.get("canonical") or "")
    if not date_value:
        date_value = str(source_context.get("date") or "")

    triplet_text = candidate.content
    if subject and predicate and obj:
        triplet_text = f"{subject} | {predicate} | {obj}"

    co_triplets = source_context.get("co_triplet_keys") or []
    co_text = "; ".join(str(x) for x in co_triplets[:5])

    parts = [
        triplet_text,
        f"doc: {doc_id}",
    ]
    if date_value:
        parts.append(f"date: {date_value}")
    if co_text:
        parts.append(f"context: {co_text}")
    return " | ".join(parts)


def _enrich_context_bundles(candidates: list[BeliefCandidate]) -> None:
    """Attach context-bundle and co-triplet metadata to candidates."""
    by_doc: dict[str, list[BeliefCandidate]] = {}
    for candidate in candidates:
        source_context = candidate.source_context or {}
        if candidate.source_context is None:
            candidate.source_context = source_context
        doc_id = _doc_id_from_source_context(source_context)
        by_doc.setdefault(doc_id, []).append(candidate)

    for doc_id, group in by_doc.items():
        triplet_keys: list[str] = []
        for candidate in group:
            if candidate.has_triplet and candidate.subject and candidate.predicate and candidate.object:
                triplet_keys.append(
                    _triplet_key(candidate.subject, candidate.predicate, candidate.object)
                )
        # Stable deterministic bundle id by document + triplets.
        digest_src = f"{doc_id}|{'|'.join(sorted(triplet_keys))}"
        bundle_id = "ctx_" + hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]

        for candidate in group:
            source_context = candidate.source_context
            source_context["context_bundle_id"] = bundle_id
            source_context["grounding_doc_id"] = doc_id

            own_key = None
            if candidate.has_triplet and candidate.subject and candidate.predicate and candidate.object:
                own_key = _triplet_key(candidate.subject, candidate.predicate, candidate.object)
                source_context["triplet_key"] = own_key

            co_keys = [k for k in triplet_keys if k != own_key][:20]
            if co_keys:
                source_context["co_triplet_keys"] = co_keys

            source_context["embedding_text"] = _build_embedding_text(candidate)


def _build_temporal_context(source_context: dict[str, Any]) -> TemporalContext | None:
    """Build temporal context from extracted source date metadata."""
    if not source_context:
        return None

    date_meta = (
        source_context.get("date_normalized")
        or source_context.get("object_date")
        or source_context.get("subject_date")
    )

    if isinstance(date_meta, dict):
        canonical = str(date_meta.get("canonical") or "")
        precision = str(date_meta.get("precision") or "")
        ambiguous = bool(date_meta.get("ambiguous"))
        relative = bool(date_meta.get("relative"))
        # Relative dates ("this year", "tomorrow") should not be anchored
        # as stable temporal coordinates in long-term beliefs.
        if ambiguous or relative:
            return None
        if precision == "day":
            observed_at = _parse_iso_date(canonical)
            if observed_at:
                return TemporalContext(observed_at=observed_at)
        elif precision == "month":
            parts = canonical.split("-")
            if len(parts) == 2:
                y, m = int(parts[0]), int(parts[1])
                last_day = calendar.monthrange(y, m)[1]
                start = datetime(y, m, 1, tzinfo=timezone.utc)
                end = datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc)
                return TemporalContext(
                    observed_at=start,
                    valid_from=start,
                    valid_until=end,
                )
        elif precision == "year":
            y = int(canonical)
            start = datetime(y, 1, 1, tzinfo=timezone.utc)
            end = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return TemporalContext(
                observed_at=start,
                valid_from=start,
                valid_until=end,
            )

    raw_date = source_context.get("date")
    if isinstance(raw_date, str) and raw_date.strip():
        source_doc = source_context.get("source_document")
        reference_datetime = ""
        if isinstance(source_doc, dict):
            reference_datetime = str(source_doc.get("occurred_at") or "")
        normalized = normalize_date(
            raw_date,
            reference_datetime=reference_datetime or None,
        )
        if normalized and not normalized.ambiguous and not normalized.relative:
            return _build_temporal_context({"date_normalized": normalized.__dict__})

    # Fallback: anchor beliefs to the source experience timestamp when available.
    source_doc = source_context.get("source_document")
    if isinstance(source_doc, dict):
        occurred_at = str(source_doc.get("occurred_at") or "").strip()
        if occurred_at:
            try:
                return TemporalContext(
                    observed_at=datetime.fromisoformat(
                        occurred_at.replace("Z", "+00:00"),
                    ),
                )
            except Exception:
                pass

    return None


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
        llm: Any | None = None,
    ) -> None:
        self._memory = memory
        self._config = config or ReflectionConfig()
        self._llm = llm
        self._claim_judge = ClaimPrecisionJudge()
        self._anti_triviality = AntiTrivialityPolicy()

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

        # Enrich with document-grounded context bundles and co-triplet links.
        _enrich_context_bundles(candidates)

        # Validate candidates against existing knowledge
        candidates = await self._validate_candidates(candidates, existing_beliefs)

        # Limit number of candidates (0 = unlimited)
        candidates = sorted(
            candidates,
            key=lambda c: c.confidence,
            reverse=True,
        )
        cap = self._config.max_beliefs_per_cycle
        if cap > 0:
            candidates = candidates[:cap]

        return candidates

    def _pattern_to_candidate(self, pattern: Pattern) -> BeliefCandidate | None:
        """Convert a pattern to a belief candidate."""
        object_value = pattern.object
        derived_date_meta: dict[str, Any] | None = None
        # Generate content based on pattern type
        if pattern.type == PatternType.FACT:
            content = pattern.description
            predicate = pattern.predicate or "is"
        elif pattern.type == PatternType.RELATIONSHIP:
            content = pattern.description
            predicate = pattern.predicate or "related to"
        elif pattern.type == PatternType.ARGUMENT:
            content = pattern.description
            predicate = pattern.predicate or "argues"
        elif pattern.type == PatternType.TIMELINE_EVENT:
            content = pattern.description
            context_date_meta = (pattern.context or {}).get("date_normalized") or {}
            canonical_date = str(
                context_date_meta.get("canonical") or "",
            ).strip()
            if not canonical_date:
                reference_datetime = ""
                source_doc = (pattern.context or {}).get("source")
                if isinstance(source_doc, dict):
                    reference_datetime = str(source_doc.get("occurred_at") or "")
                context_raw_date = str((pattern.context or {}).get("date") or "").strip()
                for raw_date in (context_raw_date, pattern.object, pattern.description):
                    if not raw_date:
                        continue
                    normalized = normalize_date(
                        raw_date,
                        reference_datetime=reference_datetime or None,
                    )
                    if normalized and not normalized.ambiguous and not normalized.relative:
                        canonical_date = normalized.canonical
                        derived_date_meta = normalized.__dict__
                        break
            if canonical_date:
                # Represent timeline events as date-anchored triples so graph
                # operators can reason over temporal progression directly.
                predicate = "occurred_on"
                object_value = canonical_date
            else:
                predicate = pattern.predicate or "event"
        elif pattern.type == PatternType.CAUSAL:
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

        # Collect source document provenance from pattern context
        source_ctx: dict[str, Any] = {}
        if pattern.context.get("source"):
            source_ctx["source_document"] = pattern.context["source"]
        if pattern.context.get("date"):
            source_ctx["date"] = pattern.context["date"]
        if pattern.context.get("date_normalized"):
            source_ctx["date_normalized"] = pattern.context["date_normalized"]
        elif derived_date_meta:
            source_ctx["date_normalized"] = derived_date_meta
        if pattern.context.get("object_date"):
            source_ctx["object_date"] = pattern.context["object_date"]
        if pattern.context.get("subject_date"):
            source_ctx["subject_date"] = pattern.context["subject_date"]
        if pattern.context.get("rhetoric"):
            source_ctx["rhetoric"] = pattern.context["rhetoric"]
        if pattern.type == PatternType.TIMELINE_EVENT and pattern.object:
            source_ctx["event_text"] = pattern.object

        return BeliefCandidate(
            id=uuid4(),
            content=content,
            subject=pattern.subject,
            predicate=predicate,
            object=object_value,
            confidence=confidence,
            source_patterns=[pattern.id],
            source_experiences=pattern.evidence,
            source_context=source_ctx,
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

    async def _decide_claim_relation(self, belief: Belief) -> tuple[ClaimDecision | None, int]:
        """Classify candidate relation against existing canonical claims."""
        backend = getattr(self._memory, "_backend", None)
        if backend is None or not hasattr(backend, "find_claim_merge_candidates"):
            return None, 0

        try:
            candidates = await backend.find_claim_merge_candidates(belief, limit=8)
        except Exception:
            return None, 0

        decision = await self._claim_judge.decide(
            belief,
            candidates,
            llm=self._llm,
        )
        return decision, len(candidates)

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

        experience_ids = [str(e) for e in candidate.source_experiences]
        source_metadata: dict[str, Any] = {
            "patterns": [str(p) for p in candidate.source_patterns],
            "experiences": experience_ids,
            "reasoning": candidate.reasoning,
        }
        if candidate.source_context:
            source_metadata.update(candidate.source_context)
        source_metadata.setdefault("source_type", "experience")
        source_metadata.setdefault("extraction_run_id", source_metadata.get("context_bundle_id", "reflection"))
        source_metadata.setdefault(
            "extractor_model",
            str(getattr(self._llm, "model", "unknown") or "unknown"),
        )
        source_metadata.setdefault("extractor_version", "belief_generator_v1")
        source_metadata.setdefault("extracted_at", utc_now().isoformat())
        source_metadata.setdefault(
            "evidence_span",
            str(source_metadata.get("triplet_key") or source_metadata.get("context_bundle_id") or candidate.id),
        )
        source_metadata.setdefault("confidence_basis", "hybrid")
        temporal = _build_temporal_context(candidate.source_context or {})
        belief_tags: set[str] = set()
        source_type = str((candidate.source_context or {}).get("type") or "").strip().lower()
        if source_type:
            belief_tags.add(source_type)
        if str((candidate.source_context or {}).get("operator") or "").strip():
            belief_tags.add("deterministic_discovery")

        belief = Belief(
            id=candidate.id,
            content=candidate.content,
            triplet=triplet,
            confidence=candidate.confidence,
            source=Source(
                id="reflection_engine",
                type=SourceType.REFLECTION,
                reliability=0.7,
                metadata=source_metadata,
            ),
            status=BeliefStatus.PROVISIONAL,
            temporal=temporal,
            evidence_for=candidate.supports,
            evidence_against=candidate.contradicts,
            tags=belief_tags,
            metadata={},
        )

        decision, candidate_count = await self._decide_claim_relation(belief)
        if decision:
            belief.metadata["claim_relation_label"] = decision.label.value
            belief.metadata["claim_decision_reason"] = decision.reason
            belief.metadata["claim_decision_confidence"] = decision.confidence
            belief.metadata["claim_decision_via"] = decision.via
            belief.metadata["claim_candidate_count"] = candidate_count
            if decision.candidate_belief_id:
                belief.metadata["canonical_claim_id"] = str(decision.candidate_belief_id)
            else:
                belief.metadata["canonical_claim_id"] = str(belief.id)
            if decision.label == ClaimDecisionLabel.MERGE:
                belief.tags.add("observation_only")
            elif decision.label == ClaimDecisionLabel.CONTRADICTION:
                belief.status = BeliefStatus.CONTESTED

        anti_triviality = self._anti_triviality.evaluate(belief)
        belief.metadata["anti_triviality_score"] = anti_triviality.score
        belief.metadata["anti_triviality_signals"] = anti_triviality.signals
        belief.metadata["anti_triviality_block_reasons"] = anti_triviality.block_reasons
        if not anti_triviality.promotable:
            belief.metadata["canonical_promotion_allowed"] = False
            belief.metadata["observation_only"] = True
            belief.tags.add("observation_only")

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
