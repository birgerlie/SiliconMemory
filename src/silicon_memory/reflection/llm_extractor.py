"""LLM-powered multi-dimensional knowledge extractor for the reflection engine.

Extracts structured knowledge from experience text across four dimensions:
1. Facts — subject-predicate-object triplets
2. Relationships — person-to-person and person-to-institution links
3. Arguments — legal/logical arguments with rhetoric classification (pathos/logos/ethos)
4. Timeline events — dated events with actors and significance

Each extraction is grounded to its source document for provenance tracking.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel

from silicon_memory.core.types import Experience
from silicon_memory.llm.model_policy import ensure_extraction_model_allowed
from silicon_memory.entities.date_normalizer import normalize_date
from silicon_memory.reflection.types import (
    ExperienceGroup,
    Pattern,
    PatternType,
    ReflectionConfig,
)

if TYPE_CHECKING:
    from silicon_memory.entities.resolver import EntityResolver
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)

# Common English words to skip when extracting entity candidates
_COMMON_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "has", "his", "how", "its",
    "may", "new", "now", "old", "see", "way", "who", "did", "get", "let",
    "say", "she", "too", "use", "also", "been", "call", "each", "find",
    "from", "give", "have", "here", "high", "just", "know", "last", "long",
    "made", "make", "many", "more", "most", "much", "must", "name", "next",
    "only", "over", "part", "said", "same", "some", "such", "take", "than",
    "that", "them", "then", "they", "this", "time", "very", "want", "well",
    "went", "were", "what", "when", "will", "with", "work", "year",
    "about", "after", "again", "being", "below", "between", "both",
    "could", "does", "down", "first", "found", "great", "house", "into",
    "large", "later", "never", "other", "place", "point", "right",
    "shall", "small", "still", "their", "there", "these", "thing",
    "think", "those", "three", "under", "water", "where", "which",
    "while", "world", "would", "before", "should", "through",
    "because", "during", "even", "every", "further", "given",
    "however", "including", "itself", "often", "rather", "since",
    "though", "upon", "whether", "therefore", "furthermore",
    "moreover", "thus", "accordingly", "nevertheless", "within",
})


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Return True when upstream LLM is unavailable/unloaded."""
    message = str(exc).lower()
    if not message:
        return False
    markers = (
        "no models loaded",
        "model has crashed",
        "invalid_request_error",
        "param': 'model'",
        'param": "model"',
    )
    return any(marker in message for marker in markers)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class ExtractedFact(BaseModel):
    subject: str
    predicate: str
    object: str
    source: str = ""  # which document/source this came from
    confidence: float = 0.7


class ExtractedRelationship(BaseModel):
    person1: str
    relationship: str  # e.g. "employer of", "collaborator with"
    person2: str
    context: str = ""  # e.g. "during 1999-2005"
    source: str = ""
    confidence: float = 0.7


class ExtractedArgument(BaseModel):
    claim: str  # what is being argued
    evidence: str  # what evidence or reasoning supports it
    rhetoric: str = "logos"  # "pathos", "logos", or "ethos"
    actor: str = ""  # who makes this argument
    source: str = ""
    confidence: float = 0.7


class ExtractedEvent(BaseModel):
    date: str  # approximate date or period
    event: str  # what happened
    actors: list[str] = []
    significance: str = ""  # why it matters
    source: str = ""
    confidence: float = 0.7


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = []
    relationships: list[ExtractedRelationship] = []
    arguments: list[ExtractedArgument] = []
    events: list[ExtractedEvent] = []


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are an expert analyst extracting structured knowledge from documents. \
Read the ENTIRE text below and extract knowledge across four dimensions. \
Be thorough — cover all sections, not just the first paragraph.

Each text segment is labelled with a source tag like [SRC:doc_id]. \
For every extracted item, set the "source" field to the doc_id of the \
document it came from. If a fact spans multiple sources, pick the primary one.

1. FACTS: Subject-predicate-object triplets.
   - People and roles (e.g. "Marie Curie" / "discovered" / "radium")
   - Organizational facts (e.g. "NASA" / "launched" / "Mars Rover")
   - Institutional relationships (e.g. "WHO" / "headquartered in" / "Geneva")
   - Outcomes and results (e.g. "Tesla" / "revenue" / "$81.5 billion")
   - Locations and properties (e.g. "Amazon" / "operates" / "AWS data centers")

2. RELATIONSHIPS: Connections between people or organisations.
   - person1, relationship type, person2, context, source
   - e.g. person1="Watson", relationship="collaborator of", person2="Crick", context="1950s DNA research"

3. ARGUMENTS: Logical or rhetorical arguments made in the text.
   - claim: the core assertion
   - evidence: supporting reasoning or cited evidence
   - rhetoric: classify as "pathos" (emotional appeal), "logos" (logical reasoning), or "ethos" (appeal to authority/credibility)
   - actor: who makes the argument
   - source: which document

4. EVENTS: Dated occurrences.
   - date, what happened, who was involved, why it matters, source

PREDICATE RULES (critical for knowledge graph quality):
- Use active voice, present tense: "owns" not "is the owner of", "convicted of" not "was convicted of"
- Remove copulas (is, was, were, are, has been): "attorney for" not "is an attorney for"
- Remove articles and prepositions where possible: "owns property" not "is the owner of the property"
- 2-4 words maximum: "represented" not "served as legal representative for"
- Reuse predicates across facts when the meaning is the same — consistency matters
- For the same relationship, always use the same predicate form

General rules:
- Extract from ALL sections of the text, not just the beginning
- Extract only what is explicitly stated or clearly implied
- Use short, clear values (proper nouns preferred)
- Assign confidence 0.0-1.0 based on clarity
- Always set the "source" field to the document identifier
- Skip boilerplate (headers, footers, document numbers)
- Maximum {max_items} items per dimension

Respond with a single JSON object matching this schema:
{{"facts": [...], "relationships": [...], "arguments": [...], "events": [...]}}
No markdown fences, no explanation, just valid JSON.

Text:
{text}"""


_EXTRACTION_MISSING_DIMENSIONS_PROMPT = """\
The previous extraction produced mostly FACTS but missed other dimensions.
Re-read the ENTIRE text and extract ONLY these dimensions:

1. RELATIONSHIPS
2. ARGUMENTS
3. EVENTS

Do not repeat factual triplets unless strictly needed to explain one of these
dimensions. Focus on recall over precision; we will merge/dedupe later.

Use source labels [SRC:doc_id] and set "source" for every item.
Maximum {max_items} items per dimension.

Respond with a single JSON object matching:
{{"facts": [], "relationships": [...], "arguments": [...], "events": [...]}}
No markdown fences.

Text:
{text}"""


# ---------------------------------------------------------------------------
# Source metadata helper
# ---------------------------------------------------------------------------

def _build_source_label(exp: Experience) -> str:
    """Build a source label from experience context metadata."""
    ctx = exp.context or {}
    doc_id = ctx.get("document_id", "")
    title = ctx.get("title", "")
    section = ctx.get("section_title", "")
    if doc_id and title:
        label = f"{doc_id} ({title})"
    elif doc_id:
        label = doc_id
    elif title:
        label = title
    else:
        label = str(exp.id)
    if section:
        label += f" § {section}"
    return label


def _build_source_meta(exp: Experience) -> dict[str, Any]:
    """Build a source metadata dict from experience context."""
    ctx = exp.context or {}
    meta: dict[str, Any] = {
        "experience_id": str(exp.id),
        "occurred_at": exp.occurred_at.isoformat(),
    }
    for key in ("document_id", "title", "section_title", "section_index",
                "format", "source_type", "file_path"):
        if key in ctx:
            meta[key] = ctx[key]
    return meta


def _canonicalize_date_value(
    value: str,
    *,
    reference_datetime: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return canonical date string + metadata when value is date-like."""
    normalized = normalize_date(value, reference_datetime=reference_datetime)
    if not normalized:
        return value, None
    return normalized.canonical, {
        "raw": normalized.raw,
        "canonical": normalized.canonical,
        "precision": normalized.precision,
        "ambiguous": normalized.ambiguous,
        "confidence": normalized.confidence,
        "relative": normalized.relative,
        "reference_date": normalized.reference_date,
    }


class LLMPatternExtractor:
    """Extracts multi-dimensional patterns from experiences using LLM.

    Set ``cache_dir`` to persist raw LLM extraction results to disk as
    JSON.  This allows replaying extractions without re-running the LLM
    (useful when changing storage backends or re-ingesting beliefs).
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        llm: Any,
        config: ReflectionConfig | None = None,
        cache_dir: str | Path | None = None,
        resolver: "EntityResolver | None" = None,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._config = config or ReflectionConfig()
        self._resolver = resolver
        self._cache_dir: Path | None = None
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Extraction cache dir: %s", self._cache_dir)
        self._reset_extraction_diagnostics()

    def _reset_extraction_diagnostics(self) -> None:
        """Reset per-run extraction diagnostics."""
        self._llm_extraction_calls = 0
        self._limit_hit_chunks = Counter()

    def extraction_diagnostics(self) -> dict[str, Any]:
        """Return extraction diagnostics from the latest run."""
        return {
            "llm_extraction_calls": int(self._llm_extraction_calls),
            "limit_hit_chunks": dict(self._limit_hit_chunks),
            "limit_hit_chunks_total": int(sum(self._limit_hit_chunks.values())),
        }

    def _record_extraction_limits(self, result: "ExtractionResult") -> None:
        """Track likely clipping when dimensions hit max-items caps."""
        self._llm_extraction_calls += 1
        max_items = int(self._config.extraction_max_items or 0)
        if max_items <= 0:
            return

        counts = {
            "facts": len(result.facts),
            "relationships": len(result.relationships),
            "arguments": len(result.arguments),
            "events": len(result.events),
        }
        hit_dims = [name for name, count in counts.items() if count >= max_items]
        if not hit_dims:
            return
        for dim in hit_dims:
            self._limit_hit_chunks[dim] += 1
        logger.warning(
            "Extraction chunk reached max_items cap (%d) for: %s. "
            "Results may be clipped for this chunk.",
            max_items,
            ", ".join(hit_dims),
        )

    @staticmethod
    def _should_retry_missing_dimensions(result: "ExtractionResult") -> bool:
        """Return True when extraction is likely too one-dimensional."""
        if len(result.facts) <= 0:
            return False
        return (
            len(result.relationships) == 0
            and len(result.arguments) == 0
            and len(result.events) == 0
        )

    @staticmethod
    def _merge_extraction_results(
        primary: "ExtractionResult",
        secondary: "ExtractionResult",
    ) -> "ExtractionResult":
        """Merge two extraction payloads and deduplicate by content keys."""
        merged = ExtractionResult(
            facts=list(primary.facts),
            relationships=list(primary.relationships),
            arguments=list(primary.arguments),
            events=list(primary.events),
        )

        seen_facts = {
            (
                str(f.subject).strip().lower(),
                str(f.predicate).strip().lower(),
                str(f.object).strip().lower(),
            )
            for f in merged.facts
        }
        seen_rels = {
            (
                str(r.person1).strip().lower(),
                str(r.relationship).strip().lower(),
                str(r.person2).strip().lower(),
                str(r.context).strip().lower(),
            )
            for r in merged.relationships
        }
        seen_args = {
            (
                str(a.actor).strip().lower(),
                str(a.claim).strip().lower(),
                str(a.rhetoric).strip().lower(),
            )
            for a in merged.arguments
        }
        seen_events = {
            (
                str(e.date).strip().lower(),
                str(e.event).strip().lower(),
                ",".join(str(x).strip().lower() for x in (e.actors or [])),
            )
            for e in merged.events
        }

        for f in secondary.facts:
            key = (
                str(f.subject).strip().lower(),
                str(f.predicate).strip().lower(),
                str(f.object).strip().lower(),
            )
            if key in seen_facts:
                continue
            seen_facts.add(key)
            merged.facts.append(f)

        for r in secondary.relationships:
            key = (
                str(r.person1).strip().lower(),
                str(r.relationship).strip().lower(),
                str(r.person2).strip().lower(),
                str(r.context).strip().lower(),
            )
            if key in seen_rels:
                continue
            seen_rels.add(key)
            merged.relationships.append(r)

        for a in secondary.arguments:
            key = (
                str(a.actor).strip().lower(),
                str(a.claim).strip().lower(),
                str(a.rhetoric).strip().lower(),
            )
            if key in seen_args:
                continue
            seen_args.add(key)
            merged.arguments.append(a)

        for e in secondary.events:
            key = (
                str(e.date).strip().lower(),
                str(e.event).strip().lower(),
                ",".join(str(x).strip().lower() for x in (e.actors or [])),
            )
            if key in seen_events:
                continue
            seen_events.add(key)
            merged.events.append(e)

        return merged

    async def extract_patterns_flat(
        self,
        experiences: list["Experience"],
    ) -> list[Pattern]:
        """Extract patterns from experiences using flat char-based chunking.

        Bypasses the group-based approach entirely.  Documents are
        deduplicated, tagged with source labels, and packed into LLM
        calls by character budget (extraction_max_chars).  This is
        ~2-3x faster per document than group-based extraction because
        the LLM call overhead is amortised across more documents.
        """
        self._enforce_model_policy()
        self._reset_extraction_diagnostics()

        # Deduplicate and tag
        seen_content: set[str] = set()
        tagged_texts: list[str] = []
        exp_map: list["Experience"] = []

        for exp in experiences:
            content = exp.content.strip()
            if len(content) < 50:
                continue
            content_key = content[:200]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            label = _build_source_label(exp)
            tagged_texts.append(f"[SRC:{label}]\n{content}")
            exp_map.append(exp)

        if not tagged_texts:
            logger.info("No texts after filtering %d experiences", len(experiences))
            return []

        logger.info(
            "Flat extraction: %d texts from %d experiences",
            len(tagged_texts), len(experiences),
        )

        # Pack into chunks by character budget
        max_chars = self._config.extraction_max_chars
        all_patterns: list[Pattern] = []
        chunk_texts: list[str] = []
        chunk_exps: list["Experience"] = []
        chunk_chars = 0
        separator = "\n\n---\n\n"
        sep_len = len(separator)

        for text, exp in zip(tagged_texts, exp_map):
            text_len = len(text)
            added_len = text_len + (sep_len if chunk_texts else 0)

            if chunk_chars + added_len > max_chars and chunk_texts:
                # Flush current chunk
                known_context = await self._build_rule_context(chunk_exps)
                all_patterns.extend(
                    await self._extract_chunk(
                        chunk_texts,
                        chunk_exps,
                        known_context=known_context,
                    )
                )
                chunk_texts = []
                chunk_exps = []
                chunk_chars = 0

            chunk_texts.append(text)
            chunk_exps.append(exp)
            chunk_chars += text_len + (sep_len if len(chunk_texts) > 1 else 0)

        # Flush remaining
        if chunk_texts:
            known_context = await self._build_rule_context(chunk_exps)
            all_patterns.extend(
                await self._extract_chunk(
                    chunk_texts,
                    chunk_exps,
                    known_context=known_context,
                )
            )

        return self._deduplicate(all_patterns)

    async def _extract_chunk(
        self,
        chunk_texts: list[str],
        chunk_exps: list["Experience"],
        known_context: str = "",
    ) -> list[Pattern]:
        """Send one chunk of tagged texts to the LLM and return patterns."""
        combined = "\n\n---\n\n".join(chunk_texts)

        source_lookup: dict[str, dict[str, Any]] = {}
        for exp in chunk_exps:
            label = _build_source_label(exp)
            source_lookup[label] = _build_source_meta(exp)

        logger.info(
            "Extracting chunk: %d docs, %d chars",
            len(chunk_texts), len(combined),
        )

        patterns: list[Pattern] = []
        try:
            patterns = await self._extract_from_text(
                combined,
                [e.id for e in chunk_exps],
                source_lookup,
                known_context=known_context,
            )
        except Exception as e:
            if _is_model_unavailable_error(e):
                raise RuntimeError(
                    "Extraction failed: local LLM reports no loaded/available model. "
                    "Load the extraction model (for example `lms load qwen/qwen3-30b-a3b-2507 --identifier qwen3-30b`) "
                    "and retry.",
                ) from e
            logger.warning("LLM extraction failed for chunk: %s", e)
            patterns = []

        if patterns:
            return patterns

        # Retry path: large chunk may fail parsing/format constraints even if
        # the transport call succeeded. Split and retry recursively.
        if len(chunk_texts) <= 1:
            return []

        mid = len(chunk_texts) // 2
        logger.info(
            "Chunk returned empty; retrying with split chunks (%d + %d docs)",
            mid, len(chunk_texts) - mid,
        )
        left = await self._extract_chunk(
            chunk_texts[:mid],
            chunk_exps[:mid],
            known_context=known_context,
        )
        right = await self._extract_chunk(
            chunk_texts[mid:],
            chunk_exps[mid:],
            known_context=known_context,
        )
        return left + right

    async def extract_patterns(
        self,
        groups: list[ExperienceGroup],
        max_groups: int = 3,
    ) -> list[Pattern]:
        """Extract patterns from experience groups using the LLM.

        Legacy group-based path.  Prefer extract_patterns_flat() for
        higher throughput.

        Args:
            groups: Experience groups to process.
            max_groups: Maximum groups to process per call to bound memory
                usage.  Overlapping grouping strategies can create many more
                groups than experiences; capping prevents runaway extraction.
        """
        self._enforce_model_policy()
        self._reset_extraction_diagnostics()
        all_patterns: list[Pattern] = []

        if len(groups) > max_groups:
            logger.info(
                "Capping groups from %d to %d to bound memory",
                len(groups), max_groups,
            )
            groups = groups[:max_groups]

        for group in groups:
            experiences = await self._fetch_experiences(group.experiences)
            logger.info(
                "Group %s: %d exp IDs -> %d fetched",
                group.id, len(group.experiences), len(experiences),
            )
            if not experiences:
                continue

            # Combine experience content (deduplicated), preserving source labels
            seen_content: set[str] = set()
            tagged_texts: list[str] = []
            exp_map: list[Experience] = []  # parallel to tagged_texts

            for exp in experiences:
                content = exp.content.strip()
                if len(content) < 50:
                    logger.debug("Skipping short content (%d chars)", len(content))
                    continue
                content_key = content[:200]
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)

                # Tag each text with its source document
                label = _build_source_label(exp)
                tagged_texts.append(f"[SRC:{label}]\n{content}")
                exp_map.append(exp)

            if not tagged_texts:
                logger.info("Group %s: no texts after filtering", group.id)
                continue

            logger.info("Group %s: %d texts to process", group.id, len(tagged_texts))

            # Process in chunks to stay within context limits
            chunk_size = self._config.extraction_chunk_size
            max_chars = self._config.extraction_max_chars
            for i in range(0, len(tagged_texts), chunk_size):
                chunk_texts = tagged_texts[i:i + chunk_size]
                chunk_exps = exp_map[i:i + chunk_size]
                combined = "\n\n---\n\n".join(chunk_texts)

                if len(combined) > max_chars:
                    combined = combined[:max_chars]

                # Build a lookup: source label -> experience metadata
                source_lookup: dict[str, dict[str, Any]] = {}
                for exp in chunk_exps:
                    label = _build_source_label(exp)
                    source_lookup[label] = _build_source_meta(exp)

                # Rules-first context from deterministic entity resolver.
                known_context = await self._build_rule_context(chunk_exps)

                try:
                    patterns = await self._extract_from_text(
                        combined,
                        [e.id for e in chunk_exps],
                        source_lookup,
                        known_context=known_context,
                    )
                    all_patterns.extend(patterns)
                except Exception as e:
                    if _is_model_unavailable_error(e):
                        raise RuntimeError(
                            "Extraction failed: local LLM reports no loaded/available model. "
                            "Load the extraction model and retry.",
                        ) from e
                    logger.warning(
                        "LLM extraction failed for group %s: %s", group.id, e
                    )

        return self._deduplicate(all_patterns)

    def _enforce_model_policy(self) -> None:
        """Enforce extraction model quality policy."""
        allow_weak = bool(getattr(self._config, "allow_weak_extraction_model", False))
        model_name = ensure_extraction_model_allowed(
            self._llm,
            allow_weak_extraction_model=allow_weak,
        )
        if model_name:
            logger.debug("Extraction model policy check passed for model=%s", model_name)

    async def _build_rule_context(self, experiences: list["Experience"]) -> str:
        """Build deterministic extraction hints from regex/entity rules."""
        if self._resolver is None or not experiences:
            return ""

        max_docs = 12
        max_refs_per_doc = 15
        max_chars = 4000
        lines: list[str] = []
        docs_with_hits = 0
        total_refs = 0

        for exp in experiences[:max_docs]:
            content = (exp.content or "").strip()
            if not content:
                continue
            try:
                resolved = await self._resolver.resolve(content)
            except Exception:
                continue
            if not resolved.resolved:
                continue

            docs_with_hits += 1
            refs: list[str] = []
            seen: set[tuple[str, str]] = set()
            for ref in resolved.resolved:
                key = (ref.entity_type, ref.canonical_id)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(f"{ref.entity_type}:{ref.canonical_id}")
                total_refs += 1
                if len(refs) >= max_refs_per_doc:
                    break

            if refs:
                label = _build_source_label(exp)
                lines.append(f"- [SRC:{label}] {', '.join(refs)}")

        if not lines:
            return ""

        header = (
            "Deterministic entity extraction hints (rules-first). "
            "Use as grounding and still extract non-entity facts/relations/events."
        )
        context = header + "\n" + "\n".join(lines)
        if len(context) > max_chars:
            context = context[:max_chars]

        logger.debug(
            "Rule context prepared: %d docs, %d refs, %d chars",
            docs_with_hits,
            total_refs,
            len(context),
        )
        return context

    async def _extract_from_text(
        self,
        text: str,
        evidence_ids: list[UUID],
        source_lookup: dict[str, dict[str, Any]],
        known_context: str = "",
    ) -> list[Pattern]:
        """Run the LLM extraction and convert results to Pattern objects."""
        max_items = self._config.extraction_max_items
        max_tokens = self._config.extraction_max_tokens
        prompt = _EXTRACTION_PROMPT.format(text=text, max_items=max_items or "no limit")

        # Inject known context before the text section
        def _inject_context(base_prompt: str) -> str:
            if not known_context:
                return base_prompt
            return base_prompt.replace(
                "\nText:\n",
                f"\n{known_context}\n\nText:\n",
            )

        prompt = _inject_context(prompt)

        logger.info("Calling LLM for extraction (%d chars of text)", len(text))
        try:
            result = await self._llm.generate_structured(
                prompt,
                ExtractionResult,
                max_tokens=max_tokens,
            )
            logger.info(
                "Structured extraction: %d facts, %d rels, %d args, %d events",
                len(result.facts), len(result.relationships),
                len(result.arguments), len(result.events),
            )
        except Exception as e:
            if _is_model_unavailable_error(e):
                raise RuntimeError(
                    "Extraction failed: local LLM reports no loaded/available model. "
                    "Load the extraction model and retry.",
                ) from e
            logger.info("Structured extraction failed (%s), falling back to raw", e)
            # Fallback: raw generate + parse
            try:
                raw = await self._llm.generate(
                    prompt, max_tokens=max_tokens, temperature=0.3
                )
                logger.info("Raw LLM response (%d chars): %.200s", len(raw), raw)
                result = self._parse_raw(raw)
            except Exception as e2:
                if _is_model_unavailable_error(e2):
                    raise RuntimeError(
                        "Extraction failed: local LLM reports no loaded/available model. "
                        "Load the extraction model and retry.",
                    ) from e2
                logger.warning("LLM extraction failed completely: %s", e2)
                return []

        self._record_extraction_limits(result)
        if self._should_retry_missing_dimensions(result):
            retry_prompt = _inject_context(
                _EXTRACTION_MISSING_DIMENSIONS_PROMPT.format(
                    text=text,
                    max_items=max_items or "no limit",
                )
            )
            logger.info(
                "Primary extraction is fact-only; retrying missing dimensions "
                "(relationships/arguments/events).",
            )
            try:
                retry_result = await self._llm.generate_structured(
                    retry_prompt,
                    ExtractionResult,
                    max_tokens=max_tokens,
                )
                logger.info(
                    "Missing-dimension retry: %d rels, %d args, %d events",
                    len(retry_result.relationships),
                    len(retry_result.arguments),
                    len(retry_result.events),
                )
                self._record_extraction_limits(retry_result)
                result = self._merge_extraction_results(result, retry_result)
            except Exception as retry_exc:
                if _is_model_unavailable_error(retry_exc):
                    raise RuntimeError(
                        "Extraction failed: local LLM reports no loaded/available model. "
                        "Load the extraction model and retry.",
                    ) from retry_exc
                logger.info("Missing-dimension retry failed (%s)", retry_exc)

        # Cache raw extraction to disk for replay
        self._cache_extraction(result, evidence_ids, source_lookup)

        patterns: list[Pattern] = []
        evidence = evidence_ids[:5]

        # Helper: resolve source label to metadata dict
        def _resolve_source(source_label: str) -> dict[str, Any]:
            """Find best-matching source metadata for a label from the LLM."""
            if not source_label:
                return {}
            # Exact match
            if source_label in source_lookup:
                return source_lookup[source_label]
            # Partial match (LLM may abbreviate)
            sl = source_label.lower()
            for key, meta in source_lookup.items():
                if sl in key.lower() or key.lower() in sl:
                    return meta
            # Match by document_id substring
            for key, meta in source_lookup.items():
                doc_id = meta.get("document_id", "")
                if doc_id and (sl in doc_id.lower() or doc_id.lower() in sl):
                    return meta
            return {}

        # --- Facts ---
        for f in (result.facts if hasattr(result, "facts") else []):
            s, p, o = _get(f, "subject"), _get(f, "predicate"), _get(f, "object")
            if not s or not p or not o:
                continue
            src = _resolve_source(_get(f, "source", ""))
            reference_datetime = ""
            if isinstance(src, dict):
                reference_datetime = str(src.get("occurred_at") or "")
            subj_norm, subj_date_meta = _canonicalize_date_value(
                s,
                reference_datetime=reference_datetime or None,
            )
            obj_norm, obj_date_meta = _canonicalize_date_value(
                o,
                reference_datetime=reference_datetime or None,
            )
            fact_ctx: dict[str, Any] = {}
            if subj_date_meta:
                fact_ctx["subject_date"] = subj_date_meta
            if obj_date_meta:
                fact_ctx["object_date"] = obj_date_meta
            if src:
                fact_ctx["source"] = src
            patterns.append(Pattern(
                type=PatternType.FACT,
                description=f"{subj_norm} {p} {obj_norm}",
                evidence=list(evidence),
                confidence=_clamp(_get(f, "confidence", 0.7)),
                subject=subj_norm,
                predicate=p,
                object=obj_norm,
                context=fact_ctx,
            ))

        # --- Relationships ---
        for r in (result.relationships if hasattr(result, "relationships") else []):
            p1 = _get(r, "person1")
            rel = _get(r, "relationship")
            p2 = _get(r, "person2")
            if not p1 or not rel or not p2:
                continue
            ctx = _get(r, "context", "")
            src = _resolve_source(_get(r, "source", ""))
            desc = f"{p1} {rel} {p2}"
            if ctx:
                desc += f" ({ctx})"
            pattern_ctx: dict[str, Any] = {}
            if ctx:
                pattern_ctx["relationship_context"] = ctx
            if src:
                pattern_ctx["source"] = src
            patterns.append(Pattern(
                type=PatternType.RELATIONSHIP,
                description=desc,
                evidence=list(evidence),
                confidence=_clamp(_get(r, "confidence", 0.7)),
                subject=p1,
                predicate=rel,
                object=p2,
                context={
                    **pattern_ctx,
                    "relationship_direction": "forward",
                    "bidirectional_edge": True,
                },
            ))

            # Emit reverse edge for bidirectional graph connectivity.
            # Keep original predicate for now and preserve direction metadata.
            if p1 != p2:
                rev_desc = f"{p2} {rel} {p1}"
                if ctx:
                    rev_desc += f" ({ctx})"
                patterns.append(Pattern(
                    type=PatternType.RELATIONSHIP,
                    description=rev_desc,
                    evidence=list(evidence),
                    confidence=_clamp(_get(r, "confidence", 0.7)),
                    subject=p2,
                    predicate=rel,
                    object=p1,
                    context={
                        **pattern_ctx,
                        "relationship_direction": "reverse",
                        "relationship_source_predicate": rel,
                        "bidirectional_edge": True,
                    },
                ))

        # --- Arguments ---
        for a in (result.arguments if hasattr(result, "arguments") else []):
            claim = _get(a, "claim")
            ev = _get(a, "evidence")
            if not claim:
                continue
            rhetoric = _get(a, "rhetoric", "logos")
            actor = _get(a, "actor", "")
            src = _resolve_source(_get(a, "source", ""))
            desc = f"[{rhetoric.upper()}] {actor + ': ' if actor else ''}{claim}"
            patterns.append(Pattern(
                type=PatternType.ARGUMENT,
                description=desc,
                evidence=list(evidence),
                confidence=_clamp(_get(a, "confidence", 0.7)),
                subject=actor or "unknown",
                predicate="argues",
                object=claim,
                context={
                    "evidence": ev or "",
                    "rhetoric": rhetoric,
                    "actor": actor,
                    "source": src,
                },
            ))

        # --- Timeline events ---
        for ev in (result.events if hasattr(result, "events") else []):
            date = _get(ev, "date")
            event = _get(ev, "event")
            if not event:
                continue
            actors = ev.actors if hasattr(ev, "actors") else _get(ev, "actors", [])
            if isinstance(actors, str):
                actors = [actors]
            sig = _get(ev, "significance", "")
            src = _resolve_source(_get(ev, "source", ""))
            reference_datetime = ""
            if isinstance(src, dict):
                reference_datetime = str(src.get("occurred_at") or "")
            normalized_date = normalize_date(
                date or "",
                reference_datetime=reference_datetime or None,
            )
            date_display = normalized_date.canonical if normalized_date else (date or "?")
            desc = f"[{date_display}] {event}"
            if sig:
                desc += f" — {sig}"
            event_ctx: dict[str, Any] = {
                "date": date_display if date else "",
                "actors": actors,
                "significance": sig,
                "source": src,
            }
            if normalized_date:
                event_ctx["date_normalized"] = {
                    "raw": normalized_date.raw,
                    "canonical": normalized_date.canonical,
                    "precision": normalized_date.precision,
                    "ambiguous": normalized_date.ambiguous,
                    "confidence": normalized_date.confidence,
                    "relative": normalized_date.relative,
                    "reference_date": normalized_date.reference_date,
                }
            patterns.append(Pattern(
                type=PatternType.TIMELINE_EVENT,
                description=desc,
                evidence=list(evidence),
                confidence=_clamp(_get(ev, "confidence", 0.7)),
                subject=", ".join(actors) if actors else "unknown",
                predicate="event",
                object=event,
                context=event_ctx,
            ))

        return patterns

    # ------------------------------------------------------------------
    # Entity-aware context gathering
    # ------------------------------------------------------------------

    async def _gather_known_context(self, text: str) -> str:
        """Query existing beliefs about entities mentioned in this text.

        Extracts potential entity names (capitalized phrases) from the text,
        queries the knowledge graph for known facts about them, and builds
        a context string so the LLM can focus on NEW information.

        Returns:
            A context string like "Previously established knowledge: ..."
            or empty string if no prior knowledge found.
        """
        # Extract potential entity names from text
        entity_candidates = set()
        # Multi-word capitalized phrases (proper nouns)
        for match in re.finditer(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text
        ):
            entity_candidates.add(match.group(1))
        # Single capitalized words that are likely names (3+ chars)
        for match in re.finditer(r'\b([A-Z][a-z]{2,})\b', text):
            word = match.group(1)
            # Skip common English words
            if word.lower() not in _COMMON_WORDS:
                entity_candidates.add(word)

        if not entity_candidates:
            return ""

        # Query existing beliefs for each entity (limit to top 15)
        known_facts: list[str] = []
        queried = 0
        for entity in sorted(entity_candidates)[:15]:
            try:
                beliefs = await self._memory._backend.get_beliefs_by_entity(
                    entity
                )
                for b in beliefs[:3]:  # Max 3 facts per entity
                    if b.triplet:
                        fact = (
                            f"{b.triplet.subject} {b.triplet.predicate} "
                            f"{b.triplet.object}"
                        )
                        if fact not in known_facts:
                            known_facts.append(fact)
                queried += 1
            except Exception:
                pass

        if not known_facts:
            return ""

        # Build context string
        context = (
            "Previously established knowledge (build on this, "
            "focus on NEW information not listed here):\n"
        )
        for fact in known_facts[:20]:  # Cap at 20 known facts
            context += f"  - {fact}\n"

        logger.debug(
            "Entity context: %d entities queried, %d known facts",
            queried, len(known_facts),
        )
        return context

    # ------------------------------------------------------------------
    # Extraction cache (disk persistence for replay)
    # ------------------------------------------------------------------

    def _cache_extraction(
        self,
        result: ExtractionResult,
        evidence_ids: list[UUID],
        source_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Persist a single LLM extraction to the cache directory."""
        if self._cache_dir is None:
            return
        try:
            cache_entry = {
                "extraction": result.model_dump(),
                "evidence_ids": [str(eid) for eid in evidence_ids],
                "source_lookup": source_lookup,
            }
            # Use a monotonic filename so entries are ordered
            existing = list(self._cache_dir.glob("extraction_*.json"))
            idx = len(existing)
            path = self._cache_dir / f"extraction_{idx:06d}.json"
            path.write_text(json.dumps(cache_entry, indent=2))
            logger.info("Cached extraction to %s", path.name)
        except Exception as e:
            logger.warning("Failed to cache extraction: %s", e)

    @staticmethod
    def load_cache(cache_dir: str | Path) -> list[dict[str, Any]]:
        """Load all cached extraction entries from disk.

        Returns a list of dicts, each with keys:
          - ``extraction``: dict matching ExtractionResult schema
          - ``evidence_ids``: list of UUID strings
          - ``source_lookup``: dict of source label → metadata

        Use with ``replay_cached_entry`` to convert back to Patterns.
        """
        cache_path = Path(cache_dir)
        entries: list[dict[str, Any]] = []
        for path in sorted(cache_path.glob("extraction_*.json")):
            entries.append(json.loads(path.read_text()))
        return entries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_raw(self, raw: str) -> ExtractionResult:
        """Parse raw LLM text output into ExtractionResult."""
        raw = raw.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        # Find JSON object
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group())
                return ExtractionResult.model_validate(parsed)
            except (json.JSONDecodeError, Exception):
                pass

        # Try to find a JSON array (old format fallback)
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            try:
                items = json.loads(match.group())
                return ExtractionResult(facts=items)
            except (json.JSONDecodeError, Exception):
                pass

        return ExtractionResult()

    def _deduplicate(self, patterns: list[Pattern]) -> list[Pattern]:
        """Deduplicate patterns by type+subject+predicate+object key."""
        seen: dict[str, Pattern] = {}

        for p in patterns:
            key = (
                f"{p.type.value}|"
                f"{(p.subject or '').lower()}|"
                f"{(p.predicate or '').lower()}|"
                f"{(p.object or '').lower()}"
            )
            if key in seen:
                existing = seen[key]
                if p.confidence > existing.confidence:
                    existing.confidence = p.confidence
                existing.occurrences += 1
                for eid in p.evidence:
                    if eid not in existing.evidence and len(existing.evidence) < 10:
                        existing.evidence.append(eid)
                # Merge source contexts
                if p.context.get("source") and not existing.context.get("source"):
                    existing.context["source"] = p.context["source"]
            else:
                seen[key] = p

        return list(seen.values())

    async def _fetch_experiences(
        self, experience_ids: list[UUID]
    ) -> list[Experience]:
        """Fetch full experience objects."""
        experiences = []
        for eid in experience_ids:
            exp = await self._memory.get_experience(eid)
            if exp:
                experiences.append(exp)
        return experiences


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get(obj: Any, attr: str, default: Any = "") -> Any:
    """Get attribute from Pydantic model or dict."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _clamp(val: Any, lo: float = 0.1, hi: float = 0.95) -> float:
    """Clamp a confidence value."""
    try:
        return max(lo, min(hi, float(val)))
    except (TypeError, ValueError):
        return 0.7
