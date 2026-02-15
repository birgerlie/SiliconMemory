"""LLM-powered multi-dimensional knowledge extractor for the reflection engine.

Extracts structured knowledge from experience text across four dimensions:
1. Facts — subject-predicate-object triplets
2. Relationships — person-to-person and person-to-institution links
3. Arguments — legal/logical arguments with rhetoric classification (pathos/logos/ethos)
4. Timeline events — dated events with actors and significance

Each extraction is grounded to its source document for provenance tracking.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel

from silicon_memory.core.types import Experience
from silicon_memory.reflection.types import (
    ExperienceGroup,
    Pattern,
    PatternType,
    ReflectionConfig,
)

if TYPE_CHECKING:
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
    "court", "case", "order", "motion", "filed", "district", "circuit",
    "pursuant", "section", "evidence", "trial", "defendant", "plaintiff",
    "government", "united", "states", "appeal", "judgment", "opinion",
    "also", "upon", "whether", "therefore", "however", "furthermore",
    "moreover", "thus", "accordingly", "nevertheless",
})


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
    relationship: str  # e.g. "employer of", "co-conspirator with"
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
You are an expert analyst extracting structured knowledge from court documents. \
Read the ENTIRE text below and extract knowledge across four dimensions. \
Be thorough — cover all sections, not just the first paragraph.

Each text segment is labelled with a source tag like [SRC:doc_id]. \
For every extracted item, set the "source" field to the doc_id of the \
document it came from. If a fact spans multiple sources, pick the primary one.

1. FACTS: Subject-predicate-object triplets.
   - People and roles (e.g. "Ghislaine Maxwell" / "was convicted of" / "sex trafficking")
   - Legal facts (e.g. "Case 22-1426" / "is an appeal in" / "Second Circuit")
   - Institutional facts (e.g. "SDNY" / "prosecuted" / "Maxwell")
   - Sentences/outcomes (e.g. "Maxwell" / "sentenced to" / "240 months")
   - Financial facts (e.g. "Epstein" / "worth" / "$577 million")
   - Locations and properties (e.g. "Epstein" / "owned" / "71st Street mansion")

2. RELATIONSHIPS: Connections between people or organisations.
   - person1, relationship type, person2, context, source
   - e.g. person1="Maxwell", relationship="associate of", person2="Epstein", context="1990s-2000s"

3. ARGUMENTS: Legal or logical arguments made in the text.
   - claim: the core assertion
   - evidence: supporting reasoning or cited evidence
   - rhetoric: classify as "pathos" (emotional appeal), "logos" (logical/legal reasoning), or "ethos" (appeal to authority/credibility)
   - actor: who makes the argument
   - source: which document

4. EVENTS: Dated occurrences.
   - date, what happened, who was involved, why it matters, source

Rules:
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
    meta: dict[str, Any] = {"experience_id": str(exp.id)}
    for key in ("document_id", "title", "section_title", "section_index",
                "format", "source_type", "file_path"):
        if key in ctx:
            meta[key] = ctx[key]
    return meta


class LLMPatternExtractor:
    """Extracts multi-dimensional patterns from experiences using LLM."""

    def __init__(
        self,
        memory: "SiliconMemory",
        llm: Any,
        config: ReflectionConfig | None = None,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._config = config or ReflectionConfig()

    async def extract_patterns(
        self,
        groups: list[ExperienceGroup],
    ) -> list[Pattern]:
        """Extract patterns from experience groups using the LLM."""
        all_patterns: list[Pattern] = []

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

                # Query existing knowledge about entities in this chunk
                known_context = await self._gather_known_context(combined)

                try:
                    patterns = await self._extract_from_text(
                        combined,
                        [e.id for e in chunk_exps],
                        source_lookup,
                        known_context=known_context,
                    )
                    all_patterns.extend(patterns)
                except Exception as e:
                    logger.warning(
                        "LLM extraction failed for group %s: %s", group.id, e
                    )

        return self._deduplicate(all_patterns)

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
        prompt = _EXTRACTION_PROMPT.format(text=text, max_items=max_items)

        # Inject known context before the text section
        if known_context:
            prompt = prompt.replace(
                "\nText:\n",
                f"\n{known_context}\n\nText:\n",
            )

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
            logger.info("Structured extraction failed (%s), falling back to raw", e)
            # Fallback: raw generate + parse
            try:
                raw = await self._llm.generate(
                    prompt, max_tokens=max_tokens, temperature=0.3
                )
                logger.info("Raw LLM response (%d chars): %.200s", len(raw), raw)
                result = self._parse_raw(raw)
            except Exception as e2:
                logger.warning("LLM extraction failed completely: %s", e2)
                return []

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
            patterns.append(Pattern(
                type=PatternType.FACT,
                description=f"{s} {p} {o}",
                evidence=list(evidence),
                confidence=_clamp(_get(f, "confidence", 0.7)),
                subject=s,
                predicate=p,
                object=o,
                context={"source": src} if src else {},
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
                context=pattern_ctx,
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
            desc = f"[{date or '?'}] {event}"
            if sig:
                desc += f" — {sig}"
            patterns.append(Pattern(
                type=PatternType.TIMELINE_EVENT,
                description=desc,
                evidence=list(evidence),
                confidence=_clamp(_get(ev, "confidence", 0.7)),
                subject=", ".join(actors) if actors else "unknown",
                predicate="event",
                object=event,
                context={
                    "date": date or "",
                    "actors": actors,
                    "significance": sig,
                    "source": src,
                },
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
