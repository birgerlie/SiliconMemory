"""LLM-based procedural memory detection.

Scans beliefs and experiences for procedural knowledge — sequences of
steps, how-to instructions, workflows — and extracts structured
Procedure objects.
"""

from __future__ import annotations

from datetime import timezone
import json
import logging
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.types import Procedure, Source, SourceType

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)

_DETECT_FROM_BELIEFS_PROMPT = """\
Analyze these beliefs/facts for procedural knowledge — sequences of steps,
how-to instructions, workflows, or repeated patterns.

Beliefs:
{beliefs_text}

For each procedure found, extract:
- name: Short descriptive name
- trigger: When/why to use this procedure
- steps: Ordered list of concrete steps
- description: Brief summary

Return JSON: {{"procedures": [{{"name": "...", "trigger": "...", "steps": ["..."], "description": "..."}}]}}
Return empty list if no procedures found."""

_DETECT_FROM_EXPERIENCES_PROMPT = """\
These are sequences of events from multiple sessions. Identify any repeated
patterns or workflows that appear across sessions.

{sessions_text}

Extract procedures that appear in 2+ sessions.
Return JSON: {{"procedures": [{{"name": "...", "trigger": "...", "steps": ["..."], "description": "..."}}]}}
Return empty list if no procedures found."""

_DEDUP_PROMPT = """\
Does this new procedure duplicate any existing procedure?

New procedure:
  Name: {new_name}
  Steps: {new_steps}

Existing procedures:
{existing_text}

Respond with JSON: {{"is_duplicate": true/false, "duplicate_of": "<name or null>"}}"""


class ProcedureDetector:
    """Detects procedural knowledge from beliefs and experiences using LLM."""

    def __init__(self, memory: "SiliconMemory", llm: Any) -> None:
        self._memory = memory
        self._llm = llm

    async def detect_from_beliefs(self, limit: int = 100) -> list[Procedure]:
        """Use LLM to scan beliefs for procedural knowledge."""
        beliefs = await self._memory.query_beliefs(
            query="how to steps process deploy build run",
            limit=limit,
            min_confidence=0.3,
        )
        if not beliefs:
            return []

        beliefs_text = "\n".join(
            f"- {b.content or (b.triplet.as_text() if b.triplet else '')}"
            for b in beliefs[:50]
        )

        prompt = _DETECT_FROM_BELIEFS_PROMPT.format(beliefs_text=beliefs_text)
        return await self._extract_procedures(prompt)

    async def detect_from_experiences(self, limit: int = 200) -> list[Procedure]:
        """Use LLM to find repeated action sequences in experiences."""
        # Group experiences by session_id, then by document_id/day fallback
        sessions: dict[str, list[str]] = {}
        try:
            experiences = await self._memory.get_recent_experiences(hours=24 * 365, limit=limit)
            for exp in experiences:
                sid = (exp.session_id or "").strip()
                if not sid:
                    doc_id = str((exp.context or {}).get("document_id") or "").strip()
                    if doc_id:
                        sid = f"doc:{doc_id}"
                if not sid:
                    day = exp.occurred_at.astimezone(timezone.utc).date().isoformat()
                    sid = f"day:{day}"
                if sid not in sessions:
                    sessions[sid] = []
                sessions[sid].append(exp.content)
        except Exception:
            return []

        if len(sessions) < 2:
            return []

        sessions_text = ""
        for i, (sid, contents) in enumerate(list(sessions.items())[:10]):
            sessions_text += f"\nSession {i + 1} ({sid}):\n"
            for c in contents[:20]:
                sessions_text += f"  - {c[:200]}\n"

        prompt = _DETECT_FROM_EXPERIENCES_PROMPT.format(sessions_text=sessions_text)
        return await self._extract_procedures(prompt)

    async def detect_and_commit(self) -> int:
        """Detect, deduplicate, and commit new procedures. Returns count."""
        # Prefer deterministic timeline-derived procedures first for speed/cost.
        from_timeline = await self._derive_from_timeline()
        from_predicates = await self._derive_from_repeated_predicates()
        if from_timeline:
            from_beliefs: list[Procedure] = []
            from_experiences: list[Procedure] = []
        else:
            from_beliefs = await self.detect_from_beliefs()
            from_experiences = await self.detect_from_experiences()

        candidates = from_beliefs + from_experiences + from_timeline + from_predicates
        if not candidates:
            return 0

        # Get existing procedures for deduplication
        existing = await self._memory.find_applicable_procedures(
            "all procedures", limit=50,
        )

        committed = 0
        for candidate in candidates:
            if existing:
                is_dup = await self._is_duplicate(candidate, existing)
                if is_dup:
                    continue

            try:
                await self._memory.commit_procedure(candidate)
                committed += 1
                existing.append(candidate)
            except Exception as e:
                logger.debug("Failed to commit procedure: %s", e)

        logger.info("Procedure detection: %d candidates, %d committed", len(candidates), committed)
        return committed

    async def _derive_from_timeline(self, limit: int = 20) -> list[Procedure]:
        """Deterministically derive procedures from timeline belief triples."""
        backend = self._memory._backend
        try:
            triples = backend._query_triples(k=5000)
        except Exception:
            return []

        by_subject: dict[str, list[tuple[str, str]]] = {}
        seen_triplets: set[tuple[str, str, str]] = set()
        for t in triples:
            subject = str(getattr(t, "subject", "") or "").strip()
            predicate = str(getattr(t, "predicate", "") or "").strip()
            date_value = str(getattr(t, "object_value", "") or "").strip()
            if not subject or predicate != "occurred_on" or not date_value:
                continue
            key = (subject, predicate, date_value)
            if key in seen_triplets:
                continue
            seen_triplets.add(key)

            metadata = getattr(t, "metadata", {}) or {}
            event_text = ""
            source_meta = metadata.get("source_metadata")
            if isinstance(source_meta, dict):
                event_text = str(source_meta.get("event_text") or "").strip()
            if not event_text:
                event_text = str(metadata.get("content") or "").strip()
            if not event_text:
                event_text = f"Event on {date_value}"

            by_subject.setdefault(subject, []).append((date_value, event_text))

        procedures: list[Procedure] = []
        for subject, items in by_subject.items():
            if len(items) < 2:
                continue
            ordered = sorted(items, key=lambda x: x[0])
            steps = [f"{date}: {event}" for date, event in ordered[:8]]
            procedures.append(
                Procedure(
                    id=uuid4(),
                    name=f"{subject} timeline playbook",
                    description=f"Derived procedural sequence from dated events for {subject}.",
                    trigger=f"When reasoning about sequence/progression for {subject}",
                    steps=steps,
                    confidence=0.55,
                    source=Source(
                        id="procedure_detection_timeline",
                        type=SourceType.REFLECTION,
                        reliability=0.55,
                    ),
                    tags={"detected", "timeline_derived"},
                ),
            )
            if len(procedures) >= limit:
                break

        return procedures

    async def _derive_from_repeated_predicates(self, limit: int = 10) -> list[Procedure]:
        """Derive generic procedures from repeated predicate patterns per subject."""
        backend = self._memory._backend
        try:
            triples = backend._query_triples(k=10_000)
        except Exception:
            return []

        by_subject: dict[str, dict[str, int]] = {}
        for t in triples:
            metadata = getattr(t, "metadata", {}) or {}
            if hasattr(backend, "_can_access") and not backend._can_access(  # noqa: SLF001
                metadata,
                _external_id=str(getattr(t, "external_id", "") or ""),
            ):
                continue
            subject = str(getattr(t, "subject", "") or "").strip()
            predicate = str(getattr(t, "predicate", "") or "").strip()
            if not subject or not predicate:
                continue
            if predicate in {"occurred_on", "event", "argues"}:
                continue
            by_subject.setdefault(subject, {})
            by_subject[subject][predicate] = by_subject[subject].get(predicate, 0) + 1

        procedures: list[Procedure] = []
        for subject, counts in by_subject.items():
            repeated = [(pred, n) for pred, n in counts.items() if n >= 2]
            if len(repeated) < 2:
                continue
            repeated.sort(key=lambda x: x[1], reverse=True)
            steps = [
                f"Assess whether {subject} {pred} (observed {n}x)"
                for pred, n in repeated[:8]
            ]
            procedures.append(
                Procedure(
                    id=uuid4(),
                    name=f"{subject} recurring-claim review",
                    description=(
                        "Deterministic procedure derived from recurring predicate patterns "
                        f"for {subject}."
                    ),
                    trigger=f"When reviewing repeated claims about {subject}",
                    steps=steps,
                    confidence=0.5,
                    source=Source(
                        id="procedure_detection_predicates",
                        type=SourceType.REFLECTION,
                        reliability=0.5,
                    ),
                    tags={"detected", "predicate_derived"},
                ),
            )
            if len(procedures) >= limit:
                break
        return procedures

    async def _extract_procedures(self, prompt: str) -> list[Procedure]:
        """Call LLM and parse procedure results."""
        try:
            response = await self._llm_call(
                "You extract procedural knowledge from text. Return valid JSON only.",
                prompt,
            )
            parsed = self._parse_json(response)
            if not parsed or "procedures" not in parsed:
                return []

            procedures = []
            for p in parsed["procedures"]:
                raw_steps = p.get("steps", [])
                if isinstance(raw_steps, str):
                    # LLM returned a string instead of a list — split on common delimiters
                    raw_steps = [s.strip() for s in raw_steps.replace(" → ", "\n").replace("→", "\n").replace(";", "\n").split("\n") if s.strip()]
                elif not isinstance(raw_steps, list):
                    raw_steps = []
                proc = Procedure(
                    id=uuid4(),
                    name=p.get("name", "Unnamed"),
                    description=p.get("description", ""),
                    trigger=p.get("trigger", ""),
                    steps=raw_steps,
                    confidence=0.6,
                    source=Source(
                        id="procedure_detection",
                        type=SourceType.REFLECTION,
                        reliability=0.6,
                    ),
                    tags={"detected", "llm_generated"},
                )
                procedures.append(proc)
            return procedures
        except Exception as e:
            logger.debug("Procedure extraction failed: %s", e)
            return []

    async def _is_duplicate(
        self, candidate: Procedure, existing: list[Procedure],
    ) -> bool:
        """Use LLM to check if candidate duplicates an existing procedure."""
        if not existing:
            return False

        existing_text = "\n".join(
            f"- {p.name}: {' → '.join(p.steps[:5])}"
            for p in existing[:10]
        )

        prompt = _DEDUP_PROMPT.format(
            new_name=candidate.name,
            new_steps=" → ".join(candidate.steps),
            existing_text=existing_text,
        )

        try:
            response = await self._llm_call(
                "You check for duplicate procedures. Return valid JSON only.",
                prompt,
            )
            parsed = self._parse_json(response)
            return parsed.get("is_duplicate", False) if parsed else False
        except Exception:
            return False

    async def _llm_call(self, system: str, user: str) -> str:
        """Make an LLM call, handling both scheduler and direct provider."""
        prompt = f"{system}\n\n{user}"
        if hasattr(self._llm, "complete"):
            return await self._llm.complete(
                prompt, system=system, temperature=0.3, max_tokens=1024,
            )
        elif hasattr(self._llm, "generate"):
            return await self._llm.generate(
                prompt, max_tokens=1024, temperature=0.3,
            )
        raise TypeError(f"Unknown LLM type: {type(self._llm)}")

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
        return None
