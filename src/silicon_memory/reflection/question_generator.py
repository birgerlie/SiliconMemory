"""LLM-based question generation for active learning.

Identifies knowledge gaps from uncertain/contested beliefs and
generates investigative questions to fill them.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from silicon_memory.core.utils import utc_now

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)

_QUESTION_PROMPT = """\
You are analyzing a knowledge base to find gaps and uncertainties.

Uncertain beliefs (high entropy — we're not sure how confident to be):
{uncertain_beliefs}

Contested beliefs (conflicting evidence):
{contested_beliefs}

Recent contradictions:
{contradictions}

Generate questions that would help resolve these uncertainties. Each question
should target a specific knowledge gap and be answerable from external sources.

Return JSON: {{"questions": [{{"question": "...", "target_belief_id": "...", "priority": "high/medium/low", "context": "why this matters"}}]}}
Return empty list if no meaningful questions can be generated."""


@dataclass
class Question:
    """A generated question targeting a knowledge gap."""

    id: UUID = field(default_factory=uuid4)
    question: str = ""
    target_belief_id: str | None = None
    priority: str = "medium"
    context: str = ""


class QuestionGenerator:
    """Generates questions to fill knowledge gaps using LLM."""

    def __init__(self, memory: "SiliconMemory", llm: Any) -> None:
        self._memory = memory
        self._llm = llm

    async def generate_questions(self, max_questions: int = 5) -> list[Question]:
        """Identify knowledge gaps and generate questions to fill them."""
        backend = self._memory._backend

        # Gather uncertain beliefs
        uncertain_text = "None found"
        try:
            uncertain = await backend.get_uncertain_beliefs(min_entropy=0.5, k=50)
            if uncertain:
                lines = []
                for u in uncertain[:20]:
                    ext_id = getattr(u, "external_id", "")
                    text = getattr(u, "text", "") or str(u)
                    entropy = getattr(u, "entropy", 0)
                    lines.append(f"- [{ext_id}] {text[:200]} (entropy={entropy:.2f})")
                uncertain_text = "\n".join(lines)
        except Exception:
            pass

        # Gather contested beliefs
        contested_text = "None found"
        try:
            from silicon_memory.core.types import BeliefStatus
            contested = await backend.query_beliefs(
                query="the", limit=50, include_contested=True,
            )
            contested_beliefs = [b for b in contested if b.status == BeliefStatus.CONTESTED]
            if contested_beliefs:
                lines = []
                for b in contested_beliefs[:20]:
                    content = b.content or (b.triplet.as_text() if b.triplet else "")
                    lines.append(f"- [{b.id}] {content[:200]} (conf={b.confidence:.2f})")
                contested_text = "\n".join(lines)
        except Exception:
            pass

        # Gather recent contradictions
        contradictions_text = "None found"
        try:
            contras = await backend.detect_triple_contradictions(min_probability=0.3)
            if contras:
                lines = []
                for c in contras[:10]:
                    lines.append(f"- {getattr(c, 'subject', '?')} {getattr(c, 'predicate', '?')}: "
                                 f"conflicting values")
                contradictions_text = "\n".join(lines)
        except Exception:
            pass

        prompt = _QUESTION_PROMPT.format(
            uncertain_beliefs=uncertain_text,
            contested_beliefs=contested_text,
            contradictions=contradictions_text,
        )
        for attempt in range(5):
            heuristic = await self._heuristic_questions(max_questions=max_questions)
            if heuristic:
                return heuristic
            if attempt < 4:
                # Freshly committed triples can be briefly invisible in search paths.
                await asyncio.sleep(0.2 * (attempt + 1))

        try:
            response = await self._llm_call(
                "You identify knowledge gaps and generate questions. Return valid JSON only.",
                prompt,
            )
            parsed = self._parse_json(response)
            if not parsed or "questions" not in parsed:
                return await self._heuristic_questions(max_questions=max_questions)

            questions = []
            for q in parsed["questions"][:max_questions]:
                questions.append(Question(
                    question=q.get("question", ""),
                    target_belief_id=q.get("target_belief_id"),
                    priority=q.get("priority", "medium"),
                    context=q.get("context", ""),
                ))
            if questions:
                return questions
            return await self._heuristic_questions(max_questions=max_questions)
        except Exception as e:
            logger.debug("Question generation failed: %s", e)
            return await self._heuristic_questions(max_questions=max_questions)

    async def generate_and_store(self) -> int:
        """Generate questions and store in working memory + belief store."""
        questions = await self.generate_questions()
        if not questions:
            for attempt in range(3):
                await asyncio.sleep(0.4 * (attempt + 1))
                questions = await self._heuristic_questions(max_questions=5)
                if questions:
                    break
        if not questions:
            return 0

        stored = 0
        for q in questions:
            # Store as working memory (available to next LLM session, with TTL)
            try:
                key = f"open_question_{q.id}"
                value = {
                    "question": q.question,
                    "priority": q.priority,
                    "context": q.context,
                    "target_belief_id": q.target_belief_id,
                    "generated_at": utc_now().isoformat(),
                }
                await self._memory.set_context(key, value, ttl_seconds=86400)  # 24h TTL
                stored += 1
            except Exception as e:
                logger.debug("Failed to store question in working memory: %s", e)

            # Also store as belief for persistence
            try:
                from silicon_memory.core.types import Belief, Source, SourceType
                belief = Belief(
                    id=q.id,
                    content=f"[QUESTION] {q.question}",
                    confidence=0.5,
                    source=Source(
                        id="question_generator",
                        type=SourceType.REFLECTION,
                        reliability=0.5,
                    ),
                    tags={"open_question", q.priority},
                    metadata={"context": q.context, "target_belief_id": q.target_belief_id},
                )
                await self._memory.commit_belief(belief)
            except Exception as e:
                logger.debug("Failed to store question as belief: %s", e)

        logger.info("Question generation: %d questions generated, %d stored", len(questions), stored)
        return stored

    async def get_open_questions(self) -> list[dict[str, Any]]:
        """Retrieve open questions from working memory and belief store."""
        questions: list[dict[str, Any]] = []

        # From working memory
        try:
            all_working = await self._memory.get_all_context()
            for key, value in all_working.items():
                if key.startswith("open_question_") and isinstance(value, dict):
                    questions.append(value)
        except Exception:
            pass

        # From belief store (tagged as open_question)
        try:
            beliefs = await self._memory.query_beliefs(
                query="QUESTION", limit=20, min_confidence=0.0,
            )
            for b in beliefs:
                if "open_question" in (b.tags or set()):
                    questions.append({
                        "question": b.content.replace("[QUESTION] ", ""),
                        "priority": next((t for t in b.tags if t in ("high", "medium", "low")), "medium"),
                        "context": b.metadata.get("context", ""),
                        "belief_id": str(b.id),
                    })
        except Exception:
            pass

        # Deduplicate by question text
        seen = set()
        unique = []
        for q in questions:
            text = q.get("question", "")
            if text and text not in seen:
                seen.add(text)
                unique.append(q)

        return unique

    async def _llm_call(self, system: str, user: str) -> str:
        """Make an LLM call, handling both scheduler and direct provider."""
        prompt = f"{system}\n\n{user}"
        if hasattr(self._llm, "complete"):
            return await self._llm.complete(
                prompt, system=system, temperature=0.5, max_tokens=1024,
            )
        elif hasattr(self._llm, "generate"):
            return await self._llm.generate(
                prompt, max_tokens=1024, temperature=0.5,
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

    async def _heuristic_questions(self, max_questions: int = 5) -> list[Question]:
        """Fallback question generation from hypothesis beliefs."""
        backend = self._memory._backend
        hypotheses = []
        low_conf_triple_questions: list[Question] = []
        coverage_triple_questions: list[Question] = []
        try:
            if hasattr(backend, "get_beliefs_by_tag"):
                hypotheses = await backend.get_beliefs_by_tag(
                    "hypothesis",
                    limit=max_questions * 6,
                    min_confidence=0.0,
                )
            if not hypotheses:
                hypotheses = await backend.query_beliefs(
                    query="hypothesis",
                    limit=max_questions * 6,
                    min_confidence=0.0,
                    include_contested=True,
                )
        except Exception:
            hypotheses = []

        if not hypotheses:
            try:
                hypotheses = await backend.query_beliefs(
                    query="the",
                    limit=max_questions * 8,
                    min_confidence=0.0,
                    include_contested=True,
                )
            except Exception:
                hypotheses = []

        if hasattr(backend, "_query_triples"):
            try:
                triples = backend._query_triples(k=max(max_questions * 200, 500))
                if inspect.isawaitable(triples):
                    triples = await triples
            except Exception:
                triples = []
            seen_statements: set[str] = set()
            for t in triples:
                if (
                    len(low_conf_triple_questions) >= max_questions
                    and len(coverage_triple_questions) >= max_questions
                ):
                    break
                confidence = float(getattr(t, "probability", 0.5) or 0.5)
                metadata = getattr(t, "metadata", {}) or {}
                if hasattr(backend, "_can_access") and not backend._can_access(  # noqa: SLF001
                    metadata,
                    _external_id=getattr(t, "external_id", ""),
                ):
                    continue
                statement = str(metadata.get("content") or "").strip()
                if not statement:
                    statement = (
                        f"{getattr(t, 'subject', '')} "
                        f"{getattr(t, 'predicate', '')} "
                        f"{getattr(t, 'object_value', '')}"
                    ).strip()
                statement = statement[:220].strip()
                if not statement or statement in seen_statements:
                    continue
                seen_statements.add(statement)
                target_belief_id = str(metadata.get("belief_id") or "") or None
                if confidence <= 0.75:
                    low_conf_triple_questions.append(
                        Question(
                            question=f"What evidence would confirm or refute: {statement}?",
                            target_belief_id=target_belief_id,
                            priority="high" if confidence < 0.5 else "medium",
                            context=(
                                "Heuristic fallback from low-confidence triple "
                                f"(confidence={confidence:.2f})."
                            ),
                        ),
                    )
                else:
                    coverage_triple_questions.append(
                        Question(
                            question=f"What independent source would corroborate: {statement}?",
                            target_belief_id=target_belief_id,
                            priority="low",
                            context=(
                                "Coverage fallback from available high-confidence triple "
                                f"(confidence={confidence:.2f})."
                            ),
                        ),
                    )

        if not hypotheses and low_conf_triple_questions:
            return low_conf_triple_questions[:max_questions]

        questions: list[Question] = []
        seen: set[str] = set()
        ordered_hypotheses = sorted(
            hypotheses,
            key=lambda b: float(getattr(b, "confidence", 0.5) or 0.5),
        )
        for b in ordered_hypotheses:
            if len(questions) >= max_questions:
                break
            conf = float(getattr(b, "confidence", 0.5) or 0.5)
            if conf > 0.75:
                continue
            statement = (getattr(b, "content", "") or "").strip()
            if not statement and getattr(b, "triplet", None):
                statement = b.triplet.as_text()
            statement = statement[:220].strip()
            if not statement or statement in seen:
                continue
            seen.add(statement)
            questions.append(
                Question(
                    question=f"What evidence would confirm or refute: {statement}?",
                    target_belief_id=str(getattr(b, "id", "")) or None,
                    priority="high" if conf < 0.5 else "medium",
                    context=f"Heuristic fallback from hypothesis belief (confidence={conf:.2f}).",
                ),
            )
        if not questions and ordered_hypotheses:
            for b in ordered_hypotheses:
                if len(questions) >= max_questions:
                    break
                conf = float(getattr(b, "confidence", 0.5) or 0.5)
                statement = (getattr(b, "content", "") or "").strip()
                if not statement and getattr(b, "triplet", None):
                    statement = b.triplet.as_text()
                statement = statement[:220].strip()
                if not statement or statement in seen:
                    continue
                seen.add(statement)
                questions.append(
                    Question(
                        question=f"What evidence would confirm or refute: {statement}?",
                        target_belief_id=str(getattr(b, "id", "")) or None,
                        priority="medium",
                        context=(
                            "Heuristic fallback from least-confident available belief "
                            f"(confidence={conf:.2f})."
                        ),
                    ),
                )
        if questions:
            return questions

        if low_conf_triple_questions:
            return low_conf_triple_questions[:max_questions]
        if coverage_triple_questions:
            return coverage_triple_questions[:max_questions]

        # Last-resort fallback: derive verification questions from fresh experiences.
        try:
            recent = await self._memory.get_recent_experiences(
                hours=24 * 30,
                limit=max_questions * 3,
            )
        except Exception:
            recent = []
        experience_questions: list[Question] = []
        seen_snippets: set[str] = set()
        for exp in recent:
            if len(experience_questions) >= max_questions:
                break
            snippet = (getattr(exp, "content", "") or "").strip()
            if not snippet:
                continue
            snippet = " ".join(snippet.split())
            snippet = snippet[:220].strip()
            if not snippet or snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            experience_questions.append(
                Question(
                    question=(
                        "What external evidence can validate this recent claim or event: "
                        f"{snippet}?"
                    ),
                    target_belief_id=None,
                    priority="low",
                    context="Fallback from recent episodic experience when belief/triple signals are sparse.",
                ),
            )
        if experience_questions:
            return experience_questions

        return []
