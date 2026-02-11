"""Decision brief synthesis engine."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING
from uuid import UUID

from silicon_memory.decision.types import (
    DecisionBrief,
    EvidencedClaim,
    Option,
    Precedent,
    Risk,
    Uncertainty,
)
from silicon_memory.retrieval.salience import PROFILES

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory, RecallContext
    from silicon_memory.reflection.llm import LLMProvider


class DecisionBriefGenerator:
    """Generates structured decision briefs from memory.

    The generator:
    1. Retrieves relevant beliefs using decision_support salience profile
    2. Retrieves past decisions as precedents
    3. Identifies contradictions and high-entropy beliefs
    4. Uses LLM (if available) to synthesize into a structured brief
    5. Every claim is linked back to a belief_id for traceability

    Example:
        >>> generator = DecisionBriefGenerator(memory)
        >>> brief = await generator.generate("Should we use PostgreSQL?")
        >>> print(brief.recommendation)
    """

    def __init__(
        self,
        memory: "SiliconMemory",
    ) -> None:
        self._memory = memory

    async def generate(
        self,
        question: str,
        llm_provider: "LLMProvider | None" = None,
        max_beliefs: int = 30,
        max_precedents: int = 5,
    ) -> DecisionBrief:
        """Generate a decision brief for a question.

        Args:
            question: The decision question
            llm_provider: Optional LLM for synthesis
            max_beliefs: Maximum beliefs to retrieve
            max_precedents: Maximum past decisions to include

        Returns:
            DecisionBrief with structured analysis
        """
        # Step 1: Retrieve relevant beliefs with decision_support profile
        from silicon_memory.memory.silicondb_router import RecallContext

        ctx = RecallContext(
            query=question,
            max_facts=max_beliefs,
            salience_profile="decision_support",
            min_confidence=0.2,
        )

        recall_response = await self._memory.recall(ctx)
        beliefs = recall_response.facts

        # Step 2: Retrieve past decisions as precedents
        precedents = []
        try:
            past_decisions = await self._memory.recall_decisions(
                question, k=max_precedents
            )
            for d in past_decisions:
                precedents.append(Precedent(
                    decision_id=d.id,
                    title=d.title,
                    outcome=d.outcome,
                    relevance_score=0.5,
                ))
        except Exception:
            pass

        # Step 3: Build key beliefs list with evidence links
        key_beliefs = []
        for fact in beliefs:
            claim = EvidencedClaim(
                claim=fact.content or "",
                belief_id=UUID(fact.source.id) if fact.source and _is_uuid(fact.source.id) else UUID(int=0),
                confidence=fact.confidence,
                evidence_count=0,
                source_description=fact.source.id if fact.source else "",
            )
            key_beliefs.append(claim)

        # Step 4: Identify uncertainties from low-confidence beliefs
        uncertainties = []
        for fact in beliefs:
            if fact.confidence < 0.5:
                uncertainties.append(Uncertainty(
                    description=f"Low confidence: {fact.content or ''}",
                    belief_id=UUID(fact.source.id) if fact.source and _is_uuid(fact.source.id) else None,
                    entropy=1.0 - fact.confidence,
                    impact="medium",
                ))

        # Step 5: Synthesize with LLM or build heuristic brief
        if llm_provider:
            brief = await self._synthesize_with_llm(
                question, key_beliefs, precedents, uncertainties, llm_provider
            )
        else:
            brief = self._synthesize_heuristic(
                question, key_beliefs, precedents, uncertainties
            )

        return brief

    async def _synthesize_with_llm(
        self,
        question: str,
        key_beliefs: list[EvidencedClaim],
        precedents: list[Precedent],
        uncertainties: list[Uncertainty],
        provider: "LLMProvider",
    ) -> DecisionBrief:
        """Use LLM to synthesize a decision brief."""
        belief_text = "\n".join(
            f"- [{c.confidence:.1f}] {c.claim}" for c in key_beliefs[:20]
        )
        precedent_text = "\n".join(
            f"- {p.title} (outcome: {p.outcome or 'unknown'})" for p in precedents
        )

        prompt = (
            f"Decision question: {question}\n\n"
            f"Relevant beliefs:\n{belief_text}\n\n"
            f"Past precedents:\n{precedent_text or 'None'}\n\n"
            "Based on the above evidence, provide a decision brief as JSON:\n"
            '{"summary": "...", "options": [{"title": "...", "description": "...", '
            '"estimated_confidence": 0.0-1.0}], "risks": [{"description": "...", '
            '"severity": "low/medium/high"}], "recommendation": "...", '
            '"confidence_in_recommendation": 0.0-1.0}\n'
            "Return ONLY the JSON."
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a decision analysis engine. Synthesize evidence into structured briefs.",
            temperature=0.3,
            max_tokens=2000,
        )

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return self._synthesize_heuristic(question, key_beliefs, precedents, uncertainties)

        options = []
        for opt_data in data.get("options", []):
            options.append(Option(
                title=opt_data.get("title", ""),
                description=opt_data.get("description", ""),
                estimated_confidence=opt_data.get("estimated_confidence", 0.5),
            ))

        risks = []
        for risk_data in data.get("risks", []):
            risks.append(Risk(
                description=risk_data.get("description", ""),
                severity=risk_data.get("severity", "medium"),
            ))

        return DecisionBrief(
            question=question,
            summary=data.get("summary", ""),
            options=options,
            key_beliefs=key_beliefs,
            risks=risks,
            uncertainties=uncertainties,
            past_precedents=precedents,
            recommendation=data.get("recommendation", ""),
            confidence_in_recommendation=data.get("confidence_in_recommendation", 0.0),
        )

    def _synthesize_heuristic(
        self,
        question: str,
        key_beliefs: list[EvidencedClaim],
        precedents: list[Precedent],
        uncertainties: list[Uncertainty],
    ) -> DecisionBrief:
        """Build a decision brief using heuristics (no LLM)."""
        avg_confidence = (
            sum(b.confidence for b in key_beliefs) / len(key_beliefs)
            if key_beliefs
            else 0.0
        )

        summary = (
            f"Analysis based on {len(key_beliefs)} relevant beliefs"
            f" and {len(precedents)} past precedents."
        )
        if uncertainties:
            summary += f" {len(uncertainties)} areas of uncertainty identified."

        return DecisionBrief(
            question=question,
            summary=summary,
            key_beliefs=key_beliefs,
            uncertainties=uncertainties,
            past_precedents=precedents,
            confidence_in_recommendation=avg_confidence,
        )


def _is_uuid(s: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        UUID(s)
        return True
    except (ValueError, AttributeError):
        return False
