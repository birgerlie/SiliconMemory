"""Tests for SM-2: Decision Synthesis."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from silicon_memory.decision.types import (
    DecisionBrief,
    EvidencedClaim,
    Option,
    Precedent,
    Risk,
    Uncertainty,
)
from silicon_memory.decision.synthesis import DecisionBriefGenerator
from silicon_memory.tools.decision_tool import DecisionTool


# ============================================================================
# Unit tests: DecisionBrief types
# ============================================================================


class TestDecisionBriefTypes:
    """Test decision brief type dataclasses."""

    def test_evidenced_claim(self):
        bid = uuid4()
        claim = EvidencedClaim(
            claim="Python is popular",
            belief_id=bid,
            confidence=0.9,
            evidence_count=5,
            source_description="internal",
        )
        assert claim.claim == "Python is popular"
        assert claim.belief_id == bid
        assert claim.confidence == 0.9

    def test_option(self):
        opt = Option(title="Use PostgreSQL", description="Relational DB")
        assert opt.title == "Use PostgreSQL"
        assert opt.supporting_evidence == []
        assert opt.opposing_evidence == []
        assert opt.estimated_confidence == 0.5

    def test_risk(self):
        risk = Risk(description="Migration may fail", severity="high")
        assert risk.severity == "high"
        assert risk.related_beliefs == []

    def test_uncertainty(self):
        unc = Uncertainty(description="Team skill unknown", entropy=0.7)
        assert unc.entropy == 0.7
        assert unc.impact == "medium"

    def test_precedent(self):
        did = uuid4()
        prec = Precedent(
            decision_id=did,
            title="Previous DB choice",
            outcome="Worked well",
            relevance_score=0.8,
        )
        assert prec.relevance_score == 0.8

    def test_decision_brief_defaults(self):
        brief = DecisionBrief(question="Should we migrate?")
        assert brief.question == "Should we migrate?"
        assert brief.options == []
        assert brief.key_beliefs == []
        assert brief.risks == []
        assert brief.has_contradictions is False
        assert brief.total_evidence == 0
        assert brief.confidence_in_recommendation == 0.0

    def test_decision_brief_has_contradictions(self):
        brief = DecisionBrief(
            question="Test",
            options=[
                Option(
                    title="A",
                    opposing_evidence=[
                        EvidencedClaim(claim="x", belief_id=uuid4(), confidence=0.5)
                    ],
                )
            ],
        )
        assert brief.has_contradictions is True

    def test_decision_brief_total_evidence(self):
        bid = uuid4()
        brief = DecisionBrief(
            question="Test",
            key_beliefs=[
                EvidencedClaim(claim="a", belief_id=bid, confidence=0.5),
                EvidencedClaim(claim="b", belief_id=bid, confidence=0.6),
            ],
            options=[
                Option(
                    title="X",
                    supporting_evidence=[
                        EvidencedClaim(claim="s", belief_id=bid, confidence=0.7)
                    ],
                    opposing_evidence=[
                        EvidencedClaim(claim="o", belief_id=bid, confidence=0.4)
                    ],
                ),
            ],
        )
        assert brief.total_evidence == 4  # 2 key + 1 supporting + 1 opposing

    def test_decision_brief_to_dict(self):
        brief = DecisionBrief(
            question="Which DB?",
            summary="Analysis summary",
            recommendation="Use PostgreSQL",
            confidence_in_recommendation=0.8,
        )
        data = brief.to_dict()
        assert data["question"] == "Which DB?"
        assert data["summary"] == "Analysis summary"
        assert data["recommendation"] == "Use PostgreSQL"
        assert data["confidence_in_recommendation"] == 0.8
        assert data["has_contradictions"] is False


# ============================================================================
# Unit tests: DecisionBriefGenerator with mocked memory
# ============================================================================


class TestDecisionBriefGenerator:
    """Test the decision brief generator with mocked memory."""

    def _make_mock_memory(self, beliefs=None, decisions=None):
        """Create a mock SiliconMemory."""
        from silicon_memory.memory.silicondb_router import RecallResponse
        from silicon_memory.core.utils import utc_now

        mock_facts = []
        for b in (beliefs or []):
            fact = MagicMock()
            fact.content = b.get("content", "")
            fact.confidence = b.get("confidence", 0.5)
            fact.source = MagicMock()
            fact.source.id = b.get("source_id", str(uuid4()))
            fact.memory_type = "belief"
            fact.relevance_score = 0.8
            mock_facts.append(fact)

        recall_response = MagicMock(spec=RecallResponse)
        recall_response.facts = mock_facts
        recall_response.experiences = []
        recall_response.procedures = []
        recall_response.working_context = {}
        recall_response.total_items = len(mock_facts)
        recall_response.query = "test"
        recall_response.as_of = utc_now()

        memory = AsyncMock()
        memory.recall = AsyncMock(return_value=recall_response)
        memory.recall_decisions = AsyncMock(return_value=decisions or [])
        return memory

    async def test_generate_with_beliefs(self):
        """Test that beliefs are mapped to key_beliefs in brief."""
        memory = self._make_mock_memory(beliefs=[
            {"content": "PostgreSQL is reliable", "confidence": 0.9},
            {"content": "Team knows SQL well", "confidence": 0.8},
            {"content": "NoSQL is trendy", "confidence": 0.4},
        ])

        generator = DecisionBriefGenerator(memory)
        brief = await generator.generate("Should we use PostgreSQL?")

        assert brief.question == "Should we use PostgreSQL?"
        assert len(brief.key_beliefs) == 3
        assert all(isinstance(c, EvidencedClaim) for c in brief.key_beliefs)
        # All claims should have belief_ids
        for claim in brief.key_beliefs:
            assert claim.belief_id is not None

    async def test_generate_identifies_uncertainties(self):
        """Test that low-confidence beliefs become uncertainties."""
        memory = self._make_mock_memory(beliefs=[
            {"content": "High confidence fact", "confidence": 0.9},
            {"content": "Low confidence guess", "confidence": 0.3},
            {"content": "Another uncertain thing", "confidence": 0.4},
        ])

        generator = DecisionBriefGenerator(memory)
        brief = await generator.generate("Test question")

        assert len(brief.uncertainties) >= 2

    async def test_generate_with_precedents(self):
        """Test that past decisions become precedents."""
        from silicon_memory.core.decision import Decision

        past = Decision(title="Previous DB choice", outcome="Worked well")
        memory = self._make_mock_memory(beliefs=[], decisions=[past])

        generator = DecisionBriefGenerator(memory)
        brief = await generator.generate("Which database?")

        assert len(brief.past_precedents) == 1
        assert brief.past_precedents[0].title == "Previous DB choice"

    async def test_generate_empty_beliefs(self):
        """Test generation with no beliefs."""
        memory = self._make_mock_memory(beliefs=[])

        generator = DecisionBriefGenerator(memory)
        brief = await generator.generate("Unknown topic")

        assert brief.question == "Unknown topic"
        assert brief.key_beliefs == []
        assert brief.confidence_in_recommendation == 0.0

    async def test_generate_heuristic_summary(self):
        """Test that heuristic brief includes summary."""
        memory = self._make_mock_memory(beliefs=[
            {"content": "Fact 1", "confidence": 0.8},
            {"content": "Fact 2", "confidence": 0.7},
        ])

        generator = DecisionBriefGenerator(memory)
        brief = await generator.generate("Test")

        assert "2 relevant beliefs" in brief.summary

    async def test_generate_with_llm(self):
        """Test LLM-powered synthesis."""
        memory = self._make_mock_memory(beliefs=[
            {"content": "PostgreSQL is reliable", "confidence": 0.9},
        ])

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps({
            "summary": "LLM synthesized summary",
            "options": [
                {"title": "PostgreSQL", "description": "Reliable choice", "estimated_confidence": 0.85},
                {"title": "MongoDB", "description": "NoSQL option", "estimated_confidence": 0.6},
            ],
            "risks": [
                {"description": "Migration complexity", "severity": "high"},
            ],
            "recommendation": "Use PostgreSQL",
            "confidence_in_recommendation": 0.85,
        }))

        generator = DecisionBriefGenerator(memory)
        brief = await generator.generate("Which DB?", llm_provider=mock_llm)

        assert brief.summary == "LLM synthesized summary"
        assert len(brief.options) == 2
        assert brief.options[0].title == "PostgreSQL"
        assert len(brief.risks) == 1
        assert brief.recommendation == "Use PostgreSQL"
        assert brief.confidence_in_recommendation == 0.85
        mock_llm.complete.assert_awaited()

    async def test_llm_fallback_on_bad_json(self):
        """Test fallback to heuristic when LLM returns invalid JSON."""
        memory = self._make_mock_memory(beliefs=[
            {"content": "A fact", "confidence": 0.7},
        ])

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="Not valid JSON at all")

        generator = DecisionBriefGenerator(memory)
        brief = await generator.generate("Test", llm_provider=mock_llm)

        # Should fall back to heuristic
        assert brief.question == "Test"
        assert "1 relevant beliefs" in brief.summary


# ============================================================================
# Unit tests: DecisionTool
# ============================================================================


class TestDecisionTool:
    """Test the decision support tool for LLM function calling."""

    def _make_mock_memory(self):
        from silicon_memory.memory.silicondb_router import RecallResponse
        from silicon_memory.core.utils import utc_now

        recall_response = MagicMock(spec=RecallResponse)
        recall_response.facts = []
        recall_response.experiences = []
        recall_response.procedures = []
        recall_response.working_context = {}
        recall_response.total_items = 0
        recall_response.query = "test"
        recall_response.as_of = utc_now()

        memory = AsyncMock()
        memory.recall = AsyncMock(return_value=recall_response)
        memory.recall_decisions = AsyncMock(return_value=[])
        return memory

    async def test_invoke_returns_dict(self):
        """Test that invoke returns a serializable dict."""
        memory = self._make_mock_memory()
        tool = DecisionTool(memory)

        result = await tool.invoke(question="Should we migrate?")

        assert isinstance(result, dict)
        assert result["question"] == "Should we migrate?"
        assert "recommendation" in result
        assert "confidence_in_recommendation" in result

    def test_openai_schema(self):
        """Test the OpenAI function calling schema."""
        schema = DecisionTool.get_openai_schema()
        assert schema["name"] == "decision_support"
        assert "parameters" in schema
        assert "question" in schema["parameters"]["properties"]
        assert "question" in schema["parameters"]["required"]

    async def test_invoke_with_llm(self):
        """Test invoke with LLM provider."""
        memory = self._make_mock_memory()
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps({
            "summary": "Brief",
            "options": [],
            "risks": [],
            "recommendation": "Go ahead",
            "confidence_in_recommendation": 0.7,
        }))

        tool = DecisionTool(memory, llm_provider=mock_llm)
        result = await tool.invoke(question="Test?")

        assert result["recommendation"] == "Go ahead"
