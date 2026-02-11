"""End-to-end tests for cognitive memory features (SM-1 through SM-6).

These tests exercise the full stack from LLM tool interfaces through
SiliconMemory to real SiliconDB storage and back.

Run with:
    SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e_cognitive.py -v
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from silicon_memory.core.types import (
    Belief,
    Experience,
    Source,
    SourceType,
    Triplet,
)
from silicon_memory.core.decision import (
    Alternative,
    Assumption,
    Decision,
    DecisionStatus,
)
from silicon_memory.memory.silicondb_router import RecallContext
from silicon_memory.tools.memory_tool import MemoryTool, MemoryAction
from silicon_memory.tools.decision_tool import DecisionTool


pytestmark = pytest.mark.e2e


# ============================================================================
# SM-1: Decision Records via MemoryTool — E2E Tests
# ============================================================================


class TestDecisionToolE2E:
    """E2E tests for the DecisionTool LLM interface."""

    @pytest.mark.asyncio
    async def test_decision_tool_invoke(self, silicon_memory, sample_source):
        """Invoke DecisionTool and get a structured brief dict."""
        # Store some beliefs first
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Python is good for prototyping", confidence=0.9, source=sample_source)
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Rust is better for performance", confidence=0.85, source=sample_source)
        )

        tool = DecisionTool(silicon_memory)
        result = await tool.invoke(question="Should we use Python or Rust?")

        assert isinstance(result, dict)
        assert result["question"] == "Should we use Python or Rust?"
        assert "recommendation" in result
        assert "confidence_in_recommendation" in result
        assert "total_evidence" in result
        assert "has_contradictions" in result

    @pytest.mark.asyncio
    async def test_decision_tool_schema(self):
        """Decision tool schema is valid for function calling."""
        schema = DecisionTool.get_openai_schema()

        assert schema["name"] == "decision_support"
        assert "parameters" in schema
        assert "question" in schema["parameters"]["properties"]
        assert "question" in schema["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_decision_tool_empty_beliefs(self, silicon_memory):
        """DecisionTool works even with no stored beliefs."""
        tool = DecisionTool(silicon_memory)
        result = await tool.invoke(question="What color should we paint the bikeshed?")

        assert isinstance(result, dict)
        assert result["question"] == "What color should we paint the bikeshed?"
        assert result["confidence_in_recommendation"] == 0.0


# ============================================================================
# SM-1: Decision Records via MemoryTool — E2E Tests
# ============================================================================


class TestMemoryToolDecisionsE2E:
    """E2E tests for decision operations through MemoryTool."""

    @pytest.mark.asyncio
    async def test_store_decision_via_tool(self, silicon_memory, sample_source):
        """Store a decision through the MemoryTool interface."""
        tool = MemoryTool(silicon_memory)

        response = await tool.invoke(
            "store_decision",
            title="Use FastAPI for the new API service",
            description="FastAPI selected for its async support and auto-docs",
            decided_by="architect",
        )

        assert response.success
        assert response.action == MemoryAction.STORE_DECISION

    @pytest.mark.asyncio
    async def test_recall_decisions_via_tool(self, silicon_memory, sample_source):
        """Store then recall decisions through the MemoryTool interface."""
        tool = MemoryTool(silicon_memory)

        # Store a decision
        await tool.invoke(
            "store_decision",
            title="Use Docker for containerization",
            description="Docker selected for packaging microservices",
            decided_by="devops-lead",
        )

        # Recall decisions
        response = await tool.invoke(
            "recall_decisions",
            query="containerization Docker",
        )

        assert response.success
        assert response.action == MemoryAction.RECALL_DECISIONS


# ============================================================================
# SM-2: Decision Synthesis — Full E2E Tests
# ============================================================================


class TestDecisionSynthesisE2E:
    """E2E tests for decision synthesis through the tool interface."""

    @pytest.mark.asyncio
    async def test_accumulate_knowledge_then_synthesize(self, silicon_memory, sample_source):
        """Store facts via MemoryTool, then synthesize via DecisionTool."""
        memory_tool = MemoryTool(silicon_memory)
        decision_tool = DecisionTool(silicon_memory)

        # Accumulate knowledge about databases
        facts = [
            ("PostgreSQL", "is", "relational database"),
            ("PostgreSQL", "supports", "ACID transactions"),
            ("MongoDB", "is", "document database"),
            ("MongoDB", "supports", "horizontal scaling"),
            ("Redis", "is", "in-memory data store"),
        ]

        for subject, predicate, obj in facts:
            await memory_tool.invoke(
                "store_fact",
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=0.85,
            )

        # Also store some experience
        await memory_tool.invoke(
            "store_experience",
            content="Previous project used PostgreSQL successfully",
            outcome="Stable production performance for 2 years",
            importance=0.8,
        )

        # Now synthesize a decision brief
        brief = await decision_tool.invoke(
            question="Which database should we use for the new e-commerce service?"
        )

        assert isinstance(brief, dict)
        assert "question" in brief
        assert "recommendation" in brief

    @pytest.mark.asyncio
    async def test_synthesize_returns_serializable_json(self, silicon_memory, sample_source):
        """Decision brief must be fully JSON-serializable."""
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Testing is important", confidence=0.9, source=sample_source)
        )

        tool = DecisionTool(silicon_memory)
        brief = await tool.invoke(question="Should we add more unit tests?")

        # Must be JSON-serializable
        json_str = json.dumps(brief)
        parsed = json.loads(json_str)
        assert parsed["question"] == "Should we add more unit tests?"


# ============================================================================
# SM-3: Salience Retrieval — E2E Tests
# ============================================================================


class TestSalienceRetrievalE2E:
    """E2E tests for salience-weighted retrieval through tool interface."""

    @pytest.mark.asyncio
    async def test_recall_with_salience_through_tool(self, silicon_memory, sample_source):
        """Store facts then recall with salience via MemoryTool."""
        tool = MemoryTool(silicon_memory)

        # Store facts
        await tool.invoke(
            "store_fact",
            content="FastAPI uses Starlette under the hood",
            confidence=0.9,
        )
        await tool.invoke(
            "store_fact",
            content="FastAPI automatically generates OpenAPI docs",
            confidence=0.95,
        )

        # Recall (MemoryTool doesn't directly expose salience_profile,
        # but we can test through the underlying memory object)
        ctx = RecallContext(
            query="FastAPI features",
            salience_profile="decision_support",
            max_facts=10,
        )
        response = await silicon_memory.recall(ctx)
        assert response is not None
        assert response.query == "FastAPI features"


# ============================================================================
# SM-5: Passive Ingestion — E2E Tests
# ============================================================================


class TestPassiveIngestionE2E:
    """E2E tests for meeting transcript ingestion."""

    @pytest.mark.asyncio
    async def test_full_meeting_workflow(self, silicon_memory, sample_source):
        """Full workflow: ingest meeting → store facts → recall everything."""
        from silicon_memory.ingestion.meeting import MeetingTranscriptAdapter

        memory_tool = MemoryTool(silicon_memory)

        # Step 1: Ingest a meeting transcript
        transcript = (
            "Alice: The deployment pipeline is too slow.\n"
            "Bob: We should parallelize the test suite.\n"
            "Alice: Good idea. Also, we need to cache Docker layers.\n"
            "Bob: I'll implement the parallel test runner by next week.\n"
            "Alice: And I'll look into Docker layer caching.\n"
        )

        adapter = MeetingTranscriptAdapter()
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=transcript,
            metadata={
                "meeting_id": "deploy-optimization",
                "title": "Deployment Pipeline Optimization",
            },
        )

        assert result.experiences_created > 0

        # Step 2: Store a related fact
        await memory_tool.invoke(
            "store_fact",
            content="Parallel test execution can reduce CI time by 50%",
            confidence=0.8,
        )

        # Step 3: Recall everything about the topic
        response = await memory_tool.invoke(
            "recall",
            query="deployment pipeline CI/CD optimization",
            max_facts=10,
            max_experiences=10,
            max_procedures=5,
        )

        assert response.success

    @pytest.mark.asyncio
    async def test_ingest_then_decision(self, silicon_memory, sample_source):
        """Ingest meeting → make decision → verify both retrievable."""
        from silicon_memory.ingestion.meeting import MeetingTranscriptAdapter

        memory_tool = MemoryTool(silicon_memory)

        # Ingest meeting
        transcript = (
            "Alice: We need to choose a logging framework.\n"
            "Bob: I recommend structured logging with JSON output.\n"
            "Alice: Agreed. Let's use structlog for Python.\n"
        )

        adapter = MeetingTranscriptAdapter()
        await silicon_memory.ingest_from(
            adapter=adapter,
            content=transcript,
            metadata={"meeting_id": "logging-decision"},
        )

        # Store the decision
        await memory_tool.invoke(
            "store_decision",
            title="Use structlog for structured logging",
            description="structlog selected for JSON-formatted structured logging",
            decided_by="Alice",
        )

        # Recall both experiences and decisions
        response = await memory_tool.invoke(
            "recall",
            query="logging framework structlog",
        )
        assert response.success

        recall_response = await memory_tool.invoke(
            "recall_decisions",
            query="logging framework",
        )
        assert recall_response.success


# ============================================================================
# SM-6: News Integration — E2E Tests
# ============================================================================


class TestNewsIntegrationE2E:
    """E2E tests for news article ingestion and cross-referencing."""

    @pytest.mark.asyncio
    async def test_ingest_news_then_cross_reference(self, silicon_memory, sample_source):
        """Ingest internal + external beliefs, then cross-reference."""
        from silicon_memory.ingestion.news import NewsArticleAdapter

        memory_tool = MemoryTool(silicon_memory)

        # Store internal belief
        await memory_tool.invoke(
            "store_fact",
            content="Microservices architecture improves team autonomy",
            confidence=0.85,
        )

        # Ingest news article with external perspective
        adapter = NewsArticleAdapter()
        await silicon_memory.ingest_from(
            adapter=adapter,
            content='{"title": "Microservices Challenges", "body": "While microservices improve team autonomy, they add significant operational complexity. Network latency becomes a major concern."}',
            metadata={"source_name": "ArchWeekly", "credibility": 0.75},
        )

        # Cross-reference
        result = await silicon_memory.cross_reference(
            "microservices architecture",
            min_confidence=0.3,
        )

        assert result.query == "microservices architecture"
        assert isinstance(result.internal_beliefs, list)
        assert isinstance(result.external_beliefs, list)

    @pytest.mark.asyncio
    async def test_news_feeds_into_decision_brief(self, silicon_memory, sample_source):
        """News-derived beliefs contribute to decision synthesis."""
        from silicon_memory.ingestion.news import NewsArticleAdapter

        memory_tool = MemoryTool(silicon_memory)
        decision_tool = DecisionTool(silicon_memory)

        # Internal knowledge
        await memory_tool.invoke(
            "store_fact",
            content="We currently use a monolith architecture",
            confidence=0.95,
        )

        # External news
        adapter = NewsArticleAdapter()
        await silicon_memory.ingest_from(
            adapter=adapter,
            content='{"title": "Monolith to Microservices Migration", "body": "Companies report 6-12 month migration timelines. Success requires strong DevOps culture and container orchestration expertise."}',
            metadata={"source_name": "TechTrends", "credibility": 0.7},
        )

        # Decision synthesis should incorporate both internal and external knowledge
        brief = await decision_tool.invoke(
            question="Should we migrate from monolith to microservices?"
        )

        assert isinstance(brief, dict)
        assert brief["question"] == "Should we migrate from monolith to microservices?"


# ============================================================================
# Full Cognitive Workflow E2E Tests
# ============================================================================


class TestFullCognitiveWorkflowE2E:
    """End-to-end tests for complete cognitive memory workflows."""

    @pytest.mark.asyncio
    async def test_learn_decide_act_reflect(self, silicon_memory, sample_source):
        """Full cognitive loop: learn → decide → act → reflect."""
        memory_tool = MemoryTool(silicon_memory)
        decision_tool = DecisionTool(silicon_memory)

        # LEARN: Accumulate knowledge
        facts = [
            "Python has excellent library ecosystem for web development",
            "Django provides batteries-included web framework",
            "FastAPI is modern and supports async natively",
            "Flask is lightweight and flexible",
        ]
        for fact in facts:
            await memory_tool.invoke("store_fact", content=fact, confidence=0.85)

        # DECIDE: Synthesize a decision
        brief = await decision_tool.invoke(
            question="Which Python web framework should we use?"
        )
        assert isinstance(brief, dict)

        # ACT: Record the decision
        await memory_tool.invoke(
            "store_decision",
            title="Use FastAPI for new API service",
            description="FastAPI chosen for async support and auto-documentation",
            decided_by="team",
        )

        # REFLECT: Record experience and verify recall
        await memory_tool.invoke(
            "store_experience",
            content="Chose FastAPI after evaluating Django, Flask, and FastAPI",
            outcome="Team aligned on FastAPI for its modern features",
            importance=0.8,
        )

        # Verify everything is retrievable
        recall = await memory_tool.invoke(
            "recall",
            query="Python web framework FastAPI",
            max_facts=10,
            max_experiences=5,
        )
        assert recall.success

    @pytest.mark.asyncio
    async def test_context_switch_and_resume(self, silicon_memory, sample_source):
        """Test context switching between sessions."""
        tool = MemoryTool(silicon_memory)

        # Session A: Working on backend
        await tool.invoke("set_context", key="current_task", value="backend API design")
        await tool.invoke(
            "store_fact",
            content="REST API should follow OpenAPI 3.0 spec",
            confidence=0.9,
        )

        # Verify context
        ctx_response = await tool.invoke("get_context", key="current_task")
        assert ctx_response.success
        assert ctx_response.data["value"] == "backend API design"

        # Switch context
        await tool.invoke("set_context", key="current_task", value="frontend review")
        await tool.invoke(
            "store_fact",
            content="React components should be functional, not class-based",
            confidence=0.85,
        )

        # Verify switched context
        ctx_response = await tool.invoke("get_context", key="current_task")
        assert ctx_response.data["value"] == "frontend review"

    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, silicon_memory, sample_source):
        """Test that bulk operations complete within reasonable time."""
        import time

        tool = MemoryTool(silicon_memory)

        # Store 30 facts
        start = time.monotonic()
        for i in range(30):
            await tool.invoke(
                "store_fact",
                content=f"Technical fact number {i} about distributed systems",
                confidence=0.7 + (i % 3) * 0.1,
            )
        store_elapsed = time.monotonic() - start

        # Store 5 decisions
        for i in range(5):
            await tool.invoke(
                "store_decision",
                title=f"Architecture decision {i}",
                description=f"Decision about component {i} in the system",
                decided_by="architect",
            )

        # Recall should work
        start = time.monotonic()
        response = await tool.invoke(
            "recall",
            query="distributed systems architecture",
            max_facts=20,
        )
        recall_elapsed = time.monotonic() - start

        assert response.success
        # Sanity check: operations should complete (not hang)
        assert store_elapsed < 60  # 30 facts in under 60s
        assert recall_elapsed < 10  # Recall in under 10s

    @pytest.mark.asyncio
    async def test_response_json_serialization(self, silicon_memory, sample_source):
        """All tool responses must be JSON-serializable for LLM consumption."""
        memory_tool = MemoryTool(silicon_memory)

        # Store fact
        store_resp = await memory_tool.invoke(
            "store_fact", content="Test fact", confidence=0.8
        )
        json.dumps(store_resp.to_dict())

        # Store decision
        dec_resp = await memory_tool.invoke(
            "store_decision",
            title="Test decision",
            description="Test",
            decided_by="user",
        )
        json.dumps(dec_resp.to_dict())

        # Recall
        recall_resp = await memory_tool.invoke("recall", query="test")
        json.dumps(recall_resp.to_dict())

        # Decision tool
        decision_tool = DecisionTool(silicon_memory)
        brief = await decision_tool.invoke(question="Test?")
        json.dumps(brief)
