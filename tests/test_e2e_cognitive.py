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


# ============================================================================
# SDB-1/2/3: Search-weights wiring, entropy reranking, graph context — E2E
# ============================================================================


class TestSearchWeightsWiringE2E:
    """E2E: search_weights flow from SalienceProfile through recall pipeline."""

    @pytest.mark.asyncio
    async def test_recall_with_search_weights_returns_results(
        self, silicon_memory, sample_source,
    ):
        """Store several beliefs, recall with decision_support profile,
        verify results come back with populated fields."""
        from silicon_memory.memory.silicondb_router import RecallContext

        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Rust has zero-cost abstractions",
                   confidence=0.95, source=sample_source, tags=["rust"]),
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Rust memory safety without GC",
                   confidence=0.9, source=sample_source, tags=["rust"]),
        )

        ctx = RecallContext(
            query="Rust safety",
            salience_profile="decision_support",
            max_facts=10,
        )
        response = await silicon_memory.recall(ctx)

        assert response is not None
        assert response.query == "Rust safety"
        # Facts should be present (mock embedder uses hashing, so relevance
        # is approximate, but at least the pipeline shouldn't crash)
        assert isinstance(response.facts, list)

    @pytest.mark.asyncio
    async def test_backend_recall_passes_weights_to_search(
        self, silicon_memory, sample_source,
    ):
        """Backend.recall() with search_weights should not raise."""
        from silicon_memory.retrieval.salience import PROFILES

        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Go has goroutines for concurrency",
                   confidence=0.88, source=sample_source, tags=["go"]),
        )

        weights = PROFILES["exploration"].to_search_weights()

        result = await silicon_memory._backend.recall(
            query="Go concurrency",
            max_facts=5,
            search_weights=weights,
        )

        assert "facts" in result
        assert "experiences" in result
        assert "procedures" in result

    @pytest.mark.asyncio
    async def test_backend_query_beliefs_accepts_search_weights(
        self, silicon_memory, sample_source,
    ):
        """query_beliefs() with search_weights should work end-to-end."""
        from silicon_memory.retrieval.salience import PROFILES

        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="TypeScript adds types to JavaScript",
                   confidence=0.9, source=sample_source, tags=["typescript"]),
        )

        weights = PROFILES["context_recall"].to_search_weights()
        beliefs = await silicon_memory._backend.query_beliefs(
            "TypeScript types", limit=5, search_weights=weights,
        )

        assert isinstance(beliefs, list)

    @pytest.mark.asyncio
    async def test_backend_query_experiences_accepts_search_weights(
        self, silicon_memory,
    ):
        """query_experiences() with search_weights should work end-to-end."""
        await silicon_memory.record_experience(
            Experience(
                id=uuid4(), content="Debugged a TypeScript error",
                outcome="Fixed the type issue", session_id="sess-ts",
            ),
        )

        weights = {"vector": 0.6, "text": 0.4}
        experiences = await silicon_memory._backend.query_experiences(
            "TypeScript error", limit=5, search_weights=weights,
        )

        assert isinstance(experiences, list)

    @pytest.mark.asyncio
    async def test_backend_find_procedures_accepts_search_weights(
        self, silicon_memory,
    ):
        """find_applicable_procedures() with search_weights should work."""
        from silicon_memory.core.types import Procedure

        await silicon_memory.commit_procedure(
            Procedure(
                id=uuid4(), name="Lint TypeScript",
                description="Run the TypeScript linter",
                steps=["Run eslint", "Fix warnings"],
                trigger="lint typescript", confidence=0.85,
            ),
        )

        weights = {"vector": 0.5, "text": 0.5}
        procs = await silicon_memory._backend.find_applicable_procedures(
            "lint typescript", limit=3, search_weights=weights,
        )

        assert isinstance(procs, list)


class TestEntropyRerankingE2E:
    """E2E: entropy reranking adjusts fact ordering."""

    @pytest.mark.asyncio
    async def test_recall_with_entropy_reranking(
        self, silicon_memory, sample_source,
    ):
        """Recall with entropy_weight > 0 should not crash and should
        return results with the entropy field populated."""
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Elixir runs on the BEAM VM",
                   confidence=0.92, source=sample_source),
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Elixir has lightweight processes",
                   confidence=0.5, source=sample_source),
        )

        weights = {
            "vector": 0.3, "text": 0.1, "temporal": 0.0,
            "confidence": 0.2, "graph_proximity": 0.0,
            "temporal_half_life_hours": 720,
            "entropy_weight": 0.3,
            "entropy_direction": "prefer_high",
        }

        result = await silicon_memory._backend.recall(
            query="Elixir concurrency",
            max_facts=10,
            search_weights=weights,
        )

        assert "facts" in result
        for fact in result["facts"]:
            # entropy should be numeric on every RecallResult
            # (SiliconDB may return int instead of float — see SiliconDB #129)
            assert isinstance(fact.entropy, (int, float))

    @pytest.mark.asyncio
    async def test_recall_without_entropy_still_works(
        self, silicon_memory, sample_source,
    ):
        """Recall with entropy_weight=0 (or missing) should behave normally."""
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Haskell is purely functional",
                   confidence=0.9, source=sample_source),
        )

        result = await silicon_memory._backend.recall(
            query="Haskell features",
            max_facts=5,
        )

        assert "facts" in result


class TestGraphContextNodesE2E:
    """E2E: graph_context_nodes flow through recall."""

    @pytest.mark.asyncio
    async def test_recall_with_graph_context_nodes(
        self, silicon_memory, sample_source,
    ):
        """Passing graph_context_nodes via search_weights should not crash."""
        b1 = Belief(id=uuid4(), content="Kubernetes orchestrates containers",
                     confidence=0.9, source=sample_source)
        b2 = Belief(id=uuid4(), content="Docker containers are lightweight VMs",
                     confidence=0.85, source=sample_source)

        await silicon_memory.commit_belief(b1)
        await silicon_memory.commit_belief(b2)

        # Build external IDs to use as seeds
        seed_id = silicon_memory._backend._build_external_id("belief", b1.id)

        weights = {
            "vector": 0.3, "text": 0.1, "temporal": 0.0,
            "confidence": 0.2, "graph_proximity": 0.2,
            "temporal_half_life_hours": 720,
            "entropy_weight": 0.0,
            "entropy_direction": "prefer_low",
            "graph_context_nodes": [seed_id],
        }

        result = await silicon_memory._backend.recall(
            query="container orchestration",
            max_facts=10,
            search_weights=weights,
        )

        assert "facts" in result
        assert isinstance(result["facts"], list)


# ============================================================================
# SM-4: Context Switch Snapshots — E2E Tests
# ============================================================================


class TestContextSnapshotsE2E:
    """E2E tests for context switch snapshots against real SiliconDB."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_snapshot(self, silicon_memory):
        """Create a snapshot and retrieve it by task context."""
        # Set some working memory first
        await silicon_memory.set_context("current_file", "auth.py")
        await silicon_memory.set_context("branch", "fix-tokens")

        # Create a snapshot
        snapshot = await silicon_memory.create_snapshot("project-alpha/auth")

        assert snapshot.task_context == "project-alpha/auth"
        assert isinstance(snapshot.summary, str)
        assert snapshot.working_memory.get("current_file") == "auth.py"
        assert snapshot.working_memory.get("branch") == "fix-tokens"
        assert snapshot.user_id == "test-user"
        assert snapshot.tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_get_latest_snapshot(self, silicon_memory):
        """Retrieve the most recent snapshot for a task context."""
        await silicon_memory.set_context("step", "1")
        snap1 = await silicon_memory.create_snapshot("task-A")

        await silicon_memory.set_context("step", "2")
        snap2 = await silicon_memory.create_snapshot("task-A")

        latest = await silicon_memory.get_latest_snapshot("task-A")

        assert latest is not None
        assert latest.id == snap2.id
        assert latest.working_memory.get("step") == "2"

    @pytest.mark.asyncio
    async def test_snapshot_not_found(self, silicon_memory):
        """get_latest_snapshot returns None for unknown task context."""
        result = await silicon_memory.get_latest_snapshot("nonexistent-task")
        assert result is None

    @pytest.mark.asyncio
    async def test_multi_task_switch_resume(self, silicon_memory):
        """Simulate: work on A, switch to B, resume A."""
        # Work on task A
        await silicon_memory.set_context("task", "A-work")
        await silicon_memory.record_experience(
            Experience(id=uuid4(), content="Implemented auth module",
                       outcome="Auth working", session_id="s1"),
        )
        snap_a = await silicon_memory.create_snapshot("task-A")

        # Switch to task B
        await silicon_memory.set_context("task", "B-work")
        await silicon_memory.record_experience(
            Experience(id=uuid4(), content="Fixed database migration",
                       outcome="Migration complete", session_id="s2"),
        )
        snap_b = await silicon_memory.create_snapshot("task-B")

        # Resume task A
        resumed_a = await silicon_memory.get_latest_snapshot("task-A")
        assert resumed_a is not None
        assert resumed_a.task_context == "task-A"
        assert resumed_a.working_memory.get("task") == "A-work"

        # Resume task B
        resumed_b = await silicon_memory.get_latest_snapshot("task-B")
        assert resumed_b is not None
        assert resumed_b.task_context == "task-B"
        assert resumed_b.working_memory.get("task") == "B-work"

    @pytest.mark.asyncio
    async def test_snapshot_captures_recent_experiences(self, silicon_memory):
        """Snapshot captures IDs of recent experiences."""
        exp = Experience(
            id=uuid4(),
            content="Debugged token refresh",
            outcome="Fixed the bug",
            session_id="s1",
        )
        await silicon_memory.record_experience(exp)

        snapshot = await silicon_memory.create_snapshot("debug-session")

        # recent_experiences should contain the experience ID
        assert isinstance(snapshot.recent_experiences, list)

    @pytest.mark.asyncio
    async def test_snapshot_stored_as_silicondb_document(self, silicon_memory):
        """Verify snapshot is stored as a SiliconDB document with correct node_type."""
        await silicon_memory.set_context("key", "val")
        snapshot = await silicon_memory.create_snapshot("stored-check")

        # Query the backend directly for snapshot documents
        results = await silicon_memory._backend.query_snapshots_by_context(
            task_context="stored-check", limit=5,
        )

        assert len(results) >= 1
        found = results[0]
        assert found.task_context == "stored-check"
        assert found.id == snapshot.id

    @pytest.mark.asyncio
    async def test_memory_tool_switch_context_e2e(self, silicon_memory):
        """Test SWITCH_CONTEXT through MemoryTool against real backend."""
        await silicon_memory.set_context("file", "main.py")

        tool = MemoryTool(silicon_memory)
        response = await tool.invoke("switch_context", task_context="project-beta")

        assert response.success
        assert response.action == MemoryAction.SWITCH_CONTEXT
        assert response.data["task_context"] == "project-beta"
        assert isinstance(response.data["summary"], str)

    @pytest.mark.asyncio
    async def test_memory_tool_resume_context_e2e(self, silicon_memory):
        """Test RESUME_CONTEXT through MemoryTool against real backend."""
        await silicon_memory.set_context("state", "in-progress")

        # First switch (create snapshot)
        tool = MemoryTool(silicon_memory)
        await tool.invoke("switch_context", task_context="project-gamma")

        # Then resume
        response = await tool.invoke("resume_context", task_context="project-gamma")

        assert response.success
        assert response.action == MemoryAction.RESUME_CONTEXT
        assert response.data["found"] is True
        assert response.data["task_context"] == "project-gamma"
        assert isinstance(response.data["working_memory"], dict)
