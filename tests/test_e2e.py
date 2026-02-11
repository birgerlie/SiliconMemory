"""End-to-end tests for silicon-memory.

These tests verify the complete flow from LLM tools through SiliconMemory
to SiliconDB storage and back.

Run with:
    SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e.py -v

For real embeddings (slower, more accurate):
    USE_REAL_EMBEDDINGS=1 SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest

from silicon_memory.core.types import (
    Belief,
    Experience,
    Procedure,
    Source,
    Triplet,
)
from silicon_memory.tools.memory_tool import MemoryTool, MemoryAction
from silicon_memory.tools.query_tool import QueryTool, QueryFormat


pytestmark = pytest.mark.e2e


# ============================================================================
# MemoryTool E2E Tests
# ============================================================================


class TestMemoryToolE2E:
    """End-to-end tests for the MemoryTool LLM interface."""

    @pytest.mark.asyncio
    async def test_store_and_recall_fact(self, silicon_memory):
        """Test storing and recalling a fact through the memory tool."""
        tool = MemoryTool(silicon_memory)

        # Store a fact using the tool
        response = await tool.invoke(
            "store_fact",
            subject="Django",
            predicate="is",
            object="Python web framework",
            confidence=0.9,
            tags=["python", "web", "framework"],
        )

        assert response.success
        assert response.action == MemoryAction.STORE_FACT
        assert "belief_id" in response.data

        # Recall
        recall_response = await tool.invoke(
            "recall",
            query="Django web framework",
            max_facts=10,
        )

        assert recall_response.success
        assert recall_response.action == MemoryAction.RECALL

    @pytest.mark.asyncio
    async def test_store_fact_with_content(self, silicon_memory):
        """Test storing a fact with content (no triplet)."""
        tool = MemoryTool(silicon_memory)

        response = await tool.invoke(
            "store_fact",
            content="Flask is a lightweight Python web framework.",
            confidence=0.85,
            tags=["python", "web"],
        )

        assert response.success
        assert response.data["content"] == "Flask is a lightweight Python web framework."
        assert response.data["triplet"] is None

    @pytest.mark.asyncio
    async def test_store_experience(self, silicon_memory):
        """Test storing an experience through the memory tool."""
        tool = MemoryTool(silicon_memory)

        response = await tool.invoke(
            "store_experience",
            content="User asked about database connection pooling",
            outcome="Explained SQLAlchemy connection pool configuration",
            emotional_valence=0.4,
            importance=0.7,
            session_id="e2e-test-session",
        )

        assert response.success
        assert response.action == MemoryAction.STORE_EXPERIENCE
        assert "experience_id" in response.data

    @pytest.mark.asyncio
    async def test_store_procedure(self, silicon_memory):
        """Test storing a procedure through the memory tool."""
        tool = MemoryTool(silicon_memory)

        response = await tool.invoke(
            "store_procedure",
            name="Deploy Flask App to Heroku",
            steps=[
                "Create requirements.txt with dependencies",
                "Create Procfile with: web: gunicorn app:app",
                "Initialize git repo if needed",
                "Create Heroku app: heroku create",
                "Push to Heroku: git push heroku main",
            ],
            trigger="deploy flask heroku",
            description="Steps to deploy a Flask application to Heroku",
            confidence=0.85,
        )

        assert response.success
        assert response.action == MemoryAction.STORE_PROCEDURE
        assert response.data["name"] == "Deploy Flask App to Heroku"
        assert len(response.data["steps"]) == 5

    @pytest.mark.asyncio
    async def test_context_operations(self, silicon_memory):
        """Test working memory context operations."""
        tool = MemoryTool(silicon_memory)

        # Set context
        set_response = await tool.invoke(
            "set_context",
            key="current_project",
            value="silicon-memory",
            ttl_seconds=600,
        )

        assert set_response.success
        assert set_response.action == MemoryAction.SET_CONTEXT

        # Get context
        get_response = await tool.invoke(
            "get_context",
            key="current_project",
        )

        assert get_response.success
        assert get_response.action == MemoryAction.GET_CONTEXT
        assert get_response.data["value"] == "silicon-memory"

    @pytest.mark.asyncio
    async def test_what_do_you_know(self, silicon_memory):
        """Test the what_do_you_know action."""
        tool = MemoryTool(silicon_memory)

        # First store some knowledge
        await tool.invoke(
            "store_fact",
            subject="Redis",
            predicate="is",
            object="in-memory data store",
            confidence=0.95,
        )

        await tool.invoke(
            "store_fact",
            subject="Redis",
            predicate="is used for",
            object="caching",
            confidence=0.9,
        )

        # Ask what we know
        response = await tool.invoke(
            "what_do_you_know",
            query="Redis caching",
            min_confidence=0.5,
        )

        assert response.success
        assert response.action == MemoryAction.WHAT_DO_YOU_KNOW
        assert "query" in response.data
        assert "report" in response.data

    @pytest.mark.asyncio
    async def test_invalid_action(self, silicon_memory):
        """Test handling of invalid action."""
        tool = MemoryTool(silicon_memory)

        response = await tool.invoke("invalid_action")

        assert not response.success
        assert "Unknown action" in response.error

    @pytest.mark.asyncio
    async def test_openai_schema(self, silicon_memory):
        """Test that the OpenAI function schema is valid."""
        schema = MemoryTool.get_openai_schema()

        assert schema["name"] == "memory"
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "action" in schema["parameters"]["properties"]

        # Check all actions are in enum
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "recall" in actions
        assert "store_fact" in actions
        assert "store_experience" in actions
        assert "store_procedure" in actions
        assert "what_do_you_know" in actions


# ============================================================================
# QueryTool E2E Tests
# ============================================================================


class TestQueryToolE2E:
    """End-to-end tests for the QueryTool."""

    @pytest.mark.asyncio
    async def test_basic_query(self, silicon_memory, sample_source):
        """Test basic query functionality."""
        # Pre-populate with data
        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                triplet=Triplet("PostgreSQL", "is", "relational database"),
                confidence=0.95,
                source=sample_source,
            )
        )

        tool = QueryTool(silicon_memory)
        response = await tool.query("PostgreSQL database", min_confidence=0.3)

        assert response.query == "PostgreSQL database"
        assert isinstance(response.total_confidence, float)
        assert isinstance(response.beliefs, list)

    @pytest.mark.asyncio
    async def test_query_response_formats(self, silicon_memory, sample_source):
        """Test different query response formats."""
        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                content="MongoDB is a NoSQL document database.",
                confidence=0.9,
                source=sample_source,
            )
        )

        tool = QueryTool(silicon_memory)
        response = await tool.query("MongoDB NoSQL")

        # Test full format
        full = response.to_dict(QueryFormat.FULL)
        assert "query" in full
        assert "beliefs" in full
        assert "sources" in full

        # Test summary format
        summary = response.to_dict(QueryFormat.SUMMARY)
        assert "query" in summary
        assert "total_confidence" in summary
        assert "belief_count" in summary

        # Test report format
        report = response.to_dict(QueryFormat.REPORT)
        assert "query" in report
        assert "report" in report

        # Test citations format
        citations = response.to_dict(QueryFormat.CITATIONS)
        assert "query" in citations
        assert "sources" in citations

    @pytest.mark.asyncio
    async def test_verify_claim(self, silicon_memory, sample_source):
        """Test claim verification."""
        # Store supporting beliefs
        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                triplet=Triplet("Python", "is", "interpreted language"),
                confidence=0.95,
                source=sample_source,
            )
        )

        tool = QueryTool(silicon_memory)

        # Verify a claim
        result = await tool.verify_claim(
            "Python is an interpreted language",
            min_confidence=0.5,
        )

        assert "claim" in result
        assert "status" in result
        assert "verification_score" in result
        assert result["status"] in ["supported", "weakly_supported", "contested", "unsupported", "unknown"]

    @pytest.mark.asyncio
    async def test_query_entity(self, silicon_memory, sample_source):
        """Test querying all knowledge about an entity."""
        # Store multiple facts about an entity
        entity = "Kubernetes"

        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                triplet=Triplet(entity, "is", "container orchestration platform"),
                confidence=0.95,
                source=sample_source,
            )
        )

        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                triplet=Triplet(entity, "was developed by", "Google"),
                confidence=0.9,
                source=sample_source,
            )
        )

        tool = QueryTool(silicon_memory)
        response = await tool.query_entity(entity)

        assert response.query == f"Entity: {entity}"
        assert response.report is not None

    @pytest.mark.asyncio
    async def test_context_string_generation(self, silicon_memory, sample_source):
        """Test generating context strings for LLM prompts."""
        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                content="Docker containers package applications with their dependencies.",
                confidence=0.85,
                source=sample_source,
            )
        )

        tool = QueryTool(silicon_memory)
        response = await tool.query("Docker containers")

        context_string = response.as_context_string()

        assert "Knowledge about: Docker containers" in context_string
        assert "Confidence:" in context_string


# ============================================================================
# Full Workflow E2E Tests
# ============================================================================


class TestFullWorkflowE2E:
    """End-to-end tests for complete LLM interaction workflows."""

    @pytest.mark.asyncio
    async def test_learning_workflow(self, silicon_memory):
        """Test a complete learning workflow: learn -> recall -> update."""
        memory_tool = MemoryTool(silicon_memory)
        query_tool = QueryTool(silicon_memory)

        # Step 1: Learn some facts
        await memory_tool.invoke(
            "store_fact",
            subject="GraphQL",
            predicate="is",
            object="query language for APIs",
            confidence=0.9,
        )

        await memory_tool.invoke(
            "store_fact",
            content="GraphQL was developed by Facebook in 2012.",
            confidence=0.85,
        )

        # Step 2: Record the learning experience
        await memory_tool.invoke(
            "store_experience",
            content="Learned about GraphQL from documentation",
            outcome="Acquired knowledge about GraphQL basics",
            importance=0.7,
        )

        # Step 3: Query what we know
        knowledge = await query_tool.query("GraphQL API")

        assert knowledge is not None

        # Step 4: Verify our understanding
        verification = await query_tool.verify_claim(
            "GraphQL is a query language",
            min_confidence=0.5,
        )

        assert verification is not None

    @pytest.mark.asyncio
    async def test_procedure_execution_workflow(self, silicon_memory):
        """Test a procedure-based workflow: learn procedure -> recall -> record outcome."""
        tool = MemoryTool(silicon_memory)

        # Step 1: Store a procedure
        await tool.invoke(
            "store_procedure",
            name="Set up CI/CD with GitHub Actions",
            steps=[
                "Create .github/workflows directory",
                "Create workflow YAML file",
                "Define trigger events",
                "Add build and test steps",
                "Configure deployment if needed",
                "Commit and push to trigger workflow",
            ],
            trigger="github actions CI/CD",
            confidence=0.8,
        )

        # Step 2: Set context for current task
        await tool.invoke(
            "set_context",
            key="current_task",
            value="setting up CI/CD",
        )

        # Step 3: Recall relevant procedures
        recall = await tool.invoke(
            "recall",
            query="GitHub Actions CI/CD setup",
            max_procedures=5,
        )

        assert recall.success

        # Step 4: Record the experience of using the procedure
        await tool.invoke(
            "store_experience",
            content="Followed CI/CD setup procedure for user",
            outcome="Successfully configured GitHub Actions workflow",
            emotional_valence=0.7,
            importance=0.8,
        )

    @pytest.mark.asyncio
    async def test_multi_session_context(self, silicon_memory):
        """Test context management across multiple interactions."""
        tool = MemoryTool(silicon_memory)

        # Simulate first interaction
        await tool.invoke("set_context", key="user_expertise", value="intermediate")
        await tool.invoke("set_context", key="preferred_language", value="Python")
        await tool.invoke("set_context", key="project_type", value="web application")

        # Check context is set
        response = await tool.invoke("get_context")
        context = response.data.get("context", {})

        assert context.get("user_expertise") == "intermediate"
        assert context.get("preferred_language") == "Python"

        # Update context (simulating later interaction)
        await tool.invoke("set_context", key="user_expertise", value="advanced")

        # Verify update
        response = await tool.invoke("get_context", key="user_expertise")
        assert response.data["value"] == "advanced"

    @pytest.mark.asyncio
    async def test_knowledge_accumulation(self, silicon_memory):
        """Test accumulating knowledge over multiple interactions."""
        memory_tool = MemoryTool(silicon_memory)
        query_tool = QueryTool(silicon_memory)

        topics = [
            ("SQLAlchemy", "is", "Python ORM"),
            ("SQLAlchemy", "supports", "PostgreSQL"),
            ("SQLAlchemy", "supports", "MySQL"),
            ("SQLAlchemy", "provides", "connection pooling"),
        ]

        # Accumulate knowledge
        for subject, predicate, obj in topics:
            await memory_tool.invoke(
                "store_fact",
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=0.9,
            )

        # Query accumulated knowledge
        knowledge = await query_tool.query("SQLAlchemy ORM database")

        assert knowledge is not None
        # The knowledge proof should have information about our stored facts

    @pytest.mark.asyncio
    async def test_response_serialization(self, silicon_memory):
        """Test that tool responses are JSON-serializable for LLM consumption."""
        import json

        tool = MemoryTool(silicon_memory)

        # Store a fact
        store_response = await tool.invoke(
            "store_fact",
            subject="Test",
            predicate="is",
            object="serializable",
            confidence=0.8,
        )

        # Convert to dict and serialize to JSON
        response_dict = store_response.to_dict()
        json_str = json.dumps(response_dict)

        assert json_str is not None
        assert "success" in json_str
        assert "action" in json_str

        # Deserialize and verify
        parsed = json.loads(json_str)
        assert parsed["success"] is True
        assert parsed["action"] == "store_fact"


# ============================================================================
# Performance E2E Tests
# ============================================================================


class TestPerformanceE2E:
    """Performance-oriented end-to-end tests."""

    @pytest.mark.asyncio
    async def test_bulk_fact_storage(self, silicon_memory):
        """Test storing many facts in sequence."""
        tool = MemoryTool(silicon_memory)

        # Store 50 facts
        for i in range(50):
            response = await tool.invoke(
                "store_fact",
                subject=f"Entity{i}",
                predicate="has_property",
                object=f"Value{i}",
                confidence=0.8,
            )
            assert response.success

    @pytest.mark.asyncio
    async def test_concurrent_context_operations(self, silicon_memory):
        """Test concurrent working memory operations."""
        tool = MemoryTool(silicon_memory)

        # Set multiple context values concurrently
        async def set_context(key: str, value: str):
            return await tool.invoke("set_context", key=key, value=value)

        tasks = [
            set_context(f"key_{i}", f"value_{i}")
            for i in range(20)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.success for r in results)

        # Verify all are retrievable
        response = await tool.invoke("get_context")
        context = response.data.get("context", {})

        for i in range(20):
            assert context.get(f"key_{i}") == f"value_{i}"

    @pytest.mark.asyncio
    async def test_rapid_recall(self, silicon_memory, sample_source):
        """Test rapid consecutive recalls."""
        tool = MemoryTool(silicon_memory)

        # Store some data first
        await tool.invoke(
            "store_fact",
            content="Test content for rapid recall testing.",
            confidence=0.8,
        )

        # Perform 20 rapid recalls
        for i in range(20):
            response = await tool.invoke(
                "recall",
                query=f"test query {i}",
                max_facts=5,
            )
            assert response.success
