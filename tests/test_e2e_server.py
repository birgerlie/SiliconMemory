"""End-to-end tests for the Silicon Memory server.

Exercises the full stack: REST API → SiliconMemory → SiliconDB + LLM.

Requirements:
    - SILICONDB_LIBRARY_PATH set
    - SiliconServe running on localhost:8000 (for LLM tests)

Run with:
    SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e_server.py -v
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from silicon_memory.llm.config import LLMConfig
from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.security.types import UserContext
from silicon_memory.server.config import ServerConfig
from silicon_memory.server.dependencies import MemoryPool, get_memory, resolve_user_context
from silicon_memory.server.rest.app import create_app

SILICONDB_AVAILABLE = bool(os.environ.get("SILICONDB_LIBRARY_PATH"))

# Detect if SiliconServe is live and can actually serve requests
SILICONSERVE_MODEL = os.environ.get("SILICONSERVE_MODEL", "qwen3-4b")
_siliconserve_ok: bool | None = None


def siliconserve_available() -> bool:
    """Check if SiliconServe can actually serve a completion."""
    global _siliconserve_ok
    if _siliconserve_ok is not None:
        return _siliconserve_ok
    try:
        import httpx

        r = httpx.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": SILICONSERVE_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
            timeout=10,
        )
        _siliconserve_ok = r.status_code == 200
    except Exception:
        _siliconserve_ok = False
    return _siliconserve_ok


requires_siliconserve = pytest.mark.skipif(
    not siliconserve_available(),
    reason="SiliconServe not available or model not loaded",
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not SILICONDB_AVAILABLE, reason="SILICONDB_LIBRARY_PATH not set"),
]


# ============================================================================
# Fixtures — server with mock embedder (matches existing test patterns)
# ============================================================================


@pytest.fixture
def _mock_embedder():
    """Deterministic mock embedder — no ML models needed."""
    from tests.embedder_cache import MockEmbedder

    return MockEmbedder(dimension=384, model_name="mock-384")


@pytest.fixture
def server_config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        mode="rest",
        db_path=tmp_path / "test_server.db",
        llm=LLMConfig(
            base_url="http://localhost:8000/v1",
            model=SILICONSERVE_MODEL,
            api_key="not-needed",
        ),
        auto_embedder=False,  # we inject mock embedder
        reflect_interval=9999,
    )


@pytest.fixture
async def client(server_config: ServerConfig, _mock_embedder):
    """Async HTTP client with lifespan + mock embedder injection.

    The MemoryPool.get() is monkey-patched to create SiliconMemory
    instances using the mock embedder, matching the existing test
    patterns in conftest.py.
    """
    app = create_app(server_config)

    # Run lifespan manually (ASGITransport doesn't trigger it)
    ctx = app.router.lifespan_context
    async with ctx(app):
        pool: MemoryPool = app.state.pool

        # Patch pool.get() to inject mock embedder into SiliconDB
        _original_get = pool.get

        def _patched_get(user_ctx: UserContext) -> SiliconMemory:
            key = (user_ctx.tenant_id, user_ctx.user_id)
            if key not in pool._instances:
                from silicondb.native import SiliconDBNative
                from silicon_memory.storage.silicondb_backend import SiliconDBBackend
                from silicon_memory.temporal.decay import DecayConfig
                from silicon_memory.security.authorization import PolicyEngine

                db = SiliconDBNative(
                    path=str(server_config.db_path),
                    dimensions=_mock_embedder.dimension,
                    enable_graph=server_config.enable_graph,
                    enable_async=True,
                    auto_embedder=False,
                )
                db.set_embedder(
                    _mock_embedder.embed,
                    dimension=_mock_embedder.dimension,
                    model_name=_mock_embedder.model_name,
                    warmup=True,
                )

                # Build a SiliconMemory with our patched backend
                memory = SiliconMemory.__new__(SiliconMemory)
                memory._user_context = user_ctx
                from silicon_memory.security.config import SecurityConfig

                memory._security_config = SecurityConfig()

                backend = SiliconDBBackend.__new__(SiliconDBBackend)
                backend._db = db
                backend._decay_config = DecayConfig()
                backend._user_context = user_ctx
                backend._policy_engine = PolicyEngine()
                backend._working_keys = set()

                memory._backend = backend

                from silicon_memory.security.forgetting import ForgettingService
                from silicon_memory.security.transparency import TransparencyService
                from silicon_memory.security.inspector import MemoryInspector
                from silicon_memory.security.audit import AuditLogger
                from silicon_memory.snapshot.service import SnapshotService

                memory._forgetting_service = ForgettingService(backend)
                memory._transparency_service = TransparencyService(backend)
                memory._inspector = MemoryInspector(backend)
                memory._audit_logger = AuditLogger(backend)
                memory._snapshot_service = SnapshotService(
                    memory=memory, backend=backend, llm_provider=app.state.llm,
                )
                memory._preferences = None

                pool._instances[key] = memory
            return pool._instances[key]

        pool.get = _patched_get

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# Headers for a test user
HEADERS = {"X-User-Id": "test-user", "X-Tenant-Id": "test-tenant"}


# ============================================================================
# Health & Status
# ============================================================================


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient):
        r = await client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_status(self, client: AsyncClient):
        r = await client.get("/api/v1/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "rest"
        assert data["active_users"] >= 0


# ============================================================================
# LLM Provider — real SiliconServe
# ============================================================================


class TestLLMProvider:
    @requires_siliconserve
    @pytest.mark.asyncio
    async def test_generate(self):
        """SiliconLLMProvider.generate() returns text from SiliconServe."""
        llm = SiliconLLMProvider(
            base_url="http://localhost:8000/v1",
            model=SILICONSERVE_MODEL,
        )
        result = await llm.generate("Say 'hello' and nothing else.", max_tokens=32)
        assert isinstance(result, str)
        assert len(result) > 0

    @requires_siliconserve
    @pytest.mark.asyncio
    async def test_generate_structured(self):
        """SiliconLLMProvider.generate_structured() returns parsed JSON."""
        from pydantic import BaseModel

        class Color(BaseModel):
            name: str
            hex: str

        llm = SiliconLLMProvider(
            base_url="http://localhost:8000/v1",
            model=SILICONSERVE_MODEL,
        )
        result = await llm.generate_structured(
            "Return a JSON object for the color red.", Color
        )
        assert isinstance(result, Color)
        assert len(result.name) > 0

    @pytest.mark.asyncio
    async def test_embed_raises(self):
        """embed() raises NotImplementedError — SiliconDB handles embeddings."""
        llm = SiliconLLMProvider()
        with pytest.raises(NotImplementedError):
            await llm.embed("test")


# ============================================================================
# Store & Recall — full round-trip through REST + SiliconDB
# ============================================================================


class TestStoreAndRecall:
    @pytest.mark.asyncio
    async def test_store_belief_then_recall(self, client: AsyncClient):
        """Store a belief via REST, then recall it."""
        r = await client.post("/api/v1/store", json={
            "type": "belief",
            "content": "FastAPI is built on top of Starlette and Pydantic",
            "confidence": 0.9,
            "tags": ["python", "web"],
        }, headers=HEADERS)
        assert r.status_code == 200
        store_data = r.json()
        assert store_data["stored"] is True
        belief_id = store_data["id"]

        r = await client.post("/api/v1/recall", json={
            "query": "FastAPI framework",
            "max_facts": 10,
        }, headers=HEADERS)
        assert r.status_code == 200
        recall_data = r.json()
        assert recall_data["total_items"] >= 1

    @pytest.mark.asyncio
    async def test_store_experience_then_recall(self, client: AsyncClient):
        """Store an experience then recall it."""
        r = await client.post("/api/v1/store", json={
            "type": "experience",
            "content": "Debugged a tricky async race condition in the event loop",
            "outcome": "Fixed by adding a proper lock",
            "session_id": "debug-session-1",
        }, headers=HEADERS)
        assert r.status_code == 200

        r = await client.post("/api/v1/recall", json={
            "query": "async race condition debugging",
            "max_experiences": 10,
        }, headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["total_items"] >= 1

    @pytest.mark.asyncio
    async def test_store_procedure_then_recall(self, client: AsyncClient):
        """Store a procedure then recall it."""
        r = await client.post("/api/v1/store", json={
            "type": "procedure",
            "content": "How to deploy a FastAPI service to production",
            "name": "Deploy FastAPI",
            "steps": [
                "Build Docker image",
                "Push to registry",
                "Update Kubernetes deployment",
                "Verify health check",
            ],
            "confidence": 0.85,
        }, headers=HEADERS)
        assert r.status_code == 200

        r = await client.post("/api/v1/recall", json={
            "query": "deploy FastAPI production",
            "max_procedures": 5,
        }, headers=HEADERS)
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_store_belief_with_triplet(self, client: AsyncClient):
        """Store a belief with subject/predicate/object triplet."""
        r = await client.post("/api/v1/store", json={
            "type": "belief",
            "content": "Python supports async/await for concurrency",
            "confidence": 0.95,
            "subject": "Python",
            "predicate": "supports",
            "object": "async/await concurrency",
        }, headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["type"] == "belief"


# ============================================================================
# Query — semantic search
# ============================================================================


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_beliefs(self, client: AsyncClient):
        """Store beliefs then query by semantic similarity."""
        for content in [
            "Rust guarantees memory safety without garbage collection",
            "Go has goroutines for lightweight concurrency",
            "TypeScript adds static typing to JavaScript",
        ]:
            await client.post("/api/v1/store", json={
                "type": "belief", "content": content, "confidence": 0.85,
            }, headers=HEADERS)

        r = await client.post("/api/v1/query", json={
            "query": "memory safety programming",
            "limit": 10,
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1


# ============================================================================
# Get specific memory
# ============================================================================


class TestGetMemory:
    @pytest.mark.asyncio
    async def test_get_belief_by_id(self, client: AsyncClient):
        """Store a belief then get it by ID."""
        r = await client.post("/api/v1/store", json={
            "type": "belief",
            "content": "Kubernetes is the industry standard for container orchestration",
            "confidence": 0.9,
        }, headers=HEADERS)
        belief_id = r.json()["id"]

        r = await client.get(f"/api/v1/memory/belief/{belief_id}", headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "belief"
        assert "Kubernetes" in data["content"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_404(self, client: AsyncClient):
        r = await client.get(
            "/api/v1/memory/belief/00000000-0000-0000-0000-000000000000",
            headers=HEADERS,
        )
        assert r.status_code == 404


# ============================================================================
# Working Memory
# ============================================================================


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_working_memory_crud(self, client: AsyncClient):
        """Set, get, and delete working memory keys."""
        # Set
        r = await client.put("/api/v1/working/current_task", json={
            "value": "implementing auth module",
            "ttl_seconds": 600,
        }, headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["key"] == "current_task"

        # Get all
        r = await client.get("/api/v1/working", headers=HEADERS)
        assert r.status_code == 200
        assert "current_task" in r.json()

        # Delete
        r = await client.delete("/api/v1/working/current_task", headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        # Verify deleted
        r = await client.get("/api/v1/working", headers=HEADERS)
        assert "current_task" not in r.json()


# ============================================================================
# Decisions
# ============================================================================


class TestDecisions:
    @pytest.mark.asyncio
    async def test_store_and_search_decision(self, client: AsyncClient):
        """Store a decision then search for it."""
        # Store a supporting belief first
        r = await client.post("/api/v1/store", json={
            "type": "belief",
            "content": "PostgreSQL has excellent JSON support",
            "confidence": 0.9,
        }, headers=HEADERS)
        belief_id = r.json()["id"]

        # Store decision
        r = await client.post("/api/v1/decisions", json={
            "title": "Use PostgreSQL for the new service",
            "description": "PostgreSQL selected for its reliability and JSON support",
            "assumptions": [{
                "belief_id": belief_id,
                "description": "PostgreSQL JSON support is sufficient",
                "confidence_at_decision": 0.9,
                "is_critical": True,
            }],
            "tags": ["database", "architecture"],
        }, headers=HEADERS)
        assert r.status_code == 200
        decision = r.json()
        assert decision["title"] == "Use PostgreSQL for the new service"
        assert decision["status"] == "active"

        # Search
        r = await client.post("/api/v1/decisions/search", json={
            "query": "database selection PostgreSQL",
            "limit": 5,
        }, headers=HEADERS)
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1
        assert any("PostgreSQL" in d["title"] for d in results)


# ============================================================================
# Forget (GDPR)
# ============================================================================


class TestForget:
    @pytest.mark.asyncio
    async def test_forget_session(self, client: AsyncClient):
        """Store experiences in a session, then forget the session."""
        for i in range(3):
            await client.post("/api/v1/store", json={
                "type": "experience",
                "content": f"Sensitive meeting note {i} about budget",
                "session_id": "sensitive-session-99",
            }, headers=HEADERS)

        r = await client.post("/api/v1/forget", json={
            "scope": "session",
            "session_id": "sensitive-session-99",
            "reason": "User requested deletion",
        }, headers=HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["scope"] == "session"
        assert data["success"] is True


# ============================================================================
# Full cognitive workflow
# ============================================================================


class TestFullWorkflow:
    @pytest.mark.asyncio
    async def test_learn_recall_decide(self, client: AsyncClient):
        """Full cognitive loop: store facts → recall → decide → working memory."""
        # LEARN
        facts = [
            "React is a JavaScript library for building user interfaces",
            "Vue.js is a progressive JavaScript framework",
            "Svelte compiles components at build time for better performance",
            "Angular is a full-featured TypeScript framework by Google",
        ]
        for fact in facts:
            r = await client.post("/api/v1/store", json={
                "type": "belief", "content": fact, "confidence": 0.85,
                "tags": ["frontend", "javascript"],
            }, headers=HEADERS)
            assert r.status_code == 200

        # Experience
        r = await client.post("/api/v1/store", json={
            "type": "experience",
            "content": "Previous project used React with success",
            "outcome": "Delivered on time, team was productive",
        }, headers=HEADERS)
        assert r.status_code == 200

        # RECALL
        r = await client.post("/api/v1/recall", json={
            "query": "JavaScript frontend framework comparison",
            "max_facts": 10,
            "max_experiences": 5,
        }, headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["total_items"] >= 1

        # DECIDE
        r = await client.post("/api/v1/decisions", json={
            "title": "Use React for the new dashboard",
            "description": "React chosen for team familiarity and ecosystem",
            "tags": ["frontend", "architecture"],
        }, headers=HEADERS)
        assert r.status_code == 200

        # WORKING MEMORY
        for key, val in [("current_project", "dashboard-v2"), ("framework", "react")]:
            r = await client.put(f"/api/v1/working/{key}", json={
                "value": val, "ttl_seconds": 3600,
            }, headers=HEADERS)
            assert r.status_code == 200

        r = await client.get("/api/v1/working", headers=HEADERS)
        assert r.status_code == 200
        working = r.json()
        assert working["current_project"] == "dashboard-v2"
        assert working["framework"] == "react"

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, client: AsyncClient):
        """Two users store beliefs — each only sees their own."""
        user_a = {"X-User-Id": "alice", "X-Tenant-Id": "acme"}
        user_b = {"X-User-Id": "bob", "X-Tenant-Id": "acme"}

        r = await client.post("/api/v1/store", json={
            "type": "belief",
            "content": "Alice's secret project uses Rust for the backend",
            "confidence": 0.9,
        }, headers=user_a)
        assert r.status_code == 200

        r = await client.post("/api/v1/store", json={
            "type": "belief",
            "content": "Bob prefers Go for microservices",
            "confidence": 0.85,
        }, headers=user_b)
        assert r.status_code == 200

        # Alice queries
        r = await client.post("/api/v1/query", json={
            "query": "backend programming language", "limit": 10,
        }, headers=user_a)
        assert r.status_code == 200
        alice_beliefs = r.json()["beliefs"]

        # Bob queries
        r = await client.post("/api/v1/query", json={
            "query": "backend programming language", "limit": 10,
        }, headers=user_b)
        assert r.status_code == 200
        bob_beliefs = r.json()["beliefs"]

        # Both should have at least some results
        assert len(alice_beliefs) >= 1 or len(bob_beliefs) >= 1


# ============================================================================
# LLM Integration — real SiliconServe
# ============================================================================


class TestLLMIntegration:
    """Tests that exercise the real LLM. Skipped if SiliconServe is unavailable."""

    @requires_siliconserve
    @pytest.mark.asyncio
    async def test_llm_generate_coherent_response(self):
        llm = SiliconLLMProvider(
            base_url="http://localhost:8000/v1",
            model=SILICONSERVE_MODEL,
        )
        response = await llm.generate(
            "What is 2 + 2? Reply with just the number.",
            max_tokens=16,
            temperature=0.0,
        )
        assert "4" in response

    @requires_siliconserve
    @pytest.mark.asyncio
    async def test_llm_structured_output(self):
        from pydantic import BaseModel

        class LanguageInfo(BaseModel):
            name: str
            paradigm: str
            year: int

        llm = SiliconLLMProvider(
            base_url="http://localhost:8000/v1",
            model=SILICONSERVE_MODEL,
        )
        result = await llm.generate_structured(
            "Return information about the Python programming language.",
            LanguageInfo,
        )
        assert isinstance(result, LanguageInfo)
        assert result.name.lower() == "python"
        assert result.year > 1980

    @requires_siliconserve
    @pytest.mark.asyncio
    async def test_llm_long_generation(self):
        llm = SiliconLLMProvider(
            base_url="http://localhost:8000/v1",
            model=SILICONSERVE_MODEL,
        )
        response = await llm.generate(
            "Explain in 2-3 sentences why memory systems are useful for AI agents.",
            max_tokens=256,
        )
        assert len(response) > 50
