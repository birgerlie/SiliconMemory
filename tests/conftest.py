"""Pytest fixtures for silicon-memory tests.

Provides fixtures for:
- Core types (Source, Triplet, Belief, Experience, Procedure)
- Fake clock for temporal testing
- SiliconDB with cached embeddings for integration tests
- SiliconMemory instances for E2E tests
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from uuid import uuid4

import pytest

from silicon_memory.core.types import (
    Belief,
    Experience,
    Procedure,
    Source,
    TemporalContext,
    Triplet,
)
from silicon_memory.temporal.clock import FakeClock

from tests.embedder_cache import FileCachedEmbedder, MockEmbedder


# ============================================================================
# Environment checks
# ============================================================================

# Check if SiliconDB library is available
SILICONDB_AVAILABLE = bool(os.environ.get("SILICONDB_LIBRARY_PATH"))

# Check if we should use real embeddings (slow but accurate)
USE_REAL_EMBEDDINGS = bool(os.environ.get("USE_REAL_EMBEDDINGS"))


# ============================================================================
# Core type fixtures
# ============================================================================


@pytest.fixture
def fake_clock() -> FakeClock:
    """Create a fake clock for testing."""
    return FakeClock(datetime(2024, 1, 1, 12, 0, 0))


@pytest.fixture
def sample_source() -> Source:
    """Create a sample source."""
    return Source(
        id="test-source-1",
        type="observation",
        reliability=0.9,
        metadata={"name": "Test Source", "url": "https://example.com"},
    )


@pytest.fixture
def sample_triplet() -> Triplet:
    """Create a sample triplet."""
    return Triplet(
        subject="Python",
        predicate="is",
        object="programming language",
    )


@pytest.fixture
def sample_belief(sample_source: Source, sample_triplet: Triplet) -> Belief:
    """Create a sample belief."""
    return Belief(
        id=uuid4(),
        triplet=sample_triplet,
        confidence=0.8,
        source=sample_source,
        tags=["programming", "language"],
    )


@pytest.fixture
def sample_belief_with_content(sample_source: Source) -> Belief:
    """Create a sample belief with content instead of triplet."""
    return Belief(
        id=uuid4(),
        content="Python is a popular programming language for data science.",
        confidence=0.85,
        source=sample_source,
        tags=["programming", "data-science"],
    )


@pytest.fixture
def sample_experience() -> Experience:
    """Create a sample experience."""
    return Experience(
        id=uuid4(),
        content="User asked about Python",
        outcome="Provided information about Python",
        session_id="session-1",
    )


@pytest.fixture
def sample_procedure() -> Procedure:
    """Create a sample procedure."""
    return Procedure(
        id=uuid4(),
        name="Install Python Package",
        description="How to install a Python package using pip",
        steps=[
            "Open terminal",
            "Run pip install package-name",
            "Verify installation",
        ],
        trigger="install python package",
        confidence=0.9,
    )


# ============================================================================
# Embedder fixtures
# ============================================================================

# Global embedder instances (session-scoped in effect)
_mock_embedder = None
_cached_embedder = None


@pytest.fixture(scope="session")
def mock_embedder() -> MockEmbedder:
    """Session-scoped mock embedder for fast tests.

    Uses deterministic hashing - no ML models needed.
    """
    global _mock_embedder
    if _mock_embedder is None:
        _mock_embedder = MockEmbedder(dimension=384, model_name="mock-384")
    return _mock_embedder


@pytest.fixture(scope="session")
def cached_embedder():
    """Session-scoped cached E5 embedder for integration tests.

    Uses a file-based cache to avoid re-embedding the same text.
    Only loads the real embedder if USE_REAL_EMBEDDINGS is set.

    Returns:
        FileCachedEmbedder wrapping E5Embedder, or MockEmbedder if
        USE_REAL_EMBEDDINGS is not set.
    """
    global _cached_embedder
    if _cached_embedder is None:
        if USE_REAL_EMBEDDINGS:
            try:
                from silicondb.embedders import E5Embedder

                real_embedder = E5Embedder("small")
                _cached_embedder = FileCachedEmbedder(
                    embed_fn=real_embedder.embed,
                    dimension=real_embedder.dimension,
                    model_name=real_embedder.model_name,
                    cache_name="e5_small",
                    embed_numpy_fn=real_embedder.embed_numpy,
                )
            except ImportError:
                # Fall back to mock if E5 not available
                _cached_embedder = MockEmbedder(dimension=384, model_name="mock-384")
        else:
            _cached_embedder = MockEmbedder(dimension=384, model_name="mock-384")
    return _cached_embedder


def pytest_sessionfinish(session, exitstatus):
    """Save embedding caches at end of test session."""
    global _cached_embedder
    if _cached_embedder and isinstance(_cached_embedder, FileCachedEmbedder):
        _cached_embedder.save()
        print(f"\nEmbedder {_cached_embedder.stats()}")


# ============================================================================
# Temp directory fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    path = Path(tempfile.mkdtemp(prefix="silicon-memory-test-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> Path:
    """Create a path for a temporary SiliconDB database."""
    return temp_dir / "test.db"


# ============================================================================
# SiliconDB fixtures (requires SILICONDB_LIBRARY_PATH)
# ============================================================================


@pytest.fixture
def silicondb_native(temp_db_path: Path, mock_embedder: MockEmbedder):
    """Create a native SiliconDB instance with mock embedder.

    Skips if SILICONDB_LIBRARY_PATH is not set.
    """
    if not SILICONDB_AVAILABLE:
        pytest.skip("SILICONDB_LIBRARY_PATH not set")

    from silicondb.native import SiliconDBNative

    db = SiliconDBNative(
        path=str(temp_db_path),
        dimensions=mock_embedder.dimension,
        enable_graph=True,
        enable_async=True,
        auto_embedder=False,
    )

    db.set_embedder(
        mock_embedder.embed,
        dimension=mock_embedder.dimension,
        model_name=mock_embedder.model_name,
        warmup=True,
    )

    yield db
    db.close()


@pytest.fixture
def silicondb(temp_db_path: Path, cached_embedder):
    """Create a SiliconDB instance with cached embedder.

    Skips if SILICONDB_LIBRARY_PATH is not set.
    """
    if not SILICONDB_AVAILABLE:
        pytest.skip("SILICONDB_LIBRARY_PATH not set")

    from silicondb import SiliconDB
    from silicondb.native import SiliconDBNative

    # Create native DB without auto-embedder
    db_native = SiliconDBNative(
        path=str(temp_db_path),
        dimensions=cached_embedder.dimension,
        enable_graph=True,
        enable_async=True,
        auto_embedder=False,
    )

    db_native.set_embedder(
        cached_embedder.embed,
        dimension=cached_embedder.dimension,
        model_name=cached_embedder.model_name,
        warmup=True,
    )

    yield db_native
    db_native.close()


# ============================================================================
# SiliconMemory fixtures (integration/e2e)
# ============================================================================


@pytest.fixture
def silicon_memory(temp_db_path: Path, mock_embedder: MockEmbedder):
    """Create a SiliconMemory instance with mock embedder.

    Skips if SILICONDB_LIBRARY_PATH is not set.
    """
    if not SILICONDB_AVAILABLE:
        pytest.skip("SILICONDB_LIBRARY_PATH not set")

    from silicon_memory import SiliconMemory
    from silicon_memory.storage.silicondb_backend import SiliconDBConfig

    # Note: We need to create the backend manually to inject the mock embedder
    from silicondb.native import SiliconDBNative

    # Create database with mock embedder
    db = SiliconDBNative(
        path=str(temp_db_path),
        dimensions=mock_embedder.dimension,
        enable_graph=True,
        enable_async=True,
        auto_embedder=False,
    )

    db.set_embedder(
        mock_embedder.embed,
        dimension=mock_embedder.dimension,
        model_name=mock_embedder.model_name,
        warmup=True,
    )

    # Create SiliconMemory with custom backend
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend
    from silicon_memory.temporal.decay import DecayConfig
    from silicon_memory.security.types import UserContext
    from silicon_memory.security.authorization import PolicyEngine

    # Create a wrapper that uses our pre-configured db
    class TestSiliconDBBackend(SiliconDBBackend):
        def __init__(self, db, decay_config=None):
            self._db = db
            self._decay_config = decay_config or DecayConfig()
            self._user_context = UserContext(
                user_id="test-user",
                tenant_id="test-tenant",
            )
            self._policy_engine = PolicyEngine()
            self._working_keys: set[str] = set()

    backend = TestSiliconDBBackend(db)

    # Create SiliconMemory-like object for testing
    class TestSiliconMemory:
        def __init__(self, backend):
            self._backend = backend

        @property
        def user_context(self):
            return self._backend._user_context

        async def recall(self, ctx):
            from silicon_memory.memory.silicondb_router import RecallResponse
            from silicon_memory.retrieval.salience import PROFILES, SalienceProfile

            # Resolve salience profile
            recall_kwargs = {
                "query": ctx.query,
                "max_facts": ctx.max_facts,
                "max_experiences": ctx.max_experiences,
                "max_procedures": ctx.max_procedures,
                "min_confidence": ctx.min_confidence,
            }
            if hasattr(ctx, "salience_profile") and ctx.salience_profile is not None:
                profile = ctx.salience_profile
                if isinstance(profile, str):
                    profile = PROFILES.get(profile)
                if isinstance(profile, SalienceProfile):
                    recall_kwargs["search_weights"] = profile.to_search_weights()

            result = await self._backend.recall(**recall_kwargs)

            return RecallResponse(
                query=result["query"],
                facts=result["facts"],
                experiences=result["experiences"],
                procedures=result["procedures"],
                working_context=result["working_context"],
                total_items=result["total_items"],
                as_of=result["as_of"],
            )

        async def what_do_you_know(self, query: str, min_confidence: float = 0.3):
            return await self._backend.build_knowledge_proof(query, min_confidence)

        async def commit_belief(self, belief):
            return await self._backend.commit_belief(belief)

        async def get_belief(self, belief_id):
            return await self._backend.get_belief(belief_id)

        async def query_beliefs(self, query, limit=10, min_confidence=0.0):
            return await self._backend.query_beliefs(query, limit, min_confidence)

        async def record_experience(self, experience):
            return await self._backend.record_experience(experience)

        async def commit_procedure(self, procedure):
            return await self._backend.commit_procedure(procedure)

        async def set_context(self, key: str, value, ttl_seconds: int = 300):
            return await self._backend.set_working(key, value, ttl_seconds)

        async def get_context(self, key: str):
            return await self._backend.get_working(key)

        async def get_all_context(self):
            return await self._backend.get_all_working()

        async def commit_decision(self, decision):
            snapshot_id = None
            if decision.assumptions:
                belief_ids = [str(a.belief_id) for a in decision.assumptions]
                try:
                    snapshot = await self._backend.snapshot_beliefs(belief_ids)
                    snapshot_id = snapshot.get("snapshot_id")
                    decision.belief_snapshot_id = snapshot_id
                except Exception:
                    pass  # Snapshot not critical for decision storage
            await self._backend.commit_decision(decision)
            return snapshot_id

        async def recall_decisions(self, query, k=10, min_confidence=0.0):
            return await self._backend.recall_decisions(query, k, min_confidence)

        async def get_decision(self, decision_id):
            return await self._backend.get_decision(decision_id)

        async def record_outcome(self, decision_id, outcome):
            return await self._backend.record_decision_outcome(decision_id, outcome)

        async def revise_decision(self, decision_id, new_decision):
            return await self._backend.revise_decision(decision_id, new_decision)

        async def cross_reference(self, query, limit=20, min_confidence=0.3):
            from silicon_memory.core.types import SourceType
            from silicon_memory.memory.silicondb_router import CrossReferenceResult

            all_beliefs = await self._backend.query_beliefs(
                query, limit=limit * 2, min_confidence=min_confidence
            )

            internal = []
            external = []
            for b in all_beliefs:
                if b.source and b.source.type == SourceType.EXTERNAL:
                    external.append(b)
                else:
                    internal.append(b)
            internal = internal[:limit]
            external = external[:limit]

            agreements = []
            contradictions_list = []
            for ib in internal:
                for eb in external:
                    ib_text = ib.content or (ib.triplet.as_text() if ib.triplet else "")
                    eb_text = eb.content or (eb.triplet.as_text() if eb.triplet else "")
                    if not ib_text or not eb_text:
                        continue
                    ib_words = set(ib_text.lower().split())
                    eb_words = set(eb_text.lower().split())
                    overlap = len(ib_words & eb_words)
                    total = min(len(ib_words), len(eb_words))
                    if total > 0 and overlap / total > 0.3:
                        contra_words = {"not", "no", "never", "incorrect", "wrong", "false"}
                        ib_has_neg = bool(ib_words & contra_words)
                        eb_has_neg = bool(eb_words & contra_words)
                        if ib_has_neg != eb_has_neg:
                            contradictions_list.append((ib, eb))
                        else:
                            agreements.append((ib, eb))

            return CrossReferenceResult(
                query=query,
                internal_beliefs=internal,
                external_beliefs=external,
                agreements=agreements,
                contradictions=contradictions_list,
            )

        async def ingest_from(self, adapter, content, metadata, llm_provider=None):
            from silicon_memory.ingestion.types import IngestionResult

            metadata = {
                **metadata,
                "user_id": self.user_context.user_id,
                "tenant_id": self.user_context.tenant_id,
            }
            try:
                return await adapter.ingest(
                    content=content,
                    metadata=metadata,
                    memory=self,
                    llm_provider=llm_provider,
                )
            except Exception as e:
                return IngestionResult(
                    source_type=adapter.source_type,
                    errors=[f"Ingestion failed: {e}"],
                )

        def close(self):
            self._backend._db.close()

    memory = TestSiliconMemory(backend)
    yield memory
    memory.close()


@pytest.fixture
def silicon_memory_with_data(silicon_memory, sample_source: Source):
    """Create a SiliconMemory instance pre-populated with test data.

    Adds several beliefs, experiences, and procedures for testing queries.
    """
    import asyncio

    async def setup():
        # Add beliefs
        beliefs = [
            Belief(
                id=uuid4(),
                triplet=Triplet("Python", "is", "programming language"),
                confidence=0.95,
                source=sample_source,
                tags=["programming"],
            ),
            Belief(
                id=uuid4(),
                triplet=Triplet("Python", "is used for", "data science"),
                confidence=0.9,
                source=sample_source,
                tags=["programming", "data-science"],
            ),
            Belief(
                id=uuid4(),
                triplet=Triplet("JavaScript", "is", "programming language"),
                confidence=0.95,
                source=sample_source,
                tags=["programming", "web"],
            ),
            Belief(
                id=uuid4(),
                content="Machine learning is a subset of artificial intelligence.",
                confidence=0.85,
                source=sample_source,
                tags=["ml", "ai"],
            ),
        ]

        for belief in beliefs:
            await silicon_memory.commit_belief(belief)

        # Add experiences
        experiences = [
            Experience(
                id=uuid4(),
                content="Helped user debug Python code",
                outcome="Successfully fixed the bug",
                session_id="session-1",
            ),
            Experience(
                id=uuid4(),
                content="Explained machine learning concepts",
                outcome="User understood the basics",
                session_id="session-2",
            ),
        ]

        for exp in experiences:
            await silicon_memory.record_experience(exp)

        # Add procedures
        procedures = [
            Procedure(
                id=uuid4(),
                name="Debug Python Code",
                description="How to debug Python code systematically",
                steps=[
                    "Read the error message carefully",
                    "Check the line number mentioned",
                    "Add print statements or use debugger",
                    "Fix the issue and test",
                ],
                trigger="debug python",
                confidence=0.85,
            ),
            Procedure(
                id=uuid4(),
                name="Install Python Package",
                description="How to install packages using pip",
                steps=[
                    "Open terminal",
                    "Run: pip install package-name",
                    "Verify: python -c 'import package'",
                ],
                trigger="install package",
                confidence=0.9,
            ),
        ]

        for proc in procedures:
            await silicon_memory.commit_procedure(proc)

    asyncio.get_event_loop().run_until_complete(setup())
    return silicon_memory
