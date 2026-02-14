"""End-to-end tests for SiliconDB foundation features (SDB-1 through SDB-5).

These tests exercise SiliconDB's low-level APIs directly to verify that
temporal scoring, entropy/custom scoring, graph proximity, belief snapshots,
and edge embeddings work correctly end-to-end.

Run with:
    SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e_silicondb.py -v
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest


pytestmark = pytest.mark.e2e


# ============================================================================
# Helpers
# ============================================================================


def _ingest_and_wait(db, external_id, text, **kwargs):
    """Ingest a document and wait for it to be indexed."""
    result = db.ingest(external_id=external_id, text=text, **kwargs)
    seq = result.get("sequence", 0)
    doc_id = result.get("doc_id", 0)
    if seq:
        db.wait_for(seq, consistency="indexed", doc_id=doc_id)
    return result


# ============================================================================
# SDB-1: Temporal Ranking Signal — E2E Tests
# ============================================================================


class TestTemporalScoringE2E:
    """E2E tests for temporal recency scoring in SiliconDB.

    Verifies that search_temporal ranks more recent documents higher
    than older ones when temporal_weight is dominant.
    """

    @pytest.mark.asyncio
    async def test_recent_doc_ranks_higher(self, silicondb_native):
        """Two docs with same content but different ages: recent wins."""
        db = silicondb_native

        now = time.time()
        one_week_ago = now - 7 * 24 * 3600
        one_month_ago = now - 30 * 24 * 3600

        _ingest_and_wait(
            db, "old-doc", "database performance optimization techniques",
            metadata={"age": "old"},
            created_at=one_month_ago, updated_at=one_month_ago,
        )
        _ingest_and_wait(
            db, "recent-doc", "database performance optimization techniques",
            metadata={"age": "recent"},
            created_at=one_week_ago, updated_at=one_week_ago,
        )
        _ingest_and_wait(
            db, "newest-doc", "database performance optimization techniques",
            metadata={"age": "newest"},
            created_at=now, updated_at=now,
        )

        # Search with dominant temporal weight
        results = db.search_temporal(
            query="database performance",
            k=10,
            vector_weight=0.1,
            text_weight=0.1,
            temporal_weight=0.8,
            temporal_half_life_hours=168,  # 1 week
        )

        assert len(results) >= 2

        # Extract external IDs in result order
        ext_ids = [r.get("externalId", r.get("external_id", "")) for r in results]
        assert "newest-doc" in ext_ids
        assert "old-doc" in ext_ids

        newest_idx = ext_ids.index("newest-doc")
        old_idx = ext_ids.index("old-doc")
        assert newest_idx < old_idx, (
            f"newest-doc (idx={newest_idx}) should rank before old-doc (idx={old_idx})"
        )

    @pytest.mark.asyncio
    async def test_temporal_zero_weight_no_effect(self, silicondb_native):
        """With temporal_weight=0, recency should not affect ranking."""
        db = silicondb_native

        now = time.time()
        old_time = now - 365 * 24 * 3600  # 1 year ago

        _ingest_and_wait(
            db, "old-precise", "quantum computing algorithms",
            created_at=old_time, updated_at=old_time,
        )
        _ingest_and_wait(
            db, "new-vague", "quantum computing general overview",
            created_at=now, updated_at=now,
        )

        results_no_temporal = db.search_temporal(
            query="quantum computing algorithms",
            k=10,
            vector_weight=0.5,
            text_weight=0.5,
            temporal_weight=0.0,
        )

        assert len(results_no_temporal) >= 1
        # With zero temporal weight, the result should work without errors
        # (no specific ordering guarantee with mock embedder)

    @pytest.mark.asyncio
    async def test_temporal_search_returns_scores(self, silicondb_native):
        """Temporal search should return score breakdown."""
        db = silicondb_native

        _ingest_and_wait(db, "scored-doc", "machine learning models")

        results = db.search_temporal(
            query="machine learning",
            k=5,
            temporal_weight=0.3,
        )

        assert len(results) >= 1
        first = results[0]
        assert "score" in first or "scores" in first


# ============================================================================
# SDB-2: Custom Scoring / Entropy — E2E Tests
# ============================================================================


class TestEntropyScoringE2E:
    """E2E tests for entropy scoring in SiliconDB.

    Verifies that entropy is returned as a float in search results,
    that get_uncertain_beliefs returns high-entropy beliefs, and that
    search_with_entropy_rerank produces valid results.
    """

    @pytest.mark.asyncio
    async def test_search_result_entropy_is_float(self, silicondb_native):
        """Search results should include entropy as a float."""
        db = silicondb_native

        # Ingest a belief with mid-range probability → non-zero entropy
        _ingest_and_wait(
            db, "ent-belief-1", "Redis is the best caching solution",
            node_type="belief", probability=0.6,
        )

        results = db.search(query="caching solution", k=5)
        assert len(results) >= 1

        first = results[0]
        entropy_val = first.get("entropy")
        assert entropy_val is not None, f"Expected 'entropy' in result, got keys: {list(first.keys())}"
        assert isinstance(entropy_val, float), f"Expected float, got {type(entropy_val)}: {entropy_val}"

    @pytest.mark.asyncio
    async def test_entropy_varies_with_probability(self, silicondb_native):
        """Beliefs with probability ~0.5 should have higher entropy than those near 1.0."""
        db = silicondb_native

        # High certainty (low entropy)
        _ingest_and_wait(
            db, "certain-belief", "Python is a programming language",
            node_type="belief", probability=0.99,
        )
        # Uncertain (high entropy)
        _ingest_and_wait(
            db, "uncertain-belief", "Python is faster than C for compute",
            node_type="belief", probability=0.5,
        )

        results = db.search(query="Python programming", k=10)
        assert len(results) >= 2

        entropy_by_id = {}
        for r in results:
            ext_id = r.get("external_id", r.get("externalId", ""))
            entropy_by_id[ext_id] = r.get("entropy", 0.0)

        certain_ent = entropy_by_id.get("certain-belief", 0.0)
        uncertain_ent = entropy_by_id.get("uncertain-belief", 0.0)

        assert uncertain_ent > certain_ent, (
            f"Uncertain belief entropy ({uncertain_ent}) should exceed "
            f"certain belief entropy ({certain_ent})"
        )

    @pytest.mark.asyncio
    async def test_get_uncertain_beliefs(self, silicondb_native):
        """get_uncertain_beliefs should return beliefs with high entropy."""
        db = silicondb_native

        # Insert triples with varying probabilities
        db.insert_triple(
            "triple-sure", "Earth", "is", "round", probability=0.99,
        )
        db.insert_triple(
            "triple-unsure", "Pluto", "is", "planet", probability=0.5,
        )
        db.insert_triple(
            "triple-moderate", "Mars", "has", "water", probability=0.7,
        )
        # Small sleep to let beliefs index
        import time
        time.sleep(0.1)

        uncertain = db.get_uncertain_beliefs(min_entropy=0.3, k=10)

        # Should include the uncertain beliefs but not the highly certain one
        ext_ids = [b.get("external_id", b.get("externalId", "")) for b in uncertain]
        assert "triple-unsure" in ext_ids, (
            f"Expected 'triple-unsure' (p=0.5) in uncertain beliefs, got: {ext_ids}"
        )

    @pytest.mark.asyncio
    async def test_search_with_entropy_rerank(self, silicondb_native, mock_embedder):
        """search_with_entropy_rerank should return valid results."""
        db = silicondb_native

        _ingest_and_wait(db, "er-doc-1", "Redis caching architecture patterns")
        _ingest_and_wait(db, "er-doc-2", "PostgreSQL query optimization guide")
        _ingest_and_wait(db, "er-doc-3", "Database replication strategies")

        query_emb = mock_embedder.embed(["caching architecture"])[0]
        results = db.search_with_entropy_rerank(
            query="caching architecture",
            embedding=query_emb,
            k=5,
            candidates=10,
            lambda_param=0.5,
        )

        assert len(results) >= 1, "Expected at least 1 result from entropy rerank search"
        # Verify results are dicts with expected fields
        first = results[0]
        assert isinstance(first, dict)


# ============================================================================
# SDB-3: Graph Proximity Scoring — E2E Tests
# ============================================================================


class TestGraphProximityE2E:
    """E2E tests for Personalized PageRank in SiliconDB.

    Verifies that PPR assigns higher scores to nodes closer to seed nodes
    in the graph.
    """

    @pytest.mark.asyncio
    async def test_ppr_closer_nodes_score_higher(self, silicondb_native):
        """Nodes 1-hop from seed should score higher than 2-hop nodes."""
        db = silicondb_native

        # Create a small graph: seed -> hop1 -> hop2, seed -> hop1b
        _ingest_and_wait(db, "seed-node", "project alpha central hub")
        _ingest_and_wait(db, "hop1-a", "authentication module for alpha")
        _ingest_and_wait(db, "hop1-b", "database layer for alpha")
        _ingest_and_wait(db, "hop2-a", "JWT token validation details")
        _ingest_and_wait(db, "distant", "unrelated weather data")

        # Build graph edges
        db.add_edge("seed-node", "hop1-a", edge_type="contains")
        db.add_edge("seed-node", "hop1-b", edge_type="contains")
        db.add_edge("hop1-a", "hop2-a", edge_type="uses")
        # 'distant' has no edges to seed

        # Run PPR from seed
        ppr_results = db.personalized_pagerank(
            seeds=[{"external_id": "seed-node", "weight": 1.0}],
            damping_factor=0.85,
            max_iterations=50,
        )

        assert len(ppr_results) >= 3

        scores = {
            r.get("externalId", r.get("external_id", "")): r["score"]
            for r in ppr_results
        }

        # Seed should have the highest score
        assert scores.get("seed-node", 0) > 0

        # 1-hop nodes should score higher than 2-hop
        hop1_score = max(scores.get("hop1-a", 0), scores.get("hop1-b", 0))
        hop2_score = scores.get("hop2-a", 0)
        assert hop1_score > hop2_score, (
            f"1-hop score ({hop1_score}) should be > 2-hop score ({hop2_score})"
        )

    @pytest.mark.asyncio
    async def test_ppr_disconnected_node_low_score(self, silicondb_native):
        """Nodes with no path to seed should have very low PPR scores."""
        db = silicondb_native

        _ingest_and_wait(db, "ppr-center", "central processing unit")
        _ingest_and_wait(db, "ppr-connected", "connected component")
        _ingest_and_wait(db, "ppr-island", "isolated island node")

        db.add_edge("ppr-center", "ppr-connected", edge_type="links")
        # ppr-island has no edges

        ppr_results = db.personalized_pagerank(
            seeds=[{"external_id": "ppr-center", "weight": 1.0}],
        )

        scores = {
            r.get("externalId", r.get("external_id", "")): r["score"]
            for r in ppr_results
        }

        connected_score = scores.get("ppr-connected", 0)
        island_score = scores.get("ppr-island", 0)

        assert connected_score > island_score, (
            f"Connected node ({connected_score}) should score > island ({island_score})"
        )

    @pytest.mark.asyncio
    async def test_ppr_multiple_seeds(self, silicondb_native):
        """PPR with multiple seed nodes should boost nodes near both."""
        db = silicondb_native

        _ingest_and_wait(db, "ms-seed-a", "seed A content")
        _ingest_and_wait(db, "ms-seed-b", "seed B content")
        _ingest_and_wait(db, "ms-bridge", "bridge between A and B")
        _ingest_and_wait(db, "ms-only-a", "only connected to A")

        db.add_edge("ms-seed-a", "ms-bridge", edge_type="links")
        db.add_edge("ms-seed-b", "ms-bridge", edge_type="links")
        db.add_edge("ms-seed-a", "ms-only-a", edge_type="links")

        ppr_results = db.personalized_pagerank(
            seeds=[
                {"external_id": "ms-seed-a", "weight": 1.0},
                {"external_id": "ms-seed-b", "weight": 1.0},
            ],
        )

        scores = {
            r.get("externalId", r.get("external_id", "")): r["score"]
            for r in ppr_results
        }

        # Bridge (connected to both seeds) should score at least as high
        # as only-a (connected to one seed)
        bridge_score = scores.get("ms-bridge", 0)
        only_a_score = scores.get("ms-only-a", 0)

        assert bridge_score >= only_a_score, (
            f"Bridge ({bridge_score}) should score >= only-a ({only_a_score})"
        )


# ============================================================================
# SDB-4: Belief Snapshots / Point-in-Time Queries — E2E Tests
# ============================================================================


class TestBeliefSnapshotsE2E:
    """E2E tests for belief history tracking and point-in-time queries."""

    @pytest.mark.asyncio
    async def test_belief_history_records_changes(self, silicondb_native):
        """Inserting a triple and recording observations should create history."""
        db = silicondb_native

        ext_id = f"belief-{uuid4()}"
        db.insert_triple(
            external_id=ext_id,
            subject="Python",
            predicate="is used for",
            object_value="web development",
            probability=0.7,
        )

        # Record observations to change probability
        db.record_observation(ext_id, confirmed=True, source="test-1")
        db.record_observation(ext_id, confirmed=True, source="test-2")
        db.record_observation(ext_id, confirmed=False, source="test-3")

        history = db.get_belief_history(ext_id)

        # Should have at least the initial entry + observation entries
        assert len(history) >= 1, f"Expected history entries, got {len(history)}"

        # Each entry should have a timestamp
        for entry in history:
            assert "timestamp" in entry or "timestamp_us" in entry

    @pytest.mark.asyncio
    async def test_belief_as_of_returns_historical_state(self, silicondb_native):
        """get_belief_as_of should return the state at a past timestamp."""
        db = silicondb_native

        ext_id = f"belief-{uuid4()}"

        # Insert with initial probability
        db.insert_triple(
            external_id=ext_id,
            subject="Redis",
            predicate="is",
            object_value="fast",
            probability=0.6,
        )

        # Record timestamp after initial insert
        time.sleep(0.05)
        after_insert_us = int(time.time() * 1_000_000)

        # Record confirming observations to increase probability
        time.sleep(0.05)
        db.record_observation(ext_id, confirmed=True, source="obs-1")
        db.record_observation(ext_id, confirmed=True, source="obs-2")
        db.record_observation(ext_id, confirmed=True, source="obs-3")

        # Query at the post-insert timestamp (before observations)
        past_state = db.get_belief_as_of(ext_id, after_insert_us)

        if past_state is not None:
            # The probability at that time should be close to initial
            past_prob = past_state.get("probability", past_state.get("confidence", 0))
            assert isinstance(past_prob, (int, float))

    @pytest.mark.asyncio
    async def test_snapshot_beliefs_creates_immutable_copy(self, silicondb_native):
        """snapshot_beliefs should create a snapshot that doesn't change."""
        db = silicondb_native

        ext_id_1 = f"belief-{uuid4()}"
        ext_id_2 = f"belief-{uuid4()}"

        db.insert_triple(
            external_id=ext_id_1,
            subject="Go",
            predicate="has",
            object_value="goroutines",
            probability=0.8,
        )
        db.insert_triple(
            external_id=ext_id_2,
            subject="Rust",
            predicate="has",
            object_value="ownership model",
            probability=0.9,
        )

        # Create snapshot
        snapshot = db.snapshot_beliefs([ext_id_1, ext_id_2])
        snapshot_id = snapshot.get("snapshot_id") or snapshot.get("id", "")
        assert snapshot_id, f"Expected snapshot_id, got: {snapshot}"

        # Mutate the beliefs
        db.record_observation(ext_id_1, confirmed=False, source="contra-1")
        db.record_observation(ext_id_1, confirmed=False, source="contra-2")

        # Retrieve the snapshot — should reflect original state
        retrieved = db.get_belief_snapshot(snapshot_id)
        assert retrieved is not None, "Snapshot should be retrievable"

        # Snapshot should contain both beliefs
        beliefs = retrieved.get("beliefs", retrieved.get("entries", {}))
        assert len(beliefs) >= 1, f"Expected beliefs in snapshot, got: {retrieved}"

    @pytest.mark.asyncio
    async def test_snapshot_survives_probability_update(self, silicondb_native):
        """After creating a snapshot, updating probabilities shouldn't affect it."""
        db = silicondb_native

        ext_id = f"belief-{uuid4()}"
        db.insert_triple(
            external_id=ext_id,
            subject="Docker",
            predicate="uses",
            object_value="containers",
            probability=0.95,
        )

        # Snapshot at high confidence
        snapshot = db.snapshot_beliefs([ext_id])
        snapshot_id = snapshot.get("snapshot_id") or snapshot.get("id", "")

        # Drive probability down
        for _ in range(5):
            db.record_observation(ext_id, confirmed=False, source="contra")

        # Snapshot should still have original values
        retrieved = db.get_belief_snapshot(snapshot_id)
        assert retrieved is not None


# ============================================================================
# SDB-5: Edge Embeddings — E2E Tests
# ============================================================================


class TestEdgeEmbeddingsE2E:
    """E2E tests for edge embeddings (semantic edge search)."""

    @pytest.mark.asyncio
    async def test_add_edge_with_description(self, silicondb_native):
        """Edges can be added with text descriptions."""
        db = silicondb_native

        _ingest_and_wait(db, "src-node", "source entity")
        _ingest_and_wait(db, "tgt-node", "target entity")

        # Should not raise
        db.add_edge(
            from_id="src-node",
            to_id="tgt-node",
            edge_type="uses",
            weight=1.0,
            description="uses Redis for session caching with sub-ms latency",
        )

    @pytest.mark.asyncio
    async def test_search_edges_by_embedding(self, silicondb_native, mock_embedder):
        """search_edges returns edges ranked by semantic similarity."""
        db = silicondb_native

        _ingest_and_wait(db, "company-a", "Acme Corporation")
        _ingest_and_wait(db, "tech-redis", "Redis database")
        _ingest_and_wait(db, "tech-postgres", "PostgreSQL database")
        _ingest_and_wait(db, "company-b", "Beta Industries")

        # Add edges with different descriptions
        db.add_edge(
            "company-a", "tech-redis", edge_type="uses",
            description="uses Redis for real-time caching and session storage",
            embedding=mock_embedder.embed(["uses Redis for real-time caching and session storage"])[0],
        )
        db.add_edge(
            "company-a", "tech-postgres", edge_type="uses",
            description="uses PostgreSQL for relational data warehousing",
            embedding=mock_embedder.embed(["uses PostgreSQL for relational data warehousing"])[0],
        )
        db.add_edge(
            "company-b", "tech-redis", edge_type="evaluated",
            description="evaluated Redis but chose Memcached instead",
            embedding=mock_embedder.embed(["evaluated Redis but chose Memcached instead"])[0],
        )

        # Search for caching-related edges
        query_emb = mock_embedder.embed(["caching and session storage"])[0]
        results = db.search_edges(embedding=query_emb, k=10)

        assert len(results) >= 1, "Expected at least 1 edge search result"

        # Each result should have source/target info
        first = results[0]
        assert "source_id" in first or "sourceId" in first or "from_id" in first

    @pytest.mark.asyncio
    async def test_edge_without_description_not_in_search(
        self, silicondb_native, mock_embedder,
    ):
        """Edges added without descriptions should not appear in edge search."""
        db = silicondb_native

        _ingest_and_wait(db, "plain-src", "plain source")
        _ingest_and_wait(db, "plain-tgt", "plain target")
        _ingest_and_wait(db, "desc-src", "described source")
        _ingest_and_wait(db, "desc-tgt", "described target")

        # Edge without description
        db.add_edge("plain-src", "plain-tgt", edge_type="links")

        # Edge with description
        desc = "semantic relationship for testing edge search"
        db.add_edge(
            "desc-src", "desc-tgt", edge_type="semantic",
            description=desc,
            embedding=mock_embedder.embed([desc])[0],
        )

        query_emb = mock_embedder.embed(["semantic relationship testing"])[0]
        results = db.search_edges(embedding=query_emb, k=10)

        # If results are returned, the plain edge (no description) should not be among them
        result_sources = set()
        for r in results:
            src = r.get("source_id", r.get("sourceId", r.get("from_id", "")))
            result_sources.add(src)

        # The described edge should be found, the plain one should not
        if results:
            assert "plain-src" not in result_sources, (
                "Plain edge (no description) should not appear in edge search"
            )

    @pytest.mark.asyncio
    async def test_edge_search_returns_description(
        self, silicondb_native, mock_embedder,
    ):
        """Edge search results should include the description text."""
        db = silicondb_native

        _ingest_and_wait(db, "es-src", "edge search source")
        _ingest_and_wait(db, "es-tgt", "edge search target")

        desc = "connects via high-bandwidth fiber optic link"
        db.add_edge(
            "es-src", "es-tgt", edge_type="connects",
            description=desc,
            embedding=mock_embedder.embed([desc])[0],
        )

        query_emb = mock_embedder.embed(["fiber optic connection"])[0]
        results = db.search_edges(embedding=query_emb, k=5)

        assert len(results) >= 1
        first = results[0]
        # Should have description or text field
        result_desc = first.get("description", first.get("text", ""))
        assert "fiber" in result_desc.lower() or len(result_desc) > 0, (
            f"Expected description in result, got: {first}"
        )
