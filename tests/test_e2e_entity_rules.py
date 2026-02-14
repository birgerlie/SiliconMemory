"""End-to-end tests for EntityRuleStore persistence.

These tests exercise the full stack: EntityRuleStore → real SiliconDB → disk,
verifying that entity rules and aliases survive a store close/reopen cycle.

Run with:
    SILICONDB_LIBRARY_PATH=/path/to/lib pytest tests/test_e2e_entity_rules.py -v
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from silicon_memory.entities.cache import EntityCache
from silicon_memory.entities.resolver import EntityResolver
from silicon_memory.entities.rules import RuleEngine
from silicon_memory.entities.types import DetectorRule, ExtractorRule

pytestmark = pytest.mark.e2e

SILICONDB_AVAILABLE = bool(os.environ.get("SILICONDB_LIBRARY_PATH"))


def _skip_unless_silicondb():
    if not SILICONDB_AVAILABLE:
        pytest.skip("SILICONDB_LIBRARY_PATH not set")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(db_path: Path, mock_embedder):
    """Create an EntityRuleStore backed by a real SiliconDB with mock embedder."""
    from silicondb.native import SiliconDBNative

    from silicon_memory.entities.store import EntityRuleStore

    # Build native DB with mock embedder
    db = SiliconDBNative(
        path=str(db_path),
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

    # Build store, injecting our pre-configured db
    store = object.__new__(EntityRuleStore)
    store._db = db
    store._manifest = {"detectors": [], "extractors": [], "aliases": []}
    store._load_manifest()
    return store


def _sample_detector(rule_id: str = "det-acme") -> DetectorRule:
    return DetectorRule(
        id=rule_id,
        pattern=r"\bAcme\b",
        description="Detect references to Acme Corp",
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


def _sample_extractor(rule_id: str = "ext-company") -> ExtractorRule:
    return ExtractorRule(
        id=rule_id,
        entity_type="company",
        detector_ids=["det-acme"],
        pattern=r"(Acme\s+\w+)",
        normalize_template="{match_lower}",
        examples=["Acme Corp", "Acme Inc"],
        context_examples=["Acme Corp announced today"],
        confidence=0.92,
        context_threshold=0.65,
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectorPersistence:
    """Save a detector, close the store, reopen, and verify it loads."""

    def test_survives_close_reopen(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        rule = _sample_detector()

        # Save
        store1 = _make_store(temp_db_path, mock_embedder)
        store1.save_detector(rule)
        store1.close()

        # Reopen
        store2 = _make_store(temp_db_path, mock_embedder)
        loaded = store2.load_all_detectors()
        store2.close()

        assert len(loaded) == 1
        r = loaded[0]
        assert r.id == rule.id
        assert r.pattern == rule.pattern
        assert r.description == rule.description
        assert r.created_at == rule.created_at


class TestExtractorPersistence:
    """Save an extractor, close, reopen, verify all fields preserved."""

    def test_survives_close_reopen(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        rule = _sample_extractor()

        store1 = _make_store(temp_db_path, mock_embedder)
        store1.save_extractor(rule)
        store1.close()

        store2 = _make_store(temp_db_path, mock_embedder)
        loaded = store2.load_all_extractors()
        store2.close()

        assert len(loaded) == 1
        r = loaded[0]
        assert r.id == rule.id
        assert r.entity_type == rule.entity_type
        assert r.detector_ids == rule.detector_ids
        assert r.pattern == rule.pattern
        assert r.normalize_template == rule.normalize_template
        assert r.examples == rule.examples
        assert r.context_examples == rule.context_examples
        assert r.confidence == rule.confidence
        assert r.context_threshold == rule.context_threshold


class TestAliasPersistence:
    """Save aliases, close, reopen, verify round-trip."""

    def test_survives_close_reopen(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        store1 = _make_store(temp_db_path, mock_embedder)
        store1.save_alias("ACME", "acme_corp", "company")
        store1.save_alias("Bob Smith", "bob.smith@acme.com", "person")
        store1.close()

        store2 = _make_store(temp_db_path, mock_embedder)
        loaded = store2.load_all_aliases()
        store2.close()

        assert len(loaded) == 2
        by_canonical = {canonical: (alias, etype) for alias, canonical, etype in loaded}
        assert "acme_corp" in by_canonical
        assert "bob.smith@acme.com" in by_canonical
        assert by_canonical["acme_corp"][1] == "company"
        assert by_canonical["bob.smith@acme.com"][1] == "person"


class TestDeletePersistence:
    """Delete a rule, close, reopen, verify it's gone."""

    def test_deleted_detector_stays_deleted(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        store1 = _make_store(temp_db_path, mock_embedder)
        store1.save_detector(_sample_detector("d1"))
        store1.save_detector(_sample_detector("d2"))
        store1.delete_detector("d1")
        store1.close()

        store2 = _make_store(temp_db_path, mock_embedder)
        loaded = store2.load_all_detectors()
        store2.close()

        assert len(loaded) == 1
        assert loaded[0].id == "d2"


class TestMultipleRules:
    """Save many rules, close, reopen, verify all present."""

    def test_batch_persist(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        store1 = _make_store(temp_db_path, mock_embedder)
        for i in range(5):
            store1.save_detector(_sample_detector(f"det-{i}"))
            store1.save_extractor(_sample_extractor(f"ext-{i}"))
        for i in range(3):
            store1.save_alias(f"alias-{i}", f"canonical-{i}", "thing")
        store1.close()

        store2 = _make_store(temp_db_path, mock_embedder)
        assert len(store2.load_all_detectors()) == 5
        assert len(store2.load_all_extractors()) == 5
        assert len(store2.load_all_aliases()) == 3
        store2.close()


class TestIdempotentPersist:
    """Saving the same rule twice doesn't create duplicates on disk."""

    def test_upsert_is_idempotent(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        rule = _sample_detector()
        store1 = _make_store(temp_db_path, mock_embedder)
        store1.save_detector(rule)
        store1.save_detector(rule)
        store1.close()

        store2 = _make_store(temp_db_path, mock_embedder)
        loaded = store2.load_all_detectors()
        store2.close()

        assert len(loaded) == 1


class TestResolverE2EIntegration:
    """Full round-trip: resolver + store → close → reload into fresh resolver."""

    @pytest.mark.asyncio
    async def test_resolver_persist_and_reload(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        # Phase 1: create resolver with store, add rules via resolver API
        store1 = _make_store(temp_db_path, mock_embedder)
        resolver1 = EntityResolver(
            cache=EntityCache(), rules=RuleEngine(), store=store1,
        )
        resolver1.add_detector(_sample_detector())
        resolver1.add_extractor(_sample_extractor())
        await resolver1.register_alias("ACME", "acme_corp", "company")

        # Verify resolver1 works
        result = await resolver1.resolve("Acme Corp announced Q4 earnings")
        assert len(result.resolved) >= 1
        store1.close()

        # Phase 2: fresh resolver, load persisted rules from store
        store2 = _make_store(temp_db_path, mock_embedder)
        cache2 = EntityCache()
        rules2 = RuleEngine()
        for d in store2.load_all_detectors():
            rules2.add_detector(d)
        for e in store2.load_all_extractors():
            rules2.add_extractor(e)
        for alias, canonical_id, entity_type in store2.load_all_aliases():
            cache2.store(alias, canonical_id, entity_type)

        resolver2 = EntityResolver(
            cache=cache2, rules=rules2, store=store2,
        )

        # Verify the reloaded resolver resolves the same entity
        result2 = await resolver2.resolve("Acme Corp announced Q4 earnings")
        assert len(result2.resolved) >= 1
        assert result2.resolved[0].entity_type == "company"

        # Alias also survived
        assert cache2.lookup("ACME") == "acme_corp"
        store2.close()


class TestVectorSearch:
    """Verify find_similar_* returns persisted rules via SiliconDB search."""

    def test_find_similar_detectors(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        store = _make_store(temp_db_path, mock_embedder)
        store.save_detector(_sample_detector())

        results = store.find_similar_detectors("Acme company references")
        assert len(results) >= 1
        assert results[0].id == "det-acme"
        store.close()

    def test_find_similar_extractors(self, temp_db_path, mock_embedder):
        _skip_unless_silicondb()

        store = _make_store(temp_db_path, mock_embedder)
        store.save_extractor(_sample_extractor())

        results = store.find_similar_extractors("company name pattern")
        assert len(results) >= 1
        assert results[0].id == "ext-company"
        store.close()
