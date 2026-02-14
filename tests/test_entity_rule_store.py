"""Unit tests for EntityRuleStore — mock SiliconDB."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silicon_memory.entities.types import DetectorRule, ExtractorRule


# ---------------------------------------------------------------------------
# Fake SiliconDB that stores documents in a plain dict
# ---------------------------------------------------------------------------


class FakeSiliconDB:
    """Minimal in-memory double for silicondb.SiliconDB."""

    def __init__(self, **kwargs):
        self._docs: dict[str, dict] = {}

    def ingest(self, *, external_id: str, text: str, metadata: dict | None = None,
               node_type: str | None = None, **kwargs) -> None:
        self._docs[external_id] = {
            "external_id": external_id,
            "text": text,
            "metadata": dict(metadata) if metadata else {},
            "node_type": node_type,
        }

    def get(self, external_id: str) -> dict | None:
        return self._docs.get(external_id)

    def update(self, external_id: str, text: str | None = None,
               metadata: dict | None = None, **kwargs) -> None:
        doc = self._docs.get(external_id)
        if doc is None:
            raise KeyError(f"Document {external_id} not found")
        if text is not None:
            doc["text"] = text
        if metadata is not None:
            doc["metadata"] = metadata

    def delete(self, external_id: str) -> None:
        self._docs.pop(external_id, None)

    def search(self, query: str, k: int = 10, filter: dict | None = None,
               **kwargs) -> list[dict]:
        results = []
        for doc in self._docs.values():
            if filter:
                match = True
                for fk, fv in filter.items():
                    if doc.get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue
            results.append(doc)
        return results[:k]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_store(fake_db: FakeSiliconDB):
    """Build an EntityRuleStore backed by the given FakeSiliconDB."""
    from silicon_memory.entities.store import EntityRuleStore

    store = object.__new__(EntityRuleStore)
    store._db = fake_db
    store._manifest = {"detectors": [], "extractors": [], "aliases": []}
    store._load_manifest()
    return store


@pytest.fixture()
def fake_db():
    return FakeSiliconDB()


@pytest.fixture()
def store(fake_db):
    return _make_store(fake_db)


def _sample_detector(rule_id: str = "det-1") -> DetectorRule:
    return DetectorRule(
        id=rule_id,
        pattern=r"\bAcme\b",
        description="Detect references to Acme Corp",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def _sample_extractor(rule_id: str = "ext-1") -> ExtractorRule:
    return ExtractorRule(
        id=rule_id,
        entity_type="company",
        detector_ids=["det-1"],
        pattern=r"(Acme\s+\w+)",
        normalize_template="{match_lower}",
        examples=["Acme Corp", "Acme Inc"],
        context_examples=["Acme Corp announced"],
        confidence=0.95,
        context_threshold=0.7,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests: save and load
# ---------------------------------------------------------------------------


class TestSaveAndLoadDetector:
    def test_round_trip(self, store):
        rule = _sample_detector()
        store.save_detector(rule)

        loaded = store.load_all_detectors()
        assert len(loaded) == 1
        r = loaded[0]
        assert r.id == rule.id
        assert r.pattern == rule.pattern
        assert r.description == rule.description
        assert r.created_at == rule.created_at


class TestSaveAndLoadExtractor:
    def test_round_trip(self, store):
        rule = _sample_extractor()
        store.save_extractor(rule)

        loaded = store.load_all_extractors()
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
        assert r.created_at == rule.created_at


class TestSaveAndLoadAlias:
    def test_round_trip(self, store):
        store.save_alias("ACME", "acme_corp", "company")

        loaded = store.load_all_aliases()
        assert len(loaded) == 1
        alias, canonical, etype = loaded[0]
        assert alias == "ACME"
        assert canonical == "acme_corp"
        assert etype == "company"


class TestSaveIsIdempotent:
    def test_detector_no_duplicate(self, store, fake_db):
        rule = _sample_detector()
        store.save_detector(rule)
        store.save_detector(rule)  # second save = upsert

        assert store._manifest["detectors"].count(rule.id) == 1
        loaded = store.load_all_detectors()
        assert len(loaded) == 1

    def test_extractor_no_duplicate(self, store):
        rule = _sample_extractor()
        store.save_extractor(rule)
        store.save_extractor(rule)

        assert store._manifest["extractors"].count(rule.id) == 1
        loaded = store.load_all_extractors()
        assert len(loaded) == 1

    def test_alias_no_duplicate(self, store):
        store.save_alias("Foo", "foo_id", "person")
        store.save_alias("Foo", "foo_id", "person")

        assert store._manifest["aliases"].count("foo") == 1
        loaded = store.load_all_aliases()
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Tests: delete
# ---------------------------------------------------------------------------


class TestDeleteDetector:
    def test_removes_from_manifest(self, store):
        rule = _sample_detector()
        store.save_detector(rule)
        assert store.delete_detector(rule.id) is True
        assert rule.id not in store._manifest["detectors"]
        assert store.load_all_detectors() == []

    def test_missing_returns_false(self, store):
        assert store.delete_detector("nonexistent") is False


class TestDeleteExtractor:
    def test_removes_from_manifest(self, store):
        rule = _sample_extractor()
        store.save_extractor(rule)
        assert store.delete_extractor(rule.id) is True
        assert rule.id not in store._manifest["extractors"]
        assert store.load_all_extractors() == []


class TestDeleteAlias:
    def test_removes_from_manifest(self, store):
        store.save_alias("Bar", "bar_id", "org")
        assert store.delete_alias("Bar") is True
        assert "bar" not in store._manifest["aliases"]
        assert store.load_all_aliases() == []


# ---------------------------------------------------------------------------
# Tests: manifest tracking
# ---------------------------------------------------------------------------


class TestManifest:
    def test_tracks_ids_on_save(self, store):
        store.save_detector(_sample_detector("d1"))
        store.save_detector(_sample_detector("d2"))
        store.save_extractor(_sample_extractor("e1"))
        store.save_alias("X", "x_id", "thing")

        assert store._manifest["detectors"] == ["d1", "d2"]
        assert store._manifest["extractors"] == ["e1"]
        assert store._manifest["aliases"] == ["x"]

    def test_updates_on_delete(self, store):
        store.save_detector(_sample_detector("d1"))
        store.save_detector(_sample_detector("d2"))
        store.delete_detector("d1")

        assert store._manifest["detectors"] == ["d2"]


# ---------------------------------------------------------------------------
# Tests: vector search (with fake search returning docs)
# ---------------------------------------------------------------------------


class TestFindSimilarDetectors:
    def test_returns_matching_rules(self, store):
        rule = _sample_detector()
        store.save_detector(rule)

        results = store.find_similar_detectors("Acme")
        assert len(results) >= 1
        assert results[0].id == rule.id


class TestFindSimilarExtractors:
    def test_returns_matching_rules(self, store):
        rule = _sample_extractor()
        store.save_extractor(rule)

        results = store.find_similar_extractors("company")
        assert len(results) >= 1
        assert results[0].id == rule.id


# ---------------------------------------------------------------------------
# Tests: EntityResolver integration
# ---------------------------------------------------------------------------


class TestResolverAutoPersists:
    @pytest.mark.asyncio
    async def test_register_alias_calls_store(self):
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        mock_store = MagicMock()
        resolver = EntityResolver(
            cache=EntityCache(), rules=RuleEngine(), store=mock_store,
        )

        await resolver.register_alias("ACME", "acme_corp", "company")
        mock_store.save_alias.assert_called_once_with("ACME", "acme_corp", "company")

    def test_add_detector_calls_store(self):
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        mock_store = MagicMock()
        resolver = EntityResolver(
            cache=EntityCache(), rules=RuleEngine(), store=mock_store,
        )

        rule = _sample_detector()
        resolver.add_detector(rule)
        mock_store.save_detector.assert_called_once_with(rule)
        assert len(resolver.rules._detectors) == 1

    def test_add_extractor_calls_store(self):
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        mock_store = MagicMock()
        resolver = EntityResolver(
            cache=EntityCache(), rules=RuleEngine(), store=mock_store,
        )

        rule = _sample_extractor()
        resolver.add_extractor(rule)
        mock_store.save_extractor.assert_called_once_with(rule)
        assert len(resolver.rules._extractors) == 1

    def test_works_without_store(self):
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        resolver = EntityResolver(cache=EntityCache(), rules=RuleEngine())

        rule = _sample_detector()
        resolver.add_detector(rule)
        assert len(resolver.rules._detectors) == 1

    @pytest.mark.asyncio
    async def test_register_alias_works_without_store(self):
        from silicon_memory.entities.cache import EntityCache
        from silicon_memory.entities.resolver import EntityResolver
        from silicon_memory.entities.rules import RuleEngine

        resolver = EntityResolver(cache=EntityCache(), rules=RuleEngine())
        await resolver.register_alias("X", "x_id", "person")
        assert resolver.cache.lookup("X") == "x_id"
