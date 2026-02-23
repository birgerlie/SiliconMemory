"""Focused tests for backend reliability controls."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from silicon_memory.core.decision import Decision
from silicon_memory.core.types import Belief, SourceType, Triplet
from silicon_memory.core.utils import utc_now
from silicon_memory.security.types import UserContext
from silicon_memory.storage.silicondb_backend import SiliconDBBackend, SiliconDBConfig
from silicon_memory.temporal.decay import DecayConfig


class _FakeDecisionDB:
    def __init__(self) -> None:
        self.ingest_calls = 0

    def ingest(self, **kwargs):  # noqa: ANN003
        self.ingest_calls += 1
        return kwargs

    def add_edge(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


@pytest.mark.asyncio
async def test_run_db_retries_transient_error():
    """Transient transport errors should be retried."""
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._config = SiliconDBConfig(path="", retry_attempts=3, retry_base_ms=1, retry_max_ms=2)
    backend._db = MagicMock()
    backend._decay_config = DecayConfig()
    backend._user_context = UserContext(user_id="u", tenant_id="t")
    backend._policy_engine = MagicMock()
    backend._working_keys = set()

    calls = {"n": 0}

    def _op():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("service unavailable")
        return "ok"

    result = await backend._run_db("retryable_op", _op, retry=True)
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_run_db_does_not_retry_non_transient():
    """Non-transient errors should fail fast."""
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._config = SiliconDBConfig(path="", retry_attempts=5, retry_base_ms=1, retry_max_ms=2)
    backend._db = MagicMock()
    backend._decay_config = DecayConfig()
    backend._user_context = UserContext(user_id="u", tenant_id="t")
    backend._policy_engine = MagicMock()
    backend._working_keys = set()

    calls = {"n": 0}

    def _op():
        calls["n"] += 1
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError):
        await backend._run_db("non_retryable_op", _op, retry=True)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_commit_decision_is_idempotent():
    """Duplicate decision commit payload should be suppressed within TTL."""
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._config = SiliconDBConfig(path="", idempotency_ttl_s=600, retry_attempts=1)
    backend._db = _FakeDecisionDB()
    backend._decay_config = DecayConfig()
    backend._user_context = UserContext(user_id="u", tenant_id="t")
    backend._policy_engine = MagicMock()
    backend._working_keys = set()

    decision = Decision(title="Use PostgreSQL", description="Primary DB")
    await backend.commit_decision(decision)
    await backend.commit_decision(decision)

    assert backend._db.ingest_calls == 1


@pytest.mark.asyncio
async def test_get_beliefs_by_tag_matches_native_tags():
    """Tag lookup should work when tags are stored as native lists."""

    class _Triple:
        external_id = "t/u/belief-1"
        subject = "Alice"
        predicate = "hypothetically"
        object_value = "Bob"
        probability = 0.6
        sources: dict[str, float] = {}
        metadata = {
            "belief_id": "25e8db47-e5ca-4a56-b89d-cf65c8fd0072",
            "status": "provisional",
            "tags": ["hypothesis", "deterministic_discovery"],
        }

    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._query_triples = lambda **kwargs: [_Triple()]  # noqa: ARG005
    backend._search_by_type = lambda *args, **kwargs: []  # noqa: ARG005
    backend._can_access = lambda *args, **kwargs: True  # noqa: ARG005

    beliefs = await backend.get_beliefs_by_tag("hypothesis", limit=10, min_confidence=0.0)
    assert len(beliefs) == 1
    assert "hypothesis" in beliefs[0].tags


@pytest.mark.asyncio
async def test_find_claim_merge_candidates_scores_exact_triplet() -> None:
    class _Triple:
        external_id = "t/u/belief-1"
        subject = "Alice"
        predicate = "works_at"
        object_value = "ACME"
        probability = 0.82
        sources: dict[str, float] = {}
        metadata = {
            "belief_id": "a5e90e3b-7b2d-45b5-939e-c3c7b8355f85",
            "status": "provisional",
            "canonical_claim_id": "a5e90e3b-7b2d-45b5-939e-c3c7b8355f85",
        }

    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._query_triples = lambda **kwargs: [_Triple()]  # noqa: ARG005
    backend._search_by_type = lambda *args, **kwargs: []  # noqa: ARG005
    backend._can_access = lambda *args, **kwargs: True  # noqa: ARG005

    belief = Belief(
        triplet=Triplet("Alice", "works_at", "ACME"),
        confidence=0.75,
    )
    candidates = await backend.find_claim_merge_candidates(belief, limit=5)
    assert len(candidates) == 1
    assert candidates[0].score >= 0.99
    assert candidates[0].subject == "Alice"
    assert candidates[0].predicate == "works_at"


@pytest.mark.asyncio
async def test_query_beliefs_collapses_same_canonical_claim() -> None:
    canonical_id = "3f9c6a8d-f813-4f8e-9e57-4153ebb3b7e4"
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._weighted_search = lambda *args, **kwargs: [  # noqa: ARG005
        {
            "node_type": "belief_surface",
            "text": "Alice works at ACME",
            "probability": 0.8,
            "external_id": "t/u/belief_surface-1",
            "metadata": {
                "belief_id": "8a73dbf6-f5e8-4124-bbd8-4f240f74776a",
                "status": "provisional",
                "canonical_claim_id": canonical_id,
                "triplet": {"subject": "Alice", "predicate": "works_at", "object": "ACME"},
                "tags": ["extracted"],
            },
        },
        {
            "node_type": "belief",
            "text": "Alice is employed by ACME",
            "probability": 0.79,
            "external_id": "t/u/belief-2",
            "metadata": {
                "belief_id": "e6be4c2d-95d8-48ec-9ae2-b7ad748eec14",
                "status": "provisional",
                "canonical_claim_id": canonical_id,
                "triplet": {"subject": "Alice", "predicate": "employed_by", "object": "ACME"},
                "tags": ["extracted"],
            },
        },
    ]
    backend._query_triples = lambda **kwargs: []  # noqa: ARG005
    backend._can_access = lambda *args, **kwargs: True  # noqa: ARG005

    beliefs = await backend.query_beliefs("Alice ACME", limit=10)
    assert len(beliefs) == 1
    assert beliefs[0].metadata.get("canonical_claim_id") == canonical_id


def test_triple_to_belief_hydrates_source_metadata():
    """Belief conversion should preserve source provenance metadata."""

    class _Triple:
        subject = "Alice"
        predicate = "works with"
        object_value = "Bob"
        probability = 0.77
        sources: dict[str, float] = {}
        metadata = {
            "belief_id": "6f16f82d-7f91-4272-b269-709c5dc5a5d0",
            "status": "provisional",
            "source_id": "reflection_engine",
            "source_type": "reflection",
            "source_reliability": 0.91,
            "source_metadata": {"grounding_doc_id": "doc-42", "evidence_span": "line 9"},
        }

    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    belief = backend._triple_to_belief(_Triple())  # noqa: SLF001
    assert belief is not None
    assert belief.source is not None
    assert belief.source.id == "reflection_engine"
    assert belief.source.type == SourceType.REFLECTION
    assert belief.source.metadata.get("grounding_doc_id") == "doc-42"


def test_triple_to_belief_hydrates_native_complex_metadata():
    """Triple conversion should preserve native tags/evidence/source metadata."""

    class _Triple:
        subject = "Alpha"
        predicate = "related_to"
        object_value = "Beta"
        probability = 0.63
        sources: dict[str, float] = {}
        metadata = {
            "belief_id": "1382d486-cf63-4e93-ac7f-65b1154d9eb4",
            "status": "provisional",
            "tags": ["hypothesis", "deterministic_discovery"],
            "evidence_for": ["6f16f82d-7f91-4272-b269-709c5dc5a5d0"],
            "source_id": "reflection_engine",
            "source_type": "reflection",
            "source_reliability": 0.7,
            "source_metadata": {"information_gain": 0.88},
        }

    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    belief = backend._triple_to_belief(_Triple())  # noqa: SLF001
    assert belief is not None
    assert "hypothesis" in belief.tags
    assert len(belief.evidence_for) == 1
    assert belief.source is not None
    assert belief.source.metadata.get("information_gain") == 0.88


def test_doc_to_experience_hydrates_context_dict_and_processed_bool() -> None:
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    exp = backend._doc_to_experience(  # noqa: SLF001
        {
            "text": "Contextful experience",
            "metadata": {
                "experience_id": "f8f9f560-a95f-4bc4-ae3d-bcdf446ca4f6",
                "occurred_at": "2026-02-19T11:00:00+00:00",
                "context": {"document_id": "doc-42", "title": "Sample"},
                "processed": "true",
            },
        },
    )
    assert exp is not None
    assert exp.context.get("document_id") == "doc-42"
    assert exp.processed is True


def test_search_result_to_experience_hydrates_context_json() -> None:
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    exp = backend._search_result_to_experience(  # noqa: SLF001
        {
            "text": "JSON context experience",
            "metadata": {
                "experience_id": "2ec5ef75-eaf9-4f64-b4ac-c8c0664f0706",
                "occurred_at": "2026-02-19T11:00:00+00:00",
                "context": "{\"document_id\": \"doc-77\"}",
                "processed": "false",
            },
        },
    )
    assert exp is not None
    assert exp.context.get("document_id") == "doc-77"
    assert exp.processed is False


class _FakeExperienceDB:
    def __init__(self, docs: dict[str, dict]) -> None:
        self.docs = docs

    def get(self, external_id: str) -> dict | None:
        return self.docs.get(external_id)

    def update(self, external_id: str, metadata: dict | None = None):  # noqa: ANN001
        if external_id not in self.docs:
            return None
        doc = dict(self.docs[external_id])
        current = dict(doc.get("metadata") or {})
        if metadata:
            current.update(metadata)
        doc["metadata"] = current
        self.docs[external_id] = doc
        return doc

    def ingest(self, external_id: str, text: str, metadata: dict, node_type: str):  # noqa: ANN001
        self.docs[external_id] = {
            "text": text,
            "metadata": dict(metadata),
            "node_type": node_type,
        }
        return self.docs[external_id]


class _FakeWorkingIndexDB:
    def __init__(self, docs: dict[str, dict]) -> None:
        self.docs = docs

    def get(self, external_id: str) -> dict | None:
        return self.docs.get(external_id)

    def update(self, external_id: str, text: str = "", metadata: dict | None = None):  # noqa: ANN001
        self.docs[external_id] = {"text": text, "metadata": metadata or {}}
        return self.docs[external_id]

    def ingest(self, external_id: str, text: str, metadata: dict, node_type: str):  # noqa: ANN001
        self.docs[external_id] = {"text": text, "metadata": metadata, "node_type": node_type}
        return self.docs[external_id]


def _build_backend_with_experience_docs(
    docs: dict[str, dict],
    *,
    use_consistency_waits: bool = False,
) -> SiliconDBBackend:
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._config = SiliconDBConfig(
        path="", retry_attempts=1, use_consistency_waits=use_consistency_waits,
    )
    backend._db = _FakeExperienceDB(docs)
    backend._decay_config = DecayConfig()
    backend._user_context = UserContext(user_id="u", tenant_id="t")
    backend._policy_engine = MagicMock()
    backend._working_keys = set()
    backend._experience_external_ids = set(docs.keys())
    backend._search_experiences_broad = lambda *args, **kwargs: []  # noqa: ARG005
    return backend


@pytest.mark.asyncio
async def test_recent_experiences_falls_back_to_direct_ids_when_search_empty() -> None:
    exp_id = uuid4()
    occurred_at = utc_now() - timedelta(hours=1)
    external_id = f"t/u/experience-{exp_id}"
    docs = {
        external_id: {
            "text": "Team synced on migration plan",
            "metadata": {
                "experience_id": str(exp_id),
                "occurred_at": occurred_at.isoformat(),
                "processed": False,
                "extracted": False,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
    }
    backend = _build_backend_with_experience_docs(docs)

    recent = await backend.get_recent_experiences(hours=24, limit=10)
    assert len(recent) == 1
    assert recent[0].id == exp_id


@pytest.mark.asyncio
async def test_unprocessed_and_unextracted_fallback_to_direct_ids() -> None:
    exp_id = uuid4()
    external_id = f"t/u/experience-{exp_id}"
    docs = {
        external_id: {
            "text": "Follow-up action captured",
            "metadata": {
                "experience_id": str(exp_id),
                "occurred_at": utc_now().isoformat(),
                "processed": False,
                "extracted": False,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
    }
    backend = _build_backend_with_experience_docs(docs)

    unprocessed = await backend.get_unprocessed_experiences(limit=10)
    unextracted = await backend.get_unextracted_experiences(limit=10)
    assert len(unprocessed) == 1
    assert len(unextracted) == 1
    assert unprocessed[0].id == exp_id
    assert unextracted[0].id == exp_id


@pytest.mark.asyncio
async def test_count_extraction_progress_uses_indexed_direct_fallback() -> None:
    unextracted_id = uuid4()
    extracted_id = uuid4()
    docs = {
        f"t/u/experience-{unextracted_id}": {
            "text": "Pending extraction item",
            "metadata": {
                "experience_id": str(unextracted_id),
                "occurred_at": utc_now().isoformat(),
                "processed": False,
                "extracted": False,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
        f"t/u/experience-{extracted_id}": {
            "text": "Already extracted item",
            "metadata": {
                "experience_id": str(extracted_id),
                "occurred_at": utc_now().isoformat(),
                "processed": True,
                "extracted": True,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
    }
    backend = _build_backend_with_experience_docs(docs)

    progress = await backend.count_extraction_progress()
    assert progress == {"extracted": 1, "unextracted": 1, "total": 2}


@pytest.mark.asyncio
async def test_count_extraction_progress_respects_access_scope() -> None:
    visible_id = uuid4()
    hidden_id = uuid4()
    docs = {
        f"t/u/experience-{visible_id}": {
            "text": "Visible extraction item",
            "metadata": {
                "experience_id": str(visible_id),
                "occurred_at": utc_now().isoformat(),
                "processed": False,
                "extracted": False,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
        f"other_t/other_u/experience-{hidden_id}": {
            "text": "Hidden extraction item",
            "metadata": {
                "experience_id": str(hidden_id),
                "occurred_at": utc_now().isoformat(),
                "processed": False,
                "extracted": False,
                "owner_id": "other_u",
                "tenant_id": "other_t",
                "privacy_level": "private",
            },
        },
    }
    backend = _build_backend_with_experience_docs(docs)

    progress = await backend.count_extraction_progress()
    assert progress == {"extracted": 0, "unextracted": 1, "total": 1}


@pytest.mark.asyncio
async def test_mark_experience_processed_preserves_existing_extracted_flag() -> None:
    exp_id = uuid4()
    external_id = f"t/u/experience-{exp_id}"
    docs = {
        external_id: {
            "text": "Experience to mark processed",
            "metadata": {
                "experience_id": str(exp_id),
                "occurred_at": utc_now().isoformat(),
                "processed": False,
                "extracted": True,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
    }
    backend = _build_backend_with_experience_docs(docs)
    await backend.mark_experiences_processed([exp_id])
    updated = docs[external_id]["metadata"]
    assert updated["processed"] is True
    assert updated["extracted"] is True


@pytest.mark.asyncio
async def test_mark_experience_extracted_preserves_existing_processed_flag() -> None:
    exp_id = uuid4()
    external_id = f"t/u/experience-{exp_id}"
    docs = {
        external_id: {
            "text": "Experience to mark extracted",
            "metadata": {
                "experience_id": str(exp_id),
                "occurred_at": utc_now().isoformat(),
                "processed": True,
                "extracted": False,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
    }
    backend = _build_backend_with_experience_docs(docs)
    await backend.mark_experiences_extracted([exp_id])
    updated = docs[external_id]["metadata"]
    assert updated["processed"] is True
    assert updated["extracted"] is True


@pytest.mark.asyncio
async def test_mark_experience_flags_preserved_in_either_update_order() -> None:
    exp_a = uuid4()
    exp_b = uuid4()
    docs = {
        f"t/u/experience-{exp_a}": {
            "text": "Order A",
            "metadata": {
                "experience_id": str(exp_a),
                "occurred_at": utc_now().isoformat(),
                "processed": False,
                "extracted": False,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
        f"t/u/experience-{exp_b}": {
            "text": "Order B",
            "metadata": {
                "experience_id": str(exp_b),
                "occurred_at": utc_now().isoformat(),
                "processed": False,
                "extracted": False,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
    }
    backend = _build_backend_with_experience_docs(docs)

    # extracted -> processed
    await backend.mark_experiences_extracted([exp_a])
    await backend.mark_experiences_processed([exp_a])

    # processed -> extracted
    await backend.mark_experiences_processed([exp_b])
    await backend.mark_experiences_extracted([exp_b])

    assert docs[f"t/u/experience-{exp_a}"]["metadata"]["processed"] is True
    assert docs[f"t/u/experience-{exp_a}"]["metadata"]["extracted"] is True
    assert docs[f"t/u/experience-{exp_b}"]["metadata"]["processed"] is True
    assert docs[f"t/u/experience-{exp_b}"]["metadata"]["extracted"] is True


@pytest.mark.asyncio
async def test_count_extraction_progress_filters_inaccessible_search_results() -> None:
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._config = SiliconDBConfig(path="", retry_attempts=1, use_consistency_waits=False)
    backend._db = MagicMock()
    backend._decay_config = DecayConfig()
    backend._user_context = UserContext(user_id="u", tenant_id="t")
    backend._policy_engine = MagicMock()
    backend._working_keys = set()
    backend._experience_external_ids = set()
    backend._search_experiences_broad = lambda *args, **kwargs: [  # noqa: ARG005
        {
            "external_id": "t/u/experience-visible",
            "metadata": {
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
                "extracted": True,
            },
        },
        {
            "external_id": "other_t/other_u/experience-hidden",
            "metadata": {
                "owner_id": "other_u",
                "tenant_id": "other_t",
                "privacy_level": "private",
                "extracted": True,
            },
        },
    ]

    progress = await backend.count_extraction_progress()
    assert progress == {"extracted": 1, "unextracted": 0, "total": 1}


@pytest.mark.asyncio
async def test_get_unprocessed_extraction_items_uses_indexed_direct_fallback() -> None:
    belief_id = uuid4()
    extraction_external_id = f"t/u/extraction-{belief_id}"
    docs = {
        extraction_external_id: {
            "text": "extraction item",
            "metadata": {
                "belief_id": str(belief_id),
                "reflection_processed": False,
                "active": True,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
            "node_type": "extraction_item",
        },
    }
    backend = _build_backend_with_experience_docs(docs, use_consistency_waits=False)
    backend._extraction_external_ids = {extraction_external_id}  # noqa: SLF001
    backend._search_by_type = lambda *args, **kwargs: []  # noqa: ARG005, SLF001
    backend._query_triples = lambda **kwargs: []  # noqa: ARG005, SLF001

    items = await backend.get_unprocessed_extraction_items(limit=10)
    assert len(items) == 1
    assert items[0]["external_id"] == extraction_external_id


@pytest.mark.asyncio
async def test_write_extraction_item_tracks_external_id() -> None:
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._config = SiliconDBConfig(path="", retry_attempts=1, use_consistency_waits=False)
    backend._db = _FakeExperienceDB({})
    backend._decay_config = DecayConfig()
    backend._user_context = UserContext(user_id="u", tenant_id="t")
    backend._policy_engine = MagicMock()
    backend._working_keys = set()
    backend._experience_external_ids = set()
    backend._extraction_external_ids = set()

    belief = Belief(
        id=uuid4(),
        content="A reflected belief",
        confidence=0.7,
        tags={"extracted"},
    )
    belief_external_id = f"t/u/belief-{belief.id}"

    await backend._write_extraction_item_for_belief(  # noqa: SLF001
        belief=belief,
        belief_external_id=belief_external_id,
        storage_type="document",
    )

    extraction_external_id = f"t/u/extraction-{belief.id}"
    assert extraction_external_id in backend._extraction_external_ids  # noqa: SLF001


@pytest.mark.asyncio
async def test_get_all_working_loads_keys_from_persisted_index() -> None:
    expires_at = (utc_now() + timedelta(minutes=10)).isoformat()
    docs = {
        "t/u/working-index-keys": {
            "metadata": {
                "keys": ["open_question_1"],
            },
        },
        "t/u/working-open_question_1": {
            "metadata": {
                "key": "open_question_1",
                "value": {"question": "What validates hypothesis X?"},
                "expires_at": expires_at,
                "owner_id": "u",
                "tenant_id": "t",
                "privacy_level": "private",
            },
        },
    }
    backend = SiliconDBBackend.__new__(SiliconDBBackend)
    backend._config = SiliconDBConfig(path="", retry_attempts=1, use_consistency_waits=False)
    backend._db = _FakeWorkingIndexDB(docs)
    backend._decay_config = DecayConfig()
    backend._user_context = UserContext(user_id="u", tenant_id="t")
    backend._policy_engine = MagicMock()
    backend._working_keys = set()

    all_working = await backend.get_all_working()

    assert "open_question_1" in all_working
    assert backend._working_keys == {"open_question_1"}
