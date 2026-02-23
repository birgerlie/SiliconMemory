"""Tests for context-bundle enrichment of extracted triplets."""

from __future__ import annotations

from silicon_memory.reflection.generator import _enrich_context_bundles
from silicon_memory.reflection.types import BeliefCandidate


def _candidate(subject: str, predicate: str, obj: str, doc_id: str) -> BeliefCandidate:
    return BeliefCandidate(
        content=f"{subject} {predicate} {obj}",
        subject=subject,
        predicate=predicate,
        object=obj,
        source_context={"source_document": {"document_id": doc_id}},
    )


def test_context_bundle_groups_by_grounding_doc() -> None:
    c1 = _candidate("A", "uses", "B", "doc-1")
    c2 = _candidate("B", "depends on", "C", "doc-1")
    c3 = _candidate("X", "references", "Y", "doc-2")

    _enrich_context_bundles([c1, c2, c3])

    assert c1.source_context["context_bundle_id"] == c2.source_context["context_bundle_id"]
    assert c1.source_context["context_bundle_id"] != c3.source_context["context_bundle_id"]
    assert c1.source_context["grounding_doc_id"] == "doc-1"
    assert c3.source_context["grounding_doc_id"] == "doc-2"


def test_context_bundle_adds_triplet_and_co_triplet_keys() -> None:
    c1 = _candidate("A", "uses", "B", "doc-1")
    c2 = _candidate("B", "depends on", "C", "doc-1")

    _enrich_context_bundles([c1, c2])

    assert "triplet_key" in c1.source_context
    assert "triplet_key" in c2.source_context
    assert c1.source_context["triplet_key"] != c2.source_context["triplet_key"]

    co1 = c1.source_context.get("co_triplet_keys", [])
    co2 = c2.source_context.get("co_triplet_keys", [])
    assert c2.source_context["triplet_key"] in co1
    assert c1.source_context["triplet_key"] in co2


def test_context_bundle_adds_embedding_text() -> None:
    c = _candidate("A", "uses", "B", "doc-1")
    _enrich_context_bundles([c])
    text = c.source_context.get("embedding_text", "")
    assert "A | uses | B" in text
    assert "doc: doc-1" in text
