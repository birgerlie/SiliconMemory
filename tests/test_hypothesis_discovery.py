"""Tests for deterministic hypothesis discovery and ranking."""

from __future__ import annotations

from uuid import uuid4

import pytest

from silicon_memory.reflection.hypothesis import CommunityEdge, HypothesisGenerator
from silicon_memory.reflection.types import BeliefCandidate


def _generator() -> HypothesisGenerator:
    return HypothesisGenerator.__new__(HypothesisGenerator)


def test_two_hop_bridge_discovery_creates_candidate() -> None:
    gen = _generator()
    edges = [
        CommunityEdge(subject="Alice", predicate="met", object="Bob", confidence=0.9),
        CommunityEdge(subject="Bob", predicate="advised", object="Carol", confidence=0.8),
    ]

    candidates = gen._generate_deterministic_hypotheses(edges)  # noqa: SLF001
    bridge = [
        c for c in candidates
        if c.predicate == "possibly_connected_via" and c.subject == "Alice" and c.object == "Carol"
    ]
    assert bridge, "Expected a two-hop bridge hypothesis"
    assert bridge[0].source_context.get("operator") == "two_hop_bridge"


def test_contradiction_discovery_creates_candidate() -> None:
    gen = _generator()
    edges = [
        CommunityEdge(subject="Case-42", predicate="status", object="open", confidence=0.8),
        CommunityEdge(subject="Case-42", predicate="status", object="closed", confidence=0.7),
    ]

    candidates = gen._generate_deterministic_hypotheses(edges)  # noqa: SLF001
    conflict = [c for c in candidates if c.predicate == "has_conflicting_claim_on"]
    assert conflict, "Expected a contradiction hypothesis"
    assert "open" in conflict[0].content.lower()
    assert "closed" in conflict[0].content.lower()


def test_temporal_sequence_discovery_creates_candidate() -> None:
    gen = _generator()
    edges = [
        CommunityEdge(subject="Case-42", predicate="argued_on", object="2024-03-12", confidence=0.8),
        CommunityEdge(subject="Case-42", predicate="decided_on", object="2024-09-17", confidence=0.9),
    ]

    candidates = gen._generate_deterministic_hypotheses(edges)  # noqa: SLF001
    temporal = [c for c in candidates if c.predicate == "timeline_progresses_from_to"]
    assert temporal, "Expected temporal sequence hypothesis"
    assert "2024-03-12" in temporal[0].object
    assert "2024-09-17" in temporal[0].object


def test_information_gain_ranking_prioritizes_novel_supported_candidate() -> None:
    gen = _generator()
    novel = BeliefCandidate(
        id=uuid4(),
        content="[HYPOTHESIS] novel",
        subject="Alice",
        predicate="possibly_connected_via",
        object="Carol",
        confidence=0.5,
        source_context={"path_support": 0.8},
    )
    known = BeliefCandidate(
        id=uuid4(),
        content="[HYPOTHESIS] known",
        subject="Alice",
        predicate="possibly_connected_via",
        object="Bob",
        confidence=0.5,
        source_context={"path_support": 0.8},
    )
    known_key = gen._candidate_key(known)  # noqa: SLF001

    ranked = gen._rank_candidates_by_information_gain(  # noqa: SLF001
        [known, novel],
        known_triplet_keys={known_key},
    )

    assert ranked
    assert ranked[0].content == novel.content
    assert ranked[0].source_context.get("information_gain", 0.0) > ranked[1].source_context.get(
        "information_gain", 0.0,
    )


@pytest.mark.asyncio
async def test_global_temporal_hypotheses_from_triples() -> None:
    class _Triple:
        def __init__(self, subject: str, predicate: str, object_value: str, probability: float = 0.7):
            self.subject = subject
            self.predicate = predicate
            self.object_value = object_value
            self.probability = probability
            self.external_id = "t/u/belief-1"
            self.metadata = {"source_metadata": {"source": {"document_id": "doc-1"}}}

    class _Backend:
        def _query_triples(self, k: int = 1000):  # noqa: ARG002
            return [
                _Triple("Case-42", "argued_on", "2024-03-12"),
                _Triple("Case-42", "decided_on", "2024-09-17"),
            ]

        def _can_access(self, metadata, _external_id=""):  # noqa: ANN001, ARG002
            return True

    class _Memory:
        def __init__(self) -> None:
            self._backend = _Backend()

    gen = HypothesisGenerator(memory=_Memory(), llm=object())
    cands = await gen._generate_global_temporal_hypotheses(limit=10)  # noqa: SLF001
    assert cands
    assert any(c.predicate == "timeline_progresses_from_to" for c in cands)


@pytest.mark.asyncio
async def test_global_temporal_ignores_observed_at_for_non_temporal_predicates() -> None:
    class _Triple:
        def __init__(
            self,
            subject: str,
            predicate: str,
            object_value: str,
            metadata: dict | None = None,
            probability: float = 0.7,
        ) -> None:
            self.subject = subject
            self.predicate = predicate
            self.object_value = object_value
            self.probability = probability
            self.external_id = "t/u/belief-1"
            self.metadata = metadata or {}

    class _Backend:
        def _query_triples(self, k: int = 1000):  # noqa: ARG002
            return [
                _Triple(
                    "Case-42",
                    "related_to",
                    "Maxwell",
                    metadata={"observed_at": "2026-02-19T00:00:00+00:00"},
                ),
                _Triple("Case-42", "decided_on", "2024-09-17"),
            ]

        def _can_access(self, metadata, _external_id=""):  # noqa: ANN001, ARG002
            return True

    class _Memory:
        def __init__(self) -> None:
            self._backend = _Backend()

    gen = HypothesisGenerator(memory=_Memory(), llm=object())
    cands = await gen._generate_global_temporal_hypotheses(limit=10)  # noqa: SLF001
    assert cands == []


@pytest.mark.asyncio
async def test_global_temporal_normalizes_month_year_text() -> None:
    class _Triple:
        def __init__(self, subject: str, predicate: str, object_value: str, probability: float = 0.7):
            self.subject = subject
            self.predicate = predicate
            self.object_value = object_value
            self.probability = probability
            self.external_id = "t/u/belief-1"
            self.metadata = {}

    class _Backend:
        def _query_triples(self, k: int = 1000):  # noqa: ARG002
            return [
                _Triple("Case-42", "signed in", "September 2007"),
                _Triple("Case-42", "decided on", "March 12, 2024"),
            ]

        def _can_access(self, metadata, _external_id=""):  # noqa: ANN001, ARG002
            return True

    class _Memory:
        def __init__(self) -> None:
            self._backend = _Backend()

    gen = HypothesisGenerator(memory=_Memory(), llm=object())
    cands = await gen._generate_global_temporal_hypotheses(limit=10)  # noqa: SLF001
    assert cands
    assert any(
        c.predicate == "timeline_progresses_from_to"
        and "2007-09" in c.object
        and "2024-03-12" in c.object
        for c in cands
    )


@pytest.mark.asyncio
async def test_global_temporal_ignores_relative_date_metadata() -> None:
    class _Triple:
        def __init__(self, subject: str, predicate: str, object_value: str, metadata: dict):
            self.subject = subject
            self.predicate = predicate
            self.object_value = object_value
            self.probability = 0.7
            self.external_id = "t/u/belief-1"
            self.metadata = metadata

    class _Backend:
        def _query_triples(self, k: int = 1000):  # noqa: ARG002
            return [
                _Triple(
                    "Case-42",
                    "occurred_on",
                    "2026",
                    metadata={
                        "source_metadata": {
                            "object_date": {
                                "canonical": "2026",
                                "precision": "year",
                                "relative": True,
                                "reference_date": "2026-02-19",
                            },
                        },
                    },
                ),
                _Triple(
                    "Case-42",
                    "occurred_on",
                    "1994",
                    metadata={
                        "source_metadata": {
                            "object_date": {
                                "canonical": "1994",
                                "precision": "year",
                                "relative": False,
                            },
                        },
                    },
                ),
            ]

        def _can_access(self, metadata, _external_id=""):  # noqa: ANN001, ARG002
            return True

    class _Memory:
        def __init__(self) -> None:
            self._backend = _Backend()

    gen = HypothesisGenerator(memory=_Memory(), llm=object())
    cands = await gen._generate_global_temporal_hypotheses(limit=10)  # noqa: SLF001
    assert cands == []
