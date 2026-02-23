"""Temporal/context regression tests for observation consolidation."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from silicon_memory.reflection.observation_consolidator import (
    Observation,
    ObservationConsolidator,
    load_from_cache,
)


def test_load_from_cache_uses_item_source_and_emits_occurred_on(tmp_path: Path) -> None:
    cache_file = tmp_path / "extraction_000000.json"
    cache_file.write_text(
        json.dumps(
            {
                "extraction": {
                    "facts": [
                        {
                            "subject": "Case 22-1426",
                            "predicate": "argued on",
                            "object": "March 12, 2024",
                            "source": "DOJ-OGR-00000002",
                            "confidence": 0.8,
                        },
                    ],
                    "events": [
                        {
                            "date": "September 17, 2024",
                            "event": "Decision issued",
                            "actors": ["Second Circuit"],
                            "source": "DOJ-OGR-00000002",
                            "confidence": 0.9,
                        },
                    ],
                },
                "source_lookup": {
                    "DOJ-OGR-00000002": {"document_id": "DOJ-OGR-00000002"},
                },
            },
        ),
        encoding="utf-8",
    )

    observations = load_from_cache(tmp_path)
    assert observations
    assert any(o.source_doc == "DOJ-OGR-00000002" for o in observations)
    assert any(
        o.predicate == "occurred_on" and o.object == "2024-09-17"
        for o in observations
    )


@pytest.mark.asyncio
async def test_ingest_observation_adds_source_and_date_metadata() -> None:
    captured: list[dict] = []

    class _DB:
        def query_triples(self, subject: str, predicate: str, k: int = 50):  # noqa: ARG002
            return []

        def record_observation(self, external_id: str, confirmed: bool, source: str):  # noqa: ARG002
            return None

        def insert_triple(
            self,
            external_id: str,
            subject: str,
            predicate: str,
            object_value: str,
            probability: float,
            metadata: dict,
        ) -> None:
            captured.append(
                {
                    "external_id": external_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object_value": object_value,
                    "probability": probability,
                    "metadata": metadata,
                },
            )

    class _Backend:
        def __init__(self) -> None:
            self._db = _DB()

        def _get_user_prefix(self) -> str:
            return "tenant/user/"

    consolidator = ObservationConsolidator(
        backend=_Backend(),
        resolver=object(),  # not used by _ingest_observations
    )
    obs = Observation(
        id=uuid4(),
        kind="fact",
        subject="Epstein's NPA",
        predicate="signed in",
        object="September 2007",
        confidence=0.8,
        source_doc="DOJ-OGR-00000002",
        raw={},
    )

    _, inserted = await consolidator._ingest_observations([obs])  # noqa: SLF001
    assert inserted == 0
    assert captured
    meta = captured[0]["metadata"]
    source_meta = meta.get("source_metadata", {})
    assert source_meta.get("source_document", {}).get("document_id") == "DOJ-OGR-00000002"
    assert source_meta.get("object_date", {}).get("canonical") == "2007-09"


def test_simple_merge_coerces_non_string_triplet_fields() -> None:
    cluster = type("ClusterLike", (), {})()
    cluster.size = 1
    cluster.kind = "fact"
    cluster.observations = [
        Observation(
            kind="fact",
            subject=["Alice", "Bob"],  # type: ignore[arg-type]
            predicate={"rel": "works_with"},  # type: ignore[arg-type]
            object=["ACME"],  # type: ignore[arg-type]
            confidence=0.8,
        ),
    ]
    cluster.best = cluster.observations[0]

    merged = ObservationConsolidator._simple_merge(cluster)  # noqa: SLF001
    assert merged.triplet is not None
    assert isinstance(merged.triplet.subject, str)
    assert isinstance(merged.triplet.predicate, str)
    assert isinstance(merged.triplet.object, str)


@pytest.mark.asyncio
async def test_llm_merge_coerces_confidence_and_list_values() -> None:
    class _LLM:
        async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2):  # noqa: ANN201, ARG002
            return '{"subject":["Alice"],"predicate":"works_with","object":["Bob"],"confidence":"0.83"}'

    class _Backend:
        def __init__(self) -> None:
            self._db = object()

    consolidator = ObservationConsolidator(
        backend=_Backend(),
        resolver=object(),
        llm=_LLM(),
    )
    cluster = type("ClusterLike", (), {})()
    cluster.subject = "Alice"
    cluster.kind = "relationship"
    cluster.size = 2
    cluster.observations = [
        Observation(kind="relationship", subject="Alice", predicate="works_with", object="Bob", confidence=0.7),
        Observation(kind="relationship", subject="Alice", predicate="works with", object="Bob", confidence=0.6),
    ]
    cluster.best = cluster.observations[0]

    merged = await consolidator._llm_merge(cluster)  # noqa: SLF001
    assert merged.triplet is not None
    assert merged.triplet.subject == "Alice"
    assert merged.triplet.object == "Bob"
    assert isinstance(merged.confidence, float)
