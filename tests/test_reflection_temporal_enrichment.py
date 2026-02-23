"""Tests for temporal enrichment in reflection belief generation."""

from __future__ import annotations

from silicon_memory.reflection.generator import BeliefGenerator, _build_temporal_context
from silicon_memory.reflection.types import Pattern, PatternType


def test_build_temporal_context_day_precision() -> None:
    temporal = _build_temporal_context(
        {"date_normalized": {"canonical": "2021-06-25", "precision": "day", "ambiguous": False}}
    )
    assert temporal is not None
    assert temporal.observed_at.year == 2021
    assert temporal.observed_at.month == 6
    assert temporal.observed_at.day == 25
    assert temporal.valid_from is None
    assert temporal.valid_until is None


def test_build_temporal_context_month_precision() -> None:
    temporal = _build_temporal_context(
        {"date_normalized": {"canonical": "2021-06", "precision": "month", "ambiguous": False}}
    )
    assert temporal is not None
    assert temporal.observed_at.year == 2021
    assert temporal.observed_at.month == 6
    assert temporal.observed_at.day == 1
    assert temporal.valid_from is not None
    assert temporal.valid_until is not None
    assert temporal.valid_until.day == 30


def test_build_temporal_context_year_precision() -> None:
    temporal = _build_temporal_context(
        {"date_normalized": {"canonical": "2019", "precision": "year", "ambiguous": False}}
    )
    assert temporal is not None
    assert temporal.observed_at.year == 2019
    assert temporal.valid_from is not None
    assert temporal.valid_until is not None
    assert temporal.valid_from.month == 1
    assert temporal.valid_until.month == 12


def test_build_temporal_context_ambiguous_skips() -> None:
    temporal = _build_temporal_context(
        {"date_normalized": {"canonical": "2021-03-04", "precision": "day", "ambiguous": True}}
    )
    assert temporal is None


def test_build_temporal_context_relative_skips() -> None:
    temporal = _build_temporal_context(
        {
            "date_normalized": {
                "canonical": "2026",
                "precision": "year",
                "ambiguous": False,
                "relative": True,
                "reference_date": "2026-02-19",
            },
        },
    )
    assert temporal is None


def test_build_temporal_context_from_raw_date_fallback() -> None:
    temporal = _build_temporal_context({"date": "June 25, 2021"})
    assert temporal is not None
    assert temporal.observed_at.year == 2021
    assert temporal.observed_at.month == 6
    assert temporal.observed_at.day == 25


def test_build_temporal_context_from_source_occurred_at() -> None:
    temporal = _build_temporal_context(
        {
            "source_document": {
                "document_id": "doc-1",
                "occurred_at": "2026-02-19T10:30:00+00:00",
            },
        },
    )
    assert temporal is not None
    assert temporal.observed_at.year == 2026
    assert temporal.observed_at.month == 2
    assert temporal.observed_at.day == 19


def test_timeline_pattern_maps_to_occurred_on_triplet() -> None:
    generator = BeliefGenerator(memory=object())  # type: ignore[arg-type]
    pattern = Pattern(
        type=PatternType.TIMELINE_EVENT,
        description="[2024-09-17] Judgment affirmed",
        subject="Case 22-1426",
        predicate="event",
        object="Judgment affirmed",
        confidence=0.9,
        context={
            "date_normalized": {
                "canonical": "2024-09-17",
                "precision": "day",
                "ambiguous": False,
                "confidence": 1.0,
            },
        },
    )
    candidate = generator._pattern_to_candidate(pattern)  # noqa: SLF001
    assert candidate is not None
    assert candidate.predicate == "occurred_on"
    assert candidate.object == "2024-09-17"
    assert candidate.source_context.get("event_text") == "Judgment affirmed"


def test_timeline_pattern_derives_date_when_context_missing() -> None:
    generator = BeliefGenerator(memory=object())  # type: ignore[arg-type]
    pattern = Pattern(
        type=PatternType.TIMELINE_EVENT,
        description="Judgment affirmed on June 25, 2021",
        subject="Case 22-1426",
        predicate="event",
        object="June 25, 2021",
        confidence=0.9,
        context={},
    )
    candidate = generator._pattern_to_candidate(pattern)  # noqa: SLF001
    assert candidate is not None
    assert candidate.predicate == "occurred_on"
    assert candidate.object == "2021-06-25"
    assert candidate.source_context.get("date_normalized", {}).get("canonical") == "2021-06-25"


def test_timeline_pattern_uses_context_date_when_available() -> None:
    generator = BeliefGenerator(memory=object())  # type: ignore[arg-type]
    pattern = Pattern(
        type=PatternType.TIMELINE_EVENT,
        description="Appellate decision issued",
        subject="Case 22-1426",
        predicate="event",
        object="Decision issued",
        confidence=0.9,
        context={"date": "2024-09-17"},
    )
    candidate = generator._pattern_to_candidate(pattern)  # noqa: SLF001
    assert candidate is not None
    assert candidate.predicate == "occurred_on"
    assert candidate.object == "2024-09-17"
    assert candidate.source_context.get("date_normalized", {}).get("canonical") == "2024-09-17"


def test_timeline_pattern_does_not_anchor_relative_date() -> None:
    generator = BeliefGenerator(memory=object())  # type: ignore[arg-type]
    pattern = Pattern(
        type=PatternType.TIMELINE_EVENT,
        description="Hearing is tomorrow",
        subject="Case 22-1426",
        predicate="event",
        object="tomorrow",
        confidence=0.9,
        context={"source": {"occurred_at": "2026-02-19T10:30:00+00:00"}},
    )
    candidate = generator._pattern_to_candidate(pattern)  # noqa: SLF001
    assert candidate is not None
    assert candidate.predicate == "event"
    assert candidate.object == "tomorrow"
    assert "date_normalized" not in candidate.source_context
