from __future__ import annotations

from uuid import uuid4

import pytest

from silicon_memory.core.claims import (
    AntiTrivialityPolicy,
    ClaimDecisionLabel,
    ClaimMergeCandidate,
    ClaimPrecisionJudge,
    build_provenance_contract,
    validate_provenance_contract,
)
from silicon_memory.core.types import Belief, Source, SourceType, Triplet


def test_build_and_validate_provenance_contract() -> None:
    belief = Belief(
        triplet=Triplet("ACME", "filed_case", "HR-2024-123"),
        source=Source(
            id="reflection_engine",
            type=SourceType.REFLECTION,
            reliability=0.7,
            metadata={
                "source_document": {"document_id": "doc-7"},
                "evidence_span": "line:12-20",
                "extraction_run_id": "run-1",
                "extractor_model": "qwen3-80b",
                "extractor_version": "v1",
                "extracted_at": "2026-02-19T12:00:00+00:00",
                "source_type": "document",
                "confidence_basis": "llm",
            },
        ),
    )

    contract = build_provenance_contract(belief)
    validation = validate_provenance_contract(contract)

    assert validation.complete is True
    assert validation.missing_fields == []
    assert contract.source_doc_id == "doc-7"
    assert contract.extractor_model == "qwen3-80b"


def test_validate_provenance_contract_detects_missing_fields() -> None:
    belief = Belief(
        triplet=Triplet("Alice", "related_to", "Bob"),
        source=Source(
            id="reflection_engine",
            type=SourceType.REFLECTION,
            metadata={},
        ),
    )

    contract = build_provenance_contract(belief)
    validation = validate_provenance_contract(contract)

    assert validation.complete is False
    assert "extraction_run_id" in validation.missing_fields
    assert validation.completeness_rate < 1.0


def test_anti_triviality_policy_flags_non_trivial_claims() -> None:
    policy = AntiTrivialityPolicy()
    belief = Belief(
        triplet=Triplet("Case HR-2024-123", "occurred_on", "2024-03-12"),
        content="Case HR-2024-123 occurred on 2024-03-12",
        source=Source(
            id="reflection_engine",
            type=SourceType.REFLECTION,
            metadata={
                "date_normalized": {
                    "canonical": "2024-03-12",
                    "precision": "day",
                    "ambiguous": False,
                    "relative": False,
                },
            },
        ),
    )

    result = policy.evaluate(belief)
    assert result.promotable is True
    assert "specific_relation" in result.signals
    assert "temporal_anchor" in result.signals


@pytest.mark.asyncio
async def test_claim_precision_judge_rule_merge() -> None:
    belief = Belief(
        triplet=Triplet("Alice", "works_at", "ACME"),
    )
    candidate = ClaimMergeCandidate(
        belief_id=uuid4(),
        score=1.0,
        reason="exact_match",
        subject="Alice",
        predicate="works_at",
        object_value="ACME",
    )
    judge = ClaimPrecisionJudge()
    decision = await judge.decide(belief, [candidate], llm=None)
    assert decision.label == ClaimDecisionLabel.MERGE
    assert decision.candidate_belief_id == candidate.belief_id


@pytest.mark.asyncio
async def test_claim_precision_judge_rule_contradiction() -> None:
    belief = Belief(triplet=Triplet("Alice", "is_active", "false"))
    candidate = ClaimMergeCandidate(
        belief_id=uuid4(),
        score=0.9,
        reason="same_sp",
        subject="Alice",
        predicate="is_active",
        object_value="true",
    )
    judge = ClaimPrecisionJudge()
    decision = await judge.decide(belief, [candidate], llm=None)
    assert decision.label == ClaimDecisionLabel.CONTRADICTION
    assert decision.candidate_belief_id == candidate.belief_id


class _StubLLM:
    async def generate_structured(self, prompt, schema, max_tokens=None):  # noqa: ANN001, ANN201
        _ = (prompt, max_tokens)
        return schema.model_validate(
            {
                "label": "RELATED",
                "candidate_belief_id": str(self._candidate_id),
                "reason": "llm_related",
                "confidence": 0.66,
            },
        )

    def __init__(self, candidate_id) -> None:  # noqa: ANN001
        self._candidate_id = candidate_id


@pytest.mark.asyncio
async def test_claim_precision_judge_llm_override() -> None:
    belief = Belief(triplet=Triplet("Alice", "works_with", "Bob"))
    candidate = ClaimMergeCandidate(
        belief_id=uuid4(),
        score=0.65,
        reason="surface_overlap",
        subject="Alice",
        predicate="works_at",
        object_value="ACME",
    )
    llm = _StubLLM(candidate.belief_id)
    judge = ClaimPrecisionJudge()
    decision = await judge.decide(belief, [candidate], llm=llm)
    assert decision.label == ClaimDecisionLabel.RELATED
    assert decision.via == "llm"
