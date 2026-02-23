"""Claim strategy helpers for canonical promotion and provenance gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from silicon_memory.core.types import Belief
from silicon_memory.core.utils import utc_now


_UNKNOWN_VALUES = {"", "unknown", "n/a", "none", "null"}
_GENERIC_PREDICATES = {
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "related to",
    "relates to",
    "event",
    "argues",
    "states",
    "mentions",
}
_NUMERIC_OR_IDENTIFIER_RE = re.compile(r"[0-9]")


class ClaimDecisionLabel(StrEnum):
    """Decision labels for mapping observations to canonical claims."""

    MERGE = "MERGE"
    CONTRADICTION = "CONTRADICTION"
    RELATED = "RELATED"
    NEW = "NEW"


@dataclass(frozen=True)
class ProvenanceContract:
    """Required provenance contract for canonical claim promotion."""

    source_doc_id: str
    evidence_span: str
    extraction_run_id: str
    extractor_model: str
    extractor_version: str
    extracted_at: str
    source_type: str
    confidence_basis: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_doc_id": self.source_doc_id,
            "evidence_span": self.evidence_span,
            "extraction_run_id": self.extraction_run_id,
            "extractor_model": self.extractor_model,
            "extractor_version": self.extractor_version,
            "extracted_at": self.extracted_at,
            "source_type": self.source_type,
            "confidence_basis": self.confidence_basis,
        }


@dataclass(frozen=True)
class ProvenanceValidation:
    """Validation result for the provenance contract."""

    contract: ProvenanceContract
    missing_fields: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.missing_fields) == 0

    @property
    def completeness_rate(self) -> float:
        required_count = 8
        present = max(0, required_count - len(self.missing_fields))
        return present / required_count


@dataclass(frozen=True)
class AntiTrivialityResult:
    """Decision data for anti-triviality promotion policy."""

    promotable: bool
    score: float
    signals: list[str] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClaimMergeCandidate:
    """Scored candidate claim for canonical decisioning."""

    belief_id: UUID
    score: float
    reason: str
    source: str = ""
    subject: str = ""
    predicate: str = ""
    object_value: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ClaimDecision:
    """Final decision from precision judge."""

    label: ClaimDecisionLabel
    candidate_belief_id: UUID | None
    reason: str
    confidence: float
    via: str = "rules"


class ClaimDecisionResponse(BaseModel):
    """Structured schema for LLM precision decisions."""

    label: str = Field(description="One of MERGE, CONTRADICTION, RELATED, NEW")
    candidate_belief_id: str | None = Field(default=None)
    reason: str = Field(default="")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def build_provenance_contract(belief: Belief) -> ProvenanceContract:
    """Build provenance contract fields from source metadata."""
    source_meta = belief.source.metadata if belief.source and isinstance(belief.source.metadata, dict) else {}

    source_doc = source_meta.get("source_document")
    if isinstance(source_doc, dict):
        source_doc_id = (
            str(source_doc.get("document_id") or "").strip()
            or str(source_doc.get("title") or "").strip()
            or str(source_doc.get("experience_id") or "").strip()
            or str(source_doc.get("source_id") or "").strip()
        )
    elif isinstance(source_doc, str):
        source_doc_id = source_doc.strip()
    else:
        source_doc_id = ""
    if not source_doc_id:
        source_doc_id = str(
            source_meta.get("source_doc_id")
            or source_meta.get("grounding_doc_id")
            or "",
        ).strip()
    if not source_doc_id and belief.source:
        source_doc_id = str(belief.source.id or "").strip()

    evidence_span = str(
        source_meta.get("evidence_span")
        or source_meta.get("triplet_key")
        or source_meta.get("context_bundle_id")
        or "",
    ).strip()
    extraction_run_id = str(
        source_meta.get("extraction_run_id")
        or source_meta.get("run_id")
        or source_meta.get("batch_id")
        or "",
    ).strip()
    extractor_model = str(
        source_meta.get("extractor_model")
        or source_meta.get("llm_model")
        or source_meta.get("model")
        or "",
    ).strip()
    extractor_version = str(
        source_meta.get("extractor_version")
        or source_meta.get("version")
        or source_meta.get("extractor_build")
        or "",
    ).strip()
    extracted_at = str(source_meta.get("extracted_at") or "").strip() or utc_now().isoformat()
    source_type = str(
        source_meta.get("source_type")
        or (belief.source.type.value if belief.source and hasattr(belief.source.type, "value") else "")
        or "",
    ).strip()
    confidence_basis = str(source_meta.get("confidence_basis") or "").strip()
    if not confidence_basis:
        if source_meta.get("reasoning"):
            confidence_basis = "hybrid"
        elif belief.source and hasattr(belief.source.type, "value") and belief.source.type.value == "reflection":
            confidence_basis = "llm"
        else:
            confidence_basis = "rule"

    return ProvenanceContract(
        source_doc_id=source_doc_id,
        evidence_span=evidence_span,
        extraction_run_id=extraction_run_id,
        extractor_model=extractor_model,
        extractor_version=extractor_version,
        extracted_at=extracted_at,
        source_type=source_type,
        confidence_basis=confidence_basis,
    )


def validate_provenance_contract(contract: ProvenanceContract) -> ProvenanceValidation:
    """Validate required provenance fields."""
    missing: list[str] = []
    values = contract.as_dict()
    for key, value in values.items():
        norm = str(value).strip().lower()
        if norm in _UNKNOWN_VALUES:
            missing.append(key)
    return ProvenanceValidation(contract=contract, missing_fields=missing)


class AntiTrivialityPolicy:
    """Rule-based anti-triviality gate for canonical promotion."""

    def __init__(self, min_signals: int = 1) -> None:
        self._min_signals = max(1, min_signals)

    def evaluate(self, belief: Belief) -> AntiTrivialityResult:
        """Evaluate whether belief is non-trivial enough for canonical promotion."""
        signals: list[str] = []
        block_reasons: list[str] = []

        source_meta = belief.source.metadata if belief.source and isinstance(belief.source.metadata, dict) else {}

        if belief.triplet:
            predicate = " ".join(str(belief.triplet.predicate or "").strip().lower().split())
            if predicate and predicate not in _GENERIC_PREDICATES:
                signals.append("specific_relation")

            if _NUMERIC_OR_IDENTIFIER_RE.search(belief.triplet.object or ""):
                signals.append("numeric_or_identifier")

        if _NUMERIC_OR_IDENTIFIER_RE.search(belief.content or ""):
            signals.append("numeric_or_identifier")

        date_meta = source_meta.get("date_normalized")
        if isinstance(date_meta, dict):
            canonical = str(date_meta.get("canonical") or "").strip()
            ambiguous = bool(date_meta.get("ambiguous", False))
            relative = bool(date_meta.get("relative", False))
            if canonical and not ambiguous and not relative:
                signals.append("temporal_anchor")

        co_triplets = source_meta.get("co_triplet_keys")
        if isinstance(co_triplets, list) and len(co_triplets) >= 1:
            signals.append("cross_claim_context")

        experience_ids = source_meta.get("experiences")
        if isinstance(experience_ids, list) and len(experience_ids) >= 2:
            signals.append("multi_source_support")

        if belief.evidence_against:
            signals.append("contradiction_surface")

        unique_signals = list(dict.fromkeys(signals))
        promotable = len(unique_signals) >= self._min_signals
        if not promotable:
            block_reasons.append("no_non_trivial_signal")
            if belief.triplet:
                predicate = " ".join(str(belief.triplet.predicate or "").strip().lower().split())
                if predicate in _GENERIC_PREDICATES:
                    block_reasons.append("generic_predicate_only")
        return AntiTrivialityResult(
            promotable=promotable,
            score=float(len(unique_signals)),
            signals=unique_signals,
            block_reasons=block_reasons,
        )


class ClaimPrecisionJudge:
    """Rules-first precision judge with optional LLM escalation."""

    async def decide(
        self,
        belief: Belief,
        candidates: list[ClaimMergeCandidate],
        *,
        llm: Any | None = None,
        llm_max_candidates: int = 6,
    ) -> ClaimDecision:
        """Return MERGE/CONTRADICTION/RELATED/NEW against candidate set."""
        sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        rule_decision = self._rules_decision(belief, sorted_candidates)

        if not llm or not sorted_candidates:
            return rule_decision
        if (
            rule_decision.label in {ClaimDecisionLabel.MERGE, ClaimDecisionLabel.CONTRADICTION}
            and rule_decision.confidence >= 0.92
        ):
            return rule_decision

        prompt = self._build_prompt(
            belief,
            sorted_candidates[: max(1, llm_max_candidates)],
        )
        try:
            llm_result = await llm.generate_structured(
                prompt,
                ClaimDecisionResponse,
                max_tokens=256,
            )
            parsed = llm_result if isinstance(llm_result, ClaimDecisionResponse) else ClaimDecisionResponse.model_validate(llm_result)
            label_text = str(parsed.label or "").strip().upper()
            if label_text not in {item.value for item in ClaimDecisionLabel}:
                return rule_decision
            label = ClaimDecisionLabel(label_text)

            candidate_ids = {str(item.belief_id) for item in sorted_candidates}
            candidate_belief_id: UUID | None = None
            if parsed.candidate_belief_id and parsed.candidate_belief_id in candidate_ids:
                candidate_belief_id = UUID(parsed.candidate_belief_id)
            elif label in {ClaimDecisionLabel.MERGE, ClaimDecisionLabel.CONTRADICTION, ClaimDecisionLabel.RELATED}:
                candidate_belief_id = sorted_candidates[0].belief_id

            if label == ClaimDecisionLabel.MERGE and candidate_belief_id is None:
                return rule_decision

            confidence = max(0.0, min(1.0, float(parsed.confidence)))
            return ClaimDecision(
                label=label,
                candidate_belief_id=candidate_belief_id,
                reason=str(parsed.reason or "llm_judgement").strip() or "llm_judgement",
                confidence=confidence,
                via="llm",
            )
        except Exception:
            return rule_decision

    def _rules_decision(
        self,
        belief: Belief,
        candidates: list[ClaimMergeCandidate],
    ) -> ClaimDecision:
        if not candidates:
            return ClaimDecision(
                label=ClaimDecisionLabel.NEW,
                candidate_belief_id=None,
                reason="no_candidate_claims",
                confidence=0.6,
            )

        best = candidates[0]
        if best.score >= 0.995:
            return ClaimDecision(
                label=ClaimDecisionLabel.MERGE,
                candidate_belief_id=best.belief_id,
                reason="exact_or_near_exact_triplet_match",
                confidence=0.97,
            )

        if belief.triplet:
            candidate_subject = " ".join(best.subject.strip().lower().split())
            candidate_predicate = " ".join(best.predicate.strip().lower().split())
            candidate_object = " ".join(best.object_value.strip().lower().split())
            belief_subject = " ".join(belief.triplet.subject.strip().lower().split())
            belief_predicate = " ".join(belief.triplet.predicate.strip().lower().split())
            belief_object = " ".join(belief.triplet.object.strip().lower().split())
            if (
                candidate_subject
                and candidate_predicate
                and belief_subject == candidate_subject
                and belief_predicate == candidate_predicate
                and candidate_object
                and belief_object
                and candidate_object != belief_object
                and self._are_mutually_exclusive(candidate_object, belief_object)
            ):
                return ClaimDecision(
                    label=ClaimDecisionLabel.CONTRADICTION,
                    candidate_belief_id=best.belief_id,
                    reason="same_subject_predicate_mutually_exclusive_object",
                    confidence=0.92,
                )

        if best.score >= 0.82:
            return ClaimDecision(
                label=ClaimDecisionLabel.RELATED,
                candidate_belief_id=best.belief_id,
                reason="high_similarity_but_not_equivalent",
                confidence=0.74,
            )

        return ClaimDecision(
            label=ClaimDecisionLabel.NEW,
            candidate_belief_id=None,
            reason="insufficient_match_confidence",
            confidence=0.58,
        )

    @staticmethod
    def _are_mutually_exclusive(value_a: str, value_b: str) -> bool:
        va = value_a.strip().lower()
        vb = value_b.strip().lower()
        if not va or not vb:
            return False
        if {va, vb} in (
            {"true", "false"},
            {"yes", "no"},
            {"present", "absent"},
            {"guilty", "not guilty"},
        ):
            return True
        if va.startswith("not ") and va[4:] == vb:
            return True
        if vb.startswith("not ") and vb[4:] == va:
            return True
        try:
            return float(va) != float(vb)
        except Exception:
            return False

    @staticmethod
    def _build_prompt(
        belief: Belief,
        candidates: list[ClaimMergeCandidate],
    ) -> str:
        lines = []
        for i, candidate in enumerate(candidates, start=1):
            lines.append(
                f"{i}. belief_id={candidate.belief_id} | score={candidate.score:.3f} | "
                f"triplet=({candidate.subject}, {candidate.predicate}, {candidate.object_value}) | "
                f"reason={candidate.reason}"
            )

        if belief.triplet:
            claim_text = f"({belief.triplet.subject}, {belief.triplet.predicate}, {belief.triplet.object})"
        else:
            claim_text = belief.content or ""

        return (
            "Classify the relation between the NEW claim and candidate canonical claims.\n"
            "Allowed labels: MERGE, CONTRADICTION, RELATED, NEW.\n"
            "Pick MERGE only if semantically equivalent claim.\n"
            "Pick CONTRADICTION only if incompatible with a specific candidate.\n"
            "Pick RELATED for contextual relation but distinct claim.\n"
            "Pick NEW if no candidate fits.\n\n"
            f"NEW claim:\n{claim_text}\n\n"
            "Candidates:\n"
            + "\n".join(lines)
            + "\n\nReturn JSON with keys: label, candidate_belief_id, reason, confidence."
        )
