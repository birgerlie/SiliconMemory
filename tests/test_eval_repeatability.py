"""Unit tests for repeatability evaluation logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "eval_repeatability.py"
    spec = importlib.util.spec_from_file_location("eval_repeatability_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_repeatability_module"] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_repeatability_passes_with_small_deltas() -> None:
    mod = _load_module()

    batch_a = {
        "doc_count": 5,
        "patterns_total": 24,
        "committed_triplet_beliefs": 20,
        "reflection": {"experiences_processed": 20},
        "retrieval_sanity": {
            "direct_experience_visibility": {"expected": 5, "visible": 5},
            "extraction_progress": {"extracted": 5},
            "unprocessed_experiences_count": 0,
            "unextracted_experiences_count": 0,
        },
    }
    batch_b = {
        "doc_count": 5,
        "patterns_total": 26,
        "committed_triplet_beliefs": 22,
        "reflection": {"experiences_processed": 21},
        "retrieval_sanity": {
            "direct_experience_visibility": {"expected": 5, "visible": 5},
            "extraction_progress": {"extracted": 5},
            "unprocessed_experiences_count": 0,
            "unextracted_experiences_count": 0,
        },
    }
    dream_a = {"dream": {"procedures_created": 4, "questions_generated": 5}, "retrieval_sanity": {"procedures_count": 8, "open_questions_count": 5}}
    dream_b = {"dream": {"procedures_created": 5, "questions_generated": 4}, "retrieval_sanity": {"procedures_count": 9, "open_questions_count": 4}}

    result = mod.evaluate_repeatability(batch_a, batch_b, dream_a, dream_b)

    assert result["gates_passed"] is True
    assert result["metrics_passed"] is True
    assert result["overall_passed"] is True


def test_evaluate_repeatability_fails_on_gate_or_large_delta() -> None:
    mod = _load_module()

    batch_a = {
        "doc_count": 5,
        "patterns_total": 10,
        "committed_triplet_beliefs": 5,
        "reflection": {"experiences_processed": 5},
        "retrieval_sanity": {
            "direct_experience_visibility": {"expected": 5, "visible": 5},
            "extraction_progress": {"extracted": 5},
            "unprocessed_experiences_count": 0,
            "unextracted_experiences_count": 0,
        },
    }
    batch_b = {
        "doc_count": 5,
        "patterns_total": 40,  # large delta should fail
        "committed_triplet_beliefs": 30,
        "reflection": {"experiences_processed": 30},
        "retrieval_sanity": {
            "direct_experience_visibility": {"expected": 5, "visible": 3},  # gate fail
            "extraction_progress": {"extracted": 2},
            "unprocessed_experiences_count": 2,
            "unextracted_experiences_count": 3,
        },
    }
    dream_a = {"dream": {"procedures_created": 1, "questions_generated": 1}, "retrieval_sanity": {"procedures_count": 1, "open_questions_count": 1}}
    dream_b = {"dream": {"procedures_created": 12, "questions_generated": 12}, "retrieval_sanity": {"procedures_count": 0, "open_questions_count": 0}}

    result = mod.evaluate_repeatability(batch_a, batch_b, dream_a, dream_b)

    assert result["metrics_passed"] is False
    assert result["gates_passed"] is False
    assert result["overall_passed"] is False


def test_evaluate_repeatability_enforces_optional_minimums() -> None:
    mod = _load_module()

    batch_a = {
        "doc_count": 5,
        "patterns_total": 20,
        "committed_triplet_beliefs": 20,
        "reflection": {"experiences_processed": 20},
        "retrieval_sanity": {
            "direct_experience_visibility": {"expected": 5, "visible": 5},
            "extraction_progress": {"extracted": 5},
            "unprocessed_experiences_count": 0,
            "unextracted_experiences_count": 0,
        },
    }
    batch_b = {
        "doc_count": 5,
        "patterns_total": 20,
        "committed_triplet_beliefs": 20,
        "reflection": {"experiences_processed": 20},
        "retrieval_sanity": {
            "direct_experience_visibility": {"expected": 5, "visible": 5},
            "extraction_progress": {"extracted": 5},
            "unprocessed_experiences_count": 0,
            "unextracted_experiences_count": 0,
        },
    }
    dream_a = {"dream": {"procedures_created": 4, "questions_generated": 4}, "retrieval_sanity": {"procedures_count": 4, "open_questions_count": 0}}
    dream_b = {"dream": {"procedures_created": 4, "questions_generated": 4}, "retrieval_sanity": {"procedures_count": 4, "open_questions_count": 0}}

    strict = mod.evaluate_repeatability(
        batch_a,
        batch_b,
        dream_a,
        dream_b,
        require_open_questions_min=1,
        require_procedures_min=1,
    )
    relaxed = mod.evaluate_repeatability(
        batch_a,
        batch_b,
        dream_a,
        dream_b,
        require_open_questions_min=0,
        require_procedures_min=1,
    )

    assert strict["gates_passed"] is False
    assert relaxed["gates_passed"] is True
