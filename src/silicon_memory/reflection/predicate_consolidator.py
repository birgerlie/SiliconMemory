"""Periodic predicate consolidation via LLM evaluation.

Collects all unique predicates from the knowledge graph, groups synonyms,
and canonicalizes them to short, active-voice, present-tense forms.
This prevents predicate drift across extraction runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of a predicate consolidation pass."""

    skipped: bool = False
    predicate_count: int = 0
    merged: int = 0
    refitted: int = 0
    groups: list[list[str]] = field(default_factory=list)


class PredicateConsolidator:
    """Consolidates predicate variants into canonical forms using LLM.

    With ~40-50 real predicates, they all fit in one LLM prompt.
    The LLM groups synonyms and picks canonical forms.
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        llm: Any,
        threshold: int = 60,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._threshold = threshold

    async def consolidate_predicates(self) -> ConsolidationResult:
        """Run predicate consolidation.

        Collects all unique predicates, sends them to the LLM for
        grouping and canonicalization, and updates triples in-place.
        """
        db = self._memory._backend._db

        # 1. Collect all unique predicates with usage counts
        predicates = db.all_predicates()
        result = ConsolidationResult(predicate_count=len(predicates))

        if len(predicates) <= self._threshold:
            result.skipped = True
            logger.info(
                "Predicate consolidation skipped: %d predicates <= threshold %d",
                len(predicates), self._threshold,
            )
            return result

        # Get usage counts per predicate
        predicate_counts: dict[str, int] = {}
        for pred in predicates:
            triples = db.query_triples(predicate=pred, k=10000)
            predicate_counts[pred] = len(triples)

        # 2. Single LLM call: evaluate all predicates, group & canonicalize
        predicate_list = "\n".join(
            f'- "{pred}" (used {predicate_counts.get(pred, 0)} times)'
            for pred in sorted(predicates)
        )
        mappings = await self._llm_evaluate(predicate_list)

        if not mappings:
            logger.info("No predicate mappings returned by LLM")
            return result

        # 3. Refit: use native rewrite_predicates() for batched rewrite + merge
        # Filter out identity mappings
        effective_mappings = {
            old: new for old, new in mappings.items() if old != new
        }

        if effective_mappings:
            # Register aliases server-side so future queries auto-resolve
            aliases: dict[str, list[str]] = {}
            for old_pred, canonical in effective_mappings.items():
                aliases.setdefault(canonical, []).append(old_pred)
            try:
                db.register_predicate_aliases(aliases)
                logger.info(
                    "Registered %d predicate alias groups server-side",
                    len(aliases),
                )
            except Exception as e:
                logger.debug("register_predicate_aliases failed (non-fatal): %s", e)

            try:
                rw_result = db.rewrite_predicates(effective_mappings)
                result.refitted = rw_result.rewritten_count
                result.merged = len(effective_mappings)
                logger.info(
                    "Predicate consolidation (native): %d mappings, "
                    "%d triples rewritten, %d merged, %d surviving",
                    len(effective_mappings),
                    rw_result.rewritten_count,
                    rw_result.merged_count,
                    rw_result.surviving_count,
                )
            except Exception as e:
                logger.warning("Native rewrite_predicates failed: %s", e)
                result.merged = 0
                result.refitted = 0
        else:
            result.merged = 0
            result.refitted = 0
            logger.info("Predicate consolidation: no effective mappings")
        return result

    async def _llm_evaluate(self, predicate_list: str) -> dict[str, str]:
        """Ask LLM to group and canonicalize predicates.

        Returns mapping of old_pred -> canonical_pred.
        """
        system = (
            "You are a knowledge graph curator. Below are all predicates "
            "currently in the graph with their usage counts. Your job:\n\n"
            "1. Group predicates that mean the same thing\n"
            "2. For each group, pick ONE canonical predicate\n"
            "3. Return a mapping of old → canonical\n\n"
            "Rules for canonical predicates:\n"
            "- Short (2-4 words)\n"
            "- Active voice, present tense\n"
            "- Domain-agnostic\n"
            "- Prefer the variant with the highest usage count\n"
            "- Leave predicates that are already unique and well-formed unchanged\n\n"
            'Respond with JSON only: {"mappings": {"old_pred": "canonical_pred", ...}}'
        )
        user = f"Predicates:\n{predicate_list}"

        try:
            response = await self._llm_call(system, user)
            parsed = self._parse_json(response)
            if parsed and "mappings" in parsed:
                return parsed["mappings"]
        except Exception as e:
            logger.warning("Predicate consolidation LLM call failed: %s", e)

        return {}

    async def _llm_call(self, system: str, user: str) -> str:
        """Make an LLM call, handling both scheduler and direct provider."""
        prompt = f"{system}\n\n{user}"
        if hasattr(self._llm, "complete"):
            return await self._llm.complete(
                prompt, system=system, temperature=0.3, max_tokens=2048,
            )
        elif hasattr(self._llm, "generate"):
            return await self._llm.generate(
                prompt, max_tokens=2048, temperature=0.3,
            )
        raise TypeError(f"Unknown LLM type: {type(self._llm)}")

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        # Try finding first { ... }
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
        return None
